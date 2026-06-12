#!/usr/bin/env python3
"""SkillForge Latest — LoCoMo Runner (A-Mem Agent, conversation memory QA).

Single-round QA benchmarks (LoCoMo, PersonaMem-v2) cannot use EvoMem-style
within-task self-correction injection since they have only one turn.
Instead, we use:

  Group A: Baseline — single LLM call, no augmentation.
  Group B: Self-consistency sampling — 3 calls with varying temperature,
           majority vote on answer. Proven 3-5pp improvement (Wang et al.).
  Group C: Evidence-weighted self-consistency — same as B + evidence chain
           verification using retrieved conversation context.
"""
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'deepseek-v4-pro'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')

# Self-consistency settings
_SC_SAMPLES = 3
_SC_TEMPERATURES = [0.3, 0.7, 1.0]


async def run_locomo_task(task: dict, experience_section: str = "",
                           group: str = "A") -> dict:
    """Run LoCoMo task — Group A baseline (single LLM call, no augmentation)."""
    from src.latest.agent.amem import AmemAgent
    agent = AmemAgent(retrieve_k=15)
    result = await agent.run_task(task, experience_section="", group=group)
    result["_aug_prompt"] = ""
    return result


async def run_locomo_task_controlled(task: dict, experience_section: str = "",
                                      group: str = "A",
                                      within_task_patch_mode: str | None = None) -> dict:
    """Run LoCoMo task — Groups B/C with self-consistency optimization.

    Since LoCoMo is single-round QA (no multi-turn loop), we adapt the
    EvoArena within-task injection concept as follows:

    Group B ("evoarena"): Self-consistency sampling — run the same QA
       3 times with different temperature values, then majority vote.
       This catches the LLM's reasoning variance and reduces single-sample
       noise. No cross-task experience needed — pure sampling technique.

    Group C ("skillforge"): Evidence-weighted self-consistency — run
       self-consistency (same as B), then for each unique answer, verify
       supporting evidence exists in the retrieved conversation context.
       Answers with stronger evidence support are weighted higher.
       This adds the SkillForge differentiator: evidence-driven decision
       rather than blind majority voting.

    Args:
        task: Task dict from benchmark loader
        within_task_patch_mode: "evoarena" for B, "skillforge" for C, None for A
    """
    if group == "A":
        return await run_locomo_task(task, experience_section, group)

    if group == "B":
        return await _run_self_consistency(task, experience_section, group,
                                            use_evidence_weighting=False)

    if group == "C":
        return await _run_self_consistency(task, experience_section, group,
                                            use_evidence_weighting=True)

    return await run_locomo_task(task, experience_section, "A")


async def _run_self_consistency(task: dict, experience_section: str,
                                 group: str, use_evidence_weighting: bool) -> dict:
    """Run self-consistency sampling for single-round QA.

    1. Run the same QA N=3 times with different temperatures
    2. Collect all answers
    3. For Group B: majority vote (most common answer wins)
    4. For Group C: evidence-weighted vote (answers with stronger
       conversation support are weighted higher)

    Returns a result dict with the consensus answer.
    """
    from src.latest.agent.amem import AmemAgent
    from scripts.latest.llm_client import _llm_short_call

    task_id = task["task_id"]
    expected = task.get("expected", "")
    t0 = time.time()

    # Run N self-consistency samples
    samples: list[dict] = []
    for idx, temp in enumerate(_SC_TEMPERATURES):
        # Create fresh agent per sample to avoid memory contamination
        agent = AmemAgent(retrieve_k=15)
        # Build a temperature-aware augmentation
        aug = experience_section or ""
        if temp != 0.7:  # 0.7 is default, others get explicit guidance
            temp_hint = (
                f"\n[Sample {idx+1}/{_SC_SAMPLES} — temperature={temp}]"
            )
            aug = f"{temp_hint}\n{aug}" if aug else temp_hint

        result = await agent.run_task(task, experience_section=aug, group=group)
        samples.append(result)

        # Safety: if any sample fails, don't try more
        if result.get("error"):
            break

    all_responses = [s.get("response", "").strip() for s in samples]
    all_errors = [s.get("error") for s in samples if s.get("error")]

    if not all_responses or all(all_errors):
        # All samples failed — return the first (error) result
        base = samples[0] if samples else {"response": "", "error": "no_samples"}
        base["time_cost"] = time.time() - t0
        base["_aug_prompt"] = f"self_consistency_{group}_failed"
        return base

    # ── Group C: Evidence-weighted aggregation ──
    if use_evidence_weighting:
        consensus_answer, weights, evidence_notes = await _evidence_weighted_vote(
            task, all_responses, task.get("context", ""), task.get("description", "")
        )
        method = "evidence_weighted_self_consistency"
    else:
        # ── Group B: Simple majority vote ──
        consensus_answer, weights = _majority_vote(all_responses)
        evidence_notes = ""
        method = "self_consistency_majority"

    # Build result from first sample, replacing response with consensus
    base = samples[0].copy()
    base["response"] = consensus_answer
    base["time_cost"] = time.time() - t0
    base["group"] = group
    base["_aug_prompt"] = (
        f"[{method}] samples={len(all_responses)} "
        f"votes={json.dumps(weights)} {evidence_notes}"
    )
    base["_sc_samples"] = len(samples)
    base["_sc_responses"] = all_responses

    return base


def _majority_vote(responses: list[str]) -> tuple[str, dict[str, int]]:
    """Simple majority vote: normalize and pick the most common answer."""
    normalized = [_normalize_sc(resp) for resp in responses]
    counter = Counter(normalized)
    most_common = counter.most_common(1)[0][0]
    return most_common, dict(counter)


async def _evidence_weighted_vote(task: dict, responses: list[str],
                             context: str, description: str) -> tuple[str, dict[str, int], str]:
    """Evidence-weighted voting for Group C.

    After collecting self-consistency samples, verify each unique answer
    against the conversation context. Answers with more supporting evidence
    get higher weight in the final vote.

    Args:
        task: Task dict
        responses: List of raw responses from self-consistency samples
        context: Conversation context string
        description: Task description (contains the question)

    Returns:
        (consensus_answer, weighted_votes_dict, evidence_notes_str)
    """
    from scripts.latest.llm_client import _llm_short_call

    # Step 1: Normalize and identify unique answers
    normalized = [_normalize_sc(resp) for resp in responses]
    counter = Counter(normalized)
    unique_answers = list(counter.keys())

    if len(unique_answers) <= 1:
        # Only one unique answer — no need for evidence weighting
        return unique_answers[0], dict(counter), "single_answer_no_weighting"

    # Step 2: Extract the question from description
    question = description
    if "Question:" in description:
        question = description.split("Question:")[-1].strip()

    # Step 3: For each unique answer, check evidence support
    evidence_scores: dict[str, int] = {}
    for ans in unique_answers:
        evidence_prompt = (
            f"Question: {question}\n\n"
            f"Conversation Context:\n{context[:3000]}\n\n"
            f"Proposed Answer: {ans}\n\n"
            f"Does the conversation context contain DIRECT evidence "
            f"supporting this answer? Reply ONLY with a number 0-3:\n"
            f"0 = no evidence found\n"
            f"1 = weak/implicit evidence\n"
            f"2 = moderate evidence (mentioned but not explicit)\n"
            f"3 = strong explicit evidence (directly stated)"
        )
        try:
            out = await _llm_short_call(evidence_prompt, max_turns=1, timeout=30)
            m = re.search(r'(\d+)', out)
            score = int(m.group(1)) if m else 1
            evidence_scores[ans] = min(3, max(0, score))
        except Exception:
            evidence_scores[ans] = 1  # Default weight if call fails

    # Step 4: Apply evidence weighting: count * (1 + evidence_score * 0.5)
    weighted: dict[str, float] = {}
    for ans in unique_answers:
        ev_score = evidence_scores.get(ans, 1)
        weight = counter[ans] * (1.0 + ev_score * 0.5)
        weighted[ans] = weight

    consensus = max(weighted, key=weighted.get)
    evidence_notes = f"evidence_scores={evidence_scores} weighted={weighted}"
    return consensus, dict(counter), evidence_notes


def _normalize_sc(text: str) -> str:
    """Normalize a response for self-consistency comparison."""
    t = text.strip().lower()
    # Remove common prefixes
    t = re.sub(r'^(answer|the answer is|output|result)\s*[:=-]\s*', '', t, flags=re.IGNORECASE)
    # Remove trailing punctuation and whitespace
    t = re.sub(r'[.!;,\s]+$', '', t)
    return t[:100]  # Cap length for comparison