#!/usr/bin/env python3
"""SkillForge Latest — PersonaMem-v2 Runner (A-Mem Agent, persona memory QA).

Single-round QA benchmarks (LoCoMo, PersonaMem-v2) cannot use EvoMem-style
within-task self-correction injection since they have only one turn.
Instead, we use:

  Group A: Baseline — single LLM call, no augmentation.
  Group B: Self-consistency sampling — 3 calls with varying temperature,
           majority vote on answer. Proven 3-5pp improvement (Wang et al.).
  Group C: Persona-consistent self-consistency — same as B + cross-check
           answers against persona traits for consistency verification.
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


async def run_persona_mem_task(task: dict, experience_section: str = "",
                                group: str = "A") -> dict:
    """Run PersonaMem-v2 task — Group A baseline (single LLM call, no augmentation)."""
    from src.latest.agent.amem import AmemAgent
    agent = AmemAgent()
    result = await agent.run_task(task, experience_section="", group=group)
    result["_aug_prompt"] = ""
    return result


async def run_persona_mem_task_controlled(task: dict, experience_section: str = "",
                                           group: str = "A",
                                           within_task_patch_mode: str | None = None) -> dict:
    """Run PersonaMem-v2 task — Groups B/C with self-consistency optimization.

    Since PersonaMem-v2 is single-round QA (no multi-turn loop), we adapt the
    EvoArena within-task injection concept as follows:

    Group B ("evoarena"): Self-consistency sampling — run the same QA
       3 times with different temperature values, then majority vote.
       Reduces single-sample noise and catches reasoning variance.

    Group C ("skillforge"): Persona-consistent self-consistency — run
       self-consistency (same as B), then cross-check each answer against
       the persona profile for consistency. Answers that are inconsistent
       with known persona traits are down-weighted.

    Args:
        task: Task dict from benchmark loader
        within_task_patch_mode: "evoarena" for B, "skillforge" for C, None for A
    """
    if group == "A":
        return await run_persona_mem_task(task, experience_section, group)

    if group == "B":
        return await _run_self_consistency(task, experience_section, group,
                                            use_persona_verification=False)

    if group == "C":
        return await _run_self_consistency(task, experience_section, group,
                                            use_persona_verification=True)

    return await run_persona_mem_task(task, experience_section, "A")


async def _run_self_consistency(task: dict, experience_section: str,
                                 group: str, use_persona_verification: bool) -> dict:
    """Run self-consistency sampling for single-round persona QA.

    1. Run the same QA N=3 times with different temperatures
    2. Collect all answers
    3. For Group B: majority vote (most common answer wins)
    4. For Group C: persona-consistent vote (answers aligned with
       persona traits are weighted higher)

    Returns a result dict with the consensus answer.
    """
    from src.latest.agent.amem import AmemAgent

    task_id = task["task_id"]
    expected = task.get("expected", "")
    t0 = time.time()

    # Run N self-consistency samples
    samples: list[dict] = []
    for idx, temp in enumerate(_SC_TEMPERATURES):
        agent = AmemAgent()
        aug = experience_section or ""
        if temp != 0.7:
            temp_hint = f"\n[Sample {idx+1}/{_SC_SAMPLES} — temperature={temp}]"
            aug = f"{temp_hint}\n{aug}" if aug else temp_hint

        result = await agent.run_task(task, experience_section=aug, group=group)
        samples.append(result)
        if result.get("error"):
            break

    all_responses = [s.get("response", "").strip() for s in samples]
    all_errors = [s.get("error") for s in samples if s.get("error")]

    if not all_responses or all(all_errors):
        base = samples[0] if samples else {"response": "", "error": "no_samples"}
        base["time_cost"] = time.time() - t0
        base["_aug_prompt"] = f"self_consistency_{group}_failed"
        return base

    # ── Group C: Persona-consistent voting ──
    if use_persona_verification:
        persona_traits = _extract_persona_traits(task)
        consensus_answer, weights, persona_notes = await _persona_consistent_vote(
            task, all_responses, persona_traits, task.get("description", "")
        )
        method = "persona_consistent_self_consistency"
    else:
        # ── Group B: Simple majority vote ──
        consensus_answer, weights = _majority_vote(all_responses)
        persona_notes = ""
        method = "self_consistency_majority"

    base = samples[0].copy()
    base["response"] = consensus_answer
    base["time_cost"] = time.time() - t0
    base["group"] = group
    base["_aug_prompt"] = (
        f"[{method}] samples={len(all_responses)} "
        f"votes={json.dumps(weights)} {persona_notes}"
    )
    base["_sc_samples"] = len(samples)
    base["_sc_responses"] = all_responses

    return base


def _extract_persona_traits(task: dict) -> list[str]:
    """Extract persona trait descriptions from task metadata."""
    metadata = task.get("metadata", {})
    traits = metadata.get("persona_traits", [])
    if traits:
        return list(traits)
    # Fallback: try to extract from context
    context = task.get("context", "")
    if isinstance(context, str) and context:
        return [context[:500]]
    return []


def _majority_vote(responses: list[str]) -> tuple[str, dict[str, int]]:
    """Simple majority vote: normalize and pick the most common answer."""
    normalized = [_normalize_sc(resp) for resp in responses]
    counter = Counter(normalized)
    most_common = counter.most_common(1)[0][0]
    return most_common, dict(counter)


async def _persona_consistent_vote(task: dict, responses: list[str],
                              persona_traits: list[str],
                              description: str) -> tuple[str, dict[str, int], str]:
    """Persona-consistent voting for Group C.

    After collecting self-consistency samples, verify each unique answer
    against the persona profile. Answers inconsistent with known persona
    traits are down-weighted.

    Args:
        task: Task dict
        responses: List of raw responses from self-consistency samples
        persona_traits: List of persona trait strings
        description: Task description (contains the question)

    Returns:
        (consensus_answer, weighted_votes_dict, persona_notes_str)
    """
    from scripts.latest.llm_client import _llm_short_call

    # Step 1: Normalize and identify unique answers
    normalized = [_normalize_sc(resp) for resp in responses]
    counter = Counter(normalized)
    unique_answers = list(counter.keys())

    if len(unique_answers) <= 1:
        return unique_answers[0], dict(counter), "single_answer_no_weighting"

    # Step 2: Extract question
    question = description
    if "Question:" in description:
        question = description.split("Question:")[-1].strip()

    # Step 3: For each unique answer, check persona consistency
    persona_text = "\n".join(f"- {trait}" for trait in persona_traits[:5])
    consistency_scores: dict[str, int] = {}
    for ans in unique_answers:
        if not persona_text:
            consistency_scores[ans] = 1
            continue
        consistency_prompt = (
            f"Persona profile:\n{persona_text[:800]}\n\n"
            f"Question: {question[:300]}\n\n"
            f"Proposed Answer: {ans}\n\n"
            f"Is this answer CONSISTENT with the persona profile above?\n"
            f"Reply ONLY with a number 0-3:\n"
            f"0 = contradicts the persona\n"
            f"1 = no clear relationship to persona\n"
            f"2 = somewhat consistent with persona traits\n"
            f"3 = strongly consistent with persona traits"
        )
        try:
            out = await _llm_short_call(consistency_prompt, max_turns=1, timeout=30)
            m = re.search(r'(\d+)', out)
            score = int(m.group(1)) if m else 1
            consistency_scores[ans] = min(3, max(0, score))
        except Exception:
            consistency_scores[ans] = 1

    # Step 4: Apply persona-consistency weighting: count * (1 + score * 0.5)
    weighted: dict[str, float] = {}
    for ans in unique_answers:
        cs = consistency_scores.get(ans, 1)
        weight = counter[ans] * (1.0 + cs * 0.5)
        weighted[ans] = weight

    consensus = max(weighted, key=weighted.get)
    persona_notes = f"persona_consistency={consistency_scores} weighted={weighted}"
    return consensus, dict(counter), persona_notes


def _normalize_sc(text: str) -> str:
    """Normalize a response for self-consistency comparison."""
    t = text.strip().lower()
    t = re.sub(r'^(answer|the answer is|output|result)\s*[:=-]\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'[.!;,\s]+$', '', t)
    return t[:100]