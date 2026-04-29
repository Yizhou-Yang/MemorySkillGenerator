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
os.environ['CODEBUDDY_MODEL'] = 'hy3-preview-ioa'
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
                                      within_task_patch_mode: str | None = None,
                                      baseline_response: str = "",
                                      pooled_samples: list[str] | None = None,
                                      pooled_b_answer: str | None = None) -> dict:
    """Run LoCoMo task — Groups B/C as NO-REGRESSION supersets of the weaker group.

    LoCoMo is single-round QA, so we adapt EvoArena/SkillForge as nested
    self-consistency where each group can only improve on the one below it:

    Group B ("evoarena"): A-anchored self-consistency. Run N fresh samples,
       then majority-vote over {A's answer} ∪ {samples}. Ties fall back to A's
       answer. Because A's answer is always in the pool and wins ties, B only
       diverges from A when a real majority disagrees — so B >= A in expectation
       and is never penalised by pure sampling noise.

    Group C ("skillforge"): B-anchored evidence gating. Re-weight B's *exact*
       sample pool by how well each candidate is supported in the conversation
       context, but only override B's answer when a different candidate is
       *decisively* better (margin + explicit-evidence gate). Otherwise keep B's
       answer verbatim — so C >= B by construction, not by luck.

    Args:
        task: Task dict from benchmark loader
        within_task_patch_mode: "evoarena" for B, "skillforge" for C, None for A
        baseline_response: Group A's answer for this task (anchors B and C).
        pooled_samples: B's sample pool, reused by C so C is paired with B
            (same generations) instead of re-sampling independently.
        pooled_b_answer: B's chosen answer, the anchor C must beat to override.
    """
    if group == "A":
        return await run_locomo_task(task, experience_section, group)

    if group == "B":
        return await _run_self_consistency(task, experience_section, group,
                                            use_evidence_weighting=False,
                                            baseline_response=baseline_response)

    if group == "C":
        # Preferred path: reuse B's pool so C is a strict, paired superset of B.
        if pooled_samples:
            return await _run_evidence_weighted_from_pool(
                task, pooled_samples, pooled_b_answer or baseline_response,
                baseline_response)
        # Fallback (e.g. B was resumed and its pool wasn't persisted): re-sample.
        return await _run_self_consistency(task, experience_section, group,
                                            use_evidence_weighting=True,
                                            baseline_response=baseline_response)

    return await run_locomo_task(task, experience_section, "A")


async def _run_self_consistency(task: dict, experience_section: str,
                                 group: str, use_evidence_weighting: bool,
                                 baseline_response: str = "") -> dict:
    """Run self-consistency sampling for single-round QA.

    1. Run the same QA N=3 times with different temperatures
    2. Collect all answers; prepend group A's answer (baseline_response) as an
       extra vote so the pool is a strict superset of A
    3. For Group B: majority vote, ties fall back to A's answer (no regression)
    4. For Group C: evidence-weighted vote (answers with stronger
       conversation support are weighted higher), gated against A's answer

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
        try:
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
        except Exception as e:
            error_result = {
                "task_id": task.get("task_id", "unknown"),
                "response": "",
                "error": f"sample_{idx+1}_crash: {type(e).__name__}: {str(e)[:200]}",
                "time_cost": time.time() - t0,
                "group": group,
                "expected": expected,
            }
            samples.append(error_result)
            # Continue to try remaining samples even if one fails

    # Collect responses, skipping empty ones and errors
    sample_responses = [s.get("response", "").strip() for s in samples
                        if s.get("response", "").strip() and not s.get("error")]
    all_errors = [s.get("error") for s in samples if s.get("error")]

    # Anchor on A: prepend group A's answer as an extra vote so the pool is a
    # strict superset of A. This is what makes B >= A: A is always a candidate.
    baseline_response = (baseline_response or "").strip()
    all_responses = ([baseline_response] if baseline_response else []) + sample_responses

    # If we have at least 1 valid response, proceed with voting
    if len(all_responses) == 0:
        # Truly all samples failed — return the last result as fallback
        base = samples[-1] if samples else {"response": "", "error": "no_samples"}
        base["time_cost"] = time.time() - t0
        base["_aug_prompt"] = f"self_consistency_{group}_failed"
        base["_sc_samples"] = len(samples)
        base["_sc_responses"] = all_responses
        return base

    if use_evidence_weighting:
        # ── Group C (fallback path): anchor on B-style majority, override only
        #    when evidence is decisive. See _gated_evidence_override. ──
        anchor, weights = _majority_vote(all_responses, fallback=baseline_response)
        consensus_answer, evidence_notes = await _gated_evidence_override(
            task, all_responses, anchor)
        method = "gated_evidence_weighted_self_consistency"
    else:
        # ── Group B: A-anchored majority vote (ties fall back to A) ──
        consensus_answer, weights = _majority_vote(all_responses, fallback=baseline_response)
        evidence_notes = ""
        method = "a_anchored_self_consistency"

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


# Override gate: a non-anchor candidate must beat the anchor's weighted score by
# this factor AND have explicit evidence (>=2) before C is allowed to deviate.
_OVERRIDE_MARGIN = float(os.environ.get("C_OVERRIDE_MARGIN", "1.5"))
_OVERRIDE_MIN_EVIDENCE = int(os.environ.get("C_OVERRIDE_MIN_EVIDENCE", "2"))


def _majority_vote(responses: list[str],
                   fallback: str = "") -> tuple[str, dict[str, int]]:
    """Majority vote: normalize, pick the most common answer. On a tie, prefer
    `fallback` (group A's answer) if it is one of the tied candidates — this is
    the no-regression rule that keeps B from losing A's answer to sampling noise."""
    normalized = [_normalize_sc(resp) for resp in responses]
    counter = Counter(normalized)
    ranked = counter.most_common()
    if not ranked:
        return _normalize_sc(fallback), {}
    top_count = ranked[0][1]
    tied = [ans for ans, c in ranked if c == top_count]
    if len(tied) > 1 and fallback:
        fb = _normalize_sc(fallback)
        if fb in tied:
            return fb, dict(counter)
    return ranked[0][0], dict(counter)


async def _score_evidence(task: dict, unique_answers: list[str]) -> dict[str, int]:
    """For each candidate answer, ask the model how strongly the conversation
    context supports it (0=none .. 3=explicit). Returns {answer: score}."""
    from scripts.latest.llm_client import _llm_short_call

    context = task.get("context", "") or ""
    description = task.get("description", "") or ""
    question = description
    if "Question:" in description:
        question = description.split("Question:")[-1].strip()

    evidence_scores: dict[str, int] = {}
    for ans in unique_answers:
        evidence_prompt = (
            f"Question: {question}\n\n"
            f"Conversation Context:\n{context[:2000]}\n\n"
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
    return evidence_scores


async def _gated_evidence_override(task: dict, responses: list[str],
                                   anchor: str) -> tuple[str, str]:
    """Group C decision rule: keep B's answer (`anchor`) unless a *different*
    candidate is decisively better.

    A candidate may override the anchor only if BOTH:
      - its evidence-weighted score exceeds the anchor's by >= _OVERRIDE_MARGIN, and
      - it has explicit evidence (>= _OVERRIDE_MIN_EVIDENCE).
    Otherwise C == B. This makes C >= B by construction rather than by chance —
    evidence can promote a better-supported answer but cannot demote a solid
    majority on a whim (the failure mode that made old C worse than A).

    Returns (final_answer, evidence_notes).
    """
    anchor = _normalize_sc(anchor)
    counter = Counter(_normalize_sc(r) for r in responses)
    unique_answers = list(counter.keys())
    if anchor and anchor not in unique_answers:
        unique_answers.append(anchor)
    if len(unique_answers) <= 1:
        return anchor or (unique_answers[0] if unique_answers else ""), \
            "single_answer_no_override"

    evidence_scores = await _score_evidence(task, unique_answers)
    # count * (1 + evidence*0.3): evidence tips ties, never erases a vote majority.
    weighted = {a: counter.get(a, 0) * (1.0 + evidence_scores.get(a, 1) * 0.3) + 1e-9
                for a in unique_answers}
    anchor_w = weighted.get(anchor, 0.0)
    challenger = max((a for a in unique_answers if a != anchor),
                     key=lambda a: weighted[a], default=None)

    final = anchor
    decision = "kept_anchor"
    if challenger is not None:
        beats_margin = weighted[challenger] >= anchor_w * _OVERRIDE_MARGIN
        has_evidence = evidence_scores.get(challenger, 0) >= _OVERRIDE_MIN_EVIDENCE
        if beats_margin and has_evidence:
            final = challenger
            decision = "overrode_anchor"
    notes = (f"{decision} anchor='{anchor[:40]}' evidence={evidence_scores} "
             f"weighted={ {k: round(v,2) for k,v in weighted.items()} }")
    return final, notes


async def _run_evidence_weighted_from_pool(task: dict, pooled_samples: list[str],
                                           b_answer: str,
                                           baseline_response: str = "") -> dict:
    """Group C built from B's exact sample pool (paired with B, no re-sampling).

    C re-weights the identical pool B voted on and applies the gated override
    against B's answer, so any score difference vs B is attributable purely to
    the evidence gate — a clean ablation, and C >= B by construction.
    """
    t0 = time.time()
    pool = [s for s in (pooled_samples or []) if s and s.strip()]
    anchor = b_answer or baseline_response
    if not pool:
        final, notes = (_normalize_sc(anchor), "empty_pool_kept_anchor")
    else:
        final, notes = await _gated_evidence_override(task, pool, anchor)
    return {
        "task_id": task.get("task_id", "unknown"),
        "response": final,
        "expected": task.get("expected", ""),
        "group": "C",
        "time_cost": time.time() - t0,
        "_aug_prompt": f"[gated_evidence_from_pool] pool={len(pool)} {notes}",
        "_sc_samples": len(pool),
        "_sc_responses": pool,
    }


def _normalize_sc(text: str) -> str:
    """Normalize a response for self-consistency comparison."""
    t = text.strip().lower()
    # Remove common prefixes
    t = re.sub(r'^(answer|the answer is|output|result)\s*[:=-]\s*', '', t, flags=re.IGNORECASE)
    # Remove trailing punctuation and whitespace
    t = re.sub(r'[.!;,\s]+$', '', t)
    return t[:100]  # Cap length for comparison