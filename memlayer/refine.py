"""Version-conditioned AI refinement + cross-agent skill quality evaluation.

Refinement restructures raw action sequences into generalized skills; the critic
scores them and forces enrichment of weak entries (never discards).
"""
from __future__ import annotations
from json_repair import repair_json
import json
import os
import time
from .experience import Experience

AI_REVIEW_PROMPT = """You are a skill quality optimizer. REFINE this experience to maximize reusability.

## CRITICAL CONSTRAINTS
1. ZERO INFORMATION LOSS: You must PRESERVE ALL DETAILS. Do not summarize, compress, or remove any steps.
   Your job is to ADD generalization (placeholders, causal reasoning) ON TOP of the existing content.
2. ZERO NOISE: Every field you output must contain ACTIONABLE, TRANSFERABLE content.
   - "causal_lesson" must explain a SPECIFIC mechanism (not "it worked" or "completed all steps")
   - "generalized_steps" must contain CONCRETE actions (not vague descriptions)
   - "avoidance_note" must describe SPECIFIC pitfalls (not generic warnings)
   - If the experience is a success with no failures, the causal_lesson should explain
     WHAT STRATEGY made it work (e.g., "Used binary search instead of linear scan")
   - NEVER output trivial content like "Completed all required steps" — that is noise, not a skill.

## Experience
Task: {task_desc}
Outcome: {outcome} (score: {score:.0%})
Steps taken:
{steps}
Missing steps: {missing}
Failure reason: {failure_reason}
{version_history_section}
## Instructions
1. Generalize: replace hard-coded IDs/dates/names with [PLACEHOLDER], but KEEP EVERY STEP.
2. Extract causal lesson: WHY did this succeed/fail? Must be a SPECIFIC mechanism, not a tautology.
3. If version history exists: what improved across attempts? What regressed?
4. DO NOT remove, compress, or summarize any steps.
5. For successes: identify the KEY STRATEGY or TOOL CHAIN that made it work.
   The lesson must be transferable to similar-but-different tasks.
6. For failures: identify the ROOT CAUSE and what SPECIFIC alternative approach should be tried.

## Response (JSON only)
{{
  "generalized_steps": "ALL original steps rewritten with placeholders — same count, same detail",
  "causal_lesson": "one sentence: the SPECIFIC mechanism/strategy that caused success/failure (NEVER 'completed all steps')",
  "avoidance_note": "SPECIFIC pitfall to avoid with concrete indicators (empty string ONLY if pure success with no lessons)",
  "transferability": "EXACT task types and conditions where this skill applies",
  "evolution_insight": "what version history reveals (empty string if no history)",
  "quality_score": 0-10
}}"""

# ── FAILURE-SPECIFIC refinement prompt ──────────────────────────────────────
# Retry when the first refinement yields an empty causal_lesson/avoidance_note.
AI_REVIEW_FAILURE_RETRY_PROMPT = """You are a FAILURE ANALYSIS specialist. The first refinement attempt produced
a trivial or empty causal lesson. You MUST extract a STRATEGIC-LEVEL analysis.

## CRITICAL DISTINCTION
- FACTUAL failure (USELESS, do NOT output): "The answer is X, not Y"
  → This has ZERO transfer value. Skip this entirely.
- STRATEGIC failure (VALUABLE, MUST output): "The tool chain failed because step N
  used a method that doesn't work for this data type."
  → This is transferable to other tasks.

## Experience
Task: {task_desc}
Outcome: {outcome} (score: {score:.0%})
Steps taken:
{steps}
Missing steps: {missing}
Failure reason: {failure_reason}
{previous_result_section}

## MANDATORY OUTPUT REQUIREMENTS
1. **causal_lesson**: MUST explain the STRATEGIC reason for failure.
   - BAD: "the count was wrong" (factual, useless)
   - GOOD: "web search returned stale data because the query lacked a date filter" (strategic, transferable)
   - GOOD: "PDF extraction failed because the tool was used on scanned images instead of digital text" (strategic)
   - GOOD: "multi-hop question failed because step 2 depended on a wrong assumption from step 1" (strategic)
2. **avoidance_note**: MUST name a SPECIFIC action pattern to avoid, with indicators.
   - BAD: "be more careful" (vague, useless)
   - GOOD: "do not use WebFetch on PDFs containing scanned images; check for [OCR needed] indicator first"
3. **generalized_steps**: MUST be PROCEDURAL patterns, NOT raw search queries.
   - BAD: "1. search 'Unlambda evaluation order 1960s article'" (raw query, useless)
   - GOOD: "1. search for [TECHNICAL_TERM] combined with [TIME_PERIOD] and [DOCUMENT_TYPE]"
   - Replace ALL concrete values with [PLACEHOLDER] descriptions of what they REPRESENT.
   - Do NOT just wrap the original query in brackets — describe the SEMANTIC ROLE of each term.

## Response (JSON only)
{{
  "generalized_steps": "ALL steps rewritten with [SEMANTIC_PLACEHOLDERS] — describe WHAT each term represents",
  "causal_lesson": "STRATEGIC reason for failure — NOT factual — one specific, transferable sentence",
  "avoidance_note": "SPECIFIC action pattern to avoid with concrete indicators",
  "transferability": "EXACT task types and conditions where this lesson applies",
  "evolution_insight": "what version history reveals (empty string if no history)",
  "quality_score": 0-10
}}"""

CROSS_AGENT_EVAL_PROMPT = """You are an independent quality evaluator for AI agent skills/experiences.
Evaluate whether this experience is high-quality and worth injecting into future tasks.

## Experience to Evaluate
Task: {task_desc}
Approach taken:
{steps}
Claimed outcome: {outcome}
Causal lesson: {causal_lesson}
Generalized steps: {generalized_steps}

## Evaluation Criteria
1. Actionability (0-3): Are the steps concrete and reproducible?
2. Generalizability (0-3): Would this help on DIFFERENT but similar tasks?
3. Correctness (0-2): Does the approach seem logically sound?
4. Novelty (0-2): Does it provide non-obvious insight?

## CRITICAL: Check for these DISQUALIFYING issues
- NOISE: Is the causal_lesson trivial/tautological? (e.g., "it worked", "completed all steps")
  → If yes, score Novelty=0 and flag as noise in reason.
- INFORMATION LOSS: Are the generalized_steps vague/compressed compared to the original steps?
  → If yes, score Actionability=0 and flag as information loss in reason.
- OVERFITTING: Is the content too task-specific to transfer? (e.g., hardcoded file paths, specific IDs)
  → If yes, score Generalizability=0.

If ANY disqualifying issue is found, verdict MUST be "low_confidence" (triggers forced re-refinement).

## Separately: judge the OUTCOME itself
Independent of the record's quality, judge whether the attempt's final
answer/result is likely CORRECT for the task, using only the task and the
approach shown (no reference answer). Be strict: "correct" only when you
would bet on this answer; when in doubt say "unsure".

## Response (JSON only)
{{
  "actionability": 0-3,
  "generalizability": 0-3,
  "correctness": 0-2,
  "novelty": 0-2,
  "total": 0-10,
  "verdict": "inject" | "skip" | "low_confidence",
  "outcome_verdict": "correct" | "wrong" | "unsure",
  "reason": "one sentence justification",
  "noise_detected": true/false,
  "info_loss_detected": true/false
}}"""

def _format_patch_history(patch_history: list) -> str:
    if not patch_history:
        return ""
    lines = ["\n## Version History"]
    for p in patch_history:
        lines.append(f"### v{p.get('from_version','?')} → v{p.get('to_version','?')} (score: {p.get('score_delta',0):+.0%})")
        if p.get("outcome_change"):
            lines.append(f"  Outcome: {p['outcome_change']}")
        if p.get("fixed_missing"):
            lines.append(f"  Fixed: {p['fixed_missing']}")
        if p.get("new_missing"):
            lines.append(f"  New gaps: {p['new_missing']}")
        if p.get("new_steps"):
            lines.append(f"  Added: {p['new_steps']}")
        if p.get("removed_steps"):
            lines.append(f"  Removed: {p['removed_steps']}")
    lines.append("\nUse this history for a STRONGER refinement.\n")
    return "\n".join(lines)

def ai_review_experience(exp: Experience, llm_fn=None) -> dict:
    """Version-conditioned refinement with a quality self-check for failures.

    Failures whose first pass yields an empty lesson/note are re-refined with a
    stricter prompt. llm_fn=None returns an unrefined fallback with all originals.
    """
    if llm_fn is None:
        return {
            "generalized_steps": "\n".join(exp.action_commands),
            "causal_lesson": "",
            "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
            "transferability": "",
            "evolution_insight": "",
            "quality_score": 0,
            "refined": False,
        }

    steps_str = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
    missing_str = ", ".join(exp.missing_steps) if exp.missing_steps else "(none)"

    reasoning_section = ""
    if exp.reasoning_trace:
        reasoning_lines = "\n".join(f"  - {r}" for r in exp.reasoning_trace[:10])
        reasoning_section = f"\nAgent's reasoning during execution:\n{reasoning_lines}\n"

    prompt = AI_REVIEW_PROMPT.format(
        task_desc=exp.task_desc, outcome=exp.outcome, score=exp.score,
        steps=steps_str, missing=missing_str,
        failure_reason=exp.failure_reason or "(none)",
        version_history_section=_format_patch_history(exp.patch_history),
    )
    if reasoning_section:
        prompt += reasoning_section

    result = _call_refine_llm(prompt, llm_fn)

    # ── Quality self-check for failures ─────────────────────────────────────
    if (exp.outcome != "success" and result and result.get("refined") and
            (not result.get("causal_lesson", "").strip() or
             not result.get("avoidance_note", "").strip())):
        prev_section = (
            f"\n## Previous refinement attempt (REJECTED - missing strategic analysis)\n"
            f"Previous causal_lesson: '{result.get('causal_lesson', '') or '(EMPTY)'}'\n"
            f"Previous avoidance_note: '{result.get('avoidance_note', '') or '(EMPTY)'}'\n"
            f"Previous generalized_steps: '{result.get('generalized_steps', '')[:300]}'\n"
            f"\nThe above was rejected because it lacked STRATEGIC-LEVEL analysis.\n"
        )
        retry_prompt = AI_REVIEW_FAILURE_RETRY_PROMPT.format(
            task_desc=exp.task_desc, outcome=exp.outcome, score=exp.score,
            steps=steps_str, missing=missing_str,
            failure_reason=exp.failure_reason or "(none)",
            previous_result_section=prev_section,
        )
        if reasoning_section:
            retry_prompt += reasoning_section

        retry_result = _call_refine_llm(retry_prompt, llm_fn)
        if retry_result and retry_result.get("refined"):
            if (retry_result.get("causal_lesson", "").strip() and
                    retry_result.get("avoidance_note", "").strip()):
                return retry_result

    return result if result else _unrefined_fallback(exp)


def _call_refine_llm(prompt: str, llm_fn) -> dict | None:
    """Call LLM for refinement and parse JSON response. Returns None on failure."""
    try:
        response = llm_fn(prompt)
        repaired = repair_json(response, return_objects=True)
        if isinstance(repaired, dict):
            repaired["refined"] = True
            repaired.setdefault("evolution_insight", "")
            return repaired
        if isinstance(repaired, list) and repaired and isinstance(repaired[0], dict):
            result = repaired[0]
            result["refined"] = True
            result.setdefault("evolution_insight", "")
            return result
    except Exception:
        pass
    return None


def _unrefined_fallback(exp: Experience) -> dict:
    """Minimal fallback preserving original data, marked as unrefined."""
    return {
        "generalized_steps": "\n".join(exp.action_commands),
        "causal_lesson": "",
        "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
        "transferability": "",
        "evolution_insight": "",
        "quality_score": 0,
        "refined": False,
    }

# Critic-call resilience. The default verdict is total=5, which is the inject
# threshold, so an unavailable critic is indistinguishable from a mediocre one
# unless failures are retried and then counted.
_CRITIC_TRIES = int(os.environ.get("C_CRITIC_TRIES", "4"))
_CRITIC_BACKOFF_S = float(os.environ.get("C_CRITIC_BACKOFF_S", "3"))
_CRITIC_FAILURES = {"n": 0}


def critic_failure_count() -> int:
    """How many experiences were graded by the fallback rather than the critic."""
    return _CRITIC_FAILURES["n"]


def cross_agent_evaluate_skill(exp: Experience, llm_fn=None) -> dict:
    """Cross-agent quality evaluation: an independent LLM judges skill quality.

    Detects noise, information loss and overfitting; verdict='low_confidence'
    triggers critic_refine_experience (forced enrichment, never discard).
    """
    default = {"total": 5, "verdict": "inject", "reason": "no evaluator available",
               "outcome_verdict": "unsure", "critic_failed": True,
               "actionability": 2, "generalizability": 2, "correctness": 1, "novelty": 0}

    if llm_fn is None:
        return default

    steps_str = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
    causal = exp.failure_taxonomy.get("causal_lesson", "")
    generalized = exp.failure_taxonomy.get("generalized_steps", "")

    reasoning_context = ""
    if exp.reasoning_trace:
        reasoning_lines = "\n".join(f"  - {r}" for r in exp.reasoning_trace[:8])
        reasoning_context = f"\nAgent reasoning during execution:\n{reasoning_lines}"

    prompt = CROSS_AGENT_EVAL_PROMPT.format(
        task_desc=exp.task_desc,
        steps=steps_str or "(no steps recorded)",
        outcome=exp.outcome,
        causal_lesson=causal or "(none)",
        generalized_steps=generalized or "(none)",
    )
    if reasoning_context:
        prompt += reasoning_context

    # The gateway serving the critic returns 503 in bursts (measured 2026-08-20:
    # one model 0/24, two others ~25% failures at concurrency 8). A failed call
    # used to fall straight through to `default`, which is total=5 -- exactly the
    # inject threshold -- so an outage silently graded every experience as barely
    # passing and nothing in the trace said so. Retry first, and when every try
    # fails mark the result so the row is countable rather than indistinguishable
    # from a real score of 5.
    last = ""
    for attempt in range(_CRITIC_TRIES):
        try:
            response = llm_fn(prompt)
            repaired = repair_json(response, return_objects=True)
            if isinstance(repaired, list) and repaired and isinstance(repaired[0], dict):
                repaired = repaired[0]
            if isinstance(repaired, dict) and repaired:
                repaired.setdefault("verdict",
                                    "inject" if repaired.get("total", 0) >= 5 else "skip")
                repaired.setdefault("outcome_verdict", "unsure")
                repaired["critic_failed"] = False
                return repaired
            last = "unparseable response"
        except Exception as e:                       # noqa: BLE001 - reported below
            last = f"{type(e).__name__}: {str(e)[:80]}"
        if attempt + 1 < _CRITIC_TRIES:
            time.sleep(_CRITIC_BACKOFF_S * (attempt + 1))
    out = dict(default)
    out["reason"] = f"critic unavailable after {_CRITIC_TRIES} tries ({last})"
    _CRITIC_FAILURES["n"] += 1
    if _CRITIC_FAILURES["n"] <= 3 or _CRITIC_FAILURES["n"] % 25 == 0:
        print(f"  [critic] FAILED {_CRITIC_FAILURES['n']}x -- experience graded at the "
              f"default 5/10, not by the critic ({last})", flush=True)
    return out


CRITIC_REFINE_PROMPT = """You are a skill quality enhancer. A cross-agent critic found this experience LOW QUALITY.
Your job: ENRICH and EXPAND it so it becomes high-quality. DO NOT compress or remove ANY information.

## CRITICAL CONSTRAINTS
1. ZERO INFORMATION LOSS: KEEP every original step and detail intact. Your output must be LONGER than input.
2. ZERO NOISE: Every sentence you add must be ACTIONABLE and SPECIFIC.
   - Do NOT add vague platitudes ("be careful", "ensure correctness")
   - DO add concrete failure modes with specific indicators
   - DO add exact recovery commands/strategies
   - DO add measurable preconditions

## Original Experience
Task: {task_desc}
Outcome: {outcome} (score: {score:.0%})
Steps taken:
{steps}
Causal lesson: {causal_lesson}
Generalized steps: {generalized_steps}
Avoidance note: {avoidance_note}

## Critic Feedback
Score: {critic_total}/10
Reason: {critic_reason}
Weak dimensions: {weak_dimensions}

## Your Job
1. KEEP every original step and detail intact — do NOT summarize or compress
2. ADD missing context: what environment setup is needed? what preconditions?
3. ADD concrete failure modes: what could go wrong at each step? (with specific error patterns)
4. ADD recovery strategies: if step N fails, what EXACT command/action should the agent try?
5. EXPAND causal reasoning: make the WHY more specific and actionable
6. ADD transfer conditions: under what exact conditions does this apply?

## Response (JSON only)
{{
  "enhanced_steps": "ALL original steps PLUS added context/failure-modes/recovery — must be LONGER than input",
  "enhanced_causal_lesson": "deeper causal analysis — more specific than original",
  "enhanced_avoidance": "concrete pitfalls with specific indicators (error messages, symptoms)",
  "enhanced_transferability": "exact conditions and task types where this applies",
  "recovery_strategies": "what to do when each step fails (specific commands/actions)",
  "preconditions": "environment/state requirements before attempting this approach",
  "quality_score": 0-10
}}"""


def critic_refine_experience(exp: Experience, critic_verdict: dict, llm_fn=None) -> dict:
    """When the critic scores low, enrich the experience (never discard).

    Adds recovery strategies, preconditions and deeper causal reasoning; the
    prompt requires output LONGER than input (zero information loss).
    """
    if llm_fn is None:
        return {"enhanced": False}

    steps_str = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
    causal = exp.failure_taxonomy.get("causal_lesson", "")
    generalized = exp.failure_taxonomy.get("generalized_steps", "")
    avoidance = exp.failure_taxonomy.get("avoidance_note", "")

    weak = []
    if critic_verdict.get("actionability", 3) < 2:
        weak.append("actionability (steps not concrete enough)")
    if critic_verdict.get("generalizability", 3) < 2:
        weak.append("generalizability (too task-specific)")
    if critic_verdict.get("correctness", 2) < 1:
        weak.append("correctness (logic may be flawed)")
    if critic_verdict.get("novelty", 2) < 1:
        weak.append("novelty (too obvious)")

    reasoning_context = ""
    if exp.reasoning_trace:
        reasoning_lines = "\n".join(f"  - {r}" for r in exp.reasoning_trace[:10])
        reasoning_context = f"\n\nAgent's reasoning during execution:\n{reasoning_lines}"

    prompt = CRITIC_REFINE_PROMPT.format(
        task_desc=exp.task_desc,
        outcome=exp.outcome,
        score=exp.score,
        steps=steps_str or "(no steps recorded)",
        causal_lesson=causal or "(none)",
        generalized_steps=generalized or "(none)",
        avoidance_note=avoidance or "(none)",
        critic_total=critic_verdict.get("total", 0),
        critic_reason=critic_verdict.get("reason", "low quality"),
        weak_dimensions=", ".join(weak) if weak else "general quality",
    )
    if reasoning_context:
        prompt += reasoning_context

    try:
        response = llm_fn(prompt)
        repaired = repair_json(response, return_objects=True)
        if isinstance(repaired, dict):
            repaired["enhanced"] = True
            return repaired
        if isinstance(repaired, list) and repaired and isinstance(repaired[0], dict):
            result = repaired[0]
            result["enhanced"] = True
            return result
    except Exception:
        pass

    return {"enhanced": False}