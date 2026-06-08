"""Version-Conditioned AI Refinement + Cross-Agent Skill Quality Evaluation."""
from __future__ import annotations
from json_repair import repair_json
import json
from .experience import Experience

AI_REVIEW_PROMPT = """You are a skill quality optimizer. REFINE this experience to maximize reusability.

CRITICAL: You must PRESERVE ALL DETAILS. Do not summarize, compress, or remove any steps.
Your job is to ADD generalization (placeholders, causal reasoning) ON TOP of the existing content.

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
2. Extract causal lesson: WHY did this succeed/fail?
3. If version history exists: what improved across attempts? What regressed?
4. DO NOT remove, compress, or summarize any steps.

## Response (JSON only)
{{
  "generalized_steps": "ALL original steps rewritten with placeholders — same count, same detail",
  "causal_lesson": "one sentence: why this worked/failed",
  "avoidance_note": "what to avoid (empty string if success)",
  "transferability": "what task types benefit from this",
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

## Response (JSON only)
{{
  "actionability": 0-3,
  "generalizability": 0-3,
  "correctness": 0-2,
  "novelty": 0-2,
  "total": 0-10,
  "verdict": "inject" | "skip" | "low_confidence",
  "reason": "one sentence justification"
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
    """Version-conditioned refinement. Uses json_repair for robust JSON extraction."""
    if llm_fn is None:
        return {
            "generalized_steps": "\n".join(exp.action_commands),
            "causal_lesson": exp.failure_reason if exp.outcome != "success" else "Completed all required steps",
            "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
            "transferability": f"Tasks involving: {', '.join(exp.tool_sequence)}",
            "evolution_insight": "",
            "quality_score": int(exp.score * 10),
            "refined": False,
        }

    steps_str = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
    missing_str = ", ".join(exp.missing_steps) if exp.missing_steps else "(none)"

    prompt = AI_REVIEW_PROMPT.format(
        task_desc=exp.task_desc, outcome=exp.outcome, score=exp.score,
        steps=steps_str, missing=missing_str,
        failure_reason=exp.failure_reason or "(none)",
        version_history_section=_format_patch_history(exp.patch_history),
    )

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

    return {
        "generalized_steps": "\n".join(exp.action_commands),
        "causal_lesson": exp.failure_reason if exp.outcome != "success" else "Completed all required steps",
        "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
        "transferability": f"Tasks involving: {', '.join(exp.tool_sequence)}",
        "evolution_insight": "",
        "quality_score": int(exp.score * 10),
        "refined": False,
    }

def cross_agent_evaluate_skill(exp: Experience, llm_fn=None) -> dict:
    """Cross-agent quality evaluation: an independent LLM judges skill quality."""
    default = {"total": 5, "verdict": "inject", "reason": "no evaluator available",
               "actionability": 2, "generalizability": 2, "correctness": 1, "novelty": 0}

    if llm_fn is None:
        return default

    steps_str = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
    causal = exp.failure_taxonomy.get("causal_lesson", "")
    generalized = exp.failure_taxonomy.get("generalized_steps", "")

    prompt = CROSS_AGENT_EVAL_PROMPT.format(
        task_desc=exp.task_desc,
        steps=steps_str or "(no steps recorded)",
        outcome=exp.outcome,
        causal_lesson=causal or "(none)",
        generalized_steps=generalized or "(none)",
    )

    try:
        response = llm_fn(prompt)
        repaired = repair_json(response, return_objects=True)
        if isinstance(repaired, dict):
            repaired.setdefault("verdict", "inject" if repaired.get("total", 0) >= 5 else "skip")
            return repaired
        if isinstance(repaired, list) and repaired and isinstance(repaired[0], dict):
            result = repaired[0]
            result.setdefault("verdict", "inject" if result.get("total", 0) >= 5 else "skip")
            return result
    except Exception:
        pass

    return default


CRITIC_REFINE_PROMPT = """You are a skill quality enhancer. A cross-agent critic found this experience LOW QUALITY.
Your job: ENRICH and EXPAND it so it becomes high-quality. DO NOT compress or remove ANY information.

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
3. ADD concrete failure modes: what could go wrong at each step?
4. ADD recovery strategies: if step N fails, what should the agent try?
5. EXPAND causal reasoning: make the WHY more specific and actionable
6. ADD transfer conditions: under what exact conditions does this apply?

## Response (JSON only)
{{
  "enhanced_steps": "ALL original steps PLUS added context/failure-modes/recovery — must be LONGER than input",
  "enhanced_causal_lesson": "deeper causal analysis — more specific than original",
  "enhanced_avoidance": "concrete pitfalls with specific indicators",
  "enhanced_transferability": "exact conditions and task types where this applies",
  "recovery_strategies": "what to do when each step fails",
  "preconditions": "environment/state requirements before attempting this approach",
  "quality_score": 0-10
}}"""


def critic_refine_experience(exp: Experience, critic_verdict: dict, llm_fn=None) -> dict:
    """When critic scores low, REFINE the experience (never discard).
    Enriches with failure modes, recovery strategies, preconditions.
    Never compresses or removes information."""
    if llm_fn is None:
        return {"enhanced": False}

    steps_str = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
    causal = exp.failure_taxonomy.get("causal_lesson", "")
    generalized = exp.failure_taxonomy.get("generalized_steps", "")
    avoidance = exp.failure_taxonomy.get("avoidance_note", "")

    # Identify weak dimensions for targeted improvement
    weak = []
    if critic_verdict.get("actionability", 3) < 2:
        weak.append("actionability (steps not concrete enough)")
    if critic_verdict.get("generalizability", 3) < 2:
        weak.append("generalizability (too task-specific)")
    if critic_verdict.get("correctness", 2) < 1:
        weak.append("correctness (logic may be flawed)")
    if critic_verdict.get("novelty", 2) < 1:
        weak.append("novelty (too obvious)")

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