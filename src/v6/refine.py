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
    """Cross-agent quality evaluation: an independent LLM judges skill quality.

    This replaces the "immediate failure retry" pattern. In production we don't
    know if a task succeeded, so we use a separate agent to assess whether the
    experience is worth injecting into future tasks.

    Returns:
        dict with keys: total (0-10), verdict ("inject"|"skip"|"low_confidence"),
        actionability, generalizability, correctness, novelty, reason
    """
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