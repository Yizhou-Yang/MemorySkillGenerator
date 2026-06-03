"""Version-Conditioned AI Refinement — ADDS information, never removes."""
from __future__ import annotations
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
2. Extract causal lesson: WHY did this succeed/fail? What's the transferable insight?
3. If version history exists: what improved across attempts? What regressed?
4. DO NOT remove, compress, or summarize any steps. Add context, don't subtract.

## Response (JSON only)
{{
  "generalized_steps": "ALL original steps rewritten with placeholders — same count, same detail level",
  "causal_lesson": "one sentence: why this worked/failed",
  "avoidance_note": "what to avoid (empty string if success)",
  "transferability": "what task types benefit from this experience",
  "evolution_insight": "what version history reveals (empty string if no history)",
  "quality_score": 0-10
}}"""


def _format_patch_history(patch_history: list) -> str:
    """Format version diffs for the AI reviewer."""
    if not patch_history:
        return ""
    lines = ["\n## Version History"]
    for p in patch_history:
        lines.append(f"### v{p.get('from_version','?')} → v{p.get('to_version','?')} (score: {p.get('score_delta',0):+.0%})")
        if p.get("outcome_change"): lines.append(f"  Outcome: {p['outcome_change']}")
        if p.get("fixed_missing"): lines.append(f"  ✅ Fixed: {p['fixed_missing']}")
        if p.get("new_missing"): lines.append(f"  ❌ New gaps: {p['new_missing']}")
        if p.get("new_steps"): lines.append(f"  ➕ Added: {p['new_steps']}")
        if p.get("removed_steps"): lines.append(f"  ➖ Removed: {p['removed_steps']}")
    lines.append("\nUse this history for a STRONGER refinement.\n")
    return "\n".join(lines)


def ai_review_experience(exp: Experience, llm_fn=None) -> dict:
    """Version-conditioned refinement. Passes ALL data to LLM — no pre-truncation.
    Returns structured 7-field dict."""
    if llm_fn is None:
        # Passthrough: return raw data as-is (no information loss)
        return {
            "generalized_steps": "\n".join(exp.action_commands),
            "causal_lesson": exp.failure_reason if exp.outcome != "success" else "Completed all required steps",
            "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
            "transferability": f"Tasks involving: {', '.join(exp.tool_sequence)}",
            "evolution_insight": "",
            "quality_score": int(exp.score * 10),
            "refined": False,
        }

    # Pass ALL steps and ALL missing steps to LLM (no [:10] or [:5] truncation)
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
        import json
        if "{" in response:
            json_str = response[response.index("{"):response.rindex("}") + 1]
            result = json.loads(json_str)
            result["refined"] = True
            result.setdefault("evolution_insight", "")
            return result
    except Exception:
        pass

    # Fallback: passthrough (no information loss)
    return {
        "generalized_steps": "\n".join(exp.action_commands),
        "causal_lesson": exp.failure_reason if exp.outcome != "success" else "Completed all required steps",
        "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
        "transferability": f"Tasks involving: {', '.join(exp.tool_sequence)}",
        "evolution_insight": "",
        "quality_score": int(exp.score * 10),
        "refined": False,
    }
