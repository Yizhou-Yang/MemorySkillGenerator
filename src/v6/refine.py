"""
SkillForge V6 — Version-Conditioned AI Refinement.

Unlike stateless reflection (Reflexion, ExpeL), this refinement is VERSION-CONDITIONED:
each iteration sees the full patch_history (v1→v2→...→vN diff chain), enabling the
refinement to synthesize cumulative lessons across all prior attempts.

Output: structured 7-field dict (not free-form natural language).
"""
from __future__ import annotations
from .experience import Experience


AI_REVIEW_PROMPT = """You are a skill quality optimizer. Your job is to REFINE an experience record to maximize its value for future similar tasks.

## Experience to Refine
Task: {task_desc}
Outcome: {outcome} (score: {score:.0%})
Steps taken:
{steps}
Missing steps: {missing}
Failure reason: {failure_reason}
{version_history_section}
## Your Job

1. **Generalize** the experience: replace hard-coded IDs, dates, and names with descriptive placeholders, but KEEP all structural details.
2. **Extract causal lesson**: WHY did this succeed/fail? What's the transferable insight?
3. **Preserve all details**: Do NOT remove steps or simplify. Add context, don't subtract.
4. **Learn from version history** (if available): identify what improved across attempts, what patterns of failure recurred, and synthesize a cumulative lesson that incorporates all prior attempts.

## Response Format (JSON only)
{{
  "generalized_steps": "The steps rewritten with placeholders instead of hard-coded values, but same level of detail",
  "causal_lesson": "One clear sentence: why this worked / why this failed, informed by version history if available",
  "avoidance_note": "What to avoid next time (empty if success), incorporating patterns from prior failed versions",
  "transferability": "What types of tasks can benefit from this experience",
  "evolution_insight": "What the version history reveals about solving this type of task (empty if no history)",
  "quality_score": 0-10
}}"""


def _format_patch_history(patch_history: list) -> str:
    """Format patch history into a readable version-diff section for the AI reviewer.
    
    This is what makes our refinement VERSION-CONDITIONED:
    the LLM sees the full evolution trace before producing its refinement.
    """
    if not patch_history:
        return ""
    
    lines = ["\n## Version History (previous attempts on this same/similar task)"]
    lines.append("Each entry shows what changed between consecutive attempts:\n")
    
    for p in patch_history:
        from_v = p.get("from_version", "?")
        to_v = p.get("to_version", "?")
        score_delta = p.get("score_delta", 0)
        outcome_change = p.get("outcome_change", "")
        new_steps = p.get("new_steps", [])
        removed_steps = p.get("removed_steps", [])
        fixed_missing = p.get("fixed_missing", [])
        new_missing = p.get("new_missing", [])
        
        lines.append(f"### v{from_v} → v{to_v} (score: {score_delta:+.0%})")
        if outcome_change:
            lines.append(f"  Outcome: {outcome_change}")
        if fixed_missing:
            lines.append(f"  ✅ Fixed (previously missing): {fixed_missing}")
        if new_missing:
            lines.append(f"  ❌ New gaps: {new_missing}")
        if new_steps:
            lines.append(f"  ➕ Added steps: {new_steps}")
        if removed_steps:
            lines.append(f"  ➖ Removed steps: {removed_steps}")
        lines.append("")
    
    lines.append("Use this history to produce a STRONGER refinement — learn from what improved and what regressed across versions.")
    lines.append("")
    return "\n".join(lines)


def ai_review_experience(exp: Experience, llm_fn=None) -> dict:
    """Version-conditioned AI refinement.
    
    Key difference from Reflexion/ExpeL:
    - Reflexion: stateless, each reflection is independent
    - ExpeL: one-time extraction, no structured output
    - Ours: sees full patch_history diff chain → structured 7-field output
            that ACCUMULATES lessons across versions
    
    Args:
        exp: Experience to refine (may contain patch_history)
        llm_fn: Callable (prompt: str) -> str. If None, returns passthrough.
    
    Returns:
        Dict with: generalized_steps, causal_lesson, avoidance_note,
                   transferability, evolution_insight, quality_score, refined
    """
    if llm_fn is None:
        return {
            "generalized_steps": "\n".join(exp.action_commands[:10]),
            "causal_lesson": exp.failure_reason if exp.outcome != "success" else "Completed all required steps",
            "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
            "transferability": f"Tasks involving: {', '.join(exp.tool_sequence[:3])}",
            "evolution_insight": "",
            "quality_score": int(exp.score * 10),
            "refined": False,
        }
    
    steps_str = "\n".join(f"  {i+1}. {cmd[:120]}" for i, cmd in enumerate(exp.action_commands[:10]))
    missing_str = ", ".join(exp.missing_steps[:5]) if exp.missing_steps else "(none)"
    version_history_section = _format_patch_history(exp.patch_history)
    
    prompt = AI_REVIEW_PROMPT.format(
        task_desc=exp.task_desc[:200],
        outcome=exp.outcome,
        score=exp.score,
        steps=steps_str,
        missing=missing_str,
        failure_reason=exp.failure_reason or "(none)",
        version_history_section=version_history_section,
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
    
    return {
        "generalized_steps": "\n".join(exp.action_commands[:10]),
        "causal_lesson": exp.failure_reason if exp.outcome != "success" else "Completed all required steps",
        "avoidance_note": exp.failure_reason if exp.outcome != "success" else "",
        "transferability": f"Tasks involving: {', '.join(exp.tool_sequence[:3])}",
        "evolution_insight": "",
        "quality_score": int(exp.score * 10),
        "refined": False,
    }
