"""Prompt Injection — full experience context injection for all task types."""
from __future__ import annotations
from .experience import Experience, ExperienceLibrary
from .gate import should_augment, classify_task_type


def format_success_experience(exp: Experience) -> str:
    """Format successful experience with version evolution context."""
    taxonomy = exp.failure_taxonomy
    parts = [f"[Successful approach for similar task]", f"Task: {exp.task_desc}"]

    if taxonomy.get("ai_refined") and taxonomy.get("generalized_steps"):
        parts.append(f"Causal lesson: {taxonomy.get('causal_lesson', '')}")
        parts.append(f"Generalized steps:\n{taxonomy['generalized_steps']}")
        parts.append(f"Transferable to: {taxonomy.get('transferability', '')}")
        if taxonomy.get("evolution_insight"):
            parts.append(f"Evolution insight: {taxonomy['evolution_insight']}")
    else:
        steps = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
        parts.append(f"Steps:\n{steps}")
    parts.append(f"Score: {exp.score:.0%}")

    # Show how this success was achieved (patch history from failures → success)
    if exp.patch_history:
        evolution = []
        for p in exp.patch_history[-2:]:
            if p.get("fixed_missing"):
                evolution.append(f"Previously missing {p['fixed_missing']} → now fixed")
            elif p.get("score_delta", 0) > 0:
                evolution.append(f"Improved from v{p.get('from_version','?')} (+{p['score_delta']:.0%})")
        if evolution:
            parts.append("How it was fixed: " + "; ".join(evolution))

    return "\n".join(parts)


def format_failure_experience(exp: Experience) -> str:
    """Format failed experience with patch history (EvoMem-style version tracking)."""
    taxonomy = exp.failure_taxonomy
    parts = [f"[⚠️ Lesson from similar failed task]", f"Task: {exp.task_desc}"]

    if taxonomy.get("ai_refined") and taxonomy.get("causal_lesson"):
        parts.append(f"Why it failed: {taxonomy['causal_lesson']}")
        if taxonomy.get("avoidance_note"):
            parts.append(f"Avoid: {taxonomy['avoidance_note']}")
        if taxonomy.get("generalized_steps"):
            parts.append(f"Attempted:\n{taxonomy['generalized_steps']}")
        if exp.missing_steps:
            parts.append("MISSING steps: " + ", ".join(exp.missing_steps))
        if taxonomy.get("transferability"):
            parts.append(f"Transferable to: {taxonomy['transferability']}")
        if taxonomy.get("evolution_insight"):
            parts.append(f"Evolution insight: {taxonomy['evolution_insight']}")
    else:
        if exp.failure_reason:
            parts.append(f"What went wrong: {exp.failure_reason}")
        if exp.missing_steps:
            parts.append("MISSING: " + ", ".join(exp.missing_steps))
        if exp.action_commands:
            steps = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
            parts.append(f"Attempted:\n{steps}")

    # EvoMem-style patch history: show how this skill evolved across attempts
    if exp.patch_history:
        patch_lines = ["Version history (what changed across attempts):"]
        for p in exp.patch_history[-3:]:  # Last 3 patches max
            delta = p.get("score_delta", 0)
            patch_lines.append(
                f"  v{p.get('from_version','?')}→v{p.get('to_version','?')}: "
                f"{p.get('outcome_change', '')} (Δ={delta:+.0%})"
            )
            if p.get("fixed_missing"):
                patch_lines.append(f"    Fixed: {p['fixed_missing']}")
            if p.get("new_missing"):
                patch_lines.append(f"    Still missing: {p['new_missing']}")
            if p.get("new_steps"):
                patch_lines.append(f"    Added: {p['new_steps']}")
        parts.append("\n".join(patch_lines))

    # Evolution trace summary
    if taxonomy.get("evolution_trace"):
        parts.append("Evolution: " + " → ".join(taxonomy["evolution_trace"][-3:]))

    # Critic-refined enrichments (recovery strategies, preconditions)
    if taxonomy.get("recovery_strategies"):
        parts.append(f"Recovery strategies: {taxonomy['recovery_strategies']}")
    if taxonomy.get("preconditions"):
        parts.append(f"Preconditions: {taxonomy['preconditions']}")

    return "\n".join(parts)


def build_augmented_prompt(task_desc: str, library: ExperienceLibrary,
                           top_k_success: int = 2, top_k_failure: int = 2,
                           expected: str = "", metadata: dict | None = None,
                           **kwargs) -> str:
    """Build augmented prompt with full experience injection.

    All task types (qa, agentic, embodied) receive the same full-context
    injection. The previous qa→lightweight hints routing was removed because
    experiments showed no meaningful difference between dynamic and static
    benchmarks — both benefit equally from full experience context.
    """
    do_augment, reason = should_augment(task_desc, library)
    if not do_augment:
        return ""

    sections = []

    successes = library.retrieve_similar(task_desc, top_k=top_k_success, outcome_filter="success")
    if successes:
        sections.append("## Relevant Experience (from similar successful tasks)\n")
        for exp in successes:
            entry = format_success_experience(exp)
            sections.append(entry + "\n")

    failures = library.retrieve_similar(task_desc, top_k=top_k_failure,
                                         outcome_filter="failure", exclude_tool_failures=True)
    if not failures:
        failures = library.retrieve_similar(task_desc, top_k=top_k_failure,
                                             outcome_filter="partial", exclude_tool_failures=True)
    if failures:
        sections.append("## Lessons from Similar Failed Attempts\n")
        for exp in failures:
            entry = format_failure_experience(exp)
            sections.append(entry + "\n")

    return "\n".join(sections) if sections else ""
