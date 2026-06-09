"""Prompt Injection — full experience context injection for all task types."""
from __future__ import annotations
from .experience import Experience, ExperienceLibrary
from .gate import should_augment, classify_task_type


def _is_quality_success(exp: Experience) -> bool:
    """Check if a success experience is high-quality enough to inject.

    Prevents overfitting by filtering out:
    - Very low-score "successes" (clearly wrong partial matches)
    - Empty experiences with no actionable content
    """
    if exp.score < 0.3:
        return False
    taxonomy = exp.failure_taxonomy
    # AI-refined with substantive generalized_steps = quality skill
    if taxonomy.get("ai_refined") and taxonomy.get("generalized_steps"):
        # Also check causal_lesson is not completely empty
        causal = taxonomy.get("causal_lesson", "")
        if len(causal) > 5:  # Must have a real lesson, not empty
            return True
    # Allow high-score unrefined successes — they still carry useful signal
    # even without AI refinement (1M context can absorb the noise)
    if exp.score >= 0.8:
        return True
    return False


def _is_quality_failure(exp: Experience) -> bool:
    """Check if a failure experience provides actionable lessons.

    Prevents noise injection by filtering out:
    - Raw unrefined failures (just error messages, no causal analysis)
    - Tool-chain failures (infra issues, not skill issues)
    - Failures with no meaningful lesson content
    """
    taxonomy = exp.failure_taxonomy
    # AI-refined failures always have quality content
    if taxonomy.get("ai_refined") and taxonomy.get("causal_lesson"):
        causal = taxonomy["causal_lesson"]
        # Filter out only completely empty/trivial causal lessons
        if len(causal) > 5:
            return True
    # Unrefined failures are noise — they contain raw error messages
    # and task-specific action sequences that don't generalize
    return False


def format_success_experience(exp: Experience) -> str:
    """Format successful experience with version evolution context.

    NOTE: This function is only called for experiences that passed _is_quality_success,
    which requires ai_refined=True with substantive generalized_steps and causal_lesson.
    """
    taxonomy = exp.failure_taxonomy
    parts = [f"[Successful approach for similar task]", f"Task: {exp.task_desc}"]

    if taxonomy.get("ai_refined") and taxonomy.get("generalized_steps"):
        parts.append(f"Causal lesson: {taxonomy.get('causal_lesson', '')}")
        parts.append(f"Generalized steps:\n{taxonomy['generalized_steps']}")
        parts.append(f"Transferable to: {taxonomy.get('transferability', '')}")
        if taxonomy.get("evolution_insight"):
            parts.append(f"Evolution insight: {taxonomy['evolution_insight']}")
    else:
        # Safety fallback — should not be reached due to quality gate
        # but if it is, output minimal non-noise content
        parts.append(f"Score: {exp.score:.0%}")
        return "\n".join(parts)

    parts.append(f"Score: {exp.score:.0%}")

    # Show how this success was achieved (patch history from failures → success)
    if exp.patch_history:
        evolution = []
        for p in exp.patch_history:
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
        for p in exp.patch_history:  # Full history — 1M context can handle it
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
        parts.append("Evolution: " + " → ".join(taxonomy["evolution_trace"]))

    # Critic-refined enrichments (recovery strategies, preconditions)
    if taxonomy.get("recovery_strategies"):
        parts.append(f"Recovery strategies: {taxonomy['recovery_strategies']}")
    if taxonomy.get("preconditions"):
        parts.append(f"Preconditions: {taxonomy['preconditions']}")

    return "\n".join(parts)


def build_augmented_prompt(task_desc: str, library: ExperienceLibrary,
                           top_k_success: int = 5, top_k_failure: int = 5,
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

    # Retrieve and quality-gate success experiences
    successes = library.retrieve_similar(task_desc, top_k=top_k_success * 2,
                                         outcome_filter="success")
    quality_successes = [exp for exp in successes if _is_quality_success(exp)][:top_k_success]
    if quality_successes:
        sections.append("## Relevant Experience (from similar successful tasks)\n")
        for exp in quality_successes:
            entry = format_success_experience(exp)
            sections.append(entry + "\n")

    # Retrieve and quality-gate failure experiences
    failures = library.retrieve_similar(task_desc, top_k=top_k_failure * 2,
                                         outcome_filter="failure", exclude_tool_failures=True)
    if not failures:
        failures = library.retrieve_similar(task_desc, top_k=top_k_failure * 2,
                                             outcome_filter="partial", exclude_tool_failures=True)
    quality_failures = [exp for exp in failures if _is_quality_failure(exp)][:top_k_failure]
    if quality_failures:
        sections.append("## Lessons from Similar Failed Attempts\n")
        for exp in quality_failures:
            entry = format_failure_experience(exp)
            sections.append(entry + "\n")

    return "\n".join(sections) if sections else ""
