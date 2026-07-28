"""Prompt Injection — Dual-channel success/failure experience injection."""
from __future__ import annotations
from .experience import Experience, ExperienceLibrary
from .experience import compute_similarity
from .gate import should_augment, classify_task_type


# Quality gates — structural signal filtering only


def _is_quality_success(exp: Experience) -> bool:
    """Injection gate for successes; the library itself is never filtered."""
    if exp.score < 0.3:
        return False
    taxonomy = exp.failure_taxonomy
    if taxonomy.get("ai_refined") and taxonomy.get("generalized_steps"):
        causal = taxonomy.get("causal_lesson", "")
        if len(causal) > 5:
            return True
    # High-score unrefined successes still carry usable signal.
    if exp.score >= 0.8:
        return True
    return False


def _is_quality_failure(exp: Experience) -> bool:
    """Injection gate for failures: only AI-refined ones with a causal lesson."""
    taxonomy = exp.failure_taxonomy
    if taxonomy.get("ai_refined") and taxonomy.get("causal_lesson"):
        causal = taxonomy["causal_lesson"]
        if len(causal) > 5:
            return True
    # Unrefined failures are raw error text — noise, not a lesson.
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  Channel 1: Success Experience Formatting
# ══════════════════════════════════════════════════════════════════════════════

def format_success_experience(exp: Experience) -> str:
    """Format a successful experience with version evolution context."""
    taxonomy = exp.failure_taxonomy
    parts = [f"[✓ Successful approach for similar task]", f"Task: {exp.task_desc}"]

    if taxonomy.get("ai_refined") and taxonomy.get("generalized_steps"):
        parts.append(f"Key strategy: {taxonomy.get('causal_lesson', '')}")
        parts.append(f"Generalized steps:\n{taxonomy['generalized_steps']}")
        parts.append(f"Applies to: {taxonomy.get('transferability', '')}")
        if taxonomy.get("evolution_insight"):
            parts.append(f"Evolution insight: {taxonomy['evolution_insight']}")
    else:
        # Safety fallback — should not be reached due to quality gate
        parts.append(f"Score: {exp.score:.0%}")
        return "\n".join(parts)

    parts.append(f"Reliability: {exp.score:.0%}")

    if exp.patch_history:
        evolution = []
        for p in exp.patch_history:
            if p.get("fixed_missing"):
                evolution.append(f"Previously missing {p['fixed_missing']} → now fixed")
            elif p.get("score_delta", 0) > 0:
                evolution.append(f"Improved from v{p.get('from_version','?')} (+{p['score_delta']:.0%})")
        if evolution:
            parts.append("How it was refined: " + "; ".join(evolution))

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  Channel 2: Failure Lesson Formatting
#
#  Phrase lessons causally ("X failed because Y"), never as imperative negation
#  ("don't do X"), to avoid negation priming.
# ══════════════════════════════════════════════════════════════════════════════

def format_failure_experience(exp: Experience) -> str:
    """Format a failed experience with causal analysis and recovery strategies."""
    taxonomy = exp.failure_taxonomy
    parts = [f"[⚠️ Lesson from similar failed task]", f"Task: {exp.task_desc}"]

    if taxonomy.get("ai_refined") and taxonomy.get("causal_lesson"):
        parts.append(f"Root cause: {taxonomy['causal_lesson']}")
        if taxonomy.get("avoidance_note"):
            parts.append(f"Known pitfall: {taxonomy['avoidance_note']}")
        if taxonomy.get("generalized_steps"):
            parts.append(f"What was attempted:\n{taxonomy['generalized_steps']}")
        if exp.missing_steps:
            parts.append("MISSING steps: " + ", ".join(exp.missing_steps))
        if taxonomy.get("transferability"):
            parts.append(f"Applies to: {taxonomy['transferability']}")
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

    if exp.patch_history:
        patch_lines = ["Version history (what changed across attempts):"]
        for p in exp.patch_history:
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

    if taxonomy.get("evolution_trace"):
        parts.append("Evolution: " + " → ".join(taxonomy["evolution_trace"]))

    if taxonomy.get("recovery_strategies"):
        parts.append(f"Recovery strategies: {taxonomy['recovery_strategies']}")
    if taxonomy.get("preconditions"):
        parts.append(f"Preconditions: {taxonomy['preconditions']}")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  Augmented Prompt Builder — Dual-Channel Assembly
# ══════════════════════════════════════════════════════════════════════════════

def build_augmented_prompt(task_desc: str, library: ExperienceLibrary,
                           top_k_success: int = 3, top_k_failure: int = 2,
                           expected: str = "", metadata: dict | None = None,
                           max_chars: int = 6000,
                           **kwargs) -> str:
    """Build an augmented prompt with dual-channel experience injection.

    The library is never modified — only the injection view is gated. max_chars
    is a hard cap: without it the block grew past 23K chars and drowned the task.
    """
    do_augment, reason = should_augment(task_desc, library)
    if not do_augment:
        return ""

    sections = []
    current_len = 0

    # ── Channel 1: Positive guidance (success experiences) ──────────────
    successes = library.retrieve_similar(task_desc, top_k=top_k_success * 2,
                                         outcome_filter="success")
    quality_successes = [exp for exp in successes if _is_quality_success(exp)][:top_k_success]
    if quality_successes:
        header = "## Relevant Experience (from similar successful tasks)\n"
        sections.append(header)
        current_len += len(header)
        for exp in quality_successes:
            entry = format_success_experience(exp) + "\n"
            if current_len + len(entry) > max_chars:
                break
            sections.append(entry)
            current_len += len(entry)

    # ── Channel 2: Negative guidance (failure lessons) ──────────────────
    if current_len < max_chars:
        failures = library.retrieve_similar(task_desc, top_k=top_k_failure * 2,
                                             outcome_filter="failure", exclude_tool_failures=True)
        if not failures:
            failures = library.retrieve_similar(task_desc, top_k=top_k_failure * 2,
                                                 outcome_filter="partial", exclude_tool_failures=True)
        quality_failures = [exp for exp in failures if _is_quality_failure(exp)][:top_k_failure]
        if quality_failures:
            header = "## Lessons from Similar Failed Attempts\n"
            sections.append(header)
            current_len += len(header)
            for exp in quality_failures:
                entry = format_failure_experience(exp) + "\n"
                if current_len + len(entry) > max_chars:
                    break
                sections.append(entry)
                current_len += len(entry)

    result = "\n".join(sections) if sections else ""
    if len(result) > max_chars:
        result = result[:max_chars].rsplit("\n", 1)[0]
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Channel 3: Within-Task Patch Memory (SkillForge D Group)
#
#  Failure-aware routing (the differentiator from EvoMem, which treats all
#  patches alike): error patches → [Avoid] framing, refinements → [Refine].
# ══════════════════════════════════════════════════════════════════════════════

def format_intermediate_state_patch(patch: dict) -> str:
    """Format one intermediate state patch with type-aware routing.

    `patch` carries IntermediateState keys: turn, conclusion,
    revised_conclusion, revision_rationale, revision_trigger, is_error_patch.
    """
    is_error = patch.get("is_error_patch", False)
    trigger = patch.get("revision_trigger", "self_correction")
    turn = patch.get("turn", -1)
    revised_at = patch.get("revised_at_turn", -1)
    conclusion_type = patch.get("conclusion_type", "assumption")

    if is_error:
        parts = [
            f"[Avoid this pitfall — from turn {turn} (revised at turn {revised_at})]",
        ]
        if patch.get("conclusion"):
            parts.append(f"Wrong conclusion: {patch['conclusion']}")
        if patch.get("revised_conclusion"):
            parts.append(f"Correct conclusion: {patch['revised_conclusion']}")
        if patch.get("revision_rationale"):
            parts.append(f"Why it was wrong: {patch['revision_rationale']}")
        parts.append(f"Trigger: {trigger}")
        return "\n".join(parts)
    else:
        parts = [
            f"[Refined strategy — from turn {turn} (improved at turn {revised_at})]",
        ]
        if patch.get("conclusion"):
            parts.append(f"Initial approach ({conclusion_type}): {patch['conclusion']}")
        if patch.get("revised_conclusion"):
            parts.append(f"Improved approach: {patch['revised_conclusion']}")
        if patch.get("revision_rationale"):
            parts.append(f"Refinement insight: {patch['revision_rationale']}")
        parts.append(f"Trigger: {trigger}")
        return "\n".join(parts)


def format_within_task_patches(exp: Experience, max_patches: int = 3) -> str:
    """Format within-task intermediate state patches from one experience.

    Error patches come first (higher learning value), then refinements, up to
    max_patches total.
    """
    patches = exp.intermediate_states
    if not patches:
        return ""

    error_patches = [p for p in patches if p.get("is_error_patch", False)]
    refinement_patches = [p for p in patches if not p.get("is_error_patch", False)]

    selected = []
    for p in error_patches[:max_patches]:
        formatted = format_intermediate_state_patch(p)
        if formatted:
            selected.append(formatted)
    remaining = max_patches - len(selected)
    for p in refinement_patches[:remaining]:
        formatted = format_intermediate_state_patch(p)
        if formatted:
            selected.append(formatted)

    if not selected:
        return ""

    header = f"## Self-Correction Patterns (from similar task: {exp.task_desc[:80]})"
    return header + "\n" + "\n\n".join(selected)


def build_skillforge_prompt(task_desc: str, library: ExperienceLibrary,
                             top_k_success: int = 3, top_k_failure: int = 2,
                             top_k_patches: int = 2, max_patches_per_exp: int = 2,
                             max_chars: int = 8000,
                             **kwargs) -> str:
    """Build the SkillForge prompt from three channels: success experiences,
    failure lessons, and within-task patches with failure-aware routing.
    """
    do_augment, reason = should_augment(task_desc, library)
    if not do_augment:
        return ""

    sections = []
    current_len = 0

    # ── Channel 1: Success experiences ────────────────────────────────────
    successes = library.retrieve_similar(task_desc, top_k=top_k_success * 2,
                                         outcome_filter="success")
    quality_successes = [exp for exp in successes if _is_quality_success(exp)][:top_k_success]
    if quality_successes:
        header = "## Relevant Experience (from similar successful tasks)\n"
        sections.append(header)
        current_len += len(header)
        for exp in quality_successes:
            entry = format_success_experience(exp) + "\n"
            if current_len + len(entry) > max_chars:
                break
            sections.append(entry)
            current_len += len(entry)

    # ── Channel 2: Failure lessons ─────────────────────────────────────────
    if current_len < max_chars:
        failures = library.retrieve_similar(task_desc, top_k=top_k_failure * 2,
                                             outcome_filter="failure", exclude_tool_failures=True)
        if not failures:
            failures = library.retrieve_similar(task_desc, top_k=top_k_failure * 2,
                                                 outcome_filter="partial", exclude_tool_failures=True)
        quality_failures = [exp for exp in failures if _is_quality_failure(exp)][:top_k_failure]
        if quality_failures:
            header = "## Lessons from Similar Failed Attempts\n"
            sections.append(header)
            current_len += len(header)
            for exp in quality_failures:
                entry = format_failure_experience(exp) + "\n"
                if current_len + len(entry) > max_chars:
                    break
                sections.append(entry)
                current_len += len(entry)

    # ── Channel 3: Within-task patch memory (SkillForge differentiator) ───
    if current_len < max_chars:
        patch_experiences = []
        for exp in library.experiences:
            if exp.intermediate_states:
                patch_experiences.append(exp)

        if patch_experiences:
            scored_patches = []
            for exp in patch_experiences:
                sim = compute_similarity(task_desc, exp.task_desc)
                # Weight by patch count, error patches worth more than refinements.
                n_errors = sum(1 for p in exp.intermediate_states if p.get("is_error_patch"))
                n_refinements = len(exp.intermediate_states) - n_errors
                patch_value = n_errors * 1.5 + n_refinements * 1.0
                scored_patches.append((sim * patch_value, exp))

            scored_patches.sort(key=lambda x: -x[0])
            top_patch_exps = [exp for _, exp in scored_patches[:top_k_patches]]

            if top_patch_exps:
                header = "## Self-Correction Patterns (from similar tasks)\n"
                sections.append(header)
                current_len += len(header)
                for exp in top_patch_exps:
                    entry = format_within_task_patches(exp, max_patches=max_patches_per_exp) + "\n"
                    if current_len + len(entry) > max_chars:
                        break
                    sections.append(entry)
                    current_len += len(entry)

    result = "\n".join(sections) if sections else ""
    if len(result) > max_chars:
        result = result[:max_chars].rsplit("\n", 1)[0]
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Within-Task Patch Injection (B/C Group — EvoArena EvoMem Foundation)
#
#  Within-task only (no cross-task injection). B组 = plain patch log;
#  C组 = failure-aware routing + critic quality gate.
# ══════════════════════════════════════════════════════════════════════════════


def format_evoarena_patch_log(patches: list[dict]) -> str:
    """Format within-task patches in plain EvoMem style for B组.

    Only revised patches (revised_at_turn >= 0) are included; unrevised
    conclusions are still pending.
    """
    revised = [p for p in patches if p.get("revised_at_turn", -1) >= 0]
    if not revised:
        return ""

    entries = []
    for p in revised:
        turn = p.get("turn", -1)
        revised_at = p.get("revised_at_turn", -1)
        entries.append(
            f"[Memory Update] Turn {turn} conclusion revised at turn {revised_at}:\n"
            f"  Original: {p.get('conclusion', '')}\n"
            f"  Revised:  {p.get('revised_conclusion', '')}\n"
            f"  Reason:   {p.get('revision_rationale', '')}"
        )

    return "\n\n--- Memory Patch Log ---\n" + "\n\n".join(entries) + "\n--- End Patch Log ---\n"


def format_skillforge_patch_log(patches: list[dict]) -> str:
    """Format within-task patches with failure-aware attention routing for C组.

    Error patches render as [Avoid This Pitfall], refinements as [Refined
    Strategy], in separate sections; rationales under 10 chars are dropped.
    """
    revised = [p for p in patches if p.get("revised_at_turn", -1) >= 0]
    if not revised:
        return ""

    error_patches = []
    refinement_patches = []

    for p in revised:
        is_error = p.get("is_error_patch", False)
        rationale = p.get("revision_rationale", "")

        if len(rationale) < 10:
            continue

        if is_error:
            error_patches.append(
                f"[Avoid This Pitfall] Turn {p.get('turn', -1)}:\n"
                f"  Wrong approach: {p.get('conclusion', '')}\n"
                f"  Corrected to:   {p.get('revised_conclusion', '')}\n"
                f"  Lesson: {rationale}"
            )
        else:
            refinement_patches.append(
                f"[Refined Strategy] Turn {p.get('turn', -1)} → {p.get('revised_at_turn', -1)}:\n"
                f"  Initial:  {p.get('conclusion', '')}\n"
                f"  Improved: {p.get('revised_conclusion', '')}\n"
                f"  Rationale: {rationale}"
            )

    parts = []
    if error_patches:
        parts.append("## Self-Corrections: Pitfalls to Avoid\n" + "\n\n".join(error_patches))
    if refinement_patches:
        parts.append("## Self-Corrections: Strategy Refinements\n" + "\n\n".join(refinement_patches))

    if not parts:
        return ""

    return "\n\n--- Correction Log ---\n" + "\n\n".join(parts) + "\n--- End Log ---\n"