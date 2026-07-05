"""Prompt Injection — Dual-channel success/failure experience injection."""
from __future__ import annotations
from .experience import Experience, ExperienceLibrary
from .experience import compute_similarity
from .gate import should_augment, classify_task_type


# Quality gates — structural signal filtering only


def _is_quality_success(exp: Experience) -> bool:
    """Quality gate for success experiences — reduces δ_att by filtering noise.

    Theory: Unrefined successes contain raw action commands that are:
    - Task-specific (high overfitting risk → hurts generalizability)
    - Unstructured (format ambiguity → increases δ_att)
    - Potentially misleading (low-score "successes" → consistency collapse)

    Only AI-refined experiences with substantive causal lessons pass.
    This does NOT affect r_M (coverage) — experiences remain in the library.
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
    """Quality gate for failure experiences — reduces δ_att by filtering noise.

    Theory: Raw failure experiences contain error messages and task-specific
    action sequences that:
    - Don't generalize (overfitting to specific error conditions)
    - May trigger negation priming if injected as "don't do X" without context
    - Dilute attention from genuinely useful failure lessons

    Only AI-refined failures with causal analysis pass.
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


# ══════════════════════════════════════════════════════════════════════════════
#  Channel 1: Success Experience Formatting
#
#  Theory: Success skills provide POSITIVE guidance — "what TO do".
#  Formatted with structured fields to maximize format clarity (↓ δ_att).
#  Includes evolution context to show HOW the skill was refined over time.
# ══════════════════════════════════════════════════════════════════════════════

def format_success_experience(exp: Experience) -> str:
    """Format successful experience with version evolution context.

    Theoretical role: Provides positive action templates that the LLM can
    directly follow. Structured format reduces format-parsing ambiguity.
    Evolution context shows reliability (multiple successful attempts → higher trust).
    """
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

    # Show how this success was achieved (patch history from failures → success)
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
#  Theory: Failure lessons provide NEGATIVE guidance — "what went wrong and why".
#  CRITICAL: We use causal framing ("X failed because Y") rather than
#  imperative negation ("don't do X") to avoid negation priming effects.
#  The dual-channel separation ensures the LLM doesn't conflate positive
#  and negative signals (consistency collapse prevention).
# ══════════════════════════════════════════════════════════════════════════════

def format_failure_experience(exp: Experience) -> str:
    """Format failed experience with causal analysis and recovery strategies.

    Theoretical role: Provides negative examples with ROOT CAUSE analysis.
    Uses causal framing to avoid negation priming (δ_att mechanism #4).
    Patch history shows what was tried and what didn't work — preventing
    the agent from repeating known-bad strategies.
    """
    taxonomy = exp.failure_taxonomy
    parts = [f"[⚠️ Lesson from similar failed task]", f"Task: {exp.task_desc}"]

    if taxonomy.get("ai_refined") and taxonomy.get("causal_lesson"):
        # Causal framing: "X failed because Y" (not "don't do X")
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


# ══════════════════════════════════════════════════════════════════════════════
#  Augmented Prompt Builder — Dual-Channel Assembly
#
#  Theory: The two channels (success + failure) are assembled as SEPARATE
#  clearly-demarcated sections. This structural separation:
#    1. Prevents consistency collapse (δ_att mechanism #3)
#    2. Allows the LLM to allocate attention independently to each channel
#    3. Makes the positive/negative signal boundary unambiguous
#
#  Coverage (r_M) is preserved because:
#    - Quality gates only filter on STRUCTURAL quality, not content relevance
#    - All experiences remain in the library regardless of injection decision
#    - The library never shrinks — only the injection view is filtered
# ══════════════════════════════════════════════════════════════════════════════

def build_augmented_prompt(task_desc: str, library: ExperienceLibrary,
                           top_k_success: int = 3, top_k_failure: int = 2,
                           expected: str = "", metadata: dict | None = None,
                           max_chars: int = 6000,
                           **kwargs) -> str:
    """Build augmented prompt with dual-channel experience injection.

    Theoretical guarantees:
    - r_M unchanged: library content is never modified, only injection is gated
    - δ_sem managed: effectiveness-weighted retrieval (in ExperienceLibrary)
      ensures historically-helpful experiences rank higher
    - δ_att reduced: quality gating + dual-channel separation + structured format
    - Context budget: hard cap at max_chars to prevent prompt explosion
      that drowns task instructions (observed: 23K+ chars after 8 tasks)

    All task types (qa, agentic, embodied) receive the same full-context
    injection. Experiments showed no meaningful difference between dynamic
    and static benchmarks — both benefit equally from full experience context.
    """
    do_augment, reason = should_augment(task_desc, library)
    if not do_augment:
        return ""

    sections = []
    current_len = 0

    # ── Channel 1: Positive guidance (success experiences) ──────────────
    # Theory: These provide action templates the LLM can directly follow.
    # Effectiveness-weighted retrieval ensures δ_sem is minimized.
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
    # Theory: These provide causal failure analysis to prevent repeating mistakes.
    # Separated from Channel 1 to prevent consistency collapse.
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
    # Final safety truncation (should rarely trigger due to per-entry checks)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit("\n", 1)[0]
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Channel 3: Within-Task Patch Memory (SkillForge D Group)
#
#  Theory: Intermediate state patches capture the agent's self-correction
#  patterns during multi-step reasoning. These are the EvoMem-style
#  "patch-based intermediate conclusion state table" — but with
#  failure-aware attention routing that EvoMem doesn't have.
#
#  Failure-Aware Attention Routing:
#    Error patches (is_error_patch=True) → channeled as [Avoid] pitfalls
#    Refinement patches (is_error_patch=False) → channeled as [Refine] templates
#
#  This is the CORE differentiator from EvoMem:
#    EvoMem: records "what changed" — treats all patches equally
#    SkillForge: routes patches by TYPE — error patches get avoidance framing
#               and refinement patches get procedural template framing
# ══════════════════════════════════════════════════════════════════════════════

def format_intermediate_state_patch(patch: dict) -> str:
    """Format a single intermediate state patch with type-aware routing.

    Failure-aware attention routing:
    - Error patches (is_error_patch=True): formatted as [Avoid this pitfall]
      with the correction rationale as an avoidance lesson.
    - Refinement patches (is_error_patch=False): formatted as [Refined strategy]
      as a procedural improvement template.

    Args:
        patch: dict with keys from IntermediateState: turn, conclusion,
               revised_conclusion, revision_rationale, revision_trigger,
               is_error_patch, etc.
    """
    is_error = patch.get("is_error_patch", False)
    trigger = patch.get("revision_trigger", "self_correction")
    turn = patch.get("turn", -1)
    revised_at = patch.get("revised_at_turn", -1)
    conclusion_type = patch.get("conclusion_type", "assumption")

    if is_error:
        # Error patch → avoidance framing (negation priming risk managed by
        # using causal framing: "X fails because Y" rather than "Don't do X")
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
        # Refinement patch → procedural template framing
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

    Selects the most informative patches (error patches first, then refinements)
    and formats them with failure-aware attention routing.

    Args:
        exp: Experience with intermediate_states populated
        max_patches: Maximum number of patches to include
    """
    patches = exp.intermediate_states
    if not patches:
        return ""

    # Prioritize error patches (higher learning value), then refinement patches
    error_patches = [p for p in patches if p.get("is_error_patch", False)]
    refinement_patches = [p for p in patches if not p.get("is_error_patch", False)]

    selected = []
    # Take up to max_patches, error patches first
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
    """Build SkillForge augmented prompt with THREE channels:
    1. Success experiences (positive guidance)
    2. Failure lessons (negative guidance with causal analysis)
    3. Within-task patches (EvoMem-style intermediate state corrections
       with failure-aware attention routing)

    Channel 3 is the SkillForge differentiator — EvoMem records patches but
    injects them as raw diffs. SkillForge routes patches by type:
    - Error patches → "[Avoid this pitfall]" causal framing
    - Refinement patches → "[Refined strategy]" procedural template

    This failure-aware attention routing is what separates SkillForge from
    both EvoMem (no routing) and standard memory injectors (no patches).

    Theoretical guarantees:
    - r_M unchanged: library content is never modified
    - δ_sem managed: effectiveness-weighted retrieval
    - δ_att reduced: quality gating + three-channel separation + type-aware routing
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
    # Theory: These intermediate state patches capture the agent's own
    # self-correction patterns. When solving a new task, seeing how
    # previous agents revised their intermediate conclusions helps the
    # current agent avoid the same pitfalls.
    #
    # Failure-aware attention routing:
    # - Error patches → avoidance framing (prevents negation priming)
    # - Refinement patches → procedural template framing
    if current_len < max_chars:
        # Collect experiences with patches from both successes and failures
        patch_experiences = []
        for exp in library.experiences:
            if exp.intermediate_states:
                patch_experiences.append(exp)

        # Score by similarity to current task
        if patch_experiences:
            scored_patches = []
            for exp in patch_experiences:
                sim = compute_similarity(task_desc, exp.task_desc)
                # Weight by number of meaningful patches (error > refinement)
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
#  EvoArena's EvoMem is a within-task mechanism: it tracks the agent's own
#  self-corrections during a single task execution and makes them available
#  for reference within the same task. This is fundamentally different from
#  cross-task library injection (which was the old B/C design).
#
#  B组 (EvoArena EvoMem):    Plain memory patch log — "here's what you corrected"
#  C组 (EvoArena + SkillForge): Failure-aware routing + critic quality gate
#    - ERROR patches → [Avoid This Pitfall] avoidance framing
#    - REFINEMENT patches → [Refined Strategy] procedural template framing
#    - Quality gate filters trivial/incomplete patches
# ══════════════════════════════════════════════════════════════════════════════


def format_evoarena_patch_log(patches: list[dict]) -> str:
    """Format within-task patches in plain EvoMem style for B组.

    This replicates EvoArena's EvoMem approach: when the agent revises an
    intermediate conclusion during a multi-turn task, the revision is captured
    and made available for reference in subsequent turns. No cross-task
    injection, no failure-aware routing — pure within-task self-correction
    visibility.

    Only includes patches that have been revised (revised_at_turn >= 0),
    since unrevised conclusions are still pending.
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

    Builds on EvoArena EvoMem's within-task patch tracking. Our contributions:
    1. Failure-aware routing: ERROR patches → [Avoid This Pitfall] format
       (prevents negation priming by reframing as actionable avoidance),
       REFINEMENT patches → [Refined Strategy] format (procedural template)
    2. Critic quality gate: only includes patches with substantive revision
       rationale (>= 10 chars), filtering trivial corrections.
    3. Type-labeled separation: error and refinement patches are clearly
       separated into distinct sections for unambiguous attention allocation.
    """
    revised = [p for p in patches if p.get("revised_at_turn", -1) >= 0]
    if not revised:
        return ""

    error_patches = []
    refinement_patches = []

    for p in revised:
        is_error = p.get("is_error_patch", False)
        rationale = p.get("revision_rationale", "")

        # Critic quality gate: skip trivial patches
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