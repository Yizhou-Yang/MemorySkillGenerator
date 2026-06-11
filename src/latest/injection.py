"""Prompt Injection — Dual-Channel Experience Injection with Signal Separation.

Theoretical Foundation (SRDP Framework — Corollary 15):
    gap ≤ (2R_max)/(1-γ)² · [ε_LLM(r_M) + δ_sem + E[δ_att]]

This module targets δ_att (attention degradation) through three mechanisms:

1. QUALITY GATING: Filters noise skills that would dilute attention budget.
   → Reduces δ_att by preventing format-parsing ambiguity and retrieval dilution.

2. DUAL-CHANNEL SEPARATION: Success experiences and failure lessons are injected
   as distinct, clearly-demarcated sections with different formatting.
   → Reduces δ_att by preventing consistency collapse (LLM silently picking one
   when given contradictory positive/negative signals in the same block).

3. STRUCTURED FORMATTING: Each experience uses a consistent, parseable format
   (labeled fields, markdown structure) rather than prose.
   → Reduces δ_att by improving format clarity for LLM attention allocation.

Design choice — NO position reordering:
    Lost-in-the-Middle effects are documented at 4K-32K context windows.
    With 1M context (DeepSeek V4), empirical evidence suggests position effects
    are negligible. We therefore prioritize signal quality over position tricks.

Coverage guarantee (r_M preservation):
    This module NEVER filters based on content relevance alone — only on
    structural quality (is it refined? does it have a causal lesson?).
    All information remains in the library; only injection is gated.
    → ε_LLM(r_M) is never increased by this module.
"""
from __future__ import annotations
from .experience import Experience, ExperienceLibrary
from .gate import should_augment, classify_task_type


# ══════════════════════════════════════════════════════════════════════════════
#  Quality Gates — δ_att reduction via noise filtering
#
#  Theory: Injecting low-quality skills increases δ_att through two mechanisms:
#    (a) Retrieval dilution: noisy skills split attention mass with useful ones
#    (b) Format ambiguity: unstructured raw commands are harder for LLM to parse
#  These gates ensure only well-structured, AI-refined skills enter the context.
# ══════════════════════════════════════════════════════════════════════════════

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
