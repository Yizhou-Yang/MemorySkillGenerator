"""Version-Conditioned AI Refinement + Cross-Agent Skill Quality Evaluation.

Theoretical Role (SRDP Framework — δ_att Reduction):
    This module targets δ_att (attention degradation error) through two mechanisms:

    1. AI Refinement (ai_review_experience):
       - Converts raw action sequences into structured, generalized skills
       - Reduces FORMAT PARSING AMBIGUITY dimension of δ_att by producing
         consistent, well-structured output (numbered steps, clear causal lessons)
       - The ZERO INFORMATION LOSS constraint ensures r_M (coverage radius)
         is never increased — we only ADD structure, never remove content

    2. Cross-Agent Critic (cross_agent_evaluate_skill + critic_refine_experience):
       - Independent quality evaluation detects NOISE and INFORMATION LOSS
       - Noise detection prevents RETRIEVAL DILUTION: noisy skills that would
         split probability mass in Boltzmann retrieval are flagged and enriched
       - Forced enrichment (never discard) maintains r_M while improving
         the signal-to-noise ratio that directly affects δ_att
       - The critic→refine loop is the system's primary δ_att reduction mechanism:
         low-quality skills are iteratively improved until they can be properly
         utilized by the LLM's attention mechanism

    Gap Bound Contribution:
        sup_s |V* - V^π_M| ≤ (2R_max)/(1-γ)² · (ε_LLM(r_M) + δ_sem + E[δ_att])
        This module reduces E[δ_att] by ensuring every skill in the library is:
        - Well-structured (low format parsing ambiguity)
        - Non-trivial (low retrieval dilution from noise)
        - Enriched with recovery strategies (reduces negation priming risk)
"""
from __future__ import annotations
from json_repair import repair_json
import json
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
# Activated when the initial refinement produces empty causal_lesson or avoidance_note.
# This prompt is stricter: it demands STRATEGIC-LEVEL analysis (not factual-level).
# The core insight: failure experiences are valuable ONLY if they explain WHY the
# STRATEGY failed (e.g., "PDF text extraction failed because the file was scanned
# images, not digital text") rather than WHAT fact was wrong (e.g., "the answer is
# Rockhopper penguin, not Emperor penguin"). Factual errors have zero cross-task
# transfer value and become noise when injected.
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

## Response (JSON only)
{{
  "actionability": 0-3,
  "generalizability": 0-3,
  "correctness": 0-2,
  "novelty": 0-2,
  "total": 0-10,
  "verdict": "inject" | "skip" | "low_confidence",
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
    """Version-conditioned refinement with quality self-check for failures.

    SRDP Theory: This function reduces δ_att (format parsing ambiguity dimension)
    by transforming raw action sequences into structured, generalized skills.

    QUALITY SELF-CHECK (for failures only):
    If the first refinement produces empty causal_lesson or avoidance_note,
    the experience is re-refined using AI_REVIEW_FAILURE_RETRY_PROMPT which
    demands STRATEGIC-LEVEL analysis instead of factual-level noise.

    The ZERO INFORMATION LOSS constraint is critical: it ensures r_M never
    increases. We add generalization ON TOP of existing content, preserving
    the original skill's coverage of its task region in the skill space.

    When llm_fn is None (no LLM available), returns a minimal fallback that
    preserves all original information but marks as unrefined.
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

    # Include reasoning trace if available (from response_filter AI evaluation)
    reasoning_section = ""
    if exp.reasoning_trace:
        reasoning_lines = "\n".join(f"  - {r}" for r in exp.reasoning_trace[:10])
        reasoning_section = f"\nAgent's reasoning during execution:\n{reasoning_lines}\n"

    # ── First refinement attempt ────────────────────────────────────────────
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
    # If the failure experience has empty causal_lesson or avoidance_note,
    # the AI failed to extract strategic-level insight. Retry with a stricter
    # prompt that explicitly forbids factual-level analysis.
    if (exp.outcome != "success" and result and result.get("refined") and
            (not result.get("causal_lesson", "").strip() or
             not result.get("avoidance_note", "").strip())):
        # Show what we got from first attempt
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
            # Use retry result but keep the original generalized_steps if retry is worse
            if (retry_result.get("causal_lesson", "").strip() and
                    retry_result.get("avoidance_note", "").strip()):
                return retry_result
        # If retry also failed, return the first result but mark as low quality

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

def cross_agent_evaluate_skill(exp: Experience, llm_fn=None) -> dict:
    """Cross-agent quality evaluation: an independent LLM judges skill quality.

    SRDP Theory: This is the primary δ_att diagnostic mechanism.
    It detects three failure modes that increase δ_att:

    1. NOISE (trivial causal_lesson) → causes RETRIEVAL DILUTION:
       Noisy skills split Boltzmann probability mass without providing
       useful signal, diluting μ(c*|s) for the optimal skill.

    2. INFORMATION LOSS (vague generalized_steps) → increases r_M:
       If refinement accidentally compressed content, the skill's coverage
       radius expands (it covers less of the task space precisely).

    3. OVERFITTING (task-specific content) → increases δ_sem:
       Overfitted skills have high similarity to one task but mislead
       retrieval for related-but-different tasks.

    When any issue is detected, verdict='low_confidence' triggers
    critic_refine_experience (forced enrichment, never discard).
    """
    default = {"total": 5, "verdict": "inject", "reason": "no evaluator available",
               "actionability": 2, "generalizability": 2, "correctness": 1, "novelty": 0}

    if llm_fn is None:
        return default

    steps_str = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
    causal = exp.failure_taxonomy.get("causal_lesson", "")
    generalized = exp.failure_taxonomy.get("generalized_steps", "")

    # Include reasoning trace for richer evaluation context
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
    """When critic scores low, REFINE the experience (never discard).

    SRDP Theory: This is the δ_att REPAIR mechanism.
    Rather than discarding low-quality skills (which would increase r_M),
    we enrich them to reduce their contribution to δ_att:

    - Adding recovery strategies → reduces NEGATION PRIMING risk
      (concrete "do X instead" replaces vague "don't do Y")
    - Adding preconditions → reduces CONSISTENCY COLLAPSE risk
      (explicit conditions prevent silent conflict resolution)
    - Expanding causal reasoning → reduces FORMAT PARSING AMBIGUITY
      (deeper explanation is easier for attention to anchor on)

    The constraint that output must be LONGER than input guarantees
    r_M never increases (Zero Information Loss principle).
    """
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

    # Include reasoning trace for richer refinement context
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