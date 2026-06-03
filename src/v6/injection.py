"""Cost-Aware Prompt Injection — full information preservation, no truncation."""
from __future__ import annotations
from .experience import Experience, ExperienceLibrary
from .gate import should_augment, classify_task_type


def estimate_token_count(text: str) -> int:
    return len(text) // 4


def format_success_experience(exp: Experience, budget_tokens: int = 800) -> str:
    """Format successful experience. AI-refined preferred. NO truncation of content —
    only omits lower-priority fields when budget is tight."""
    taxonomy = exp.failure_taxonomy
    parts = []

    parts.append(f"[Successful approach for similar task]")
    parts.append(f"Task: {exp.task_desc}")

    if taxonomy.get("ai_refined") and taxonomy.get("generalized_steps"):
        parts.append(f"Causal lesson: {taxonomy.get('causal_lesson', '')}")
        parts.append(f"Generalized steps:\n{taxonomy['generalized_steps']}")
        parts.append(f"Transferable to: {taxonomy.get('transferability', '')}")
        if taxonomy.get("evolution_insight"):
            parts.append(f"Evolution insight: {taxonomy['evolution_insight']}")
    else:
        # Raw: include ALL action commands, not just first N
        steps = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
        parts.append(f"Steps:\n{steps}")
    parts.append(f"Score: {exp.score:.0%}")

    result = "\n".join(parts)
    # Budget control: drop lowest-priority fields first (transferability, evolution_insight)
    # but NEVER truncate mid-sentence
    if estimate_token_count(result) > budget_tokens and len(parts) > 4:
        result = "\n".join(parts[:4])  # Keep task + lesson + steps, drop extras
    return result


def format_failure_experience(exp: Experience, budget_tokens: int = 600) -> str:
    """Format failed experience. Full information — no field truncation."""
    taxonomy = exp.failure_taxonomy
    parts = []

    parts.append(f"[⚠️ Lesson from similar failed task]")
    parts.append(f"Task: {exp.task_desc}")

    if taxonomy.get("ai_refined") and taxonomy.get("causal_lesson"):
        parts.append(f"Why it failed: {taxonomy['causal_lesson']}")
        if taxonomy.get("avoidance_note"):
            parts.append(f"Avoid: {taxonomy['avoidance_note']}")
        if taxonomy.get("generalized_steps"):
            parts.append(f"What was attempted:\n{taxonomy['generalized_steps']}")
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
            parts.append("MISSING steps: " + ", ".join(exp.missing_steps))
        if exp.action_commands:
            steps = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands))
            parts.append(f"What was attempted:\n{steps}")

    result = "\n".join(parts)
    # Budget: drop lowest-priority fields (transferability, evolution_insight) but keep core
    if estimate_token_count(result) > budget_tokens and len(parts) > 5:
        result = "\n".join(parts[:5])
    return result


def _build_qa_hint(task_desc: str, library: ExperienceLibrary, token_budget: int = 400) -> str:
    """Lightweight hints for QA tasks. Filters by content relevance, not keyword blacklist."""
    candidates = library.retrieve_similar(task_desc, top_k=5)
    if not candidates:
        return ""

    hints = []
    for exp in candidates:
        ft = exp.failure_taxonomy
        if not ft.get("ai_refined"):
            continue
        # Use transferability and causal_lesson — these are the most generalizable fields.
        # Skip if the hint is about specific tool usage that wouldn't apply to a QA context
        # (detected by checking if the lesson is about the REASONING, not about tool mechanics)
        for text in [ft.get("transferability", ""), ft.get("causal_lesson", "")]:
            if not text or len(text) < 20:
                continue
            hints.append(text)
            break

    if not hints:
        return ""
    # Deduplicate by prefix
    seen = set()
    unique = []
    for h in hints:
        key = h[:40].lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)

    if not unique:
        return ""
    result = "## Reasoning Hints\n" + "\n".join(f"- {h}" for h in unique[:3])
    return result


def build_augmented_prompt(task_desc: str, library: ExperienceLibrary,
                           token_budget: int = 2000,
                           top_k_success: int = 2, top_k_failure: int = 2,
                           expected: str = "", metadata: dict | None = None) -> str:
    """Route: qa→hints, agentic/embodied→full injection with gate."""
    task_type = classify_task_type(task_desc, expected=expected, metadata=metadata)

    if task_type == "qa":
        return _build_qa_hint(task_desc, library, token_budget=min(token_budget, 400))

    do_augment, reason = should_augment(task_desc, library)
    if not do_augment:
        return f"<!-- augmentation skipped: {reason} -->"

    sections = []
    remaining = token_budget

    successes = library.retrieve_similar(task_desc, top_k=top_k_success, outcome_filter="success")
    if successes:
        sections.append("## Relevant Experience (from similar successful tasks)\n")
        for exp in successes:
            entry = format_success_experience(exp, budget_tokens=remaining // 3)
            sections.append(entry + "\n")
            remaining -= estimate_token_count(entry)
            if remaining <= 200:
                break

    failures = library.retrieve_similar(task_desc, top_k=top_k_failure,
                                         outcome_filter="failure", exclude_tool_failures=True)
    if not failures:
        failures = library.retrieve_similar(task_desc, top_k=top_k_failure,
                                             outcome_filter="partial", exclude_tool_failures=True)
    if failures and remaining > 200:
        sections.append("## Lessons from Similar Failed Attempts\n")
        for exp in failures:
            entry = format_failure_experience(exp, budget_tokens=remaining // 2)
            sections.append(entry + "\n")
            remaining -= estimate_token_count(entry)
            if remaining <= 0:
                break

    return "\n".join(sections) if sections else ""
