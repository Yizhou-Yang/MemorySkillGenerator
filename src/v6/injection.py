"""Cost-Aware Prompt Injection — routes by task type, enforces token budget."""
from __future__ import annotations
import tiktoken
from .experience import Experience, ExperienceLibrary
from .gate import should_augment, classify_task_type

# Lazy-loaded tokenizer (cl100k_base covers GPT-3.5/4/most models)
_enc = None
def estimate_token_count(text: str) -> int:
    """Accurate token count via tiktoken."""
    global _enc
    if _enc is None:
        try:
            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return len(text) // 4  # fallback
    return len(_enc.encode(text, disallowed_special=()))


def format_success_experience(exp: Experience, budget_tokens: int = 800) -> str:
    """Format successful experience. AI-refined preferred. No content truncation."""
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

    result = "\n".join(parts)
    # Budget: drop low-priority fields (never truncate mid-content)
    if estimate_token_count(result) > budget_tokens and len(parts) > 4:
        result = "\n".join(parts[:4])
    return result


def format_failure_experience(exp: Experience, budget_tokens: int = 600) -> str:
    """Format failed experience. Full information preserved."""
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

    result = "\n".join(parts)
    if estimate_token_count(result) > budget_tokens and len(parts) > 5:
        result = "\n".join(parts[:5])
    return result


def _build_qa_hint(task_desc: str, library: ExperienceLibrary, token_budget: int = 400) -> str:
    """Lightweight hints for QA tasks."""
    candidates = library.retrieve_similar(task_desc, top_k=5)
    if not candidates:
        return ""
    hints = []
    for exp in candidates:
        ft = exp.failure_taxonomy
        if not ft.get("ai_refined"):
            continue
        for text in [ft.get("transferability", ""), ft.get("causal_lesson", "")]:
            if not text or len(text) < 20:
                continue
            hints.append(text)
            break
    if not hints:
        return ""
    seen = set()
    unique = [h for h in hints if not (h[:40].lower() in seen or seen.add(h[:40].lower()))]
    if not unique:
        return ""
    return "## Reasoning Hints\n" + "\n".join(f"- {h}" for h in unique[:3])


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
