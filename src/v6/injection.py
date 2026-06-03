"""
SkillForge V6 — Cost-Aware Prompt Injection.

Routes injection by task type:
- Agentic: full experience (success steps + failure warnings)
- QA: lightweight reasoning hints only (~80 tokens)
- Embodied: full experience (action patterns)

Token budget enforcement prevents prompt bloat (TRACE finding #3: 48% overhead).
"""
from __future__ import annotations
import re
from .experience import Experience, ExperienceLibrary
from .gate import should_augment, classify_task_type


def estimate_token_count(text: str) -> int:
    """Rough estimation: 1 token ≈ 4 chars for English."""
    return len(text) // 4


def format_success_experience(exp: Experience, budget_tokens: int = 500) -> str:
    """Format successful experience. Prefers AI-refined version (generalized + causal)."""
    taxonomy = exp.failure_taxonomy
    
    if taxonomy.get("ai_refined") and taxonomy.get("generalized_steps"):
        result = f"""[Successful approach for similar task]
Task: {exp.task_desc[:150]}
Causal lesson: {taxonomy.get('causal_lesson', '')}
Generalized steps:
{taxonomy['generalized_steps']}
Transferable to: {taxonomy.get('transferability', '')}
Original score: {exp.score:.0%}"""
    else:
        max_steps = 8 if exp.task_complexity == "complex" else 5
        steps = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands[:max_steps]))
        result = f"""[Successful approach for similar task]
Task: {exp.task_desc[:150]}
Steps taken:
{steps}
Result: Task completed successfully (score: {exp.score:.0%})."""
    
    if estimate_token_count(result) > budget_tokens:
        key_steps = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands[:3]))
        result = f"""[Successful approach for similar task]
Task: {exp.task_desc[:100]}
Key steps: {key_steps}"""
    
    return result


def format_failure_experience(exp: Experience, budget_tokens: int = 400) -> str:
    """Format failed experience. Prefers AI-refined version (causal lesson + avoidance)."""
    taxonomy = exp.failure_taxonomy
    
    if taxonomy.get("ai_refined") and taxonomy.get("causal_lesson"):
        lines = ["[⚠️ Lesson from similar failed task]"]
        lines.append(f"Task: {exp.task_desc[:120]}")
        lines.append(f"Why it failed: {taxonomy['causal_lesson']}")
        if taxonomy.get("avoidance_note"):
            lines.append(f"Avoid: {taxonomy['avoidance_note']}")
        if taxonomy.get("generalized_steps"):
            lines.append(f"What was attempted:\n{taxonomy['generalized_steps']}")
        if exp.missing_steps:
            lines.append("Steps that were MISSING:")
            for step in exp.missing_steps[:4]:
                lines.append(f"  - {step}")
        lines.append(f"Transferable to: {taxonomy.get('transferability', '')}")
    else:
        lines = ["[⚠️ Warning from similar failed task]"]
        lines.append(f"Task: {exp.task_desc[:120]}")
        category = taxonomy.get("category", "")
        if category == "tool_failure":
            lines.append(f"⚠️ Tool/environment issue: {taxonomy.get('root_cause', exp.failure_reason)}")
        elif category == "over_action":
            lines.append(f"⚠️ Over-action: Agent did too much. {taxonomy.get('root_cause', '')}")
        elif category == "task_mismatch":
            lines.append(f"⚠️ Task misunderstood: {taxonomy.get('root_cause', exp.failure_reason)}")
        else:
            if exp.failure_reason:
                lines.append(f"What went wrong: {exp.failure_reason}")
        if exp.missing_steps:
            lines.append("Steps that were MISSING (make sure to do these):")
            for step in exp.missing_steps[:4]:
                lines.append(f"  - {step}")
    
    result = "\n".join(lines)
    if estimate_token_count(result) > budget_tokens:
        result = "\n".join(lines[:5])
    return result


def _build_qa_hint(task_desc: str, library: ExperienceLibrary, token_budget: int = 400) -> str:
    """Lightweight hints for QA tasks (~80 tokens). Filters out tool-specific noise."""
    candidates = library.retrieve_similar(task_desc, top_k=5)
    if not candidates:
        return ""
    
    tool_noise = {"websearch", "webfetch", "tool", "bash", "cli", "api", "curl",
                  "search for", "fetch", "download", "execute", "invoke"}
    
    hints = []
    for exp in candidates:
        ft = exp.failure_taxonomy
        if not ft.get("ai_refined"):
            continue
        for text in [ft.get("transferability", ""), ft.get("causal_lesson", ""), ft.get("avoidance_note", "")]:
            if not text:
                continue
            if any(kw in text.lower() for kw in tool_noise):
                continue
            if len(text) > 30:
                hints.append(text[:150])
                break
    
    if not hints:
        return ""
    
    seen = set()
    unique = []
    for h in hints:
        key = h[:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    
    if not unique:
        return ""
    
    hint_text = "\n".join(f"- {h}" for h in unique[:2])
    result = f"## Reasoning Hints\n{hint_text}"
    if estimate_token_count(result) > token_budget:
        result = result[:token_budget * 4]
    return result


def build_augmented_prompt(task_desc: str, library: ExperienceLibrary,
                           token_budget: int = 2000,
                           top_k_success: int = 2, top_k_failure: int = 2,
                           expected: str = "", metadata: dict | None = None) -> str:
    """Task-type-aware experience injection router.
    
    Routing logic:
    - qa → _build_qa_hint (lightweight, max ~100 tokens)
    - agentic/embodied → full injection (success + failure, within budget)
    
    Gate logic (for agentic/embodied):
    - simple task → skip
    - low relevance → skip
    - historically ineffective → skip
    """
    task_type = classify_task_type(task_desc, expected=expected, metadata=metadata)
    
    # QA: lightweight hints only
    if task_type == "qa":
        return _build_qa_hint(task_desc, library, token_budget=min(token_budget, 400))
    
    # Agentic/Embodied: full injection with gate
    do_augment, reason = should_augment(task_desc, library)
    if not do_augment:
        return f"<!-- augmentation skipped: {reason} -->"
    
    sections = []
    remaining_budget = token_budget
    
    # Success experiences
    successes = library.retrieve_similar(task_desc, top_k=top_k_success, outcome_filter="success")
    if successes:
        sections.append("## Relevant Experience (from similar successful tasks)\n")
        for exp in successes:
            entry = format_success_experience(exp, budget_tokens=remaining_budget // 3)
            entry_tokens = estimate_token_count(entry)
            if remaining_budget - entry_tokens > 200:
                sections.append(entry)
                sections.append("")
                remaining_budget -= entry_tokens
    
    # Failure experiences (excluding tool-chain noise)
    failures = library.retrieve_similar(
        task_desc, top_k=top_k_failure,
        outcome_filter="failure", exclude_tool_failures=True
    )
    if not failures:
        failures = library.retrieve_similar(
            task_desc, top_k=top_k_failure,
            outcome_filter="partial", exclude_tool_failures=True
        )
    
    if failures and remaining_budget > 200:
        sections.append("## Lessons from Similar Failed Attempts\n")
        for exp in failures:
            entry = format_failure_experience(exp, budget_tokens=remaining_budget // 2)
            entry_tokens = estimate_token_count(entry)
            if remaining_budget - entry_tokens > 0:
                sections.append(entry)
                sections.append("")
                remaining_budget -= entry_tokens
    
    return "\n".join(sections) if sections else ""
