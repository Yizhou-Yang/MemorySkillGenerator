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

    result = "\n".join(parts)
    # Budget: drop low-priority fields (never truncate mid-content)
    if estimate_token_count(result) > budget_tokens and len(parts) > 4:
        result = "\n".join(parts[:4])
    return result

def format_failure_experience(exp: Experience, budget_tokens: int = 600) -> str:
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

    result = "\n".join(parts)
    if estimate_token_count(result) > budget_tokens and len(parts) > 5:
        result = "\n".join(parts[:5])
    return result

def _build_qa_hint(task_desc: str, library: ExperienceLibrary, token_budget: int = 600) -> str:
    """Enhanced hints for QA tasks — includes both reasoning patterns and pitfall warnings."""
    candidates = library.retrieve_similar(task_desc, top_k=5)
    if not candidates:
        return ""
    
    hints = []
    pitfalls = []
    
    for exp in candidates:
        ft = exp.failure_taxonomy
        
        # Successful experiences → reasoning hints
        if exp.outcome == "success" and exp.score >= 0.5:
            if ft.get("ai_refined") and ft.get("generalized_steps"):
                hints.append(f"✓ {ft.get('causal_lesson', '')}")
            elif exp.action_commands:
                # Extract the key reasoning from the successful answer
                answer_preview = exp.action_commands[0][:150] if exp.action_commands else ""
                if answer_preview:
                    hints.append(f"✓ Similar question answered: {exp.task_desc[:80]}")
        
        # Failed experiences → pitfall warnings
        elif exp.outcome == "failure" and ft.get("ai_refined"):
            causal = ft.get("causal_lesson", "")
            avoidance = ft.get("avoidance_note", "")
            if avoidance and len(avoidance) > 20:
                pitfalls.append(f"⚠ {avoidance}")
            elif causal and len(causal) > 20 and "mismatch" not in causal.lower():
                pitfalls.append(f"⚠ {causal}")
    
    sections = []
    if hints:
        seen = set()
        unique_hints = [h for h in hints if not (h[:40].lower() in seen or seen.add(h[:40].lower()))]
        if unique_hints:
            sections.append("## Reasoning Patterns from Similar Tasks")
            sections.extend(f"- {h}" for h in unique_hints[:3])
    
    if pitfalls:
        seen = set()
        unique_pitfalls = [p for p in pitfalls if not (p[:40].lower() in seen or seen.add(p[:40].lower()))]
        if unique_pitfalls:
            sections.append("\n## Common Pitfalls to Avoid")
            sections.extend(f"- {p}" for p in unique_pitfalls[:2])
    
    if not sections:
        return ""
    
    result = "\n".join(sections)
    if estimate_token_count(result) > token_budget:
        result = "\n".join(sections[:4])
    return result

def build_augmented_prompt(task_desc: str, library: ExperienceLibrary,
                           token_budget: int = 2000,
                           top_k_success: int = 2, top_k_failure: int = 2,
                           expected: str = "", metadata: dict | None = None) -> str:
    """Route: qa→enhanced hints, agentic/embodied→full injection with gate."""
    task_type = classify_task_type(task_desc, expected=expected, metadata=metadata)

    if task_type == "qa":
        return _build_qa_hint(task_desc, library, token_budget=min(token_budget, 600))

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
