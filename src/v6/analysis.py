"""Format-adaptive execution analysis + failure classification. No hardcoded patterns."""
from __future__ import annotations
import re
from .experience import Experience, FailureTaxonomy
from .gate import assess_task_complexity


def classify_failure(agent_actions: list[dict], oracle_actions: list[dict],
                     score: float, missing: list[str], extra: list[str]) -> FailureTaxonomy:
    """4 categories: tool_failure / over_action / task_mismatch / model_failure."""
    # Tool failure: detect error patterns in action outputs
    error_count = 0
    for a in agent_actions:
        output = str(a.get("output", "") or a.get("observation", "")).lower()
        # Generic error indicators (not domain-specific)
        if any(kw in output for kw in ("error", "traceback", "exception", "timeout",
                                        "permission denied", "not found", "refused")):
            error_count += 1

    if error_count >= 3:
        return FailureTaxonomy(category="tool_failure",
            root_cause=f"Multiple errors in execution ({error_count})", is_tool_chain=True)
    if oracle_actions and len(agent_actions) > len(oracle_actions) * 2.5 and score < 0.5:
        return FailureTaxonomy(category="over_action",
            root_cause=f"{len(agent_actions)} actions vs {len(oracle_actions)} expected")
    if oracle_actions and score < 0.3 and len(missing) > len(oracle_actions) * 0.7:
        return FailureTaxonomy(category="task_mismatch",
            root_cause=f"Missed {len(missing)}/{len(oracle_actions)} required steps")
    return FailureTaxonomy(category="model_failure",
        root_cause=f"Correct approach but {len(missing)} missing steps")


def _extract_action_key(action: dict) -> str | None:
    """Extract a comparable key from any action format. Returns None if empty."""
    # Format 1: Bash tool call (e.g. Gaia2)
    if action.get('tool') == 'Bash':
        cmd = action.get('input', {}).get('command', '') if isinstance(action.get('input'), dict) else ''
        # Strip env var prefixes (GAIA2_STATE_DIR=... or any VAR=value pattern)
        clean = re.sub(r'\b[A-Z_]+=\S+\s*', '', cmd).strip()
        parts = clean.split()
        if parts:
            return f"{parts[0]} {parts[1]}" if len(parts) > 1 and not parts[1].startswith('-') else parts[0]
    # Format 2: Direct command dict (e.g. ALFWorld, generic)
    if action.get('command'):
        return action['command'].lower()
    # Format 3: Tool use block (e.g. SWE-bench agent)
    if action.get('tool') and action.get('input'):
        return f"{action['tool']}:{str(action['input'])[:50]}".lower()
    # Format 4: LLM output — extract action-like lines
    if action.get('output'):
        for line in action['output'].split('\n'):
            stripped = line.strip().lower()
            # An action line typically starts with a verb and is short
            if stripped and len(stripped.split()) <= 10 and not stripped.startswith(('#', '//', '"""')):
                words = stripped.split()
                if words[0].isalpha():
                    return stripped
    # Format 5: app.fn oracle format
    if action.get('app') and action.get('fn'):
        return f"{action['app']}.{action['fn']}"
    return None


def _match_actions(key_a: str, key_b: str) -> bool:
    """Check if two action keys match (word overlap ≥2 or substring containment)."""
    a_parts = set(key_a.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
    b_parts = set(key_b.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
    if len(a_parts & b_parts) >= 2:
        return True
    if key_a.lower() in key_b.lower() or key_b.lower() in key_a.lower():
        return True
    return False


def analyze_execution(task_id: str, task_desc: str,
                      agent_actions: list[dict], oracle_actions: list[dict],
                      token_cost: int = 0, time_cost: float = 0.0,
                      augmentation_used: str = "") -> Experience:
    """Format-adaptive: extracts action keys from any format, then matches."""
    # Extract keys
    agent_keys = [k for a in agent_actions if (k := _extract_action_key(a)) is not None]
    agent_cmds = [str(a.get('command', '') or a.get('input', {}).get('command', '')
                      or a.get('output', '')[:200])
                  for a in agent_actions]
    oracle_keys = [k for o in oracle_actions if (k := _extract_action_key(o)) is not None]

    # Match oracle → agent (ordered, greedy)
    matched, used = 0, set()
    for ok in oracle_keys:
        for j, ak in enumerate(agent_keys):
            if j not in used and _match_actions(ok, ak):
                matched += 1
                used.add(j)
                break

    score = matched / len(oracle_keys) if oracle_keys else 0
    missing = [ok for ok in oracle_keys if not any(_match_actions(ok, ak) for ak in agent_keys)]
    extra = [ak for j, ak in enumerate(agent_keys) if j not in used]

    outcome = "success" if score >= 1.0 else "partial" if score >= 0.5 else "failure"
    taxonomy = FailureTaxonomy()
    failure_reason = ""
    if outcome != "success":
        taxonomy = classify_failure(agent_actions, oracle_actions, score, missing, extra)
        failure_reason = taxonomy.root_cause

    return Experience(
        task_id=task_id, task_desc=task_desc,
        tool_sequence=agent_keys, action_commands=agent_cmds,
        outcome=outcome, score=score,
        missing_steps=missing, extra_steps=extra,
        failure_reason=failure_reason,
        failure_taxonomy={"category": taxonomy.category, "root_cause": taxonomy.root_cause,
                          "is_tool_chain": taxonomy.is_tool_chain, "recoverable": taxonomy.recoverable},
        token_cost=token_cost, time_cost=time_cost,
        task_complexity=assess_task_complexity(task_desc),
        augmentation_used=augmentation_used,
        timestamp=__import__("time").time(),
    )
