"""
SkillForge V6 — Execution Analysis + Failure Classification.

Format-adaptive: supports Gaia2 (Bash), ALFWorld (command), SWE-bench (LLM output),
and any new benchmark without format-specific code.
"""
from __future__ import annotations
import re
from .experience import Experience, FailureTaxonomy
from .gate import assess_task_complexity


def classify_failure(agent_actions: list[dict], oracle_actions: list[dict],
                     score: float, missing: list[str], extra: list[str]) -> FailureTaxonomy:
    """Classify failure into taxonomy (4 categories).
    
    - tool_failure: CLI errors, timeouts, permission issues (env noise)
    - over_action: agent did 2.5x+ more actions than needed
    - task_mismatch: most oracle steps missing (agent didn't understand task)
    - model_failure: correct tools but wrong args/logic (recoverable)
    """
    tool_error_patterns = [
        r'error:', r'timeout', r'permission denied', r'not found',
        r'connection refused', r'traceback', r'exception',
    ]
    
    error_count = 0
    for a in agent_actions:
        output = str(a.get("output", "")).lower()
        for pat in tool_error_patterns:
            if re.search(pat, output):
                error_count += 1
                break
    
    if error_count >= 3:
        return FailureTaxonomy(
            category="tool_failure",
            root_cause=f"Multiple tool errors ({error_count} actions had errors)",
            is_tool_chain=True, recoverable=True,
        )
    
    if len(agent_actions) > len(oracle_actions) * 2.5 and score < 0.5:
        return FailureTaxonomy(
            category="over_action",
            root_cause=f"Agent used {len(agent_actions)} actions vs {len(oracle_actions)} expected",
            is_tool_chain=False, recoverable=True,
        )
    
    if score < 0.3 and len(missing) > len(oracle_actions) * 0.7:
        return FailureTaxonomy(
            category="task_mismatch",
            root_cause=f"Agent missed {len(missing)}/{len(oracle_actions)} required steps",
            is_tool_chain=False, recoverable=True,
        )
    
    return FailureTaxonomy(
        category="model_failure",
        root_cause=f"Correct approach but {len(missing)} missing steps",
        is_tool_chain=False, recoverable=True,
    )


def analyze_execution(task_id: str, task_desc: str,
                      agent_actions: list[dict], oracle_actions: list[dict],
                      token_cost: int = 0, time_cost: float = 0.0,
                      augmentation_used: str = "") -> Experience:
    """Format-adaptive execution analysis.
    
    Supports:
    - Gaia2:    agent={"tool":"Bash","input":{"command":"..."}}  oracle={"app":"X","fn":"Y"}
    - Generic:  agent={"command":"go to desk"}                   oracle={"command":"go to desk"}
    - LLM:      agent={"tool":"LLM","output":"response text"}    oracle={"output":"expected"}
    """
    # ─── Extract agent tool sequence ──────────────────────────────────
    agent_tools = []
    agent_cmds = []
    for a in agent_actions:
        if a.get('tool') == 'Bash':
            cmd = a.get('input', {}).get('command', '') if isinstance(a.get('input'), dict) else ''
            agent_cmds.append(cmd)
            clean = re.sub(r'GAIA2_STATE_DIR=\S+\s*', '', cmd).strip()
            parts = clean.split()
            if parts:
                tool_fn = f"{parts[0]} {parts[1]}" if len(parts) > 1 and not parts[1].startswith('-') else parts[0]
                agent_tools.append(tool_fn)
        elif a.get('command'):
            cmd = a['command']
            agent_cmds.append(cmd)
            agent_tools.append(cmd.lower())
        elif a.get('output'):
            output = a['output']
            agent_cmds.append(output[:200])
            for line in output.split('\n'):
                line_clean = line.strip().lower()
                if any(line_clean.startswith(v) for v in ('go to', 'take', 'put', 'use', 'open', 'close', 'clean', 'heat', 'cool')):
                    agent_tools.append(line_clean)
            if not agent_tools:
                agent_tools.append(output[:50].lower())
    
    # ─── Extract oracle tool sequence ─────────────────────────────────
    oracle_tools = []
    for o in oracle_actions:
        if o.get('app') and o.get('fn'):
            oracle_tools.append(f"{o['app']}.{o['fn']}")
        elif o.get('command'):
            oracle_tools.append(o['command'].lower())
        elif o.get('output'):
            oracle_tools.append(o['output'][:50].lower())
    
    # ─── Match (word overlap ≥2 or substring) ─────────────────────────
    matched = 0
    used = set()
    for ot in oracle_tools:
        ot_parts = set(ot.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
        for j, at in enumerate(agent_tools):
            if j not in used:
                at_parts = set(at.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
                overlap = ot_parts & at_parts
                if len(overlap) >= 2 or ot.lower() in at.lower() or at.lower() in ot.lower():
                    matched += 1
                    used.add(j)
                    break
    
    score = matched / len(oracle_tools) if oracle_tools else 0
    
    # Missing + Extra
    missing = []
    for ot in oracle_tools:
        found = False
        ot_parts = set(ot.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
        for at in agent_tools:
            at_parts = set(at.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ').split())
            overlap = ot_parts & at_parts
            if len(overlap) >= 2 or ot.lower() in at.lower() or at.lower() in ot.lower():
                found = True
                break
        if not found:
            missing.append(ot)
    
    extra = [at for j, at in enumerate(agent_tools) if j not in used]
    
    # Outcome
    if score >= 1.0:
        outcome = "success"
    elif score >= 0.5:
        outcome = "partial"
    else:
        outcome = "failure"
    
    # Failure taxonomy
    taxonomy = FailureTaxonomy()
    failure_reason = ""
    if outcome != "success":
        taxonomy = classify_failure(agent_actions, oracle_actions, score, missing, extra)
        failure_reason = taxonomy.root_cause or f"Missing {len(missing)} required steps"
    
    complexity = assess_task_complexity(task_desc)
    
    return Experience(
        task_id=task_id, task_desc=task_desc,
        tool_sequence=agent_tools, action_commands=agent_cmds[:15],
        outcome=outcome, score=score,
        missing_steps=missing, extra_steps=extra[:10],
        failure_reason=failure_reason,
        failure_taxonomy={
            "category": taxonomy.category,
            "root_cause": taxonomy.root_cause,
            "is_tool_chain": taxonomy.is_tool_chain,
            "recoverable": taxonomy.recoverable,
        },
        token_cost=token_cost, time_cost=time_cost,
        task_complexity=complexity,
        augmentation_used=augmentation_used,
        timestamp=__import__("time").time(),
    )
