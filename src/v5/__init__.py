"""
SkillForge V5 — EvoMem Reproduction + Failure-Based Feedback Loop

Architecture:
  Round 1: Run baseline, record ALL experiences (success + failure)
  Round 2: For each new task:
    1. [EvoMem] Retrieve similar SUCCESSFUL experiences → inject into prompt
    2. [Failure Feedback] Retrieve similar FAILED experiences → inject "what went wrong"
    
  Key principle: ALWAYS ADD INFORMATION, NEVER REDUCE.
  No blocking, no limiting, no efficiency prompts.
  Give the agent MORE context to make better decisions.

EvoMem reproduction:
  - After each task: store {task_desc, action_sequence, outcome, tool_sequence}
  - Before each task: retrieve top-K similar experiences by task description
  - Inject as "Here's what worked on similar tasks: ..."

Failure Feedback Loop (our contribution):
  - Store failed experiences with analysis: "what was missing", "what went wrong"
  - Before each task: also retrieve similar FAILURES
  - Inject as "On similar tasks, agents failed because: ... Make sure to also: ..."
  - This adds information the agent wouldn't otherwise have
"""
from __future__ import annotations
import json
import os
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


# ─── Experience Store ─────────────────────────────────────────────────────

@dataclass
class Experience:
    """A recorded task execution experience."""
    task_id: str
    task_desc: str
    tool_sequence: list[str]       # Ordered list of tools called
    action_commands: list[str]     # Full bash commands executed
    outcome: str                   # "success" | "partial" | "failure"
    score: float                   # 0-1 metric
    missing_steps: list[str]       # What oracle had that agent didn't do
    extra_steps: list[str]         # What agent did that wasn't needed
    failure_reason: str            # Why it failed (if applicable)
    timestamp: float = 0.0


class ExperienceLibrary:
    """Stores and retrieves task execution experiences."""
    
    def __init__(self):
        self.experiences: list[Experience] = []
    
    def record(self, exp: Experience):
        self.experiences.append(exp)
    
    def retrieve_similar(self, task_desc: str, top_k: int = 3, 
                         outcome_filter: str | None = None) -> list[Experience]:
        """Retrieve most similar experiences by keyword overlap.
        
        Args:
            task_desc: Current task description
            top_k: Number of experiences to return
            outcome_filter: "success" | "failure" | None (all)
        """
        candidates = self.experiences
        if outcome_filter:
            candidates = [e for e in candidates if e.outcome == outcome_filter]
        
        if not candidates:
            return []
        
        # Simple keyword similarity (production would use embeddings)
        task_words = set(task_desc.lower().split())
        scored = []
        for exp in candidates:
            exp_words = set(exp.task_desc.lower().split())
            if not exp_words:
                continue
            overlap = len(task_words & exp_words) / max(len(task_words | exp_words), 1)
            scored.append((overlap, exp))
        
        scored.sort(key=lambda x: -x[0])
        return [exp for _, exp in scored[:top_k]]
    
    def get_successful(self) -> list[Experience]:
        return [e for e in self.experiences if e.outcome == "success"]
    
    def get_failed(self) -> list[Experience]:
        return [e for e in self.experiences if e.outcome in ("failure", "partial")]
    
    def to_dict(self) -> list[dict]:
        return [
            {
                "task_id": e.task_id, "task_desc": e.task_desc,
                "tool_sequence": e.tool_sequence, "action_commands": e.action_commands,
                "outcome": e.outcome, "score": e.score,
                "missing_steps": e.missing_steps, "extra_steps": e.extra_steps,
                "failure_reason": e.failure_reason, "timestamp": e.timestamp,
            }
            for e in self.experiences
        ]
    
    def from_dict(self, data: list[dict]):
        for d in data:
            self.experiences.append(Experience(**d))
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                self.from_dict(json.load(f))


# ─── Prompt Augmentation ──────────────────────────────────────────────────

def format_success_experience(exp: Experience) -> str:
    """Format a successful experience for prompt injection."""
    steps = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(exp.action_commands[:10]))
    return f"""[Successful approach for similar task]
Task: {exp.task_desc[:150]}
Steps taken:
{steps}
Result: Task completed successfully."""


def format_failure_experience(exp: Experience) -> str:
    """Format a failed experience for prompt injection (our contribution)."""
    lines = [f"[Warning from similar failed task]"]
    lines.append(f"Task: {exp.task_desc[:150]}")
    if exp.failure_reason:
        lines.append(f"What went wrong: {exp.failure_reason}")
    if exp.missing_steps:
        lines.append(f"Steps that were MISSING (make sure to do these):")
        for step in exp.missing_steps[:5]:
            lines.append(f"  - {step}")
    return "\n".join(lines)


def build_augmented_prompt(task_desc: str, library: ExperienceLibrary,
                           top_k_success: int = 2, top_k_failure: int = 2) -> str:
    """Build the experience-augmented section to inject into prompt.
    
    This is the core of EvoMem + Failure Feedback:
    - Retrieve similar successes → "here's what worked"
    - Retrieve similar failures → "here's what went wrong, don't repeat"
    
    Always ADDS information, never removes or restricts.
    """
    sections = []
    
    # EvoMem: successful experiences
    successes = library.retrieve_similar(task_desc, top_k=top_k_success, outcome_filter="success")
    if successes:
        sections.append("## Relevant Experience (from similar successful tasks)\n")
        for exp in successes:
            sections.append(format_success_experience(exp))
            sections.append("")
    
    # Failure Feedback: failed experiences
    failures = library.retrieve_similar(task_desc, top_k=top_k_failure, outcome_filter="failure")
    if not failures:
        failures = library.retrieve_similar(task_desc, top_k=top_k_failure, outcome_filter="partial")
    
    if failures:
        sections.append("## Lessons from Similar Failed Attempts\n")
        for exp in failures:
            sections.append(format_failure_experience(exp))
            sections.append("")
    
    return "\n".join(sections) if sections else ""


# ─── Experience Recording (post-task analysis) ────────────────────────────

def analyze_execution(task_id: str, task_desc: str, 
                      agent_actions: list[dict], oracle_actions: list[dict]) -> Experience:
    """Analyze a completed task execution and create an Experience record.
    
    Compares agent's actions against oracle to determine:
    - What the agent did right
    - What was missing (oracle had, agent didn't)
    - What was extra (agent did, oracle didn't need)
    """
    # Extract tool sequences
    agent_tools = []
    agent_cmds = []
    for a in agent_actions:
        if a.get('tool') == 'Bash':
            cmd = a.get('input', {}).get('command', '') if isinstance(a.get('input'), dict) else ''
            agent_cmds.append(cmd)
            # Extract CLI tool name
            import re
            clean = re.sub(r'GAIA2_STATE_DIR=\S+\s*', '', cmd).strip()
            parts = clean.split()
            if parts:
                tool_fn = f"{parts[0]} {parts[1]}" if len(parts) > 1 and not parts[1].startswith('-') else parts[0]
                agent_tools.append(tool_fn)
    
    oracle_tools = []
    for o in oracle_actions:
        oracle_tools.append(f"{o.get('app','')}.{o.get('fn','')}")
    
    # Match (function-level)
    matched = 0
    used = set()
    for ot in oracle_tools:
        for j, at in enumerate(agent_tools):
            if j not in used:
                # Rough match: oracle "Calendar.delete_calendar_event" ≈ agent "calendar delete-event"
                ot_parts = ot.lower().replace('.', ' ').replace('_', ' ').split()
                at_parts = at.lower().replace('-', ' ').replace('_', ' ').split()
                if set(ot_parts) & set(at_parts):
                    matched += 1
                    used.add(j)
                    break
    
    score = matched / len(oracle_tools) if oracle_tools else 0
    
    # Determine missing steps
    missing = []
    for i, ot in enumerate(oracle_tools):
        found = False
        for at in agent_tools:
            ot_parts = set(ot.lower().replace('.', ' ').replace('_', ' ').split())
            at_parts = set(at.lower().replace('-', ' ').replace('_', ' ').split())
            if ot_parts & at_parts:
                found = True
                break
        if not found:
            missing.append(ot)
    
    # Extra steps
    extra = [at for j, at in enumerate(agent_tools) if j not in used]
    
    # Determine outcome
    if score >= 1.0:
        outcome = "success"
    elif score >= 0.5:
        outcome = "partial"
    else:
        outcome = "failure"
    
    # Failure reason
    failure_reason = ""
    if outcome != "success":
        if missing:
            failure_reason = f"Missing {len(missing)} required steps: {', '.join(missing[:3])}"
        elif len(agent_tools) > len(oracle_tools) * 2:
            failure_reason = "Too many unnecessary actions, lost focus on core task"
    
    return Experience(
        task_id=task_id,
        task_desc=task_desc,
        tool_sequence=agent_tools,
        action_commands=agent_cmds[:15],  # Keep first 15 commands
        outcome=outcome,
        score=score,
        missing_steps=missing,
        extra_steps=extra[:10],
        failure_reason=failure_reason,
        timestamp=time.time(),
    )


# ─── SkillForge V5 Module (for runner integration) ────────────────────────

class SkillForgeV5:
    """
    EvoMem + Failure Feedback module.
    
    Principle: ALWAYS ADD INFORMATION.
    - Injects relevant success experiences
    - Injects failure warnings with missing steps
    - Never blocks, never limits, never removes
    """
    
    def __init__(self, library_path: str | None = None):
        self.library = ExperienceLibrary()
        if library_path:
            self.library.load(library_path)
    
    def get_augmentation(self, task_desc: str) -> str:
        """Get experience-based prompt augmentation for a task."""
        return build_augmented_prompt(task_desc, self.library)
    
    def record_experience(self, task_id: str, task_desc: str,
                          agent_actions: list[dict], oracle_actions: list[dict]):
        """Record an experience after task completion."""
        exp = analyze_execution(task_id, task_desc, agent_actions, oracle_actions)
        self.library.record(exp)
    
    def save(self, path: str):
        self.library.save(path)
    
    def load(self, path: str):
        self.library.load(path)
    
    @property
    def stats(self) -> dict:
        return {
            "total": len(self.library.experiences),
            "success": len(self.library.get_successful()),
            "failed": len(self.library.get_failed()),
        }
