"""
SkillForge V4 Runtime — Cross-Benchmark Generic Intervention.

Strategy for +5pp F1 (benchmark-agnostic):
  1. Efficiency prompt: tell agent to minimize actions
  2. Fail-retry block: if exact same action just failed, block retry
  3. Exploration cap: 3 consecutive failures on same tool → block that tool
  4. Self-learned budget: total actions capped at 2x successful-task average

NO oracle data used. NO benchmark-specific rules.
"""
from __future__ import annotations
import json
import hashlib
import time
from collections import deque
from typing import Any


# === Efficiency System Prompt (cross-benchmark) ===

EFFICIENCY_PROMPT = """Important execution constraints:
- Complete the task with the MINIMUM number of actions possible.
- Never retry an action that just failed with the same parameters.
- If a tool call fails 2-3 times, switch to a completely different approach.
- Each action should make clear progress toward the goal.
- Do NOT explore or gather unnecessary information — act decisively."""


class SkillForgeV4:
    """
    Cross-benchmark intervention runtime for CodeBuddy Agent SDK.
    
    Integrates via:
      - canUseTool callback (action-level blocking)
      - system prompt injection (efficiency instruction)
    
    All heuristics are generic (no oracle, no benchmark-specific rules).
    """

    def __init__(self, max_total_actions: int = 30):
        # Config
        self.max_total_actions = max_total_actions  # Task-level budget
        
        # Per-task state
        self._action_count = 0
        self._last_action_sig: str = ""
        self._last_action_failed: bool = False
        self._tool_fail_streak: dict[str, int] = {}  # tool → consecutive failures
        self._tool_success_seen: dict[str, bool] = {}  # tool → ever succeeded?
        self._history: deque[str] = deque(maxlen=50)
        
        # Stats
        self._blocked_retry = 0
        self._blocked_exploration = 0
        self._blocked_budget = 0
        self._total_tasks = 0
    
    # === SDK Integration ===
    
    async def can_use_tool(self, tool_name: str, input_data: dict, options: Any):
        """CodeBuddy SDK canUseTool callback."""
        from codebuddy_agent_sdk import PermissionResultAllow, PermissionResultDeny
        
        sig = self._make_sig(tool_name, input_data)
        
        # 1. Total action budget
        if self._action_count >= self.max_total_actions:
            self._blocked_budget += 1
            return PermissionResultDeny(
                message="Action budget reached. Wrap up and finish the task now."
            )
        
        # 2. Fail-retry block: exact same action just failed → block
        if self._last_action_failed and sig == self._last_action_sig:
            self._blocked_retry += 1
            return PermissionResultDeny(
                message="This exact action just failed. Try a different approach or different parameters."
            )
        
        # 3. Exploration cap: tool failed 3+ times consecutively → block
        if self._tool_fail_streak.get(tool_name, 0) >= 3:
            if not self._tool_success_seen.get(tool_name, False):
                self._blocked_exploration += 1
                return PermissionResultDeny(
                    message=f"{tool_name} has failed 3 times consecutively. Use a different tool."
                )
        
        # 4. Exact duplicate detection (same sig already in history)
        if sig in self._history:
            # Allow if this tool has succeeded before (might be legitimate reuse)
            if not self._tool_success_seen.get(tool_name, False):
                self._blocked_retry += 1
                return PermissionResultDeny(
                    message="You already tried this exact action. Try something different."
                )
        
        # Allow
        self._action_count += 1
        self._last_action_sig = sig
        self._history.append(sig)
        
        return PermissionResultAllow(updated_input=input_data)
    
    def record_tool_result(self, tool_name: str, success: bool):
        """
        Call after each tool execution to track success/failure.
        Used by exploration cap logic.
        """
        if success:
            self._tool_fail_streak[tool_name] = 0
            self._tool_success_seen[tool_name] = True
            self._last_action_failed = False
        else:
            self._tool_fail_streak[tool_name] = self._tool_fail_streak.get(tool_name, 0) + 1
            self._last_action_failed = True
    
    # === Task Lifecycle ===
    
    def start_task(self, task_id: str = ""):
        """Reset state for new task."""
        self._action_count = 0
        self._last_action_sig = ""
        self._last_action_failed = False
        self._tool_fail_streak.clear()
        self._tool_success_seen.clear()
        self._history.clear()
        self._total_tasks += 1
    
    @property
    def system_prompt(self) -> str:
        """Return the efficiency prompt to inject."""
        return EFFICIENCY_PROMPT
    
    # === Helpers ===
    
    def _make_sig(self, tool_name: str, args: dict) -> str:
        """Create a signature for dedup."""
        args_str = json.dumps(args, sort_keys=True, default=str)
        h = hashlib.md5(args_str.encode()).hexdigest()[:16]
        return f"{tool_name}:{h}"
    
    @property
    def stats(self) -> dict:
        return {
            "blocked_retry": self._blocked_retry,
            "blocked_exploration": self._blocked_exploration,
            "blocked_budget": self._blocked_budget,
            "total_tasks": self._total_tasks,
            "current_actions": self._action_count,
        }


def create_runtime(successful_task_avg_actions: float = 15.0) -> SkillForgeV4:
    """
    Create runtime with self-learned budget.
    
    Args:
        successful_task_avg_actions: Average actions in successful R1 tasks.
            Set to 2x this value as the budget cap.
            Default 15 is conservative (most benchmarks have 5-15 oracle actions).
    """
    budget = int(successful_task_avg_actions * 2)
    return SkillForgeV4(max_total_actions=budget)
