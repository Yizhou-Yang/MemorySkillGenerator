"""
BudgetGuard — per-tool call limits learned from execution statistics.

Key fix from v3: check() has NO side effects. budget = ceil(avg * 1.5), not max.
"""
from __future__ import annotations
import math


class BudgetGuard:
    DEFAULT_BUDGET = 10

    def __init__(self):
        self._budgets: dict[str, int] = {}
        self._counts: dict[str, int] = {}

    def check(self, tool_name: str) -> tuple[bool, str]:
        """Check budget. Pure query, no side effects."""
        count = self._counts.get(tool_name, 0)
        budget = self._budgets.get(tool_name, self.DEFAULT_BUDGET)
        if count >= budget:
            return True, f"Budget exceeded: {tool_name} ({count}/{budget})"
        return False, ""

    def record_call(self, tool_name: str):
        """Explicitly record a tool call (separated from check)."""
        self._counts[tool_name] = self._counts.get(tool_name, 0) + 1

    def learn(self, tool_avg_counts: dict[str, float]):
        """Learn budgets from oracle/successful execution averages."""
        for tool, avg in tool_avg_counts.items():
            self._budgets[tool] = max(3, math.ceil(avg * 1.5))

    def reset_task(self):
        self._counts.clear()

    def to_dict(self) -> dict:
        return dict(self._budgets)

    def from_dict(self, data: dict):
        for k, v in data.items():
            self._budgets[k] = int(v) if isinstance(v, (int, float)) else v.get("max", 10)
