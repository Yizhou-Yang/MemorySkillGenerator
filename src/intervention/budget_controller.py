"""
Budget Controller — per-tool call limits learned from execution statistics.

Generic: works on any benchmark. Learns natural call frequencies from successful executions.
If a tool is called way more than its learned budget → block further calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolBudget:
    """Budget for a single tool, learned from execution history."""
    tool_name: str
    max_calls: int  # Maximum calls per task
    avg_calls: float = 0.0  # Average from successful tasks
    learned_from_n: int = 0  # How many tasks contributed to this estimate
    current_calls: int = 0  # Current task call count

    def is_exceeded(self) -> bool:
        return self.current_calls > self.max_calls

    def record_call(self):
        self.current_calls += 1

    def reset(self):
        self.current_calls = 0


class BudgetController:
    """
    Controls per-tool call budgets.
    
    Learning:
      - After each successful task, record how many times each tool was called.
      - Budget = max(observed_avg * 2, 3). Generous but prevents runaway loops.
    
    Enforcement:
      - If tool exceeds budget → block further calls.
      - Budget can be overridden for specific tools if needed.
    """

    DEFAULT_BUDGET = 15  # If no data, allow up to 15 calls per tool per task

    def __init__(self):
        self._budgets: dict[str, ToolBudget] = {}
        self._current_counts: dict[str, int] = {}

    def check(self, tool_name: str) -> tuple[bool, str]:
        """
        Check if tool call is within budget.
        Returns: (should_block, reason)
        """
        self._current_counts[tool_name] = self._current_counts.get(tool_name, 0) + 1
        count = self._current_counts[tool_name]

        budget = self._budgets.get(tool_name)
        if budget:
            if count > budget.max_calls:
                return True, f"Budget exceeded: {tool_name} ({count}/{budget.max_calls})"
        else:
            if count > self.DEFAULT_BUDGET:
                return True, f"Default budget exceeded: {tool_name} ({count}/{self.DEFAULT_BUDGET})"

        return False, ""

    def learn_from_execution(self, tool_counts: dict[str, int]):
        """
        Learn budgets from a completed task.
        tool_counts: {tool_name: call_count} from one successful execution.
        """
        for tool, count in tool_counts.items():
            if tool not in self._budgets:
                self._budgets[tool] = ToolBudget(
                    tool_name=tool,
                    max_calls=max(count * 2, 3),
                    avg_calls=float(count),
                    learned_from_n=1,
                )
            else:
                budget = self._budgets[tool]
                # Running average
                n = budget.learned_from_n
                budget.avg_calls = (budget.avg_calls * n + count) / (n + 1)
                budget.max_calls = max(int(budget.avg_calls * 2.5), 3)
                budget.learned_from_n = n + 1

    def reset_task(self):
        """Reset current counts for new task."""
        self._current_counts.clear()

    def set_budget(self, tool_name: str, max_calls: int):
        """Manually set budget for a tool."""
        self._budgets[tool_name] = ToolBudget(tool_name=tool_name, max_calls=max_calls)

    def to_dict(self) -> dict:
        return {
            name: {"max": b.max_calls, "avg": b.avg_calls, "n": b.learned_from_n}
            for name, b in self._budgets.items()
        }

    def from_dict(self, data: dict):
        for name, d in data.items():
            self._budgets[name] = ToolBudget(
                tool_name=name,
                max_calls=d.get("max", self.DEFAULT_BUDGET),
                avg_calls=d.get("avg", 0.0),
                learned_from_n=d.get("n", 0),
            )

    @property
    def size(self) -> int:
        return len(self._budgets)
