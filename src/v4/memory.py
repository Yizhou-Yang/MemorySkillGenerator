"""
ExperienceMemory — simple outcome store for statistical learning.

No embeddings, no EMA, no layers. Just facts.
Used by BudgetGuard (tool frequency) and reporting (improvement tracking).
"""
from __future__ import annotations
import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class TaskOutcome:
    """Record of one task execution."""
    task_id: str
    task_desc: str
    tools_used: list[str]
    tool_counts: dict[str, int]
    success: bool
    score: float
    failure_patterns: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ExperienceMemory:
    def __init__(self):
        self.outcomes: list[TaskOutcome] = []

    def record(self, outcome: TaskOutcome):
        self.outcomes.append(outcome)

    def get_tool_stats(self) -> dict[str, float]:
        """Average tool usage across successful tasks."""
        successful = [o for o in self.outcomes if o.success]
        if not successful:
            return {}
        total: Counter = Counter()
        for o in successful:
            total.update(o.tool_counts)
        return {tool: count / len(successful) for tool, count in total.items()}

    def get_failure_patterns(self) -> list[str]:
        patterns: Counter = Counter()
        for o in self.outcomes:
            if not o.success:
                patterns.update(o.failure_patterns)
        return [p for p, _ in patterns.most_common(10)]

    @property
    def success_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.success) / len(self.outcomes)

    def to_dict(self) -> list[dict]:
        return [{"task_id": o.task_id, "desc": o.task_desc[:100], "tools": o.tools_used,
                 "counts": o.tool_counts, "success": o.success, "score": o.score,
                 "failures": o.failure_patterns, "ts": o.timestamp} for o in self.outcomes]

    def from_dict(self, data: list[dict]):
        for d in data:
            self.outcomes.append(TaskOutcome(
                task_id=d["task_id"], task_desc=d.get("desc", ""),
                tools_used=d.get("tools", []), tool_counts=d.get("counts", {}),
                success=d.get("success", False), score=d.get("score", 0.0),
                failure_patterns=d.get("failures", []), timestamp=d.get("ts", 0.0)
            ))
