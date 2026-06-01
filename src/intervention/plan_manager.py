"""
Plan Manager — enforced action sequences learned from successful executions.

A PlanRule is NOT a prompt hint. It's a deterministic execution sequence
that the agent MUST follow when triggered.

Learning: When a task is completed successfully, the action sequence is recorded.
Next time a similar task appears, the agent executes the learned plan directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanRule:
    """
    A learned execution plan — a fixed sequence of actions for a task type.
    
    When triggered, the agent follows this sequence instead of free-form reasoning.
    This is plan-level intervention: skip the "think what to do" step.
    """
    plan_id: str
    keywords: list[str]  # Task matching keywords
    steps: list[dict]  # [{"tool": "X", "arg_keys": [...], "hint": "..."}, ...]
    confidence: float = 0.5
    times_used: int = 0
    avg_outcome: float = 0.0
    source_task: str = ""

    def match_score(self, task_desc: str) -> float:
        """How well does this plan match a task description?"""
        if not self.keywords:
            return 0.0
        task_lower = task_desc.lower()
        matched = sum(1 for kw in self.keywords if kw in task_lower)
        return matched / len(self.keywords)

    def get_step(self, index: int) -> dict | None:
        """Get a specific step from the plan."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    @property
    def length(self) -> int:
        return len(self.steps)


class PlanManager:
    """
    Manages learned execution plans.
    
    Lifecycle:
      1. record_success(task_desc, action_sequence) — learn from successful execution
      2. get_plan(task_desc) — retrieve matching plan for new task
      3. strengthen(plan_id, outcome) — increase confidence if plan worked again
      4. weaken(plan_id) — decrease confidence if plan failed
    """

    def __init__(self, min_confidence: float = 0.3):
        self.plans: list[PlanRule] = []
        self.min_confidence = min_confidence
        self._plan_counter = 0
        self._stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'and', 'or', 'but', 'if', 'that', 'this', 'it', 'my', 'all',
            'i', 'me', 'we', 'you', 'them', 'their', 'please', 'each',
            'after', 'before', 'also', 'then', 'not', 'no', 'any',
        }

    def get_plan(self, task_desc: str) -> PlanRule | None:
        """
        Find the best matching plan for a task.
        Returns None if no plan matches above threshold.
        """
        best_plan = None
        best_score = 0.3  # Minimum match score

        for plan in self.plans:
            if plan.confidence < self.min_confidence:
                continue
            score = plan.match_score(task_desc)
            if score > best_score:
                best_plan = plan
                best_score = score

        return best_plan

    def get_plan_as_sequence(self, task_desc: str) -> list[str]:
        """
        Get plan as a list of tool names to execute in order.
        This is what the agent loop uses to enforce the plan.
        """
        plan = self.get_plan(task_desc)
        if not plan:
            return []
        plan.times_used += 1
        return [step.get("tool", "") for step in plan.steps if step.get("tool")]

    def record_success(self, task_desc: str, action_sequence: list[dict], outcome: float = 1.0):
        """
        Learn a new plan from a successful execution.
        
        action_sequence: [{"tool": "X__Y", "args": {...}}, ...]
        """
        keywords = self._extract_keywords(task_desc)
        if not keywords:
            return None

        # Check if similar plan exists → strengthen
        existing = self._find_similar(keywords)
        if existing:
            existing.times_used += 1
            existing.avg_outcome = 0.7 * existing.avg_outcome + 0.3 * outcome
            existing.confidence = min(0.95, existing.confidence + 0.05)
            return existing

        # Create new plan
        self._plan_counter += 1
        steps = []
        for action in action_sequence[:12]:  # Max 12 steps
            tool = action.get("tool", "")
            args = action.get("args", {})
            arg_keys = list(args.keys())[:3] if isinstance(args, dict) else []
            steps.append({
                "tool": tool,
                "arg_keys": arg_keys,
                "hint": f"({', '.join(arg_keys)})" if arg_keys else "",
            })

        plan = PlanRule(
            plan_id=f"plan_{self._plan_counter:04d}",
            keywords=keywords,
            steps=steps,
            confidence=min(0.7, outcome),
            avg_outcome=outcome,
            source_task=task_desc[:100],
        )
        self.plans.append(plan)
        return plan

    def strengthen(self, plan_id: str, outcome: float):
        """Plan succeeded again → increase confidence."""
        for plan in self.plans:
            if plan.plan_id == plan_id:
                plan.confidence = min(0.95, plan.confidence + 0.1)
                plan.avg_outcome = 0.7 * plan.avg_outcome + 0.3 * outcome
                plan.times_used += 1
                return

    def weaken(self, plan_id: str):
        """Plan failed → decrease confidence."""
        for plan in self.plans:
            if plan.plan_id == plan_id:
                plan.confidence = max(0.0, plan.confidence - 0.2)
                return

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        words = text.lower().split()[:25]
        keywords = [
            w.strip('.,!?"\'()[]{}') for w in words
            if w.strip('.,!?"\'()[]{}') not in self._stop_words and len(w) > 2
        ]
        return keywords[:12]

    def _find_similar(self, keywords: list[str]) -> PlanRule | None:
        """Find existing plan with similar keywords."""
        keyword_set = set(keywords)
        for plan in self.plans:
            plan_set = set(plan.keywords)
            if not plan_set:
                continue
            overlap = len(keyword_set & plan_set) / max(len(plan_set), 1)
            if overlap > 0.5:
                return plan
        return None

    def to_dict(self) -> list[dict]:
        """Serialize for persistence."""
        return [
            {
                "plan_id": p.plan_id,
                "keywords": p.keywords,
                "steps": p.steps,
                "confidence": p.confidence,
                "times_used": p.times_used,
                "avg_outcome": p.avg_outcome,
                "source_task": p.source_task,
            }
            for p in self.plans
        ]

    def from_dict(self, data: list[dict]):
        """Load from persisted data."""
        for d in data:
            self.plans.append(PlanRule(
                plan_id=d.get("plan_id", f"plan_{len(self.plans)}"),
                keywords=d.get("keywords", []),
                steps=d.get("steps", []),
                confidence=d.get("confidence", 0.5),
                times_used=d.get("times_used", 0),
                avg_outcome=d.get("avg_outcome", 0.0),
                source_task=d.get("source_task", ""),
            ))

    @property
    def size(self) -> int:
        return len(self.plans)

    @property
    def active_plans(self) -> int:
        return sum(1 for p in self.plans if p.confidence >= self.min_confidence)
