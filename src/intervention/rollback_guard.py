"""
Rollback Guard — monitors intervention effectiveness, disables harmful interventions.

Principle: If intervention makes things WORSE than baseline, disable it.
Uses A/B comparison: track outcome with vs without intervention on similar tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InterventionRecord:
    """Record of an intervention's effect on a task."""
    task_hash: str
    intervention_type: str  # "action", "plan", "budget"
    intervention_id: str
    outcome_with: float  # Outcome when intervention was active
    outcome_without: float | None = None  # Baseline outcome (if known)


class RollbackGuard:
    """
    Monitors whether interventions help or hurt.
    
    Logic:
      - Track outcomes per intervention type
      - If an intervention's avg outcome < baseline avg → disable it
      - Disabled interventions can be re-enabled after cooldown
    
    This prevents the system from degrading performance through bad learned rules.
    """

    def __init__(self, cooldown_steps: int = 10, harm_threshold: float = 0.05):
        self.cooldown_steps = cooldown_steps
        self.harm_threshold = harm_threshold

        self._records: list[InterventionRecord] = []
        self._disabled: dict[str, int] = {}  # intervention_id → disabled_until_step
        self._current_step = 0
        self._baseline_outcomes: list[float] = []

    def is_enabled(self, intervention_id: str) -> bool:
        """Check if a specific intervention is currently enabled."""
        if intervention_id not in self._disabled:
            return True
        return self._current_step >= self._disabled[intervention_id]

    def record_baseline(self, outcome: float):
        """Record a baseline (no intervention) outcome."""
        self._baseline_outcomes.append(outcome)

    def record_intervention(self, intervention_id: str, intervention_type: str, outcome: float):
        """Record an intervention's outcome."""
        self._current_step += 1
        self._records.append(InterventionRecord(
            task_hash="",
            intervention_type=intervention_type,
            intervention_id=intervention_id,
            outcome_with=outcome,
        ))

        # Check if this intervention is hurting
        intervention_outcomes = [
            r.outcome_with for r in self._records
            if r.intervention_id == intervention_id
        ]

        if len(intervention_outcomes) >= 3 and self._baseline_outcomes:
            avg_intervention = sum(intervention_outcomes[-5:]) / len(intervention_outcomes[-5:])
            avg_baseline = sum(self._baseline_outcomes[-10:]) / len(self._baseline_outcomes[-10:])

            if avg_intervention < avg_baseline - self.harm_threshold:
                # Intervention is hurting → disable
                self._disabled[intervention_id] = self._current_step + self.cooldown_steps

    def step(self):
        """Advance step counter (call once per task)."""
        self._current_step += 1

    @property
    def disabled_count(self) -> int:
        return sum(1 for step in self._disabled.values() if self._current_step < step)

    @property
    def stats(self) -> dict:
        return {
            "total_records": len(self._records),
            "disabled_interventions": self.disabled_count,
            "baseline_avg": (
                sum(self._baseline_outcomes) / len(self._baseline_outcomes)
                if self._baseline_outcomes else 0
            ),
            "current_step": self._current_step,
        }
