"""Completion Gate — Generic conditional completion control.

Framework-level mechanism to block premature task completion in multi-phase agentic tasks.
"""
from __future__ import annotations


class CompletionGate:
    """Framework-level mechanism to block premature task completion.

    In multi-phase tasks (e.g., GAIA2 twist tasks), the agent may try to
    output ALL_DONE after completing only the first phase. This gate
    enforces that certain conditions must be met before completion is allowed.

    Usage:
        gate = CompletionGate()
        gate.set_condition("wait_for_reply", required=True)
        ...
        gate.mark_satisfied("notify_user")  # Phase 1 done
        ...
        if processed.is_completion:
            if not gate.can_complete():
                # Inject rejection message
                hint = gate.get_rejection_hint()
                conversation_history += hint
                continue
            break

    SRDP Theory: This prevents δ_att(consistency_collapse) — where the agent
    "forgets" the second phase exists and silently terminates early.
    """

    def __init__(self):
        self._conditions: dict[str, bool] = {}
        self._required: set[str] = set()
        self._rejection_count: int = 0
        self._max_rejections: int = 5  # Safety valve: don't loop forever

    def set_condition(self, name: str, required: bool = True) -> None:
        """Register a condition that must be satisfied before completion."""
        self._conditions[name] = False
        if required:
            self._required.add(name)

    def mark_satisfied(self, name: str) -> None:
        """Mark a condition as satisfied."""
        if name in self._conditions:
            self._conditions[name] = True

    def can_complete(self) -> bool:
        """Check if all required conditions are met for completion.

        Returns True if:
        - All required conditions are satisfied, OR
        - Max rejections exceeded (safety valve to prevent infinite loops)
        """
        if self._rejection_count >= self._max_rejections:
            return True  # Safety valve: allow completion after too many rejections
        return all(self._conditions.get(c, False) for c in self._required)

    def get_rejection_hint(self) -> str:
        """Generate a rejection message listing unsatisfied conditions."""
        self._rejection_count += 1
        unsatisfied = [c for c in self._required if not self._conditions.get(c, False)]
        conditions_text = ", ".join(unsatisfied)
        return (
            f"\n\n🛑 ALL_DONE REJECTED (attempt {self._rejection_count}/{self._max_rejections}): "
            f"Required conditions not met: [{conditions_text}].\n"
            f"You MUST satisfy these before completing the task.\n"
        )

    @property
    def rejection_count(self) -> int:
        return self._rejection_count