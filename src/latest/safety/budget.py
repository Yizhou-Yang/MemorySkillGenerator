"""Budget Tracker — Generic turn budget management for agentic loops.

Injects budget awareness prompts at key thresholds to help the agent
prioritize actions and avoid wasting turns on stuck sub-tasks.
"""
from __future__ import annotations


class BudgetTracker:
    """Framework-level turn budget management for agentic loops.

    Injects budget awareness prompts at key thresholds to help the agent
    prioritize actions and avoid wasting turns on stuck sub-tasks.

    SRDP Theory: Budget awareness reduces δ_att(retrieval_dilution) by
    preventing the conversation history from growing too large with
    repetitive failed attempts.
    """

    def __init__(self, max_turns: int, thresholds: tuple[int, ...] = (75, 50, 30, 15, 5)):
        self._max_turns = max_turns
        self._thresholds = thresholds

    def get_budget_hint(self, current_turn: int) -> str:
        """Get a budget awareness hint for the current turn, if at a threshold.

        Returns empty string if not at a threshold.
        """
        remaining = self._max_turns - current_turn - 1
        if remaining not in self._thresholds:
            return ""

        if remaining <= 5:
            return (
                f"\n[⏱ {remaining} turns remaining. "
                "FINAL: Complete task NOW or output ALL_DONE.]"
            )
        elif remaining <= 15:
            return (
                f"\n[⏱ {remaining} turns remaining. "
                "URGENT: Complete primary task NOW. Skip any stuck sub-tasks.]"
            )
        elif remaining <= 30:
            return (
                f"\n[⏱ {remaining} turns remaining. "
                "Prioritize core actions. Don't waste turns on exhaustive searches.]"
            )
        else:
            return (
                f"\n[⏱ {remaining} turns remaining. "
                "On track. Reserve turns for any follow-up/twist.]"
            )