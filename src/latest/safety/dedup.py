"""No-Repeat Guard — Runtime idempotency guard for agentic loops.

Prevents re-executing identical (tool, args) pairs in multi-turn agentic tasks.
"""
from __future__ import annotations


class NoRepeatGuard:
    """Runtime idempotency guard — prevents re-executing identical (tool, args) pairs.

    Theory (SRDP — δ_att reduction):
        In multi-turn agentic loops, LLMs can get stuck repeating the same operation
        with identical parameters, especially when awaiting a reply that never arrives.
        Each repeated execution adds noise to the conversation history and wastes the
        context budget. This guard provides a HARD runtime block: if the agent tries
        to execute an (op_id, args) tuple that was already executed in the current
        task, execution is skipped and a warning is injected into the conversation.

        This is a general safety mechanism, not benchmark-specific. It applies to
        any agentic task where the LLM may repeat actions.

    Usage:
        guard = NoRepeatGuard()

        # In agent loop, before executing:
        if guard.would_repeat(op_id, args):
            warning = guard.get_warning(op_id)
            # Inject warning into conversation, skip execution
            continue
        guard.record(op_id, args)
        # ... execute tool ...
    """

    def __init__(self):
        self._seen: set[tuple] = set()

    def would_repeat(self, op_id: str, args: dict[str, object]) -> bool:
        """Check if this exact (op_id, args) was already executed."""
        key = self._make_key(op_id, args)
        return key in self._seen

    def record(self, op_id: str, args: dict[str, object]) -> None:
        """Record that this (op_id, args) was executed."""
        key = self._make_key(op_id, args)
        self._seen.add(key)

    def reset(self) -> None:
        """Reset for a new task."""
        self._seen.clear()

    def get_warning(self, op_id: str) -> str:
        """Get the warning message to inject when a repeat is detected."""
        return (
            f"\n\n⚠️ DUPLICATE CALL DETECTED: {op_id} with same params was already executed.\n"
            f"This operation has NO effect. You MUST do something different.\n"
            f"Choose a different operation or use ALL_DONE if the task is complete.\n"
        )

    @staticmethod
    def _make_key(op_id: str, args: dict[str, object]) -> tuple:
        """Create a hashable key from op_id and args, sorting for determinism."""
        import json
        return (op_id, json.dumps(args, sort_keys=True, default=str))