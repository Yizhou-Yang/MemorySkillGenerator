"""
LoopGuard — blocks exact-repeat and cyclic actions.

Pure defense: an agent repeating the same failed action is ALWAYS wasting budget.
"""
from __future__ import annotations
import json
import hashlib
from collections import deque


class LoopGuard:
    def __init__(self, max_repeat: int = 2, window: int = 15):
        self.max_repeat = max_repeat
        self.window = window
        self._history: deque[str] = deque(maxlen=window)
        self._blocked = 0

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Should this action be blocked? Returns (blocked, reason)."""
        sig = f"{tool_name}:{hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:12]}"

        recent = list(self._history)[-self.max_repeat:]
        if len(recent) == self.max_repeat and all(s == sig for s in recent):
            self._blocked += 1
            return True, f"Blocked: exact repeat ({self.max_repeat}x) of {tool_name}"

        self._history.append(sig)
        return False, ""

    def reset(self):
        self._history.clear()

    @property
    def stats(self) -> dict:
        return {"blocked": self._blocked, "history_len": len(self._history)}
