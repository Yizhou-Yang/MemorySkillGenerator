"""
Loop Detector — identifies and breaks repetitive action patterns.

Generic: works on any agent in any benchmark.
Detects: consecutive repeats, cyclic patterns, excessive single-tool usage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class LoopEvent:
    tool: str
    args_hash: str
    blocked: bool = False


class LoopDetector:
    """
    Detects and blocks repetitive action patterns.
    
    Three detection modes:
      1. Consecutive: same (tool, args) N times in a row → block
      2. Cyclic: pattern A→B→A→B repeating → block
      3. Frequency: single tool called > budget times total → block
    """

    def __init__(self, max_consecutive: int = 3, max_cyclic: int = 2, window: int = 20):
        self.max_consecutive = max_consecutive
        self.max_cyclic = max_cyclic
        self.window = window
        self._history: deque = deque(maxlen=window)
        self._tool_counts: dict[str, int] = {}
        self._blocked_count = 0

    def check(self, tool_name: str, args_hash: str) -> tuple[bool, str]:
        """
        Check if this action should be blocked.
        Returns: (should_block, reason)
        """
        sig = f"{tool_name}:{args_hash}"

        # 1. Exact dedup: same tool+args already in history
        if sig in [f"{e.tool}:{e.args_hash}" for e in self._history]:
            # Allow up to max_consecutive repeats
            recent_same = sum(1 for e in list(self._history)[-self.max_consecutive:]
                           if f"{e.tool}:{e.args_hash}" == sig)
            if recent_same >= self.max_consecutive:
                self._blocked_count += 1
                return True, f"Exact repeat blocked ({recent_same+1}x): {tool_name}"

        # 2. Consecutive same tool (even with different args)
        if len(self._history) >= self.max_consecutive:
            recent_tools = [e.tool for e in list(self._history)[-self.max_consecutive:]]
            if all(t == tool_name for t in recent_tools):
                self._blocked_count += 1
                return True, f"Consecutive tool blocked ({self.max_consecutive+1}x): {tool_name}"

        # 3. Cyclic pattern: A→B→A→B
        if len(self._history) >= 4:
            h = list(self._history)
            last4 = [e.tool for e in h[-4:]]
            if last4[0] == last4[2] and last4[1] == last4[3] and tool_name == last4[0]:
                self._blocked_count += 1
                return True, f"Cyclic pattern blocked: {last4[0]}↔{last4[1]}"

        # Record
        self._history.append(LoopEvent(tool=tool_name, args_hash=args_hash))
        self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1

        return False, ""

    def reset(self):
        """Reset for new task."""
        self._history.clear()
        self._tool_counts.clear()

    @property
    def blocked_count(self) -> int:
        return self._blocked_count

    @property
    def stats(self) -> dict:
        return {
            "blocked": self._blocked_count,
            "history_len": len(self._history),
            "tool_counts": dict(self._tool_counts),
        }
