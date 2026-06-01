"""
ArgFixer — minimal, high-confidence argument correction.

Key difference from v3 ActionCorrector:
  - NO auto-learning from error→success (too noisy)
  - Only manually curated or oracle-derived rules
  - Confidence threshold = 0.8 (v3 used 0.3 → too many false positives)
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FixRule:
    """A single high-confidence argument fix."""
    tool_pattern: str       # Substring match on tool name
    param: str              # Which parameter to fix
    condition: str          # "relative_path" | "missing"
    fix: str | bool         # "prepend_slash" | "set_true" | literal value
    confidence: float = 0.9
    fires: int = 0
    successes: int = 0

    def matches(self, tool_name: str, args: dict) -> bool:
        if self.tool_pattern not in tool_name:
            return False
        if self.condition == "relative_path":
            val = str(args.get(self.param, ""))
            return bool(val) and not val.startswith("/")
        if self.condition == "missing":
            return self.param not in args
        return False

    def apply(self, args: dict) -> dict:
        result = dict(args)
        if self.fix == "prepend_slash":
            val = str(result.get(self.param, ""))
            if val and not val.startswith("/"):
                result[self.param] = "/" + val
        elif self.fix == "set_true":
            result[self.param] = True
        elif isinstance(self.fix, str):
            result[self.param] = self.fix
        self.fires += 1
        return result


class ArgFixer:
    def __init__(self, min_confidence: float = 0.8):
        self.rules: list[FixRule] = []
        self.min_confidence = min_confidence

    def intercept(self, tool_name: str, args: dict) -> tuple[dict, bool, str]:
        """Apply matching high-confidence fixes."""
        modified = False
        reasons = []
        for rule in self.rules:
            if rule.confidence >= self.min_confidence and rule.matches(tool_name, args):
                args = rule.apply(args)
                modified = True
                reasons.append(f"{rule.tool_pattern}.{rule.param}")
        return args, modified, "; ".join(reasons)

    def add_rule(self, rule: FixRule):
        self.rules.append(rule)

    def to_dict(self) -> list[dict]:
        return [{"tool": r.tool_pattern, "param": r.param, "cond": r.condition,
                 "fix": r.fix, "conf": r.confidence, "fires": r.fires} for r in self.rules]

    def from_dict(self, data: list[dict]):
        for d in data:
            self.rules.append(FixRule(
                tool_pattern=d["tool"], param=d["param"],
                condition=d["cond"], fix=d["fix"],
                confidence=d.get("conf", 0.9), fires=d.get("fires", 0)
            ))
