"""
Action Corrector — deterministic action-level skill intervention.

Each ActionRule is a learned correction that fires on EVERY matching action.
Not a suggestion, not a prompt — a direct parameter modification.

Learning: When agent calls tool(bad_args) → error, then retries with tool(good_args) → success,
we learn ActionRule(condition=matches_bad, transform=apply_good).
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ActionRule:
    """
    A single learned action correction rule.
    
    Fires deterministically: if condition matches, transform is applied.
    No LLM involved, no prompt, no chance of being ignored.
    """
    rule_id: str
    tool_pattern: str  # Tool name pattern (e.g., "Files" matches any Files__* tool)
    condition: dict  # {param: pattern} — when to fire
    transform: dict  # {param: new_value_or_rule} — what to change
    learned_from: str = ""  # Which task/error taught us this
    confidence: float = 0.5
    times_fired: int = 0
    times_helped: int = 0  # Tracked post-hoc

    def matches(self, tool_name: str, args: dict) -> bool:
        """Check if this rule should fire for given action."""
        # Tool pattern match
        if self.tool_pattern not in tool_name:
            return False
        # Condition match
        for param, pattern in self.condition.items():
            if param not in args:
                if pattern == "__MISSING__":
                    return True  # Param is missing, rule wants to add it
                continue
            val = str(args[param])
            if pattern == "__RELATIVE_PATH__" and not val.startswith('/'):
                return True
            if pattern == "__EMPTY__" and (not val or val == '{}' or val == '[]'):
                return True
            if val == str(pattern):
                return True
        return False

    def apply(self, args: dict) -> dict:
        """Apply correction to args. Returns modified args."""
        corrected = dict(args)
        for param, transform in self.transform.items():
            if transform == "__PREPEND_SLASH__":
                if param in corrected and not str(corrected[param]).startswith('/'):
                    corrected[param] = '/' + str(corrected[param])
            elif transform == "__ADD_IF_MISSING__":
                pass  # Handled by specific value below
            else:
                # Direct value assignment or addition
                corrected[param] = transform
        self.times_fired += 1
        return corrected


class ActionCorrector:
    """
    Manages a collection of ActionRules.
    
    Lifecycle:
      1. observe_error(tool, args, error) — record a failed attempt
      2. observe_success(tool, args) — if same tool succeeds after, learn correction
      3. intercept(tool, args) — apply all matching corrections to an action
    """

    def __init__(self):
        self.rules: list[ActionRule] = []
        self._pending_errors: dict[str, dict] = {}  # tool → {args, error}
        self._rule_counter = 0

    def intercept(self, tool_name: str, args: dict) -> tuple[dict, bool, str]:
        """
        Apply all matching corrections to an action.
        Returns: (corrected_args, was_modified, reason)
        """
        modified = False
        reasons = []

        for rule in self.rules:
            if rule.confidence < 0.3:
                continue  # Skip low-confidence rules
            if rule.matches(tool_name, args):
                args = rule.apply(args)
                modified = True
                reasons.append(f"{rule.rule_id}: {rule.learned_from[:40]}")

        return args, modified, "; ".join(reasons)

    def observe_error(self, tool_name: str, args: dict, error_msg: str):
        """Record a failed action. Waiting for subsequent success to learn correction."""
        self._pending_errors[tool_name] = {
            "args": dict(args),
            "error": error_msg[:200],
        }

    def observe_success(self, tool_name: str, args: dict):
        """
        Record a successful action. If there was a prior error on same tool,
        learn the correction (bad_args → good_args).
        """
        if tool_name not in self._pending_errors:
            return None

        pending = self._pending_errors.pop(tool_name)
        bad_args = pending["args"]
        good_args = dict(args)

        # Compute diff: what changed between bad and good?
        condition = {}
        transform = {}

        for key in set(list(bad_args.keys()) + list(good_args.keys())):
            bad_val = str(bad_args.get(key, ''))
            good_val = str(good_args.get(key, ''))

            if bad_val != good_val:
                # Generalize the condition
                if not bad_val.startswith('/') and good_val.startswith('/'):
                    condition[key] = "__RELATIVE_PATH__"
                    transform[key] = "__PREPEND_SLASH__"
                elif bad_val == '' and good_val:
                    condition[key] = "__EMPTY__"
                    transform[key] = good_val
                else:
                    condition[key] = bad_val
                    transform[key] = good_val

            # Check for missing params in bad that exist in good
            if key not in bad_args and key in good_args:
                condition[key] = "__MISSING__"
                transform[key] = good_val

        if not condition:
            return None

        # Create rule
        self._rule_counter += 1
        rule = ActionRule(
            rule_id=f"ar_{self._rule_counter:04d}",
            tool_pattern=tool_name.split('__')[0] if '__' in tool_name else tool_name,
            condition=condition,
            transform=transform,
            learned_from=pending["error"],
            confidence=0.6,
        )
        self.rules.append(rule)
        return rule

    def add_rule(self, rule: ActionRule):
        """Manually add a rule (e.g., loaded from persistence)."""
        self.rules.append(rule)

    def to_dict(self) -> list[dict]:
        """Serialize rules for persistence."""
        return [
            {
                "rule_id": r.rule_id,
                "tool_pattern": r.tool_pattern,
                "condition": r.condition,
                "transform": r.transform,
                "learned_from": r.learned_from,
                "confidence": r.confidence,
                "times_fired": r.times_fired,
            }
            for r in self.rules
        ]

    def from_dict(self, data: list[dict]):
        """Load rules from persisted data."""
        for d in data:
            self.rules.append(ActionRule(
                rule_id=d.get("rule_id", f"ar_{len(self.rules)}"),
                tool_pattern=d.get("tool_pattern", ""),
                condition=d.get("condition", {}),
                transform=d.get("transform", {}),
                learned_from=d.get("learned_from", ""),
                confidence=d.get("confidence", 0.5),
                times_fired=d.get("times_fired", 0),
            ))

    @property
    def size(self) -> int:
        return len(self.rules)

    def summary(self) -> str:
        return f"ActionCorrector: {self.size} rules, {sum(r.times_fired for r in self.rules)} total fires"
