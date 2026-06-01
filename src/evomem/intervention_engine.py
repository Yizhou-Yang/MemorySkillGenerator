"""
SkillForge Intervention Engine — Cross-Benchmark, Multi-Level.

This is the RUNTIME component that sits inside the agent loop.
It learns corrections, plans, and patterns FROM EXECUTION (not hardcoded).

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  Generic Framework (works on ANY benchmark)                  │
  │    - Loop detector (repeated action → break)                 │
  │    - Error pattern learner (tool error → remember correction)│
  │    - Plan generator (successful sequences → reusable plans)  │
  │    - Action corrector (learned arg fixes)                    │
  │    - Rollback guard (if intervention hurts → disable it)     │
  └─────────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  Learned State (populated during execution, persisted)       │
  │    - error_corrections: {(tool, bad_args) → good_args}       │
  │    - successful_plans: {task_embedding → action_sequence}     │
  │    - blocked_patterns: {action_sig → reason}                 │
  │    - tool_schemas: {tool_name → observed_return_format}      │
  └─────────────────────────────────────────────────────────────┘

Learning happens ONLINE:
  - Agent calls tool with args → gets error → engine records (bad_args, error)
  - Agent retries with different args → succeeds → engine records correction
  - Agent completes a task → engine records successful plan
  - Next time similar situation arises → engine intervenes
"""
import json
import os
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class LearnedCorrection:
    """A learned arg correction: when tool+bad_args fails, use good_args."""
    tool_name: str
    bad_pattern: dict  # Args that caused error (generalized)
    good_pattern: dict  # Args that succeeded after
    error_message: str
    times_applied: int = 0
    times_helped: int = 0  # Did the correction lead to success?
    confidence: float = 0.5

    def matches(self, tool: str, args: dict) -> bool:
        """Check if this correction applies to given action."""
        if tool != self.tool_name:
            return False
        for key, bad_val in self.bad_pattern.items():
            if key in args and self._pattern_match(str(args[key]), str(bad_val)):
                return True
        return False

    def apply(self, args: dict) -> dict:
        """Apply correction to args."""
        corrected = dict(args)
        for key, good_val in self.good_pattern.items():
            if key in corrected:
                corrected[key] = good_val
        self.times_applied += 1
        return corrected

    @staticmethod
    def _pattern_match(actual: str, pattern: str) -> bool:
        """Generalized match: exact or prefix match."""
        if actual == pattern:
            return True
        # Relative path pattern: doesn't start with /
        if pattern == "__RELATIVE_PATH__" and not actual.startswith('/'):
            return True
        return False


@dataclass
class LearnedPlan:
    """A successful action sequence that can be reused."""
    task_pattern: str  # Generalized task description keywords
    action_sequence: list  # [{"tool": ..., "args_template": ...}, ...]
    times_used: int = 0
    avg_outcome: float = 0.0
    confidence: float = 0.5

    def match_score(self, task_desc: str) -> float:
        """How well does this plan match the given task?"""
        task_lower = task_desc.lower()
        keywords = self.task_pattern.lower().split()
        if not keywords:
            return 0.0
        matched = sum(1 for kw in keywords if kw in task_lower)
        return matched / len(keywords)


@dataclass 
class ToolObservation:
    """Recorded observation of what a tool returns."""
    tool_name: str
    return_type: str  # "list", "dict", "string", "error"
    key_fields: list  # Fields observed in return value
    sample_size: int = 0


class InterventionEngine:
    """
    Cross-benchmark multi-level intervention engine.
    Learns from execution experience, no hardcoding.
    """

    def __init__(self, state_path: str = "/workspace/intervention_state.json"):
        self.state_path = state_path

        # Learned state (populated during execution)
        self.corrections: list = []  # LearnedCorrection
        self.plans: list = []  # LearnedPlan
        self.tool_observations: dict = {}  # tool_name → ToolObservation
        self.blocked_actions: dict = {}  # action_sig → block_reason

        # Runtime tracking
        self._action_history: list = []  # Recent actions for loop detection
        self._pending_error: dict = {}  # tool → (bad_args, error) awaiting correction
        self._current_task: str = ""
        self._current_plan_actions: list = []  # Actions in current task (for plan learning)
        self._intervention_stats = {
            "corrections_applied": 0,
            "plans_injected": 0,
            "loops_broken": 0,
            "total_steps": 0,
        }

        # Rollback protection
        self._intervention_enabled = True
        self._disable_until_step = 0

        # Load persisted state
        self._load_state()

    # ================================================================
    # PLANNING LEVEL: Inject learned plans
    # ================================================================

    def get_plan_for_task(self, task_desc: str) -> str:
        """If we have a learned plan for this task type, return it."""
        if not self._intervention_enabled:
            return ""
        if not self.plans:
            return ""

        best_plan = None
        best_score = 0.3  # Minimum threshold

        for plan in self.plans:
            score = plan.match_score(task_desc)
            if score > best_score and plan.confidence > 0.3:
                best_plan = plan
                best_score = score

        if not best_plan:
            return ""

        lines = [f"[Learned Plan (confidence={best_plan.confidence:.0%}, used {best_plan.times_used}x)]"]
        for i, step in enumerate(best_plan.action_sequence[:8], 1):
            tool = step.get('tool', '?')
            hint = step.get('hint', '')
            lines.append(f"  {i}. {tool} — {hint}")
        lines.append("[Follow this sequence. One action per step.]")

        self._intervention_stats["plans_injected"] += 1
        return "\n".join(lines)

    # ================================================================
    # ACTION LEVEL: Intercept and correct actions
    # ================================================================

    def intercept_action(self, tool_name: str, arguments: dict) -> tuple:
        """
        Check if this action should be corrected based on learned patterns.
        Returns: (corrected_args, was_corrected, correction_reason)
        """
        if not self._intervention_enabled:
            return arguments, False, ""

        self._intervention_stats["total_steps"] += 1

        # --- Loop detection (generic) ---
        action_sig = f"{tool_name}:{json.dumps(arguments, sort_keys=True)[:200]}"
        self._action_history.append(action_sig)
        self._action_history = self._action_history[-15:]

        # Detect 3+ consecutive identical actions
        if len(self._action_history) >= 3:
            if self._action_history[-1] == self._action_history[-2] == self._action_history[-3]:
                self._intervention_stats["loops_broken"] += 1
                # Don't block, but mark for the model to see
                return arguments, False, ""

        # --- Apply learned corrections ---
        for correction in self.corrections:
            if correction.matches(tool_name, arguments):
                corrected = correction.apply(arguments)
                self._intervention_stats["corrections_applied"] += 1
                return corrected, True, f"Applied correction: {correction.error_message[:50]}"

        return arguments, False, ""

    # ================================================================
    # TOOL LEVEL: Learn from observations
    # ================================================================

    def observe_result(self, tool_name: str, arguments: dict, observation: str, had_error: bool):
        """
        Called AFTER a tool executes. Learns from the result.
        This is where the engine actually LEARNS.
        """
        # Track for plan building
        self._current_plan_actions.append({
            "tool": tool_name,
            "args_keys": list(arguments.keys()) if isinstance(arguments, dict) else [],
            "success": not had_error,
            "hint": self._generate_hint(tool_name, arguments, observation, had_error),
        })

        if had_error:
            # Record failed attempt → waiting for correction
            self._pending_error[tool_name] = {
                "bad_args": dict(arguments) if isinstance(arguments, dict) else {},
                "error": str(observation)[:200],
                "step": len(self._action_history),
            }
        else:
            # Success! Check if this corrects a previous error
            if tool_name in self._pending_error:
                pending = self._pending_error.pop(tool_name)
                # Learn correction: bad_args → good_args
                self._learn_correction(
                    tool_name,
                    pending["bad_args"],
                    dict(arguments) if isinstance(arguments, dict) else {},
                    pending["error"]
                )

            # Learn tool observation schema
            self._learn_tool_schema(tool_name, observation)

    def _learn_correction(self, tool_name: str, bad_args: dict, good_args: dict, error_msg: str):
        """Learn a new correction from error→success pattern."""
        # Generalize the pattern (don't store exact values, store patterns)
        bad_pattern = {}
        good_pattern = {}

        for key in set(list(bad_args.keys()) + list(good_args.keys())):
            bad_val = str(bad_args.get(key, ''))
            good_val = str(good_args.get(key, ''))

            if bad_val != good_val:
                # Generalize: if bad was relative path, pattern = __RELATIVE_PATH__
                if not bad_val.startswith('/') and good_val.startswith('/'):
                    bad_pattern[key] = "__RELATIVE_PATH__"
                    good_pattern[key] = good_val  # Store the working absolute path
                else:
                    bad_pattern[key] = bad_val
                    good_pattern[key] = good_val

        if bad_pattern:
            correction = LearnedCorrection(
                tool_name=tool_name,
                bad_pattern=bad_pattern,
                good_pattern=good_pattern,
                error_message=error_msg,
            )
            self.corrections.append(correction)
            self._save_state()

    def _learn_tool_schema(self, tool_name: str, observation: str):
        """Learn what a tool returns (for future plan generation)."""
        obs_str = str(observation)[:500]

        # Detect return type
        if obs_str.startswith('[') or obs_str.startswith('{'):
            try:
                data = json.loads(obs_str[:2000])
                if isinstance(data, list):
                    return_type = "list"
                    key_fields = list(data[0].keys())[:5] if data and isinstance(data[0], dict) else []
                elif isinstance(data, dict):
                    return_type = "dict"
                    key_fields = list(data.keys())[:5]
                else:
                    return_type = "string"
                    key_fields = []
            except:
                return_type = "string"
                key_fields = []
        elif "error" in obs_str.lower() or "Error" in obs_str:
            return_type = "error"
            key_fields = []
        else:
            return_type = "string"
            key_fields = []

        self.tool_observations[tool_name] = {
            "return_type": return_type,
            "key_fields": key_fields,
            "sample_size": self.tool_observations.get(tool_name, {}).get("sample_size", 0) + 1,
        }

    def _generate_hint(self, tool_name: str, arguments: dict, observation: str, had_error: bool) -> str:
        """Generate a brief hint about this step for plan storage."""
        if had_error:
            return f"ERROR with {list(arguments.keys())[:2]}"
        args_brief = ", ".join(f"{k}" for k in list(arguments.keys())[:2])
        return f"{args_brief} → ok"

    # ================================================================
    # TASK LIFECYCLE: Learn plans from complete executions
    # ================================================================

    def start_task(self, task_desc: str):
        """Called at the beginning of a new task."""
        self._current_task = task_desc
        self._current_plan_actions = []
        self._pending_error = {}

    def end_task(self, outcome: float):
        """
        Called at end of task. If successful, save as a learned plan.
        """
        if not self._current_task or not self._current_plan_actions:
            return

        # Only learn from reasonably successful executions
        if outcome >= 0.3 and len(self._current_plan_actions) <= 30:
            # Extract keywords from task for matching
            task_keywords = self._extract_keywords(self._current_task)

            # Only keep successful steps for the plan
            successful_steps = [a for a in self._current_plan_actions if a.get('success', True)]

            if successful_steps:
                plan = LearnedPlan(
                    task_pattern=task_keywords,
                    action_sequence=successful_steps[:10],
                    times_used=0,
                    avg_outcome=outcome,
                    confidence=min(0.8, outcome),
                )

                # Check if similar plan exists → strengthen it
                existing = self._find_similar_plan(task_keywords)
                if existing:
                    existing.times_used += 1
                    existing.avg_outcome = 0.7 * existing.avg_outcome + 0.3 * outcome
                    existing.confidence = min(0.95, existing.confidence + 0.1)
                else:
                    self.plans.append(plan)

                self._save_state()

    def _extract_keywords(self, task_desc: str) -> str:
        """Extract meaningful keywords from task description."""
        # Remove common words, keep domain-specific ones
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of',
                     'in', 'for', 'on', 'with', 'at', 'by', 'from', 'that', 'this',
                     'it', 'its', 'my', 'your', 'all', 'and', 'or', 'but', 'if',
                     'then', 'them', 'their', 'there', 'here', 'i', 'me', 'we', 'you'}
        words = task_desc.lower().split()[:30]
        keywords = [w.strip('.,!?"\'') for w in words if w.strip('.,!?"\'') not in stop_words and len(w) > 2]
        return " ".join(keywords[:15])

    def _find_similar_plan(self, keywords: str) -> LearnedPlan:
        """Find existing plan with similar keywords."""
        for plan in self.plans:
            score = plan.match_score(keywords)
            if score > 0.5:
                return plan
        return None

    # ================================================================
    # ROLLBACK PROTECTION
    # ================================================================

    def report_outcome(self, with_intervention: float, without_intervention: float = None):
        """
        Compare outcome with/without intervention.
        If intervention hurts, disable temporarily.
        """
        if without_intervention is not None and with_intervention < without_intervention - 0.1:
            # Intervention made things worse → disable for next 5 steps
            self._intervention_enabled = False
            self._disable_until_step = self._intervention_stats["total_steps"] + 5

        # Re-enable after cooldown
        if not self._intervention_enabled and self._intervention_stats["total_steps"] >= self._disable_until_step:
            self._intervention_enabled = True

    # ================================================================
    # Persistence
    # ================================================================

    def _save_state(self):
        """Persist learned state to disk."""
        state = {
            "corrections": [
                {"tool": c.tool_name, "bad": c.bad_pattern, "good": c.good_pattern,
                 "error": c.error_message, "applied": c.times_applied, "confidence": c.confidence}
                for c in self.corrections
            ],
            "plans": [
                {"pattern": p.task_pattern, "actions": p.action_sequence,
                 "used": p.times_used, "outcome": p.avg_outcome, "confidence": p.confidence}
                for p in self.plans
            ],
            "tool_observations": self.tool_observations,
            "stats": self._intervention_stats,
        }
        try:
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except:
            pass

    def _load_state(self):
        """Load persisted state."""
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)

            for c in state.get("corrections", []):
                self.corrections.append(LearnedCorrection(
                    tool_name=c["tool"], bad_pattern=c["bad"], good_pattern=c["good"],
                    error_message=c["error"], times_applied=c.get("applied", 0),
                    confidence=c.get("confidence", 0.5)
                ))
            for p in state.get("plans", []):
                self.plans.append(LearnedPlan(
                    task_pattern=p["pattern"], action_sequence=p["actions"],
                    times_used=p.get("used", 0), avg_outcome=p.get("outcome", 0.5),
                    confidence=p.get("confidence", 0.5)
                ))
            self.tool_observations = state.get("tool_observations", {})
            self._intervention_stats = state.get("stats", self._intervention_stats)
        except:
            pass

    def get_summary(self) -> dict:
        """Return learning summary."""
        return {
            "corrections_learned": len(self.corrections),
            "plans_learned": len(self.plans),
            "tools_observed": len(self.tool_observations),
            "stats": self._intervention_stats,
            "intervention_enabled": self._intervention_enabled,
        }
