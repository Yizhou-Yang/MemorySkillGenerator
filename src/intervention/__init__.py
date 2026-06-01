"""
SkillForge Intervention Module — Cross-benchmark multi-level agent intervention.

Skills are NOT prompt-level hints. They are EXECUTABLE RULES that directly modify agent behavior.

Three intervention levels:
  1. Action-level: Intercept each action, apply learned corrections (deterministic)
  2. Plan-level: Enforce learned action sequences (agent follows plan, not free-form)
  3. Tool-level: Budget control, dedup cache, loop breaking (prevent waste)

All rules are LEARNED from execution experience, not hardcoded.
"""

from src.intervention.action_corrector import ActionCorrector, ActionRule
from src.intervention.plan_manager import PlanManager, PlanRule
from src.intervention.loop_detector import LoopDetector
from src.intervention.budget_controller import BudgetController
from src.intervention.rollback_guard import RollbackGuard

__all__ = [
    "ActionCorrector",
    "ActionRule",
    "PlanManager",
    "PlanRule",
    "LoopDetector",
    "BudgetController",
    "RollbackGuard",
]
