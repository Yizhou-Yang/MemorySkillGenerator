"""
Skill evaluator.

Evaluates the quality of induced skills, including:
- Success rate: re-running tasks with the skill injected.
- Compression ratio: tokens(trajectory) / tokens(skill).
- Reuse frequency: how often a skill is matched.
- Transfer capability: performance on different datasets.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.models import (
    Skill,
    SkillEvalResult,
    Trajectory,
    TransformVariant,
)
from src.utils.llm import LLMClient


class SkillEvaluator:
    """Skill quality evaluator."""

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.num_validation_runs: int = self.config.get("num_validation_runs", 3)

    def evaluate_skill(
        self,
        skill: Skill,
        validation_tasks: list[dict[str, str]],
        source_trajectory: Trajectory | None = None,
    ) -> SkillEvalResult:
        """
        Evaluate a single skill's quality.

        Args:
            skill: The skill to evaluate.
            validation_tasks: Validation task list, each containing
                ``task_id``, ``description``, and ``expected``.
            source_trajectory: Source trajectory (used to compute compression ratio).

        Returns:
            The evaluation result.
        """
        logger.info(
            f"Evaluating skill: {skill.name} (variant={skill.source_variant})"
        )

        # 1. Success rate: run validation tasks with the skill injected
        success_count = 0
        validation_details: list[dict[str, Any]] = []

        for task in validation_tasks:
            task_result = self._validate_with_skill(skill, task)
            validation_details.append(task_result)
            if task_result.get("success", False):
                success_count += 1

        success_rate = (
            success_count / len(validation_tasks) if validation_tasks else 0.0
        )

        # 2. Compression ratio
        compression_ratio = 0.0
        if source_trajectory:
            traj_chars = sum(
                len(step.content) for step in source_trajectory.steps
            )
            skill_chars = skill.compactness
            compression_ratio = (
                traj_chars / skill_chars if skill_chars > 0 else 0.0
            )

        eval_result = SkillEvalResult(
            skill_id=skill.skill_id,
            variant=skill.source_variant or TransformVariant.TRAJ_TO_SKILL,
            success_rate=success_rate,
            compression_ratio=compression_ratio,
            validation_details=validation_details,
        )

        logger.info(
            f"Skill evaluation complete: {skill.name}, "
            f"success_rate={success_rate:.1%}, "
            f"compression_ratio={compression_ratio:.1f}x"
        )
        return eval_result

    def _validate_with_skill(
        self,
        skill: Skill,
        task: dict[str, str],
    ) -> dict[str, Any]:
        """
        Run a single validation task with the skill injected.

        Args:
            skill: The skill to inject.
            task: The validation task.

        Returns:
            A result dict with ``task_id``, ``success``, ``response``, etc.
        """
        skill_prompt = self._format_skill_as_prompt(skill)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task-execution agent. "
                    "Below is a skill you can use:\n\n"
                    f"{skill_prompt}\n\n"
                    "Apply the skill above to complete the task. "
                    "Give your final answer directly."
                ),
            },
            {"role": "user", "content": task.get("description", "")},
        ]

        try:
            response = self.llm_client.chat(messages)
            expected = task.get("expected", "")

            # Simple answer matching (MVP phase)
            success = self._check_answer(response, expected)

            return {
                "task_id": task.get("task_id", ""),
                "success": success,
                "response": response[:500],
                "expected": expected,
            }
        except Exception as exc:
            logger.error(f"Validation task failed: {exc}")
            return {
                "task_id": task.get("task_id", ""),
                "success": False,
                "error": str(exc),
            }

    def _format_skill_as_prompt(self, skill: Skill) -> str:
        """Format a skill as an injectable prompt."""
        parts = [f"## Skill: {skill.name}", f"{skill.description}", ""]

        if skill.preconditions:
            parts.append("**Preconditions:**")
            for precondition in skill.preconditions:
                parts.append(f"- {precondition}")
            parts.append("")

        if skill.procedure:
            parts.append("**Procedure:**")
            for idx, step in enumerate(skill.procedure, 1):
                parts.append(f"{idx}. {step}")
            parts.append("")

        if skill.constraints:
            parts.append("**Constraints:**")
            for constraint in skill.constraints:
                parts.append(f"- {constraint}")
            parts.append("")

        if skill.facts:
            parts.append("**Facts:**")
            for fact in skill.facts:
                parts.append(f"- {fact}")

        return "\n".join(parts)

    def _check_answer(self, response: str, expected: str) -> bool:
        """
        Check whether the answer is correct.

        MVP phase: simple substring containment check.
        Can be replaced with a more precise method (e.g. LLM-as-judge) later.
        """
        if not expected:
            return True  # No expected answer — pass by default
        return expected.lower().strip() in response.lower()

    def compare_variants(
        self,
        results: dict[TransformVariant, list[SkillEvalResult]],
    ) -> dict[str, Any]:
        """
        Compare evaluation results across the three variants.

        Args:
            results: Evaluation results keyed by variant.

        Returns:
            A comparison summary dict.
        """
        comparison: dict[str, Any] = {}
        for variant, eval_results in results.items():
            if not eval_results:
                continue
            avg_success = sum(
                result.success_rate for result in eval_results
            ) / len(eval_results)
            avg_compression = sum(
                result.compression_ratio for result in eval_results
            ) / len(eval_results)
            comparison[variant.value] = {
                "num_skills": len(eval_results),
                "avg_success_rate": round(avg_success, 4),
                "avg_compression_ratio": round(avg_compression, 2),
            }

        logger.info(
            f"Variant comparison results: {json.dumps(comparison, indent=2)}"
        )
        return comparison