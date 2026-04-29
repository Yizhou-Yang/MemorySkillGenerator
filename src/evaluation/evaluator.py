"""
Skill evaluator.

Evaluates the quality of induced skills using multiple metrics:
- LLM-as-judge scoring (0-10): a separate LLM call rates the skill's
  ability to guide task completion.  This replaces naive substring matching
  and produces continuous scores that differentiate variant quality.
- Compression ratio: chars(trajectory) / chars(skill).
- Skill quality score: LLM rates the skill structure itself (0-10).
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
    """Skill quality evaluator with LLM-as-judge scoring."""

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

        Uses LLM-as-judge for continuous scoring instead of binary
        substring matching.
        """
        logger.info(
            f"Evaluating skill: {skill.name} (variant={skill.source_variant})"
        )

        # 1. LLM-as-judge: score skill-guided task completion (0-10)
        total_score = 0.0
        validation_details: list[dict[str, Any]] = []

        for task in validation_tasks:
            task_result = self._validate_with_skill(skill, task)
            validation_details.append(task_result)
            total_score += task_result.get("score", 0.0)

        # Normalise to 0-1 range
        success_rate = (
            total_score / (len(validation_tasks) * 10.0)
            if validation_tasks
            else 0.0
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

        # 3. Skill quality score (structure, specificity, reusability)
        quality_score = self._score_skill_quality(skill)

        eval_result = SkillEvalResult(
            skill_id=skill.skill_id,
            variant=skill.source_variant or TransformVariant.TRAJ_TO_SKILL,
            success_rate=round(success_rate, 4),
            compression_ratio=round(compression_ratio, 2),
            transfer_score=round(quality_score, 4),
            validation_details=validation_details,
        )

        logger.info(
            f"Skill evaluation complete: {skill.name}, "
            f"task_score={success_rate:.1%}, "
            f"quality={quality_score:.2f}, "
            f"compression={compression_ratio:.1f}x"
        )
        return eval_result

    def _validate_with_skill(
        self,
        skill: Skill,
        task: dict[str, str],
    ) -> dict[str, Any]:
        """
        Run a validation task with the skill injected, then use
        LLM-as-judge to score the response (0-10).
        """
        skill_prompt = self._format_skill_as_prompt(skill)

        # Step 1: Generate response using the skill
        gen_messages = [
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
            response = self.llm_client.chat(gen_messages)
        except Exception as exc:
            logger.error(f"Validation generation failed: {exc}")
            return {
                "task_id": task.get("task_id", ""),
                "success": False,
                "score": 0.0,
                "error": str(exc),
            }

        expected = task.get("expected", "")

        # Step 2: LLM-as-judge scores the response
        score = self._llm_judge_score(
            task_description=task.get("description", ""),
            expected_answer=expected,
            actual_response=response[:1000],
            skill_name=skill.name,
        )

        return {
            "task_id": task.get("task_id", ""),
            "success": score >= 7.0,
            "score": score,
            "response": response[:500],
            "expected": expected,
        }

    def _llm_judge_score(
        self,
        task_description: str,
        expected_answer: str,
        actual_response: str,
        skill_name: str,
    ) -> float:
        """
        Use LLM-as-judge to score a response on a 0-10 scale.

        Scoring criteria:
        - Correctness (does the response contain the right answer?)
        - Completeness (is the reasoning thorough?)
        - Relevance (does it address the actual question?)
        """
        judge_prompt = f"""You are a strict evaluation judge. Score the following response on a 0-10 scale.

Task: {task_description[:500]}

Expected answer: {expected_answer}

Actual response (generated using skill "{skill_name}"):
{actual_response}

Scoring criteria:
- 9-10: Correct answer with clear, complete reasoning
- 7-8: Correct answer but reasoning could be better
- 5-6: Partially correct or correct answer with flawed reasoning
- 3-4: Mostly incorrect but shows some relevant understanding
- 1-2: Incorrect with minimal relevance
- 0: Completely wrong or no answer

Return JSON: {{"score": <number 0-10>, "reason": "<one sentence>"}}"""

        messages = [
            {
                "role": "system",
                "content": "You are a strict, fair evaluation judge. Return only JSON.",
            },
            {"role": "user", "content": judge_prompt},
        ]

        try:
            result = self.llm_client.chat_json(messages, temperature=0.1)
            data = json.loads(result)
            score = float(data.get("score", 0))
            return max(0.0, min(10.0, score))
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error(f"Judge scoring failed: {exc}")
            # Fallback: simple substring check
            if expected_answer and expected_answer.lower().strip() in actual_response.lower():
                return 7.0
            return 3.0

    def _score_skill_quality(self, skill: Skill) -> float:
        """
        Use LLM to score the skill's structural quality (0-10).

        Evaluates 5 dimensions with a strict rubric that penalises
        common failure modes differently for each variant:
        - Specificity: penalises vague steps (common in memory_to_skill)
        - Reusability: penalises over-fitting (common in traj_to_skill)
        - Structure: penalises missing sections
        - Denoising: penalises inclusion of errors/retries (key differentiator!)
        - Completeness: penalises missing reasoning steps
        """
        skill_text = self._format_skill_as_prompt(skill)

        judge_prompt = f"""You are a strict skill quality evaluator. Rate this skill on 5 dimensions.

{skill_text}

STRICT SCORING RUBRIC (be harsh — most skills should score 5-7, not 8-10):

1. **Specificity** (0-10): Are steps concrete and actionable?
   - 9-10: Every step has specific actions with clear inputs/outputs
   - 6-8: Most steps are specific but some are vague
   - 3-5: Mix of specific and vague steps
   - 0-2: Mostly vague advice like "be careful" or "think about it"

2. **Reusability** (0-10): Could this skill work on DIFFERENT but similar tasks?
   - 9-10: Fully generic — no task-specific details, pure methodology
   - 6-8: Mostly generic with minor task-specific references
   - 3-5: Contains several task-specific details that limit reuse
   - 0-2: Completely tied to one specific task instance

3. **Structure** (0-10): Are all sections well-defined?
   - 9-10: All of preconditions, procedure (3+ steps), constraints, facts present
   - 6-8: Most sections present, procedure has 2+ steps
   - 3-5: Missing 1-2 sections or procedure is too short
   - 0-2: Missing most sections

4. **Denoising** (0-10): Is it free of noise, errors, dead-ends, retries?
   - 9-10: Clean — no trace of errors, retries, or irrelevant tangents
   - 6-8: Mostly clean with minor noise
   - 3-5: Contains some error-handling steps that are task-specific noise
   - 0-2: Full of error traces, retry logic, and dead-end reasoning

5. **Completeness** (0-10): Does it cover the full reasoning process?
   - 9-10: Complete end-to-end methodology with verification steps
   - 6-8: Covers main steps but missing verification or edge cases
   - 3-5: Covers only the core steps, missing important details
   - 0-2: Incomplete — missing critical reasoning steps

Return JSON: {{"specificity": <0-10>, "reusability": <0-10>, "structure": <0-10>, "denoising": <0-10>, "completeness": <0-10>, "overall": <0-10>, "reason": "<one sentence>"}}"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict skill quality evaluator. "
                    "Be harsh — most skills should score 5-7, not 8-10. "
                    "Return only JSON."
                ),
            },
            {"role": "user", "content": judge_prompt},
        ]

        try:
            result = self.llm_client.chat_json(messages, temperature=0.1)
            data = json.loads(result)
            # Use the average of 5 dimensions (not the LLM's "overall")
            dims = ["specificity", "reusability", "structure", "denoising", "completeness"]
            scores = [float(data.get(d, 5)) for d in dims]
            avg_score = sum(scores) / len(scores)
            return max(0.0, min(10.0, avg_score)) / 10.0  # Normalise to 0-1
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error(f"Skill quality scoring failed: {exc}")
            # Heuristic fallback based on structure completeness
            has_procedure = len(skill.procedure) >= 2
            has_constraints = len(skill.constraints) >= 1
            has_facts = len(skill.facts) >= 1
            has_preconditions = len(skill.preconditions) >= 1
            completeness = sum([has_procedure, has_constraints, has_facts, has_preconditions])
            return completeness / 4.0 * 0.6  # Max 0.6 for heuristic

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

        if skill.rules:
            parts.append("")
            parts.append("**Rules:**")
            for rule in skill.rules:
                parts.append(f"- {rule}")

        return "\n".join(parts)

    def compare_variants(
        self,
        results: dict[TransformVariant, list[SkillEvalResult]],
    ) -> dict[str, Any]:
        """Compare evaluation results across the three variants."""
        comparison: dict[str, Any] = {}
        for variant, eval_results in results.items():
            if not eval_results:
                continue
            n = len(eval_results)
            avg_success = sum(r.success_rate for r in eval_results) / n
            avg_compression = sum(r.compression_ratio for r in eval_results) / n
            avg_quality = sum(r.transfer_score for r in eval_results) / n
            comparison[variant.value] = {
                "num_skills": n,
                "avg_task_score": round(avg_success, 4),
                "avg_quality_score": round(avg_quality, 4),
                "avg_compression_ratio": round(avg_compression, 2),
            }

        logger.info(
            f"Variant comparison results: {json.dumps(comparison, indent=2)}"
        )
        return comparison