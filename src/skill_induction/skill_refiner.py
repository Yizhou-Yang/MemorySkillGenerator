"""
Skill refiner — iterative skill improvement via validation feedback.

Implements the Skill Lifecycle (P2) from Mem2Evolve analysis:
- Skill Validation Loop: test skill on similar tasks, refine if EM=0
- Skill Accumulation: multiple trajectories refine the same skill
- Skill Retirement: deprecate skills with persistent low performance

Reference: docs/internal/mem2evolve_analysis.md §3.3 Backward Evolution
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.models import MemoryStore, Skill, Trajectory, TransformVariant
from src.skill_induction.base import BaseSkillInducer
from src.utils.llm import LLMClient


class SkillRefiner:
    """
    Iteratively refines skills based on validation feedback.

    Unlike one-shot skill induction, the refiner:
    1. Validates a skill against test tasks (using EM/F1)
    2. If validation fails, uses LLM to improve the skill
    3. Tracks version history and retires persistently failing skills
    """

    DEFAULT_MAX_ITERATIONS = 3
    DEFAULT_RETIREMENT_THRESHOLD = 3  # Retire after N consecutive failures

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.max_iterations: int = self.config.get(
            "max_iterations", self.DEFAULT_MAX_ITERATIONS
        )
        self.retirement_threshold: int = self.config.get(
            "retirement_threshold", self.DEFAULT_RETIREMENT_THRESHOLD
        )

    def refine(
        self,
        skill: Skill,
        validation_results: list[dict[str, Any]],
    ) -> Skill:
        """
        Refine a skill based on validation feedback.

        Args:
            skill: The skill to refine.
            validation_results: List of validation results, each containing
                'task_description', 'expected', 'response', 'em', 'f1'.

        Returns:
            A refined version of the skill (or the original if no improvement needed).
        """
        # Check if skill needs refinement
        failures = [r for r in validation_results if r.get("em", 0) == 0]
        if not failures:
            logger.info(f"[SkillRefiner] Skill '{skill.name}' passed all validations, no refinement needed")
            return skill

        failure_rate = len(failures) / len(validation_results) if validation_results else 0
        logger.info(
            f"[SkillRefiner] Skill '{skill.name}' failed {len(failures)}/{len(validation_results)} "
            f"validations (failure_rate={failure_rate:.1%}), attempting refinement"
        )

        if self.llm_client is None:
            logger.warning("[SkillRefiner] No LLM client, cannot refine")
            return skill

        # Build refinement prompt from failure cases
        refined_skill = self._refine_with_feedback(skill, failures)
        refined_skill.version = skill.version + 1
        refined_skill.metadata["refined_from"] = skill.skill_id
        refined_skill.metadata["failure_count"] = len(failures)

        logger.info(
            f"[SkillRefiner] Refined skill '{refined_skill.name}' "
            f"v{skill.version} -> v{refined_skill.version}"
        )
        return refined_skill

    def should_retire(self, skill: Skill, consecutive_failures: int) -> bool:
        """
        Determine if a skill should be retired (deprecated).

        A skill is retired if it has failed consecutively more than
        the retirement threshold.
        """
        return consecutive_failures >= self.retirement_threshold

    def accumulate(
        self,
        existing_skill: Skill,
        new_trajectory: Trajectory,
        new_memory: MemoryStore | None = None,
    ) -> Skill:
        """
        Accumulate new trajectory evidence into an existing skill.

        Instead of creating a new skill from scratch, this enriches
        the existing skill with insights from a new trajectory on
        a similar task.

        Args:
            existing_skill: The current skill to enrich.
            new_trajectory: A new trajectory on a similar task.
            new_memory: Optional memory from the new trajectory.

        Returns:
            An enriched version of the skill.
        """
        if self.llm_client is None:
            logger.warning("[SkillRefiner] No LLM client, cannot accumulate")
            return existing_skill

        # Build trajectory summary
        traj_summary = self._trajectory_to_summary(new_trajectory)

        prompt = f"""You have an existing skill and a new trajectory from a similar task.
Enrich the skill with any NEW insights from the trajectory that are not already captured.

Existing skill:
- Name: {existing_skill.name}
- Description: {existing_skill.description}
- Procedure: {json.dumps(existing_skill.procedure)}
- Constraints: {json.dumps(existing_skill.constraints)}
- Facts: {json.dumps(existing_skill.facts)}
- Rules: {json.dumps(existing_skill.rules)}

New trajectory summary:
{traj_summary}

Instructions:
1. Keep all existing procedure steps that are still valid
2. ADD new steps/constraints/facts only if the trajectory reveals something new
3. REMOVE steps that the trajectory shows are unnecessary or harmful
4. Do NOT duplicate existing information

Return JSON:
{{
  "name": "skill name (keep or improve)",
  "description": "updated description",
  "procedure": ["step 1", "step 2", ...],
  "constraints": ["constraint 1", ...],
  "facts": ["fact 1", ...],
  "rules": ["rule 1", ...]
}}"""

        messages = [
            {
                "role": "system",
                "content": "You are a skill refinement agent. Enrich existing skills with new evidence.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm_client.chat_json(messages)
            data = json.loads(response)
            enriched = Skill(
                skill_id=existing_skill.skill_id,
                name=data.get("name", existing_skill.name),
                description=data.get("description", existing_skill.description),
                procedure=data.get("procedure", existing_skill.procedure),
                constraints=data.get("constraints", existing_skill.constraints),
                facts=data.get("facts", existing_skill.facts),
                rules=data.get("rules", existing_skill.rules),
                source_tasks=existing_skill.source_tasks + [new_trajectory.task_id],
                source_variant=existing_skill.source_variant,
                version=existing_skill.version + 1,
                metadata={**existing_skill.metadata, "accumulated_from": new_trajectory.trajectory_id},
            )
            logger.info(
                f"[SkillRefiner] Accumulated into '{enriched.name}' "
                f"v{existing_skill.version} -> v{enriched.version}"
            )
            return enriched
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error(f"[SkillRefiner] Accumulation failed: {exc}")
            return existing_skill

    def _refine_with_feedback(
        self, skill: Skill, failures: list[dict[str, Any]]
    ) -> Skill:
        """Use LLM to refine a skill based on failure cases."""
        failure_descriptions = []
        for f in failures[:5]:  # Limit to 5 failure cases
            failure_descriptions.append(
                f"- Task: {f.get('task_description', 'N/A')[:200]}\n"
                f"  Expected: {f.get('expected', 'N/A')[:100]}\n"
                f"  Got: {f.get('response', 'N/A')[:100]}"
            )
        failures_text = "\n".join(failure_descriptions)

        prompt = f"""A skill failed on several validation tasks. Improve it based on the failure patterns.

Current skill:
- Name: {skill.name}
- Description: {skill.description}
- Procedure: {json.dumps(skill.procedure)}
- Constraints: {json.dumps(skill.constraints)}
- Facts: {json.dumps(skill.facts)}
- Rules: {json.dumps(skill.rules)}

Failure cases:
{failures_text}

Analyse the failure pattern and improve the skill:
1. Identify what the skill is missing or doing wrong
2. Add/modify procedure steps to address the failures
3. Add constraints to prevent the failure pattern
4. Add any missing facts or rules

Return JSON:
{{
  "name": "improved skill name",
  "description": "improved description",
  "procedure": ["step 1", "step 2", ...],
  "constraints": ["constraint 1", ...],
  "facts": ["fact 1", ...],
  "rules": ["rule 1", ...]
}}"""

        messages = [
            {
                "role": "system",
                "content": "You are a skill improvement agent. Fix skills based on failure analysis.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm_client.chat_json(messages)
            data = json.loads(response)
            return Skill(
                skill_id=skill.skill_id,
                name=data.get("name", skill.name),
                description=data.get("description", skill.description),
                preconditions=skill.preconditions,
                procedure=data.get("procedure", skill.procedure),
                constraints=data.get("constraints", skill.constraints),
                facts=data.get("facts", skill.facts),
                rules=data.get("rules", skill.rules),
                source_tasks=skill.source_tasks,
                source_variant=skill.source_variant,
                version=skill.version,
                metadata=skill.metadata,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error(f"[SkillRefiner] Refinement LLM call failed: {exc}")
            return skill

    @staticmethod
    def _trajectory_to_summary(trajectory: Trajectory) -> str:
        """Convert trajectory to a concise summary for accumulation."""
        lines = [
            f"Task: {trajectory.task_description}",
            f"Result: {'success' if trajectory.success else 'failure'}",
            f"Steps ({trajectory.num_steps}):",
        ]
        for step in trajectory.steps[:10]:  # Limit to first 10 steps
            lines.append(f"  [{step.step_type.value}] {step.content[:100]}")
        if trajectory.num_steps > 10:
            lines.append(f"  ... ({trajectory.num_steps - 10} more steps)")
        return "\n".join(lines)
