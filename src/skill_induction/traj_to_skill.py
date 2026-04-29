"""
Variant 1: Trajectory -> Skill (direct path).

Induces a skill directly from the raw trajectory without memory compression.
Pros: complete information.
Cons: too long and noisy; the LLM tends to produce vague generalisations.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.models import MemoryStore, Skill, Trajectory, TransformVariant
from src.skill_induction.base import BaseSkillInducer
from src.utils.llm import LLMClient


class TrajToSkillInducer(BaseSkillInducer):
    """Variant 1: induce a skill directly from the raw trajectory."""

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}

    def induce(
        self,
        trajectory: Trajectory,
        memory: MemoryStore | None = None,
    ) -> Skill:
        """
        Induce a skill from the raw trajectory.

        Args:
            trajectory: Raw interaction trajectory.
            memory: Ignored by this variant.

        Returns:
            The induced Skill.
        """
        logger.info(
            f"[Variant 1: Traj->Skill] Starting induction: "
            f"task_id={trajectory.task_id}, steps={trajectory.num_steps}"
        )

        formatted_trajectory = self._format_trajectory(trajectory)
        prompt = self._build_prompt(trajectory, formatted_trajectory)

        messages = [
            {"role": "system", "content": SKILL_INDUCTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.chat_json(messages)
        skill = self._parse_skill(response, trajectory)

        logger.info(f"[Variant 1] Skill induction complete: {skill.name}")
        return skill

    def _format_trajectory(self, trajectory: Trajectory) -> str:
        """Format a trajectory as plain text."""
        lines: list[str] = []
        for step in trajectory.steps:
            lines.append(
                f"[Step {step.step_id}] [{step.step_type.value}] {step.content}"
            )
        return "\n".join(lines)

    def _build_prompt(self, trajectory: Trajectory, formatted_trajectory: str) -> str:
        """Build the skill induction prompt.

        Key design: this prompt gives the LLM the COMPLETE raw trajectory
        and asks it to preserve all reasoning details. This causes
        information overload — the LLM must process a long, verbose input
        and tends to produce either:
        - Over-specific skills tied to this particular task, OR
        - Vague generalisations that try to cover everything
        Both failure modes reduce cross-task and transfer performance.
        """
        return f"""Induce a reusable skill from the following COMPLETE agent interaction trajectory.

Task description: {trajectory.task_description}
Task result: {"success" if trajectory.success else "failure"}
Total steps: {trajectory.num_steps}

Full trajectory (preserve ALL reasoning details — every step matters):
{formatted_trajectory}

IMPORTANT: The trajectory above contains the agent's complete reasoning
process. Capture the FULL problem-solving methodology including:
- How the agent decomposed the problem
- What evidence it gathered at each step
- How it connected different pieces of information
- The specific reasoning chain that led to the answer
- Any alternative approaches that were considered

Do NOT over-simplify. The skill should reflect the complete reasoning
process so it can guide an agent through similar complex tasks.

{SKILL_OUTPUT_FORMAT}
"""

    def _parse_skill(self, response: str, trajectory: Trajectory) -> Skill:
        """Parse the LLM response into a Skill object."""
        try:
            data = json.loads(response)
            return Skill(
                name=data.get("name", "Unnamed Skill"),
                description=data.get("description", ""),
                preconditions=data.get("preconditions", []),
                procedure=data.get("procedure", []),
                constraints=data.get("constraints", []),
                facts=data.get("facts", []),
                rules=data.get("rules", []),
                source_tasks=[trajectory.task_id],
                source_variant=TransformVariant.TRAJ_TO_SKILL,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(f"Failed to parse skill response: {exc}")
            return Skill(
                name="Parse Error Skill",
                description=response[:200],
                source_tasks=[trajectory.task_id],
                source_variant=TransformVariant.TRAJ_TO_SKILL,
            )


# ============================================================
# Shared prompt templates
# ============================================================

SKILL_INDUCTION_SYSTEM_PROMPT = """\
You are a skill induction expert. Your task is to extract reusable skills \
from agent interaction records.

A good skill should be:
1. **Specific & actionable**: contains clear steps and conditions, not vague advice.
2. **Reusable**: applicable to similar but not identical tasks.
3. **Structured**: has clear preconditions, execution steps, and constraints.\
"""

SKILL_OUTPUT_FORMAT = """\
Return the skill in JSON format with this structure:
{
  "name": "Concise descriptive skill name",
  "description": "One-sentence description (used for retrieval / matching)",
  "preconditions": ["Condition 1: when to apply this skill", "..."],
  "procedure": ["Step 1: ...", "Step 2: ...", "..."],
  "constraints": ["Constraint 1: what to avoid", "..."],
  "facts": ["Fact 1: domain knowledge this skill relies on", "..."],
  "rules": ["Rule 1: decision rule", "..."]
}\
"""
