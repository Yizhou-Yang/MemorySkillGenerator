"""
Variant 2: Memory -> Skill (compressed path).

First compresses the trajectory into structured memory, then induces a
skill from the memory.
Pros: denoised, structured.
Cons: may lose critical details (over-compression).
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.models import MemoryStore, Skill, Trajectory, TransformVariant
from src.skill_induction.base import BaseSkillInducer
from src.skill_induction.traj_to_skill import SKILL_INDUCTION_SYSTEM_PROMPT, SKILL_OUTPUT_FORMAT
from src.utils.llm import LLMClient


class MemoryToSkillInducer(BaseSkillInducer):
    """Variant 2: induce a skill from structured memory."""

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}

    def induce(
        self,
        trajectory: Trajectory,
        memory: MemoryStore | None = None,
    ) -> Skill:
        """
        Induce a skill from structured memory.

        Args:
            trajectory: Raw trajectory (used only for metadata in this variant).
            memory: Structured memory (required).

        Returns:
            The induced Skill.
        """
        if memory is None:
            raise ValueError("Variant 2 (Memory->Skill) requires structured memory")

        logger.info(
            f"[Variant 2: Memory->Skill] Starting induction: "
            f"task_id={trajectory.task_id}, memories={memory.num_entries}"
        )

        formatted_memory = self._format_memory(memory)
        prompt = self._build_prompt(trajectory, formatted_memory)

        messages = [
            {"role": "system", "content": SKILL_INDUCTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.chat_json(messages)
        skill = self._parse_skill(response, trajectory)

        logger.info(f"[Variant 2] Skill induction complete: {skill.name}")
        return skill

    def _format_memory(self, memory: MemoryStore) -> str:
        """Format memory entries as plain text."""
        lines: list[str] = []
        for idx, entry in enumerate(memory.entries):
            lines.append(
                f"[Memory {idx + 1}] [{entry.category}] "
                f"(specificity={entry.specificity_score:.1f}, "
                f"importance={entry.importance:.1f})\n"
                f"  {entry.content}"
            )
        return "\n".join(lines)

    def _build_prompt(self, trajectory: Trajectory, formatted_memory: str) -> str:
        """Build the skill induction prompt.

        Key design: this prompt gives the LLM ONLY compressed memory entries
        (no raw trajectory). The LLM must work with pre-structured,
        categorised knowledge. This produces clean, well-structured skills
        but may miss concrete operational details that were lost during
        memory compression.
        """
        return f"""Induce a reusable skill STRICTLY from the following structured memories.

These memories were extracted and compressed from an agent interaction trajectory.
You have NO access to the original trajectory — work only with what is provided.

Task description: {trajectory.task_description}
Task result: {"success" if trajectory.success else "failure"}

Structured memories:
{formatted_memory}

STRICT RULES:
1. Use ONLY the information present in the memories above.
2. Do NOT invent or speculate about details not in the memories.
3. Prioritise memories with high specificity (>= 0.7) and importance (>= 0.6).
4. Discard any memory that is vague or generic (specificity < 0.4).
5. The skill should be CONCISE — focus on the essential methodology.
6. Integrate related memories into coherent, reusable procedure steps.

The resulting skill should be a clean, generic methodology that can
guide an agent through SIMILAR (not identical) tasks.

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
                source_variant=TransformVariant.MEMORY_TO_SKILL,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(f"Failed to parse skill response: {exc}")
            return Skill(
                name="Parse Error Skill",
                description=response[:200],
                source_tasks=[trajectory.task_id],
                source_variant=TransformVariant.MEMORY_TO_SKILL,
            )
