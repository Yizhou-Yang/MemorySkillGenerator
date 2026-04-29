"""
Variant 3: Memory + Evidence Trajectory -> Skill (hybrid path).

Uses memory as an index to trace back key evidence from the raw trajectory,
then induces a skill from the combined input.
Pros: balances compression and completeness.
Hypothesis: most likely the best variant.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.models import MemoryStore, Skill, Trajectory, TransformVariant
from src.skill_induction.base import BaseSkillInducer
from src.skill_induction.traj_to_skill import SKILL_INDUCTION_SYSTEM_PROMPT, SKILL_OUTPUT_FORMAT
from src.utils.llm import LLMClient


class HybridToSkillInducer(BaseSkillInducer):
    """Variant 3: Memory + Evidence Trajectory -> Skill (hybrid path)."""

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.evidence_top_k: int = self.config.get("evidence_retrieval_top_k", 5)
        self.evidence_max_tokens: int = self.config.get("evidence_max_tokens", 2048)

    def induce(
        self,
        trajectory: Trajectory,
        memory: MemoryStore | None = None,
    ) -> Skill:
        """
        Induce a skill from Memory + Evidence Trajectory.

        Pipeline:
        1. Use memory entries as an index to identify key knowledge points.
        2. Trace back to the raw trajectory to find supporting evidence.
        3. Feed memory + evidence into the LLM to induce a skill.

        Args:
            trajectory: Raw trajectory.
            memory: Structured memory (required).

        Returns:
            The induced Skill.
        """
        if memory is None:
            raise ValueError("Variant 3 (Hybrid) requires structured memory")

        logger.info(
            f"[Variant 3: Hybrid->Skill] Starting induction: "
            f"task_id={trajectory.task_id}, "
            f"memories={memory.num_entries}, steps={trajectory.num_steps}"
        )

        # Step 1: Retrieve evidence using memory as an index
        evidence_pairs = self._retrieve_evidence(trajectory, memory)

        # Step 2: Build hybrid input
        hybrid_text = self._format_hybrid_input(evidence_pairs)
        prompt = self._build_prompt(trajectory, hybrid_text)

        messages = [
            {"role": "system", "content": SKILL_INDUCTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.chat_json(messages)
        skill = self._parse_skill(response, trajectory)

        logger.info(f"[Variant 3] Skill induction complete: {skill.name}")
        return skill

    def _retrieve_evidence(
        self,
        trajectory: Trajectory,
        memory: MemoryStore,
    ) -> list[dict[str, str]]:
        """
        Retrieve key evidence from the trajectory using memory as an index.

        MVP phase: uses the LLM for evidence retrieval.
        Can be replaced with embedding-based similarity search later.

        Returns:
            A list of dicts, each containing ``{"memory": ..., "evidence": ...}``.
        """
        # Format trajectory steps
        formatted_steps: list[str] = []
        for step in trajectory.steps:
            formatted_steps.append(
                f"[Step {step.step_id}] [{step.step_type.value}] {step.content}"
            )
        trajectory_text = "\n".join(formatted_steps)

        # Format memory list
        memory_contents = [entry.content for entry in memory.entries]

        prompt = f"""Given the following structured memories and raw trajectory, \
find the most relevant evidence snippet in the trajectory for each memory.

Structured memories:
{json.dumps(memory_contents, ensure_ascii=False, indent=2)}

Raw trajectory:
{trajectory_text}

Return JSON with up to {self.evidence_top_k} relevant steps per memory:
{{
  "evidence_pairs": [
    {{
      "memory": "memory content",
      "evidence_steps": [step_id_list],
      "evidence_summary": "one-sentence evidence summary"
    }}
  ]
}}
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evidence retrieval expert. "
                    "Precisely match memories to evidence in the trajectory."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.chat_json(messages)

        try:
            data = json.loads(response)
            raw_pairs = data.get("evidence_pairs", [])
            # Build a step lookup map
            step_map = {step.step_id: step for step in trajectory.steps}
            result: list[dict[str, str]] = []
            for pair in raw_pairs:
                evidence_fragments: list[str] = []
                for step_id in pair.get("evidence_steps", []):
                    if step_id in step_map:
                        matched_step = step_map[step_id]
                        evidence_fragments.append(
                            f"[{matched_step.step_type.value}] {matched_step.content}"
                        )
                result.append({
                    "memory": pair.get("memory", ""),
                    "evidence": (
                        "\n".join(evidence_fragments)
                        if evidence_fragments
                        else pair.get("evidence_summary", "")
                    ),
                })
            return result
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(f"Failed to parse evidence response: {exc}")
            # Fallback: pair all memories with a failure notice
            return [
                {"memory": entry.content, "evidence": "(evidence retrieval failed)"}
                for entry in memory.entries
            ]

    def _format_hybrid_input(
        self,
        evidence_pairs: list[dict[str, str]],
    ) -> str:
        """Format the hybrid (memory + evidence) input as plain text."""
        lines: list[str] = []
        for idx, pair in enumerate(evidence_pairs):
            lines.append(f"--- Knowledge Point {idx + 1} ---")
            lines.append(f"Memory: {pair['memory']}")
            lines.append(f"Trajectory Evidence: {pair['evidence']}")
            lines.append("")
        return "\n".join(lines)

    def _build_prompt(self, trajectory: Trajectory, hybrid_text: str) -> str:
        """Build the skill induction prompt."""
        return f"""Induce a reusable skill from the following memory + trajectory evidence pairs.

Each knowledge point contains:
- Memory: compressed structured knowledge.
- Trajectory evidence: key snippets from the raw trajectory supporting that memory.

Task description: {trajectory.task_description}
Task result: {"success" if trajectory.success else "failure"}

Memory + Evidence:
{hybrid_text}

Notes:
- Memories provide the structured knowledge framework.
- Trajectory evidence provides concrete operational details and context.
- Synthesise both to produce a skill that is both structured and detailed.
- Pay special attention to error->fix patterns — these are the most valuable skill sources.

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
                source_variant=TransformVariant.HYBRID_TO_SKILL,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(f"Failed to parse skill response: {exc}")
            return Skill(
                name="Parse Error Skill",
                description=response[:200],
                source_tasks=[trajectory.task_id],
                source_variant=TransformVariant.HYBRID_TO_SKILL,
            )
