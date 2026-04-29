"""
Variant 3: Memory + Evidence Trajectory -> Skill (hybrid path).

Evidence-as-Filter approach (v6):
Instead of injecting trajectory details INTO the skill, we use the
trajectory as a FILTER to validate and rank memory entries. Only
memories that are strongly supported by trajectory evidence survive.

This gives hybrid the best of both worlds:
- Memory-level abstraction (clean, generic procedures)
- Trajectory-validated relevance (only the most important memories)
- Confidence-weighted prioritisation (high-evidence memories first)

The key insight: the trajectory's role is to SELECT which memories
matter, not to ADD concrete details to the skill.
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
    """Variant 3: Memory + Evidence-Filtered Trajectory -> Skill."""

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.evidence_top_k: int = self.config.get("evidence_retrieval_top_k", 5)

    def induce(
        self,
        trajectory: Trajectory,
        memory: MemoryStore | None = None,
    ) -> Skill:
        """
        Induce a skill from evidence-filtered memory.

        Pipeline:
        1. Use trajectory to VALIDATE and RANK memory entries.
        2. Keep only high-confidence, evidence-supported memories.
        3. Feed the filtered memories (WITHOUT raw trajectory) to the LLM
           for skill induction — preserving memory-level abstraction.

        Args:
            trajectory: Raw trajectory (used for validation only).
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

        # Step 1: Validate and rank memories using trajectory as evidence
        validated_memories = self._validate_and_rank_memories(trajectory, memory)
        logger.info(
            f"[Variant 3] Evidence filtering: "
            f"{memory.num_entries} → {len(validated_memories)} memories"
        )

        # Step 2: Build skill from filtered memories ONLY (no trajectory details)
        filtered_memory_text = self._format_validated_memories(validated_memories)
        prompt = self._build_prompt(trajectory, filtered_memory_text)

        messages = [
            {"role": "system", "content": SKILL_INDUCTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.chat_json(messages)
        skill = self._parse_skill(response, trajectory)

        logger.info(f"[Variant 3] Skill induction complete: {skill.name}")
        return skill

    def _validate_and_rank_memories(
        self,
        trajectory: Trajectory,
        memory: MemoryStore,
    ) -> list[dict[str, Any]]:
        """
        Use trajectory as evidence to validate and rank memory entries.

        This is the KEY differentiator of hybrid (v6 Evidence-as-Filter):
        - The trajectory is used ONLY to assess which memories are real
          and important (not to extract additional details).
        - Memories with strong trajectory support get boosted.
        - Memories with no evidence or contradictions get dropped.
        - The output is a ranked list of MEMORY CONTENT ONLY — no
          trajectory fragments leak into the skill.

        Returns:
            A ranked list of validated memory dicts with confidence scores.
        """
        # Format trajectory steps for the validation LLM call
        formatted_steps: list[str] = []
        for step in trajectory.steps:
            formatted_steps.append(
                f"[Step {step.step_id}] [{step.step_type.value}] {step.content}"
            )
        trajectory_text = "\n".join(formatted_steps)

        # Format memory list
        memory_items: list[str] = []
        for idx, entry in enumerate(memory.entries):
            memory_items.append(
                f"[M{idx}] (category={entry.category}, "
                f"specificity={entry.specificity_score:.1f}, "
                f"importance={entry.importance:.1f}) {entry.content}"
            )
        memory_text = "\n".join(memory_items)

        prompt = f"""You are a memory validation expert. Your job is to assess which
memories are genuinely supported by the trajectory evidence.

TASK: For each memory, determine:
1. Is it SUPPORTED by concrete evidence in the trajectory?
2. How IMPORTANT is it for solving similar tasks (not just this one)?
3. How GENERALIZABLE is it (does it capture a reusable pattern, or
   is it specific to this one task instance)?

Memories to validate:
{memory_text}

Trajectory evidence:
{trajectory_text}

Return JSON:
{{
  "validated_memories": [
    {{
      "memory_index": <int>,
      "content": "the original memory content (DO NOT modify it)",
      "category": "original category",
      "evidence_strength": "strong|moderate|weak|none",
      "generalizability": "high|medium|low",
      "importance_for_similar_tasks": 0.0-1.0,
      "reasoning": "brief explanation of why this memory matters or not"
    }}
  ]
}}

CRITICAL RULES:
- DO NOT rewrite or enrich the memory content. Copy it exactly.
- DO NOT add trajectory-specific details to the memory.
- Focus on FILTERING: which memories capture reusable methodology
  vs. which are task-specific trivia?
- A memory about "how to approach the problem" is more generalizable
  than a memory about "the answer is X".
- Rank by generalizability first, then by evidence strength.
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory validation expert. "
                    "Assess which memories are evidence-supported and "
                    "generalizable. Do NOT modify memory content."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.chat_json(messages)

        try:
            data = json.loads(response)
            raw_validated = data.get("validated_memories", [])

            # Filter: keep only memories with at least moderate evidence
            # and at least medium generalizability
            filtered: list[dict[str, Any]] = []
            for item in raw_validated:
                evidence = item.get("evidence_strength", "none")
                generalizability = item.get("generalizability", "low")
                importance = float(item.get("importance_for_similar_tasks", 0.0))

                # Filter criteria: must have evidence AND be generalizable
                if evidence in ("strong", "moderate") and generalizability in ("high", "medium"):
                    filtered.append({
                        "content": item.get("content", ""),
                        "category": item.get("category", "general"),
                        "evidence_strength": evidence,
                        "generalizability": generalizability,
                        "importance": importance,
                    })

            # Sort: high generalizability + strong evidence first
            gen_order = {"high": 2, "medium": 1, "low": 0}
            ev_order = {"strong": 2, "moderate": 1, "weak": 0, "none": -1}
            filtered.sort(
                key=lambda x: (
                    gen_order.get(x["generalizability"], 0),
                    ev_order.get(x["evidence_strength"], 0),
                    x["importance"],
                ),
                reverse=True,
            )

            # Keep top-K
            return filtered[: self.evidence_top_k]

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(f"Failed to parse validation response: {exc}")
            # Fallback: use all memories sorted by importance
            return [
                {
                    "content": entry.content,
                    "category": entry.category,
                    "evidence_strength": "unknown",
                    "generalizability": "medium",
                    "importance": entry.importance,
                }
                for entry in sorted(
                    memory.entries, key=lambda e: e.importance, reverse=True
                )
            ]

    def _format_validated_memories(
        self,
        validated_memories: list[dict[str, Any]],
    ) -> str:
        """Format validated memories as plain text for skill induction.

        CRITICAL: This output contains ONLY memory content + validation
        metadata. NO trajectory fragments are included.
        """
        lines: list[str] = []
        for idx, mem in enumerate(validated_memories):
            gen = mem.get("generalizability", "medium")
            ev = mem.get("evidence_strength", "moderate")
            imp = mem.get("importance", 0.5)
            lines.append(
                f"[Memory {idx + 1}] [{mem['category']}] "
                f"(evidence={ev}, generalizability={gen}, "
                f"importance={imp:.1f})"
            )
            lines.append(f"  {mem['content']}")
            lines.append("")
        return "\n".join(lines)

    def _build_prompt(self, trajectory: Trajectory, filtered_memory_text: str) -> str:
        """Build the skill induction prompt.

        Key design (v6 Evidence-as-Filter):
        - The LLM receives ONLY validated, ranked memories.
        - NO raw trajectory details are included.
        - The memories have been pre-filtered by trajectory evidence,
          so only the most relevant and generalizable ones survive.
        - This produces skills at memory-level abstraction, but with
          BETTER memory selection than memory→skill (which uses all
          memories indiscriminately).
        """
        return f"""Induce a reusable skill from the following evidence-validated memories.

These memories have been extracted from an agent interaction and then
VALIDATED against the actual execution trajectory. Only memories with
strong evidence support and high generalizability have been retained.
They are ranked by relevance (most important first).

Task description: {trajectory.task_description}
Task result: {"success" if trajectory.success else "failure"}

Evidence-Validated Memories (ranked by generalizability):
{filtered_memory_text}

INSTRUCTIONS:
1. These memories have already been quality-filtered — trust them.
2. Focus on the METHODOLOGY captured in these memories, not task-specific details.
3. The skill procedure should describe a GENERAL approach that works
   for similar tasks, not a recipe for this specific task.
4. Keep the procedure CONCISE (4-6 steps). Each step should describe
   WHAT to do and WHY, not low-level implementation details.
5. Constraints and rules should capture DECISION CRITERIA that help
   an agent choose between alternatives.
6. Facts should capture DOMAIN KNOWLEDGE that is reusable across tasks,
   not task-specific data points.
7. The skill must be abstract enough to transfer to unseen tasks of
   the same type, while being specific enough to provide real guidance.

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
