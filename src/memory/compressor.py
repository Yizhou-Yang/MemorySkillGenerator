"""
Memory compressor base class and Mem0 adapter.

Compresses raw trajectories into structured memory entries.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from src.models import MemoryEntry, MemoryStore, Trajectory
from src.utils.llm import LLMClient


class BaseMemoryCompressor(ABC):
    """Abstract base class for memory compressors."""

    @abstractmethod
    def compress(self, trajectory: Trajectory) -> MemoryStore:
        """Compress a trajectory into structured memory."""
        ...


class Mem0Compressor(BaseMemoryCompressor):
    """
    Mem0-based memory compressor.

    MVP phase: uses an LLM to simulate Mem0's memory extraction process.
    Can be replaced with the real Mem0 SDK later.
    """

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}

    def compress(self, trajectory: Trajectory) -> MemoryStore:
        """
        Compress a trajectory into structured memory.

        Args:
            trajectory: The raw interaction trajectory.

        Returns:
            A structured memory store.
        """
        logger.info(
            f"Compressing trajectory to memory: "
            f"trajectory_id={trajectory.trajectory_id}"
        )

        # Serialise the trajectory to text
        trajectory_text = self._trajectory_to_text(trajectory)

        # Use the LLM to extract structured memories
        prompt = self._build_extraction_prompt(trajectory_text)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory extraction expert. "
                    "Extract key structured memories from the interaction trajectory."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.chat_json(messages)

        # Parse the response into MemoryEntry objects
        entries = self._parse_memories(response, trajectory.trajectory_id)

        memory_store = MemoryStore(
            task_id=trajectory.task_id,
            framework="mem0",
            entries=entries,
            source_trajectory_id=trajectory.trajectory_id,
        )

        logger.info(
            f"Memory compression complete: {memory_store.num_entries} entries, "
            f"avg_specificity={memory_store.avg_specificity:.2f}"
        )
        return memory_store

    def _trajectory_to_text(self, trajectory: Trajectory) -> str:
        """Convert a trajectory to a plain-text representation."""
        lines = [
            f"Task: {trajectory.task_description}",
            f"Result: {'success' if trajectory.success else 'failure'}",
            "",
        ]
        for step in trajectory.steps:
            lines.append(f"[{step.step_type.value}] {step.content}")
        return "\n".join(lines)

    def _build_extraction_prompt(self, trajectory_text: str) -> str:
        """Build the memory extraction prompt."""
        return f"""Extract key memories from the following agent interaction trajectory.

Each memory should be specific and actionable knowledge, not vague generalisations.

Return JSON in this format:
{{
  "memories": [
    {{
      "content": "memory content",
      "category": "fact/rule/procedure/insight",
      "specificity_score": 0.0-1.0,
      "importance": 0.0-1.0
    }}
  ]
}}

Category guide:
- fact: domain facts (e.g. "Python's list.sort() sorts in-place")
- rule: decision rules (e.g. "On a 404 error, check the URL spelling first")
- procedure: operational procedures (e.g. "Run the test suite before deploying")
- insight: discoveries (e.g. "This API rate-limits above 100 concurrent requests")

Specificity scoring:
- 0.0-0.3: vague (e.g. "be careful", "pay attention to details")
- 0.4-0.6: somewhat specific but still fuzzy
- 0.7-1.0: highly specific and actionable (includes concrete conditions, steps, or values)

Interaction trajectory:
{trajectory_text}
"""

    def _parse_memories(
        self, response: str, trajectory_id: str
    ) -> list[MemoryEntry]:
        """Parse the LLM response into a list of MemoryEntry objects."""
        try:
            data = json.loads(response)
            raw_memories = data.get("memories", [])
            entries: list[MemoryEntry] = []
            for mem in raw_memories:
                entry = MemoryEntry(
                    content=mem.get("content", ""),
                    category=mem.get("category", "general"),
                    source_trajectory_id=trajectory_id,
                    specificity_score=float(mem.get("specificity_score", 0.5)),
                    importance=float(mem.get("importance", 0.5)),
                )
                entries.append(entry)
            return entries
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(f"Failed to parse memory response: {exc}")
            # Fallback: treat the entire response as a single memory
            return [
                MemoryEntry(
                    content=response,
                    category="general",
                    source_trajectory_id=trajectory_id,
                )
            ]


def create_compressor(
    framework: str, llm_client: LLMClient, config: dict[str, Any] | None = None
) -> BaseMemoryCompressor:
    """
    Factory: create a memory compressor for the given framework.

    Args:
        framework: Framework name (mem0 / amem / memorybank).
        llm_client: LLM client instance.
        config: Additional configuration.
    """
    compressors: dict[str, type[BaseMemoryCompressor]] = {
        "mem0": Mem0Compressor,
        # TODO: Add later
        # "amem": AMEMCompressor,
        # "memorybank": MemoryBankCompressor,
    }
    if framework not in compressors:
        raise ValueError(
            f"Unsupported memory framework: {framework}. "
            f"Available: {list(compressors.keys())}"
        )
    return compressors[framework](llm_client, config)
