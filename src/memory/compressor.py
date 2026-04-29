"""
Memory compressor implementations.

Three memory compression strategies, each producing structured memory
from raw trajectories via different LLM prompting approaches:

- Mem0:       flat key-value extraction (simple, production-style)
- A-MEM:      agentic self-organising memory with linking and reflection
- MemoryBank: hierarchical memory with importance-based tiering
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

    # -- shared helpers ---------------------------------------------------

    @staticmethod
    def _trajectory_to_text(trajectory: Trajectory) -> str:
        """Convert a trajectory to a plain-text representation."""
        lines = [
            f"Task: {trajectory.task_description}",
            f"Result: {'success' if trajectory.success else 'failure'}",
            "",
        ]
        for step in trajectory.steps:
            lines.append(f"[{step.step_type.value}] {step.content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_memory_json(
        response: str, trajectory_id: str, key: str = "memories"
    ) -> list[MemoryEntry]:
        """Parse an LLM JSON response into a list of MemoryEntry objects."""
        try:
            data = json.loads(response)
            raw_items = data.get(key, [])
            entries: list[MemoryEntry] = []
            for item in raw_items:
                entry = MemoryEntry(
                    content=item.get("content", ""),
                    category=item.get("category", "general"),
                    source_trajectory_id=trajectory_id,
                    specificity_score=float(item.get("specificity_score", 0.5)),
                    importance=float(item.get("importance", 0.5)),
                    metadata={
                        k: v
                        for k, v in item.items()
                        if k not in {"content", "category", "specificity_score", "importance"}
                    },
                )
                entries.append(entry)
            return entries
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(f"Failed to parse memory response: {exc}")
            return [
                MemoryEntry(
                    content=response,
                    category="general",
                    source_trajectory_id=trajectory_id,
                )
            ]


# =====================================================================
# Mem0 Compressor — flat key-value extraction
# =====================================================================


class Mem0Compressor(BaseMemoryCompressor):
    """
    Mem0-style memory compressor.

    Extracts flat, independent memory entries from the trajectory.
    Each entry is a standalone piece of knowledge with category and scores.
    Mirrors the production Mem0 SDK's extraction behaviour.
    """

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}

    def compress(self, trajectory: Trajectory) -> MemoryStore:
        logger.info(
            f"[Mem0] Compressing trajectory: "
            f"trajectory_id={trajectory.trajectory_id}"
        )

        trajectory_text = self._trajectory_to_text(trajectory)
        prompt = self._build_prompt(trajectory_text)
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
        entries = self._parse_memory_json(response, trajectory.trajectory_id)

        store = MemoryStore(
            task_id=trajectory.task_id,
            framework="mem0",
            entries=entries,
            source_trajectory_id=trajectory.trajectory_id,
        )
        logger.info(
            f"[Mem0] Done: {store.num_entries} entries, "
            f"avg_specificity={store.avg_specificity:.2f}"
        )
        return store

    def _build_prompt(self, trajectory_text: str) -> str:
        return f"""Extract key memories from the following agent interaction trajectory.

Each memory should be specific and actionable knowledge, not vague generalisations.

Return JSON:
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
- 0.0-0.3: vague (e.g. "be careful")
- 0.4-0.6: somewhat specific
- 0.7-1.0: highly specific and actionable

Interaction trajectory:
{trajectory_text}
"""


# =====================================================================
# A-MEM Compressor — agentic memory with linking and reflection
# =====================================================================


class AMEMCompressor(BaseMemoryCompressor):
    """
    A-MEM (Agentic Memory) style compressor.

    Inspired by the A-MEM paper: the agent autonomously decides how to
    organise memories.  This implementation uses a two-pass approach:

    Pass 1 — Raw extraction:  extract atomic memory entries.
    Pass 2 — Reflection & linking:  the LLM reviews the raw entries,
             merges related ones, adds cross-references, and generates
             higher-level "reflection" memories that summarise patterns.
    """

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}

    def compress(self, trajectory: Trajectory) -> MemoryStore:
        logger.info(
            f"[A-MEM] Compressing trajectory: "
            f"trajectory_id={trajectory.trajectory_id}"
        )

        trajectory_text = self._trajectory_to_text(trajectory)

        # Pass 1: raw extraction
        raw_entries = self._extract_raw(trajectory_text, trajectory.trajectory_id)
        logger.info(f"[A-MEM] Pass 1 — extracted {len(raw_entries)} raw entries")

        # Pass 2: reflection & linking
        refined_entries = self._reflect_and_link(raw_entries, trajectory.trajectory_id)
        logger.info(
            f"[A-MEM] Pass 2 — refined to {len(refined_entries)} entries "
            f"(with reflections)"
        )

        store = MemoryStore(
            task_id=trajectory.task_id,
            framework="amem",
            entries=refined_entries,
            source_trajectory_id=trajectory.trajectory_id,
        )
        logger.info(
            f"[A-MEM] Done: {store.num_entries} entries, "
            f"avg_specificity={store.avg_specificity:.2f}"
        )
        return store

    def _extract_raw(
        self, trajectory_text: str, trajectory_id: str
    ) -> list[MemoryEntry]:
        """Pass 1: extract atomic memory entries."""
        prompt = f"""Extract atomic memory entries from this trajectory.
Each entry should capture ONE specific piece of knowledge.

Return JSON:
{{
  "memories": [
    {{
      "content": "one atomic piece of knowledge",
      "category": "fact/rule/procedure/insight",
      "specificity_score": 0.0-1.0,
      "importance": 0.0-1.0
    }}
  ]
}}

Trajectory:
{trajectory_text}
"""
        messages = [
            {
                "role": "system",
                "content": "You are an atomic memory extraction agent.",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.chat_json(messages)
        return self._parse_memory_json(response, trajectory_id)

    def _reflect_and_link(
        self, raw_entries: list[MemoryEntry], trajectory_id: str
    ) -> list[MemoryEntry]:
        """Pass 2: reflect on raw entries, merge related ones, add reflections."""
        raw_text_items: list[str] = []
        for idx, entry in enumerate(raw_entries):
            raw_text_items.append(
                f"[{idx}] [{entry.category}] {entry.content}"
            )
        raw_text = "\n".join(raw_text_items)

        prompt = f"""You are an agentic memory organiser.  Below are raw memory entries
extracted from an agent trajectory.  Your job:

1. **Merge** entries that describe the same knowledge into one.
2. **Link** related entries by noting connections in a "related_to" field.
3. **Reflect**: generate 1-3 higher-level "reflection" memories that
   summarise patterns or meta-strategies observed across multiple entries.

Return JSON:
{{
  "memories": [
    {{
      "content": "refined or reflected memory",
      "category": "fact/rule/procedure/insight/reflection",
      "specificity_score": 0.0-1.0,
      "importance": 0.0-1.0,
      "related_to": ["indices or short descriptions of related memories"]
    }}
  ]
}}

Raw entries:
{raw_text}
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agentic memory organiser that merges, "
                    "links, and reflects on raw memory entries."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.chat_json(messages)
        return self._parse_memory_json(response, trajectory_id)


# =====================================================================
# MemoryBank Compressor — hierarchical with importance-based tiering
# =====================================================================


class MemoryBankCompressor(BaseMemoryCompressor):
    """
    MemoryBank-style compressor.

    Inspired by the MemoryBank paper: organises memories into three tiers
    based on importance, and applies an Ebbinghaus-inspired forgetting
    mechanism to discard low-value memories.

    Tiers:
    - core:      importance >= 0.7  (long-term, never forgotten)
    - working:   0.4 <= importance < 0.7  (medium-term)
    - ephemeral: importance < 0.4  (short-term, subject to forgetting)
    """

    CORE_THRESHOLD = 0.7
    WORKING_THRESHOLD = 0.4

    def __init__(self, llm_client: LLMClient, config: dict[str, Any] | None = None) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.forget_ephemeral: bool = self.config.get("forget_ephemeral", True)

    def compress(self, trajectory: Trajectory) -> MemoryStore:
        logger.info(
            f"[MemoryBank] Compressing trajectory: "
            f"trajectory_id={trajectory.trajectory_id}"
        )

        trajectory_text = self._trajectory_to_text(trajectory)

        # Step 1: extract with importance-aware prompt
        all_entries = self._extract_tiered(trajectory_text, trajectory.trajectory_id)
        logger.info(f"[MemoryBank] Extracted {len(all_entries)} entries before tiering")

        # Step 2: tier and optionally forget
        core, working, ephemeral = self._tier_entries(all_entries)
        logger.info(
            f"[MemoryBank] Tiered: core={len(core)}, "
            f"working={len(working)}, ephemeral={len(ephemeral)}"
        )

        # Apply forgetting: drop ephemeral entries if configured
        if self.forget_ephemeral:
            kept_entries = core + working
            logger.info(
                f"[MemoryBank] Forgetting {len(ephemeral)} ephemeral entries"
            )
        else:
            kept_entries = core + working + ephemeral

        # Tag each entry's tier in metadata
        for entry in core:
            entry.metadata["tier"] = "core"
        for entry in working:
            entry.metadata["tier"] = "working"
        if not self.forget_ephemeral:
            for entry in ephemeral:
                entry.metadata["tier"] = "ephemeral"

        store = MemoryStore(
            task_id=trajectory.task_id,
            framework="memorybank",
            entries=kept_entries,
            source_trajectory_id=trajectory.trajectory_id,
        )
        logger.info(
            f"[MemoryBank] Done: {store.num_entries} entries, "
            f"avg_specificity={store.avg_specificity:.2f}"
        )
        return store

    def _extract_tiered(
        self, trajectory_text: str, trajectory_id: str
    ) -> list[MemoryEntry]:
        """Extract memories with explicit importance scoring for tiering."""
        prompt = f"""Extract memories from this trajectory.  For each memory,
carefully assess its **importance** for future task reuse:

- importance >= 0.7: **core** knowledge (critical facts, key procedures)
- 0.4 <= importance < 0.7: **working** knowledge (useful but not critical)
- importance < 0.4: **ephemeral** knowledge (minor details, may be forgotten)

Return JSON:
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

Trajectory:
{trajectory_text}
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a hierarchical memory manager. "
                    "Extract memories and carefully score their importance "
                    "for long-term retention."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.chat_json(messages)
        return self._parse_memory_json(response, trajectory_id)

    def _tier_entries(
        self, entries: list[MemoryEntry]
    ) -> tuple[list[MemoryEntry], list[MemoryEntry], list[MemoryEntry]]:
        """Split entries into core / working / ephemeral tiers."""
        core: list[MemoryEntry] = []
        working: list[MemoryEntry] = []
        ephemeral: list[MemoryEntry] = []

        for entry in entries:
            if entry.importance >= self.CORE_THRESHOLD:
                core.append(entry)
            elif entry.importance >= self.WORKING_THRESHOLD:
                working.append(entry)
            else:
                ephemeral.append(entry)

        return core, working, ephemeral


# =====================================================================
# Factory
# =====================================================================


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
    compressor_classes: dict[str, type[BaseMemoryCompressor]] = {
        "mem0": Mem0Compressor,
        "amem": AMEMCompressor,
        "memorybank": MemoryBankCompressor,
    }
    if framework not in compressor_classes:
        raise ValueError(
            f"Unsupported memory framework: {framework}. "
            f"Available: {list(compressor_classes.keys())}"
        )
    return compressor_classes[framework](llm_client, config)