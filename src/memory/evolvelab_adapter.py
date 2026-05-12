"""
EvolveLab Adapter — bridges SkillForge memory system with EvolveLab's
BaseMemoryProvider interface.

This adapter enables:
1. SkillForge compressors to be used as EvolveLab providers (outbound)
2. EvolveLab providers to be used within SkillForge pipelines (inbound)
3. Unified benchmark evaluation across both systems

Reference: MemEvolve paper (ICML'26) — EvolveLab unified memory interface
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger

from src.models import MemoryEntry, MemoryStore, Trajectory, TrajectoryStep, StepType
from src.memory.compressor import BaseMemoryCompressor
from src.memory.consolidation import MemoryConsolidator
from src.memory.evolvelab.base_memory import BaseMemoryProvider
from src.memory.evolvelab.memory_types import (
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    MemoryItem,
    MemoryItemType,
    TrajectoryData,
)


# ============================================================
# Outbound Adapter: SkillForge → EvolveLab
# ============================================================


class SkillForgeAsEvolveLabProvider(BaseMemoryProvider):
    """
    Wraps a SkillForge BaseMemoryCompressor + MemoryConsolidator
    as an EvolveLab BaseMemoryProvider.

    This allows SkillForge's compression pipeline to participate
    in EvolveLab's unified evaluation framework.

    Mapping:
    - take_in_memory() → compressor.compress() + consolidator.consolidate()
    - provide_memory() → similarity-based retrieval from stored memories
    - initialize() → no-op (SkillForge compressors are stateless)
    """

    def __init__(
        self,
        compressor: BaseMemoryCompressor,
        consolidator: MemoryConsolidator | None = None,
        memory_type: MemoryType = MemoryType.AGENT_KB,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(memory_type=memory_type, config=config)
        self.compressor = compressor
        self.consolidator = consolidator
        self._memory_stores: list[MemoryStore] = []
        self._all_entries: list[MemoryEntry] = []

    def initialize(self) -> bool:
        """Initialize — SkillForge compressors are stateless, always succeeds."""
        logger.info("[Adapter] SkillForgeAsEvolveLabProvider initialized")
        return True

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        """
        Ingest trajectory data via SkillForge compression pipeline.

        Converts EvolveLab TrajectoryData → SkillForge Trajectory,
        then runs compress() + optional consolidate().
        """
        try:
            # Convert TrajectoryData → Trajectory
            trajectory = self._convert_trajectory(trajectory_data)

            # Compress
            store = self.compressor.compress(trajectory)

            # Optionally consolidate
            if self.consolidator and self.consolidator.should_consolidate(store):
                store = self.consolidator.consolidate(store)

            self._memory_stores.append(store)
            self._all_entries.extend(store.entries)

            summary = "; ".join(e.content[:80] for e in store.entries[:3])
            logger.info(
                f"[Adapter] Ingested {store.num_entries} memories "
                f"from trajectory"
            )
            return (True, summary)

        except Exception as exc:
            logger.error(f"[Adapter] take_in_memory failed: {exc}")
            return (False, f"Failed: {exc}")

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        """
        Retrieve relevant memories via text similarity matching.

        Uses Jaccard similarity as a lightweight retrieval mechanism.
        """
        if not self._all_entries:
            return MemoryResponse(
                memories=[],
                memory_type=self.memory_type,
                total_count=0,
                request_id=str(uuid.uuid4()),
            )

        # Compute similarity scores
        query_tokens = set(request.query.lower().split())
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._all_entries:
            entry_tokens = set(entry.content.lower().split())
            if not query_tokens or not entry_tokens:
                scored.append((0.0, entry))
                continue
            intersection = query_tokens & entry_tokens
            union = query_tokens | entry_tokens
            sim = len(intersection) / len(union) if union else 0.0
            scored.append((sim, entry))

        # Sort by similarity descending, take top-k
        top_k = self.config.get("top_k", 3)
        scored.sort(key=lambda x: x[0], reverse=True)
        top_entries = scored[:top_k]

        memories = [
            MemoryItem(
                id=entry.memory_id,
                content=entry.content,
                metadata={
                    "category": entry.category,
                    "importance": entry.importance,
                    "specificity_score": entry.specificity_score,
                },
                score=score,
                type=MemoryItemType.TEXT,
            )
            for score, entry in top_entries
        ]

        return MemoryResponse(
            memories=memories,
            memory_type=self.memory_type,
            total_count=len(memories),
            request_id=str(uuid.uuid4()),
        )

    @staticmethod
    def _convert_trajectory(data: TrajectoryData) -> Trajectory:
        """Convert EvolveLab TrajectoryData to SkillForge Trajectory."""
        steps: list[TrajectoryStep] = []
        for i, step_dict in enumerate(data.trajectory):
            step_type_str = step_dict.get("type", "action")
            try:
                step_type = StepType(step_type_str)
            except ValueError:
                step_type = StepType.ACTION

            steps.append(
                TrajectoryStep(
                    step_id=i,
                    step_type=step_type,
                    content=step_dict.get("content", str(step_dict)),
                )
            )

        return Trajectory(
            task_id=data.metadata.get("task_id", "unknown") if data.metadata else "unknown",
            task_description=data.query,
            steps=steps,
            success=bool(data.result),
            final_answer=str(data.result) if data.result else None,
        )

    @property
    def num_memories(self) -> int:
        """Total number of stored memory entries."""
        return len(self._all_entries)

    def clear(self) -> None:
        """Clear all stored memories."""
        self._memory_stores.clear()
        self._all_entries.clear()


# ============================================================
# Inbound Adapter: EvolveLab → SkillForge
# ============================================================


class EvolveLabAsSkillForgeCompressor(BaseMemoryCompressor):
    """
    Wraps an EvolveLab BaseMemoryProvider as a SkillForge
    BaseMemoryCompressor.

    This allows any of EvolveLab's 13 memory providers (Voyager,
    ExpeL, SkillWeaver, etc.) to be used in SkillForge's pipeline.

    Mapping:
    - compress() → provider.take_in_memory() + provider.provide_memory()
    """

    def __init__(
        self,
        provider: BaseMemoryProvider,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or {}

    def compress(self, trajectory: Trajectory) -> MemoryStore:
        """
        Compress a trajectory using the EvolveLab provider.

        Converts SkillForge Trajectory → EvolveLab TrajectoryData,
        calls take_in_memory(), then retrieves stored memories.
        """
        # Convert Trajectory → TrajectoryData
        traj_data = self._convert_to_evolvelab(trajectory)

        # Ingest
        success, description = self.provider.take_in_memory(traj_data)

        if not success:
            logger.warning(
                f"[Adapter] EvolveLab provider ingestion failed: {description}"
            )
            return MemoryStore(
                task_id=trajectory.task_id,
                framework=f"evolvelab_{self.provider.get_memory_type().value}",
                entries=[],
                source_trajectory_id=trajectory.trajectory_id,
            )

        # Retrieve memories for this task
        request = MemoryRequest(
            query=trajectory.task_description,
            context="",
            status=MemoryStatus.BEGIN,
        )
        response = self.provider.provide_memory(request)

        # Convert EvolveLab MemoryItems → SkillForge MemoryEntries
        entries: list[MemoryEntry] = []
        for item in response.memories:
            entries.append(
                MemoryEntry(
                    memory_id=item.id,
                    content=str(item.content),
                    category="general",
                    source_trajectory_id=trajectory.trajectory_id,
                    importance=item.score if item.score is not None else 0.5,
                    specificity_score=0.5,
                    metadata=item.metadata,
                )
            )

        store = MemoryStore(
            task_id=trajectory.task_id,
            framework=f"evolvelab_{self.provider.get_memory_type().value}",
            entries=entries,
            source_trajectory_id=trajectory.trajectory_id,
        )

        logger.info(
            f"[Adapter] EvolveLab provider produced {store.num_entries} entries"
        )
        return store

    @staticmethod
    def _convert_to_evolvelab(trajectory: Trajectory) -> TrajectoryData:
        """Convert SkillForge Trajectory to EvolveLab TrajectoryData."""
        traj_steps: list[dict[str, Any]] = []
        for step in trajectory.steps:
            traj_steps.append({
                "type": step.step_type.value,
                "content": step.content,
            })

        return TrajectoryData(
            query=trajectory.task_description,
            trajectory=traj_steps,
            result=trajectory.final_answer,
            metadata={
                "task_id": trajectory.task_id,
                "success": trajectory.success,
            },
        )


# ============================================================
# Provider Registry — list all available EvolveLab providers
# ============================================================


EVOLVELAB_PROVIDER_NAMES = [
    "agent_kb",
    "skillweaver",
    "mobilee",
    "expel",
    "lightweight_memory",
    "cerebra_fusion_memory",
    "voyager",
    "dilu",
    "generative",
    "memp",
    "dynamic_cheatsheet",
    "agent_workflow_memory",
    "evolver",
]


def list_available_providers() -> list[str]:
    """List all available EvolveLab memory provider names."""
    return EVOLVELAB_PROVIDER_NAMES.copy()


def get_provider_info() -> list[dict[str, str]]:
    """Get info about all available EvolveLab providers."""
    from src.memory.evolvelab.memory_types import PROVIDER_MAPPING

    info = []
    for mem_type, (class_name, module_name) in PROVIDER_MAPPING.items():
        info.append({
            "type": mem_type.value,
            "class": class_name,
            "module": module_name,
        })
    return info
