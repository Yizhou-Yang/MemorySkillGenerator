"""
Memory consolidation module.

Implements memory deduplication and merging to address the "only-accumulate,
never-compress" problem identified in Mem2Evolve analysis (§2.2-2.4).

Key mechanisms:
- Cosine similarity-based deduplication between memory entries
- LLM-driven merge pass to combine redundant memories into abstractions
- Compactness objective: post-consolidation count <= 70% of original

Reference: docs/internal/mem2evolve_analysis.md §2.4 Level 1
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.models import MemoryEntry, MemoryStore
from src.utils.llm import LLMClient


class MemoryConsolidator:
    """
    Consolidates a MemoryStore by deduplicating and merging redundant entries.

    Two-phase approach:
    1. Similarity-based clustering: group entries with cosine sim > threshold
    2. LLM merge pass: merge each cluster into a single, more abstract entry

    This addresses the "compression as side-effect" limitation where memories
    accumulate without bound and never get compressed.
    """

    DEFAULT_SIMILARITY_THRESHOLD = 0.75
    DEFAULT_TARGET_RATIO = 0.7  # Target: keep <= 70% of original entries

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.similarity_threshold: float = self.config.get(
            "similarity_threshold", self.DEFAULT_SIMILARITY_THRESHOLD
        )
        self.target_ratio: float = self.config.get(
            "target_ratio", self.DEFAULT_TARGET_RATIO
        )

    def consolidate(self, store: MemoryStore) -> MemoryStore:
        """
        Consolidate a memory store by deduplicating and merging entries.

        Args:
            store: The memory store to consolidate.

        Returns:
            A new MemoryStore with consolidated entries.
        """
        if store.num_entries <= 2:
            logger.info("[Consolidation] Too few entries to consolidate, skipping")
            return store

        logger.info(
            f"[Consolidation] Starting: {store.num_entries} entries, "
            f"target_ratio={self.target_ratio}"
        )

        # Phase 1: Compute similarity and cluster
        clusters = self._cluster_by_similarity(store.entries)
        logger.info(
            f"[Consolidation] Phase 1: {len(clusters)} clusters formed "
            f"from {store.num_entries} entries"
        )

        # Phase 2: Merge clusters
        merged_entries = self._merge_clusters(clusters, store.task_id)
        logger.info(
            f"[Consolidation] Phase 2: merged to {len(merged_entries)} entries "
            f"(ratio={len(merged_entries)/store.num_entries:.2f})"
        )

        return MemoryStore(
            task_id=store.task_id,
            framework=store.framework,
            entries=merged_entries,
            source_trajectory_id=store.source_trajectory_id,
        )

    def compute_similarity(self, entry_a: MemoryEntry, entry_b: MemoryEntry) -> float:
        """
        Compute text-based similarity between two memory entries.

        Uses token overlap (Jaccard similarity) as a lightweight proxy
        for cosine similarity when no embedding model is available.
        """
        tokens_a = set(entry_a.content.lower().split())
        tokens_b = set(entry_b.content.lower().split())

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union) if union else 0.0

    def _cluster_by_similarity(
        self, entries: list[MemoryEntry]
    ) -> list[list[MemoryEntry]]:
        """
        Cluster entries by pairwise similarity using greedy clustering.

        Each entry is assigned to the first cluster whose centroid
        (first entry) has similarity >= threshold. If no match, a new
        cluster is created.
        """
        clusters: list[list[MemoryEntry]] = []
        assigned: set[int] = set()

        for i, entry_i in enumerate(entries):
            if i in assigned:
                continue

            cluster = [entry_i]
            assigned.add(i)

            for j in range(i + 1, len(entries)):
                if j in assigned:
                    continue
                sim = self.compute_similarity(entry_i, entries[j])
                if sim >= self.similarity_threshold:
                    cluster.append(entries[j])
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def _merge_clusters(
        self, clusters: list[list[MemoryEntry]], task_id: str
    ) -> list[MemoryEntry]:
        """
        Merge each multi-entry cluster into a single consolidated entry.

        Single-entry clusters are kept as-is.
        Multi-entry clusters are merged via LLM (if available) or heuristic.
        """
        merged: list[MemoryEntry] = []

        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                merged_entry = self._merge_single_cluster(cluster, task_id)
                merged.append(merged_entry)

        return merged

    def _merge_single_cluster(
        self, cluster: list[MemoryEntry], task_id: str
    ) -> MemoryEntry:
        """
        Merge a cluster of similar entries into one consolidated entry.

        Uses LLM if available, otherwise falls back to heuristic merge.
        """
        if self.llm_client is not None:
            return self._llm_merge(cluster, task_id)
        return self._heuristic_merge(cluster)

    def _llm_merge(self, cluster: list[MemoryEntry], task_id: str) -> MemoryEntry:
        """Use LLM to merge a cluster of similar memories into one."""
        entries_text = "\n".join(
            f"- [{e.category}] (importance={e.importance:.1f}) {e.content}"
            for e in cluster
        )

        prompt = f"""You have {len(cluster)} similar memory entries that should be merged into ONE more abstract, comprehensive entry.

Entries to merge:
{entries_text}

Requirements:
1. The merged entry must capture ALL key information from the originals
2. It should be MORE abstract and general than any single original
3. It should be concise — shorter than the sum of originals
4. Preserve the most important category and highest importance score

Return JSON:
{{
  "content": "merged memory content",
  "category": "fact/rule/procedure/insight",
  "importance": 0.0-1.0,
  "specificity_score": 0.0-1.0
}}"""

        messages = [
            {
                "role": "system",
                "content": "You are a memory consolidation agent. Merge similar memories into one concise, abstract entry.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm_client.chat_json(messages)
            data = json.loads(response)
            return MemoryEntry(
                content=data.get("content", cluster[0].content),
                category=data.get("category", cluster[0].category),
                importance=float(data.get("importance", max(e.importance for e in cluster))),
                specificity_score=float(data.get("specificity_score", 0.7)),
                source_trajectory_id=cluster[0].source_trajectory_id,
                metadata={"merged_from": len(cluster), "source_task_id": task_id},
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error(f"[Consolidation] LLM merge failed: {exc}, using heuristic")
            return self._heuristic_merge(cluster)

    def _heuristic_merge(self, cluster: list[MemoryEntry]) -> MemoryEntry:
        """
        Heuristic merge: pick the highest-importance entry and enrich it.

        Strategy: use the most important entry as base, append unique
        information from others.
        """
        # Sort by importance descending
        sorted_entries = sorted(cluster, key=lambda e: e.importance, reverse=True)
        base = sorted_entries[0]

        # Collect unique content fragments from other entries
        additional_info: list[str] = []
        base_tokens = set(base.content.lower().split())
        for entry in sorted_entries[1:]:
            entry_tokens = set(entry.content.lower().split())
            unique_tokens = entry_tokens - base_tokens
            if len(unique_tokens) > 3:  # Only add if meaningful new info
                additional_info.append(entry.content)

        # Combine
        if additional_info:
            merged_content = base.content + " Additionally: " + "; ".join(additional_info)
        else:
            merged_content = base.content

        return MemoryEntry(
            content=merged_content,
            category=base.category,
            importance=max(e.importance for e in cluster),
            specificity_score=max(e.specificity_score for e in cluster),
            source_trajectory_id=base.source_trajectory_id,
            metadata={"merged_from": len(cluster)},
        )

    def should_consolidate(self, store: MemoryStore, trigger_threshold: int = 10) -> bool:
        """
        Determine if consolidation should be triggered.

        Triggers when the store has accumulated more than trigger_threshold entries.
        """
        return store.num_entries >= trigger_threshold
