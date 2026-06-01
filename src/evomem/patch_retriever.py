"""
Patch Retriever — retrieves relevant historical patches for version-aware reasoning.

Implements the Patch-Augmented Retrieval from EvoArena/EvoMem:
- Given a query, retrieves top-k relevant patches from history
- Augments standard skill retrieval with version-aware context
- Helps agent understand WHY skills changed and WHAT was before
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.evomem.patch_recorder import PatchRecorder, SkillPatch


class PatchRetriever:
    """
    Retrieves relevant patches from history to augment skill retrieval.

    Uses token-overlap similarity (same as SkillLibrary) for lightweight
    patch matching without requiring an embedding model.
    """

    def __init__(
        self,
        recorder: PatchRecorder,
        top_k: int = 3,
        min_similarity: float = 0.05,
    ) -> None:
        self.recorder = recorder
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[SkillPatch, float]]:
        """
        Retrieve patches relevant to a query.

        Args:
            query: Task description or question.
            top_k: Number of patches to return.

        Returns:
            List of (patch, similarity_score) tuples.
        """
        if not self.recorder.patches:
            return []

        top_k = top_k or self.top_k
        query_tokens = set(query.lower().split())

        if not query_tokens:
            return []

        scored: list[tuple[SkillPatch, float]] = []
        for patch in self.recorder.patches:
            patch_text = patch.to_retrieval_text()
            patch_tokens = set(patch_text.lower().split())

            if not patch_tokens:
                continue

            # Jaccard similarity
            intersection = query_tokens & patch_tokens
            union = query_tokens | patch_tokens
            sim = len(intersection) / len(union) if union else 0.0

            if sim >= self.min_similarity:
                scored.append((patch, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        if results:
            logger.debug(
                f"[EvoMem] Retrieved {len(results)} patches for query "
                f"(best sim: {results[0][1]:.3f})"
            )

        return results

    def format_patches_for_context(
        self,
        patches: list[tuple[SkillPatch, float]],
        max_tokens: int = 1000,
    ) -> str:
        """
        Format retrieved patches as context string for LLM.

        Args:
            patches: Retrieved (patch, score) tuples.
            max_tokens: Approximate token budget (chars / 4).

        Returns:
            Formatted string with patch history context.
        """
        if not patches:
            return ""

        lines = ["[Skill Evolution History (EvoMem patches)]"]
        char_budget = max_tokens * 4  # ~4 chars per token

        for patch, score in patches:
            entry = (
                f"\n--- Patch {patch.patch_id} ({patch.change_type}) ---\n"
                f"Skill: {patch.skill_name}\n"
                f"Change: {patch.summary}\n"
                f"Before: {patch.content_before[:150]}...\n"
                f"After: {patch.content_after[:150]}...\n"
                f"Reason: {patch.rationale}\n"
            )

            if sum(len(l) for l in lines) + len(entry) > char_budget:
                break
            lines.append(entry)

        return "\n".join(lines)

    def augmented_retrieval(
        self,
        query: str,
        skill_context: str,
        top_k_patches: int | None = None,
    ) -> str:
        """
        Perform patch-augmented retrieval (EvoMem §4.3):
        1. Take standard skill retrieval result (c_mem)
        2. Retrieve relevant patches (P_q)
        3. Concatenate: c(q) = Concat(c_mem, P_q)

        Args:
            query: Task description.
            skill_context: Standard retrieved skill context (c_mem).
            top_k_patches: Number of patches to retrieve.

        Returns:
            Augmented context string.
        """
        patches = self.retrieve(query, top_k=top_k_patches)

        if not patches:
            return skill_context

        patch_context = self.format_patches_for_context(patches)

        # Concatenate: standard skills + patch history
        augmented = f"{skill_context}\n\n{patch_context}"

        logger.info(
            f"[EvoMem] Augmented retrieval: {len(patches)} patches added to context"
        )

        return augmented
