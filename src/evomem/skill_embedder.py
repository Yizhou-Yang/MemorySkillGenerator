"""
Skill Embedder — vector-based skill comparison and retrieval.

Replaces Jaccard token-overlap with proper semantic embedding:
  - Uses sentence-transformers for local embedding
  - Cosine similarity for skill matching
  - Supports incremental index updates (add/remove/update)
  - Version-aware: can embed and compare different versions of same skill

Key advantage over token-overlap:
  "Decompose multi-hop questions" ≈ "Break complex queries into sub-questions"
  (semantically similar but zero token overlap)
"""

from __future__ import annotations

import numpy as np
from typing import Any
from loguru import logger


class SkillEmbedder:
    """
    Semantic embedding for skills using sentence-transformers.

    Maintains an in-memory index of skill embeddings for fast
    cosine similarity search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

        # Index: skill_id → embedding vector
        self._index: dict[str, np.ndarray] = {}
        # Cache: text → embedding (avoid recomputing)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"[Embedder] Loaded model: {self.model_name} on {self.device}")
            except ImportError:
                logger.warning(
                    "[Embedder] sentence-transformers not available, "
                    "falling back to TF-IDF embeddings"
                )
                self._model = "tfidf_fallback"
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a text string into a vector."""
        if text in self._cache:
            return self._cache[text]

        if self.model == "tfidf_fallback":
            vec = self._tfidf_embed(text)
        else:
            vec = self.model.encode(text, normalize_embeddings=True)

        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts."""
        if self.model == "tfidf_fallback":
            return np.array([self._tfidf_embed(t) for t in texts])
        return self.model.encode(texts, normalize_embeddings=True)

    def add_to_index(self, skill_id: str, text: str) -> None:
        """Add or update a skill in the embedding index."""
        self._index[skill_id] = self.embed(text)

    def remove_from_index(self, skill_id: str) -> None:
        """Remove a skill from the index."""
        self._index.pop(skill_id, None)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[tuple[str, float]]:
        """
        Find the most similar skills to a query.

        Returns:
            List of (skill_id, cosine_similarity) sorted descending.
        """
        if not self._index:
            return []

        q_vec = self.embed(query)
        scores = []

        for skill_id, s_vec in self._index.items():
            sim = float(np.dot(q_vec, s_vec))
            if sim >= threshold:
                scores.append((skill_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        return float(np.dot(vec_a, vec_b))

    def version_drift(self, texts_over_time: list[str]) -> list[float]:
        """
        Compute semantic drift across versions.

        Returns cosine distance between consecutive versions.
        Useful for detecting when a skill has drifted too far from its origin.
        """
        if len(texts_over_time) < 2:
            return []

        embeddings = self.embed_batch(texts_over_time)
        drifts = []
        for i in range(1, len(embeddings)):
            sim = float(np.dot(embeddings[i], embeddings[i - 1]))
            drifts.append(1.0 - sim)  # distance = 1 - similarity
        return drifts

    def find_merge_candidates(self, threshold: float = 0.75) -> list[tuple[str, str, float]]:
        """
        Find pairs of skills that are semantically similar enough to merge.

        Returns:
            List of (skill_id_a, skill_id_b, similarity) above threshold.
        """
        ids = list(self._index.keys())
        if len(ids) < 2:
            return []

        candidates = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = float(np.dot(self._index[ids[i]], self._index[ids[j]]))
                if sim >= threshold:
                    candidates.append((ids[i], ids[j], sim))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    @property
    def index_size(self) -> int:
        return len(self._index)

    def _tfidf_embed(self, text: str, dim: int = 256) -> np.ndarray:
        """
        Fallback: simple hash-based embedding when sentence-transformers unavailable.
        Not semantic, but deterministic and fast.
        """
        tokens = text.lower().split()
        vec = np.zeros(dim, dtype=np.float32)
        for token in tokens:
            h = hash(token) % dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
