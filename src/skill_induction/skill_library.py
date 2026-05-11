"""
Skill library — persistent storage, retrieval, and reuse of validated skills.

Implements the Asset Recruitment mechanism (P3) from Mem2Evolve analysis:
- Skill Library: persistent storage of validated skills
- Skill Retrieval: cosine similarity-based search for matching skills
- Recruit vs Create: threshold-based decision (Γ function from Mem2Evolve §4)

Reference: docs/internal/mem2evolve_analysis.md §4 "Asset Recruitment"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.models import Skill


class SkillLibrary:
    """
    Persistent skill library with retrieval capabilities.

    Stores validated skills and provides similarity-based retrieval
    to enable skill reuse (recruit) vs creation (create) decisions.
    """

    DEFAULT_RECRUIT_THRESHOLD = 0.6  # Cosine sim threshold for recruitment
    DEFAULT_MAX_RESULTS = 5

    def __init__(
        self,
        storage_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        self.storage_path = Path(storage_path) if storage_path else None
        self.recruit_threshold: float = self.config.get(
            "recruit_threshold", self.DEFAULT_RECRUIT_THRESHOLD
        )
        self.max_results: int = self.config.get(
            "max_results", self.DEFAULT_MAX_RESULTS
        )
        # In-memory skill store
        self._skills: dict[str, Skill] = {}
        # Performance tracking: skill_id -> list of (em, f1) tuples
        self._performance: dict[str, list[tuple[float, float]]] = {}

        # Load from disk if path exists
        if self.storage_path and self.storage_path.exists():
            self._load()

    @property
    def size(self) -> int:
        """Number of skills in the library."""
        return len(self._skills)

    def add(self, skill: Skill, validated: bool = True) -> None:
        """
        Add a skill to the library.

        Args:
            skill: The skill to add.
            validated: Whether the skill has passed validation.
                      Only validated skills should be added.
        """
        if not validated:
            logger.warning(
                f"[SkillLibrary] Skill '{skill.name}' not validated, skipping"
            )
            return

        self._skills[skill.skill_id] = skill
        if skill.skill_id not in self._performance:
            self._performance[skill.skill_id] = []

        logger.info(
            f"[SkillLibrary] Added skill '{skill.name}' "
            f"(id={skill.skill_id[:8]}..., library_size={self.size})"
        )

        if self.storage_path:
            self._save()

    def remove(self, skill_id: str) -> bool:
        """
        Remove a skill from the library (retirement).

        Args:
            skill_id: The skill ID to remove.

        Returns:
            True if the skill was found and removed.
        """
        if skill_id in self._skills:
            name = self._skills[skill_id].name
            del self._skills[skill_id]
            self._performance.pop(skill_id, None)
            logger.info(f"[SkillLibrary] Retired skill '{name}' (id={skill_id[:8]}...)")
            if self.storage_path:
                self._save()
            return True
        return False

    def get(self, skill_id: str) -> Skill | None:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def search(self, query: str, top_k: int | None = None) -> list[tuple[Skill, float]]:
        """
        Search for skills matching a query description.

        Uses token-overlap (Jaccard) similarity as a lightweight proxy
        for semantic similarity.

        Args:
            query: Task description or query string.
            top_k: Maximum number of results (default: self.max_results).

        Returns:
            List of (skill, similarity_score) tuples, sorted by score descending.
        """
        if not self._skills:
            return []

        top_k = top_k or self.max_results
        query_tokens = set(query.lower().split())

        if not query_tokens:
            return []

        scored: list[tuple[Skill, float]] = []
        for skill in self._skills.values():
            # Combine skill name + description + procedure for matching
            skill_text = f"{skill.name} {skill.description} {' '.join(skill.procedure)}"
            skill_tokens = set(skill_text.lower().split())

            if not skill_tokens:
                continue

            # Jaccard similarity
            intersection = query_tokens & skill_tokens
            union = query_tokens | skill_tokens
            sim = len(intersection) / len(union) if union else 0.0

            if sim > 0:
                scored.append((skill, sim))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def recruit_or_create(self, task_description: str) -> tuple[Skill | None, float]:
        """
        Decide whether to recruit an existing skill or create a new one.

        Implements the Γ(s_i) function from Mem2Evolve §4:
        - If max similarity >= threshold: recruit (return the skill)
        - Otherwise: create (return None)

        Args:
            task_description: The task description to match against.

        Returns:
            Tuple of (best_matching_skill_or_None, similarity_score).
        """
        results = self.search(task_description, top_k=1)

        if not results:
            logger.info("[SkillLibrary] No skills in library, must create")
            return None, 0.0

        best_skill, best_sim = results[0]

        if best_sim >= self.recruit_threshold:
            logger.info(
                f"[SkillLibrary] RECRUIT: '{best_skill.name}' "
                f"(sim={best_sim:.3f} >= threshold={self.recruit_threshold})"
            )
            return best_skill, best_sim
        else:
            logger.info(
                f"[SkillLibrary] CREATE: best match '{best_skill.name}' "
                f"(sim={best_sim:.3f} < threshold={self.recruit_threshold})"
            )
            return None, best_sim

    def record_performance(self, skill_id: str, em: float, f1: float) -> None:
        """
        Record a performance observation for a skill.

        Used to track skill effectiveness over time for retirement decisions.
        """
        if skill_id not in self._performance:
            self._performance[skill_id] = []
        self._performance[skill_id].append((em, f1))

    def get_performance_history(self, skill_id: str) -> list[tuple[float, float]]:
        """Get the performance history for a skill."""
        return self._performance.get(skill_id, [])

    def get_consecutive_failures(self, skill_id: str) -> int:
        """
        Count consecutive failures (EM=0) from the most recent observations.

        Used for retirement decisions.
        """
        history = self._performance.get(skill_id, [])
        if not history:
            return 0

        count = 0
        for em, _ in reversed(history):
            if em == 0.0:
                count += 1
            else:
                break
        return count

    def list_all(self) -> list[Skill]:
        """List all skills in the library."""
        return list(self._skills.values())

    def _save(self) -> None:
        """Persist the library to disk as JSON."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "skills": {
                sid: skill.model_dump(mode="json")
                for sid, skill in self._skills.items()
            },
            "performance": self._performance,
        }
        self.storage_path.write_text(json.dumps(data, indent=2, default=str))
        logger.debug(f"[SkillLibrary] Saved {self.size} skills to {self.storage_path}")

    def _load(self) -> None:
        """Load the library from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text())
            for sid, skill_data in data.get("skills", {}).items():
                self._skills[sid] = Skill.model_validate(skill_data)
            self._performance = {
                k: [tuple(v) for v in vals]
                for k, vals in data.get("performance", {}).items()
            }
            logger.info(f"[SkillLibrary] Loaded {self.size} skills from {self.storage_path}")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(f"[SkillLibrary] Failed to load: {exc}")
