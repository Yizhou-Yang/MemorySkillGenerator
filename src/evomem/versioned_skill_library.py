"""
Versioned Skill Library — git-like version management for skills.

Unlike the flat SkillLibrary, this tracks the FULL evolution history
of each skill:
  - Every mutation (create/update/merge/retire) produces a new version
  - Old versions are retained (never overwritten)
  - Can diff any two versions
  - Can rollback to a previous version
  - Can branch (fork a skill into specialized variants)

Mirrors EvoArena's core insight: knowledge is VERSION-DEPENDENT,
not simply "latest is best".

Data model:
  SkillVersion = immutable snapshot of a skill at a point in time
  SkillHistory = ordered list of versions for one skill lineage
  VersionedSkillLibrary = collection of skill histories with retrieval
"""

from __future__ import annotations

import json
import time
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger

from src.models import Skill


# ============================================================
# Version Data Model
# ============================================================

@dataclass
class SkillVersion:
    """Immutable snapshot of a skill at a specific version."""

    version: int
    timestamp: float
    skill_snapshot: dict[str, Any]  # Full Skill dict at this version
    change_type: str  # "create", "refine", "merge", "reformat", "rollback"
    change_reason: str  # Why this version was created
    parent_version: int | None = None  # None for v1 (initial create)
    merged_from: str | None = None  # skill_id merged into this one
    triggered_by_task: str = ""  # Task that triggered this evolution
    performance_at_creation: float = 0.0  # F1 when version was created

    @property
    def description(self) -> str:
        return self.skill_snapshot.get("description", "")

    @property
    def name(self) -> str:
        return self.skill_snapshot.get("name", "")


@dataclass
class SkillHistory:
    """Complete evolution history of one skill lineage."""

    skill_id: str
    versions: list[SkillVersion] = field(default_factory=list)
    is_retired: bool = False
    retired_at_version: int | None = None
    retire_reason: str = ""

    @property
    def current_version(self) -> int:
        return len(self.versions)

    @property
    def latest(self) -> SkillVersion | None:
        return self.versions[-1] if self.versions else None

    @property
    def latest_skill(self) -> dict[str, Any]:
        return self.latest.skill_snapshot if self.latest else {}

    def get_version(self, v: int) -> SkillVersion | None:
        """Get a specific version (1-indexed)."""
        if 1 <= v <= len(self.versions):
            return self.versions[v - 1]
        return None

    def diff(self, v1: int, v2: int) -> dict[str, Any]:
        """Compare two versions, return changed fields."""
        ver1 = self.get_version(v1)
        ver2 = self.get_version(v2)
        if not ver1 or not ver2:
            return {}

        changes = {}
        s1, s2 = ver1.skill_snapshot, ver2.skill_snapshot
        for key in set(list(s1.keys()) + list(s2.keys())):
            if s1.get(key) != s2.get(key):
                changes[key] = {"before": s1.get(key), "after": s2.get(key)}
        return changes

    def evolution_summary(self) -> str:
        """Human-readable evolution timeline."""
        lines = [f"Skill: {self.skill_id[:8]}... ({self.current_version} versions)"]
        for v in self.versions:
            lines.append(
                f"  v{v.version} [{v.change_type}] {v.change_reason[:60]} "
                f"(perf={v.performance_at_creation:.2f})"
            )
        if self.is_retired:
            lines.append(f"  [RETIRED at v{self.retired_at_version}] {self.retire_reason}")
        return "\n".join(lines)


# ============================================================
# Versioned Skill Library
# ============================================================

class VersionedSkillLibrary:
    """
    Git-like versioned skill library.

    Every skill mutation creates a new version. Nothing is ever lost.
    Supports: create, evolve, merge, retire, rollback, branch, diff.
    """

    def __init__(self, storage_path: str | Path | None = None):
        self._histories: dict[str, SkillHistory] = {}  # skill_id → history
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path and self.storage_path.exists():
            self._load()

    # --- Properties ---

    @property
    def size(self) -> int:
        """Number of active (non-retired) skills."""
        return sum(1 for h in self._histories.values() if not h.is_retired)

    @property
    def total_versions(self) -> int:
        """Total version count across all skills."""
        return sum(len(h.versions) for h in self._histories.values())

    @property
    def total_lineages(self) -> int:
        """Total skill lineages (including retired)."""
        return len(self._histories)

    # --- Core Operations ---

    def create(
        self,
        skill: Skill,
        reason: str = "Initial creation",
        task_id: str = "",
        performance: float = 0.0,
    ) -> SkillHistory:
        """Create a new skill (v1)."""
        history = SkillHistory(skill_id=skill.skill_id)
        version = SkillVersion(
            version=1,
            timestamp=time.time(),
            skill_snapshot=skill.model_dump(mode="json"),
            change_type="create",
            change_reason=reason,
            parent_version=None,
            triggered_by_task=task_id,
            performance_at_creation=performance,
        )
        history.versions.append(version)
        self._histories[skill.skill_id] = history

        logger.info(
            f"[VersionedLib] Created '{skill.name}' v1 "
            f"(active={self.size}, total_versions={self.total_versions})"
        )
        self._persist()
        return history

    def evolve(
        self,
        skill_id: str,
        updated_skill: Skill,
        change_type: str = "refine",
        reason: str = "",
        task_id: str = "",
        performance: float = 0.0,
        merged_from: str | None = None,
    ) -> SkillVersion | None:
        """
        Evolve a skill to a new version.

        The old version is preserved — this creates a NEW version
        on top of the existing history.
        """
        history = self._histories.get(skill_id)
        if not history or history.is_retired:
            logger.warning(f"[VersionedLib] Cannot evolve {skill_id[:8]}: not found or retired")
            return None

        new_version_num = history.current_version + 1
        version = SkillVersion(
            version=new_version_num,
            timestamp=time.time(),
            skill_snapshot=updated_skill.model_dump(mode="json"),
            change_type=change_type,
            change_reason=reason,
            parent_version=history.current_version,
            merged_from=merged_from,
            triggered_by_task=task_id,
            performance_at_creation=performance,
        )
        history.versions.append(version)

        logger.info(
            f"[VersionedLib] Evolved '{updated_skill.name}' → v{new_version_num} "
            f"[{change_type}] {reason[:50]}"
        )
        self._persist()
        return version

    def retire(self, skill_id: str, reason: str = "", task_id: str = "") -> bool:
        """Retire a skill (mark as inactive, but preserve history)."""
        history = self._histories.get(skill_id)
        if not history:
            return False

        history.is_retired = True
        history.retired_at_version = history.current_version
        history.retire_reason = reason

        logger.info(
            f"[VersionedLib] Retired '{history.latest.name}' at v{history.current_version}: {reason[:50]}"
        )
        self._persist()
        return True

    def rollback(self, skill_id: str, to_version: int, reason: str = "") -> SkillVersion | None:
        """
        Rollback a skill to a previous version.

        This doesn't delete history — it creates a NEW version
        with the content of the old version (like git revert).
        """
        history = self._histories.get(skill_id)
        if not history:
            return None

        target = history.get_version(to_version)
        if not target:
            logger.warning(f"[VersionedLib] Version {to_version} not found for {skill_id[:8]}")
            return None

        # Create a new version with the old content
        rolled_back_skill = Skill.model_validate(target.skill_snapshot)
        rolled_back_skill.version = history.current_version + 1

        version = SkillVersion(
            version=history.current_version + 1,
            timestamp=time.time(),
            skill_snapshot=rolled_back_skill.model_dump(mode="json"),
            change_type="rollback",
            change_reason=reason or f"Rolled back to v{to_version}",
            parent_version=history.current_version,
        )
        history.versions.append(version)

        # Un-retire if was retired
        if history.is_retired:
            history.is_retired = False
            history.retired_at_version = None

        logger.info(
            f"[VersionedLib] Rolled back '{target.name}' to v{to_version} → new v{version.version}"
        )
        self._persist()
        return version

    # --- Retrieval ---

    def search(self, query: str, top_k: int = 3) -> list[tuple[SkillHistory, float]]:
        """Search active skills by query (Jaccard similarity on latest version)."""
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return []

        scored = []
        for history in self._histories.values():
            if history.is_retired:
                continue
            latest = history.latest
            if not latest:
                continue

            skill_text = f"{latest.name} {latest.description}"
            skill_tokens = set(skill_text.lower().split())
            if not skill_tokens:
                continue

            intersection = query_tokens & skill_tokens
            union = query_tokens | skill_tokens
            sim = len(intersection) / len(union) if union else 0.0
            if sim > 0:
                scored.append((history, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_history(self, skill_id: str) -> SkillHistory | None:
        return self._histories.get(skill_id)

    def list_active(self) -> list[SkillHistory]:
        return [h for h in self._histories.values() if not h.is_retired]

    def list_retired(self) -> list[SkillHistory]:
        return [h for h in self._histories.values() if h.is_retired]

    # --- Version-aware Context ---

    def get_evolution_context(self, skill_id: str, max_versions: int = 3) -> str:
        """
        Get version-aware context for a skill (for LLM augmentation).

        Returns a formatted string showing how this skill evolved,
        enabling version-aware reasoning.
        """
        history = self._histories.get(skill_id)
        if not history or not history.versions:
            return ""

        lines = [f"[Skill Evolution: {history.latest.name}]"]

        # Show recent versions
        recent = history.versions[-max_versions:]
        for v in recent:
            lines.append(
                f"  v{v.version} ({v.change_type}): {v.change_reason[:80]}"
            )

        # If there were significant changes, show diff
        if len(history.versions) >= 2:
            v_prev = history.versions[-2]
            v_curr = history.versions[-1]
            if v_prev.description != v_curr.description:
                lines.append(f"  Latest change: {v_prev.description[:100]} → {v_curr.description[:100]}")

        return "\n".join(lines)

    # --- Stats ---

    def stats(self) -> dict[str, Any]:
        return {
            "active_skills": self.size,
            "retired_skills": len(self.list_retired()),
            "total_lineages": self.total_lineages,
            "total_versions": self.total_versions,
            "avg_versions_per_skill": (
                self.total_versions / self.total_lineages if self.total_lineages else 0
            ),
        }

    # --- Persistence ---

    def _persist(self):
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for sid, history in self._histories.items():
            data[sid] = {
                "skill_id": history.skill_id,
                "is_retired": history.is_retired,
                "retired_at_version": history.retired_at_version,
                "retire_reason": history.retire_reason,
                "versions": [asdict(v) for v in history.versions],
            }
        self.storage_path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self):
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text())
            for sid, hdata in data.items():
                history = SkillHistory(
                    skill_id=hdata["skill_id"],
                    is_retired=hdata.get("is_retired", False),
                    retired_at_version=hdata.get("retired_at_version"),
                    retire_reason=hdata.get("retire_reason", ""),
                )
                for vdata in hdata.get("versions", []):
                    history.versions.append(SkillVersion(**vdata))
                self._histories[sid] = history
            logger.info(
                f"[VersionedLib] Loaded: {self.size} active, "
                f"{self.total_versions} total versions"
            )
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"[VersionedLib] Load failed: {e}")
