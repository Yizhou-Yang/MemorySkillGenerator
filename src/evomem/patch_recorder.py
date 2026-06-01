"""
Patch Recorder — monitors skill library updates and records meaningful diffs.

Implements the Patch Recording component from EvoArena/EvoMem:
- Detects non-additive changes (modifications, overwrites, merges)
- Records structured patches: (timestamp, old_content, new_content, rationale, summary, evidence)
- Append-only patch log for full version history
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class SkillPatch:
    """A single patch recording a skill library change."""

    patch_id: str
    timestamp: float
    change_type: str  # "merge", "update", "delete", "refine", "reformat"
    skill_id: str
    skill_name: str

    # Core patch fields (EvoMem §4.2)
    content_before: str  # C_t^- : skill state before update
    content_after: str   # C_t^+ : skill state after update
    rationale: str       # r_t : why the change happened
    summary: str         # z_t : semantic summary of the change
    evidence: str        # e_t : triggering context/task

    # Optional metadata
    related_task_id: str = ""
    related_benchmark: str = ""
    step_index: int = 0

    def to_retrieval_text(self) -> str:
        """Format patch for retrieval matching."""
        return (
            f"[{self.change_type.upper()}] {self.skill_name}: {self.summary}\n"
            f"Before: {self.content_before[:200]}\n"
            f"After: {self.content_after[:200]}\n"
            f"Reason: {self.rationale}"
        )


class PatchRecorder:
    """
    Records skill library changes as append-only patches.

    Acts as a transparent monitoring layer — does NOT modify the skill
    library itself, only records diffs alongside it (like git reflog).
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.patches: list[SkillPatch] = []
        self.storage_path = Path(storage_path) if storage_path else None
        self._patch_counter = 0

        if self.storage_path and self.storage_path.exists():
            self._load()

    @property
    def size(self) -> int:
        return len(self.patches)

    def record_merge(
        self,
        skill_id: str,
        skill_name: str,
        content_before: str,
        content_after: str,
        merged_with: str,
        rationale: str = "",
        task_id: str = "",
        benchmark: str = "",
        step: int = 0,
    ) -> SkillPatch:
        """Record a skill merge event."""
        patch = self._create_patch(
            change_type="merge",
            skill_id=skill_id,
            skill_name=skill_name,
            content_before=content_before,
            content_after=content_after,
            rationale=rationale or f"Merged with skill '{merged_with}' due to high similarity",
            summary=f"MERGE: {skill_name} + {merged_with} → consolidated skill",
            evidence=f"Task: {task_id}, Benchmark: {benchmark}",
            task_id=task_id,
            benchmark=benchmark,
            step=step,
        )
        return patch

    def record_update(
        self,
        skill_id: str,
        skill_name: str,
        content_before: str,
        content_after: str,
        rationale: str = "",
        task_id: str = "",
        benchmark: str = "",
        step: int = 0,
    ) -> SkillPatch:
        """Record a skill content update (refinement)."""
        patch = self._create_patch(
            change_type="update",
            skill_id=skill_id,
            skill_name=skill_name,
            content_before=content_before,
            content_after=content_after,
            rationale=rationale or "Skill refined based on new task experience",
            summary=f"UPDATE: {skill_name} refined",
            evidence=f"Task: {task_id}",
            task_id=task_id,
            benchmark=benchmark,
            step=step,
        )
        return patch

    def record_delete(
        self,
        skill_id: str,
        skill_name: str,
        content_before: str,
        rationale: str = "",
        task_id: str = "",
        benchmark: str = "",
        step: int = 0,
    ) -> SkillPatch:
        """Record a skill deletion/retirement."""
        patch = self._create_patch(
            change_type="delete",
            skill_id=skill_id,
            skill_name=skill_name,
            content_before=content_before,
            content_after="[DELETED]",
            rationale=rationale or "Skill retired due to consistent underperformance",
            summary=f"DELETE: {skill_name} retired from library",
            evidence=f"Task: {task_id}",
            task_id=task_id,
            benchmark=benchmark,
            step=step,
        )
        return patch

    def record_reformat(
        self,
        skill_id: str,
        skill_name: str,
        content_before: str,
        content_after: str,
        rationale: str = "",
        step: int = 0,
    ) -> SkillPatch:
        """Record an attention-optimization reformat."""
        patch = self._create_patch(
            change_type="reformat",
            skill_id=skill_id,
            skill_name=skill_name,
            content_before=content_before,
            content_after=content_after,
            rationale=rationale or "Format standardized for attention optimization",
            summary=f"REFORMAT: {skill_name} format/position optimized",
            evidence="δ_attention optimization pass",
            step=step,
        )
        return patch

    def get_patches_for_skill(self, skill_id: str) -> list[SkillPatch]:
        """Get all patches related to a specific skill."""
        return [p for p in self.patches if p.skill_id == skill_id]

    def get_recent_patches(self, n: int = 10) -> list[SkillPatch]:
        """Get the N most recent patches."""
        return self.patches[-n:]

    def _create_patch(
        self,
        change_type: str,
        skill_id: str,
        skill_name: str,
        content_before: str,
        content_after: str,
        rationale: str,
        summary: str,
        evidence: str,
        task_id: str = "",
        benchmark: str = "",
        step: int = 0,
    ) -> SkillPatch:
        """Create and store a new patch."""
        self._patch_counter += 1
        patch = SkillPatch(
            patch_id=f"patch_{self._patch_counter:04d}",
            timestamp=time.time(),
            change_type=change_type,
            skill_id=skill_id,
            skill_name=skill_name,
            content_before=content_before,
            content_after=content_after,
            rationale=rationale,
            summary=summary,
            evidence=evidence,
            related_task_id=task_id,
            related_benchmark=benchmark,
            step_index=step,
        )
        self.patches.append(patch)

        logger.info(
            f"[EvoMem] Recorded {change_type.upper()} patch for '{skill_name}' "
            f"(total patches: {self.size})"
        )

        if self.storage_path:
            self._save()

        return patch

    def _save(self) -> None:
        """Persist patches to disk."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(p) for p in self.patches]
        self.storage_path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self) -> None:
        """Load patches from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text())
            self.patches = [SkillPatch(**d) for d in data]
            self._patch_counter = len(self.patches)
            logger.info(f"[EvoMem] Loaded {self.size} patches from {self.storage_path}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[EvoMem] Failed to load patches: {e}")
