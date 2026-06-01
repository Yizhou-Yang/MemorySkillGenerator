"""
EvoMem — Git-like Skill Evolution with Version Management.

Implements the core ideas from EvoArena (2026):
1. PatchRecorder: append-only log of all skill mutations
2. PatchRetriever: version-aware retrieval for augmented reasoning
3. VersionedSkillLibrary: git-like version management for skills
4. SkillEmbedder: vector-based semantic skill comparison
5. SkillEvolutionEngine: unified evolution decision engine
   - Vector similarity drives decisions (not token overlap)
   - Version history informs rollback/retire
   - 6 evolution actions: create, strengthen, refine, branch, retire, rollback
   - Periodic consolidation of redundant skills
"""

from src.evomem.patch_recorder import PatchRecorder, SkillPatch
from src.evomem.patch_retriever import PatchRetriever
from src.evomem.versioned_skill_library import (
    VersionedSkillLibrary,
    SkillHistory,
    SkillVersion,
)
from src.evomem.skill_embedder import SkillEmbedder
from src.evomem.evolution_engine import SkillEvolutionEngine, EvolutionDecision, SkillState

__all__ = [
    "PatchRecorder",
    "PatchRetriever",
    "SkillPatch",
    "VersionedSkillLibrary",
    "SkillHistory",
    "SkillVersion",
    "SkillEmbedder",
    "SkillEvolutionEngine",
    "EvolutionDecision",
    "SkillState",
]
