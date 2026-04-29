"""
Skill induction factory.

Creates the appropriate skill inducer based on the variant name.
"""

from __future__ import annotations

from typing import Any

from src.models import TransformVariant
from src.skill_induction.base import BaseSkillInducer
from src.skill_induction.hybrid_to_skill import HybridToSkillInducer
from src.skill_induction.memory_to_skill import MemoryToSkillInducer
from src.skill_induction.traj_to_skill import TrajToSkillInducer
from src.utils.llm import LLMClient


def create_inducer(
    variant: str | TransformVariant,
    llm_client: LLMClient,
    config: dict[str, Any] | None = None,
) -> BaseSkillInducer:
    """
    Factory: create a skill inducer for the given variant.

    Args:
        variant: Induction pathway name or enum value.
        llm_client: LLM client instance.
        config: Additional configuration.

    Returns:
        The corresponding skill inducer instance.
    """
    if isinstance(variant, str):
        variant = TransformVariant(variant)

    inducer_classes: dict[TransformVariant, type[BaseSkillInducer]] = {
        TransformVariant.TRAJ_TO_SKILL: TrajToSkillInducer,
        TransformVariant.MEMORY_TO_SKILL: MemoryToSkillInducer,
        TransformVariant.HYBRID_TO_SKILL: HybridToSkillInducer,
    }

    if variant not in inducer_classes:
        raise ValueError(
            f"Unsupported variant: {variant}. "
            f"Available: {[v.value for v in inducer_classes]}"
        )

    return inducer_classes[variant](llm_client, config)
