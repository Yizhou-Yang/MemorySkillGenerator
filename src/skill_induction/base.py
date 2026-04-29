"""
Skill induction base class.

Defines the unified interface for skill inducers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import MemoryStore, Skill, Trajectory


class BaseSkillInducer(ABC):
    """Abstract base class for skill inducers."""

    @abstractmethod
    def induce(
        self,
        trajectory: Trajectory,
        memory: MemoryStore | None = None,
    ) -> Skill:
        """
        Induce a reusable skill from the given inputs.

        Args:
            trajectory: Raw interaction trajectory.
            memory: Structured memory (not required by all variants).

        Returns:
            The induced Skill.
        """
        ...
