"""Abstract base classes for SkillForge pipeline — enables dependency injection and testability."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseSafetyGuard(ABC):
    """Interface for runtime safety guards that block unsafe actions in agentic loops."""

    @abstractmethod
    def check(self, **kwargs) -> bool:
        """Return True if the action should be blocked."""
        ...


class BaseResponseProcessor(ABC):
    """Interface for parsing and cleaning raw LLM responses."""

    @abstractmethod
    async def process_response(self, raw_response: str, task_context: str = "") -> Any:
        """Parse raw LLM response, extract actions, evaluate non-action text."""
        ...
