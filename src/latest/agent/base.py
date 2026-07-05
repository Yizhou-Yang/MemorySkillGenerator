"""
Abstract agent interfaces for SkillForge.

Enables different agent backends (A-Mem, Terminus 2, etc.) to be
swapped in without changing the evaluation pipeline.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract agent that executes a task and returns a result.

    All benchmark-specific agents (AmemAgent, TerminalAgent, etc.)
    implement this interface, so the runner can dispatch uniformly.
    """

    @abstractmethod
    async def run_task(self, task: dict, experience_section: str = "",
                       group: str = "A") -> dict:
        """Execute a single benchmark task.

        Args:
            task: Task dict from benchmark loader (task_id, description, etc.)
            experience_section: Augmented prompt from SkillForge experience library.
            group: Split group label (train/test/A/B/C).

        Returns:
            Result dict with keys: task_id, response, expected, error,
            time_cost, augmented, group (and optional: event_log, actions).
        """
        ...

    @abstractmethod
    def supports_benchmark(self, benchmark: str) -> bool:
        """Check whether this agent supports a given benchmark."""
        ...


class AgentFactory:
    """Registry-based factory for creating agents by benchmark.

    Usage:
        factory = AgentFactory()
        factory.register("locomo", AmemAgent(...))
        factory.register("gaia2", TerminalAgent(...))

        agent = factory.get("locomo")
        result = await agent.run_task(task, experience_section)
    """

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def register(self, benchmark: str, agent: BaseAgent):
        self._agents[benchmark] = agent

    def get(self, benchmark: str) -> BaseAgent:
        if benchmark not in self._agents:
            raise KeyError(f"No agent registered for benchmark '{benchmark}'")
        return self._agents[benchmark]
