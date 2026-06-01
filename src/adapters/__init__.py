"""
Base Adapter — abstract interface for benchmark-specific trace parsing.

Each benchmark has its own format for:
  - Task description extraction
  - Action trace parsing (tool name, args, observation)
  - Oracle action extraction
  - Outcome evaluation

The adapter translates benchmark-specific formats into SkillForge's generic format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GenericAction:
    """Benchmark-agnostic representation of an agent action."""
    tool: str
    args: dict
    observation: str = ""
    had_error: bool = False
    step_index: int = 0


@dataclass
class GenericTask:
    """Benchmark-agnostic representation of a task."""
    task_id: str
    task_description: str
    oracle_actions: list[GenericAction]
    agent_actions: list[GenericAction]


class BaseAdapter(ABC):
    """Abstract adapter for benchmark-specific trace parsing."""

    @abstractmethod
    def parse_task(self, scenario_path: str, trace_path: str) -> GenericTask:
        """Parse a scenario + trace into generic format."""
        ...

    @abstractmethod
    def evaluate(self, task: GenericTask) -> dict[str, float]:
        """Evaluate agent performance. Returns {er, precision, f1, ...}."""
        ...

    @abstractmethod
    def extract_tool_calls(self, trace_path: str) -> list[GenericAction]:
        """Extract agent's tool calls from a trace file."""
        ...

    @abstractmethod
    def extract_oracle(self, scenario_path: str) -> list[GenericAction]:
        """Extract oracle (expected) actions from a scenario file."""
        ...

    @abstractmethod
    def get_task_description(self, scenario_path: str) -> str:
        """Extract task description from scenario."""
        ...
