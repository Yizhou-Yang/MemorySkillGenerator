"""
MemorySkillGenerator core data model definitions.

Defines Trajectory, Memory, Skill and other core data structures
using Pydantic v2 for type validation and serialization.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# Enumerations
# ============================================================


class TaskType(str, Enum):
    """Task type enumeration."""

    QA = "qa"
    DIALOGUE = "dialogue"
    CODE_REPAIR = "code_repair"
    TOOL_USE = "tool_use"
    REASONING = "reasoning"
    OTHER = "other"


class TransformVariant(str, Enum):
    """Skill induction pathway enumeration (three competing variants)."""

    TRAJ_TO_SKILL = "traj_to_skill"         # Variant 1: direct path
    MEMORY_TO_SKILL = "memory_to_skill"     # Variant 2: compressed path
    HYBRID_TO_SKILL = "hybrid_to_skill"     # Variant 3: hybrid path


class StepType(str, Enum):
    """Trajectory step type."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ERROR = "error"
    FEEDBACK = "feedback"


# ============================================================
# Trajectory models
# ============================================================


class TrajectoryStep(BaseModel):
    """A single step within a trajectory."""

    step_id: int = Field(..., description="Step index, starting from 0")
    step_type: StepType = Field(..., description="Type of this step")
    content: str = Field(..., description="Step content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class Trajectory(BaseModel):
    """Complete agent interaction trajectory."""

    trajectory_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique trajectory identifier",
    )
    task_id: str = Field(..., description="Associated task ID")
    task_description: str = Field(..., description="Task description")
    task_type: TaskType = Field(default=TaskType.OTHER, description="Task type")
    steps: list[TrajectoryStep] = Field(default_factory=list, description="Step sequence")
    success: bool = Field(default=False, description="Whether the task completed successfully")
    final_answer: str | None = Field(default=None, description="Final answer")
    total_tokens: int = Field(default=0, description="Total token consumption")
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def error_rate(self) -> float:
        """Fraction of steps that are errors."""
        if not self.steps:
            return 0.0
        error_count = sum(1 for step in self.steps if step.step_type == StepType.ERROR)
        return error_count / len(self.steps)

    @property
    def repetition_rate(self) -> float:
        """Fraction of duplicate steps (based on content deduplication)."""
        if not self.steps:
            return 0.0
        contents = [step.content for step in self.steps]
        unique_count = len(set(contents))
        return 1.0 - (unique_count / len(contents))


# ============================================================
# Memory models
# ============================================================


class MemoryEntry(BaseModel):
    """A single structured memory entry."""

    memory_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique memory identifier",
    )
    content: str = Field(..., description="Memory content")
    category: str = Field(
        default="general",
        description="Memory category (fact / rule / procedure / insight)",
    )
    source_trajectory_id: str | None = Field(
        default=None, description="Source trajectory ID"
    )
    specificity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Specificity score: 0 = vague, 1 = highly specific and actionable",
    )
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Importance score"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryStore(BaseModel):
    """A collection of structured memory entries."""

    task_id: str = Field(..., description="Associated task ID")
    framework: str = Field(default="mem0", description="Memory framework used")
    entries: list[MemoryEntry] = Field(
        default_factory=list, description="Memory entry list"
    )
    source_trajectory_id: str | None = Field(
        default=None, description="Source trajectory ID"
    )
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    @property
    def avg_specificity(self) -> float:
        if not self.entries:
            return 0.0
        return sum(entry.specificity_score for entry in self.entries) / len(self.entries)


# ============================================================
# Skill models
# ============================================================


class Skill(BaseModel):
    """A reusable agent skill."""

    skill_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique skill identifier",
    )
    name: str = Field(..., description="Skill name")
    description: str = Field(
        ..., description="Brief skill description (used for retrieval / matching)"
    )
    preconditions: list[str] = Field(
        default_factory=list,
        description="Preconditions: when this skill should be applied",
    )
    procedure: list[str] = Field(
        default_factory=list,
        description="Execution steps",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints: what to avoid, edge cases",
    )
    facts: list[str] = Field(
        default_factory=list,
        description="Domain facts this skill relies on",
    )
    rules: list[str] = Field(
        default_factory=list,
        description="Decision rules encoded in this skill",
    )
    source_tasks: list[str] = Field(
        default_factory=list,
        description="Source task ID list",
    )
    source_variant: TransformVariant | None = Field(
        default=None,
        description="Induction pathway used to generate this skill",
    )
    version: int = Field(default=1, description="Version number")
    success_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Validation success rate"
    )
    transfer_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Transfer capability score"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def compactness(self) -> int:
        """Skill compactness: approximate total character count."""
        total = len(self.description)
        total += sum(len(step) for step in self.procedure)
        total += sum(len(constraint) for constraint in self.constraints)
        total += sum(len(fact) for fact in self.facts)
        total += sum(len(rule) for rule in self.rules)
        return total


# ============================================================
# Experiment result models
# ============================================================


class SkillEvalResult(BaseModel):
    """Evaluation result for a single skill."""

    skill_id: str
    variant: TransformVariant
    success_rate: float = Field(
        default=0.0, description="Success rate on validation tasks"
    )
    reuse_count: int = Field(default=0, description="Number of times reused")
    compression_ratio: float = Field(
        default=0.0, description="Compression ratio = tokens(traj) / tokens(skill)"
    )
    transfer_score: float = Field(
        default=0.0, description="Cross-dataset transfer score"
    )
    validation_details: list[dict[str, Any]] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """Result of a complete experiment run."""

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config_name: str = Field(..., description="Config file name used")
    benchmark: str = Field(..., description="Benchmark name")
    memory_framework: str = Field(..., description="Memory framework name")
    variant: TransformVariant = Field(..., description="Induction pathway")
    num_tasks: int = Field(default=0)
    num_skills_generated: int = Field(default=0)
    skill_results: list[SkillEvalResult] = Field(default_factory=list)
    avg_success_rate: float = Field(default=0.0)
    avg_compression_ratio: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.now)
    notes: str = Field(default="")
