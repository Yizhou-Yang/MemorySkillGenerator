"""Unit tests for core data models."""

import pytest

from src.models import (
    MemoryEntry,
    MemoryStore,
    Skill,
    StepType,
    TaskType,
    Trajectory,
    TrajectoryStep,
    TransformVariant,
)

class TestTrajectory:
    """Trajectory model tests."""

    def test_create_trajectory(self):
        trajectory = Trajectory(
            task_id="test_001",
            task_description="Test task",
            task_type=TaskType.QA,
        )
        assert trajectory.task_id == "test_001"
        assert trajectory.num_steps == 0
        assert trajectory.error_rate == 0.0
        assert trajectory.success is False

    def test_trajectory_with_steps(self):
        steps = [
            TrajectoryStep(step_id=0, step_type=StepType.THOUGHT, content="Thinking..."),
            TrajectoryStep(step_id=1, step_type=StepType.ACTION, content="Execute action"),
            TrajectoryStep(step_id=2, step_type=StepType.ERROR, content="Error occurred"),
            TrajectoryStep(step_id=3, step_type=StepType.OBSERVATION, content="Observation"),
        ]
        trajectory = Trajectory(
            task_id="test_002",
            task_description="Test task",
            steps=steps,
        )
        assert trajectory.num_steps == 4
        assert trajectory.error_rate == 0.25  # 1/4

    def test_trajectory_repetition_rate(self):
        steps = [
            TrajectoryStep(step_id=0, step_type=StepType.THOUGHT, content="same content"),
            TrajectoryStep(step_id=1, step_type=StepType.THOUGHT, content="same content"),
            TrajectoryStep(step_id=2, step_type=StepType.THOUGHT, content="different content"),
        ]
        trajectory = Trajectory(
            task_id="test_003",
            task_description="Test task",
            steps=steps,
        )
        # 3 steps, 2 unique -> repetition_rate = 1 - 2/3 ~ 0.333
        assert abs(trajectory.repetition_rate - 1 / 3) < 0.01

class TestMemory:
    """Memory model tests."""

    def test_create_memory_entry(self):
        entry = MemoryEntry(
            content="Python's list.sort() sorts in-place",
            category="fact",
            specificity_score=0.9,
            importance=0.8,
        )
        assert entry.category == "fact"
        assert entry.specificity_score == 0.9

    def test_memory_store_avg_specificity(self):
        entries = [
            MemoryEntry(content="memory 1", specificity_score=0.8),
            MemoryEntry(content="memory 2", specificity_score=0.6),
        ]
        store = MemoryStore(task_id="test", entries=entries)
        assert store.avg_specificity == 0.7
        assert store.num_entries == 2

class TestSkill:
    """Skill model tests."""

    def test_create_skill(self):
        skill = Skill(
            name="Test Skill",
            description="This is a test skill",
            procedure=["Step 1", "Step 2"],
            source_variant=TransformVariant.TRAJ_TO_SKILL,
        )
        assert skill.name == "Test Skill"
        assert skill.version == 1
        assert skill.compactness > 0

    def test_skill_serialization(self):
        skill = Skill(
            name="Serialization Test",
            description="Test JSON serialization",
            facts=["Fact 1"],
            rules=["Rule 1"],
        )
        json_str = skill.model_dump_json()
        assert "Serialization Test" in json_str

        # Deserialise
        restored = Skill.model_validate_json(json_str)
        assert restored.name == skill.name
        assert restored.facts == skill.facts

class TestEnums:
    """Enumeration type tests."""

    def test_transform_variant(self):
        assert TransformVariant.TRAJ_TO_SKILL.value == "traj_to_skill"
        assert TransformVariant("memory_to_skill") == TransformVariant.MEMORY_TO_SKILL

    def test_task_type(self):
        assert TaskType.QA.value == "qa"
        assert TaskType.CODE_REPAIR.value == "code_repair"
