"""Unit tests for memory compressors (with mock LLM, no network required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.memory.compressor import (
    AMEMCompressor,
    BaseMemoryCompressor,
    Mem0Compressor,
    MemoryBankCompressor,
    create_compressor,
)
from src.models import (
    MemoryEntry,
    MemoryStore,
    StepType,
    Trajectory,
    TrajectoryStep,
)


# ============================================================
# Fixtures
# ============================================================


def _make_trajectory(
    task_id: str = "test_task_001",
    success: bool = True,
    steps: list[TrajectoryStep] | None = None,
) -> Trajectory:
    """Create a sample trajectory for testing."""
    if steps is None:
        steps = [
            TrajectoryStep(step_id=0, step_type=StepType.THOUGHT, content="Analysing the question"),
            TrajectoryStep(step_id=1, step_type=StepType.ACTION, content="Search for relevant info"),
            TrajectoryStep(step_id=2, step_type=StepType.OBSERVATION, content="Found key facts"),
            TrajectoryStep(step_id=3, step_type=StepType.THOUGHT, content="Combining evidence"),
            TrajectoryStep(step_id=4, step_type=StepType.ACTION, content="Formulate answer"),
        ]
    return Trajectory(
        task_id=task_id,
        task_description="Answer a multi-hop question about history",
        success=success,
        steps=steps,
    )


def _make_mock_llm(response_json: dict | str) -> MagicMock:
    """Create a mock LLM client that returns a fixed JSON response."""
    mock = MagicMock()
    if isinstance(response_json, dict):
        mock.chat_json.return_value = json.dumps(response_json)
    else:
        mock.chat_json.return_value = response_json
    return mock


SAMPLE_MEMORIES_RESPONSE = {
    "memories": [
        {
            "content": "Multi-hop questions require combining evidence from multiple sources",
            "category": "procedure",
            "specificity_score": 0.8,
            "importance": 0.9,
        },
        {
            "content": "Always verify intermediate conclusions before the final answer",
            "category": "rule",
            "specificity_score": 0.7,
            "importance": 0.6,
        },
        {
            "content": "Search queries should be specific to each hop",
            "category": "insight",
            "specificity_score": 0.6,
            "importance": 0.3,
        },
    ]
}


# ============================================================
# BaseMemoryCompressor shared helpers
# ============================================================


class TestTrajectoryToText:
    """Tests for BaseMemoryCompressor._trajectory_to_text."""

    def test_basic_conversion(self):
        traj = _make_trajectory()
        text = BaseMemoryCompressor._trajectory_to_text(traj)
        assert "Task: Answer a multi-hop question about history" in text
        assert "Result: success" in text
        assert "[thought] Analysing the question" in text
        assert "[action] Search for relevant info" in text
        assert "[observation] Found key facts" in text

    def test_failure_result(self):
        traj = _make_trajectory(success=False)
        text = BaseMemoryCompressor._trajectory_to_text(traj)
        assert "Result: failure" in text

    def test_empty_steps(self):
        traj = _make_trajectory(steps=[])
        text = BaseMemoryCompressor._trajectory_to_text(traj)
        assert "Task:" in text
        assert "Result:" in text
        # No step lines — only Task + Result + trailing empty line
        lines = text.split("\n")
        # Should have Task, Result, empty separator, and no step lines
        non_empty = [l for l in lines if l.strip()]
        assert len(non_empty) == 2  # Task and Result only


class TestParseMemoryJson:
    """Tests for BaseMemoryCompressor._parse_memory_json."""

    def test_valid_json(self):
        response = json.dumps(SAMPLE_MEMORIES_RESPONSE)
        entries = BaseMemoryCompressor._parse_memory_json(response, "traj_001")
        assert len(entries) == 3
        assert entries[0].content == "Multi-hop questions require combining evidence from multiple sources"
        assert entries[0].category == "procedure"
        assert entries[0].specificity_score == 0.8
        assert entries[0].importance == 0.9
        assert entries[0].source_trajectory_id == "traj_001"

    def test_missing_optional_fields(self):
        response = json.dumps({
            "memories": [
                {"content": "bare minimum entry"}
            ]
        })
        entries = BaseMemoryCompressor._parse_memory_json(response, "traj_002")
        assert len(entries) == 1
        assert entries[0].content == "bare minimum entry"
        assert entries[0].category == "general"  # default
        assert entries[0].specificity_score == 0.5  # default
        assert entries[0].importance == 0.5  # default

    def test_invalid_json_fallback(self):
        response = "This is not valid JSON at all"
        entries = BaseMemoryCompressor._parse_memory_json(response, "traj_003")
        assert len(entries) == 1
        assert entries[0].content == response
        assert entries[0].category == "general"
        assert entries[0].source_trajectory_id == "traj_003"

    def test_empty_memories_list(self):
        response = json.dumps({"memories": []})
        entries = BaseMemoryCompressor._parse_memory_json(response, "traj_004")
        assert len(entries) == 0

    def test_custom_key(self):
        response = json.dumps({
            "results": [
                {"content": "custom key entry", "category": "fact"}
            ]
        })
        entries = BaseMemoryCompressor._parse_memory_json(
            response, "traj_005", key="results"
        )
        assert len(entries) == 1
        assert entries[0].content == "custom key entry"

    def test_extra_metadata_preserved(self):
        response = json.dumps({
            "memories": [
                {
                    "content": "entry with extras",
                    "category": "fact",
                    "specificity_score": 0.7,
                    "importance": 0.8,
                    "related_to": ["other_memory"],
                    "source": "step_3",
                }
            ]
        })
        entries = BaseMemoryCompressor._parse_memory_json(response, "traj_006")
        assert len(entries) == 1
        assert entries[0].metadata["related_to"] == ["other_memory"]
        assert entries[0].metadata["source"] == "step_3"
        # Standard fields should NOT be in metadata
        assert "content" not in entries[0].metadata
        assert "category" not in entries[0].metadata


# ============================================================
# Mem0Compressor
# ============================================================


class TestMem0Compressor:
    """Tests for Mem0Compressor with mock LLM."""

    def test_compress_basic(self):
        mock_llm = _make_mock_llm(SAMPLE_MEMORIES_RESPONSE)
        compressor = Mem0Compressor(mock_llm)
        traj = _make_trajectory()

        store = compressor.compress(traj)

        assert isinstance(store, MemoryStore)
        assert store.task_id == "test_task_001"
        assert store.framework == "mem0"
        assert store.num_entries == 3
        assert store.source_trajectory_id == traj.trajectory_id
        mock_llm.chat_json.assert_called_once()

    def test_compress_preserves_scores(self):
        mock_llm = _make_mock_llm(SAMPLE_MEMORIES_RESPONSE)
        compressor = Mem0Compressor(mock_llm)
        traj = _make_trajectory()

        store = compressor.compress(traj)

        assert store.entries[0].specificity_score == 0.8
        assert store.entries[0].importance == 0.9
        assert store.entries[1].category == "rule"

    def test_compress_with_llm_returning_invalid_json(self):
        mock_llm = _make_mock_llm("not json")
        compressor = Mem0Compressor(mock_llm)
        traj = _make_trajectory()

        store = compressor.compress(traj)

        # Fallback: single entry with raw response
        assert store.num_entries == 1
        assert store.entries[0].content == "not json"

    def test_compress_empty_trajectory(self):
        mock_llm = _make_mock_llm({"memories": [{"content": "inferred from empty", "category": "insight"}]})
        compressor = Mem0Compressor(mock_llm)
        traj = _make_trajectory(steps=[])

        store = compressor.compress(traj)

        assert store.num_entries == 1
        mock_llm.chat_json.assert_called_once()


# ============================================================
# AMEMCompressor
# ============================================================


class TestAMEMCompressor:
    """Tests for AMEMCompressor with mock LLM (two-pass)."""

    def test_compress_two_pass(self):
        # Pass 1 returns raw entries, Pass 2 returns refined entries
        raw_response = json.dumps({
            "memories": [
                {"content": "raw fact 1", "category": "fact", "specificity_score": 0.6, "importance": 0.7},
                {"content": "raw fact 2", "category": "fact", "specificity_score": 0.5, "importance": 0.5},
            ]
        })
        refined_response = json.dumps({
            "memories": [
                {
                    "content": "merged and refined fact",
                    "category": "fact",
                    "specificity_score": 0.8,
                    "importance": 0.9,
                    "related_to": ["raw fact 1", "raw fact 2"],
                },
                {
                    "content": "reflection: pattern of evidence combination",
                    "category": "reflection",
                    "specificity_score": 0.7,
                    "importance": 0.8,
                },
            ]
        })
        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [raw_response, refined_response]

        compressor = AMEMCompressor(mock_llm)
        traj = _make_trajectory()

        store = compressor.compress(traj)

        assert isinstance(store, MemoryStore)
        assert store.framework == "amem"
        assert store.num_entries == 2  # refined entries
        assert store.entries[0].content == "merged and refined fact"
        assert store.entries[1].category == "reflection"
        assert mock_llm.chat_json.call_count == 2  # two passes

    def test_compress_pass1_failure_fallback(self):
        # Pass 1 returns invalid JSON, Pass 2 gets the fallback entry
        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [
            "invalid json for pass 1",
            json.dumps({"memories": [{"content": "reflected from fallback", "category": "insight"}]}),
        ]

        compressor = AMEMCompressor(mock_llm)
        traj = _make_trajectory()

        store = compressor.compress(traj)

        assert store.num_entries >= 1
        assert mock_llm.chat_json.call_count == 2


# ============================================================
# MemoryBankCompressor
# ============================================================


class TestMemoryBankCompressor:
    """Tests for MemoryBankCompressor with mock LLM and tiering logic."""

    def test_compress_with_forgetting(self):
        mock_llm = _make_mock_llm(SAMPLE_MEMORIES_RESPONSE)
        compressor = MemoryBankCompressor(mock_llm, config={"forget_ephemeral": True})
        traj = _make_trajectory()

        store = compressor.compress(traj)

        assert store.framework == "memorybank"
        # importance: 0.9 (core), 0.6 (working), 0.3 (ephemeral -> forgotten)
        assert store.num_entries == 2  # ephemeral dropped
        tiers = [e.metadata.get("tier") for e in store.entries]
        assert "core" in tiers
        assert "working" in tiers
        assert "ephemeral" not in tiers

    def test_compress_without_forgetting(self):
        mock_llm = _make_mock_llm(SAMPLE_MEMORIES_RESPONSE)
        compressor = MemoryBankCompressor(mock_llm, config={"forget_ephemeral": False})
        traj = _make_trajectory()

        store = compressor.compress(traj)

        assert store.num_entries == 3  # all kept
        tiers = [e.metadata.get("tier") for e in store.entries]
        assert "core" in tiers
        assert "working" in tiers
        assert "ephemeral" in tiers

    def test_tier_entries_boundary_values(self):
        compressor = MemoryBankCompressor(llm_client=None)
        entries = [
            MemoryEntry(content="exactly core", importance=0.7),
            MemoryEntry(content="exactly working", importance=0.4),
            MemoryEntry(content="just below working", importance=0.39),
            MemoryEntry(content="zero importance", importance=0.0),
            MemoryEntry(content="max importance", importance=1.0),
        ]
        core, working, ephemeral = compressor._tier_entries(entries)

        assert len(core) == 2  # 0.7 and 1.0
        assert len(working) == 1  # 0.4
        assert len(ephemeral) == 2  # 0.39 and 0.0

    def test_tier_entries_empty(self):
        compressor = MemoryBankCompressor(llm_client=None)
        core, working, ephemeral = compressor._tier_entries([])
        assert core == []
        assert working == []
        assert ephemeral == []

    def test_tier_entries_all_core(self):
        compressor = MemoryBankCompressor(llm_client=None)
        entries = [
            MemoryEntry(content="high 1", importance=0.9),
            MemoryEntry(content="high 2", importance=0.8),
        ]
        core, working, ephemeral = compressor._tier_entries(entries)
        assert len(core) == 2
        assert len(working) == 0
        assert len(ephemeral) == 0

    def test_tier_entries_all_ephemeral(self):
        compressor = MemoryBankCompressor(llm_client=None)
        entries = [
            MemoryEntry(content="low 1", importance=0.1),
            MemoryEntry(content="low 2", importance=0.2),
        ]
        core, working, ephemeral = compressor._tier_entries(entries)
        assert len(core) == 0
        assert len(working) == 0
        assert len(ephemeral) == 2


# ============================================================
# Factory
# ============================================================


class TestCompressorFactory:
    """Tests for the create_compressor factory function."""

    def test_create_all_frameworks(self):
        for name, cls in [
            ("mem0", Mem0Compressor),
            ("amem", AMEMCompressor),
            ("memorybank", MemoryBankCompressor),
        ]:
            compressor = create_compressor(name, llm_client=None)
            assert isinstance(compressor, cls)

    def test_unsupported_framework(self):
        with pytest.raises(ValueError, match="Unsupported memory framework"):
            create_compressor("nonexistent", llm_client=None)

    def test_config_passed_through(self):
        compressor = create_compressor(
            "memorybank", llm_client=None, config={"forget_ephemeral": False}
        )
        assert isinstance(compressor, MemoryBankCompressor)
        assert compressor.forget_ephemeral is False
