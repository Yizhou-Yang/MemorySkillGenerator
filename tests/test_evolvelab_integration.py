"""
Tests for EvolveLab integration — verifies the adapter layer
bridges SkillForge and EvolveLab correctly.

Tests cover:
1. EvolveLab types import and instantiation
2. Outbound adapter: SkillForge compressor → EvolveLab provider
3. Inbound adapter: EvolveLab provider → SkillForge compressor
4. Provider registry and listing
5. Round-trip data conversion fidelity
"""

from __future__ import annotations

import pytest
import uuid
from unittest.mock import MagicMock, patch

from src.models import (
    MemoryEntry,
    MemoryStore,
    Trajectory,
    TrajectoryStep,
    StepType,
    Skill,
)
from src.memory.compressor import BaseMemoryCompressor, Mem0Compressor
from src.memory.consolidation import MemoryConsolidator


# ============================================================
# Test 1: EvolveLab types import correctly
# ============================================================


class TestEvolveLabImports:
    """Verify EvolveLab core types are importable and functional."""

    def test_import_base_memory(self):
        from src.memory.evolvelab.base_memory import BaseMemoryProvider
        assert BaseMemoryProvider is not None

    def test_import_memory_types(self):
        from src.memory.evolvelab.memory_types import (
            MemoryRequest,
            MemoryResponse,
            MemoryStatus,
            MemoryType,
            MemoryItem,
            MemoryItemType,
            TrajectoryData,
            PROVIDER_MAPPING,
        )
        assert MemoryStatus.BEGIN.value == "begin"
        assert MemoryStatus.IN.value == "in"
        assert len(MemoryType) >= 13  # At least 13 provider types
        assert len(PROVIDER_MAPPING) >= 13

    def test_memory_request_creation(self):
        from src.memory.evolvelab.memory_types import (
            MemoryRequest,
            MemoryStatus,
        )
        req = MemoryRequest(
            query="How to sort a list in Python?",
            context="Programming task",
            status=MemoryStatus.BEGIN,
        )
        assert req.query == "How to sort a list in Python?"
        assert req.status == MemoryStatus.BEGIN

    def test_memory_response_creation(self):
        from src.memory.evolvelab.memory_types import (
            MemoryResponse,
            MemoryType,
            MemoryItem,
            MemoryItemType,
        )
        item = MemoryItem(
            id="test-1",
            content="Use list.sort() for in-place sorting",
            metadata={"source": "test"},
            score=0.95,
            type=MemoryItemType.TEXT,
        )
        resp = MemoryResponse(
            memories=[item],
            memory_type=MemoryType.VOYAGER,
            total_count=1,
        )
        assert resp.total_count == 1
        assert resp.memories[0].score == 0.95

    def test_trajectory_data_creation(self):
        from src.memory.evolvelab.memory_types import TrajectoryData
        td = TrajectoryData(
            query="Find the capital of France",
            trajectory=[
                {"type": "thought", "content": "I need to search for France"},
                {"type": "action", "content": "search('capital of France')"},
                {"type": "observation", "content": "Paris is the capital"},
            ],
            result="Paris",
            metadata={"task_id": "test-001"},
        )
        assert len(td.trajectory) == 3
        assert td.result == "Paris"

    def test_memory_type_enum_values(self):
        from src.memory.evolvelab.memory_types import MemoryType
        expected_types = [
            "agent_kb", "skillweaver", "mobilee", "expel",
            "lightweight_memory", "cerebra_fusion_memory", "voyager",
            "dilu", "generative", "memp", "dynamic_cheatsheet",
            "agent_workflow_memory", "evolver",
        ]
        actual_values = [mt.value for mt in MemoryType]
        for expected in expected_types:
            assert expected in actual_values, f"Missing MemoryType: {expected}"

    def test_config_module(self):
        from src.memory.evolvelab.config import (
            get_memory_config,
            get_evolve_lab_config,
        )
        from src.memory.evolvelab.memory_types import MemoryType

        lab_config = get_evolve_lab_config()
        assert "default_top_k" in lab_config
        assert lab_config["default_top_k"] == 3

        voyager_config = get_memory_config(MemoryType.VOYAGER)
        assert "db_path" in voyager_config


# ============================================================
# Test 2: Outbound Adapter (SkillForge → EvolveLab)
# ============================================================


def _make_test_trajectory() -> Trajectory:
    """Create a test trajectory for adapter tests."""
    return Trajectory(
        task_id="test-task-001",
        task_description="What is the capital of France?",
        steps=[
            TrajectoryStep(
                step_id=0,
                step_type=StepType.THOUGHT,
                content="I need to find the capital of France",
            ),
            TrajectoryStep(
                step_id=1,
                step_type=StepType.ACTION,
                content="search('capital of France')",
            ),
            TrajectoryStep(
                step_id=2,
                step_type=StepType.OBSERVATION,
                content="Paris is the capital of France",
            ),
        ],
        success=True,
        final_answer="Paris",
    )


def _make_test_memory_store() -> MemoryStore:
    """Create a test MemoryStore."""
    return MemoryStore(
        task_id="test-task-001",
        framework="mem0",
        entries=[
            MemoryEntry(
                content="Paris is the capital of France",
                category="fact",
                importance=0.9,
                specificity_score=0.8,
            ),
            MemoryEntry(
                content="Use search tool for geographic questions",
                category="procedure",
                importance=0.7,
                specificity_score=0.6,
            ),
        ],
    )


class TestOutboundAdapter:
    """Test SkillForge → EvolveLab adapter."""

    def test_adapter_initialization(self):
        from src.memory.evolvelab_adapter import SkillForgeAsEvolveLabProvider
        from src.memory.evolvelab.memory_types import MemoryType

        mock_compressor = MagicMock(spec=BaseMemoryCompressor)
        adapter = SkillForgeAsEvolveLabProvider(
            compressor=mock_compressor,
            memory_type=MemoryType.AGENT_KB,
        )
        assert adapter.initialize() is True
        assert adapter.num_memories == 0

    def test_take_in_memory(self):
        from src.memory.evolvelab_adapter import SkillForgeAsEvolveLabProvider
        from src.memory.evolvelab.memory_types import (
            MemoryType,
            TrajectoryData,
        )

        mock_compressor = MagicMock(spec=BaseMemoryCompressor)
        mock_compressor.compress.return_value = _make_test_memory_store()

        adapter = SkillForgeAsEvolveLabProvider(
            compressor=mock_compressor,
            memory_type=MemoryType.AGENT_KB,
        )
        adapter.initialize()

        traj_data = TrajectoryData(
            query="What is the capital of France?",
            trajectory=[
                {"type": "thought", "content": "I need to find the capital"},
                {"type": "action", "content": "search('capital of France')"},
            ],
            result="Paris",
            metadata={"task_id": "test-001"},
        )

        success, description = adapter.take_in_memory(traj_data)
        assert success is True
        assert adapter.num_memories == 2
        mock_compressor.compress.assert_called_once()

    def test_take_in_memory_with_consolidation(self):
        from src.memory.evolvelab_adapter import SkillForgeAsEvolveLabProvider
        from src.memory.evolvelab.memory_types import (
            MemoryType,
            TrajectoryData,
        )

        mock_compressor = MagicMock(spec=BaseMemoryCompressor)
        store = _make_test_memory_store()
        mock_compressor.compress.return_value = store

        mock_consolidator = MagicMock(spec=MemoryConsolidator)
        mock_consolidator.should_consolidate.return_value = True
        # Consolidation reduces entries
        consolidated = MemoryStore(
            task_id="test-task-001",
            framework="mem0",
            entries=[store.entries[0]],  # Only keep first entry
        )
        mock_consolidator.consolidate.return_value = consolidated

        adapter = SkillForgeAsEvolveLabProvider(
            compressor=mock_compressor,
            consolidator=mock_consolidator,
            memory_type=MemoryType.AGENT_KB,
        )
        adapter.initialize()

        traj_data = TrajectoryData(
            query="Test query",
            trajectory=[{"type": "action", "content": "test"}],
            result="test",
            metadata={"task_id": "test"},
        )

        success, _ = adapter.take_in_memory(traj_data)
        assert success is True
        assert adapter.num_memories == 1  # Consolidated to 1
        mock_consolidator.consolidate.assert_called_once()

    def test_provide_memory(self):
        from src.memory.evolvelab_adapter import SkillForgeAsEvolveLabProvider
        from src.memory.evolvelab.memory_types import (
            MemoryType,
            MemoryRequest,
            MemoryStatus,
            TrajectoryData,
        )

        mock_compressor = MagicMock(spec=BaseMemoryCompressor)
        mock_compressor.compress.return_value = _make_test_memory_store()

        adapter = SkillForgeAsEvolveLabProvider(
            compressor=mock_compressor,
            memory_type=MemoryType.AGENT_KB,
            config={"top_k": 2},
        )
        adapter.initialize()

        # First ingest some memories
        traj_data = TrajectoryData(
            query="What is the capital of France?",
            trajectory=[{"type": "action", "content": "search"}],
            result="Paris",
            metadata={"task_id": "test"},
        )
        adapter.take_in_memory(traj_data)

        # Now retrieve
        request = MemoryRequest(
            query="capital of France",
            context="",
            status=MemoryStatus.BEGIN,
        )
        response = adapter.provide_memory(request)

        assert response.total_count > 0
        assert response.memory_type == MemoryType.AGENT_KB
        # The "Paris is the capital of France" entry should rank highest
        assert any("capital" in m.content.lower() or "paris" in m.content.lower()
                    for m in response.memories)

    def test_provide_memory_empty(self):
        from src.memory.evolvelab_adapter import SkillForgeAsEvolveLabProvider
        from src.memory.evolvelab.memory_types import (
            MemoryType,
            MemoryRequest,
            MemoryStatus,
        )

        mock_compressor = MagicMock(spec=BaseMemoryCompressor)
        adapter = SkillForgeAsEvolveLabProvider(
            compressor=mock_compressor,
            memory_type=MemoryType.AGENT_KB,
        )
        adapter.initialize()

        request = MemoryRequest(
            query="anything",
            context="",
            status=MemoryStatus.BEGIN,
        )
        response = adapter.provide_memory(request)
        assert response.total_count == 0

    def test_clear(self):
        from src.memory.evolvelab_adapter import SkillForgeAsEvolveLabProvider
        from src.memory.evolvelab.memory_types import (
            MemoryType,
            TrajectoryData,
        )

        mock_compressor = MagicMock(spec=BaseMemoryCompressor)
        mock_compressor.compress.return_value = _make_test_memory_store()

        adapter = SkillForgeAsEvolveLabProvider(
            compressor=mock_compressor,
            memory_type=MemoryType.AGENT_KB,
        )
        adapter.initialize()

        traj_data = TrajectoryData(
            query="test",
            trajectory=[{"type": "action", "content": "test"}],
            result="test",
            metadata={},
        )
        adapter.take_in_memory(traj_data)
        assert adapter.num_memories > 0

        adapter.clear()
        assert adapter.num_memories == 0


# ============================================================
# Test 3: Inbound Adapter (EvolveLab → SkillForge)
# ============================================================


class TestInboundAdapter:
    """Test EvolveLab → SkillForge adapter."""

    def test_compress_with_mock_provider(self):
        from src.memory.evolvelab_adapter import EvolveLabAsSkillForgeCompressor
        from src.memory.evolvelab.base_memory import BaseMemoryProvider as EvolveLabBaseProvider
        from src.memory.evolvelab.memory_types import (
            MemoryType,
            MemoryResponse,
            MemoryItem,
            MemoryItemType,
        )

        mock_provider = MagicMock(spec=EvolveLabBaseProvider)
        mock_provider.get_memory_type.return_value = MemoryType.VOYAGER
        mock_provider.take_in_memory.return_value = (True, "Memory stored")
        mock_provider.provide_memory.return_value = MemoryResponse(
            memories=[
                MemoryItem(
                    id="mem-1",
                    content="Paris is the capital of France",
                    metadata={"source": "voyager"},
                    score=0.9,
                    type=MemoryItemType.TEXT,
                ),
            ],
            memory_type=MemoryType.VOYAGER,
            total_count=1,
        )

        adapter = EvolveLabAsSkillForgeCompressor(provider=mock_provider)
        trajectory = _make_test_trajectory()

        store = adapter.compress(trajectory)

        assert store.num_entries == 1
        assert store.framework == "evolvelab_voyager"
        assert "Paris" in store.entries[0].content
        mock_provider.take_in_memory.assert_called_once()
        mock_provider.provide_memory.assert_called_once()

    def test_compress_with_failed_ingestion(self):
        from src.memory.evolvelab_adapter import EvolveLabAsSkillForgeCompressor
        from src.memory.evolvelab.base_memory import BaseMemoryProvider as EvolveLabBaseProvider
        from src.memory.evolvelab.memory_types import MemoryType

        mock_provider = MagicMock(spec=EvolveLabBaseProvider)
        mock_provider.get_memory_type.return_value = MemoryType.EXPEL
        mock_provider.take_in_memory.return_value = (False, "Error occurred")

        adapter = EvolveLabAsSkillForgeCompressor(provider=mock_provider)
        trajectory = _make_test_trajectory()

        store = adapter.compress(trajectory)

        assert store.num_entries == 0
        assert store.framework == "evolvelab_expel"

    def test_trajectory_conversion_roundtrip(self):
        """Verify data fidelity in SkillForge → EvolveLab → SkillForge conversion."""
        from src.memory.evolvelab_adapter import (
            SkillForgeAsEvolveLabProvider,
            EvolveLabAsSkillForgeCompressor,
        )

        original = _make_test_trajectory()

        # SkillForge → EvolveLab
        evolvelab_data = EvolveLabAsSkillForgeCompressor._convert_to_evolvelab(original)
        assert evolvelab_data.query == original.task_description
        assert len(evolvelab_data.trajectory) == len(original.steps)
        assert evolvelab_data.result == original.final_answer

        # EvolveLab → SkillForge
        roundtrip = SkillForgeAsEvolveLabProvider._convert_trajectory(evolvelab_data)
        assert roundtrip.task_description == original.task_description
        assert len(roundtrip.steps) == len(original.steps)
        assert roundtrip.final_answer == original.final_answer

        # Verify step content preserved
        for orig_step, rt_step in zip(original.steps, roundtrip.steps):
            assert rt_step.content == orig_step.content
            assert rt_step.step_type == orig_step.step_type


# ============================================================
# Test 4: Provider Registry
# ============================================================


class TestProviderRegistry:
    """Test provider listing and info."""

    def test_list_available_providers(self):
        from src.memory.evolvelab_adapter import list_available_providers
        providers = list_available_providers()
        assert len(providers) == 13
        assert "voyager" in providers
        assert "expel" in providers
        assert "skillweaver" in providers
        assert "evolver" in providers

    def test_get_provider_info(self):
        from src.memory.evolvelab_adapter import get_provider_info
        info = get_provider_info()
        assert len(info) >= 13

        # Check structure
        for item in info:
            assert "type" in item
            assert "class" in item
            assert "module" in item

        # Check specific providers
        types = [i["type"] for i in info]
        assert "voyager" in types
        assert "expel" in types


# ============================================================
# Test 5: Integration with SkillForge RL Controller
# ============================================================


class TestRLControllerIntegration:
    """Test that EvolveLab adapter works with RL Controller."""

    def test_adapter_memories_usable_by_controller(self):
        """Verify adapter-produced memories can be used in RL controller context."""
        from src.memory.evolvelab_adapter import SkillForgeAsEvolveLabProvider
        from src.memory.evolvelab.memory_types import (
            MemoryType,
            MemoryRequest,
            MemoryStatus,
            TrajectoryData,
        )

        mock_compressor = MagicMock(spec=BaseMemoryCompressor)
        mock_compressor.compress.return_value = _make_test_memory_store()

        adapter = SkillForgeAsEvolveLabProvider(
            compressor=mock_compressor,
            memory_type=MemoryType.AGENT_KB,
        )
        adapter.initialize()

        # Ingest
        traj_data = TrajectoryData(
            query="What is the capital of France?",
            trajectory=[{"type": "action", "content": "search"}],
            result="Paris",
            metadata={"task_id": "test"},
        )
        adapter.take_in_memory(traj_data)

        # Retrieve
        request = MemoryRequest(
            query="France capital city",
            context="",
            status=MemoryStatus.BEGIN,
        )
        response = adapter.provide_memory(request)

        # Verify memories have the structure needed for RL controller
        for mem in response.memories:
            assert isinstance(mem.id, str)
            assert isinstance(mem.content, str)
            assert mem.score is not None
            assert isinstance(mem.metadata, dict)


# ============================================================
# Test 6: Benchmark Integration Smoke Test
# ============================================================


class TestBenchmarkIntegration:
    """Smoke test: adapter works in a mini benchmark loop."""

    def test_mini_benchmark_loop(self):
        """Simulate a tiny benchmark: ingest 3 trajectories, retrieve for 2 queries."""
        from src.memory.evolvelab_adapter import SkillForgeAsEvolveLabProvider
        from src.memory.evolvelab.memory_types import (
            MemoryType,
            MemoryRequest,
            MemoryStatus,
            TrajectoryData,
        )

        mock_compressor = MagicMock(spec=BaseMemoryCompressor)

        # Each call returns a different store
        stores = [
            MemoryStore(
                task_id=f"task-{i}",
                framework="mem0",
                entries=[
                    MemoryEntry(
                        content=content,
                        category="fact",
                        importance=0.8,
                    )
                ],
            )
            for i, content in enumerate([
                "Python lists support sort() and sorted() methods",
                "HTTP 404 means resource not found on the server",
                "Binary search requires a sorted array as input",
            ])
        ]
        mock_compressor.compress.side_effect = stores

        adapter = SkillForgeAsEvolveLabProvider(
            compressor=mock_compressor,
            memory_type=MemoryType.AGENT_KB,
            config={"top_k": 2},
        )
        adapter.initialize()

        # Ingest 3 trajectories
        for i in range(3):
            traj_data = TrajectoryData(
                query=f"Task {i}",
                trajectory=[{"type": "action", "content": f"step {i}"}],
                result=f"result {i}",
                metadata={"task_id": f"task-{i}"},
            )
            success, _ = adapter.take_in_memory(traj_data)
            assert success

        assert adapter.num_memories == 3

        # Query 1: should match Python/sort
        req1 = MemoryRequest(
            query="How to sort a list in Python?",
            context="",
            status=MemoryStatus.BEGIN,
        )
        resp1 = adapter.provide_memory(req1)
        assert resp1.total_count > 0
        # The Python sort entry should be among top results
        contents = [m.content for m in resp1.memories]
        assert any("sort" in c.lower() for c in contents)

        # Query 2: should match HTTP/404
        req2 = MemoryRequest(
            query="What does HTTP 404 error mean?",
            context="",
            status=MemoryStatus.BEGIN,
        )
        resp2 = adapter.provide_memory(req2)
        assert resp2.total_count > 0
