"""Unit tests for benchmark loader and memory compressors (integration)."""

import pytest

from benchmarks.loader import BenchmarkLoader

class TestBenchmarkLoader:
    """Benchmark loader tests — loads real data from HuggingFace."""

    def test_load_hotpotqa(self):
        loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": 3})
        tasks = loader.load()
        assert len(tasks) == 3
        for task in tasks:
            assert "task_id" in task
            assert "description" in task
            assert "expected" in task
            assert task["task_id"].startswith("hotpotqa_")
            assert len(task["description"]) > 50
            assert len(task["expected"]) > 0

    def test_load_hotpotqa_hard(self):
        loader = BenchmarkLoader({"name": "hotpotqa_hard", "num_samples": 3})
        tasks = loader.load()
        assert len(tasks) >= 1  # at least 1 hard sample
        for task in tasks:
            assert task["metadata"]["level"] == "hard"

    def test_load_swebench(self):
        loader = BenchmarkLoader({"name": "swebench", "num_samples": 3})
        tasks = loader.load()
        assert len(tasks) == 3
        for task in tasks:
            assert task["task_id"].startswith("swebench_")
            assert "repo" in task["metadata"]
            assert len(task["description"]) > 50

    def test_unsupported_benchmark(self):
        loader = BenchmarkLoader({"name": "nonexistent"})
        with pytest.raises(ValueError, match="Unsupported benchmark"):
            loader.load()

class TestMemoryCompressors:
    """Memory compressor factory tests (no LLM calls)."""

    def test_create_mem0(self):
        from src.memory.compressor import create_compressor, Mem0Compressor
        # Pass a mock — we just test the factory, not the LLM call
        compressor = create_compressor("mem0", llm_client=None)
        assert isinstance(compressor, Mem0Compressor)

    def test_create_amem(self):
        from src.memory.compressor import create_compressor, AMEMCompressor
        compressor = create_compressor("amem", llm_client=None)
        assert isinstance(compressor, AMEMCompressor)

    def test_create_memorybank(self):
        from src.memory.compressor import create_compressor, MemoryBankCompressor
        compressor = create_compressor("memorybank", llm_client=None)
        assert isinstance(compressor, MemoryBankCompressor)

    def test_unsupported_framework(self):
        from src.memory.compressor import create_compressor
        with pytest.raises(ValueError, match="Unsupported memory framework"):
            create_compressor("nonexistent", llm_client=None)

    def test_memorybank_tiering(self):
        from src.memory.compressor import MemoryBankCompressor
        from src.models import MemoryEntry

        compressor = MemoryBankCompressor(llm_client=None)
        entries = [
            MemoryEntry(content="core fact", importance=0.9),
            MemoryEntry(content="working knowledge", importance=0.5),
            MemoryEntry(content="ephemeral detail", importance=0.2),
        ]
        core, working, ephemeral = compressor._tier_entries(entries)
        assert len(core) == 1
        assert len(working) == 1
        assert len(ephemeral) == 1
        assert core[0].content == "core fact"
        assert working[0].content == "working knowledge"
        assert ephemeral[0].content == "ephemeral detail"
