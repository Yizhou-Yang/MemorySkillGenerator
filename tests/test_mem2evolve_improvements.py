"""
Unit tests for Mem2Evolve-inspired improvements:
- P0: MultiJudgeVerifier (external verifier, echo chamber breaker)
- P1: MemoryConsolidator (deduplication and merging)
- P2: SkillRefiner (iterative refinement and retirement)
- P3: SkillLibrary (retrieval and reuse)

All tests use mock LLM — no network required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation.multi_judge import MultiJudgeVerifier
from src.memory.consolidation import MemoryConsolidator
from src.models import (
    MemoryEntry,
    MemoryStore,
    Skill,
    StepType,
    Trajectory,
    TrajectoryStep,
    TransformVariant,
)
from src.skill_induction.skill_library import SkillLibrary
from src.skill_induction.skill_refiner import SkillRefiner


# ============================================================
# Shared fixtures
# ============================================================


def _make_trajectory(
    task_id: str = "test_task_001",
    success: bool = True,
) -> Trajectory:
    """Create a sample trajectory for testing."""
    return Trajectory(
        task_id=task_id,
        task_description="Answer a multi-hop question about history",
        success=success,
        steps=[
            TrajectoryStep(step_id=0, step_type=StepType.THOUGHT, content="Analysing the question"),
            TrajectoryStep(step_id=1, step_type=StepType.ACTION, content="Search for relevant info"),
            TrajectoryStep(step_id=2, step_type=StepType.OBSERVATION, content="Found key facts"),
            TrajectoryStep(step_id=3, step_type=StepType.THOUGHT, content="Combining evidence"),
        ],
    )


def _make_skill(
    name: str = "Test Skill",
    skill_id: str = "skill_001",
) -> Skill:
    """Create a sample skill for testing."""
    return Skill(
        skill_id=skill_id,
        name=name,
        description="A test skill for multi-hop QA",
        preconditions=["Question requires multiple reasoning steps"],
        procedure=["Step 1: Identify sub-questions", "Step 2: Search evidence", "Step 3: Combine"],
        constraints=["Do not guess without evidence"],
        facts=["Multi-hop QA requires bridging entities"],
        rules=["Prefer specific sources over general ones"],
        source_tasks=["task_001"],
        source_variant=TransformVariant.TRAJ_TO_SKILL,
    )


def _make_memory_store(num_entries: int = 5) -> MemoryStore:
    """Create a sample memory store with configurable number of entries."""
    entries = []
    for i in range(num_entries):
        entries.append(
            MemoryEntry(
                content=f"Memory entry {i}: some knowledge about topic {i}",
                category="fact" if i % 2 == 0 else "procedure",
                importance=min(0.5 + (i * 0.03), 1.0),
                specificity_score=0.6,
                source_trajectory_id="traj_001",
            )
        )
    return MemoryStore(
        task_id="test_task_001",
        framework="mem0",
        entries=entries,
        source_trajectory_id="traj_001",
    )


# ============================================================
# P0: MultiJudgeVerifier Tests
# ============================================================


class TestMultiJudgeVerifier:
    """Tests for MultiJudgeVerifier with mock LLM."""

    def test_verify_basic(self):
        """Multi-judge returns scores from all judges."""
        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [
            json.dumps({"score": 8.0, "reason": "Correct answer"}),
            json.dumps({"score": 7.0, "reason": "Good methodology"}),
            json.dumps({"score": 6.0, "reason": "Minor issues found"}),
        ]

        verifier = MultiJudgeVerifier(mock_llm)
        result = verifier.verify(
            task_description="What is 2+2?",
            expected_answer="4",
            actual_response="The answer is 4.",
            skill_name="Math Skill",
        )

        assert result["median_score"] == 7.0
        assert len(result["scores"]) == 3
        assert result["consensus"] is True  # max-min = 2.0 <= 2.0
        assert len(result["details"]) == 3

    def test_verify_no_consensus(self):
        """Judges disagree significantly -> no consensus."""
        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [
            json.dumps({"score": 9.0, "reason": "Perfect"}),
            json.dumps({"score": 3.0, "reason": "Wrong methodology"}),
            json.dumps({"score": 5.0, "reason": "Partially correct"}),
        ]

        verifier = MultiJudgeVerifier(mock_llm)
        result = verifier.verify(
            task_description="Complex question",
            expected_answer="42",
            actual_response="I think it's 42",
        )

        assert result["median_score"] == 5.0
        assert result["consensus"] is False  # max-min = 6.0 > 2.0

    def test_verify_no_llm_client(self):
        """Without LLM client, returns default values."""
        verifier = MultiJudgeVerifier(llm_client=None)
        result = verifier.verify(
            task_description="test",
            expected_answer="test",
            actual_response="test",
        )

        assert result["median_score"] == 5.0
        assert result["scores"] == []
        assert result["consensus"] is False

    def test_verify_judge_parse_error(self):
        """When a judge returns invalid JSON, fallback score is used."""
        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [
            "not valid json",
            json.dumps({"score": 7.0, "reason": "OK"}),
            json.dumps({"score": 6.0, "reason": "Fine"}),
        ]

        verifier = MultiJudgeVerifier(mock_llm)
        result = verifier.verify(
            task_description="test",
            expected_answer="answer",
            actual_response="response",
        )

        assert len(result["scores"]) == 3
        assert result["scores"][0] == 5.0  # fallback score

    def test_should_accept_skill_passes(self):
        """Skill accepted when both EM and judge criteria pass."""
        verifier = MultiJudgeVerifier(llm_client=None)

        verification_results = [
            {"median_score": 8.0},
            {"median_score": 7.0},
        ]
        em_scores = [1.0, 1.0]

        assert verifier.should_accept_skill(verification_results, em_scores) is True

    def test_should_accept_skill_low_em(self):
        """Skill rejected when EM pass rate is too low."""
        verifier = MultiJudgeVerifier(llm_client=None)

        verification_results = [
            {"median_score": 8.0},
            {"median_score": 7.0},
            {"median_score": 9.0},
        ]
        em_scores = [0.0, 0.0, 1.0]  # Only 33% pass

        assert verifier.should_accept_skill(verification_results, em_scores) is False

    def test_should_accept_skill_low_judge(self):
        """Skill rejected when judge scores are too low."""
        verifier = MultiJudgeVerifier(llm_client=None)

        verification_results = [
            {"median_score": 3.0},
            {"median_score": 4.0},
        ]
        em_scores = [1.0, 1.0]  # EM passes but judge fails

        assert verifier.should_accept_skill(verification_results, em_scores) is False

    def test_should_accept_skill_empty(self):
        """Empty scores -> rejected."""
        verifier = MultiJudgeVerifier(llm_client=None)
        assert verifier.should_accept_skill([], []) is False

    def test_custom_num_judges(self):
        """Configurable number of judges."""
        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [
            json.dumps({"score": 7.0, "reason": "OK"}),
            json.dumps({"score": 8.0, "reason": "Good"}),
        ]

        verifier = MultiJudgeVerifier(mock_llm, config={"num_judges": 2})
        result = verifier.verify(
            task_description="test",
            expected_answer="answer",
            actual_response="response",
        )

        assert len(result["scores"]) == 2
        assert mock_llm.chat_json.call_count == 2


# ============================================================
# P1: MemoryConsolidator Tests
# ============================================================


class TestMemoryConsolidator:
    """Tests for MemoryConsolidator (deduplication and merging)."""

    def test_consolidate_no_duplicates(self):
        """Entries with no similarity remain unchanged."""
        entries = [
            MemoryEntry(content="Python uses indentation for blocks", category="fact", importance=0.8),
            MemoryEntry(content="HTTP status 404 means not found", category="fact", importance=0.7),
            MemoryEntry(content="Always run tests before deploying", category="procedure", importance=0.9),
        ]
        store = MemoryStore(task_id="t1", framework="mem0", entries=entries)

        consolidator = MemoryConsolidator(llm_client=None)
        result = consolidator.consolidate(store)

        # No duplicates -> same number of entries
        assert result.num_entries == 3

    def test_consolidate_with_duplicates(self):
        """Similar entries get merged."""
        entries = [
            MemoryEntry(
                content="multi-hop questions require combining evidence from multiple sources",
                category="procedure", importance=0.8,
            ),
            MemoryEntry(
                content="multi-hop questions need combining evidence from different sources",
                category="procedure", importance=0.7,
            ),
            MemoryEntry(
                content="HTTP status 404 means not found",
                category="fact", importance=0.6,
            ),
        ]
        store = MemoryStore(task_id="t1", framework="mem0", entries=entries)

        consolidator = MemoryConsolidator(llm_client=None, config={"similarity_threshold": 0.5})
        result = consolidator.consolidate(store)

        # First two should be merged -> 2 entries total
        assert result.num_entries == 2

    def test_consolidate_too_few_entries(self):
        """With <= 2 entries, no consolidation happens."""
        entries = [
            MemoryEntry(content="single entry", category="fact", importance=0.8),
        ]
        store = MemoryStore(task_id="t1", framework="mem0", entries=entries)

        consolidator = MemoryConsolidator(llm_client=None)
        result = consolidator.consolidate(store)

        assert result.num_entries == 1
        assert result.entries[0].content == "single entry"

    def test_consolidate_preserves_metadata(self):
        """Consolidated store preserves task_id and framework."""
        store = _make_memory_store(5)
        consolidator = MemoryConsolidator(llm_client=None)
        result = consolidator.consolidate(store)

        assert result.task_id == store.task_id
        assert result.framework == store.framework

    def test_compute_similarity_identical(self):
        """Identical entries have similarity 1.0."""
        consolidator = MemoryConsolidator(llm_client=None)
        entry = MemoryEntry(content="the quick brown fox", category="fact")

        sim = consolidator.compute_similarity(entry, entry)
        assert sim == 1.0

    def test_compute_similarity_disjoint(self):
        """Completely different entries have similarity 0.0."""
        consolidator = MemoryConsolidator(llm_client=None)
        entry_a = MemoryEntry(content="python programming language", category="fact")
        entry_b = MemoryEntry(content="chocolate cake recipe", category="fact")

        sim = consolidator.compute_similarity(entry_a, entry_b)
        assert sim == 0.0

    def test_compute_similarity_partial(self):
        """Partially overlapping entries have intermediate similarity."""
        consolidator = MemoryConsolidator(llm_client=None)
        entry_a = MemoryEntry(content="multi-hop questions require evidence combination", category="fact")
        entry_b = MemoryEntry(content="multi-hop questions need evidence from multiple sources", category="fact")

        sim = consolidator.compute_similarity(entry_a, entry_b)
        assert 0.0 < sim < 1.0

    def test_compute_similarity_empty(self):
        """Empty content returns 0.0."""
        consolidator = MemoryConsolidator(llm_client=None)
        entry_a = MemoryEntry(content="", category="fact")
        entry_b = MemoryEntry(content="some content", category="fact")

        assert consolidator.compute_similarity(entry_a, entry_b) == 0.0

    def test_heuristic_merge_picks_highest_importance(self):
        """Heuristic merge uses the highest-importance entry as base."""
        entries = [
            MemoryEntry(content="low importance entry with unique info alpha beta", category="fact", importance=0.3),
            MemoryEntry(content="high importance entry", category="procedure", importance=0.9),
        ]

        consolidator = MemoryConsolidator(llm_client=None)
        merged = consolidator._heuristic_merge(entries)

        assert merged.importance == 0.9
        assert "high importance entry" in merged.content

    def test_llm_merge(self):
        """LLM merge produces a consolidated entry."""
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = json.dumps({
            "content": "Merged: multi-hop QA requires combining evidence from multiple sources",
            "category": "procedure",
            "importance": 0.9,
            "specificity_score": 0.8,
        })

        entries = [
            MemoryEntry(content="multi-hop questions require evidence", category="procedure", importance=0.8),
            MemoryEntry(content="multi-hop QA needs combining sources", category="procedure", importance=0.7),
        ]

        consolidator = MemoryConsolidator(llm_client=mock_llm)
        merged = consolidator._llm_merge(entries, "task_001")

        assert "Merged" in merged.content
        assert merged.importance == 0.9
        assert merged.metadata["merged_from"] == 2

    def test_llm_merge_fallback_on_error(self):
        """LLM merge falls back to heuristic on parse error."""
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = "not valid json"

        entries = [
            MemoryEntry(content="entry one", category="fact", importance=0.8),
            MemoryEntry(content="entry two with different unique words", category="fact", importance=0.6),
        ]

        consolidator = MemoryConsolidator(llm_client=mock_llm)
        merged = consolidator._llm_merge(entries, "task_001")

        # Should fall back to heuristic
        assert merged.importance == 0.8
        assert "entry one" in merged.content

    def test_should_consolidate(self):
        """Consolidation triggers when entries exceed threshold."""
        consolidator = MemoryConsolidator(llm_client=None)

        small_store = _make_memory_store(5)
        assert consolidator.should_consolidate(small_store, trigger_threshold=10) is False

        large_store = _make_memory_store(15)
        assert consolidator.should_consolidate(large_store, trigger_threshold=10) is True

    def test_cluster_by_similarity(self):
        """Clustering groups similar entries together."""
        entries = [
            MemoryEntry(content="the quick brown fox jumps over the lazy dog", category="fact"),
            MemoryEntry(content="the quick brown fox leaps over the lazy dog", category="fact"),
            MemoryEntry(content="python is a programming language", category="fact"),
        ]

        consolidator = MemoryConsolidator(llm_client=None, config={"similarity_threshold": 0.5})
        clusters = consolidator._cluster_by_similarity(entries)

        # First two should cluster together, third is separate
        assert len(clusters) == 2
        # One cluster has 2 entries, the other has 1
        cluster_sizes = sorted([len(c) for c in clusters])
        assert cluster_sizes == [1, 2]


# ============================================================
# P2: SkillRefiner Tests
# ============================================================


class TestSkillRefiner:
    """Tests for SkillRefiner (iterative refinement)."""

    def test_refine_no_failures(self):
        """Skill with no failures is returned unchanged."""
        refiner = SkillRefiner(llm_client=None)
        skill = _make_skill()

        validation_results = [
            {"task_description": "q1", "expected": "a1", "response": "a1", "em": 1.0, "f1": 1.0},
            {"task_description": "q2", "expected": "a2", "response": "a2", "em": 1.0, "f1": 1.0},
        ]

        refined = refiner.refine(skill, validation_results)
        assert refined.skill_id == skill.skill_id
        assert refined.version == skill.version  # No version bump

    def test_refine_with_failures(self):
        """Skill with failures gets refined via LLM."""
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = json.dumps({
            "name": "Improved Multi-hop QA",
            "description": "Better skill for multi-hop QA",
            "procedure": ["Step 1: Decompose", "Step 2: Search each", "Step 3: Verify", "Step 4: Combine"],
            "constraints": ["Do not guess", "Verify each hop"],
            "facts": ["Bridging entities connect hops"],
            "rules": ["Always verify intermediate answers"],
        })

        refiner = SkillRefiner(llm_client=mock_llm)
        skill = _make_skill()

        validation_results = [
            {"task_description": "q1", "expected": "a1", "response": "wrong", "em": 0.0, "f1": 0.2},
            {"task_description": "q2", "expected": "a2", "response": "a2", "em": 1.0, "f1": 1.0},
        ]

        refined = refiner.refine(skill, validation_results)

        assert refined.name == "Improved Multi-hop QA"
        assert refined.version == skill.version + 1
        assert len(refined.procedure) == 4
        assert "refined_from" in refined.metadata

    def test_refine_no_llm_client(self):
        """Without LLM client, returns original skill."""
        refiner = SkillRefiner(llm_client=None)
        skill = _make_skill()

        validation_results = [
            {"task_description": "q1", "expected": "a1", "response": "wrong", "em": 0.0, "f1": 0.1},
        ]

        refined = refiner.refine(skill, validation_results)
        assert refined.skill_id == skill.skill_id
        assert refined.version == skill.version

    def test_should_retire(self):
        """Retirement triggered after consecutive failures."""
        refiner = SkillRefiner(llm_client=None, config={"retirement_threshold": 3})

        assert refiner.should_retire(_make_skill(), consecutive_failures=2) is False
        assert refiner.should_retire(_make_skill(), consecutive_failures=3) is True
        assert refiner.should_retire(_make_skill(), consecutive_failures=5) is True

    def test_accumulate_basic(self):
        """Accumulation enriches an existing skill with new trajectory."""
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = json.dumps({
            "name": "Enhanced Multi-hop QA",
            "description": "Enriched with new evidence",
            "procedure": ["Step 1: Decompose", "Step 2: Search", "Step 3: Cross-verify", "Step 4: Combine"],
            "constraints": ["Do not guess", "Check source reliability"],
            "facts": ["Bridging entities connect hops", "Wikipedia is generally reliable"],
            "rules": ["Prefer specific sources"],
        })

        refiner = SkillRefiner(llm_client=mock_llm)
        skill = _make_skill()
        traj = _make_trajectory(task_id="new_task_002")

        accumulated = refiner.accumulate(skill, traj)

        assert accumulated.version == skill.version + 1
        assert "new_task_002" in accumulated.source_tasks
        assert "accumulated_from" in accumulated.metadata

    def test_accumulate_no_llm(self):
        """Without LLM, accumulation returns original skill."""
        refiner = SkillRefiner(llm_client=None)
        skill = _make_skill()
        traj = _make_trajectory()

        result = refiner.accumulate(skill, traj)
        assert result.skill_id == skill.skill_id
        assert result.version == skill.version

    def test_trajectory_to_summary(self):
        """Trajectory summary is concise and informative."""
        traj = _make_trajectory()
        summary = SkillRefiner._trajectory_to_summary(traj)

        assert "Task: Answer a multi-hop question" in summary
        assert "Result: success" in summary
        assert "[thought]" in summary
        assert "[action]" in summary


# ============================================================
# P3: SkillLibrary Tests
# ============================================================


class TestSkillLibrary:
    """Tests for SkillLibrary (retrieval and reuse)."""

    def test_add_and_get(self):
        """Add a skill and retrieve it by ID."""
        library = SkillLibrary()
        skill = _make_skill()

        library.add(skill)

        assert library.size == 1
        retrieved = library.get(skill.skill_id)
        assert retrieved is not None
        assert retrieved.name == skill.name

    def test_add_unvalidated_rejected(self):
        """Unvalidated skills are not added."""
        library = SkillLibrary()
        skill = _make_skill()

        library.add(skill, validated=False)

        assert library.size == 0

    def test_remove(self):
        """Remove a skill from the library."""
        library = SkillLibrary()
        skill = _make_skill()
        library.add(skill)

        assert library.remove(skill.skill_id) is True
        assert library.size == 0
        assert library.get(skill.skill_id) is None

    def test_remove_nonexistent(self):
        """Removing a non-existent skill returns False."""
        library = SkillLibrary()
        assert library.remove("nonexistent_id") is False

    def test_search_basic(self):
        """Search finds relevant skills."""
        library = SkillLibrary()
        skill1 = _make_skill(name="Multi-hop QA Reasoning", skill_id="s1")
        skill2 = Skill(
            skill_id="s2",
            name="Code Debugging",
            description="Debug Python code errors",
            procedure=["Read error", "Find root cause", "Fix"],
        )
        library.add(skill1)
        library.add(skill2)

        results = library.search("multi-hop question answering reasoning")

        assert len(results) >= 1
        # First result should be the QA skill (higher similarity)
        assert results[0][0].name == "Multi-hop QA Reasoning"
        assert results[0][1] > 0  # Non-zero similarity

    def test_search_empty_library(self):
        """Search on empty library returns empty list."""
        library = SkillLibrary()
        results = library.search("anything")
        assert results == []

    def test_search_empty_query(self):
        """Empty query returns empty results."""
        library = SkillLibrary()
        library.add(_make_skill())
        results = library.search("")
        assert results == []

    def test_recruit_or_create_recruit(self):
        """High similarity -> recruit existing skill."""
        library = SkillLibrary(config={"recruit_threshold": 0.1})
        skill = _make_skill(name="Multi-hop QA Reasoning")
        library.add(skill)

        # Use query with high token overlap to the skill's text
        result_skill, sim = library.recruit_or_create(
            "multi-hop QA reasoning skill for answering questions"
        )

        assert result_skill is not None
        assert result_skill.name == "Multi-hop QA Reasoning"
        assert sim >= 0.1

    def test_recruit_or_create_create(self):
        """Low similarity -> create new skill."""
        library = SkillLibrary(config={"recruit_threshold": 0.9})
        skill = _make_skill(name="Multi-hop QA Reasoning")
        library.add(skill)

        result_skill, sim = library.recruit_or_create(
            "completely unrelated topic about cooking recipes"
        )

        assert result_skill is None
        assert sim < 0.9

    def test_recruit_or_create_empty_library(self):
        """Empty library -> must create."""
        library = SkillLibrary()
        result_skill, sim = library.recruit_or_create("any task")

        assert result_skill is None
        assert sim == 0.0

    def test_record_performance(self):
        """Performance recording works correctly."""
        library = SkillLibrary()
        skill = _make_skill()
        library.add(skill)

        library.record_performance(skill.skill_id, em=1.0, f1=0.9)
        library.record_performance(skill.skill_id, em=0.0, f1=0.3)
        library.record_performance(skill.skill_id, em=1.0, f1=0.8)

        history = library.get_performance_history(skill.skill_id)
        assert len(history) == 3
        assert history[0] == (1.0, 0.9)
        assert history[1] == (0.0, 0.3)

    def test_get_consecutive_failures(self):
        """Consecutive failure counting from most recent."""
        library = SkillLibrary()
        skill = _make_skill()
        library.add(skill)

        library.record_performance(skill.skill_id, em=1.0, f1=0.9)
        library.record_performance(skill.skill_id, em=0.0, f1=0.2)
        library.record_performance(skill.skill_id, em=0.0, f1=0.1)
        library.record_performance(skill.skill_id, em=0.0, f1=0.0)

        assert library.get_consecutive_failures(skill.skill_id) == 3

    def test_get_consecutive_failures_no_failures(self):
        """No failures -> count is 0."""
        library = SkillLibrary()
        skill = _make_skill()
        library.add(skill)

        library.record_performance(skill.skill_id, em=1.0, f1=0.9)
        library.record_performance(skill.skill_id, em=1.0, f1=0.8)

        assert library.get_consecutive_failures(skill.skill_id) == 0

    def test_get_consecutive_failures_empty(self):
        """No history -> count is 0."""
        library = SkillLibrary()
        assert library.get_consecutive_failures("nonexistent") == 0

    def test_list_all(self):
        """List all skills in the library."""
        library = SkillLibrary()
        skill1 = _make_skill(name="Skill 1", skill_id="s1")
        skill2 = _make_skill(name="Skill 2", skill_id="s2")
        library.add(skill1)
        library.add(skill2)

        all_skills = library.list_all()
        assert len(all_skills) == 2
        names = {s.name for s in all_skills}
        assert "Skill 1" in names
        assert "Skill 2" in names

    def test_persistence_save_and_load(self):
        """Skills persist to disk and can be reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skills.json"

            # Save
            library1 = SkillLibrary(storage_path=path)
            skill = _make_skill()
            library1.add(skill)
            library1.record_performance(skill.skill_id, em=1.0, f1=0.9)

            assert path.exists()

            # Load
            library2 = SkillLibrary(storage_path=path)
            assert library2.size == 1
            loaded = library2.get(skill.skill_id)
            assert loaded is not None
            assert loaded.name == skill.name

    def test_persistence_nonexistent_path(self):
        """Non-existent path doesn't crash on init."""
        library = SkillLibrary(storage_path="/tmp/nonexistent_dir_xyz/skills.json")
        assert library.size == 0


# ============================================================
# Integration: P0 + P2 + P3 working together
# ============================================================


class TestIntegrationP0P2P3:
    """Integration tests: MultiJudge + SkillRefiner + SkillLibrary."""

    def test_full_lifecycle_accept(self):
        """Full lifecycle: create skill -> validate -> accept -> store."""
        # Setup
        library = SkillLibrary()
        verifier = MultiJudgeVerifier(llm_client=None)
        skill = _make_skill()

        # Simulate validation results
        em_scores = [1.0, 1.0, 0.0]  # 67% pass rate
        verification_results = [
            {"median_score": 8.0},
            {"median_score": 7.0},
            {"median_score": 6.0},
        ]

        # Decision
        accepted = verifier.should_accept_skill(verification_results, em_scores)
        assert accepted is True

        # Store
        library.add(skill, validated=accepted)
        assert library.size == 1

    def test_full_lifecycle_reject_and_refine(self):
        """Full lifecycle: create skill -> validate -> reject -> refine."""
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = json.dumps({
            "name": "Improved Skill",
            "description": "Better version",
            "procedure": ["Step 1", "Step 2", "Step 3"],
            "constraints": ["New constraint"],
            "facts": ["New fact"],
            "rules": ["New rule"],
        })

        library = SkillLibrary()
        verifier = MultiJudgeVerifier(llm_client=None)
        refiner = SkillRefiner(llm_client=mock_llm)
        skill = _make_skill()

        # Simulate poor validation
        em_scores = [0.0, 0.0, 1.0]  # Only 33% pass
        verification_results = [
            {"median_score": 3.0},
            {"median_score": 4.0},
            {"median_score": 7.0},
        ]

        # Decision: reject
        accepted = verifier.should_accept_skill(verification_results, em_scores)
        assert accepted is False

        # Refine
        validation_results = [
            {"task_description": "q1", "expected": "a1", "response": "wrong", "em": 0.0, "f1": 0.1},
            {"task_description": "q2", "expected": "a2", "response": "wrong2", "em": 0.0, "f1": 0.2},
        ]
        refined = refiner.refine(skill, validation_results)
        assert refined.version == skill.version + 1

    def test_retirement_flow(self):
        """Skill gets retired after consecutive failures."""
        library = SkillLibrary()
        refiner = SkillRefiner(llm_client=None, config={"retirement_threshold": 3})
        skill = _make_skill()
        library.add(skill)

        # Record consecutive failures
        for _ in range(3):
            library.record_performance(skill.skill_id, em=0.0, f1=0.1)

        # Check retirement
        failures = library.get_consecutive_failures(skill.skill_id)
        if refiner.should_retire(skill, failures):
            library.remove(skill.skill_id)

        assert library.size == 0
