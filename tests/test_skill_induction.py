"""Unit tests for skill induction variants and evaluator (with mock LLM)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.evaluation.evaluator import SkillEvaluator
from src.models import (
    MemoryEntry,
    MemoryStore,
    Skill,
    SkillEvalResult,
    StepType,
    Trajectory,
    TrajectoryStep,
    TransformVariant,
)
from src.skill_induction.factory import create_inducer
from src.skill_induction.hybrid_to_skill import HybridToSkillInducer
from src.skill_induction.memory_to_skill import MemoryToSkillInducer
from src.skill_induction.traj_to_skill import TrajToSkillInducer

# Shared fixtures

def _make_trajectory() -> Trajectory:
    """Create a sample trajectory for testing."""
    return Trajectory(
        task_id="test_task_001",
        task_description="Answer a multi-hop question",
        success=True,
        steps=[
            TrajectoryStep(step_id=0, step_type=StepType.THOUGHT, content="Analysing the question"),
            TrajectoryStep(step_id=1, step_type=StepType.ACTION, content="Search for info"),
            TrajectoryStep(step_id=2, step_type=StepType.OBSERVATION, content="Found relevant facts"),
            TrajectoryStep(step_id=3, step_type=StepType.THOUGHT, content="Combining evidence"),
        ],
    )

def _make_memory_store(trajectory_id: str = "traj_001") -> MemoryStore:
    """Create a sample memory store for testing."""
    return MemoryStore(
        task_id="test_task_001",
        framework="mem0",
        entries=[
            MemoryEntry(
                content="Multi-hop questions need evidence combination",
                category="procedure",
                source_trajectory_id=trajectory_id,
                specificity_score=0.8,
                importance=0.9,
            ),
            MemoryEntry(
                content="Always verify intermediate conclusions",
                category="rule",
                source_trajectory_id=trajectory_id,
                specificity_score=0.7,
                importance=0.6,
            ),
        ],
        source_trajectory_id=trajectory_id,
    )

VALID_SKILL_RESPONSE = json.dumps({
    "name": "Multi-hop QA Reasoning",
    "description": "Skill for answering multi-hop questions by combining evidence",
    "preconditions": ["Question requires multiple reasoning steps"],
    "procedure": [
        "Step 1: Identify sub-questions",
        "Step 2: Search for evidence per sub-question",
        "Step 3: Combine evidence to form final answer",
    ],
    "constraints": ["Do not guess without evidence"],
    "facts": ["Multi-hop QA requires bridging entities"],
    "rules": ["If evidence conflicts, prefer the more specific source"],
})

# Skill Induction Factory

class TestSkillInductionFactory:
    """Tests for the skill inducer factory."""

    def test_create_traj_to_skill(self):
        inducer = create_inducer("traj_to_skill", llm_client=None)
        assert isinstance(inducer, TrajToSkillInducer)

    def test_create_memory_to_skill(self):
        inducer = create_inducer("memory_to_skill", llm_client=None)
        assert isinstance(inducer, MemoryToSkillInducer)

    def test_create_hybrid_to_skill(self):
        inducer = create_inducer("hybrid_to_skill", llm_client=None)
        assert isinstance(inducer, HybridToSkillInducer)

    def test_create_from_enum(self):
        inducer = create_inducer(TransformVariant.TRAJ_TO_SKILL, llm_client=None)
        assert isinstance(inducer, TrajToSkillInducer)

    def test_unsupported_variant(self):
        with pytest.raises(ValueError):
            create_inducer("nonexistent_variant", llm_client=None)

# Variant 1: Trajectory -> Skill

class TestTrajToSkillInducer:
    """Tests for TrajToSkillInducer with mock LLM."""

    def test_induce_basic(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = VALID_SKILL_RESPONSE

        inducer = TrajToSkillInducer(mock_llm)
        traj = _make_trajectory()
        skill = inducer.induce(trajectory=traj)

        assert isinstance(skill, Skill)
        assert skill.name == "Multi-hop QA Reasoning"
        assert skill.source_variant == TransformVariant.TRAJ_TO_SKILL
        assert "test_task_001" in skill.source_tasks
        assert len(skill.procedure) == 3
        assert len(skill.constraints) == 1
        mock_llm.chat_json.assert_called_once()

    def test_induce_invalid_json_fallback(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = "This is not valid JSON"

        inducer = TrajToSkillInducer(mock_llm)
        traj = _make_trajectory()
        skill = inducer.induce(trajectory=traj)

        assert skill.name == "Parse Error Skill"
        assert skill.source_variant == TransformVariant.TRAJ_TO_SKILL

    def test_induce_ignores_memory(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = VALID_SKILL_RESPONSE

        inducer = TrajToSkillInducer(mock_llm)
        traj = _make_trajectory()
        memory = _make_memory_store()

        # Memory is passed but should be ignored
        skill = inducer.induce(trajectory=traj, memory=memory)
        assert isinstance(skill, Skill)

# Variant 2: Memory -> Skill

class TestMemoryToSkillInducer:
    """Tests for MemoryToSkillInducer with mock LLM."""

    def test_induce_basic(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = VALID_SKILL_RESPONSE

        inducer = MemoryToSkillInducer(mock_llm)
        traj = _make_trajectory()
        memory = _make_memory_store()
        skill = inducer.induce(trajectory=traj, memory=memory)

        assert isinstance(skill, Skill)
        assert skill.name == "Multi-hop QA Reasoning"
        assert skill.source_variant == TransformVariant.MEMORY_TO_SKILL
        mock_llm.chat_json.assert_called_once()

    def test_induce_requires_memory(self):
        mock_llm = MagicMock()
        inducer = MemoryToSkillInducer(mock_llm)
        traj = _make_trajectory()

        with pytest.raises(ValueError, match="requires structured memory"):
            inducer.induce(trajectory=traj, memory=None)

    def test_induce_invalid_json_fallback(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = "not json"

        inducer = MemoryToSkillInducer(mock_llm)
        traj = _make_trajectory()
        memory = _make_memory_store()
        skill = inducer.induce(trajectory=traj, memory=memory)

        assert skill.name == "Parse Error Skill"
        assert skill.source_variant == TransformVariant.MEMORY_TO_SKILL

# Variant 3: Hybrid -> Skill

class TestHybridToSkillInducer:
    """Tests for HybridToSkillInducer with mock LLM (two LLM calls)."""

    def test_induce_basic(self):
        validation_response = json.dumps({
            "validated_memories": [
                {
                    "memory_index": 0,
                    "content": "Multi-hop questions need evidence combination",
                    "category": "procedure",
                    "evidence_strength": "strong",
                    "generalizability": "high",
                    "importance_for_similar_tasks": 0.9,
                    "reasoning": "Directly supported by trajectory steps",
                },
            ]
        })
        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [validation_response, VALID_SKILL_RESPONSE]

        inducer = HybridToSkillInducer(mock_llm)
        traj = _make_trajectory()
        memory = _make_memory_store()
        skill = inducer.induce(trajectory=traj, memory=memory)

        assert isinstance(skill, Skill)
        assert skill.name == "Multi-hop QA Reasoning"
        assert skill.source_variant == TransformVariant.HYBRID_TO_SKILL
        assert mock_llm.chat_json.call_count == 2

    def test_induce_requires_memory(self):
        mock_llm = MagicMock()
        inducer = HybridToSkillInducer(mock_llm)
        traj = _make_trajectory()

        with pytest.raises(ValueError, match="requires structured memory"):
            inducer.induce(trajectory=traj, memory=None)

    def test_induce_validation_fallback(self):
        # Validation returns invalid JSON -> fallback uses all memories
        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = ["invalid validation json", VALID_SKILL_RESPONSE]

        inducer = HybridToSkillInducer(mock_llm)
        traj = _make_trajectory()
        memory = _make_memory_store()
        skill = inducer.induce(trajectory=traj, memory=memory)

        # Should still produce a skill (validation fallback is handled)
        assert isinstance(skill, Skill)
        assert mock_llm.chat_json.call_count == 2

# Skill Evaluator

class TestSkillEvaluator:
    """Tests for SkillEvaluator with mock LLM (LLM-as-judge scoring)."""

    def _make_skill(self) -> Skill:
        return Skill(
            name="Test Skill",
            description="A test skill for evaluation",
            procedure=["Step 1", "Step 2"],
            constraints=["Avoid errors"],
            facts=["Fact A"],
            source_variant=TransformVariant.TRAJ_TO_SKILL,
        )

    def _judge_response(self, score: float) -> str:
        """Create a mock LLM-as-judge JSON response."""
        return json.dumps({"score": score, "reason": "test"})

    def _quality_response(self, score: float = 7.0) -> str:
        """Create a mock quality scoring JSON response."""
        return json.dumps({
            "specificity": score, "reusability": score,
            "structure": score, "denoising": score,
            "completeness": score, "overall": score,
            "reason": "test",
        })

    def test_evaluate_skill_success(self):
        mock_llm = MagicMock()
        # chat() for generation, chat_json() for judge + quality
        mock_llm.chat.return_value = "The answer is Yes."
        mock_llm.chat_json.side_effect = [
            self._judge_response(9.0),   # judge score
            self._quality_response(7.0), # quality score
        ]

        evaluator = SkillEvaluator(mock_llm)
        skill = self._make_skill()
        validation_tasks = [
            {"task_id": "val_001", "description": "Test question", "expected": "Yes"},
        ]

        result = evaluator.evaluate_skill(
            skill=skill,
            validation_tasks=validation_tasks,
        )

        assert isinstance(result, SkillEvalResult)
        assert result.success_rate == 0.9  # 9.0 / 10.0
        assert len(result.validation_details) == 1
        assert result.validation_details[0]["success"] is True  # score >= 7

    def test_evaluate_skill_failure(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "I don't know."
        mock_llm.chat_json.side_effect = [
            self._judge_response(2.0),   # low judge score
            self._quality_response(5.0), # quality
        ]

        evaluator = SkillEvaluator(mock_llm)
        skill = self._make_skill()
        validation_tasks = [
            {"task_id": "val_001", "description": "Test question", "expected": "Yes"},
        ]

        result = evaluator.evaluate_skill(
            skill=skill,
            validation_tasks=validation_tasks,
        )

        assert result.success_rate == 0.2  # 2.0 / 10.0
        assert result.validation_details[0]["success"] is False  # score < 7

    def test_evaluate_skill_empty_expected(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Some response"
        mock_llm.chat_json.side_effect = [
            self._judge_response(8.0),   # judge
            self._quality_response(6.0), # quality
        ]

        evaluator = SkillEvaluator(mock_llm)
        skill = self._make_skill()
        validation_tasks = [
            {"task_id": "val_001", "description": "Test question", "expected": ""},
        ]

        result = evaluator.evaluate_skill(
            skill=skill,
            validation_tasks=validation_tasks,
        )

        assert result.success_rate == 0.8  # 8.0 / 10.0

    def test_evaluate_skill_compression_ratio(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Yes"
        mock_llm.chat_json.side_effect = [
            self._judge_response(9.0),
            self._quality_response(7.0),
        ]

        evaluator = SkillEvaluator(mock_llm)
        skill = self._make_skill()
        traj = _make_trajectory()

        result = evaluator.evaluate_skill(
            skill=skill,
            validation_tasks=[{"task_id": "v1", "description": "q", "expected": "Yes"}],
            source_trajectory=traj,
        )

        assert result.compression_ratio > 0.0

    def test_evaluate_skill_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("API timeout")
        mock_llm.chat_json.side_effect = [
            self._quality_response(5.0),
        ]

        evaluator = SkillEvaluator(mock_llm)
        skill = self._make_skill()
        validation_tasks = [
            {"task_id": "val_001", "description": "Test question", "expected": "Yes"},
        ]

        result = evaluator.evaluate_skill(
            skill=skill,
            validation_tasks=validation_tasks,
        )

        assert result.success_rate == 0.0
        assert "error" in result.validation_details[0]

    def test_evaluate_skill_multiple_tasks(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            "The answer is Yes",
            "I have no idea",
        ]
        mock_llm.chat_json.side_effect = [
            self._judge_response(9.0),   # task 1 judge
            self._judge_response(2.0),   # task 2 judge
            self._quality_response(6.0), # quality
        ]

        evaluator = SkillEvaluator(mock_llm)
        skill = self._make_skill()
        validation_tasks = [
            {"task_id": "v1", "description": "q1", "expected": "Yes"},
            {"task_id": "v2", "description": "q2", "expected": "42"},
        ]

        result = evaluator.evaluate_skill(
            skill=skill,
            validation_tasks=validation_tasks,
        )

        # (9.0 + 2.0) / (2 * 10.0) = 0.55
        assert result.success_rate == 0.55
        assert len(result.validation_details) == 2

    def test_evaluate_empty_tasks(self):
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = self._quality_response(6.0)

        evaluator = SkillEvaluator(mock_llm)
        skill = self._make_skill()

        result = evaluator.evaluate_skill(
            skill=skill,
            validation_tasks=[],
        )

        assert result.success_rate == 0.0
        assert len(result.validation_details) == 0

    def test_llm_judge_fallback_on_parse_error(self):
        """When judge returns invalid JSON, fallback to substring check."""
        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = ["not json"]

        evaluator = SkillEvaluator(mock_llm)
        # Substring match: expected in response -> 7.0
        score = evaluator._llm_judge_score("q", "yes", "The answer is yes", "skill")
        assert score == 7.0

        # No match -> 3.0
        mock_llm.chat_json.side_effect = ["not json"]
        score = evaluator._llm_judge_score("q", "yes", "no match here", "skill")
        assert score == 3.0

    def test_format_skill_as_prompt(self):
        evaluator = SkillEvaluator(llm_client=None)
        skill = Skill(
            name="Test Skill",
            description="A test skill",
            preconditions=["Has context"],
            procedure=["Step 1: Do X", "Step 2: Do Y"],
            constraints=["Avoid Z"],
            facts=["Fact A"],
        )
        prompt = evaluator._format_skill_as_prompt(skill)

        assert "## Skill: Test Skill" in prompt
        assert "A test skill" in prompt
        assert "**Preconditions:**" in prompt
        assert "Has context" in prompt
        assert "**Procedure:**" in prompt
        assert "1. Step 1: Do X" in prompt
        assert "2. Step 2: Do Y" in prompt
        assert "**Constraints:**" in prompt
        assert "Avoid Z" in prompt
        assert "**Facts:**" in prompt
        assert "Fact A" in prompt

# Variant Comparison

class TestVariantComparison:
    """Tests for SkillEvaluator.compare_variants."""

    def test_compare_basic(self):
        evaluator = SkillEvaluator(llm_client=None)

        results = {
            TransformVariant.TRAJ_TO_SKILL: [
                SkillEvalResult(skill_id="s1", variant=TransformVariant.TRAJ_TO_SKILL, success_rate=0.8, compression_ratio=5.0),
                SkillEvalResult(skill_id="s2", variant=TransformVariant.TRAJ_TO_SKILL, success_rate=0.6, compression_ratio=3.0),
            ],
            TransformVariant.MEMORY_TO_SKILL: [
                SkillEvalResult(skill_id="s3", variant=TransformVariant.MEMORY_TO_SKILL, success_rate=0.9, compression_ratio=10.0),
            ],
            TransformVariant.HYBRID_TO_SKILL: [],
        }

        comparison = evaluator.compare_variants(results)

        assert "traj_to_skill" in comparison
        assert comparison["traj_to_skill"]["num_skills"] == 2
        assert comparison["traj_to_skill"]["avg_task_score"] == 0.7
        assert comparison["traj_to_skill"]["avg_compression_ratio"] == 4.0

        assert "memory_to_skill" in comparison
        assert comparison["memory_to_skill"]["num_skills"] == 1
        assert comparison["memory_to_skill"]["avg_task_score"] == 0.9

        # Empty variant should not appear
        assert "hybrid_to_skill" not in comparison

    def test_compare_empty(self):
        evaluator = SkillEvaluator(llm_client=None)
        results = {
            TransformVariant.TRAJ_TO_SKILL: [],
            TransformVariant.MEMORY_TO_SKILL: [],
            TransformVariant.HYBRID_TO_SKILL: [],
        }
        comparison = evaluator.compare_variants(results)
        assert comparison == {}
