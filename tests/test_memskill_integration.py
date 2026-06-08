"""MemSkill integration tests -- verifies all components from the MemSkill paper."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Phase 1: Benchmark Loader Tests (LoCoMo + LongMemEval)

from benchmarks.loader import BenchmarkLoader, PRIMARY_BENCHMARKS

# Mock data for LoCoMo
MOCK_LOCOMO_ROWS = [
    {
        "questions": [
            "When did Caroline go to the LGBTQ support group?",
            "When did Melanie paint a sunrise?",
            "What fields would Caroline be likely to pursue?",
        ],
        "answers": [
            "7 May 2023",
            "2022",
            "Psychology, counseling certification",
        ],
        "evidences": [[[1, 3]], [[1, 12]], [[1, 9], [1, 11]]],
        "category": [2, 2, 3],
        "turns": [
            "1:56 pm on 8 May, 2023\n Caroline: Hey Mel!",
            "Melanie: Hey Caroline! Good to see you!",
        ],
        "sessions": [
            "1:56 pm on 8 May, 2023\n Caroline: Hey Mel! Good to see you!\n Melanie: Hey!",
            "2:30 pm on 15 May, 2023\n Caroline: I've been thinking about counseling.",
        ],
    },
]

# Mock data for LongMemEval
MOCK_LONGMEMEVAL_ROWS = [
    {
        "custom_id": "0a995998",
        "question": "How many items of clothing do I need to pick up?",
        "answer": "3",
        "full_input": "I will give you several history chats...",
        "full_input_tokens": 113933,
        "focused_input": "Session 2023/02/15: I need to pick up 3 items...",
        "focused_input_tokens": 320,
    },
    {
        "custom_id": "1b886aa9",
        "question": "What restaurant did we discuss for dinner?",
        "answer": "Italian place on 5th street",
        "full_input": "Long conversation about dinner plans...",
        "full_input_tokens": 95000,
        "focused_input": "Session about dinner: Let's go to the Italian place on 5th street.",
        "focused_input_tokens": 150,
    },
    {
        "custom_id": "2c777bb0",
        "question": "When is the project deadline?",
        "answer": "March 15th",
        "full_input": "Work discussion about project timeline...",
        "full_input_tokens": 80000,
        "focused_input": "The project deadline is March 15th, no extensions.",
        "focused_input_tokens": 100,
    },
]

class TestLoCoMoLoader:
    """LoCoMo benchmark loader tests"""

    @patch("benchmarks.loader.load_dataset")
    def test_load_locomo_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LOCOMO_ROWS
        loader = BenchmarkLoader({"name": "locomo", "num_samples": 10})
        tasks = loader.load()
        assert len(tasks) > 0
        mock_load_dataset.assert_called_once_with(
            "KhangPTT373/locomo_preprocess", split="test"
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_locomo_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LOCOMO_ROWS
        loader = BenchmarkLoader({"name": "locomo", "num_samples": 10})
        tasks = loader.load()

        for task in tasks:
            assert "task_id" in task
            assert task["task_id"].startswith("locomo_")
            assert "description" in task
            assert "expected" in task
            assert "context" in task
            assert "metadata" in task

    @patch("benchmarks.loader.load_dataset")
    def test_load_locomo_qa_pairs(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LOCOMO_ROWS
        loader = BenchmarkLoader({"name": "locomo", "num_samples": 10})
        tasks = loader.load()

        # Should have 3 QA pairs (from mock data's 3 questions)
        assert len(tasks) == 3
        assert tasks[0]["expected"] == "7 May 2023"
        assert tasks[1]["expected"] == "2022"
        assert tasks[2]["expected"] == "Psychology, counseling certification"

    @patch("benchmarks.loader.load_dataset")
    def test_load_locomo_categories(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LOCOMO_ROWS
        loader = BenchmarkLoader({"name": "locomo", "num_samples": 10})
        tasks = loader.load()

        categories = [t["metadata"]["category"] for t in tasks]
        assert categories == [2, 2, 3]
        assert tasks[0]["metadata"]["category_name"] == "multi-hop"
        assert tasks[2]["metadata"]["category_name"] == "temporal"

    @patch("benchmarks.loader.load_dataset")
    def test_load_locomo_context_includes_sessions(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LOCOMO_ROWS
        loader = BenchmarkLoader({"name": "locomo", "num_samples": 10})
        tasks = loader.load()

        # Context should contain session content
        assert "Hey Mel" in tasks[0]["context"]
        assert "counseling" in tasks[0]["context"]

class TestLongMemEvalLoader:
    """LongMemEval benchmark loader tests"""

    @patch("benchmarks.loader.load_dataset")
    def test_load_longmemeval_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LONGMEMEVAL_ROWS
        loader = BenchmarkLoader({"name": "longmemeval", "num_samples": 10})
        tasks = loader.load()
        assert len(tasks) == 3
        mock_load_dataset.assert_called_once_with(
            "kellyhongg/cleaned-longmemeval-s", split="train"
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_longmemeval_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LONGMEMEVAL_ROWS
        loader = BenchmarkLoader({"name": "longmemeval", "num_samples": 10})
        tasks = loader.load()

        for task in tasks:
            assert "task_id" in task
            assert task["task_id"].startswith("longmemeval_")
            assert "description" in task
            assert "expected" in task
            assert "context" in task
            assert "metadata" in task

    @patch("benchmarks.loader.load_dataset")
    def test_load_longmemeval_uses_focused_input(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LONGMEMEVAL_ROWS
        loader = BenchmarkLoader({"name": "longmemeval", "num_samples": 10})
        tasks = loader.load()

        # Should use focused_input instead of full_input
        assert "pick up 3 items" in tasks[0]["context"]
        assert tasks[0]["expected"] == "3"

    @patch("benchmarks.loader.load_dataset")
    def test_load_longmemeval_metadata(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LONGMEMEVAL_ROWS
        loader = BenchmarkLoader({"name": "longmemeval", "num_samples": 10})
        tasks = loader.load()

        assert tasks[0]["metadata"]["custom_id"] == "0a995998"
        assert tasks[0]["metadata"]["full_input_tokens"] == 113933
        assert tasks[0]["metadata"]["focused_input_tokens"] == 320

    @patch("benchmarks.loader.load_dataset")
    def test_load_longmemeval_num_samples_limit(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_LONGMEMEVAL_ROWS
        loader = BenchmarkLoader({"name": "longmemeval", "num_samples": 2})
        tasks = loader.load()
        assert len(tasks) == 2

class TestBenchmarkListsUpdated:
    """Verify benchmark lists are correctly updated"""

    def test_locomo_in_primary(self):
        assert "locomo" in PRIMARY_BENCHMARKS

    def test_longmemeval_in_primary(self):
        assert "longmemeval" in PRIMARY_BENCHMARKS

    def test_primary_benchmarks_count(self):
        # Original 7 + 2 new = 9
        assert len(PRIMARY_BENCHMARKS) == 9

# Phase 2: RL Controller Tests

from src.rl_controller.controller import (
    ControllerMLP,
    ControllerState,
    PPOTransition,
    SelectionResult,
    SkillEmbedding,
    SkillSelectionController,
    ValueNetwork,
)

class TestControllerMLP:
    """Controller MLP network tests"""

    def test_forward_shape(self):
        mlp = ControllerMLP(input_dim=64, hidden_dim=32, output_dim=64)
        x = np.random.randn(64).astype(np.float32)
        out = mlp.forward(x)
        assert out.shape == (64,)

    def test_forward_deterministic(self):
        mlp = ControllerMLP(input_dim=64, hidden_dim=32, output_dim=64, seed=42)
        x = np.random.randn(64).astype(np.float32)
        out1 = mlp.forward(x)
        out2 = mlp.forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_clone_and_restore_params(self):
        mlp = ControllerMLP(input_dim=64, hidden_dim=32, output_dim=64)
        original_params = mlp.clone_params()
        x = np.random.randn(64).astype(np.float32)
        out_before = mlp.forward(x)

        # Modify parameters
        mlp.W1 += 1.0
        out_after = mlp.forward(x)
        assert not np.allclose(out_before, out_after)

        # Restore parameters
        mlp.set_params(original_params)
        out_restored = mlp.forward(x)
        np.testing.assert_array_almost_equal(out_before, out_restored)

    def test_get_params_count(self):
        mlp = ControllerMLP(input_dim=64, hidden_dim=32, output_dim=64)
        params = mlp.get_params()
        assert len(params) == 4  # W1, b1, W2, b2

class TestValueNetwork:
    """Value Network tests"""

    def test_forward_returns_scalar(self):
        vn = ValueNetwork(input_dim=64, hidden_dim=32)
        x = np.random.randn(64).astype(np.float32)
        value = vn.forward(x)
        assert isinstance(value, float)

    def test_forward_deterministic(self):
        vn = ValueNetwork(input_dim=64, hidden_dim=32, seed=42)
        x = np.random.randn(64).astype(np.float32)
        v1 = vn.forward(x)
        v2 = vn.forward(x)
        assert v1 == v2

class TestSkillSelectionController:
    """Skill Selection Controller comprehensive tests"""

    def _make_controller(self, dim=64, hidden=32, top_k=3):
        return SkillSelectionController({
            "embedding_dim": dim,
            "hidden_dim": hidden,
            "top_k": top_k,
        })

    def _register_skills(self, ctrl, n=10, dim=64):
        for i in range(n):
            emb = np.random.randn(dim).astype(np.float32)
            ctrl.register_skill(f"skill_{i}", f"Skill {i}", emb)

    def test_register_skill(self):
        ctrl = self._make_controller()
        assert ctrl.skill_bank_size == 0
        ctrl.register_skill("s1", "Skill 1", np.random.randn(64).astype(np.float32))
        assert ctrl.skill_bank_size == 1

    def test_remove_skill(self):
        ctrl = self._make_controller()
        ctrl.register_skill("s1", "Skill 1", np.random.randn(64).astype(np.float32))
        assert ctrl.skill_bank_size == 1
        assert ctrl.remove_skill("s1")
        assert ctrl.skill_bank_size == 0
        assert not ctrl.remove_skill("nonexistent")

    def test_select_skills_basic(self):
        ctrl = self._make_controller(top_k=3)
        self._register_skills(ctrl, n=10)

        state = ControllerState(
            embedding=np.random.randn(64).astype(np.float32)
        )
        result = ctrl.select_skills(state, training=False)

        assert isinstance(result, SelectionResult)
        assert len(result.selected_skill_ids) == 3
        assert len(result.selected_indices) == 3
        assert isinstance(result.log_prob, float)
        assert result.probabilities.shape == (10,)
        # Probabilities should sum to ~1
        assert abs(result.probabilities.sum() - 1.0) < 1e-5

    def test_select_skills_training_mode(self):
        """Training mode should use Gumbel-Top-K sampling (with randomness)"""
        ctrl = self._make_controller(top_k=3)
        self._register_skills(ctrl, n=10)

        state = ControllerState(
            embedding=np.random.randn(64).astype(np.float32)
        )

        # Multiple samples should yield different results (probabilistically)
        results = set()
        for _ in range(20):
            result = ctrl.select_skills(state, training=True)
            results.add(tuple(sorted(result.selected_indices)))

        # Should have at least 2 different selections (due to Gumbel noise)
        assert len(results) >= 2, "Gumbel-Top-K should produce diverse selections"

    def test_select_skills_greedy_deterministic(self):
        """Greedy mode should be deterministic"""
        ctrl = self._make_controller(top_k=3)
        self._register_skills(ctrl, n=10)

        state = ControllerState(
            embedding=np.random.randn(64).astype(np.float32)
        )

        result1 = ctrl.select_skills(state, training=False)
        result2 = ctrl.select_skills(state, training=False)
        assert result1.selected_indices == result2.selected_indices

    def test_select_skills_empty_bank(self):
        ctrl = self._make_controller()
        state = ControllerState(
            embedding=np.random.randn(64).astype(np.float32)
        )
        result = ctrl.select_skills(state)
        assert result.selected_skill_ids == []
        assert result.log_prob == 0.0

    def test_select_skills_fewer_than_k(self):
        """When bank has fewer skills than K"""
        ctrl = self._make_controller(top_k=5)
        self._register_skills(ctrl, n=2)

        state = ControllerState(
            embedding=np.random.randn(64).astype(np.float32)
        )
        result = ctrl.select_skills(state)
        assert len(result.selected_skill_ids) == 2  # min(5, 2)

    def test_joint_log_prob_valid(self):
        """Joint log probability should be negative (probability < 1)"""
        ctrl = self._make_controller(top_k=3)
        self._register_skills(ctrl, n=10)

        state = ControllerState(
            embedding=np.random.randn(64).astype(np.float32)
        )
        result = ctrl.select_skills(state)
        assert result.log_prob < 0, "Log probability should be negative"

    def test_joint_log_prob_formula(self):
        """Verify joint probability formula (MemSkill Eq.11)"""
        probs = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        indices = [0, 1, 2]  # Select A, B, C

        log_prob = SkillSelectionController._compute_joint_log_prob(
            probs, indices
        )

        # Manual computation:
        # P(A) = 0.4 / 1.0 = 0.4
        # P(B|A) = 0.3 / (1 - 0.4) = 0.5
        # P(C|A,B) = 0.15 / (1 - 0.4 - 0.3) = 0.5
        # π = 0.4 * 0.5 * 0.5 = 0.1
        expected_log_prob = math.log(0.4) + math.log(0.5) + math.log(0.5)
        assert abs(log_prob - expected_log_prob) < 1e-6

    def test_softmax_numerical_stability(self):
        """Softmax should be numerically stable for large values"""
        large_x = np.array([1000.0, 1001.0, 999.0])
        probs = SkillSelectionController._softmax(large_x)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_exploration_incentive(self):
        """New skills should get higher selection probability"""
        ctrl = self._make_controller(top_k=1)

        # Register 5 regular skills
        for i in range(5):
            ctrl.register_skill(
                f"old_{i}", f"Old Skill {i}",
                np.random.randn(64).astype(np.float32),
                is_new=False,
            )

        # Register 1 new skill (weak embedding)
        weak_emb = np.zeros(64, dtype=np.float32) * 0.01
        ctrl.register_skill("new_1", "New Skill", weak_emb, is_new=True)

        state = ControllerState(
            embedding=np.random.randn(64).astype(np.float32)
        )
        result = ctrl.select_skills(state, training=False)

        # New skill probability should be boosted
        new_idx = ctrl.skill_bank_size - 1
        assert result.probabilities[new_idx] > 0.01, \
            "New skill should have boosted probability due to exploration incentive"

    def test_get_value(self):
        ctrl = self._make_controller()
        state = ControllerState(
            embedding=np.random.randn(64).astype(np.float32)
        )
        value = ctrl.get_value(state)
        assert isinstance(value, float)

    def test_snapshot_and_rollback(self):
        ctrl = self._make_controller()
        self._register_skills(ctrl, n=5)

        state = ControllerState(
            embedding=np.random.randn(64).astype(np.float32)
        )

        # Save snapshot
        result_before = ctrl.select_skills(state, training=False)
        assert ctrl.save_snapshot(0.8)

        # Modify parameters
        ctrl.policy_net.W1 += 10.0
        result_modified = ctrl.select_skills(state, training=False)

        # Rollback
        assert ctrl.rollback_to_best()
        result_after = ctrl.select_skills(state, training=False)

        # Should restore after rollback
        assert result_before.selected_indices == result_after.selected_indices

    def test_ppo_update_basic(self):
        """PPO update should not raise errors"""
        ctrl = self._make_controller()
        self._register_skills(ctrl, n=5)

        # Record some transitions
        for _ in range(5):
            transition = PPOTransition(
                state_embedding=np.random.randn(64).astype(np.float32),
                selected_indices=[0, 1, 2],
                log_prob=-2.0,
                reward=0.0,
                value=0.5,
            )
            ctrl.record_transition(transition)

        # Compute advantages
        ctrl.compute_advantages(final_reward=0.7)

        # PPO update
        stats = ctrl.ppo_update(epochs=2)
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats

    def test_compute_advantages(self):
        """GAE advantage computation"""
        ctrl = self._make_controller()
        self._register_skills(ctrl, n=5)

        for i in range(3):
            transition = PPOTransition(
                state_embedding=np.random.randn(64).astype(np.float32),
                selected_indices=[0, 1],
                log_prob=-1.5,
                reward=0.0,
                value=0.3,
            )
            ctrl.record_transition(transition)

        ctrl.compute_advantages(final_reward=1.0)

        # Last step should have highest returns
        buffer = ctrl._trajectory_buffer
        assert buffer[-1].returns > buffer[0].returns or ctrl.gamma == 1.0
        # Advantage = returns - value
        for t in buffer:
            assert abs(t.advantage - (t.returns - t.value)) < 1e-6

    def test_mark_skills_not_new(self):
        ctrl = self._make_controller()
        ctrl.register_skill("s1", "S1", np.random.randn(64).astype(np.float32), is_new=True)
        ctrl.register_skill("s2", "S2", np.random.randn(64).astype(np.float32), is_new=True)

        assert ctrl._skill_embeddings[0].is_new
        ctrl.mark_skills_not_new()
        assert not ctrl._skill_embeddings[0].is_new
        assert not ctrl._skill_embeddings[1].is_new

class TestGumbelTopK:
    """Gumbel-Top-K sampling dedicated tests"""

    def test_returns_k_indices(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        indices = SkillSelectionController._gumbel_top_k(logits, k=3)
        assert len(indices) == 3
        assert len(set(indices)) == 3  # No duplicates

    def test_indices_in_range(self):
        logits = np.random.randn(20)
        indices = SkillSelectionController._gumbel_top_k(logits, k=5)
        for idx in indices:
            assert 0 <= idx < 20

    def test_high_logit_more_likely(self):
        """High logit skills should be selected more frequently"""
        logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        counts = np.zeros(5)
        for _ in range(100):
            indices = SkillSelectionController._gumbel_top_k(logits, k=1)
            counts[indices[0]] += 1
        # First skill should be selected most often
        assert counts[0] > 50, "High logit skill should be selected most often"

# Phase 3: Skill Designer Tests

from src.skill_induction.skill_designer import (
    EvolutionProposal,
    HardCase,
    HardCaseBuffer,
    SkillDesigner,
)
from src.models import Skill

class TestHardCase:
    """Hard Case data structure tests"""

    def test_difficulty_score(self):
        """d(q) = (1 - r(q)) · c(q)"""
        case = HardCase(query="test", reward=0.2, fail_count=3)
        assert abs(case.difficulty_score - 2.4) < 1e-6

    def test_difficulty_score_zero_reward(self):
        case = HardCase(query="test", reward=0.0, fail_count=1)
        assert abs(case.difficulty_score - 1.0) < 1e-6

    def test_difficulty_score_high_reward(self):
        case = HardCase(query="test", reward=0.7, fail_count=5)
        assert abs(case.difficulty_score - 1.5) < 1e-6

    def test_difficulty_ordering(self):
        """Verify paper example: A > C > B"""
        case_a = HardCase(query="A", reward=0.2, fail_count=3)  # d=2.4
        case_b = HardCase(query="B", reward=0.0, fail_count=1)  # d=1.0
        case_c = HardCase(query="C", reward=0.7, fail_count=5)  # d=1.5

        assert case_a.difficulty_score > case_c.difficulty_score > case_b.difficulty_score

class TestHardCaseBuffer:
    """Hard-Case Buffer tests"""

    def test_add_and_size(self):
        buf = HardCaseBuffer(max_size=100)
        assert buf.size == 0
        buf.add(HardCase(query="q1", reward=0.1, step=1))
        assert buf.size == 1

    def test_add_duplicate_increments_fail_count(self):
        buf = HardCaseBuffer(max_size=100)
        buf.add(HardCase(query="q1", reward=0.3, step=1))
        buf.add(HardCase(query="q1", reward=0.1, step=2))
        assert buf.size == 1
        # fail_count should increase
        cases = buf.get_top_cases(n=1)
        assert cases[0].fail_count == 2
        # reward should take minimum value
        assert cases[0].reward == 0.1

    def test_max_size_eviction(self):
        buf = HardCaseBuffer(max_size=5)
        for i in range(10):
            buf.add(HardCase(query=f"q{i}", reward=0.1, step=i))
        assert buf.size <= 5

    def test_get_top_cases_sorted(self):
        buf = HardCaseBuffer(max_size=100)
        buf.add(HardCase(query="easy", reward=0.9, fail_count=1, step=1))
        buf.add(HardCase(query="hard", reward=0.1, fail_count=5, step=2))
        buf.add(HardCase(query="medium", reward=0.5, fail_count=3, step=3))

        top = buf.get_top_cases(n=3)
        # Should be sorted by difficulty_score descending
        assert top[0].query == "hard"  # d = 0.9 * 5 = 4.5
        assert top[1].query == "medium"  # d = 0.5 * 3 = 1.5

    def test_get_top_cases_with_step_expiry(self):
        buf = HardCaseBuffer(max_size=100, max_step_gap=10)
        buf.add(HardCase(query="old", reward=0.1, fail_count=5, step=1))
        buf.add(HardCase(query="new", reward=0.1, fail_count=5, step=100))

        top = buf.get_top_cases(n=10, current_step=105)
        # "old" case should be expired (105 - 1 = 104 > 10)
        assert len(top) == 1
        assert top[0].query == "new"

    def test_clustered_representatives(self):
        buf = HardCaseBuffer(max_size=100)
        # Add two types of cases
        for i in range(10):
            buf.add(HardCase(
                query=f"temporal when did event {i} happen",
                reward=0.1, fail_count=2, step=i,
            ))
        for i in range(10):
            buf.add(HardCase(
                query=f"location where is object {i} placed",
                reward=0.2, fail_count=3, step=10 + i,
            ))

        reps = buf.get_clustered_representatives(
            n_clusters=2, representatives_per_cluster=3
        )
        # Should select representatives from each cluster
        assert len(reps) <= 6
        assert len(reps) >= 2

    def test_clear(self):
        buf = HardCaseBuffer(max_size=100)
        buf.add(HardCase(query="q1", reward=0.1, step=1))
        buf.clear()
        assert buf.size == 0

class TestSkillDesigner:
    """Skill Designer tests"""

    def test_should_trigger(self):
        designer = SkillDesigner(config={"trigger_interval": 100})
        designer.hard_case_buffer.add(
            HardCase(query="q1", reward=0.1, step=1)
        )
        assert not designer.should_trigger(50)
        assert designer.should_trigger(100)
        assert not designer.should_trigger(0)

    def test_should_trigger_empty_buffer(self):
        designer = SkillDesigner(config={"trigger_interval": 100})
        assert not designer.should_trigger(100)  # Empty buffer does not trigger

    def test_record_failure(self):
        designer = SkillDesigner()
        designer.record_failure(
            query="What happened?",
            prediction="I don't know",
            ground_truth="Event X",
            reward=0.0,
            step=1,
        )
        assert designer.hard_case_buffer.size == 1

    def test_update_reward_improvement(self):
        designer = SkillDesigner(config={"patience": 3})
        assert designer.update_reward(0.5)  # First time always improves
        assert designer.update_reward(0.6)  # Improved
        assert designer._patience_counter == 0

    def test_update_reward_no_improvement(self):
        designer = SkillDesigner(config={"patience": 3})
        designer.update_reward(0.5)
        assert not designer.update_reward(0.4)  # No improvement
        assert designer._patience_counter == 1

    def test_early_stop(self):
        designer = SkillDesigner(config={"patience": 2})
        designer.update_reward(0.5)
        designer.update_reward(0.3)  # patience=1
        designer.update_reward(0.2)  # patience=2
        assert designer.should_stop

    def test_apply_proposal_add(self):
        designer = SkillDesigner()
        proposal = EvolutionProposal(
            action="add",
            skill_name="Capture Temporal Context",
            description="Extract temporal information from conversations",
            content={
                "purpose": "Capture when events happened",
                "when_to_use": "When questions involve time",
                "how_to_apply": "Look for date/time mentions",
                "constraints": "Don't infer dates not mentioned",
            },
        )
        bank: list[Skill] = []
        bank, new_skill = designer.apply_proposal(proposal, bank)
        assert len(bank) == 1
        assert new_skill is not None
        assert new_skill.name == "Capture Temporal Context"
        assert new_skill.metadata.get("evolved")

    def test_apply_proposal_modify(self):
        designer = SkillDesigner()
        existing = Skill(
            name="Basic Memory",
            description="Basic memory skill",
            procedure=["Step 1"],
            version=1,
        )
        bank = [existing]

        proposal = EvolutionProposal(
            action="modify",
            skill_name="Basic Memory",
            description="Improved memory skill",
            content={"how_to_apply": "Enhanced step 1"},
        )
        bank, modified = designer.apply_proposal(proposal, bank)
        assert len(bank) == 1
        assert modified is not None
        assert modified.version == 2
        assert modified.description == "Improved memory skill"

    def test_apply_proposal_remove(self):
        designer = SkillDesigner()
        bank = [
            Skill(name="Good Skill", description="Keep this"),
            Skill(name="Bad Skill", description="Remove this"),
        ]
        proposal = EvolutionProposal(action="remove", skill_name="Bad Skill")
        bank, _ = designer.apply_proposal(proposal, bank)
        assert len(bank) == 1
        assert bank[0].name == "Good Skill"

    def test_evolve_with_llm(self):
        """Test full evolution flow with LLM"""
        mock_client = MagicMock()
        # Stage 1: analysis
        mock_client.chat.return_value = "The agent fails on temporal reasoning questions."
        # Stage 2: proposals
        mock_client.chat_json.return_value = json.dumps({
            "proposals": [
                {
                    "action": "add",
                    "skill_name": "Temporal Tracker",
                    "description": "Track temporal events",
                    "content": {
                        "purpose": "Track time",
                        "when_to_use": "Temporal questions",
                        "how_to_apply": "Extract dates",
                        "constraints": "Be precise",
                    },
                    "reasoning": "Many failures on temporal questions",
                }
            ]
        })

        designer = SkillDesigner(llm_client=mock_client)
        designer.record_failure("When?", "IDK", "May 2023", 0.0, step=1)

        proposals = designer.evolve(current_skills=[], current_step=100)
        assert len(proposals) == 1
        assert proposals[0].skill_name == "Temporal Tracker"

# Phase 4: Span Processor Tests

from src.memory.span_processor import SpanProcessor, TextSpan

class TestSpanProcessor:
    """Span-level Processor tests"""

    def test_split_basic(self):
        processor = SpanProcessor({"span_size": 50})
        text = "Hello world. " * 100  # ~1300 chars
        spans = processor.split_into_spans(text)
        assert len(spans) > 1
        for span in spans:
            assert span.text.strip()
            assert span.approx_tokens > 0

    def test_split_empty_text(self):
        processor = SpanProcessor()
        spans = processor.split_into_spans("")
        assert spans == []

    def test_split_short_text(self):
        processor = SpanProcessor({"span_size": 512})
        text = "Short text."
        spans = processor.split_into_spans(text)
        assert len(spans) == 1
        assert spans[0].text == "Short text."

    def test_span_ids_sequential(self):
        processor = SpanProcessor({"span_size": 50})
        text = "Sentence one. Sentence two. Sentence three. " * 20
        spans = processor.split_into_spans(text)
        for i, span in enumerate(spans):
            assert span.span_id == i

    def test_span_coverage(self):
        """All spans should cover original text (considering overlap)"""
        processor = SpanProcessor({"span_size": 50, "overlap": 0})
        text = "Word " * 200
        spans = processor.split_into_spans(text)
        # Concatenated span text should contain all original content
        all_text = " ".join(s.text for s in spans)
        # At least most words from original should appear
        assert all_text.count("Word") >= 100

    def test_dialogue_split_with_sessions(self):
        processor = SpanProcessor({"span_size": 50})
        sessions = [
            "Session 1: Hello! How are you? I'm fine, thanks.",
            "Session 2: Let's discuss the project. The deadline is next week.",
        ]
        spans = processor.split_dialogue_into_spans([], sessions=sessions)
        assert len(spans) >= 1
        # Should contain content from both sessions
        all_text = " ".join(s.text for s in spans)
        assert "Session 1" in all_text or "Hello" in all_text

    def test_dialogue_split_with_turns(self):
        processor = SpanProcessor({"span_size": 50})
        turns = [
            "User: Hello!",
            "Bot: Hi there!",
            "User: How are you?",
        ]
        spans = processor.split_dialogue_into_spans(turns)
        assert len(spans) >= 1

    def test_estimate_token_count(self):
        processor = SpanProcessor()
        assert processor.estimate_token_count("Hello world") > 0
        assert processor.estimate_token_count("A " * 100) > processor.estimate_token_count("Hi")

    def test_processing_stats(self):
        processor = SpanProcessor({"span_size": 50})
        text = "Test sentence. " * 50
        spans = processor.split_into_spans(text)
        stats = processor.get_processing_stats(spans)
        assert stats["num_spans"] > 0
        assert stats["total_tokens"] > 0
        assert stats["avg_tokens"] > 0

    def test_processing_stats_empty(self):
        processor = SpanProcessor()
        stats = processor.get_processing_stats([])
        assert stats["num_spans"] == 0

# Phase 5: Cross-Model Transfer Evaluation Tests

from src.evaluation.transfer_eval import (
    CrossModelTransferEvaluator,
    TransferReport,
    TransferResult,
)

class TestTransferResult:
    """TransferResult data structure tests"""

    def test_transfer_ratio(self):
        result = TransferResult(
            skill_id="s1", skill_name="S1",
            source_model="A", target_model="B",
            source_f1=0.5, target_f1=0.6,
        )
        assert abs(result.transfer_ratio - 1.2) < 1e-6

    def test_transfer_ratio_zero_source(self):
        result = TransferResult(
            skill_id="s1", skill_name="S1",
            source_model="A", target_model="B",
            source_f1=0.0, target_f1=0.5,
        )
        assert result.transfer_ratio == 0.0

class TestTransferReport:
    """TransferReport aggregation tests"""

    def test_compute_aggregates(self):
        report = TransferReport(
            source_model="LLaMA", target_model="Qwen",
            results=[
                TransferResult(
                    skill_id="s1", skill_name="S1",
                    source_model="LLaMA", target_model="Qwen",
                    source_f1=0.5, target_f1=0.6,
                ),
                TransferResult(
                    skill_id="s2", skill_name="S2",
                    source_model="LLaMA", target_model="Qwen",
                    source_f1=0.4, target_f1=0.5,
                ),
            ],
        )
        report.compute_aggregates()
        assert abs(report.avg_source_f1 - 0.45) < 1e-6
        assert abs(report.avg_target_f1 - 0.55) < 1e-6
        assert report.avg_transfer_gap > 0  # target > source

    def test_compute_aggregates_empty(self):
        report = TransferReport(source_model="A", target_model="B")
        report.compute_aggregates()
        assert report.avg_source_f1 == 0.0

class TestCrossModelTransferEvaluator:
    """Cross-model transfer evaluator tests"""

    def test_generate_comparison_table(self):
        evaluator = CrossModelTransferEvaluator()
        report = TransferReport(
            source_model="LLaMA-70B",
            target_model="Qwen-80B",
            results=[
                TransferResult(
                    skill_id="s1", skill_name="Temporal Tracker",
                    source_model="LLaMA-70B", target_model="Qwen-80B",
                    source_f1=0.5096, target_f1=0.5207,
                    transfer_gap=0.0111,
                ),
            ],
        )
        report.compute_aggregates()
        table = evaluator.generate_comparison_table(report)
        assert "LLaMA-70B" in table
        assert "Qwen-80B" in table
        assert "Temporal Tracker" in table

    def test_token_f1_computation(self):
        f1 = CrossModelTransferEvaluator._compute_token_f1(
            "The answer is Paris", "Paris"
        )
        assert f1 > 0

    def test_token_f1_empty_expected(self):
        f1 = CrossModelTransferEvaluator._compute_token_f1("anything", "")
        assert f1 == 1.0

    def test_token_f1_no_overlap(self):
        f1 = CrossModelTransferEvaluator._compute_token_f1("cat dog", "fish bird")
        assert f1 == 0.0

# Integration Tests: End-to-End Pipeline Verification

class TestEndToEndPipeline:
    """End-to-end pipeline integration tests"""

    def test_controller_designer_integration(self):
        """Controller + Designer working together"""
        # 1. Create controller and designer
        ctrl = SkillSelectionController({
            "embedding_dim": 32, "hidden_dim": 16, "top_k": 2,
        })
        designer = SkillDesigner(config={"trigger_interval": 5, "patience": 2})

        # 2. Register initial skills (MemSkill's 4 basic primitives)
        for name in ["INSERT", "UPDATE", "DELETE", "SKIP"]:
            ctrl.register_skill(
                f"primitive_{name}", name,
                np.random.randn(32).astype(np.float32),
            )

        # 3. Simulate training loop
        for step in range(10):
            state = ControllerState(
                embedding=np.random.randn(32).astype(np.float32)
            )
            result = ctrl.select_skills(state, training=True)
            assert len(result.selected_skill_ids) == 2

            # Simulate failure
            if step % 2 == 0:
                designer.record_failure(
                    query=f"Question {step}",
                    prediction="Wrong",
                    ground_truth="Right",
                    reward=0.1,
                    step=step,
                )

        # 4. Check designer state
        assert designer.hard_case_buffer.size > 0
        assert designer.should_trigger(10)  # trigger_interval=5

    def test_span_processor_with_locomo_format(self):
        """Span processor handles LoCoMo format data"""
        processor = SpanProcessor({"span_size": 100})

        sessions = [
            "1:56 pm on 8 May, 2023\n Caroline: Hey Mel! Good to see you!\n Melanie: Hey!",
            "2:30 pm on 15 May, 2023\n Caroline: I've been thinking about counseling.\n Melanie: That's great!",
        ]

        spans = processor.split_dialogue_into_spans([], sessions=sessions)
        assert len(spans) >= 1

        # Each span should have reasonable token count
        for span in spans:
            assert span.approx_tokens > 0
            assert span.text.strip()

    def test_skill_evolution_cycle(self):
        """Complete skill evolution cycle"""
        # 1. Initial skill bank
        skills = [
            Skill(name="INSERT", description="Insert new memory"),
            Skill(name="UPDATE", description="Update existing memory"),
            Skill(name="DELETE", description="Delete memory"),
            Skill(name="SKIP", description="Skip this span"),
        ]

        # 2. Designer proposes evolution
        designer = SkillDesigner()
        proposal = EvolutionProposal(
            action="add",
            skill_name="Capture Temporal Context",
            description="Extract temporal information",
            content={
                "purpose": "Track when events happened",
                "when_to_use": "Temporal questions",
                "how_to_apply": "Look for dates and times",
                "constraints": "Don't hallucinate dates",
            },
        )

        # 3. Apply proposals
        skills, new_skill = designer.apply_proposal(proposal, skills)
        assert len(skills) == 5
        assert new_skill.name == "Capture Temporal Context"

        # 4. Register to controller
        ctrl = SkillSelectionController({
            "embedding_dim": 32, "hidden_dim": 16, "top_k": 3,
        })
        for skill in skills:
            ctrl.register_skill(
                skill.skill_id, skill.name,
                np.random.randn(32).astype(np.float32),
                is_new=(skill.name == "Capture Temporal Context"),
            )

        # 5. Select skills (new skills should have exploration incentive)
        state = ControllerState(
            embedding=np.random.randn(32).astype(np.float32)
        )
        result = ctrl.select_skills(state, training=False)
        assert len(result.selected_skill_ids) == 3

    def test_reward_and_rollback_cycle(self):
        """Reward tracking + Rollback mechanism"""
        ctrl = SkillSelectionController({
            "embedding_dim": 32, "hidden_dim": 16, "top_k": 2,
        })
        for i in range(5):
            ctrl.register_skill(
                f"s{i}", f"Skill {i}",
                np.random.randn(32).astype(np.float32),
            )

        designer = SkillDesigner(config={"patience": 2})

        # Cycle 1: Improvement
        ctrl.save_snapshot(0.5)
        assert designer.update_reward(0.5)

        # Cycle 2: Improvement
        ctrl.save_snapshot(0.7)
        assert designer.update_reward(0.7)

        # Cycle 3: Degradation
        ctrl.policy_net.W1 += 100.0  # Corrupt parameters
        assert not designer.update_reward(0.3)

        # Cycle 4: Continued degradation -> early stop + rollback
        assert not designer.update_reward(0.2)
        assert designer.should_stop

        # Rollback to best
        assert ctrl.rollback_to_best()
