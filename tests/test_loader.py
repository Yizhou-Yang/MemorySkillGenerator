"""Unit tests for benchmark dataset loader (offline, with mock HuggingFace datasets)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from benchmarks.loader import BenchmarkLoader


# ============================================================
# Mock HuggingFace dataset rows
# ============================================================

MOCK_HOTPOTQA_ROWS = [
    {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "Yes",
        "type": "comparison",
        "level": "medium",
        "context": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sentences": [
                ["Scott Derrickson is an American director. ", "He is best known for Sinister."],
                ["Ed Wood was an American filmmaker. ", "He is known for Plan 9."],
            ],
        },
        "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
    },
    {
        "id": "5a8c7595554299585d9e36b6",
        "question": "What government position was held by the woman who portrayed Nora Batty?",
        "answer": "Chancellor of the Exchequer",
        "type": "bridge",
        "level": "hard",
        "context": {
            "title": ["Kathy Staff", "Nora Batty"],
            "sentences": [
                ["Kathy Staff was a British actress. "],
                ["Nora Batty is a fictional character. "],
            ],
        },
        "supporting_facts": {"title": ["Kathy Staff"], "sent_id": [0]},
    },
    {
        "id": "5ae2c5d2554299657d4e8a0f",
        "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "type": "comparison",
        "level": "medium",
        "context": {
            "title": ["Arthur's Magazine", "First for Women"],
            "sentences": [
                ["Arthur's Magazine was founded in 1844. "],
                ["First for Women is published by Bauer Media. "],
            ],
        },
        "supporting_facts": {"title": ["Arthur's Magazine", "First for Women"], "sent_id": [0, 0]},
    },
]

MOCK_TRIVIAQA_ROWS = [
    {
        "question": "Who was the man behind The Chipmunks?",
        "question_id": "tc_1",
        "question_source": "www.triviacountry.com",
        "entity_pages": {},
        "search_results": {},
        "answer": {
            "aliases": ["David Seville", "Ross Bagdasarian"],
            "normalized_aliases": ["david seville", "ross bagdasarian"],
            "matched_wiki_entity_name": "Ross Bagdasarian Sr.",
            "normalized_matched_wiki_entity_name": "ross bagdasarian sr.",
            "normalized_value": "david seville",
            "type": "WikipediaEntity",
            "value": "David Seville",
        },
    },
    {
        "question": "Which Lloyd Webber musical premiered in the US on 10th December 1993?",
        "question_id": "tc_2",
        "question_source": "www.triviacountry.com",
        "entity_pages": {},
        "search_results": {},
        "answer": {
            "aliases": ["Sunset Boulevard"],
            "normalized_aliases": ["sunset boulevard"],
            "matched_wiki_entity_name": "Sunset Boulevard",
            "normalized_matched_wiki_entity_name": "sunset boulevard",
            "normalized_value": "sunset boulevard",
            "type": "WikipediaEntity",
            "value": "Sunset Boulevard",
        },
    },
]

MOCK_GSM8K_ROWS = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmers' market.\n#### 18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber.\nSo the total is 2+1=<<2+1=3>>3 bolts.\n#### 3",
    },
]

MOCK_MUSIQUE_ROWS = [
    {
        "id": "2hop__131818_161450",
        "question": "Who is the spouse of the Green performer?",
        "answer": "Miquette Giraudy",
        "answer_aliases": ["Miquette Giraudy"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "Green (Steve Hillage album)",
                "paragraph_text": "Green is the fifth solo album by Steve Hillage.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Steve Hillage",
                "paragraph_text": "Steve Hillage is married to Miquette Giraudy.",
                "is_supporting": True,
            },
        ],
        "question_decomposition": [
            {"id": 1, "question": "Who performed Green?", "answer": "Steve Hillage", "paragraph_support_idx": 0},
            {"id": 2, "question": "Who is the spouse of Steve Hillage?", "answer": "Miquette Giraudy", "paragraph_support_idx": 1},
        ],
    },
    {
        "id": "2hop__100234_200567",
        "question": "What country is the birthplace of the director of Jaws?",
        "answer": "United States",
        "answer_aliases": ["United States", "USA", "US"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "Jaws (film)",
                "paragraph_text": "Jaws is a 1975 film directed by Steven Spielberg.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Steven Spielberg",
                "paragraph_text": "Steven Spielberg was born in Cincinnati, Ohio, United States.",
                "is_supporting": True,
            },
        ],
        "question_decomposition": [
            {"id": 1, "question": "Who directed Jaws?", "answer": "Steven Spielberg", "paragraph_support_idx": 0},
            {"id": 2, "question": "What country was Steven Spielberg born in?", "answer": "United States", "paragraph_support_idx": 1},
        ],
    },
]

MOCK_SWEBENCH_ROWS = [
    {
        "instance_id": "astropy__astropy-12907",
        "repo": "astropy/astropy",
        "problem_statement": "Modeling compound model separability issue",
        "hints_text": "Check the separability matrix",
        "patch": "diff --git a/astropy/modeling/separable.py ...",
        "base_commit": "abc123",
        "version": "4.3",
        "FAIL_TO_PASS": "[test_separability]",
    },
    {
        "instance_id": "django__django-11099",
        "repo": "django/django",
        "problem_statement": "UsernameValidator allows trailing newline",
        "hints_text": "",
        "patch": "diff --git a/django/contrib/auth/validators.py ...",
        "base_commit": "def456",
        "version": "3.0",
        "FAIL_TO_PASS": "[test_validators]",
    },
]


# ============================================================
# HotpotQA loader tests
# ============================================================


class TestHotpotQALoader:
    """Tests for HotpotQA loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_hotpotqa_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_HOTPOTQA_ROWS
        loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": 3})
        tasks = loader.load()

        assert len(tasks) == 3
        mock_load_dataset.assert_called_once_with(
            "hotpotqa/hotpot_qa", "distractor", split="validation"
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_hotpotqa_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_HOTPOTQA_ROWS
        loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "hotpotqa_5a8b57f25542995d1e6f1371"
        assert "Were Scott Derrickson" in task["description"]
        assert task["expected"] == "Yes"
        assert "[Scott Derrickson]" in task["context"]
        assert task["metadata"]["type"] == "comparison"
        assert task["metadata"]["level"] == "medium"

    @patch("benchmarks.loader.load_dataset")
    def test_load_hotpotqa_num_samples_limit(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_HOTPOTQA_ROWS
        loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": 2})
        tasks = loader.load()
        assert len(tasks) == 2

    @patch("benchmarks.loader.load_dataset")
    def test_load_hotpotqa_context_formatting(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_HOTPOTQA_ROWS
        loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        # Context should contain paragraph titles and sentences
        assert "[Scott Derrickson]" in task["context"]
        assert "American director" in task["context"]
        assert "[Ed Wood]" in task["context"]
        assert "American filmmaker" in task["context"]


# ============================================================
# HotpotQA hard subset tests
# ============================================================


class TestHotpotQAHardLoader:
    """Tests for HotpotQA hard subset loading."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_hard_only(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_HOTPOTQA_ROWS
        loader = BenchmarkLoader({"name": "hotpotqa_hard", "num_samples": 10})
        tasks = loader.load()

        # Only 1 row in mock data has level="hard"
        assert len(tasks) == 1
        assert tasks[0]["metadata"]["level"] == "hard"

    @patch("benchmarks.loader.load_dataset")
    def test_load_hard_empty_result(self, mock_load_dataset):
        # All rows are non-hard
        rows = [dict(r, level="medium") for r in MOCK_HOTPOTQA_ROWS]
        for r in rows:
            r["context"] = MOCK_HOTPOTQA_ROWS[0]["context"]
        mock_load_dataset.return_value = rows
        loader = BenchmarkLoader({"name": "hotpotqa_hard", "num_samples": 10})
        tasks = loader.load()
        assert len(tasks) == 0


# ============================================================
# SWE-bench loader tests
# ============================================================


class TestSWEBenchLoader:
    """Tests for SWE-bench loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_swebench_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_SWEBENCH_ROWS
        loader = BenchmarkLoader({"name": "swebench", "num_samples": 2})
        tasks = loader.load()

        assert len(tasks) == 2
        mock_load_dataset.assert_called_once_with(
            "princeton-nlp/SWE-bench_Lite", split="test"
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_swebench_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_SWEBENCH_ROWS
        loader = BenchmarkLoader({"name": "swebench", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "swebench_astropy__astropy-12907"
        assert "astropy/astropy" in task["description"]
        assert "separability" in task["description"].lower()
        assert task["expected"].startswith("diff --git")
        assert task["metadata"]["repo"] == "astropy/astropy"
        assert task["metadata"]["instance_id"] == "astropy__astropy-12907"

    @patch("benchmarks.loader.load_dataset")
    def test_load_swebench_with_hints(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_SWEBENCH_ROWS
        loader = BenchmarkLoader({"name": "swebench", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert "Hints:" in task["description"]
        assert "separability matrix" in task["description"]

    @patch("benchmarks.loader.load_dataset")
    def test_load_swebench_without_hints(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_SWEBENCH_ROWS
        loader = BenchmarkLoader({"name": "swebench", "num_samples": 2})
        tasks = loader.load()

        # Second row has empty hints
        task = tasks[1]
        assert "Hints:" not in task["description"]


# ============================================================
# General loader tests
# ============================================================


# ============================================================
# TriviaQA loader tests
# ============================================================


class TestTriviaQALoader:
    """Tests for TriviaQA loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_triviaqa_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_TRIVIAQA_ROWS
        loader = BenchmarkLoader({"name": "triviaqa", "num_samples": 2})
        tasks = loader.load()

        assert len(tasks) == 2
        mock_load_dataset.assert_called_once_with(
            "mandarjoshi/trivia_qa", "rc", split="validation"
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_triviaqa_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_TRIVIAQA_ROWS
        loader = BenchmarkLoader({"name": "triviaqa", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "triviaqa_tc_1"
        assert "Chipmunks" in task["description"]
        assert task["expected"] == "David Seville"
        assert "David Seville" in task["metadata"]["aliases"]

    @patch("benchmarks.loader.load_dataset")
    def test_load_triviaqa_num_samples_limit(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_TRIVIAQA_ROWS
        loader = BenchmarkLoader({"name": "triviaqa", "num_samples": 1})
        tasks = loader.load()
        assert len(tasks) == 1


# ============================================================
# GSM8K loader tests
# ============================================================


class TestGSM8KLoader:
    """Tests for GSM8K loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_gsm8k_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_GSM8K_ROWS
        loader = BenchmarkLoader({"name": "gsm8k", "num_samples": 2})
        tasks = loader.load()

        assert len(tasks) == 2
        mock_load_dataset.assert_called_once_with(
            "openai/gsm8k", "main", split="test"
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_gsm8k_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_GSM8K_ROWS
        loader = BenchmarkLoader({"name": "gsm8k", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "gsm8k_0"
        assert "Janet" in task["description"]
        assert task["expected"] == "18"  # extracted from #### 18
        assert "full_solution" in task["metadata"]

    @patch("benchmarks.loader.load_dataset")
    def test_load_gsm8k_answer_extraction(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_GSM8K_ROWS
        loader = BenchmarkLoader({"name": "gsm8k", "num_samples": 2})
        tasks = loader.load()

        assert tasks[0]["expected"] == "18"
        assert tasks[1]["expected"] == "3"

    def test_extract_gsm8k_answer_helper(self):
        """Test the static answer extraction helper directly."""
        assert BenchmarkLoader._extract_gsm8k_answer("some steps\n#### 42") == "42"
        assert BenchmarkLoader._extract_gsm8k_answer("#### 100") == "100"
        assert BenchmarkLoader._extract_gsm8k_answer("no marker here") == "no marker here"
        assert BenchmarkLoader._extract_gsm8k_answer("step1\nstep2\n#### 7,500") == "7,500"


# ============================================================
# MuSiQue loader tests
# ============================================================


class TestMuSiQueLoader:
    """Tests for MuSiQue loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_musique_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_MUSIQUE_ROWS
        loader = BenchmarkLoader({"name": "musique", "num_samples": 2})
        tasks = loader.load()

        assert len(tasks) == 2
        mock_load_dataset.assert_called_once_with(
            "dgslibisey/MuSiQue", split="validation"
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_musique_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_MUSIQUE_ROWS
        loader = BenchmarkLoader({"name": "musique", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "musique_2hop__131818_161450"
        assert "Green performer" in task["description"]
        assert task["expected"] == "Miquette Giraudy"
        assert task["metadata"]["answerable"] is True
        assert len(task["metadata"]["decomposition"]) == 2
        assert "Who performed Green?" in task["metadata"]["decomposition"]

    @patch("benchmarks.loader.load_dataset")
    def test_load_musique_context_formatting(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_MUSIQUE_ROWS
        loader = BenchmarkLoader({"name": "musique", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert "[Green (Steve Hillage album)]" in task["context"]
        assert "Steve Hillage" in task["context"]


# ============================================================
# General loader tests
# ============================================================


class TestBenchmarkLoaderGeneral:
    """General tests for BenchmarkLoader."""

    def test_unsupported_benchmark(self):
        loader = BenchmarkLoader({"name": "nonexistent_benchmark"})
        with pytest.raises(ValueError, match="Unsupported benchmark"):
            loader.load()

    def test_default_config(self):
        loader = BenchmarkLoader()
        assert loader.benchmark_name == "hotpotqa"
        assert loader.num_samples == 20

    def test_custom_config(self):
        loader = BenchmarkLoader({"name": "swebench", "num_samples": 5})
        assert loader.benchmark_name == "swebench"
        assert loader.num_samples == 5

    def test_none_config(self):
        loader = BenchmarkLoader(None)
        assert loader.benchmark_name == "hotpotqa"
        assert loader.num_samples == 20

    def test_all_benchmark_names_valid(self):
        """Verify all documented benchmark names are accepted by the factory."""
        valid_names = ["hotpotqa", "hotpotqa_hard", "triviaqa", "gsm8k", "musique", "swebench"]
        for name in valid_names:
            loader = BenchmarkLoader({"name": name})
            assert loader.benchmark_name == name
