"""Unit tests for benchmark dataset loader (offline, with mock HuggingFace datasets)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.loader import BenchmarkLoader, PRIMARY_BENCHMARKS, LEGACY_BENCHMARKS

# Mock HuggingFace dataset rows

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

# New benchmark mock data

MOCK_GAIA_ROWS = [
    {
        "task_id": "00d579ea-0889-4fd9-a771-2c8d79835c8d",
        "Question": "What is the capital of France?",
        "Level": "1",
        "file_name": "",
        "Annotator Metadata": "{'Steps': '1'}",
        "trace": "[]",
        "prediction": "Paris",
        "Final answer": "Paris",
        "judge": "True",
    },
    {
        "task_id": "11e680fb-1234-5678-abcd-ef0123456789",
        "Question": "How many planets are in our solar system?",
        "Level": "2",
        "file_name": "",
        "Annotator Metadata": "{'Steps': '2'}",
        "trace": "[]",
        "prediction": "8",
        "Final answer": "8",
        "judge": "True",
    },
    {
        "task_id": "22f791gc-2345-6789-bcde-f01234567890",
        "Question": "Who wrote the theory of relativity?",
        "Level": "3",
        "file_name": "relativity.pdf",
        "Annotator Metadata": "{'Steps': '3'}",
        "trace": "[]",
        "prediction": "Albert Einstein",
        "Final answer": "Albert Einstein",
        "judge": "True",
    },
]

MOCK_ALFWORLD_ROWS = [
    {
        "id": "alfworld__pick_and_place_simple__001",
        "task_type": "pick_and_place_simple",
        "game_file_path": "pick_and_place_simple-Mug-None-Shelf-301/trial_T20190908_141942/game.tw-pddl",
        "game_content": json.dumps({
            "pddl_domain": "(define ...)",
            "grammar": "",
            "pddl_problem": "(define ...)",
            "solvable": True,
            "walkthrough": [
                "go to countertop_1",
                "take mug_1 from countertop_1",
                "go to shelf_1",
                "put mug_1 in shelf_1",
            ],
        }),
    },
    {
        "id": "alfworld__look_at_obj_in_light__002",
        "task_type": "look_at_obj_in_light",
        "game_file_path": "look_at_obj_in_light-CD-None-DeskLamp-308/trial_T20190908_142000/game.tw-pddl",
        "game_content": json.dumps({
            "pddl_domain": "(define ...)",
            "grammar": "",
            "pddl_problem": "(define ...)",
            "solvable": True,
            "walkthrough": [
                "go to desk_1",
                "take cd_1 from desk_1",
                "go to desklamp_1",
                "use desklamp_1",
            ],
        }),
    },
]

MOCK_2WIKI_ROWS = [
    {
        "_id": "8813f87c0bdd11eba7f7acde48001122",
        "type": "compositional",
        "question": "Who is the mother of the director of film Polish-Russian War?",
        "context": json.dumps([
            ["Polish-Russian War (film)", ["Polish-Russian War is a 2009 film directed by Xawery Żuławski."]],
            ["Xawery Żuławski", ["Xawery Żuławski is a Polish film director.", "His mother is Małgorzata Braunek."]],
        ]),
        "supporting_facts": json.dumps([["Polish-Russian War (film)", 0], ["Xawery Żuławski", 1]]),
        "evidences": "[]",
        "answer": "Małgorzata Braunek",
    },
    {
        "_id": "3B1cc0F0f77g4c4C5cE8B5d9DfGCbF6f",
        "type": "bridge_comparison",
        "question": "Are both directors of Jaws and E.T. the same person?",
        "context": json.dumps([
            ["Jaws (film)", ["Jaws is a 1975 film directed by Steven Spielberg."]],
            ["E.T. the Extra-Terrestrial", ["E.T. is a 1982 film directed by Steven Spielberg."]],
        ]),
        "supporting_facts": json.dumps([["Jaws (film)", 0], ["E.T. the Extra-Terrestrial", 0]]),
        "evidences": "[]",
        "answer": "yes",
    },
]

MOCK_AIME_ROWS = [
    {
        "ID": "2024-I-1",
        "Problem": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 minutes more than if she walks at $s+2$ kilometers per hour. Find $s$.",
        "Solution": "Let the time at speed s be t hours. Then 9/s - 9/(s+2) = 4/60...",
        "Answer": "4",
    },
    {
        "ID": "2024-I-2",
        "Problem": "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x(y^x) = \\log_y(x^{4y}) = 10$. Find $xy$.",
        "Solution": "From the equations we get x*log_x(y) = 10 and 4y*log_y(x) = 10...",
        "Answer": "25",
    },
    {
        "ID": "2024-II-1",
        "Problem": "Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. Find the number of residents who own all four items.",
        "Solution": "By inclusion-exclusion...",
        "Answer": "124",
    },
]

MOCK_TRAVELPLANNER_ROWS = [
    {
        "org": "Washington",
        "dest": "Myrtle Beach",
        "days": "3",
        "visiting_city_number": "1",
        "date": "['2022-03-13', '2022-03-14', '2022-03-15']",
        "people_number": "1",
        "local_constraint": "{'house rule': None, 'cuisine': None, 'room type': None, 'transportation': None}",
        "budget": "1400",
        "query": "Please create a travel plan for me departing from Washington to Myrtle Beach for 3 days.",
        "level": "easy",
        "reference_information": "[{'day': 1, 'transportation': 'flight', 'hotel': 'Beach Resort'}]",
    },
    {
        "org": "New York",
        "dest": "Los Angeles",
        "days": "5",
        "visiting_city_number": "2",
        "date": "['2022-04-01', '2022-04-02', '2022-04-03', '2022-04-04', '2022-04-05']",
        "people_number": "2",
        "local_constraint": "{'house rule': 'no smoking', 'cuisine': 'Italian', 'room type': 'suite', 'transportation': 'car'}",
        "budget": "5000",
        "query": "Plan a 5-day trip from New York to Los Angeles for 2 people with Italian food preference.",
        "level": "medium",
        "reference_information": "[{'day': 1, 'transportation': 'flight', 'hotel': 'Hilton LA'}]",
    },
]

MOCK_WEBSHOP_ROWS = [
    {
        "id": "traj_1_step_5",
        "prompt": "You are a shopping agent.\n\nInstruction:\ni need a blue cotton t-shirt size medium, price lower than 30.00 dollars\n\nHistory:\nStep 1: Searched for 'blue cotton t-shirt medium' -> Found 5 results",
        "response": "I will click on the first result that matches the criteria and buy it.",
    },
    {
        "id": "traj_2_step_3",
        "prompt": "You are a shopping agent.\n\nInstruction:\nfind me a wireless mouse with ergonomic design under 50 dollars\n\nHistory:\nStep 1: Searched for 'wireless ergonomic mouse' -> Found 10 results",
        "response": "I found a Logitech MX Ergo that matches. Clicking Buy Now.",
    },
    {
        "id": "traj_3_step_8",
        "prompt": "You are a shopping agent.\n\nInstruction:\ni want running shoes nike size 10 black color\n\nHistory:\nStep 1: Searched for 'nike running shoes size 10 black' -> Found 3 results",
        "response": "Selected Nike Air Zoom Pegasus in black, size 10. Proceeding to purchase.",
    },
]

# GAIA loader tests

class TestGAIALoader:
    """Tests for GAIA loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_gaia_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_GAIA_ROWS
        loader = BenchmarkLoader({"name": "gaia", "num_samples": 3})
        tasks = loader.load()

        assert len(tasks) == 3
        mock_load_dataset.assert_called_once_with(
            "Intelligent-Internet/ii-agent_gaia-benchmark_validation",
            split="train",
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_gaia_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_GAIA_ROWS
        loader = BenchmarkLoader({"name": "gaia", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "gaia_00d579ea-0889-4fd9-a771-2c8d79835c8d"
        assert "capital of France" in task["description"]
        assert task["expected"] == "Paris"
        assert task["metadata"]["level"] == "1"

    @patch("benchmarks.loader.load_dataset")
    def test_load_gaia_levels(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_GAIA_ROWS
        loader = BenchmarkLoader({"name": "gaia", "num_samples": 3})
        tasks = loader.load()

        levels = [t["metadata"]["level"] for t in tasks]
        assert "1" in levels
        assert "2" in levels
        assert "3" in levels

    @patch("benchmarks.loader.load_dataset")
    def test_load_gaia_num_samples_limit(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_GAIA_ROWS
        loader = BenchmarkLoader({"name": "gaia", "num_samples": 2})
        tasks = loader.load()
        assert len(tasks) == 2

# ALFWorld loader tests

class TestALFWorldLoader:
    """Tests for ALFWorld loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_alfworld_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_ALFWORLD_ROWS
        loader = BenchmarkLoader({"name": "alfworld", "num_samples": 2})
        tasks = loader.load()

        assert len(tasks) == 2
        mock_load_dataset.assert_called_once_with(
            "awawa-agi/alfworld-raw",
            split="eval_out_of_distribution",
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_alfworld_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_ALFWORLD_ROWS
        loader = BenchmarkLoader({"name": "alfworld", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "alfworld_alfworld__pick_and_place_simple__001"
        assert "household task" in task["description"]
        assert "pick and place simple" in task["description"]
        assert task["metadata"]["task_type"] == "pick_and_place_simple"
        assert task["metadata"]["num_steps"] == 4
        # Expected is the walkthrough joined
        assert "go to countertop_1" in task["expected"]
        assert "put mug_1 in shelf_1" in task["expected"]

    @patch("benchmarks.loader.load_dataset")
    def test_load_alfworld_walkthrough_parsing(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_ALFWORLD_ROWS
        loader = BenchmarkLoader({"name": "alfworld", "num_samples": 2})
        tasks = loader.load()

        # Check walkthrough steps are preserved in metadata
        assert tasks[0]["metadata"]["walkthrough_steps"] == [
            "go to countertop_1",
            "take mug_1 from countertop_1",
            "go to shelf_1",
            "put mug_1 in shelf_1",
        ]
        assert tasks[1]["metadata"]["walkthrough_steps"] == [
            "go to desk_1",
            "take cd_1 from desk_1",
            "go to desklamp_1",
            "use desklamp_1",
        ]

    @patch("benchmarks.loader.load_dataset")
    def test_load_alfworld_task_description(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_ALFWORLD_ROWS
        loader = BenchmarkLoader({"name": "alfworld", "num_samples": 2})
        tasks = loader.load()

        # Task description should include task type
        assert "pick and place simple" in tasks[0]["description"].lower()
        assert "look at obj in light" in tasks[1]["description"].lower()

# 2WikiMultihopQA loader tests

class Test2WikiMultihopQALoader:
    """Tests for 2WikiMultihopQA loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_2wiki_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_2WIKI_ROWS
        loader = BenchmarkLoader({"name": "2wikimultihopqa", "num_samples": 2})
        tasks = loader.load()

        assert len(tasks) == 2
        mock_load_dataset.assert_called_once_with(
            "scholarly-shadows-syndicate/2WikiMultiHopQA",
            split="validation",
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_2wiki_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_2WIKI_ROWS
        loader = BenchmarkLoader({"name": "2wikimultihopqa", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "2wikimqa_8813f87c0bdd11eba7f7acde48001122"
        assert "Polish-Russian War" in task["description"]
        assert task["expected"] == "Małgorzata Braunek"
        assert task["metadata"]["type"] == "compositional"

    @patch("benchmarks.loader.load_dataset")
    def test_load_2wiki_context_parsing(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_2WIKI_ROWS
        loader = BenchmarkLoader({"name": "2wikimultihopqa", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        # Context should be parsed from JSON into readable format
        assert "[Polish-Russian War (film)]" in task["context"]
        assert "[Xawery Żuławski]" in task["context"]
        assert "Małgorzata Braunek" in task["context"]

    @patch("benchmarks.loader.load_dataset")
    def test_load_2wiki_num_samples_limit(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_2WIKI_ROWS
        loader = BenchmarkLoader({"name": "2wikimultihopqa", "num_samples": 1})
        tasks = loader.load()
        assert len(tasks) == 1

# AIME loader tests

class TestAIMELoader:
    """Tests for AIME loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_aime_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_AIME_ROWS
        loader = BenchmarkLoader({"name": "aime", "num_samples": 3})
        tasks = loader.load()

        assert len(tasks) == 3
        mock_load_dataset.assert_called_once_with(
            "Maxwell-Jia/AIME_2024",
            split="train",
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_aime_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_AIME_ROWS
        loader = BenchmarkLoader({"name": "aime", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "aime_2024-I-1"
        assert "9-kilometer" in task["description"] or "9$-kilometer" in task["description"]
        assert task["expected"] == "4"
        assert task["metadata"]["problem_id"] == "2024-I-1"
        assert "solution" in task["metadata"]

    @patch("benchmarks.loader.load_dataset")
    def test_load_aime_answers_are_integers(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_AIME_ROWS
        loader = BenchmarkLoader({"name": "aime", "num_samples": 3})
        tasks = loader.load()

        # All AIME answers should be numeric strings
        for task in tasks:
            assert task["expected"].isdigit(), f"Expected integer answer, got: {task['expected']}"

    @patch("benchmarks.loader.load_dataset")
    def test_load_aime_num_samples_limit(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_AIME_ROWS
        loader = BenchmarkLoader({"name": "aime", "num_samples": 2})
        tasks = loader.load()
        assert len(tasks) == 2

# TravelPlanner loader tests

class TestTravelPlannerLoader:
    """Tests for TravelPlanner loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_travelplanner_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_TRAVELPLANNER_ROWS
        loader = BenchmarkLoader({"name": "travelplanner", "num_samples": 2})
        tasks = loader.load()

        assert len(tasks) == 2
        mock_load_dataset.assert_called_once_with(
            "osunlp/TravelPlanner",
            "validation",
            split="validation",
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_travelplanner_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_TRAVELPLANNER_ROWS
        loader = BenchmarkLoader({"name": "travelplanner", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "travelplanner_0"
        assert "Washington" in task["description"]
        assert "Myrtle Beach" in task["description"]
        assert "$1400" in task["description"]
        assert task["metadata"]["org"] == "Washington"
        assert task["metadata"]["dest"] == "Myrtle Beach"
        assert task["metadata"]["days"] == "3"
        assert task["metadata"]["level"] == "easy"

    @patch("benchmarks.loader.load_dataset")
    def test_load_travelplanner_constraints(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_TRAVELPLANNER_ROWS
        loader = BenchmarkLoader({"name": "travelplanner", "num_samples": 2})
        tasks = loader.load()

        # Second task has more constraints
        task = tasks[1]
        assert "New York" in task["description"]
        assert "Los Angeles" in task["description"]
        assert "$5000" in task["description"]
        assert task["metadata"]["people_number"] == "2"

    @patch("benchmarks.loader.load_dataset")
    def test_load_travelplanner_num_samples_limit(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_TRAVELPLANNER_ROWS
        loader = BenchmarkLoader({"name": "travelplanner", "num_samples": 1})
        tasks = loader.load()
        assert len(tasks) == 1

# WebShop loader tests

class TestWebShopLoader:
    """Tests for WebShop loading with mocked HuggingFace datasets."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_webshop_basic(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_WEBSHOP_ROWS
        loader = BenchmarkLoader({"name": "webshop", "num_samples": 3})
        tasks = loader.load()

        assert len(tasks) == 3
        mock_load_dataset.assert_called_once_with(
            "Skyler215/webshop-agent-cot",
            split="test",
        )

    @patch("benchmarks.loader.load_dataset")
    def test_load_webshop_task_format(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_WEBSHOP_ROWS
        loader = BenchmarkLoader({"name": "webshop", "num_samples": 1})
        tasks = loader.load()

        task = tasks[0]
        assert task["task_id"] == "webshop_traj_1_step_5"
        assert "shopping task" in task["description"]
        assert "blue cotton t-shirt" in task["description"]
        assert task["expected"] == "I will click on the first result that matches the criteria and buy it."
        assert task["metadata"]["original_id"] == "traj_1_step_5"

    @patch("benchmarks.loader.load_dataset")
    def test_load_webshop_instruction_extraction(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_WEBSHOP_ROWS
        loader = BenchmarkLoader({"name": "webshop", "num_samples": 3})
        tasks = loader.load()

        # Instruction should be extracted from prompt
        assert "blue cotton t-shirt" in tasks[0]["description"]
        assert "wireless mouse" in tasks[1]["description"]
        assert "running shoes" in tasks[2]["description"]

    @patch("benchmarks.loader.load_dataset")
    def test_load_webshop_num_samples_limit(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_WEBSHOP_ROWS
        loader = BenchmarkLoader({"name": "webshop", "num_samples": 2})
        tasks = loader.load()
        assert len(tasks) == 2

# HotpotQA loader tests (kept from original)

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
        assert "[Scott Derrickson]" in task["context"]
        assert "American director" in task["context"]
        assert "[Ed Wood]" in task["context"]
        assert "American filmmaker" in task["context"]

# HotpotQA hard subset tests (legacy)

class TestHotpotQAHardLoader:
    """Tests for HotpotQA hard subset loading."""

    @patch("benchmarks.loader.load_dataset")
    def test_load_hard_only(self, mock_load_dataset):
        mock_load_dataset.return_value = MOCK_HOTPOTQA_ROWS
        loader = BenchmarkLoader({"name": "hotpotqa_hard", "num_samples": 10})
        tasks = loader.load()

        assert len(tasks) == 1
        assert tasks[0]["metadata"]["level"] == "hard"

    @patch("benchmarks.loader.load_dataset")
    def test_load_hard_empty_result(self, mock_load_dataset):
        rows = [dict(r, level="medium") for r in MOCK_HOTPOTQA_ROWS]
        for r in rows:
            r["context"] = MOCK_HOTPOTQA_ROWS[0]["context"]
        mock_load_dataset.return_value = rows
        loader = BenchmarkLoader({"name": "hotpotqa_hard", "num_samples": 10})
        tasks = loader.load()
        assert len(tasks) == 0

# SWE-bench loader tests (legacy)

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

        task = tasks[1]
        assert "Hints:" not in task["description"]

# TriviaQA loader tests (legacy)

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

# GSM8K loader tests (legacy)

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
        assert task["expected"] == "18"
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

# MuSiQue loader tests (legacy)

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

# General loader tests

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

    def test_all_primary_benchmark_names_valid(self):
        """Verify all primary benchmark names are accepted by the factory."""
        for name in PRIMARY_BENCHMARKS:
            loader = BenchmarkLoader({"name": name})
            assert loader.benchmark_name == name

    def test_all_legacy_benchmark_names_valid(self):
        """Verify all legacy benchmark names are still accepted by the factory."""
        for name in LEGACY_BENCHMARKS:
            loader = BenchmarkLoader({"name": name})
            assert loader.benchmark_name == name

    def test_all_benchmark_names_valid(self):
        """Verify all documented benchmark names are accepted by the factory."""
        all_names = PRIMARY_BENCHMARKS + LEGACY_BENCHMARKS
        for name in all_names:
            loader = BenchmarkLoader({"name": name})
            assert loader.benchmark_name == name

    def test_primary_benchmarks_list(self):
        """Verify PRIMARY_BENCHMARKS contains the expected benchmarks."""
        expected = {"gaia", "alfworld", "hotpotqa", "2wikimultihopqa", "aime", "travelplanner", "webshop"}
        assert set(PRIMARY_BENCHMARKS) == expected

    def test_legacy_benchmarks_list(self):
        """Verify LEGACY_BENCHMARKS contains the disabled benchmarks."""
        expected = {"hotpotqa_hard", "triviaqa", "gsm8k", "musique", "swebench"}
        assert set(LEGACY_BENCHMARKS) == expected

# Helper method tests

class TestHelperMethods:
    """Tests for static/class helper methods."""

    def test_alfworld_task_description_full(self):
        desc = BenchmarkLoader._alfworld_task_description(
            "pick_and_place_simple",
            "pick_and_place_simple-Mug-None-Shelf-301/trial_T20190908/game.tw-pddl",
        )
        assert "pick and place simple" in desc
        assert "Mug" in desc

    def test_alfworld_task_description_minimal(self):
        desc = BenchmarkLoader._alfworld_task_description(
            "look_at_obj_in_light",
            "look_at_obj_in_light-CD-None-DeskLamp-308/trial/game.tw-pddl",
        )
        assert "look at obj in light" in desc

    def test_parse_2wiki_context_valid_json(self):
        context_json = json.dumps([
            ["Title A", ["Sentence 1.", "Sentence 2."]],
            ["Title B", ["Sentence 3."]],
        ])
        result = BenchmarkLoader._parse_2wiki_context(context_json)
        assert "[Title A]" in result
        assert "Sentence 1. Sentence 2." in result
        assert "[Title B]" in result
        assert "Sentence 3." in result

    def test_parse_2wiki_context_invalid_json(self):
        result = BenchmarkLoader._parse_2wiki_context("not valid json {{{")
        assert "not valid json" in result

    def test_parse_2wiki_context_non_list(self):
        result = BenchmarkLoader._parse_2wiki_context(json.dumps({"key": "value"}))
        assert "key" in result

    def test_extract_webshop_instruction(self):
        prompt = "You are a shopping agent.\n\nInstruction:\nbuy a red hat under 20 dollars\n\nHistory:\n..."
        result = BenchmarkLoader._extract_webshop_instruction(prompt)
        assert result == "buy a red hat under 20 dollars"

    def test_extract_webshop_instruction_fallback(self):
        prompt = "No instruction marker here, just some text about shopping"
        result = BenchmarkLoader._extract_webshop_instruction(prompt)
        assert "No instruction marker" in result

    def test_extract_gsm8k_answer(self):
        assert BenchmarkLoader._extract_gsm8k_answer("some steps\n#### 42") == "42"
        assert BenchmarkLoader._extract_gsm8k_answer("#### 100") == "100"
        assert BenchmarkLoader._extract_gsm8k_answer("no marker here") == "no marker here"
        assert BenchmarkLoader._extract_gsm8k_answer("step1\nstep2\n#### 7,500") == "7,500"

# MuSiQue loader tests

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

# General loader tests

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
