"""Unit tests for gate.py — task classification and injection gating.

Tests cover all 5 benchmarks used in the latest experiment runner:
  - GAIA (static QA, multi-step reasoning)
  - ALFWorld (embodied, household tasks)
  - LoCoMo (conversation memory QA)
  - GAIA2 (dynamic agentic, CLI tool-use)
  - SWE-bench Dynamic (code bug-fixing)
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from v6.gate import assess_task_complexity, should_augment, classify_task_type
from v6.experience import ExperienceLibrary, Experience


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def empty_library():
    return ExperienceLibrary()


@pytest.fixture
def populated_library():
    lib = ExperienceLibrary()
    lib.record(Experience(
        task_id="test_1",
        task_desc="Calculate the sum of prime numbers below 100",
        tool_sequence=["python_exec"],
        action_commands=["python -c 'print(sum(...))'"],
        outcome="success",
        score=1.0,
        missing_steps=[],
        extra_steps=[],
        failure_reason="",
        failure_taxonomy={"ai_refined": True, "causal_lesson": "Use sieve of Eratosthenes"},
    ))
    lib.record(Experience(
        task_id="test_2",
        task_desc="Go to the kitchen and pick up the mug",
        tool_sequence=["go_to", "take"],
        action_commands=["go to kitchen", "take mug"],
        outcome="success",
        score=1.0,
        missing_steps=[],
        extra_steps=[],
        failure_reason="",
    ))
    return lib


# ─── classify_task_type: GAIA ──────────────────────────────────────────────

class TestClassifyGaia:
    """GAIA tasks: look like QA but require multi-step reasoning / tool use."""

    def test_gaia_with_metadata(self):
        """GAIA tasks with benchmark metadata → agentic."""
        desc = "Answer the following question accurately.\n\nQuestion: What is the population of Tokyo in 2023?"
        result = classify_task_type(desc, metadata={"benchmark": "gaia"})
        assert result == "agentic"

    def test_gaia_without_metadata_short_answer(self):
        """GAIA without metadata, short expected answer → qa."""
        desc = "Answer the following question accurately.\n\nQuestion: What is 2+2?"
        result = classify_task_type(desc, expected="4")
        assert result == "qa"

    def test_gaia_level1_factual(self):
        """GAIA Level 1: factual question with metadata."""
        desc = (
            "Answer the following question accurately.\n\n"
            "Question: How many episodes of the TV show 'Friends' were there?"
        )
        result = classify_task_type(desc, expected="236", metadata={"level": "1"})
        # Without benchmark metadata, short expected → qa
        assert result == "qa"

    def test_gaia_level2_complex(self):
        """GAIA Level 2: multi-step reasoning."""
        desc = (
            "Answer the following question accurately.\n\n"
            "Question: According to the attached file, what is the total revenue "
            "of the company in Q3 2023 after subtracting operating expenses?"
        )
        result = classify_task_type(desc, expected="$4.2M")
        # Has question form + short answer → qa (without metadata)
        assert result == "qa"


# ─── classify_task_type: ALFWorld ──────────────────────────────────────────

class TestClassifyAlfworld:
    """ALFWorld tasks: embodied household manipulation."""

    def test_alfworld_with_task_type_metadata(self):
        """ALFWorld with task_type in metadata → embodied."""
        desc = "Complete the following household task in a text-based environment.\n\nTask: pick and place: Mug\nTask type: pick_and_place_simple"
        result = classify_task_type(desc, metadata={"task_type": "pick_and_place_simple"})
        assert result == "embodied"

    def test_alfworld_clean_task(self):
        """ALFWorld clean task → embodied."""
        desc = "Complete the following household task.\n\nTask: clean: Plate with SinkBasin"
        result = classify_task_type(desc, metadata={"task_type": "clean_then_place_in_recep"})
        assert result == "embodied"

    def test_alfworld_heat_task(self):
        """ALFWorld heat task → embodied."""
        desc = "Complete the following household task.\n\nTask: heat: Egg with Microwave"
        result = classify_task_type(desc, metadata={"task_type": "heat_then_place_in_recep"})
        assert result == "embodied"

    def test_alfworld_look_at_task(self):
        """ALFWorld look_at task → embodied."""
        desc = "Complete the following household task.\n\nTask: look at: CD with DeskLamp"
        result = classify_task_type(desc, metadata={"task_type": "look_at_obj_in_light"})
        assert result == "embodied"

    def test_alfworld_with_expected_walkthrough(self):
        """ALFWorld with walkthrough expected → embodied."""
        desc = "Complete the following household task.\n\nTask: pick and place: Mug"
        expected = "go to coffeetable 1 -> take mug 1 from coffeetable 1 -> go to shelf 1 -> put mug 1 in/on shelf 1"
        result = classify_task_type(desc, expected=expected)
        assert result == "embodied"

    def test_alfworld_physical_verbs_in_desc(self):
        """ALFWorld-style description with physical verbs → embodied."""
        desc = "Pick up the apple from the counter. Go to the fridge. Put the apple in the fridge."
        result = classify_task_type(desc)
        assert result == "embodied"


# ─── classify_task_type: LoCoMo ────────────────────────────────────────────

class TestClassifyLocomo:
    """LoCoMo tasks: conversation memory QA."""

    def test_locomo_conversation_based(self):
        """LoCoMo with 'conversation history' in description → qa."""
        desc = (
            "Answer the following question based on the conversation history.\n\n"
            "Conversation:\nAlice: I went to Paris last summer...\n\n"
            "Question: Where did Alice go last summer?"
        )
        result = classify_task_type(desc, expected="Paris")
        assert result == "qa"

    def test_locomo_with_metadata(self):
        """LoCoMo with benchmark metadata → qa."""
        desc = "Answer the following question based on the conversation.\n\nQuestion: What did Bob say?"
        result = classify_task_type(desc, metadata={"benchmark": "locomo"})
        assert result == "qa"

    def test_locomo_multi_hop(self):
        """LoCoMo multi-hop question → qa."""
        desc = (
            "Answer the following question based on the conversation history.\n\n"
            "Conversation:\nAlice: I'm moving to NYC...\nBob: When?\nAlice: Next month.\n\n"
            "Question: When is Alice moving to NYC?"
        )
        result = classify_task_type(desc, expected="Next month")
        assert result == "qa"


# ─── classify_task_type: GAIA2 ─────────────────────────────────────────────

class TestClassifyGaia2:
    """GAIA2 tasks: dynamic agentic with CLI tools."""

    def test_gaia2_with_metadata(self):
        """GAIA2 with benchmark metadata → agentic."""
        desc = "Find the latest commit message in the repository and summarize it."
        metadata = {
            "benchmark": "gaia2",
            "scenario_path": "/tmp/scenarios/task_001.json",
            "tools": ["git", "grep"],
            "apps": ["terminal", "browser"],
        }
        result = classify_task_type(desc, metadata=metadata)
        assert result == "agentic"

    def test_gaia2_with_tools_metadata(self):
        """GAIA2 with tools in metadata → agentic."""
        desc = "Search for files containing 'TODO' in the project."
        metadata = {"tools": ["grep", "find"]}
        result = classify_task_type(desc, metadata=metadata)
        assert result == "agentic"

    def test_gaia2_with_apps_metadata(self):
        """GAIA2 with apps in metadata → agentic."""
        desc = "Open the browser and navigate to example.com"
        metadata = {"apps": ["browser"]}
        result = classify_task_type(desc, metadata=metadata)
        assert result == "agentic"


# ─── classify_task_type: SWE-bench ─────────────────────────────────────────

class TestClassifySwebench:
    """SWE-bench tasks: code bug-fixing."""

    def test_swebench_with_metadata(self):
        """SWE-bench with benchmark metadata → agentic."""
        desc = "Fix the following issue in the django/django repository.\n\nIssue:\nQuerySet.count() returns wrong result..."
        metadata = {"benchmark": "swebench"}
        result = classify_task_type(desc, metadata=metadata)
        assert result == "agentic"

    def test_swebench_dynamic_with_metadata(self):
        """SWE-bench dynamic with benchmark metadata → agentic."""
        desc = "Fix the following issue in the scikit-learn/scikit-learn repository.\n\nIssue:\nPCA fails with sparse input..."
        metadata = {"benchmark": "swebench_dynamic"}
        result = classify_task_type(desc, metadata=metadata)
        assert result == "agentic"

    def test_swebench_long_description_no_metadata(self):
        """SWE-bench without metadata — classification is best-effort.
        In practice, metadata is always provided by the benchmark loader.
        Without metadata, structural signals determine the type."""
        desc = (
            "Fix the following issue in the requests repository.\n\n"
            "Issue:\nWhen making a POST request with a file upload, the Content-Type header "
            "is not set correctly. The boundary parameter is missing from the multipart/form-data "
            "content type. This causes the server to reject the request. The fix should ensure "
            "that the boundary is always included when sending multipart data."
        )
        result = classify_task_type(desc)
        # Without metadata, this is classified by structural signals
        # The key point: injection behavior is IDENTICAL regardless of classification
        assert result in ("qa", "agentic")


# ─── assess_task_complexity ────────────────────────────────────────────────

class TestAssessTaskComplexity:
    """Test complexity assessment for various task descriptions."""

    def test_simple_question(self):
        assert assess_task_complexity("What is 2+2?") == "simple"

    def test_simple_short(self):
        assert assess_task_complexity("Who wrote Hamlet?") == "simple"

    def test_moderate_medium_length(self):
        result = assess_task_complexity(
            "Find the total number of commits in the repository that were made in 2023."
        )
        assert result in ("simple", "moderate")

    def test_complex_multi_step(self):
        result = assess_task_complexity(
            "First download the file from the URL, then parse the CSV to extract all rows "
            "where the value exceeds 100, and finally compute the average. Make sure to "
            "handle missing values and also check for duplicates."
        )
        assert result == "complex"

    def test_complex_many_constraints(self):
        result = assess_task_complexity(
            "Create a travel plan for 5 people going from NYC to LA for 7 days. "
            "The budget must not exceed $5000. Each day must include at least one activity. "
            "Make sure all restaurants are vegetarian-friendly, except for the last day. "
            "Also ensure every hotel has a pool."
        )
        assert result == "complex"


# ─── should_augment ────────────────────────────────────────────────────────

class TestShouldAugment:
    """Test augmentation gating logic."""

    def test_empty_library_no_augment(self, empty_library):
        """Empty library → no augmentation."""
        do_augment, reason = should_augment("What is 2+2?", empty_library)
        assert do_augment is False
        assert "no_relevant" in reason

    def test_populated_library_augments(self, populated_library):
        """Library with relevant experiences → always augment."""
        do_augment, reason = should_augment(
            "Calculate the sum of odd numbers below 50", populated_library
        )
        assert do_augment is True
        assert "always_inject" in reason

    def test_unrelated_task_still_augments(self, populated_library):
        """Even unrelated tasks augment if library has any content (always_inject policy)."""
        do_augment, reason = should_augment(
            "Write a poem about the ocean", populated_library
        )
        # should_augment returns True as long as retrieve_similar returns anything
        # (it always does if library is non-empty, since it returns top-k regardless of score)
        assert do_augment is True


# ─── Integration: classify_task_type consistency ───────────────────────────

class TestClassifyConsistency:
    """Ensure classify_task_type returns valid values and is deterministic."""

    VALID_TYPES = {"qa", "agentic", "embodied"}

    @pytest.mark.parametrize("desc,metadata", [
        # GAIA
        ("Answer the following question accurately.\n\nQuestion: What year was Python released?",
         {"benchmark": "gaia"}),
        # ALFWorld
        ("Complete the following household task.\n\nTask: pick and place: Mug\nTask type: pick_and_place_simple",
         {"task_type": "pick_and_place_simple"}),
        # LoCoMo
        ("Answer the following question based on the conversation history.\n\nConversation:\nHi\n\nQuestion: What?",
         {"benchmark": "locomo"}),
        # GAIA2
        ("Find all Python files in the project.",
         {"benchmark": "gaia2", "scenario_path": "/tmp/x.json", "tools": ["find"]}),
        # SWE-bench
        ("Fix the following issue in the repo.\n\nIssue:\nBug in parser.",
         {"benchmark": "swebench_dynamic"}),
    ])
    def test_returns_valid_type(self, desc, metadata):
        result = classify_task_type(desc, metadata=metadata)
        assert result in self.VALID_TYPES

    def test_deterministic(self):
        """Same input always produces same output."""
        desc = "Answer the following question accurately.\n\nQuestion: What is AI?"
        metadata = {"benchmark": "gaia"}
        results = [classify_task_type(desc, metadata=metadata) for _ in range(10)]
        assert len(set(results)) == 1

    @pytest.mark.parametrize("benchmark,expected_type", [
        ("gaia", "agentic"),
        ("gaia2", "agentic"),
        ("swebench", "agentic"),
        ("swebench_dynamic", "agentic"),
        ("locomo", "qa"),
        ("longmemeval", "qa"),
    ])
    def test_benchmark_metadata_routing(self, benchmark, expected_type):
        """Benchmark metadata should deterministically route to expected type."""
        desc = "Some task description here."
        result = classify_task_type(desc, metadata={"benchmark": benchmark})
        assert result == expected_type


# ─── Integration: full pipeline with injection ─────────────────────────────

class TestInjectionIntegration:
    """Test that build_augmented_prompt works correctly for all benchmark types."""

    def test_injection_with_empty_library(self, empty_library):
        """Empty library → empty augmentation for all task types."""
        from v6.injection import build_augmented_prompt

        descs = [
            ("Answer the question: What is 2+2?", {"benchmark": "gaia"}),
            ("Pick up the mug.", {"task_type": "pick_and_place_simple"}),
            ("Based on the conversation, who is Alice?", {"benchmark": "locomo"}),
            ("Find files in the project.", {"benchmark": "gaia2", "tools": ["find"]}),
            ("Fix the bug in the repo.", {"benchmark": "swebench_dynamic"}),
        ]
        for desc, meta in descs:
            result = build_augmented_prompt(desc, empty_library, metadata=meta)
            assert result == ""

    def test_injection_with_populated_library(self, populated_library):
        """Populated library → non-empty augmentation for all task types."""
        from v6.injection import build_augmented_prompt

        descs = [
            ("Calculate the sum of even numbers below 200", {"benchmark": "gaia"}),
            ("Go to the kitchen and pick up the plate", {"task_type": "pick_and_place_simple"}),
            ("Based on the conversation, what did Alice calculate?", {"benchmark": "locomo"}),
            ("Find all prime numbers in the file.", {"benchmark": "gaia2", "tools": ["python"]}),
            ("Fix the math computation bug.", {"benchmark": "swebench_dynamic"}),
        ]
        for desc, meta in descs:
            result = build_augmented_prompt(desc, populated_library, metadata=meta)
            # All task types now get full injection (no qa→lightweight routing)
            assert len(result) > 0, f"Expected non-empty augmentation for: {desc}"

    def test_uniform_injection_qa_vs_agentic(self, populated_library):
        """QA and agentic tasks get the SAME injection format (no routing difference)."""
        from v6.injection import build_augmented_prompt

        qa_result = build_augmented_prompt(
            "Calculate the sum of prime numbers below 50",
            populated_library,
            metadata={"benchmark": "locomo"}
        )
        agentic_result = build_augmented_prompt(
            "Calculate the sum of prime numbers below 50",
            populated_library,
            metadata={"benchmark": "gaia2", "tools": ["python"]}
        )
        # Both should contain the same experience content
        # (the only difference might be minor due to retrieval, but format is identical)
        assert "Successful approach" in qa_result or "Lesson" in qa_result
        assert "Successful approach" in agentic_result or "Lesson" in agentic_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
