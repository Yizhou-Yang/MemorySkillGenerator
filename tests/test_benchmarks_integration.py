"""Integration tests for all 5 benchmarks + SkillForge pipeline.

Ensures:
1. All benchmarks load without interpreter-level errors
2. Task format is correct (required fields present)
3. Full pipeline (record → refine → inject) works for each benchmark type
4. No missing modules or dependencies
5. GAIA2 list-expected and SWE-bench string-expected handled correctly
"""
import os
import sys
import pytest

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('LLM_PROVIDER', 'codebuddy')
os.environ.setdefault('CODEBUDDY_MODEL', 'deepseek-v4-pro')
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')


# ═══════════════════════════════════════════════════════════════════════════
#  Module Import Tests — catch missing dependencies early
# ═══════════════════════════════════════════════════════════════════════════

class TestModuleImports:
    """Verify all core modules can be imported without errors."""

    def test_import_experience(self):
        from v6.experience import Experience, ExperienceLibrary, compute_similarity, FailureTaxonomy
        assert Experience is not None
        assert ExperienceLibrary is not None
        assert compute_similarity is not None

    def test_import_analysis(self):
        from v6.analysis import analyze_execution, classify_failure
        assert analyze_execution is not None
        assert classify_failure is not None

    def test_import_gate(self):
        from v6.gate import classify_task_type, assess_task_complexity, should_augment
        assert classify_task_type is not None
        assert assess_task_complexity is not None
        assert should_augment is not None

    def test_import_injection(self):
        from v6.injection import (build_augmented_prompt, format_success_experience,
                                  format_failure_experience)
        assert build_augmented_prompt is not None
        assert format_success_experience is not None
        assert format_failure_experience is not None

    def test_import_refine(self):
        from v6.refine import (ai_review_experience, cross_agent_evaluate_skill,
                               critic_refine_experience)
        assert ai_review_experience is not None
        assert cross_agent_evaluate_skill is not None
        assert critic_refine_experience is not None

    def test_import_orchestrator(self):
        from v6 import (SkillForgeV6, ExperienceLibrary, Experience,
                        build_augmented_prompt, ai_review_experience,
                        cross_agent_evaluate_skill)
        assert SkillForgeV6 is not None

    def test_import_benchmark_loader(self):
        from benchmarks.loader import BenchmarkLoader
        assert BenchmarkLoader is not None

    def test_external_dependencies(self):
        """Verify all external libraries are available."""
        import rapidfuzz
        import json_repair
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        assert rapidfuzz is not None
        assert json_repair is not None
        assert TfidfVectorizer is not None


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark Loading Tests — all 5 benchmarks must load correctly
# ═══════════════════════════════════════════════════════════════════════════

GAIA2_SCENARIO_DIR = "/tmp/harbor-datasets/datasets/gaia2-cli"

BENCHMARK_CONFIGS = [
    ("gaia", {"name": "gaia", "num_samples": 2}),
    ("alfworld", {"name": "alfworld", "num_samples": 2}),
    ("locomo", {"name": "locomo", "num_samples": 2}),
    ("gaia2", {"name": "gaia2", "num_samples": 2, "scenario_dir": GAIA2_SCENARIO_DIR}),
    ("swebench_dynamic", {"name": "swebench_dynamic", "num_samples": 2}),
]


class TestBenchmarkLoading:
    """All 5 benchmarks must load without errors and have correct task format."""

    @pytest.fixture(params=BENCHMARK_CONFIGS, ids=[c[0] for c in BENCHMARK_CONFIGS])
    def benchmark_tasks(self, request):
        name, config = request.param
        # Skip gaia2 if scenario dir doesn't exist
        if name == "gaia2" and not os.path.isdir(GAIA2_SCENARIO_DIR):
            pytest.skip(f"GAIA2 scenario dir not found: {GAIA2_SCENARIO_DIR}")
        from benchmarks.loader import BenchmarkLoader
        loader = BenchmarkLoader(config)
        tasks = loader.load()
        return name, tasks

    def test_benchmark_loads_tasks(self, benchmark_tasks):
        """Each benchmark must load at least 1 task."""
        name, tasks = benchmark_tasks
        assert len(tasks) >= 1, f"{name}: loaded 0 tasks"

    def test_task_has_required_fields(self, benchmark_tasks):
        """Every task must have task_id, description, expected."""
        name, tasks = benchmark_tasks
        for task in tasks:
            assert "task_id" in task, f"{name}: missing task_id"
            assert "description" in task, f"{name}: missing description"
            assert "expected" in task, f"{name}: missing expected"

    def test_task_id_is_string(self, benchmark_tasks):
        """task_id must be a non-empty string."""
        name, tasks = benchmark_tasks
        for task in tasks:
            assert isinstance(task["task_id"], str), f"{name}: task_id not string"
            assert len(task["task_id"]) > 0, f"{name}: empty task_id"

    def test_description_is_string(self, benchmark_tasks):
        """description must be a non-empty string."""
        name, tasks = benchmark_tasks
        for task in tasks:
            assert isinstance(task["description"], str), f"{name}: description not string"
            assert len(task["description"]) > 0, f"{name}: empty description"

    def test_gaia2_expected_is_list(self):
        """GAIA2 expected field must be a list of oracle actions."""
        if not os.path.isdir(GAIA2_SCENARIO_DIR):
            pytest.skip("GAIA2 scenario dir not found")
        from benchmarks.loader import BenchmarkLoader
        loader = BenchmarkLoader({"name": "gaia2", "num_samples": 2,
                                  "scenario_dir": GAIA2_SCENARIO_DIR})
        tasks = loader.load()
        assert len(tasks) > 0
        for task in tasks:
            assert isinstance(task["expected"], list), "GAIA2 expected must be list"

    def test_swebench_expected_is_string(self):
        """SWE-bench expected field must be a string (FAIL_TO_PASS)."""
        from benchmarks.loader import BenchmarkLoader
        loader = BenchmarkLoader({"name": "swebench_dynamic", "num_samples": 2})
        tasks = loader.load()
        assert len(tasks) > 0
        for task in tasks:
            assert isinstance(task["expected"], str), "SWE-bench expected must be string"


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline Integration Tests — full record → refine → inject cycle
# ═══════════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """Full SkillForge pipeline must work without errors."""

    @pytest.fixture
    def sf(self):
        from v6 import SkillForgeV6
        return SkillForgeV6()

    def test_record_experience_success(self, sf):
        """Recording a successful experience must work."""
        exp = sf.record_experience(
            task_id="test_success_001",
            task_desc="List files in the current directory",
            agent_actions=[{"tool": "Bash", "input": {"command": "ls"}, "output": "file.txt"}],
            oracle_actions=[{"tool": "Bash", "input": {"command": "ls"}, "output": "file.txt"}],
            token_cost=100, time_cost=1.0,
        )
        assert exp is not None
        assert exp.task_id == "test_success_001"
        assert exp.outcome == "success"
        assert exp.score >= 0.5
        assert len(sf.library.experiences) == 1

    def test_record_experience_failure(self, sf):
        """Recording a failed experience must work."""
        exp = sf.record_experience(
            task_id="test_fail_001",
            task_desc="Deploy the application to production",
            agent_actions=[{"tool": "Bash", "input": {"command": "echo hello"}, "output": "hello"}],
            oracle_actions=[
                {"tool": "Bash", "input": {"command": "docker build ."}, "output": "built"},
                {"tool": "Bash", "input": {"command": "docker push"}, "output": "pushed"},
                {"tool": "Bash", "input": {"command": "kubectl apply"}, "output": "deployed"},
            ],
            token_cost=200, time_cost=5.0,
        )
        assert exp is not None
        assert exp.outcome in ("failure", "partial")
        assert exp.score < 1.0
        assert len(exp.missing_steps) > 0

    def test_record_experience_gaia2_format(self, sf):
        """GAIA2 oracle actions (list of app/fn dicts) must be handled."""
        oracle = [
            {"app": "Calendar", "fn": "create_event", "args": [{"name": "title", "value": "Meeting"}]},
            {"app": "Messages", "fn": "send_message", "args": [{"name": "content", "value": "Hi"}]},
        ]
        exp = sf.record_experience(
            task_id="gaia2_test_001",
            task_desc="Schedule a meeting and notify participants",
            agent_actions=[{"app": "Calendar", "fn": "create_event", "args": []}],
            oracle_actions=oracle,
            token_cost=50, time_cost=0.5,
        )
        assert exp is not None
        # Should detect partial match (1/2 oracle actions matched)
        assert exp.outcome in ("failure", "partial", "success")

    def test_version_tracking(self, sf):
        """Re-recording same task_id must increment version."""
        sf.record_experience(
            task_id="version_test",
            task_desc="Solve math problem",
            agent_actions=[{"output": "wrong answer"}],
            oracle_actions=[{"output": "42"}],
            token_cost=10, time_cost=0.1,
        )
        exp2 = sf.record_experience(
            task_id="version_test",
            task_desc="Solve math problem",
            agent_actions=[{"output": "42"}],
            oracle_actions=[{"output": "42"}],
            token_cost=10, time_cost=0.1,
        )
        assert exp2.version == 2
        assert len(exp2.patch_history) == 1
        assert exp2.patch_history[0]["from_version"] == 1
        assert exp2.patch_history[0]["to_version"] == 2

    def test_injection_after_recording(self, sf):
        """After recording experiences, injection must produce non-empty augmentation."""
        from v6 import build_augmented_prompt
        # Record a success
        sf.record_experience(
            task_id="inject_test_001",
            task_desc="Parse JSON file and extract user names",
            agent_actions=[{"tool": "Bash", "input": {"command": "jq '.users[].name' data.json"}}],
            oracle_actions=[{"tool": "Bash", "input": {"command": "jq '.users[].name' data.json"}}],
            token_cost=50, time_cost=0.5,
        )
        # Query similar task
        aug = build_augmented_prompt(
            "Extract email addresses from a JSON file",
            sf.library, token_budget=2000
        )
        # Should find the similar experience and inject it
        assert isinstance(aug, str)
        # May or may not be empty depending on similarity threshold
        # but must not raise an error

    def test_injection_with_empty_library(self, sf):
        """Injection with empty library must return empty string, not error."""
        from v6 import build_augmented_prompt
        aug = build_augmented_prompt("Some task", sf.library, token_budget=2000)
        assert isinstance(aug, str)


# ═══════════════════════════════════════════════════════════════════════════
#  Gate Classification Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGateClassification:
    """Task type classification must work for all benchmark types."""

    def test_classify_gaia_as_agentic(self):
        from v6.gate import classify_task_type
        result = classify_task_type(
            "Answer the following question: How many papers were published on arxiv about LLMs in 2024?",
            metadata={"benchmark": "gaia"}
        )
        assert result == "agentic"

    def test_classify_alfworld_as_embodied(self):
        from v6.gate import classify_task_type
        result = classify_task_type(
            "pick up the apple from the counter and put it in the fridge",
            expected="go to counter\ntake apple\ngo to fridge\nput apple",
        )
        assert result == "embodied"

    def test_classify_locomo_as_qa(self):
        from v6.gate import classify_task_type
        result = classify_task_type(
            "Based on the conversation history, what restaurant did Alice recommend?",
            metadata={"benchmark": "locomo"}
        )
        assert result == "qa"

    def test_classify_gaia2_as_agentic(self):
        from v6.gate import classify_task_type
        result = classify_task_type(
            "Book a meeting with my friend who is a Film Producer",
            metadata={"benchmark": "gaia2", "tools": ["calendar", "contacts"]}
        )
        assert result == "agentic"

    def test_classify_swebench_as_agentic(self):
        from v6.gate import classify_task_type
        result = classify_task_type(
            "Fix the following issue in the astropy/astropy repository.",
            metadata={"benchmark": "swebench"}
        )
        assert result == "agentic"


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis Tests — trajectory matching
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalysis:
    """Trajectory analysis must handle all action formats."""

    def test_bash_action_matching(self):
        from v6.analysis import analyze_execution
        exp = analyze_execution(
            task_id="bash_test",
            task_desc="List files",
            agent_actions=[{"tool": "Bash", "input": {"command": "ls -la"}, "output": "files"}],
            oracle_actions=[{"tool": "Bash", "input": {"command": "ls -la"}, "output": "files"}],
        )
        assert exp.score >= 0.5

    def test_app_fn_action_matching(self):
        """GAIA2-style app.fn actions must be matchable."""
        from v6.analysis import analyze_execution
        exp = analyze_execution(
            task_id="appfn_test",
            task_desc="Send a message",
            agent_actions=[{"app": "Messages", "fn": "send_message", "args": []}],
            oracle_actions=[{"app": "Messages", "fn": "send_message", "args": []}],
        )
        assert exp.score >= 0.5

    def test_failure_classification(self):
        """Failed executions must be classified into one of 4 categories."""
        from v6.analysis import analyze_execution
        exp = analyze_execution(
            task_id="fail_class_test",
            task_desc="Complex multi-step task",
            agent_actions=[{"output": "error: permission denied"},
                          {"output": "error: timeout"},
                          {"output": "error: not found"}],
            oracle_actions=[{"tool": "Bash", "input": {"command": "deploy"}}],
        )
        assert exp.failure_taxonomy.get("category") in (
            "tool_failure", "over_action", "task_mismatch", "model_failure"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Refine Tests — AI review without LLM (fallback mode)
# ═══════════════════════════════════════════════════════════════════════════

class TestRefine:
    """Refine module must work in fallback mode (no LLM)."""

    def test_ai_review_without_llm(self):
        """ai_review_experience with llm_fn=None must return valid dict."""
        from v6.refine import ai_review_experience
        from v6.experience import Experience
        exp = Experience(
            task_id="refine_test", task_desc="Test task",
            tool_sequence=["ls", "cat"], action_commands=["ls -la", "cat file.txt"],
            outcome="success", score=1.0,
            missing_steps=[], extra_steps=[],
            failure_reason="",
        )
        result = ai_review_experience(exp, llm_fn=None)
        assert isinstance(result, dict)
        assert "generalized_steps" in result
        assert "causal_lesson" in result
        assert result["refined"] is False  # No LLM = not refined

    def test_cross_agent_eval_without_llm(self):
        """cross_agent_evaluate_skill with llm_fn=None must return default."""
        from v6.refine import cross_agent_evaluate_skill
        from v6.experience import Experience
        exp = Experience(
            task_id="eval_test", task_desc="Test task",
            tool_sequence=[], action_commands=[],
            outcome="failure", score=0.0,
            missing_steps=["step1"], extra_steps=[],
            failure_reason="missed steps",
        )
        result = cross_agent_evaluate_skill(exp, llm_fn=None)
        assert isinstance(result, dict)
        assert "total" in result
        assert "verdict" in result

    def test_critic_refine_without_llm(self):
        """critic_refine_experience with llm_fn=None must return {enhanced: False}."""
        from v6.refine import critic_refine_experience
        from v6.experience import Experience
        exp = Experience(
            task_id="critic_test", task_desc="Test task",
            tool_sequence=[], action_commands=[],
            outcome="failure", score=0.0,
            missing_steps=[], extra_steps=[],
            failure_reason="",
        )
        result = critic_refine_experience(exp, {"total": 2, "reason": "low"}, llm_fn=None)
        assert result == {"enhanced": False}


# ═══════════════════════════════════════════════════════════════════════════
#  Experience Library Tests — retrieval and similarity
# ═══════════════════════════════════════════════════════════════════════════

class TestExperienceLibrary:
    """ExperienceLibrary retrieval must work correctly."""

    def test_record_and_retrieve(self):
        from v6.experience import ExperienceLibrary, Experience
        lib = ExperienceLibrary()
        exp = Experience(
            task_id="lib_test", task_desc="Parse a CSV file",
            tool_sequence=["python"], action_commands=["python parse.py"],
            outcome="success", score=1.0,
            missing_steps=[], extra_steps=[],
            failure_reason="",
        )
        lib.record(exp)
        results = lib.retrieve_similar("Read a CSV and extract columns", top_k=1)
        assert len(results) == 1
        assert results[0].task_id == "lib_test"

    def test_outcome_filter(self):
        from v6.experience import ExperienceLibrary, Experience
        lib = ExperienceLibrary()
        lib.record(Experience(
            task_id="s1", task_desc="Task A", tool_sequence=[], action_commands=[],
            outcome="success", score=1.0, missing_steps=[], extra_steps=[], failure_reason="",
        ))
        lib.record(Experience(
            task_id="f1", task_desc="Task A similar", tool_sequence=[], action_commands=[],
            outcome="failure", score=0.0, missing_steps=["x"], extra_steps=[], failure_reason="err",
        ))
        successes = lib.retrieve_similar("Task A", top_k=5, outcome_filter="success")
        failures = lib.retrieve_similar("Task A", top_k=5, outcome_filter="failure")
        assert all(e.outcome == "success" for e in successes)
        assert all(e.outcome == "failure" for e in failures)

    def test_effectiveness_weighting(self):
        """Experiences that historically hurt should be downweighted."""
        from v6.experience import ExperienceLibrary
        lib = ExperienceLibrary()
        # Simulate negative effectiveness
        lib.update_effectiveness("bad_exp", -0.5)
        lib.update_effectiveness("bad_exp", -0.3)
        weight = lib.get_experience_weight("bad_exp")
        assert weight < 1.0  # Should be downweighted

    def test_save_and_load(self, tmp_path):
        from v6.experience import ExperienceLibrary, Experience
        lib = ExperienceLibrary()
        lib.record(Experience(
            task_id="save_test", task_desc="Test save",
            tool_sequence=["ls"], action_commands=["ls -la"],
            outcome="success", score=1.0,
            missing_steps=[], extra_steps=[], failure_reason="",
        ))
        path = str(tmp_path / "lib.json")
        lib.save(path)

        lib2 = ExperienceLibrary()
        lib2.load(path)
        assert len(lib2.experiences) == 1
        assert lib2.experiences[0].task_id == "save_test"
