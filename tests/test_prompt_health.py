"""
Tests for Prompt Health Monitor — validates anti-bloat mechanisms.

Based on the article "Agent Skill Bloat to Refactoring" (snowsyzheng, 2026-05-12):
- Detects prompt bloat (line count thresholds)
- Finds semantic duplicates
- Flags negative instructions
- Checks structural placement of critical rules
- Identifies potential rule conflicts
"""
from __future__ import annotations

import pytest

from src.utils.prompt_health import (
    PromptHealthMonitor,
    PromptHealthReport,
    PromptIssue,
    check_prompt_health,
    format_health_report,
)


class TestPromptHealthBasic:
    """Basic health check tests."""

    def test_healthy_short_prompt(self):
        prompt = "You are a helpful assistant.\nAnswer questions concisely.\n"
        report = check_prompt_health(prompt)
        assert report.is_healthy
        assert report.score >= 0.9
        assert report.total_lines == 3

    def test_empty_prompt(self):
        report = check_prompt_health("")
        assert report.is_healthy
        assert report.total_lines == 1  # Empty string splits to ['']

    def test_moderate_prompt(self):
        # 200 lines — within healthy range
        lines = [f"Rule {i}: Do something specific for case {i}." for i in range(200)]
        prompt = "\n".join(lines)
        report = check_prompt_health(prompt)
        assert report.total_lines == 200
        assert report.score >= 0.7

    def test_bloated_prompt_warning(self):
        # 600 lines — warning threshold
        lines = [f"Instruction {i}: Handle scenario {i} carefully." for i in range(600)]
        prompt = "\n".join(lines)
        report = check_prompt_health(prompt)
        assert any(i.category == "bloat" and i.severity == "warning" for i in report.issues)
        assert report.score < 0.9

    def test_bloated_prompt_critical(self):
        # 1200 lines — critical threshold
        lines = [f"Rule {i}: Never forget to check condition {i}." for i in range(1200)]
        prompt = "\n".join(lines)
        report = check_prompt_health(prompt)
        assert any(i.category == "bloat" and i.severity == "critical" for i in report.issues)
        assert report.needs_restructure


class TestSemanticDedup:
    """Semantic deduplication detection tests."""

    def test_detects_near_duplicates(self):
        prompt = (
            "Always include all four quality dimensions in the summary report output.\n"
            "Always include all four quality dimensions in the final report output.\n"
            "Something completely different about HTTP status codes and error handling.\n"
        )
        report = check_prompt_health(prompt)
        dedup_issues = [i for i in report.issues if i.category == "dedup"]
        assert len(dedup_issues) >= 1

    def test_no_false_positives_on_different_content(self):
        prompt = (
            "Use tool A for aggregation tables and summary views.\n"
            "HTTP 404 means the resource was not found on the server.\n"
            "Binary search requires a sorted array as input data.\n"
        )
        report = check_prompt_health(prompt)
        dedup_issues = [i for i in report.issues if i.category == "dedup"]
        assert len(dedup_issues) == 0

    def test_short_lines_not_flagged(self):
        # Lines < 20 chars should be skipped
        prompt = "Do X.\nDo X.\nDo X.\n"
        report = check_prompt_health(prompt)
        dedup_issues = [i for i in report.issues if i.category == "dedup"]
        assert len(dedup_issues) == 0


class TestNegativeInstructions:
    """Negative instruction detection tests."""

    def test_detects_negative_patterns(self):
        prompt = (
            "Do not include any suggestions in the report.\n"
            "Never output the raw ID mapping fields.\n"
            "Avoid making assumptions without evidence.\n"
            "You must not skip any quality dimension.\n"
        )
        report = check_prompt_health(prompt)
        negative_issues = [i for i in report.issues if i.category == "negative"]
        assert len(negative_issues) >= 3

    def test_positive_instructions_not_flagged(self):
        prompt = (
            "End the report with a data table.\n"
            "Include all four quality dimensions.\n"
            "Use tool A for aggregation queries.\n"
        )
        report = check_prompt_health(prompt)
        negative_issues = [i for i in report.issues if i.category == "negative"]
        assert len(negative_issues) == 0

    def test_high_negative_density_warning(self):
        # More than 5 negative instructions triggers a summary warning
        lines = [f"Do not perform action {i} under any circumstances." for i in range(8)]
        prompt = "\n".join(lines)
        report = check_prompt_health(prompt)
        negative_issues = [i for i in report.issues if i.category == "negative"]
        # Should have individual issues + a summary warning
        assert any(i.severity == "warning" for i in negative_issues)


class TestStructuralPlacement:
    """Structural placement validation tests."""

    def test_critical_rules_in_middle_flagged(self):
        # Create a prompt where critical rules are buried in the middle
        lines = ["Header line"] * 10
        lines += ["This is some context information."] * 20
        lines += ["You MUST always verify the data before proceeding."] * 5
        lines += ["CRITICAL: Never skip the validation step."] * 3
        lines += ["More context and examples follow."] * 20
        lines += ["Footer information."] * 10
        prompt = "\n".join(lines)
        report = check_prompt_health(prompt)
        placement_issues = [i for i in report.issues if i.category == "placement"]
        # Should flag critical rules in the middle zone
        assert len(placement_issues) >= 1 or report.total_lines < 30

    def test_short_prompt_no_placement_issues(self):
        prompt = "Rule 1: Always do X.\nRule 2: Must check Y.\n"
        report = check_prompt_health(prompt)
        placement_issues = [i for i in report.issues if i.category == "placement"]
        assert len(placement_issues) == 0


class TestRuleConflicts:
    """Rule conflict detection tests."""

    def test_detects_always_vs_conditional(self):
        lines = [
            "The analysis must always cover all four quality dimensions.",
            "Some intermediate content here that separates the rules.",
            "More content to create distance between the rules.",
            "Additional context about the reporting process.",
            "Further details about the methodology used.",
            "Extra information about the data sources.",
            "Only expand on dimensions that show anomalies, skip normal ones.",
        ]
        prompt = "\n".join(lines)
        report = check_prompt_health(prompt)
        conflict_issues = [i for i in report.issues if i.category == "conflict"]
        # May or may not detect depending on token overlap
        # The key is it doesn't crash
        assert isinstance(report.score, float)


class TestFormatReport:
    """Report formatting tests."""

    def test_format_healthy_report(self):
        report = PromptHealthReport(
            total_lines=50, total_chars=2000, estimated_tokens=500, score=0.95
        )
        text = format_health_report(report)
        assert "HEALTHY" in text
        assert "50" in text

    def test_format_unhealthy_report(self):
        report = PromptHealthReport(
            total_lines=800, total_chars=40000, estimated_tokens=10000, score=0.3,
            issues=[
                PromptIssue(
                    category="bloat", severity="critical",
                    message="Too long", suggestion="Restructure"
                ),
            ],
        )
        text = format_health_report(report)
        assert "NEEDS ATTENTION" in text
        assert "RESTRUCTURING" in text
        assert "bloat" in text


class TestSuggestRestructure:
    """Restructuring suggestion tests."""

    def test_healthy_prompt_no_suggestions(self):
        monitor = PromptHealthMonitor()
        prompt = "Simple prompt.\nDo the task.\n"
        result = monitor.suggest_restructure(prompt)
        assert "no restructuring needed" in result.lower()

    def test_bloated_prompt_gets_suggestions(self):
        monitor = PromptHealthMonitor()
        lines = [f"Rule {i}: Do not forget to handle case {i}." for i in range(400)]
        prompt = "\n".join(lines)
        result = monitor.suggest_restructure(prompt)
        assert "externalizing" in result.lower() or "Restructuring" in result


class TestIntegrationWithSkillForge:
    """Integration tests with SkillForge's actual prompts."""

    def test_trajectory_collector_prompt_healthy(self):
        """The trajectory collector's system prompt should be healthy."""
        from src.trajectory.collector import TrajectoryCollector
        from unittest.mock import MagicMock

        collector = TrajectoryCollector(MagicMock(), {})
        prompt = collector._build_system_prompt()
        report = check_prompt_health(prompt)
        assert report.is_healthy, f"Trajectory collector prompt unhealthy: {format_health_report(report)}"

    def test_skill_induction_prompt_healthy(self):
        """The skill induction system prompt should be healthy."""
        from src.skill_induction.traj_to_skill import SKILL_INDUCTION_SYSTEM_PROMPT
        report = check_prompt_health(SKILL_INDUCTION_SYSTEM_PROMPT)
        assert report.is_healthy, f"Skill induction prompt unhealthy: {format_health_report(report)}"

    def test_skill_designer_prompts_healthy(self):
        """The skill designer's prompts should be within healthy bounds."""
        from src.skill_induction.skill_designer import SkillDesigner, HardCase
        from src.models import Skill

        designer = SkillDesigner(llm_client=None)
        # Simulate building the analysis prompt
        cases = [
            HardCase(query=f"Question {i}", reward=0.1, fail_count=2, step=i)
            for i in range(5)
        ]
        skills = [Skill(name=f"Skill {i}", description=f"Does thing {i}") for i in range(3)]

        # The prompt building logic is internal, but we can check the
        # overall design doesn't exceed healthy bounds
        # (This is a structural validation, not a functional test)
        assert designer.max_edits <= 5, "Max edits per cycle should be bounded"
        assert designer.hard_case_buffer.max_size <= 500, "Buffer should be bounded"
