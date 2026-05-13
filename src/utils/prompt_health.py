"""
Prompt Health Monitor — prevents skill/prompt bloat via structural checks.

Implements anti-bloat mechanisms inspired by the article
"Agent Skill Bloat to Refactoring" (snowsyzheng, 2026-05-12):

1. Line Budget: enforce max prompt length, trigger restructure mode
2. Semantic Dedup: detect near-duplicate rules/instructions
3. Structural Layering: validate priority placement (critical rules first)
4. Positive Instruction Bias: flag negative instructions that should be positive
5. Conflict Detection: find implicit rule conflicts

Key insight from the article:
  "The more you emphasize, the longer the document;
   the longer the document, the more attention is diluted;
   the more diluted, the more violations;
   the more violations, the more you add another emphasis."

This module breaks that cycle by enforcing structural health.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ============================================================
# Data Structures
# ============================================================


@dataclass
class PromptHealthReport:
    """Health report for a prompt/skill document."""

    total_lines: int = 0
    total_chars: int = 0
    estimated_tokens: int = 0
    issues: list[PromptIssue] = field(default_factory=list)
    score: float = 1.0  # 0.0 = critical, 1.0 = healthy

    @property
    def is_healthy(self) -> bool:
        return self.score >= 0.7 and not any(
            i.severity == "critical" for i in self.issues
        )

    @property
    def needs_restructure(self) -> bool:
        return self.score < 0.5 or self.total_lines > 500


@dataclass
class PromptIssue:
    """A single issue found in a prompt."""

    category: str  # "bloat" | "dedup" | "placement" | "negative" | "conflict"
    severity: str  # "info" | "warning" | "critical"
    message: str
    line_range: tuple[int, int] | None = None
    suggestion: str = ""


# ============================================================
# Prompt Health Monitor
# ============================================================


class PromptHealthMonitor:
    """
    Monitors prompt/skill health and prevents bloat.

    Applies the article's engineering solutions:
    - Solution 1: Layered architecture (position = priority)
    - Solution 2: Eliminate implicit conflicts
    - Solution 3: Positive instructions over negations
    - Solution 4: Structured format (tables > prose)
    - Solution 5: Instruction sandwich (first/last emphasis)
    - Solution 6: Externalize reference content
    - Solution 7: Reference-rule consistency
    - Solution 8: Anti-bloat brakes for auto-repair
    """

    # Thresholds (from article's "sweet spot" analysis)
    MAX_LINES_HEALTHY = 300
    MAX_LINES_WARNING = 500
    MAX_LINES_CRITICAL = 1000
    MAX_RULE_REPETITIONS = 2
    CHARS_PER_TOKEN = 4.0

    # Negative instruction patterns (article Solution 3)
    NEGATIVE_PATTERNS = [
        r"\bdo\s+not\b",
        r"\bnever\b",
        r"\bdon'?t\b",
        r"\bforbid(?:den)?\b",
        r"\bprohibit(?:ed)?\b",
        r"\bmust\s+not\b",
        r"\bshould\s+not\b",
        r"\bavoid\b",
        r"\bstrictly\s+(?:forbidden|prohibited)\b",
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.max_lines = self.config.get("max_lines", self.MAX_LINES_HEALTHY)
        self.max_repetitions = self.config.get(
            "max_repetitions", self.MAX_RULE_REPETITIONS
        )

    def check(self, prompt_text: str) -> PromptHealthReport:
        """
        Run all health checks on a prompt/skill text.

        Args:
            prompt_text: The full prompt or skill document text.

        Returns:
            PromptHealthReport with issues and score.
        """
        lines = prompt_text.split("\n")
        report = PromptHealthReport(
            total_lines=len(lines),
            total_chars=len(prompt_text),
            estimated_tokens=int(len(prompt_text) / self.CHARS_PER_TOKEN),
        )

        # Run all checks
        self._check_length(report, lines)
        self._check_semantic_dedup(report, lines)
        self._check_negative_instructions(report, lines)
        self._check_structural_placement(report, lines)
        self._check_rule_conflicts(report, lines)

        # Compute overall score
        report.score = self._compute_score(report)

        return report

    def suggest_restructure(self, prompt_text: str) -> str:
        """
        Suggest a restructured version of a bloated prompt.

        Applies the article's layered architecture:
        - Layer 1 (top): Critical rules (first 10%)
        - Layer 2: Core methodology
        - Layer 3: Execution flow (middle, can tolerate some loss)
        - Layer 4 (bottom): Output format (recency effect)

        Returns:
            Restructuring suggestions as text.
        """
        report = self.check(prompt_text)
        if report.is_healthy:
            return "Prompt is healthy, no restructuring needed."

        suggestions = ["## Restructuring Suggestions\n"]

        if report.total_lines > self.MAX_LINES_HEALTHY:
            suggestions.append(
                f"⚠️ Prompt is {report.total_lines} lines "
                f"(target: <{self.MAX_LINES_HEALTHY}). "
                f"Consider externalizing reference content."
            )

        dedup_issues = [i for i in report.issues if i.category == "dedup"]
        if dedup_issues:
            suggestions.append(
                f"\n### Duplicate Rules ({len(dedup_issues)} found)\n"
                "Merge these into a single authoritative statement at the top:"
            )
            for issue in dedup_issues:
                suggestions.append(f"  - {issue.message}")

        negative_issues = [i for i in report.issues if i.category == "negative"]
        if negative_issues:
            suggestions.append(
                f"\n### Negative Instructions ({len(negative_issues)} found)\n"
                "Convert to positive instructions (article Solution 3):"
            )
            for issue in negative_issues[:5]:
                suggestions.append(f"  - {issue.message}")
                if issue.suggestion:
                    suggestions.append(f"    → {issue.suggestion}")

        return "\n".join(suggestions)

    # ================================================================
    # Individual Checks
    # ================================================================

    def _check_length(self, report: PromptHealthReport, lines: list[str]) -> None:
        """Check prompt length against thresholds."""
        n = len(lines)
        if n > self.MAX_LINES_CRITICAL:
            report.issues.append(PromptIssue(
                category="bloat",
                severity="critical",
                message=f"Prompt is {n} lines (critical threshold: {self.MAX_LINES_CRITICAL})",
                suggestion="Immediate restructuring required. Externalize reference content.",
            ))
        elif n > self.MAX_LINES_WARNING:
            report.issues.append(PromptIssue(
                category="bloat",
                severity="warning",
                message=f"Prompt is {n} lines (warning threshold: {self.MAX_LINES_WARNING})",
                suggestion="Consider splitting into core rules + external references.",
            ))
        elif n > self.MAX_LINES_HEALTHY:
            report.issues.append(PromptIssue(
                category="bloat",
                severity="info",
                message=f"Prompt is {n} lines (healthy target: <{self.MAX_LINES_HEALTHY})",
            ))

    def _check_semantic_dedup(
        self, report: PromptHealthReport, lines: list[str]
    ) -> None:
        """Detect semantically similar lines (potential duplicates)."""
        # Use token-set overlap to find near-duplicates
        line_tokens: list[set[str]] = []
        meaningful_lines: list[tuple[int, str]] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip empty lines, comments, separators
            if not stripped or stripped.startswith("#") or len(stripped) < 20:
                line_tokens.append(set())
                continue
            tokens = set(stripped.lower().split())
            line_tokens.append(tokens)
            meaningful_lines.append((i, stripped))

        # Find pairs with high overlap
        seen_duplicates: set[str] = set()
        for idx_a in range(len(meaningful_lines)):
            i, line_a = meaningful_lines[idx_a]
            tokens_a = line_tokens[i]
            if not tokens_a or len(tokens_a) < 5:
                continue

            for idx_b in range(idx_a + 1, len(meaningful_lines)):
                j, line_b = meaningful_lines[idx_b]
                tokens_b = line_tokens[j]
                if not tokens_b or len(tokens_b) < 5:
                    continue

                # Jaccard similarity
                intersection = tokens_a & tokens_b
                union = tokens_a | tokens_b
                sim = len(intersection) / len(union) if union else 0

                if sim > 0.7:
                    key = f"{min(i,j)}-{max(i,j)}"
                    if key not in seen_duplicates:
                        seen_duplicates.add(key)
                        report.issues.append(PromptIssue(
                            category="dedup",
                            severity="warning",
                            message=(
                                f"Near-duplicate lines {i+1} and {j+1}: "
                                f"'{line_a[:60]}...' ≈ '{line_b[:60]}...'"
                            ),
                            line_range=(i + 1, j + 1),
                            suggestion="Merge into one authoritative statement.",
                        ))

    def _check_negative_instructions(
        self, report: PromptHealthReport, lines: list[str]
    ) -> None:
        """Flag negative instructions that could be rewritten positively."""
        negative_re = re.compile(
            "|".join(self.NEGATIVE_PATTERNS), re.IGNORECASE
        )

        negative_count = 0
        for i, line in enumerate(lines):
            if negative_re.search(line):
                negative_count += 1
                if negative_count <= 5:  # Only report first 5
                    report.issues.append(PromptIssue(
                        category="negative",
                        severity="info",
                        message=f"Line {i+1}: Negative instruction: '{line.strip()[:80]}'",
                        line_range=(i + 1, i + 1),
                        suggestion="Rewrite as positive: specify WHAT to do, not what to avoid.",
                    ))

        if negative_count > 5:
            report.issues.append(PromptIssue(
                category="negative",
                severity="warning",
                message=f"Total {negative_count} negative instructions found (showing first 5)",
                suggestion=(
                    "High negative instruction density increases violation probability. "
                    "Convert to positive table format: [Scenario | ✅ Do This | ❌ Avoid]"
                ),
            ))

    def _check_structural_placement(
        self, report: PromptHealthReport, lines: list[str]
    ) -> None:
        """Check if critical rules are in the attention-optimal positions."""
        if len(lines) < 20:
            return  # Too short to have placement issues

        # Look for "critical" keywords in the middle zone (attention dead zone)
        critical_keywords = [
            "must", "critical", "important", "always", "never",
            "required", "mandatory", "essential",
        ]
        critical_re = re.compile(
            r"\b(" + "|".join(critical_keywords) + r")\b", re.IGNORECASE
        )

        total = len(lines)
        top_10_pct = int(total * 0.1)
        bottom_10_pct = total - int(total * 0.1)
        middle_start = int(total * 0.3)
        middle_end = int(total * 0.7)

        middle_critical_count = 0
        for i in range(middle_start, middle_end):
            if i < len(lines) and critical_re.search(lines[i]):
                middle_critical_count += 1

        if middle_critical_count > 3:
            report.issues.append(PromptIssue(
                category="placement",
                severity="warning",
                message=(
                    f"{middle_critical_count} critical rules found in the middle zone "
                    f"(lines {middle_start+1}-{middle_end}), where attention is weakest"
                ),
                suggestion=(
                    "Move critical rules to the first 10% (primacy effect) "
                    "or last 10% (recency effect) of the prompt."
                ),
            ))

    def _check_rule_conflicts(
        self, report: PromptHealthReport, lines: list[str]
    ) -> None:
        """Detect potential implicit conflicts between rules."""
        # Simple heuristic: look for contradictory patterns
        # e.g., "always include X" vs "only include X when Y"
        always_rules: list[tuple[int, str]] = []
        conditional_rules: list[tuple[int, str]] = []

        always_re = re.compile(r"\b(always|must|every)\b", re.IGNORECASE)
        conditional_re = re.compile(
            r"\b(only\s+(?:if|when)|except|unless|but\s+not)\b", re.IGNORECASE
        )

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) < 10:
                continue
            if always_re.search(stripped):
                always_rules.append((i, stripped))
            if conditional_re.search(stripped):
                conditional_rules.append((i, stripped))

        # Check for potential conflicts (same topic, different conditions)
        for a_idx, a_text in always_rules:
            a_tokens = set(a_text.lower().split())
            for c_idx, c_text in conditional_rules:
                c_tokens = set(c_text.lower().split())
                overlap = a_tokens & c_tokens
                # If they share significant vocabulary, might conflict
                if len(overlap) > 5 and abs(a_idx - c_idx) > 5:
                    report.issues.append(PromptIssue(
                        category="conflict",
                        severity="info",
                        message=(
                            f"Potential conflict: line {a_idx+1} (always-rule) "
                            f"vs line {c_idx+1} (conditional-rule)"
                        ),
                        line_range=(a_idx + 1, c_idx + 1),
                        suggestion="Make priority explicit: which rule wins when they conflict?",
                    ))
                    break  # One conflict per always-rule is enough

    # ================================================================
    # Scoring
    # ================================================================

    def _compute_score(self, report: PromptHealthReport) -> float:
        """Compute overall health score (0.0 - 1.0)."""
        score = 1.0

        # Length penalty
        if report.total_lines > self.MAX_LINES_CRITICAL:
            score -= 0.4
        elif report.total_lines > self.MAX_LINES_WARNING:
            score -= 0.2
        elif report.total_lines > self.MAX_LINES_HEALTHY:
            score -= 0.1

        # Issue penalties
        for issue in report.issues:
            if issue.severity == "critical":
                score -= 0.3
            elif issue.severity == "warning":
                score -= 0.1
            elif issue.severity == "info":
                score -= 0.02

        return max(0.0, min(1.0, score))


# ============================================================
# Convenience Functions
# ============================================================


def check_prompt_health(prompt_text: str) -> PromptHealthReport:
    """Quick health check on a prompt string."""
    monitor = PromptHealthMonitor()
    return monitor.check(prompt_text)


def format_health_report(report: PromptHealthReport) -> str:
    """Format a health report as human-readable text."""
    lines = [
        f"Prompt Health: {'✅ HEALTHY' if report.is_healthy else '⚠️ NEEDS ATTENTION'}",
        f"  Score: {report.score:.2f}/1.00",
        f"  Lines: {report.total_lines} | Chars: {report.total_chars} | ~Tokens: {report.estimated_tokens}",
    ]

    if report.issues:
        lines.append(f"  Issues ({len(report.issues)}):")
        for issue in report.issues[:10]:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}[issue.severity]
            lines.append(f"    {icon} [{issue.category}] {issue.message}")
            if issue.suggestion:
                lines.append(f"       → {issue.suggestion}")

    if report.needs_restructure:
        lines.append("  ⚡ RESTRUCTURING RECOMMENDED")

    return "\n".join(lines)
