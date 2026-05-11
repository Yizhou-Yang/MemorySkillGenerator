"""
Multi-judge verifier — breaks the echo chamber with diverse evaluation.

Implements the External Verifier mechanism (P0) from Mem2Evolve analysis:
- Multiple judge prompts with different perspectives
- Majority voting to reduce single-judge bias
- EM/F1 as primary objective metrics (no LLM dependency)

Reference: docs/internal/mem2evolve_analysis.md §9.2 "LLM-as-Judge echo chamber"
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.models import Skill
from src.utils.llm import LLMClient


class MultiJudgeVerifier:
    """
    Multi-judge evaluation to break the LLM-as-Judge echo chamber.

    Uses multiple judge prompts with different evaluation perspectives,
    then aggregates via majority voting (median score).

    This addresses the "positive feedback loop of error" where a single
    LLM judge can systematically confirm its own biases.
    """

    # Different judge personas to reduce systematic bias
    JUDGE_PERSONAS = [
        {
            "name": "strict_correctness",
            "system": (
                "You are a strict correctness evaluator. "
                "Focus ONLY on whether the answer is factually correct. "
                "Ignore style, verbosity, and reasoning quality. "
                "If the answer is wrong, score 0-3 regardless of reasoning."
            ),
            "weight": 1.0,
        },
        {
            "name": "methodology_critic",
            "system": (
                "You are a methodology critic. "
                "Focus on whether the reasoning process is sound and complete. "
                "A correct answer with flawed reasoning should score lower than "
                "a correct answer with solid reasoning."
            ),
            "weight": 0.8,
        },
        {
            "name": "adversarial_checker",
            "system": (
                "You are an adversarial checker looking for flaws. "
                "Actively try to find errors, hallucinations, or unsupported claims. "
                "Be skeptical of confident-sounding but potentially wrong answers. "
                "Score harshly — most responses should get 4-6."
            ),
            "weight": 0.9,
        },
    ]

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or {}
        self.num_judges: int = self.config.get("num_judges", 3)

    def verify(
        self,
        task_description: str,
        expected_answer: str,
        actual_response: str,
        skill_name: str = "",
    ) -> dict[str, Any]:
        """
        Run multi-judge verification on a response.

        Args:
            task_description: The original task.
            expected_answer: Ground truth answer.
            actual_response: The agent's response.
            skill_name: Name of the skill used (for context).

        Returns:
            Dict with 'median_score', 'scores', 'consensus', and 'details'.
        """
        if self.llm_client is None:
            logger.warning("[MultiJudge] No LLM client, returning default")
            return {
                "median_score": 5.0,
                "scores": [],
                "consensus": False,
                "details": [],
            }

        judges = self.JUDGE_PERSONAS[: self.num_judges]
        scores: list[float] = []
        details: list[dict[str, Any]] = []

        for judge in judges:
            score, reason = self._single_judge(
                judge_persona=judge,
                task_description=task_description,
                expected_answer=expected_answer,
                actual_response=actual_response,
                skill_name=skill_name,
            )
            scores.append(score)
            details.append({
                "judge": judge["name"],
                "score": score,
                "reason": reason,
                "weight": judge["weight"],
            })

        # Compute weighted median
        if scores:
            sorted_scores = sorted(scores)
            median_score = sorted_scores[len(sorted_scores) // 2]
        else:
            median_score = 0.0

        # Consensus: all judges agree within 2 points
        consensus = (max(scores) - min(scores) <= 2.0) if scores else False

        result = {
            "median_score": median_score,
            "scores": scores,
            "consensus": consensus,
            "details": details,
        }

        logger.info(
            f"[MultiJudge] Scores={scores}, median={median_score:.1f}, "
            f"consensus={consensus}"
        )
        return result

    def _single_judge(
        self,
        judge_persona: dict[str, Any],
        task_description: str,
        expected_answer: str,
        actual_response: str,
        skill_name: str,
    ) -> tuple[float, str]:
        """Run a single judge evaluation."""
        prompt = f"""Score the following response on a 0-10 scale.

Task: {task_description[:500]}
Expected answer: {expected_answer[:200]}
Actual response (using skill "{skill_name}"):
{actual_response[:800]}

Return JSON: {{"score": <0-10>, "reason": "<one sentence>"}}"""

        messages = [
            {"role": "system", "content": judge_persona["system"]},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm_client.chat_json(messages, temperature=0.1)
            data = json.loads(response)
            score = float(data.get("score", 5.0))
            reason = data.get("reason", "")
            return max(0.0, min(10.0, score)), reason
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error(f"[MultiJudge] Judge '{judge_persona['name']}' failed: {exc}")
            return 5.0, f"Parse error: {exc}"

    def should_accept_skill(
        self,
        verification_results: list[dict[str, Any]],
        em_scores: list[float],
        acceptance_threshold: float = 0.5,
    ) -> bool:
        """
        Decide whether a skill should be accepted into the library.

        Uses a combination of:
        1. EM scores (primary, objective) — at least 50% tasks must pass
        2. Multi-judge median (secondary, reference) — median >= 6.0

        This breaks the echo chamber by requiring BOTH objective and
        subjective criteria to pass.

        Args:
            verification_results: List of multi-judge results.
            em_scores: List of EM scores (0.0 or 1.0).
            acceptance_threshold: Minimum fraction of EM=1 required.

        Returns:
            True if the skill should be accepted.
        """
        if not em_scores:
            return False

        # Primary criterion: EM pass rate
        em_pass_rate = sum(1 for s in em_scores if s >= 1.0) / len(em_scores)

        # Secondary criterion: median judge score
        median_scores = [r.get("median_score", 0) for r in verification_results]
        avg_median = sum(median_scores) / len(median_scores) if median_scores else 0

        accepted = em_pass_rate >= acceptance_threshold and avg_median >= 6.0

        logger.info(
            f"[MultiJudge] Acceptance decision: em_pass_rate={em_pass_rate:.1%}, "
            f"avg_median_judge={avg_median:.1f}, accepted={accepted}"
        )
        return accepted
