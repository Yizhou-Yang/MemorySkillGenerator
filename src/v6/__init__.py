"""SkillForge V6 — orchestrator module."""
from __future__ import annotations
import json

from .experience import Experience, ExperienceLibrary, FailureTaxonomy
from .gate import assess_task_complexity, should_augment, classify_task_type
from .injection import (build_augmented_prompt, format_success_experience,
                        format_failure_experience, estimate_token_count)
from .analysis import analyze_execution, classify_failure
from .refine import ai_review_experience, cross_agent_evaluate_skill, critic_refine_experience, _format_patch_history

class SkillForgeV6:
    """Orchestrates: record_experience → version tracking → AI refine → injection."""

    def __init__(self, library_path: str | None = None, token_budget: int = 2000):
        self.library = ExperienceLibrary()
        self.token_budget = token_budget
        self._gate_log: list[dict] = []
        if library_path:
            self.library.load(library_path)

    def get_augmentation(self, task_desc: str, **kwargs) -> tuple[str, dict]:
        complexity = assess_task_complexity(task_desc)
        do_augment, reason = should_augment(task_desc, self.library)
        meta = {"gated": not do_augment, "reason": reason,
                "complexity": complexity, "token_estimate": 0}
        if not do_augment:
            self._gate_log.append(meta)
            return "", meta
        augmentation = build_augmented_prompt(task_desc, self.library,
                                              token_budget=self.token_budget, **kwargs)
        meta["token_estimate"] = estimate_token_count(augmentation)
        self._gate_log.append(meta)
        return augmentation, meta

    def record_experience(self, task_id: str, task_desc: str,
                          agent_actions: list[dict], oracle_actions: list[dict],
                          token_cost: int = 0, time_cost: float = 0.0,
                          augmentation_used: str = "",
                          baseline_score: float | None = None,
                          llm_reviewer=None,
                          critic_fn=None, critic_threshold: int = 5):
        exp = analyze_execution(task_id, task_desc, agent_actions, oracle_actions,
                                token_cost=token_cost, time_cost=time_cost,
                                augmentation_used=augmentation_used)
        if baseline_score is not None and augmentation_used:
            exp.augmentation_helped = exp.score > baseline_score

        # Version history — find previous attempts at same/similar task
        prev = self._find_previous_versions(task_id, task_desc)
        if prev:
            latest = prev[-1]
            exp.version = latest.version + 1
            exp.patch_history = latest.patch_history + [{
                "from_version": latest.version, "to_version": exp.version,
                "score_delta": exp.score - latest.score,
                "outcome_change": f"{latest.outcome} → {exp.outcome}",
                "new_steps": [s for s in exp.tool_sequence if s not in latest.tool_sequence],
                "removed_steps": [s for s in latest.tool_sequence if s not in exp.tool_sequence],
                "fixed_missing": [s for s in latest.missing_steps if s not in exp.missing_steps],
                "new_missing": [s for s in exp.missing_steps if s not in latest.missing_steps],
            }]

        # AI refinement
        review = ai_review_experience(exp, llm_fn=llm_reviewer)
        exp.failure_taxonomy.update({
            "ai_refined": review.get("refined", False),
            "causal_lesson": review.get("causal_lesson", ""),
            "avoidance_note": review.get("avoidance_note", ""),
            "transferability": review.get("transferability", ""),
            "generalized_steps": review.get("generalized_steps", ""),
            "evolution_insight": review.get("evolution_insight", ""),
            "quality_score": review.get("quality_score", 0),
        })
        if exp.patch_history:
            exp.failure_taxonomy["evolution_trace"] = [
                f"v{p['from_version']}→v{p['to_version']}: "
                + (f"fixed {p['fixed_missing']}" if p.get("fixed_missing") else f"+{p.get('score_delta',0):.0%}")
                for p in exp.patch_history if p.get("fixed_missing") or p.get("score_delta", 0) > 0
            ]

        # Cross-agent critic: ALWAYS evaluate, low-score triggers forced refine (never discard)
        if critic_fn is not None:
            from .refine import cross_agent_evaluate_skill, critic_refine_experience
            verdict = cross_agent_evaluate_skill(exp, llm_fn=critic_fn)
            exp.failure_taxonomy["critic_quality"] = verdict.get("total", 5)
            exp.failure_taxonomy["critic_verdict"] = verdict.get("verdict", "inject")

            # Low quality → forced refine/expand (never discard information)
            if verdict.get("total", 5) < critic_threshold:
                refinement = critic_refine_experience(exp, verdict, llm_fn=critic_fn)
                if refinement.get("enhanced"):
                    # Expand existing fields — never overwrite with shorter content
                    existing_steps = exp.failure_taxonomy.get("generalized_steps", "")
                    enhanced_steps = refinement.get("enhanced_steps", "")
                    if len(enhanced_steps) > len(existing_steps):
                        exp.failure_taxonomy["generalized_steps"] = enhanced_steps

                    existing_causal = exp.failure_taxonomy.get("causal_lesson", "")
                    enhanced_causal = refinement.get("enhanced_causal_lesson", "")
                    if len(enhanced_causal) > len(existing_causal):
                        exp.failure_taxonomy["causal_lesson"] = enhanced_causal

                    existing_avoid = exp.failure_taxonomy.get("avoidance_note", "")
                    enhanced_avoid = refinement.get("enhanced_avoidance", "")
                    if len(enhanced_avoid) > len(existing_avoid):
                        exp.failure_taxonomy["avoidance_note"] = enhanced_avoid

                    existing_transfer = exp.failure_taxonomy.get("transferability", "")
                    enhanced_transfer = refinement.get("enhanced_transferability", "")
                    if len(enhanced_transfer) > len(existing_transfer):
                        exp.failure_taxonomy["transferability"] = enhanced_transfer

                    # Add new fields (recovery strategies, preconditions)
                    if refinement.get("recovery_strategies"):
                        exp.failure_taxonomy["recovery_strategies"] = refinement["recovery_strategies"]
                    if refinement.get("preconditions"):
                        exp.failure_taxonomy["preconditions"] = refinement["preconditions"]

                    exp.failure_taxonomy["critic_refined"] = True
                    exp.failure_taxonomy["critic_quality_post_refine"] = refinement.get("quality_score", 5)

        self.library.record(exp)
        return exp

    def _find_previous_versions(self, task_id: str, task_desc: str) -> list[Experience]:
        exact = [e for e in self.library.experiences if e.task_id == task_id]
        if exact:
            return sorted(exact, key=lambda e: e.version)
        stop_words = {"the","a","an","to","and","or","in","on","at","for","of","with",
                      "is","are","was","were","be","been","that","this","it","my","all","i","me"}
        task_words = set(task_desc.lower().split()) - stop_words
        similar = [e for e in self.library.experiences
                   if len(task_words & (set(e.task_desc.lower().split()) - stop_words))
                   / max(len(task_words | (set(e.task_desc.lower().split()) - stop_words)), 1) > 0.7]
        return sorted(similar, key=lambda e: e.version) if similar else []

    def save(self, path: str):
        data = self.library.to_dict()
        data["gate_log"] = self._gate_log
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        self.library.load(path)

    @property
    def stats(self) -> dict:
        return {
            "total_experiences": len(self.library.experiences),
            "success": len(self.library.get_successful()),
            "failed": len(self.library.get_failed()),
            "gate_decisions": len(self._gate_log),
            "gated_out": sum(1 for g in self._gate_log if g.get("gated")),
            "augmented": sum(1 for g in self._gate_log if not g.get("gated")),
            "avg_token_overhead": f"{self.library.get_avg_token_overhead():.2f}x",
        }
