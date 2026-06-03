"""
SkillForge V6 — EvoMem + Applicability Gate + Cost-Aware + Version-Conditioned AI Refine.

Module structure:
  experience.py  — Experience dataclass + ExperienceLibrary (storage + retrieval)
  gate.py        — Applicability Gate + Task Type Classification
  injection.py   — Cost-Aware Prompt Augmentation (routing + formatting)
  analysis.py    — Execution Analysis + Failure Taxonomy
  refine.py      — Version-Conditioned AI Refinement

Usage:
    from v6 import SkillForgeV6, build_augmented_prompt
    
    sf = SkillForgeV6(token_budget=2000)
    sf.record_experience(task_id, task_desc, agent_actions, oracle_actions)
    augmentation = build_augmented_prompt(task_desc, sf.library)
"""
from __future__ import annotations
import json

# Re-export all public APIs
from .experience import Experience, ExperienceLibrary, FailureTaxonomy
from .gate import assess_task_complexity, should_augment, classify_task_type
from .injection import (
    build_augmented_prompt, format_success_experience, format_failure_experience,
    estimate_token_count,
)
from .analysis import analyze_execution, classify_failure
from .refine import ai_review_experience, _format_patch_history


class SkillForgeV6:
    """Main entry point: orchestrates experience recording, version tracking, and injection.
    
    Lifecycle:
    1. record_experience() — after each task execution
       - analyze_execution() → compute score, missing steps, failure taxonomy
       - _find_previous_versions() → link to prior attempts on same task
       - ai_review_experience() → version-conditioned refinement (if llm_reviewer provided)
       - library.record() → persist
       
    2. get_augmentation() / build_augmented_prompt() — before next task
       - classify_task_type() → qa / agentic / embodied
       - should_augment() → gate check
       - format + inject relevant experiences within token budget
    """
    
    def __init__(self, library_path: str | None = None, token_budget: int = 2000):
        self.library = ExperienceLibrary()
        self.token_budget = token_budget
        self._gate_log: list[dict] = []
        
        if library_path:
            self.library.load(library_path)
    
    def get_augmentation(self, task_desc: str, **kwargs) -> tuple[str, dict]:
        """Get experience augmentation with gate decision metadata."""
        complexity = assess_task_complexity(task_desc)
        do_augment, reason = should_augment(task_desc, self.library)
        
        meta = {"gated": not do_augment, "reason": reason,
                "complexity": complexity, "token_estimate": 0}
        
        if not do_augment:
            self._gate_log.append(meta)
            return "", meta
        
        augmentation = build_augmented_prompt(
            task_desc, self.library, token_budget=self.token_budget, **kwargs
        )
        meta["token_estimate"] = estimate_token_count(augmentation)
        self._gate_log.append(meta)
        return augmentation, meta
    
    def record_experience(self, task_id: str, task_desc: str,
                          agent_actions: list[dict], oracle_actions: list[dict],
                          token_cost: int = 0, time_cost: float = 0.0,
                          augmentation_used: str = "",
                          baseline_score: float | None = None,
                          llm_reviewer=None):
        """Record experience with version history + AI refinement.
        
        Version history: links to prior attempts on same task → computes patch diff.
        AI refinement: version-conditioned prompt sees full evolution trace.
        """
        exp = analyze_execution(
            task_id, task_desc, agent_actions, oracle_actions,
            token_cost=token_cost, time_cost=time_cost,
            augmentation_used=augmentation_used,
        )
        
        if baseline_score is not None and augmentation_used:
            exp.augmentation_helped = exp.score > baseline_score
        
        # ─── Version History (EvoMem-inspired) ────────────────────────
        prev_versions = self._find_previous_versions(task_id, task_desc)
        
        if prev_versions:
            latest = prev_versions[-1]
            exp.version = latest.version + 1
            patch = {
                "from_version": latest.version,
                "to_version": exp.version,
                "score_delta": exp.score - latest.score,
                "outcome_change": f"{latest.outcome} → {exp.outcome}",
                "new_steps": [s for s in exp.tool_sequence if s not in latest.tool_sequence],
                "removed_steps": [s for s in latest.tool_sequence if s not in exp.tool_sequence],
                "fixed_missing": [s for s in latest.missing_steps if s not in exp.missing_steps],
                "new_missing": [s for s in exp.missing_steps if s not in latest.missing_steps],
            }
            exp.patch_history = latest.patch_history + [patch]
        
        # ─── AI Refinement (version-conditioned) ──────────────────────
        review_result = ai_review_experience(exp, llm_fn=llm_reviewer)
        
        exp.failure_taxonomy["ai_refined"] = review_result.get("refined", False)
        exp.failure_taxonomy["causal_lesson"] = review_result.get("causal_lesson", "")
        exp.failure_taxonomy["avoidance_note"] = review_result.get("avoidance_note", "")
        exp.failure_taxonomy["transferability"] = review_result.get("transferability", "")
        exp.failure_taxonomy["generalized_steps"] = review_result.get("generalized_steps", "")
        exp.failure_taxonomy["evolution_insight"] = review_result.get("evolution_insight", "")
        exp.failure_taxonomy["quality_score"] = review_result.get("quality_score", 0)
        
        if exp.patch_history:
            lessons = []
            for p in exp.patch_history:
                if p.get("fixed_missing"):
                    lessons.append(f"v{p['from_version']}→v{p['to_version']}: fixed {p['fixed_missing']}")
                if p.get("score_delta", 0) > 0:
                    lessons.append(f"v{p['from_version']}→v{p['to_version']}: +{p['score_delta']:.0%} by adding {p.get('new_steps', [])[:2]}")
            exp.failure_taxonomy["evolution_trace"] = lessons
        
        self.library.record(exp)
    
    def _find_previous_versions(self, task_id: str, task_desc: str) -> list[Experience]:
        """Find prior attempts: exact task_id match, or high word overlap (>70%)."""
        exact = [e for e in self.library.experiences if e.task_id == task_id]
        if exact:
            return sorted(exact, key=lambda e: e.version)
        
        stop_words = {"the", "a", "an", "to", "and", "or", "in", "on", "at", "for",
                      "of", "with", "is", "are", "was", "were", "be", "been",
                      "that", "this", "it", "my", "all", "i", "me"}
        task_words = set(task_desc.lower().split()) - stop_words
        
        similar = []
        for exp in self.library.experiences:
            exp_words = set(exp.task_desc.lower().split()) - stop_words
            if exp_words:
                overlap = len(task_words & exp_words) / max(len(task_words | exp_words), 1)
                if overlap > 0.7:
                    similar.append(exp)
        
        return sorted(similar, key=lambda e: e.version) if similar else []
    
    def save(self, path: str):
        data = self.library.to_dict()
        data["gate_log"] = self._gate_log
        data["token_budget"] = self.token_budget
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        self.library.load(path)
    
    @property
    def stats(self) -> dict:
        gated = sum(1 for g in self._gate_log if g.get("gated"))
        augmented = sum(1 for g in self._gate_log if not g.get("gated"))
        return {
            "total_experiences": len(self.library.experiences),
            "success": len(self.library.get_successful()),
            "failed": len(self.library.get_failed()),
            "gate_decisions": len(self._gate_log),
            "gated_out": gated,
            "augmented": augmented,
            "avg_token_overhead": f"{self.library.get_avg_token_overhead():.2f}x",
        }
