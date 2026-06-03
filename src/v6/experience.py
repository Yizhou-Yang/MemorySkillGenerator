"""
SkillForge V6 — Experience dataclass + ExperienceLibrary.

Core data structures for storing, serializing, and retrieving task experiences.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailureTaxonomy:
    """Structured failure analysis (4 categories from TRACE finding #6).
    
    Categories:
    - model_failure: reasoning/planning errors (recoverable with better prompt)
    - tool_failure: CLI/API errors, timeouts, format issues (NOT model's fault)
    - task_mismatch: agent fundamentally misunderstood the task
    - over_action: agent did 2x+ more actions than needed (exploration waste)
    """
    category: str = ""
    root_cause: str = ""
    is_tool_chain: bool = False  # True → don't inject as "lesson" (it's env noise)
    recoverable: bool = True


@dataclass
class Experience:
    """Enhanced experience record with version history and failure taxonomy.
    
    Fields:
        task_id:            Unique identifier for this task instance
        task_desc:          Natural language task description
        tool_sequence:      Ordered list of tools/actions used by agent
        action_commands:    Raw commands executed (for replay/debugging)
        outcome:            "success" | "partial" | "failure"
        score:              Numeric score (0-1), fraction of oracle steps completed
        missing_steps:      Oracle steps that agent failed to execute
        extra_steps:        Agent steps not in oracle (exploration waste)
        failure_reason:     Human-readable reason for failure
        
        # V6 additions:
        failure_taxonomy:   Dict with FailureTaxonomy fields + AI refinement fields
        token_cost:         Total tokens consumed by agent
        time_cost:          Wall-clock seconds
        task_complexity:    "simple" | "moderate" | "complex"
        augmentation_used:  What experience text was injected (empty = baseline)
        augmentation_helped: Did injection improve outcome vs prior attempt?
        version:            Which attempt is this on the same task? (1, 2, 3...)
        patch_history:      [{from_version, to_version, score_delta, new_steps, ...}]
        timestamp:          Unix epoch of recording
    """
    task_id: str
    task_desc: str
    tool_sequence: list[str]
    action_commands: list[str]
    outcome: str
    score: float
    missing_steps: list[str]
    extra_steps: list[str]
    failure_reason: str
    failure_taxonomy: dict = field(default_factory=dict)
    token_cost: int = 0
    time_cost: float = 0.0
    task_complexity: str = ""
    augmentation_used: str = ""
    augmentation_helped: bool | None = None
    version: int = 1
    patch_history: list = field(default_factory=list)
    timestamp: float = 0.0


class ExperienceLibrary:
    """Storage + retrieval for experiences with augmentation effectiveness tracking.
    
    Retrieval uses bag-of-words Jaccard similarity (sufficient for small libraries
    of 20-100 experiences; scales to O(n) linear scan).
    
    Augmentation effectiveness is tracked per-complexity-level: if past injections
    for "simple" tasks historically hurt performance, the gate will block future
    augmentation for simple tasks.
    """
    
    def __init__(self):
        self.experiences: list[Experience] = []
        self._augment_stats: dict[str, dict] = {}  # complexity → {helped, hurt, neutral}
    
    def record(self, exp: Experience):
        """Add experience and update augmentation stats."""
        self.experiences.append(exp)
        if exp.augmentation_used:
            key = exp.task_complexity or "unknown"
            if key not in self._augment_stats:
                self._augment_stats[key] = {"helped": 0, "hurt": 0, "neutral": 0}
            if exp.augmentation_helped is True:
                self._augment_stats[key]["helped"] += 1
            elif exp.augmentation_helped is False:
                self._augment_stats[key]["hurt"] += 1
            else:
                self._augment_stats[key]["neutral"] += 1
    
    def retrieve_similar(self, task_desc: str, top_k: int = 3,
                         outcome_filter: str | None = None,
                         exclude_tool_failures: bool = False) -> list[Experience]:
        """Retrieve most similar experiences by word overlap.
        
        Args:
            task_desc: Query task description
            top_k: Max results
            outcome_filter: "success" | "failure" | "partial" | None (all)
            exclude_tool_failures: If True, skip experiences where failure was
                                   caused by tool/env issues (not model reasoning)
        """
        candidates = self.experiences
        if outcome_filter:
            candidates = [e for e in candidates if e.outcome == outcome_filter]
        if exclude_tool_failures:
            candidates = [e for e in candidates
                         if not e.failure_taxonomy.get("is_tool_chain", False)]
        
        if not candidates:
            return []
        
        # Jaccard similarity on word sets (minus stop words)
        stop_words = {"the", "a", "an", "to", "and", "or", "in", "on", "at", "for",
                      "of", "with", "is", "are", "was", "were", "be", "been", "being",
                      "that", "this", "it", "my", "all", "i", "me"}
        task_words = set(task_desc.lower().split()) - stop_words
        
        scored = []
        for exp in candidates:
            exp_words = set(exp.task_desc.lower().split()) - stop_words
            if not exp_words:
                continue
            overlap = len(task_words & exp_words) / max(len(task_words | exp_words), 1)
            scored.append((overlap, exp))
        
        scored.sort(key=lambda x: -x[0])
        return [exp for _, exp in scored[:top_k]]
    
    def get_augmentation_effectiveness(self, task_complexity: str) -> float:
        """Historical success rate of augmentation for this complexity level.
        
        Returns 0.7 (default optimistic) if < 3 data points.
        """
        stats = self._augment_stats.get(task_complexity, {})
        total = stats.get("helped", 0) + stats.get("hurt", 0) + stats.get("neutral", 0)
        if total < 3:
            return 0.7
        return stats.get("helped", 0) / total
    
    def get_avg_token_overhead(self) -> float:
        """Ratio: avg tokens for augmented tasks / avg tokens for baseline tasks."""
        augmented = [e for e in self.experiences if e.augmentation_used and e.token_cost > 0]
        non_augmented = [e for e in self.experiences if not e.augmentation_used and e.token_cost > 0]
        if not augmented or not non_augmented:
            return 1.0
        avg_aug = sum(e.token_cost for e in augmented) / len(augmented)
        avg_no = sum(e.token_cost for e in non_augmented) / len(non_augmented)
        return avg_aug / avg_no if avg_no > 0 else 1.0
    
    def get_successful(self) -> list[Experience]:
        return [e for e in self.experiences if e.outcome == "success"]
    
    def get_failed(self) -> list[Experience]:
        return [e for e in self.experiences if e.outcome in ("failure", "partial")]
    
    def to_dict(self) -> dict:
        return {
            "experiences": [
                {
                    "task_id": e.task_id, "task_desc": e.task_desc,
                    "tool_sequence": e.tool_sequence, "action_commands": e.action_commands,
                    "outcome": e.outcome, "score": e.score,
                    "missing_steps": e.missing_steps, "extra_steps": e.extra_steps,
                    "failure_reason": e.failure_reason,
                    "failure_taxonomy": e.failure_taxonomy,
                    "token_cost": e.token_cost, "time_cost": e.time_cost,
                    "task_complexity": e.task_complexity,
                    "augmentation_used": e.augmentation_used,
                    "augmentation_helped": e.augmentation_helped,
                    "timestamp": e.timestamp,
                    "version": e.version,
                    "patch_history": e.patch_history,
                }
                for e in self.experiences
            ],
            "augment_stats": self._augment_stats,
        }
    
    def from_dict(self, data: dict | list):
        if isinstance(data, list):
            for d in data:
                d.setdefault("failure_taxonomy", {})
                d.setdefault("token_cost", 0)
                d.setdefault("time_cost", 0.0)
                d.setdefault("task_complexity", "")
                d.setdefault("augmentation_used", "")
                d.setdefault("augmentation_helped", None)
                d.setdefault("version", 1)
                d.setdefault("patch_history", [])
                self.experiences.append(Experience(**d))
        else:
            for d in data.get("experiences", []):
                d.setdefault("version", 1)
                d.setdefault("patch_history", [])
                self.experiences.append(Experience(**d))
            self._augment_stats = data.get("augment_stats", {})
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                self.from_dict(json.load(f))
