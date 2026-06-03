"""Experience dataclass + ExperienceLibrary with n-gram similarity retrieval."""
from __future__ import annotations
import json
import os
import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class FailureTaxonomy:
    category: str = ""          # model_failure | tool_failure | task_mismatch | over_action
    root_cause: str = ""
    is_tool_chain: bool = False
    recoverable: bool = True


@dataclass
class Experience:
    task_id: str
    task_desc: str
    tool_sequence: list[str]
    action_commands: list[str]
    outcome: str                    # "success" | "partial" | "failure"
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


def _tokenize(text: str) -> list[str]:
    """Lowercase split + basic normalization."""
    import re
    return re.findall(r'[a-z0-9]+', text.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _compute_similarity(query_text: str, doc_text: str) -> float:
    """Hybrid similarity: unigram TF overlap + bigram bonus.
    
    Better than Jaccard: captures word frequency and word-order (bigram).
    No external dependencies (no sklearn/numpy needed).
    """
    q_tokens = _tokenize(query_text)
    d_tokens = _tokenize(doc_text)
    if not q_tokens or not d_tokens:
        return 0.0

    # Unigram: TF-weighted overlap (not just set intersection)
    q_tf = Counter(q_tokens)
    d_tf = Counter(d_tokens)
    shared_terms = set(q_tf) & set(d_tf)
    if not shared_terms:
        return 0.0

    # Cosine on TF vectors (lightweight, no IDF needed for small corpus)
    dot = sum(q_tf[t] * d_tf[t] for t in shared_terms)
    norm_q = math.sqrt(sum(v * v for v in q_tf.values()))
    norm_d = math.sqrt(sum(v * v for v in d_tf.values()))
    unigram_sim = dot / (norm_q * norm_d) if norm_q > 0 and norm_d > 0 else 0.0

    # Bigram bonus: rewards matching word order
    q_bigrams = set(_ngrams(q_tokens, 2))
    d_bigrams = set(_ngrams(d_tokens, 2))
    if q_bigrams and d_bigrams:
        bigram_overlap = len(q_bigrams & d_bigrams) / max(len(q_bigrams | d_bigrams), 1)
    else:
        bigram_overlap = 0.0

    return 0.7 * unigram_sim + 0.3 * bigram_overlap


class ExperienceLibrary:
    def __init__(self):
        self.experiences: list[Experience] = []
        self._augment_stats: dict[str, dict] = {}

    def record(self, exp: Experience):
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
        candidates = self.experiences
        if outcome_filter:
            candidates = [e for e in candidates if e.outcome == outcome_filter]
        if exclude_tool_failures:
            candidates = [e for e in candidates
                         if not e.failure_taxonomy.get("is_tool_chain", False)]
        if not candidates:
            return []

        scored = [((_compute_similarity(task_desc, exp.task_desc)), exp) for exp in candidates]
        scored.sort(key=lambda x: -x[0])
        return [exp for _, exp in scored[:top_k]]

    def get_augmentation_effectiveness(self, task_complexity: str) -> float:
        stats = self._augment_stats.get(task_complexity, {})
        total = stats.get("helped", 0) + stats.get("hurt", 0) + stats.get("neutral", 0)
        if total < 3:
            return 0.7
        return stats.get("helped", 0) / total

    def get_avg_token_overhead(self) -> float:
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
                    "version": e.version, "patch_history": e.patch_history,
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
