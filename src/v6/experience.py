"""Experience dataclass + ExperienceLibrary with semantic embedding retrieval."""
from __future__ import annotations
import json
import os
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

# ══════════════════════════════════════════════════════════════════════════
#  Similarity: semantic embedding with TF-cosine fallback
# ══════════════════════════════════════════════════════════════════════════

_embedding_model = None
_embedding_available = None

def _get_embedding_model():
    """Lazy-load sentence-transformers model. Returns None if unavailable."""
    global _embedding_model, _embedding_available
    if _embedding_available is False:
        return None
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        _embedding_available = True
        return _embedding_model
    except Exception:
        _embedding_available = False
        return None

def _tf_idf_fallback(query: str, doc: str) -> float:
    """Fallback: word-overlap cosine when embeddings unavailable.

    Uses CountVectorizer (not TF-IDF) because with only 2 documents,
    IDF weights are meaningless. CountVectorizer + cosine gives a
    reasonable word-overlap similarity measure.
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = CountVectorizer()
        counts = vec.fit_transform([query, doc])
        sim = float(cosine_similarity(counts[0:1], counts[1:2])[0][0])
        return sim
    except Exception:
        # Ultimate fallback: simple word overlap (Jaccard)
        q_words = set(query.lower().split())
        d_words = set(doc.lower().split())
        if not q_words or not d_words:
            return 0.0
        intersection = q_words & d_words
        union = q_words | d_words
        return len(intersection) / len(union) if union else 0.0

def compute_similarity(query: str, doc: str) -> float:
    """Semantic similarity: embedding cosine (preferred) or TF-IDF cosine (fallback)."""
    model = _get_embedding_model()
    if model is not None:
        import numpy as np
        embs = model.encode([query, doc], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))
    return _tf_idf_fallback(query, doc)

# ══════════════════════════════════════════════════════════════════════════
#  ExperienceLibrary
# ══════════════════════════════════════════════════════════════════════════

class ExperienceLibrary:
    def __init__(self):
        self.experiences: list[Experience] = []
        self._augment_stats: dict[str, dict] = {}
        self._exp_effectiveness: dict[str, dict] = {}  # exp.task_id → {injected_count, total_score_delta}

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

    def update_effectiveness(self, source_exp_id: str, score_delta: float):
        """Track per-experience injection effectiveness (EvoMem-style)."""
        if source_exp_id not in self._exp_effectiveness:
            self._exp_effectiveness[source_exp_id] = {"count": 0, "total_delta": 0.0}
        self._exp_effectiveness[source_exp_id]["count"] += 1
        self._exp_effectiveness[source_exp_id]["total_delta"] += score_delta

    def get_experience_weight(self, exp_id: str) -> float:
        """Weight for retrieval ranking: downweight experiences that historically hurt."""
        stats = self._exp_effectiveness.get(exp_id)
        if not stats or stats["count"] < 2:
            return 1.0  # Default: full weight (cold start)
        avg_delta = stats["total_delta"] / stats["count"]
        # Positive delta → weight ≥ 1; negative → weight < 1 (min 0.3)
        return max(0.3, min(1.5, 1.0 + avg_delta * 2))

    def retrieve_similar(self, task_desc: str, top_k: int = 3,
                         outcome_filter: str | None = None,
                         exclude_tool_failures: bool = False,
                         min_similarity: float = 0.1) -> list[Experience]:
        """Retrieve top-k similar experiences above minimum similarity threshold.

        Args:
            min_similarity: Minimum cosine similarity to include (default 0.1).
                Low threshold ensures early-stage experiences (when library is small)
                are still retrievable. With 1M context, injecting slightly less
                relevant experiences is acceptable — the model can ignore them.
        """
        candidates = self.experiences
        if outcome_filter:
            candidates = [e for e in candidates if e.outcome == outcome_filter]
        if exclude_tool_failures:
            candidates = [e for e in candidates
                         if not e.failure_taxonomy.get("is_tool_chain", False)]
        if not candidates:
            return []

        # Score: similarity × effectiveness weight
        scored = []
        for exp in candidates:
            sim = compute_similarity(task_desc, exp.task_desc)
            if sim < min_similarity:
                continue  # Skip irrelevant experiences
            weight = self.get_experience_weight(exp.task_id)
            scored.append((sim * weight, exp))

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
            "exp_effectiveness": self._exp_effectiveness,
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
            self._exp_effectiveness = data.get("exp_effectiveness", {})

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                self.from_dict(json.load(f))
