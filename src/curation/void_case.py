"""
SRDP Void-Case Prior c_∅ — Discrete Realization.

Theoretical Foundation (Memento-2 §3.2, Eq. 7-8):
    μ_0(c|x) = λ(x) · μ_mem(c|x) + (1 - λ(x)) · δ_{c_∅}(c)

Where:
    - μ_mem(c|x): Parzen-kernel posterior over the memory/skill library
    - c_∅: "void case" — the action of NOT injecting any skill
            (equivalent to falling back to π_LLM zero-shot policy)
    - λ(x): mass assigned to the memory-based prior, automatically
            computed by the Parzen kernel

In Memento-2 the kernel form gives:
    λ(x) = (Σ_c K(x, c)) / (1 + Σ_c K(x, c))

Hard-Threshold Realization (this module):
    Since LLM cannot "fractionally inject" a skill, we discretize
    λ(x) to {0, 1} via a similarity threshold τ_void:
        λ(x) = 1[s_max(x) ≥ τ_void]
    where s_max(x) = max_{c ∈ S} cos(emb(x), emb(c)).

When λ(x) = 0 (no skill is sufficiently relevant), the void case
is selected → the executor reduces to the zero-shot baseline π_LLM.
This unifies the per-task N*(x) phenomenon: when no skill in the
library matches the task, the framework automatically recovers
zero-shot performance instead of injecting noise.

This is NOT a new mechanism — it is a faithful (discrete) implementation
of SRDP's pre-existing void-case prior, which was overlooked in our v1-v4
implementations.

Reference:
    Memento-2 (Wang et al., 2025) §3.2 "Semantic-Retrieval Decision Process"
    SkillCurator paper §4.5 "Void-Case Realization for Per-Task N*(x)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


# ============================================================
# Configuration
# ============================================================

@dataclass
class VoidCaseConfig:
    """Configuration for the void-case prior c_∅."""

    enable: bool = True
    """Master switch — set False to disable c_∅ (fallback to v4 behavior)."""

    tau_void: float = 0.35
    """Similarity threshold for triggering c_∅.

    s_max(x) < tau_void → λ(x) = 0 → no skill injection.
    s_max(x) ≥ tau_void → λ(x) = 1 → normal A3 pipeline.

    Calibrated via LOBO-CV on training benchmarks. Default 0.35 is
    a conservative initial value that should be replaced by the
    output of scripts/calibrate_void_threshold.py.
    """

    smoothing: str = "hard"
    """Quantization mode for λ(x).

    - "hard": λ(x) ∈ {0, 1}, threshold τ_void (default, paper main result)
    - "soft": λ(x) ∈ [0, 1] continuous (ablation, prompt-strength hint)
    """

    soft_temperature: float = 5.0
    """Temperature for soft sigmoid: λ(x) = σ((s_max - τ) * T)."""

    log_decisions: bool = False
    """If True, log every gate decision (verbose, for debugging)."""


# ============================================================
# Statistics — for analysis & paper figures
# ============================================================

@dataclass
class VoidCaseStats:
    """Per-evaluation-run statistics on void-case decisions."""

    n_total: int = 0
    n_void: int = 0  # tasks routed to c_∅
    n_inject: int = 0  # tasks with skill injection
    s_max_history: list[float] = field(default_factory=list)
    lambda_history: list[float] = field(default_factory=list)

    @property
    def void_rate(self) -> float:
        return self.n_void / max(self.n_total, 1)

    @property
    def avg_s_max(self) -> float:
        return float(np.mean(self.s_max_history)) if self.s_max_history else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_void": self.n_void,
            "n_inject": self.n_inject,
            "void_rate": self.void_rate,
            "avg_s_max": self.avg_s_max,
            "s_max_history": self.s_max_history,
            "lambda_history": self.lambda_history,
        }


# ============================================================
# Core API
# ============================================================

def compute_lambda_x(
    s_max: float,
    config: VoidCaseConfig,
) -> float:
    """Compute λ(x) given the maximum query-skill similarity.

    Args:
        s_max: max cosine similarity between query embedding and
               any skill embedding in the library.
        config: VoidCaseConfig.

    Returns:
        λ(x) ∈ [0, 1]. With smoothing="hard" returns either 0.0 or 1.0.
    """
    if not config.enable:
        return 1.0  # disabled → always inject
    if config.smoothing == "hard":
        return 1.0 if s_max >= config.tau_void else 0.0
    elif config.smoothing == "soft":
        # σ((s_max - τ) * T) — smooth sigmoid around τ_void
        z = (s_max - config.tau_void) * config.soft_temperature
        return 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError(f"Unknown smoothing mode: {config.smoothing}")


def apply_void_case(
    similarities: np.ndarray | list[float],
    config: VoidCaseConfig,
    stats: VoidCaseStats | None = None,
) -> tuple[bool, float, float]:
    """Decide whether to inject skills (c_∅ gate).

    Args:
        similarities: 1D array of cosine similarities between the query
                      and every skill in the library. Can be empty (no skills).
        config: VoidCaseConfig.
        stats: Optional VoidCaseStats to update in-place for analysis.

    Returns:
        (inject, lambda_x, s_max):
            inject: True → proceed with skill injection (standard A3 path);
                    False → c_∅ chosen, executor falls back to zero-shot.
            lambda_x: λ(x) value (0.0 / 1.0 for hard, [0,1] for soft).
            s_max: maximum similarity observed (0.0 if library empty).
    """
    if similarities is None or len(similarities) == 0:
        # Empty library → always c_∅
        if stats is not None:
            stats.n_total += 1
            stats.n_void += 1
            stats.s_max_history.append(0.0)
            stats.lambda_history.append(0.0)
        if config.log_decisions:
            logger.debug("[c_∅] empty library → void case")
        return False, 0.0, 0.0

    sims = np.asarray(similarities, dtype=np.float32)
    s_max = float(sims.max())
    lam = compute_lambda_x(s_max, config)

    if config.smoothing == "hard":
        inject = lam >= 0.5
    else:
        # For soft mode we still need a binary decision for the LLM —
        # we use 0.5 as the discretization threshold but the lambda
        # value can be used downstream (e.g. as prompt-strength hint).
        inject = lam >= 0.5

    if stats is not None:
        stats.n_total += 1
        if inject:
            stats.n_inject += 1
        else:
            stats.n_void += 1
        stats.s_max_history.append(s_max)
        stats.lambda_history.append(lam)

    if config.log_decisions:
        decision = "INJECT" if inject else "VOID"
        logger.debug(f"[c_∅] s_max={s_max:.3f}, λ={lam:.3f}, decision={decision}")

    return inject, lam, s_max


def gate_topk_skills(
    query_emb: np.ndarray,
    skill_embs: np.ndarray,
    skills: list,
    k: int,
    config: VoidCaseConfig,
    stats: VoidCaseStats | None = None,
) -> tuple[list, float, float]:
    """Top-k retrieval with c_∅ gate.

    Args:
        query_emb: query embedding, shape (D,) or (1, D).
        skill_embs: skill embeddings, shape (N, D).
        skills: list of N Skill objects (parallel to skill_embs).
        k: top-k count.
        config: VoidCaseConfig.
        stats: Optional VoidCaseStats.

    Returns:
        (selected_skills, lambda_x, s_max):
            selected_skills: top-k skills if inject else [].
            lambda_x, s_max: void-case decision values.
    """
    if not skills or skill_embs is None or len(skill_embs) == 0:
        return [], 0.0, 0.0

    q = query_emb.reshape(1, -1) if query_emb.ndim == 1 else query_emb
    # All embeddings already L2-normalized → dot product = cosine similarity
    sims = (skill_embs @ q.T).flatten()

    inject, lam, s_max = apply_void_case(sims, config, stats=stats)
    if not inject:
        return [], lam, s_max

    top_indices = np.argsort(sims)[::-1][:k]
    return [skills[i] for i in top_indices], lam, s_max
