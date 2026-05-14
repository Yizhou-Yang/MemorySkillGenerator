"""
δ_M Semantic-Attention Causal Decomposition — core theoretical contribution.

Paper §4.1: The SRDP gap bound's δ_M can be decomposed into two independent
components that are driven by different factors and require different operators:

  δ_M(x) = δ_semantic(x) + E_{c~μ}[δ_attention(x, c)]

Where:
  - δ_semantic: probability of retrieving the WRONG skill (selection error)
  - δ_attention: even if the RIGHT skill is retrieved, the LLM fails to
    attend to its critical parts (attention allocation error)

Key insight: All existing works (SkillOS, COSPLAY, Memento-Skills) only
optimize δ_semantic. Nobody optimizes δ_attention. This module provides
the formal framework for both.

Definitions (from paper §4.1.2):

  Definition (Semantic Error):
    δ_semantic(x) = Σ_{c∈M} μ(c|x) · 1[c ≠ c*(x)]
    where c*(x) = argmax_c Q(x,c) is the optimal skill for state x.

  Definition (Attention Error):
    δ_attention(x, c) = D_TV(p_LLM(·|s,c) || p_LLM^ideal(·|s,c))
    where p_LLM^ideal is the action distribution if LLM perfectly
    understood all content of skill c.

  Sources of δ_attention (Proposition 0+):
    - Position effect: middle skills get less attention (Lost in the Middle)
    - Length effect: longer skills → attention dilution
    - Format effect: prose < table < structured list
    - Polarity effect: negations activate forbidden concepts (pink elephant)
    - Consistency effect: conflicting skills cause LLM "hesitation"

Propositions:
  Prop 1 (Retrieval Dilution): K redundant pairs → Δδ_semantic ≤ Σ p_k/2
  Prop 2 (Phase Transition): ∃ N* s.t. gap is minimized
  Prop 3 (Bound Tightening): MERGE K pairs → δ_M(S') ≤ δ_M(S) - Σ p_k/2
  Prop 4 (Entropy Health): δ_M ≥ 1 - N_eff/|S|

Reference: SkillCurator paper §4.1-4.4
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from src.models import Skill
from src.utils.skill_formatter import (
    attention_weight,
    effective_attention,
    retrieval_noise,
    coverage_gap,
    L_OPTIMAL_TOKENS,
)


# ============================================================
# Core Decomposition
# ============================================================


@dataclass
class DeltaDecomposition:
    """Result of decomposing δ_M into semantic and attention components."""

    delta_semantic: float  # Probability of selecting wrong skill
    delta_attention: float  # Expected attention failure given correct skill
    delta_total: float  # δ_semantic + δ_attention
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def semantic_fraction(self) -> float:
        """Fraction of total error from semantic component."""
        if self.delta_total <= 0:
            return 0.0
        return self.delta_semantic / self.delta_total

    @property
    def attention_fraction(self) -> float:
        """Fraction of total error from attention component."""
        if self.delta_total <= 0:
            return 0.0
        return self.delta_attention / self.delta_total


def decompose_delta(
    skills: list[Skill],
    retrieval_distribution: list[float] | None = None,
    library_tokens: int | None = None,
) -> DeltaDecomposition:
    """
    Decompose δ_M into δ_semantic + δ_attention for a skill library.

    Args:
        skills: Current skill library
        retrieval_distribution: μ(c|x) for each skill (uniform if None)
        library_tokens: Total tokens in the formatted library

    Returns:
        DeltaDecomposition with both components estimated
    """
    n = len(skills)
    if n == 0:
        return DeltaDecomposition(delta_semantic=0.0, delta_attention=0.0, delta_total=0.0)

    # --- δ_semantic estimation ---
    # Based on redundancy: more redundant pairs → higher dilution
    delta_sem = estimate_delta_semantic(skills, retrieval_distribution)

    # --- δ_attention estimation ---
    # Based on library length, format, and position effects
    if library_tokens is None:
        # Estimate from skill content
        library_tokens = sum(
            len(f"{s.name} {s.description} {' '.join(s.procedure)}".split())
            for s in skills
        ) * 2  # Rough token estimate

    delta_att = estimate_delta_attention(skills, library_tokens)

    total = delta_sem + delta_att

    return DeltaDecomposition(
        delta_semantic=delta_sem,
        delta_attention=delta_att,
        delta_total=total,
        details={
            "num_skills": n,
            "library_tokens": library_tokens,
            "redundant_pairs": _count_redundant_pairs(skills),
            "avg_skill_length": library_tokens / n if n > 0 else 0,
        },
    )


# ============================================================
# δ_semantic Estimation (§4.1.3)
# ============================================================


def estimate_delta_semantic(
    skills: list[Skill],
    retrieval_distribution: list[float] | None = None,
) -> float:
    """
    Estimate δ_semantic — the probability of selecting the wrong skill.

    Proposition 1 (Retrieval Dilution):
      When K redundant pairs exist with similar Q-values,
      the Boltzmann retrieval probability is diluted:
        Δδ_semantic ≤ Σ_{k=1}^K p_k / 2

    In practice, we estimate this from:
    1. Number of redundant pairs (Jaccard similarity > threshold)
    2. Uniformity of retrieval distribution (more uniform = more noise)
    """
    n = len(skills)
    if n <= 1:
        return 0.0

    # Factor 1: Redundancy-based dilution
    redundant_pairs = _count_redundant_pairs(skills)
    # Each redundant pair contributes ~1/(2N) to δ_semantic
    dilution = redundant_pairs / (2 * n) if n > 0 else 0.0

    # Factor 2: Distribution uniformity
    if retrieval_distribution is not None:
        # More uniform = higher noise (ideal is concentrated on best skill)
        entropy = -sum(p * math.log(p + 1e-10) for p in retrieval_distribution if p > 0)
        max_entropy = math.log(n)
        uniformity = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        # Assume uniform (worst case)
        uniformity = 1.0

    # Combined estimate
    delta_sem = min(1.0, dilution + 0.3 * uniformity)

    return delta_sem


def _count_redundant_pairs(skills: list[Skill], threshold: float = 0.4) -> int:
    """Count pairs of skills with Jaccard similarity > threshold."""
    n = len(skills)
    count = 0
    for i in range(n):
        text_i = f"{skills[i].name} {skills[i].description} {' '.join(skills[i].procedure)}".lower()
        tokens_i = set(text_i.split())
        for j in range(i + 1, n):
            text_j = f"{skills[j].name} {skills[j].description} {' '.join(skills[j].procedure)}".lower()
            tokens_j = set(text_j.split())
            if not tokens_i or not tokens_j:
                continue
            intersection = tokens_i & tokens_j
            union = tokens_i | tokens_j
            sim = len(intersection) / len(union) if union else 0.0
            if sim > threshold:
                count += 1
    return count


# ============================================================
# δ_attention Estimation (§4.1.4)
# ============================================================


def estimate_delta_attention(
    skills: list[Skill],
    library_tokens: int,
) -> float:
    """
    Estimate δ_attention — the expected attention failure.

    Sources (Proposition 0+):
    1. Position effect: U-curve attention distribution
    2. Length effect: longer library → more dilution
    3. Format effect: (handled by formatter, not estimated here)
    4. Polarity effect: negative constraints
    5. Consistency effect: conflicting rules

    δ_attention ≈ w_pos · f_position + w_len · f_length + w_neg · f_negation + w_con · f_conflict
    """
    n = len(skills)
    if n == 0:
        return 0.0

    # Factor 1: Position effect (Lost in the Middle)
    # Average attention weight across all positions
    avg_alpha = sum(attention_weight(i, n) for i in range(n)) / n
    # Lower average attention = higher δ_attention
    f_position = 1.0 - avg_alpha  # 0 if all get full attention, ~0.7 if middle is lost

    # Factor 2: Length effect (attention dilution)
    # Deviation from optimal length
    length_ratio = library_tokens / L_OPTIMAL_TOKENS
    f_length = max(0.0, (length_ratio - 1.0) * 0.3)  # Penalty for exceeding optimal

    # Factor 3: Negation effect (pink elephant)
    total_constraints = sum(len(s.constraints) for s in skills)
    negative_count = sum(
        1 for s in skills for c in s.constraints
        if any(neg in c.lower() for neg in ["do not", "never", "avoid", "don't", "must not"])
    )
    f_negation = negative_count / max(total_constraints, 1) * 0.2

    # Factor 4: Consistency effect (conflicting rules)
    # Simple heuristic: skills with overlapping but different constraints
    f_conflict = _estimate_conflict_score(skills) * 0.3

    # Weighted combination
    delta_att = 0.4 * f_position + 0.3 * f_length + 0.15 * f_negation + 0.15 * f_conflict
    return min(1.0, delta_att)


def _estimate_conflict_score(skills: list[Skill]) -> float:
    """Estimate the degree of conflicting rules between skills."""
    if len(skills) <= 1:
        return 0.0

    # Simple heuristic: count pairs where constraints mention similar topics
    # but with different instructions
    conflict_count = 0
    total_pairs = 0

    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            if not skills[i].constraints or not skills[j].constraints:
                continue
            total_pairs += 1
            # Check if any constraint pair has high token overlap but different verbs
            for c_i in skills[i].constraints:
                for c_j in skills[j].constraints:
                    tokens_i = set(c_i.lower().split())
                    tokens_j = set(c_j.lower().split())
                    overlap = len(tokens_i & tokens_j) / max(len(tokens_i | tokens_j), 1)
                    if overlap > 0.3 and c_i != c_j:
                        conflict_count += 1

    return conflict_count / max(total_pairs, 1)


# ============================================================
# Library Health Summary (§4.3.1)
# ============================================================


@dataclass
class LibraryHealthSummary:
    """
    Lightweight online statistics for the skill library.

    Computed before each curation decision to give the curator
    global context about library state.

    This is the key differentiator from SkillOS:
    SkillOS curator only sees ξ_t + retrieved_skills (local).
    We additionally provide health_summary (global).
    """

    total_skills: int
    effective_count: float  # exp(H[retrieval_dist])
    effective_ratio: float  # effective_count / total_skills
    redundant_pairs: int
    redundancy_ratio: float  # redundant_pairs / C(N, 2)
    avg_skill_length: float  # Average tokens per skill
    conflict_count: int
    delta_semantic: float
    delta_attention: float
    delta_total: float
    capacity_utilization: float  # |S| / K(t)
    top_redundant_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    least_used_skills: list[str] = field(default_factory=list)


def compute_library_health(
    skills: list[Skill],
    retrieval_counts: dict[str, int] | None = None,
    current_step: int = 0,
) -> LibraryHealthSummary:
    """
    Compute the library health summary.

    This is injected into the curator's context before each curation decision,
    giving it global awareness of library state.

    Args:
        skills: Current skill library
        retrieval_counts: How many times each skill was retrieved (for N_eff)
        current_step: Current training step (for capacity calculation)

    Returns:
        LibraryHealthSummary with all metrics
    """
    n = len(skills)
    if n == 0:
        return LibraryHealthSummary(
            total_skills=0, effective_count=0, effective_ratio=0,
            redundant_pairs=0, redundancy_ratio=0, avg_skill_length=0,
            conflict_count=0, delta_semantic=0, delta_attention=0,
            delta_total=0, capacity_utilization=0,
        )

    # Effective count: exp(H[retrieval_distribution])
    # Proposition 4: N_eff = exp(H[p_hat])
    if retrieval_counts:
        total_retrievals = sum(retrieval_counts.values())
        if total_retrievals > 0:
            probs = [retrieval_counts.get(s.skill_id, 0) / total_retrievals for s in skills]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
            effective_count = math.exp(entropy)
        else:
            effective_count = float(n)  # Assume uniform if no data
    else:
        effective_count = float(n)  # Assume uniform

    effective_ratio = effective_count / n if n > 0 else 0.0

    # Redundancy
    redundant_pairs = _count_redundant_pairs(skills)
    max_pairs = n * (n - 1) / 2
    redundancy_ratio = redundant_pairs / max_pairs if max_pairs > 0 else 0.0

    # Average skill length
    total_tokens = sum(
        len(f"{s.name} {s.description} {' '.join(s.procedure)}".split())
        for s in skills
    )
    avg_skill_length = total_tokens / n if n > 0 else 0.0

    # Conflict count
    conflict_score = _estimate_conflict_score(skills)
    conflict_count = int(conflict_score * max_pairs) if max_pairs > 0 else 0

    # Delta decomposition
    decomp = decompose_delta(skills, library_tokens=total_tokens * 2)

    # Capacity utilization
    from src.memory.adaptive_capacity import capacity_at_step
    capacity = capacity_at_step(current_step)
    capacity_utilization = n / capacity if capacity > 0 else 0.0

    # Top redundant pairs (for curator context)
    top_pairs = _get_top_redundant_pairs(skills, top_k=3)

    # Least used skills
    least_used = []
    if retrieval_counts:
        sorted_by_use = sorted(
            [(s.skill_id, retrieval_counts.get(s.skill_id, 0)) for s in skills],
            key=lambda x: x[1],
        )
        least_used = [sid for sid, _ in sorted_by_use[:3]]

    return LibraryHealthSummary(
        total_skills=n,
        effective_count=effective_count,
        effective_ratio=effective_ratio,
        redundant_pairs=redundant_pairs,
        redundancy_ratio=redundancy_ratio,
        avg_skill_length=avg_skill_length,
        conflict_count=conflict_count,
        delta_semantic=decomp.delta_semantic,
        delta_attention=decomp.delta_attention,
        delta_total=decomp.delta_total,
        capacity_utilization=capacity_utilization,
        top_redundant_pairs=top_pairs,
        least_used_skills=least_used,
    )


def _get_top_redundant_pairs(skills: list[Skill], top_k: int = 3) -> list[tuple[str, str, float]]:
    """Get the top-K most redundant skill pairs."""
    pairs = []
    for i in range(len(skills)):
        text_i = f"{skills[i].name} {skills[i].description}".lower()
        tokens_i = set(text_i.split())
        for j in range(i + 1, len(skills)):
            text_j = f"{skills[j].name} {skills[j].description}".lower()
            tokens_j = set(text_j.split())
            if not tokens_i or not tokens_j:
                continue
            intersection = tokens_i & tokens_j
            union = tokens_i | tokens_j
            sim = len(intersection) / len(union) if union else 0.0
            if sim > 0.2:
                pairs.append((skills[i].name, skills[j].name, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


# ============================================================
# Phase Transition (Proposition 2, §4.4.1)
# ============================================================


def compute_optimal_library_size(
    a: float = 1.0,
    b: float = 0.5,
    d: int = 384,
) -> int:
    """
    Compute the optimal library size N* (Proposition 2).

    In the gap bound:
      ε_LLM(r_M) ∝ |M|^{-a/d}  (coverage improves)
      δ_M ∝ |M|^b              (retrieval degrades)

    The optimal size minimizes their sum:
      N* = (a / (b·d))^{d/(a+bd)}

    Args:
        a: Coverage improvement exponent
        b: Retrieval degradation exponent
        d: Embedding dimension

    Returns:
        Optimal library size N*
    """
    if b <= 0 or d <= 0:
        return 100  # Default

    exponent = d / (a + b * d)
    base = a / (b * d)

    if base <= 0:
        return 100

    n_star = base ** exponent
    return max(5, int(n_star))


def phase_transition_curve(
    max_n: int = 200,
    a: float = 1.0,
    b: float = 0.5,
    d: int = 384,
) -> list[tuple[int, float]]:
    """
    Compute the inverted-U performance curve as a function of library size.

    Returns (N, gap_value) pairs showing the phase transition.
    """
    curve = []
    for n in range(1, max_n + 1, 5):
        # Coverage term: decreases with N
        epsilon = n ** (-a / d)
        # Retrieval term: increases with N
        delta = n ** b / 100  # Normalized
        gap = epsilon + delta
        curve.append((n, gap))
    return curve


# ============================================================
# Proposition 3: Bound Tightening from MERGE
# ============================================================


def bound_tightening_from_merge(
    skills_before: list[Skill],
    skills_after: list[Skill],
) -> dict[str, float]:
    """
    Compute the bound tightening from a MERGE operation (Proposition 3).

    MERGE K redundant pairs → δ_M(S') ≤ δ_M(S) - Σ p_k/2

    Also verifies Safe-MERGE condition: r_M(S') ≤ r_M(S)
    """
    decomp_before = decompose_delta(skills_before)
    decomp_after = decompose_delta(skills_after)

    return {
        "delta_before": decomp_before.delta_total,
        "delta_after": decomp_after.delta_total,
        "delta_improvement": decomp_before.delta_total - decomp_after.delta_total,
        "semantic_improvement": decomp_before.delta_semantic - decomp_after.delta_semantic,
        "attention_improvement": decomp_before.delta_attention - decomp_after.delta_attention,
        "skills_before": len(skills_before),
        "skills_after": len(skills_after),
        "bound_tightened": decomp_after.delta_total < decomp_before.delta_total,
    }


# ============================================================
# Proposition 4: Entropy-based Health Metric
# ============================================================


def compute_effective_skill_count(retrieval_distribution: list[float]) -> float:
    """
    Compute N_eff = exp(H[p_hat]) (Proposition 4).

    N_eff measures how many skills are "effectively" being used.
    If all skills are equally retrieved: N_eff = |S|
    If only one skill is ever retrieved: N_eff = 1

    Proposition 4: δ_M ≥ 1 - N_eff/|S|
    The "waste ratio" (1 - N_eff/|S|) is a lower bound on δ_M.
    """
    if not retrieval_distribution:
        return 0.0

    # Filter out zeros
    probs = [p for p in retrieval_distribution if p > 0]
    if not probs:
        return 0.0

    # Normalize
    total = sum(probs)
    probs = [p / total for p in probs]

    # Entropy
    entropy = -sum(p * math.log(p) for p in probs)

    # N_eff = exp(H)
    return math.exp(entropy)


def library_waste_ratio(n_eff: float, total_skills: int) -> float:
    """
    Compute the library waste ratio (Proposition 4).

    waste_ratio = 1 - N_eff / |S|

    This is a lower bound on δ_M.
    Higher waste = more retrieval noise = worse performance.
    """
    if total_skills <= 0:
        return 0.0
    return 1.0 - (n_eff / total_skills)
