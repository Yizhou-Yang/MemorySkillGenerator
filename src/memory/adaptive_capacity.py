"""
Adaptive Memory Capacity — theoretically-grounded capacity scheduling.

Designs a memory capacity function K(t) that:
1. Grows unboundedly (K → ∞ as t → ∞)
2. Grows sub-linearly (concave: K'' < 0)
3. Satisfies SRDP convergence (Theorem 10, Assumption 9(iv))
4. Has information-theoretic justification

============================================================
THEORETICAL FOUNDATION
============================================================

From SRDP (Memento-2, arXiv:2512.22716):

  Gap Bound (Corollary 15):
    |V* - V^{π_M}| ≤ (2R_max / (1-γ)²) · (ε_LLM(r_M) + δ_M)

  Where:
    - r_M = coverage radius (how far the nearest memory is from any query)
    - δ_M = retrieval noise (probability of retrieving wrong memory)

  Key insight: As |M| grows:
    - r_M decreases (better coverage) → ε_LLM decreases → gap tightens ✓
    - δ_M may INCREASE (more candidates → more confusion) → gap loosens ✗

  Therefore: optimal |M| balances coverage gain vs retrieval noise.

  Convergence condition (Theorem 10):
    ρ_t / η_t → 0  (memory update rate ≪ policy update rate)

  This means: memory growth rate must DECELERATE over time.

============================================================
PROPOSED CAPACITY FUNCTION
============================================================

We propose a logarithmic-power capacity schedule:

  K(t) = K_0 + α · t^β · ln(1 + t/τ)

Where:
  - K_0: initial capacity (cold-start budget)
  - α: growth amplitude
  - β ∈ (0, 1): sub-linear power (controls deceleration)
  - τ: time constant (controls when growth starts to slow)

Properties:
  1. K(0) = K_0 (starts at initial capacity)
  2. K(t) → ∞ as t → ∞ (unbounded growth)
  3. K'(t) → 0 as t → ∞ (growth rate decelerates)
  4. K''(t) < 0 for large t (concave = diminishing returns)

Special cases:
  - β = 0.5, τ = 100: K(t) = K_0 + α·√t·ln(1+t/100)
    → "square-root-log" growth (very conservative)
  - β = 0.5, τ = 1: K(t) ≈ K_0 + α·√t·ln(t)
    → "moderate" growth

============================================================
INFORMATION-THEORETIC JUSTIFICATION
============================================================

Theorem (Capacity-Entropy Balance):

  The optimal capacity K*(t) minimizes the total information cost:

    J(K) = H_retrieval(K) + H_coverage(K)

  Where:
    H_retrieval(K) = log(K)
      → Retrieval entropy: more memories = harder to find the right one
      → This is δ_M in SRDP terms

    H_coverage(K) = C / K^(1/d)
      → Coverage gap entropy: fewer memories = larger uncovered regions
      → This is ε_LLM(r_M) in SRDP terms (d = embedding dimension)

  Minimizing J(K) w.r.t. K:
    dJ/dK = 1/K - C/(d·K^(1/d + 1)) = 0
    → K* ∝ C^(d/(d+1))

  But C grows with experience (more tasks seen = more state space explored):
    C(t) ∝ t^(1/d)  (covering number growth in d dimensions)

  Therefore:
    K*(t) ∝ t^(1/(d+1))

  For d = 384 (our embedding dimension):
    K*(t) ∝ t^(1/385) ≈ t^0.0026

  This is EXTREMELY slow growth — practically logarithmic.
  Our proposed K(t) with β=0.5 is more aggressive but still sub-linear.

============================================================
SRDP CONVERGENCE VERIFICATION
============================================================

For Theorem 10 (Two-Time-Scale convergence):
  Need: ρ_t / η_t → 0

  Memory update rate: ρ_t ∝ K'(t) / K(t) (fractional growth rate)
  Policy update rate: η_t = constant (or slowly decaying)

  K'(t) = α·β·t^(β-1)·ln(1+t/τ) + α·t^β/(τ+t)

  For β < 1:
    K'(t) / K(t) → 0 as t → ∞  ✓

  Therefore: ρ_t / η_t → 0 is satisfied for any β ∈ (0, 1).

============================================================
COMPACTION INTEGRATION
============================================================

The capacity K(t) defines when compaction is triggered:

  if |M_t| > K(t):
      trigger_compaction(M_t, target_size=K(t))

  Compaction operations (from SRDP §6):
    - MERGE: combine semantically similar entries (reduces δ_M)
    - PRUNE: remove entries with redundant coverage (maintains r_M)

  After compaction: |M_t| ≤ K(t) ≤ K(t+1)
  → Memory is bounded at each t (satisfies Assumption 9(iv))
  → But bound grows over time (allows unbounded learning)

This is the key insight: K(t) is a MOVING ceiling, not a fixed one.
The ceiling rises slowly, and compaction keeps memory below it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


# ============================================================
# Core Capacity Function
# ============================================================

@dataclass
class CapacityConfig:
    """Configuration for adaptive memory capacity."""

    # Initial capacity (cold-start)
    K_0: int = 20

    # Growth amplitude
    alpha: float = 5.0

    # Sub-linear power exponent (0 < beta < 1)
    # Lower = slower growth. 0.5 = square-root growth.
    beta: float = 0.5

    # Time constant (controls when growth starts to slow)
    # Higher = growth stays fast longer before decelerating.
    tau: float = 100.0

    # Embedding dimension (for information-theoretic optimal)
    embedding_dim: int = 384

    # Compaction trigger ratio: compact when |M| > K(t) * trigger_ratio
    compaction_trigger_ratio: float = 1.0

    # Compaction target ratio: after compaction, |M| ≤ K(t) * target_ratio
    compaction_target_ratio: float = 0.8

    # Minimum capacity (never go below this)
    K_min: int = 10

    # Maximum capacity (hard ceiling for resource constraints)
    K_max: int = 10000


def capacity_at_step(t: int, config: CapacityConfig | None = None) -> int:
    """
    Compute memory capacity K(t) at training step t.

    Formula:
        K(t) = K_0 + α · t^β · ln(1 + t/τ)

    Properties:
        - K(0) = K_0 (initial capacity)
        - K(t) → ∞ as t → ∞ (unbounded growth)
        - K'(t) → 0 as t → ∞ (decelerating growth)
        - K''(t) < 0 for large t (concave / diminishing returns)

    Args:
        t: Current training step (number of tasks completed)
        config: Capacity configuration

    Returns:
        Integer capacity K(t)
    """
    if config is None:
        config = CapacityConfig()

    if t <= 0:
        return config.K_0

    # K(t) = K_0 + α · t^β · ln(1 + t/τ)
    growth = config.alpha * (t ** config.beta) * math.log(1 + t / config.tau)
    K = config.K_0 + growth

    # Clamp to [K_min, K_max]
    K = max(config.K_min, min(int(K), config.K_max))

    return K


def capacity_growth_rate(t: int, config: CapacityConfig | None = None) -> float:
    """
    Compute the instantaneous growth rate K'(t).

    Formula:
        K'(t) = α · [β · t^(β-1) · ln(1 + t/τ) + t^β / (τ + t)]

    This represents ρ_t in SRDP Theorem 10.
    For convergence, we need K'(t)/K(t) → 0.

    Args:
        t: Current training step
        config: Capacity configuration

    Returns:
        Growth rate (memories per step)
    """
    if config is None:
        config = CapacityConfig()

    if t <= 0:
        return config.alpha * config.beta  # Initial rate

    beta = config.beta
    tau = config.tau
    alpha = config.alpha

    # K'(t) = α · [β · t^(β-1) · ln(1 + t/τ) + t^β / (τ + t)]
    term1 = beta * (t ** (beta - 1)) * math.log(1 + t / tau)
    term2 = (t ** beta) / (tau + t)
    dK = alpha * (term1 + term2)

    return dK


def fractional_growth_rate(t: int, config: CapacityConfig | None = None) -> float:
    """
    Compute the fractional growth rate K'(t) / K(t).

    This is the effective ρ_t for SRDP convergence.
    Must → 0 as t → ∞ for Theorem 10 to hold.

    Args:
        t: Current training step
        config: Capacity configuration

    Returns:
        Fractional growth rate (dimensionless)
    """
    K = capacity_at_step(t, config)
    dK = capacity_growth_rate(t, config)
    if K <= 0:
        return float('inf')
    return dK / K


# ============================================================
# Information-Theoretic Optimal Capacity
# ============================================================

def optimal_capacity_info_theoretic(
    t: int,
    d: int = 384,
    C_0: float = 1.0,
) -> float:
    """
    Compute the information-theoretically optimal capacity K*(t).

    Minimizes total information cost:
        J(K) = H_retrieval(K) + H_coverage(K)
             = log(K) + C(t) / K^(1/d)

    Where C(t) ∝ t^(1/d) (covering number growth).

    Solution:
        K*(t) = (C(t) / d)^(d/(d+1))

    For d=384: K*(t) ∝ t^(1/385) — extremely slow growth.

    Args:
        t: Current training step
        d: Embedding dimension
        C_0: Initial covering constant

    Returns:
        Optimal capacity (float, not rounded)
    """
    if t <= 0:
        return C_0

    # C(t) = C_0 · t^(1/d) — covering number growth
    C_t = C_0 * (t ** (1.0 / d))

    # K*(t) = (C(t) / d)^(d/(d+1))
    K_star = (C_t / d) ** (d / (d + 1))

    return max(1.0, K_star)


def retrieval_entropy(K: int) -> float:
    """
    H_retrieval(K) = log₂(K)

    Retrieval entropy: more memories → harder to find the right one.
    Corresponds to δ_M in SRDP gap bound.
    """
    if K <= 1:
        return 0.0
    return math.log2(K)


def coverage_entropy(K: int, d: int = 384, C: float = 100.0) -> float:
    """
    H_coverage(K) = C / K^(1/d)

    Coverage gap entropy: fewer memories → larger uncovered regions.
    Corresponds to ε_LLM(r_M) in SRDP gap bound.
    """
    if K <= 0:
        return float('inf')
    return C / (K ** (1.0 / d))


def total_information_cost(K: int, d: int = 384, C: float = 100.0) -> float:
    """
    J(K) = H_retrieval(K) + H_coverage(K)
         = log₂(K) + C / K^(1/d)

    Total information cost that the capacity function minimizes.
    """
    return retrieval_entropy(K) + coverage_entropy(K, d, C)


# ============================================================
# SRDP Convergence Verification
# ============================================================

def verify_srdp_convergence(
    config: CapacityConfig | None = None,
    eta: float = 0.01,
    steps: list[int] | None = None,
) -> dict[str, Any]:
    """
    Verify that the capacity schedule satisfies SRDP Theorem 10.

    Condition: ρ_t / η_t → 0 as t → ∞

    Where:
        ρ_t = K'(t) / K(t) (fractional memory growth rate)
        η_t = policy learning rate (assumed constant or slowly decaying)

    Args:
        config: Capacity configuration
        eta: Policy learning rate (η_t)
        steps: Steps at which to check convergence

    Returns:
        Dict with convergence analysis
    """
    if config is None:
        config = CapacityConfig()
    if steps is None:
        steps = [1, 10, 50, 100, 500, 1000, 5000, 10000]

    results = {
        "config": {
            "K_0": config.K_0,
            "alpha": config.alpha,
            "beta": config.beta,
            "tau": config.tau,
        },
        "eta": eta,
        "steps": [],
        "converges": True,
    }

    prev_ratio = float('inf')
    for t in steps:
        K_t = capacity_at_step(t, config)
        dK_t = capacity_growth_rate(t, config)
        rho_t = fractional_growth_rate(t, config)
        ratio = rho_t / eta if eta > 0 else float('inf')

        results["steps"].append({
            "t": t,
            "K(t)": K_t,
            "K'(t)": round(dK_t, 6),
            "rho_t": round(rho_t, 6),
            "rho/eta": round(ratio, 6),
            "decreasing": ratio < prev_ratio,
        })

        if t > 100 and ratio > prev_ratio * 1.1:  # Allow 10% noise
            results["converges"] = False

        prev_ratio = ratio

    # Check final ratio is small
    final_ratio = results["steps"][-1]["rho/eta"]
    results["final_rho_over_eta"] = final_ratio
    results["converges"] = results["converges"] and final_ratio < 1.0

    return results


# ============================================================
# Compaction Scheduler
# ============================================================

@dataclass
class CompactionDecision:
    """Decision from the compaction scheduler."""
    should_compact: bool
    current_size: int
    capacity: int
    target_size: int
    reason: str


def should_compact(
    current_memory_size: int,
    current_step: int,
    config: CapacityConfig | None = None,
) -> CompactionDecision:
    """
    Determine if compaction should be triggered.

    Compaction is triggered when:
        |M_t| > K(t) · trigger_ratio

    After compaction, target size is:
        target = K(t) · target_ratio

    Args:
        current_memory_size: Current number of memory entries
        current_step: Current training step
        config: Capacity configuration

    Returns:
        CompactionDecision with recommendation
    """
    if config is None:
        config = CapacityConfig()

    K_t = capacity_at_step(current_step, config)
    trigger_threshold = int(K_t * config.compaction_trigger_ratio)
    target_size = int(K_t * config.compaction_target_ratio)

    if current_memory_size > trigger_threshold:
        return CompactionDecision(
            should_compact=True,
            current_size=current_memory_size,
            capacity=K_t,
            target_size=target_size,
            reason=(
                f"|M|={current_memory_size} > K(t)={K_t} "
                f"(trigger at {trigger_threshold}). "
                f"Compact to target={target_size}."
            ),
        )
    else:
        return CompactionDecision(
            should_compact=False,
            current_size=current_memory_size,
            capacity=K_t,
            target_size=target_size,
            reason=(
                f"|M|={current_memory_size} ≤ K(t)={K_t}. "
                f"No compaction needed."
            ),
        )


# ============================================================
# Gap Bound Estimation
# ============================================================

def _compute_gap_components(
    K: int,
    d: int = 384,
    R_max: float = 1.0,
    gamma: float = 0.99,
    epsilon_base: float = 0.1,
    delta_base: float = 0.05,
) -> tuple[float, float, float]:
    """Compute gap bound components without optimal_K (avoids recursion)."""
    if K <= 1:
        return (float('inf'), 1.0, 0.0)

    epsilon = epsilon_base * (K ** (-1.0 / d))
    K_0 = 20
    delta = delta_base * math.log(K) / math.log(K_0) if K > 1 else delta_base
    prefactor = 2 * R_max / ((1 - gamma) ** 2)
    gap = prefactor * (epsilon + delta)
    return (gap, epsilon, delta)


def estimate_gap_bound(
    K: int,
    d: int = 384,
    R_max: float = 1.0,
    gamma: float = 0.99,
    epsilon_base: float = 0.1,
    delta_base: float = 0.05,
) -> dict[str, float]:
    """
    Estimate the SRDP gap bound given current memory size.

    Gap <= (2R_max / (1-gamma)^2) * (epsilon_LLM(r_M) + delta_M)

    Where:
        epsilon_LLM(r_M) ~ epsilon_base * K^(-1/d)  (coverage improves with more memories)
        delta_M ~ delta_base * log(K) / log(K_0)  (retrieval noise grows with log(K))

    Args:
        K: Current memory size
        d: Embedding dimension
        R_max: Maximum reward
        gamma: Discount factor
        epsilon_base: Base LLM generalization error
        delta_base: Base retrieval noise

    Returns:
        Dict with gap bound components
    """
    gap, epsilon, delta = _compute_gap_components(K, d, R_max, gamma, epsilon_base, delta_base)
    prefactor = 2 * R_max / ((1 - gamma) ** 2)

    return {
        "gap_bound": gap,
        "epsilon_LLM": epsilon,
        "delta_M": delta,
        "prefactor": prefactor,
        "K": K,
        "optimal_K": _find_optimal_K(d, R_max, gamma, epsilon_base, delta_base),
    }


def _find_optimal_K(
    d: int, R_max: float, gamma: float,
    epsilon_base: float, delta_base: float,
) -> int:
    """Find K that minimizes the gap bound (linear search)."""
    best_K = 20
    best_gap = float('inf')

    for K in range(10, 5000, 10):
        gap, _, _ = _compute_gap_components(K, d, R_max, gamma, epsilon_base, delta_base)
        if gap < best_gap:
            best_gap = gap
            best_K = K

    return best_K


# ============================================================
# Visualization / Summary
# ============================================================

def capacity_schedule_summary(
    config: CapacityConfig | None = None,
    max_steps: int = 10000,
    sample_points: int = 20,
) -> str:
    """
    Generate a human-readable summary of the capacity schedule.

    Returns a formatted string showing K(t) at various steps.
    """
    if config is None:
        config = CapacityConfig()

    steps = [int(max_steps * i / sample_points) for i in range(sample_points + 1)]
    steps[0] = 1  # Avoid t=0

    header = "  Step t |   K(t) |   K'(t) |   rho/eta |   Gap dM"
    lines = [
        "Memory Capacity Schedule: K(t) = K_0 + alpha*t^beta*ln(1 + t/tau)",
        f"  K_0={config.K_0}, alpha={config.alpha}, beta={config.beta}, tau={config.tau}",
        "",
        header,
        "-" * 55,
    ]

    eta = 0.01  # Assumed policy learning rate
    for t in steps:
        K = capacity_at_step(t, config)
        dK = capacity_growth_rate(t, config)
        rho_eta = fractional_growth_rate(t, config) / eta
        gap = estimate_gap_bound(K)
        lines.append(
            f"{t:>8} | {K:>6} | {dK:>8.4f} | {rho_eta:>8.4f} | {gap['delta_M']:>8.4f}"
        )

    lines.extend([
        "",
        "Properties:",
        f"  K(1) = {capacity_at_step(1, config)}",
        f"  K(100) = {capacity_at_step(100, config)}",
        f"  K(1000) = {capacity_at_step(1000, config)}",
        f"  K(10000) = {capacity_at_step(10000, config)}",
        f"  Growth rate at t=10000: {capacity_growth_rate(10000, config):.6f} memories/step",
        f"  SRDP convergence: ρ/η at t=10000 = {fractional_growth_rate(10000, config)/eta:.6f}",
    ])

    return "\n".join(lines)
