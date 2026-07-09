#!/usr/bin/env python3
"""Pooled PRIMARY endpoint (pre-registered in the paper's experimental setup):
one stratified paired test per arm pair, pooling final-iteration per-task
deltas across benchmarks and backbones. Per-benchmark results are SECONDARY.

Why: single-benchmark C-B effects (~2-4pp) sit below any single benchmark's
minimum detectable effect at n=100 (8-13pp). The paired deltas are the same
estimand everywhere (score in [0,1], same tasks across arms), so pooling
~600 pairs across strata brings the MDE down to ~4pp. This file is the ONE
implementation of that test, declared before the final data lands.

Test: sign-flip permutation on the pooled mean of per-task paired deltas
(exchangeability under H0 within each pair), stratified bootstrap for the CI.
CUPED=1 (default) additionally reports a variance-reduced estimate using the
iteration-0 paired delta as covariate: at iteration 0 no memory is injected in
a fresh chain, so d0 is pure judge/agent noise correlated with the final
delta's noise term; subtracting theta*d0 is unbiased for the treatment effect.

Usage:
  python scripts/latest/pooled_stats.py <model> [<model2> ...]
  POOL_BENCHMARKS=gaia,gaia2,locomo BASE=latest_evolving  # defaults
  TB2 strata are added automatically when a harbor trace exists.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

try:
    from scripts.latest.arms import norm_group as _norm_group
except ImportError:
    _LEGACY_MAP = {"A_baseline": "no_mem", "B_evomem": "raw_patch", "C_gpr": "curated_patch"}
    def _norm_group(g): return _LEGACY_MAP.get(g, g)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE = os.environ.get("BASE", "latest_evolving")
BENCHES = [b.strip() for b in
           os.environ.get("POOL_BENCHMARKS", "gaia,gaia2,locomo").split(",") if b.strip()]
PAIRS = [("curated_patch", "raw_patch"), ("curated_patch", "no_mem"), ("raw_patch", "no_mem")]
N_PERM = int(os.environ.get("N_PERM", "10000"))
N_BOOT = int(os.environ.get("N_BOOT", "4000"))
USE_CUPED = os.environ.get("CUPED", "1") == "1"
# Metric mapping. Primary endpoint = the continuous per-task `score` for every
# benchmark (most powerful, disclosed as such). NATIVE_METRIC=1 reruns the same
# test under each benchmark's *native* metric: strict exact-match for GAIA and
# LoCoMo (the `em` field), and the continuous score for GAIA2 (soft recall) and
# TB-2 (verified pass fraction). Reported as an honest robustness view.
NATIVE_METRIC = os.environ.get("NATIVE_METRIC", "0") == "1"
_EM_BENCH = {"gaia", "locomo"}
random.seed(20260705)


def _task_score(r: dict) -> float:
    if NATIVE_METRIC and (r.get("benchmark", "") or "").lower() in _EM_BENCH \
            and r.get("em") is not None:
        return float(r.get("em") or 0.0)
    return float(r.get("score") or 0.0)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line); r["group"] = _norm_group(r.get("group","")); out.append(r)
    return out


def _by_task_iter(rows: list[dict], group: str) -> dict:
    """(task_id, iteration) -> score for one arm."""
    out = {}
    for r in rows:
        if r.get("group") == group:
            out[(r.get("task_id"), int(r.get("iteration", 0) or 0))] = _task_score(r)
    return out


def _stratum_deltas(rows: list[dict], g1: str, g2: str) -> tuple[list[float], list[float]]:
    """Final-iteration paired deltas (and iteration-0 deltas for CUPED) over
    tasks whose chains FINISHED (row at iter_total-1) in both arms."""
    if not rows:
        return [], []
    k_final = max(int(r.get("iter_total", 1) or 1) for r in rows) - 1
    m1, m2 = _by_task_iter(rows, g1), _by_task_iter(rows, g2)
    d_fin, d_zero = [], []
    tasks = sorted({t for (t, i) in m1 if i == k_final} &
                   {t for (t, i) in m2 if i == k_final})
    for t in tasks:
        d_fin.append(m1[(t, k_final)] - m2[(t, k_final)])
        d_zero.append(m1.get((t, 0), 0.0) - m2.get((t, 0), 0.0))
    return d_fin, d_zero


def _cuped(d_fin: list[float], d_zero: list[float]) -> list[float]:
    n = len(d_fin)
    if n < 10:
        return d_fin
    mz = sum(d_zero) / n
    var = sum((z - mz) ** 2 for z in d_zero) / n
    if var <= 1e-12:
        return d_fin
    mf = sum(d_fin) / n
    cov = sum((f - mf) * (z - mz) for f, z in zip(d_fin, d_zero)) / n
    theta = cov / var
    # subtracting theta*(d0 - mean(d0)) keeps the estimate unbiased
    return [f - theta * (z - mz) for f, z in zip(d_fin, d_zero)]


def _pooled_test(strata: list[list[float]]) -> tuple[float, float, float, float, int]:
    """Pooled mean (task-weighted), stratified bootstrap CI, sign-flip p."""
    alld = [d for s in strata for d in s]
    n = len(alld)
    if n == 0:
        return float("nan"), 0.0, 0.0, 1.0, 0
    mean = sum(alld) / n
    # stratified bootstrap
    ms = []
    for _ in range(N_BOOT):
        tot, cnt = 0.0, 0
        for s in strata:
            k = len(s)
            tot += sum(s[random.randrange(k)] for _ in range(k))
            cnt += k
        ms.append(tot / cnt)
    ms.sort()
    lo, hi = ms[int(0.025 * N_BOOT)], ms[int(0.975 * N_BOOT)]
    # sign-flip permutation (two-sided)
    hits = 0
    for _ in range(N_PERM):
        tot = sum(d if random.random() < 0.5 else -d for d in alld)
        if abs(tot / n) >= abs(mean) - 1e-12:
            hits += 1
    p = (hits + 1) / (N_PERM + 1)
    return mean, lo, hi, p, n


def main() -> None:
    models = [m.lower() for m in sys.argv[1:]] or ["hy3"]
    strata_rows: list[tuple[str, list[dict]]] = []
    for m in models:
        for b in BENCHES:
            strata_rows.append((f"{m}/{b}",
                                _rows(PROJECT_ROOT / "experiments_results" / BASE / m / b / "trace.jsonl")))
        tb2 = PROJECT_ROOT / "experiments_results/harbor_tb2" / m / "terminal_bench_2/trace.jsonl"
        if tb2.exists():
            strata_rows.append((f"{m}/tb2", _rows(tb2)))

    print(f"# Pooled primary endpoint — models={models}, strata="
          f"{[s for s, r in strata_rows if r]}, CUPED={USE_CUPED}\n")
    print("| pair | pooled Δ | 95% CI | p (sign-flip) | n | per-stratum n |")
    print("|---|---|---|---|---|---|")
    dead_warned = set()
    for g1, g2 in PAIRS:
        strata, names = [], []
        for name, rows in strata_rows:
            d_fin, d_zero = _stratum_deltas(rows, g1, g2)
            if not d_fin:
                continue
            # Dead-stratum guard: a benchmark whose scores are ALL zero in both
            # arms (e.g. an infra-failed TB2 round) contributes only 0-deltas,
            # dragging the pooled estimate toward 0 without carrying signal.
            scores = [_task_score(r) for r in rows
                      if r.get("group") in (g1, g2)]
            if scores and not any(scores):
                if name not in dead_warned:
                    print(f"[!] stratum {name}: all scores 0 in both arms — "
                          "excluded as infra-dead, investigate before final")
                    dead_warned.add(name)
                continue
            strata.append(_cuped(d_fin, d_zero) if USE_CUPED else d_fin)
            names.append(f"{name}:{len(d_fin)}")
        mean, lo, hi, p, n = _pooled_test(strata)
        tag = " **SIG**" if p < 0.05 else ""
        print(f"| {g1}−{g2} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] "
              f"| {p:.4f}{tag} | {n} | {', '.join(names)} |")
    print("\nSecondary: per-benchmark deltas via scripts/latest/soft_stats.py; "
          "gate first with scripts/latest/check_run.sh <model>.")


if __name__ == "__main__":
    main()
