#!/usr/bin/env python3
"""Analyze SkillForge A/B/C trace results with paired statistics.

Reads experiments_results/latest/<bench>/trace.jsonl and reports, per benchmark:
  - per-group mean EM (+/- std) and n
  - whether the A <= B <= C ordering holds
  - PAIRED significance on the tasks all groups share:
      * delta (mean EM difference) for B-A and C-B
      * McNemar exact p-value (binomial) on discordant pairs
      * bootstrap 95% CI on the paired delta

Single-run point estimates on ~30 tasks are noisy; this surfaces whether an
apparent A<B<C ordering is real or within noise. No third-party deps.

Usage:
    python scripts/latest/analyze_results.py [results_dir]
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "experiments_results" / "latest"

GROUP_ORDER = ["A_baseline", "B_evomem", "C_gpr"]
GROUP_SHORT = {"A_baseline": "A", "B_evomem": "B", "C_gpr": "C"}
# Legacy trace keys (pre-retrofit) → canonical keys, so old runs still analyze.
GROUP_ALIASES = {"B_evoarena": "B_evomem", "C_skillforge": "C_gpr"}
BENCHMARKS = ["gaia", "gaia2", "locomo", "terminal_bench_2"]

_BOOT_ITERS = 5000
_rng = random.Random(12345)  # fixed seed -> reproducible CIs


def _metric(rec: dict) -> float:
    """Binary correctness per task. Prefer logged EM; fall back to score>=0.5."""
    if "em" in rec and rec["em"] is not None:
        return 1.0 if float(rec["em"]) >= 0.5 else 0.0
    return 1.0 if float(rec.get("score", 0.0)) >= 0.5 else 0.0


def _load(bench_dir: Path) -> dict[str, dict[str, float]]:
    """Return {group: {task_id: em}} for one benchmark (last record per pair wins)."""
    trace = bench_dir / "trace.jsonl"
    by_group: dict[str, dict[str, float]] = {}
    if not trace.exists():
        return by_group
    for line in trace.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        g = rec.get("group", "?")
        g = GROUP_ALIASES.get(g, g)        # normalize legacy keys to canonical
        tid = rec.get("task_id", "")
        by_group.setdefault(g, {})[tid] = _metric(rec)
    return by_group


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


def _mcnemar_p(pairs: list[tuple[float, float]]) -> tuple[int, int, float]:
    """Exact McNemar on binary paired outcomes (hi = second group).
    Returns (b, c, p) where b = lo-correct/hi-wrong, c = lo-wrong/hi-correct."""
    b = sum(1 for lo, hi in pairs if lo == 1 and hi == 0)
    c = sum(1 for lo, hi in pairs if lo == 0 and hi == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    # two-sided exact binomial p at p0=0.5
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * cdf)


def _bootstrap_ci(deltas: list[float]) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    n = len(deltas)
    means = []
    for _ in range(_BOOT_ITERS):
        s = sum(deltas[_rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * _BOOT_ITERS)]
    hi = means[int(0.975 * _BOOT_ITERS)]
    return lo, hi


def _paired(lo_map: dict[str, float], hi_map: dict[str, float]):
    ids = sorted(set(lo_map) & set(hi_map))
    pairs = [(lo_map[i], hi_map[i]) for i in ids]
    deltas = [hi - lo for lo, hi in pairs]
    return pairs, deltas, len(ids)


def analyze_benchmark(name: str, bench_dir: Path) -> None:
    by_group = _load(bench_dir)
    if not by_group:
        print(f"\n### {name}: no trace data")
        return

    print(f"\n### {name}")
    present = [g for g in GROUP_ORDER if g in by_group]
    means = {}
    for g in present:
        vals = list(by_group[g].values())
        m, sd = _mean_std(vals)
        means[g] = m
        print(f"    {GROUP_SHORT[g]} ({g:<13}) EM={m:6.1%} ± {sd:4.1%}  n={len(vals)}")

    missing = [GROUP_SHORT[g] for g in GROUP_ORDER if g not in by_group]
    if missing:
        print(f"    !! incomplete: missing groups {missing} "
              f"(run with RESUME=1 to finish them)")

    # Ordering check
    if all(g in means for g in GROUP_ORDER):
        a, b, c = (means[g] for g in GROUP_ORDER)
        ok = a <= b <= c
        print(f"    ordering A<=B<=C: {'HOLDS' if ok else 'VIOLATED'} "
              f"(A={a:.1%} B={b:.1%} C={c:.1%})")

    # Paired significance on shared tasks
    for lo_g, hi_g, label in (("A_baseline", "B_evomem", "B vs A"),
                              ("B_evomem", "C_gpr", "C vs B"),
                              ("A_baseline", "C_gpr", "C vs A")):
        if lo_g not in by_group or hi_g not in by_group:
            continue
        pairs, deltas, n = _paired(by_group[lo_g], by_group[hi_g])
        if n == 0:
            continue
        d = sum(deltas) / n
        bb, cc, p = _mcnemar_p(pairs)
        ci_lo, ci_hi = _bootstrap_ci(deltas)
        sig = "significant" if p < 0.05 else "n.s."
        print(f"    {label}: Δ={d:+.1%} on {n} shared  "
              f"[95% CI {ci_lo:+.1%},{ci_hi:+.1%}]  "
              f"McNemar b={bb} c={cc} p={p:.3f} ({sig})")


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    print("=" * 70)
    print(f"  SkillForge A/B/C analysis  ({results_dir})")
    print("  EM = binary correctness; paired stats on tasks all groups share")
    print("=" * 70)
    for name in BENCHMARKS:
        analyze_benchmark(name, results_dir / name)
    print("\nNote: with n≈30, a few-point EM gap is usually n.s. Increase task")
    print("count and/or seeds, and prefer the paired Δ + CI over raw group means.")


if __name__ == "__main__":
    main()
