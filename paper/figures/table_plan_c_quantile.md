# Plan C — Per-Benchmark Quantile Heuristic (Paper main result)

Method: τ_b = quantile_{q*(b)} of train s_max, q* selected by 5-fold CV averaged over 5 seeds.

**Aggregate (excl. locomo, n=6):** Δ vs max(B0, A3) = **+2.38 pp** (wins 6/6).

**Aggregate (incl. locomo, n=7):** Δ = **+2.32 pp** (wins 7/7). Note: locomo is flagged ⚠ EM not applicable (see appendix).

| Benchmark | metric | n | B0 | A3 | max(base) | **A3+c_∅ (Plan C)** | Δ pp | q* (mode) | τ_b | stable |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | EM | 50 | 74.0% | 74.0% | 74.0% | **74.0%** ± 0.0pp | +0.00 | 0.10 | 0.115 | 100% |
| 2wikimultihopqa | EM | 50 | 58.0% | 60.0% | 60.0% | **66.0%** ± 0.0pp | +6.00 | 0.60 | 0.285 | 60% |
| musique | EM | 50 | 42.0% | 40.0% | 42.0% | **43.6%** ± 0.8pp | +1.60 | 0.40 | 0.197 | 80% |
| triviaqa | EM | 30 | 76.7% | 80.0% | 80.0% | **80.0%** ± 0.0pp | +0.00 | 0.20 | 0.250 | 40% |
| gsm8k | EM | 30 | 60.0% | 63.3% | 63.3% | **70.0%** ± 0.0pp | +6.67 | 0.10 | 0.331 | 100% |
| longmemeval | EM | 20 | 40.0% | 30.0% | 40.0% | **40.0%** ± 0.0pp | +0.00 | 0.60 | 0.494 | 40% |
| locomo ⚠ | F1 | 20 | 9.8% | 13.0% | 13.0% | **14.9%** ± 0.0pp | +1.95 | 0.10 | 0.459 | 100% |