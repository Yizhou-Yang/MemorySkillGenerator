#!/usr/bin/env python3
"""Paired SOFT-score stats (final iteration): the paper metric for gaia2 is
soft recall (fractional), so McNemar on binarized EM under-reads it — A=9/B=11/
C=8 EM on gaia2 is 2-3 tasks of noise while the soft means differ. This prints
per-arm soft means + paired bootstrap CIs for C-B / B-A / C-A.

Usage: python scripts/latest/soft_stats.py experiments_results/latest/<model> [bench...]
"""
import json, random, sys
_LEGACY_MAP = {"A_baseline": "no_mem", "B_evomem": "raw_patch", "C_gpr": "curated_patch"}
def _norm_group(g): return _LEGACY_MAP.get(g, g)
from pathlib import Path

def final_rows(trace):
    last = {}
    dropped = 0
    for line in open(trace):
        line = line.strip()
        if not line: continue
        r = json.loads(line); r["group"] = _norm_group(r.get("group",""))
        # Infra rows (endpoint down / timeout / empty agent loop) carry an error
        # and no response; averaging their 0.0 as real scores poisons every mean
        # (llama-33 gaia had 770 such rows). Exclude them, loudly.
        if str(r.get("error") or "").strip() and not str(r.get("response") or "").strip():
            dropped += 1; continue
        k = (r["group"], r["task_id"])
        cur = last.get(k)
        if cur is None or int(r.get("iteration", 0) or 0) >= int(cur.get("iteration", 0) or 0):
            last[k] = r
    if dropped:
        print(f"  [filter] {trace}: dropped {dropped} infra error-rows (error set, empty response)")
    return last

def boot_ci(deltas, n=5000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(n))
    return means[int(0.025*n)], means[int(0.975*n)]

def main():
    base = Path(sys.argv[1])
    benches = sys.argv[2:] or ["gaia", "gaia2", "locomo", "terminal_bench_2"]
    for b in benches:
        tr = base / b / "trace.jsonl"
        if not tr.exists(): continue
        last = final_rows(tr)
        arms = {}
        for (g, t), r in last.items():
            arms.setdefault(g, {})[t] = r.get("score") or 0.0
        print(f"\n### {b} (soft score, final iteration)")
        for g in sorted(arms):
            v = list(arms[g].values())
            print(f"  {g:12s} mean={sum(v)/len(v):.3f} n={len(v)}")
        pairs = [("curated_patch","raw_patch"),("raw_patch","no_mem"),("curated_patch","no_mem")]
        for hi, lo in pairs:
            if hi not in arms or lo not in arms: continue
            shared = sorted(set(arms[hi]) & set(arms[lo]))
            if not shared: continue
            d = [arms[hi][t]-arms[lo][t] for t in shared]
            m = sum(d)/len(d); lo_ci, hi_ci = boot_ci(d)
            sig = "SIGNIFICANT" if lo_ci > 0 or hi_ci < 0 else "n.s."
            print(f"  {hi.split('_')[0]} vs {lo.split('_')[0]}: Δ={m:+.3f} "
                  f"[95% CI {lo_ci:+.3f},{hi_ci:+.3f}] on {len(d)} shared ({sig})")

if __name__ == "__main__":
    main()
