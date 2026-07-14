#!/usr/bin/env python3
"""pass@k variance/compute control (Wen 07/06): k independent no-memory
samples per task versus the memory arms at a matched call budget.

Reads groups no_mem_passk_s0..s{k-1} (written by latest_runner with
PASSK=<k> ITER_CHAIN=1 ARMS=A) plus the normal arms, and reports:

  1. pass@j for j=1..k  (a task counts if ANY of j samples scores >= 0.5)
  2. mean single-sample accuracy with a between-sample std (the variance
     the control exists to expose)
  3. the matched-budget comparison: curated_patch spends ~1 solve call
     + W write-time calls per iteration (W ~= 2.5: refine + critic
     [+ enrich]) and reaches its accuracy with ONE final answer, while
     pass@k spends k solve calls and needs an oracle to pick among k
     answers - so pass@k is an UPPER bound for resampling at that budget,
     not an achievable agent.

Usage: python scripts/latest/passk_report.py <model> [bench ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE = os.environ.get("BASE", "latest_evolving")
W_WRITE = float(os.environ.get("PASSK_W_WRITE", "2.5"))

try:
    from scripts.latest.arms import norm_group
except ImportError:
    _L = {"A_baseline": "no_mem", "B_evomem": "raw_patch", "C_gpr": "curated_patch"}
    def norm_group(g): return _L.get(g, g)


def rows(p: Path):
    if not p.exists():
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            r["group"] = norm_group(r.get("group", ""))
            out.append(r)
    return out


def main() -> None:
    model = (sys.argv[1] if len(sys.argv) > 1 else "hy3-preview-ioa").lower()
    benches = sys.argv[2:] or ["gaia", "gaia2", "locomo"]
    for bench in benches:
        rs = rows(PROJECT_ROOT / "experiments_results" / BASE / model / bench / "trace.jsonl")
        if not rs:
            print(f"[{bench}] no trace")
            continue
        kf = max((int(r.get("iter_total", 1) or 1) for r in rs), default=1) - 1
        # no-memory control: k independent single-shot samples
        nm = {}               # task -> {sample_idx: score}
        # raw-patch control: chain run once, FINAL iteration resampled k times
        # (sample 0 = the arm's own final row; s1.. = raw_patch_passk_s<i>)
        rp = {}
        for r in rs:
            g = r["group"]
            if g.startswith("no_mem_passk_s"):
                s = int(g.rsplit("_s", 1)[1])
                nm.setdefault(r["task_id"], {})[s] = r.get("score") or 0.0
            elif g.startswith("raw_patch_passk_s"):
                s = int(g.rsplit("_s", 1)[1])
                rp.setdefault(r["task_id"], {})[s] = r.get("score") or 0.0
            elif g == "raw_patch" and (r.get("iteration", 0) or 0) == kf:
                rp.setdefault(r["task_id"], {})[0] = r.get("score") or 0.0
        cur = {r["task_id"]: r.get("score") or 0.0 for r in rs
               if r["group"] == "curated_patch"
               and (r.get("iteration", 0) or 0) == kf}

        def _passk_block(label, samples, chain_calls):
            if not samples:
                return None
            k = max(max(v) for v in samples.values()) + 1
            tasks = sorted(t for t, v in samples.items() if len(v) == k)
            if not tasks or k < 2:
                return None
            print(f"  {label} (n={len(tasks)}, k={k}):")
            for j in range(1, k + 1):
                pk = sum(1 for t in tasks
                         if any(samples[t][s] >= 0.5 for s in range(j))) / len(tasks)
                print(f"    pass@{j}: {100*pk:.2f}%   "
                      f"(budget: {chain_calls + j - 1 + 1:.0f} solve calls, "
                      "oracle answer-selection)" if chain_calls else
                      f"    pass@{j}: {100*pk:.2f}%   (budget: {j} solve calls, "
                      "oracle answer-selection)")
            means = [sum(samples[t][s] >= 0.5 for t in tasks) / len(tasks)
                     for s in range(k)]
            mu = sum(means) / k
            sd = (sum((m - mu) ** 2 for m in means) / k) ** 0.5
            print(f"    single-sample: {100*mu:.2f}% ± {100*sd:.2f}pp "
                  "(between-sample std)")
            return set(tasks)

        print(f"\n[{bench}]")
        t1 = _passk_block("no_mem pass@k (single-shot)", nm, 0)
        t2 = _passk_block(f"raw_patch pass@k (chain kept, final iter resampled)",
                          rp, kf)
        if t1 is None and t2 is None:
            print("  no pass@k groups — run PASSK=k ITER_CHAIN=1 ARMS=A "
                  "(no_mem) and/or PASSK_FINAL=k ARMS=B (raw_patch) first")
            continue
        if cur:
            base = t2 or t1 or set(cur)
            common = [t for t in base if t in cur]
            if common:
                ca = sum(cur[t] >= 0.5 for t in common) / len(common)
                print(f"  curated_patch (final iter, n={len(common)}): "
                      f"{100*ca:.2f}% at ~{kf + 1 + (kf + 1) * W_WRITE:.1f} "
                      "calls/chain, single final answer (no oracle)")
                print("  note: write-time calls are short single-shot "
                      "completions; solve calls are full agent loops — on "
                      "agentic benches one solve dwarfs one write in tokens, "
                      "so call counts overstate curation's relative cost.")


if __name__ == "__main__":
    main()
