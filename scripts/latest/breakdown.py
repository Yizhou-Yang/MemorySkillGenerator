#!/usr/bin/env python3
"""Aggregate trace.jsonl into the paper's breakdown sub-tables.

Reads each benchmark's `trace.jsonl` under a results dir and produces, per
benchmark and per A/B/C arm:

  * by-task-type  (Table 3)  -- grouped by `category`
  * by-difficulty            -- grouped by `level`
  * patch-injection isolation (Figure) -- accuracy where a patch WAS injected
    vs where it was NOT (derived from `patch_injected`, or from a non-empty
    `augmented_prompt` on older traces that predate that field)

Single sweep in, all sub-tables out -- no re-runs. Companion to
`analyze_results.py` (which does the main A/B/C table + significance).

Usage:
  python scripts/latest/breakdown.py experiments_results/latest
  python scripts/latest/breakdown.py experiments_results/latest/<model>
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

BENCHES = ["gaia", "gaia2", "locomo", "terminal_bench_2"]
# tolerate both the legacy and current group labels
GROUP = {
    "A_baseline": "A",
    "B_evoarena": "B", "B_evomem": "B",
    "C_skillforge": "C", "C_gpr": "C",
}


def _rows(path: Path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _injected(r: dict) -> bool:
    v = r.get("patch_injected")
    if v is not None:
        return bool(v)
    return bool((r.get("augmented_prompt") or "").strip())


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(m):
    return f"{m:.1%}" if m is not None else "--"


def _grouped_table(rows, field, default="(none)"):
    """Return {group: {field_value: [scores]}}."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g = GROUP.get(r.get("group", ""))
        if not g:
            continue
        key = str(r.get(field) or "") or default
        out[g][key].append(r.get("score", 0.0))
    return out


def report_benchmark(bench: str, rows):
    print(f"\n## {bench}  (n_rows={len(rows)})")

    # ---- by category (Table 3) ----
    cat = _grouped_table(rows, "category")
    cats = sorted({c for g in cat.values() for c in g})
    if cats and not (len(cats) == 1 and cats[0] == "(none)"):
        print("\n### by task type (category)")
        print("| category | A | B | C | C-B |")
        print("|---|---|---|---|---|")
        for c in cats:
            a = _mean(cat["A"].get(c, []))
            b = _mean(cat["B"].get(c, []))
            cc = _mean(cat["C"].get(c, []))
            d = (cc - b) if (cc is not None and b is not None) else None
            print(f"| {c} | {_fmt(a)} | {_fmt(b)} | {_fmt(cc)} | "
                  f"{('%+.1f%%' % (d*100)) if d is not None else '--'} |")
    else:
        print("\n### by task type: no `category` field in trace "
              "(re-run with enriched logging to populate Table 3)")

    # ---- by difficulty ----
    lvl = _grouped_table(rows, "level")
    lvls = sorted({c for g in lvl.values() for c in g})
    if lvls and not (len(lvls) == 1 and lvls[0] == "(none)"):
        print("\n### by difficulty (level)")
        print("| level | A | B | C |")
        print("|---|---|---|---|")
        for c in lvls:
            print(f"| {c} | {_fmt(_mean(lvl['A'].get(c, [])))} | "
                  f"{_fmt(_mean(lvl['B'].get(c, [])))} | "
                  f"{_fmt(_mean(lvl['C'].get(c, [])))} |")

    # ---- patch-injection isolation (Figure) ----
    print("\n### patch-injection isolation (B/C only)")
    print("| arm | injected acc (n) | not-injected acc (n) |")
    print("|---|---|---|")
    by_g = collections.defaultdict(list)
    for r in rows:
        g = GROUP.get(r.get("group", ""))
        if g:
            by_g[g].append(r)
    any_inject = False
    for g in ("B", "C"):
        inj = [r.get("score", 0.0) for r in by_g.get(g, []) if _injected(r)]
        non = [r.get("score", 0.0) for r in by_g.get(g, []) if not _injected(r)]
        any_inject = any_inject or bool(inj)
        print(f"| {g} | {_fmt(_mean(inj))} ({len(inj)}) | "
              f"{_fmt(_mean(non))} ({len(non)}) |")
    if not any_inject:
        print("\n  [!] NO patches were injected in B or C on this benchmark — "
              "the arms are identical to A (sampling noise). If unexpected, the "
              "memory did not fire (single-pass run of independent tasks has no "
              "in-chain history; use ITER_CHAIN>1, or check evomem_bridge).")

    # ---- chain-level (multi-iteration) accuracy (tab:chain) ----
    # A chain (one task_id) counts as correct only if EVERY iteration succeeded;
    # we also report the final-iteration accuracy (the main-table metric under
    # ITER_CHAIN>1). Only meaningful when iterations were actually run.
    iters = max((int(r.get("iter_total", 1) or 1) for r in rows), default=1)
    if iters > 1:
        print(f"\n### chain-level accuracy (ITER_CHAIN={iters})")
        print("| arm | chain-level (all iters correct) | final-iter acc | n chains |")
        print("|---|---|---|---|")
        for g in ("A", "B", "C"):
            chains = collections.defaultdict(dict)  # task_id -> {iter: score}
            for r in by_g.get(g, []):
                chains[r.get("task_id")][int(r.get("iteration", 0) or 0)] = r.get("score", 0.0)
            if not chains:
                continue
            chain_ok = [1.0 if all(s >= 0.5 for s in it.values()) else 0.0
                        for it in chains.values()]
            final = [it[max(it)] for it in chains.values() if it]
            print(f"| {g} | {_fmt(_mean(chain_ok))} | {_fmt(_mean(final))} | {len(chains)} |")


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments_results/latest")
    print(f"# Breakdown sub-tables — {base}")
    found = False
    for bench in BENCHES:
        tp = base / bench / "trace.jsonl"
        if tp.exists():
            found = True
            report_benchmark(bench, _rows(tp))
    if not found:
        print(f"\nNo trace.jsonl found under {base}/<benchmark>/. "
              f"For a per-model sweep pass experiments_results/latest/<model>.")


if __name__ == "__main__":
    main()
