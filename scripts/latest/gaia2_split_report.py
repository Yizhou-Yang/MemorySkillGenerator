#!/usr/bin/env python3
"""GAIA2 per-split A/B/C report (execution / search / adaptability / ambiguity /
time) — the axis the paper's thesis actually targets (dynamic splits).

The loader has always stored metadata.config, but pre-fix traces did not log it
(the trace `category` fallback chain missed "config"; fixed alongside this
script). This script needs NO rerun: it rebuilds {task_id -> config} from the
dataset's task_metadata.json files, annotates the trace in place (adds a
"split" field), and prints the per-split final-iteration soft table with the
C-B delta per split.

Usage (server, where the dataset lives):
  GAIA2_SCENARIO_DIR=$PWD/.datasets/gaia2-cli \
    python scripts/latest/gaia2_split_report.py experiments_results/latest/hy3-preview
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_split_map(scenario_dir: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    for td in sorted(scenario_dir.iterdir()):
        meta = td / "task_metadata.json"
        if not td.is_dir() or not meta.exists():
            continue
        try:
            d = json.loads(meta.read_text())
        except Exception:
            continue
        sid = d.get("source_id", td.name)
        m[f"gaia2_{sid}"] = d.get("config", "unknown")
    return m


def main() -> None:
    base = Path(sys.argv[1] if len(sys.argv) > 1 else
                "experiments_results/latest/hy3-preview")
    trace = base / "gaia2" / "trace.jsonl"
    sdir = Path(os.environ.get("GAIA2_SCENARIO_DIR",
                               PROJECT_ROOT / ".datasets" / "gaia2-cli"))
    if not sdir.exists():
        sys.exit(f"dataset dir not found: {sdir} (set GAIA2_SCENARIO_DIR)")
    smap = build_split_map(sdir)
    print(f"split map: {len(smap)} tasks from {sdir}")

    rows = [json.loads(l) for l in open(trace) if l.strip()]
    unmapped = 0
    for r in rows:
        s = smap.get(r.get("task_id", ""))
        if s is None:
            unmapped += 1
        r["split"] = s or r.get("category") or "unknown"
    with open(trace, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"annotated {len(rows)} rows in place ({unmapped} unmapped)")

    # final iteration per (group, task), then per-split table
    last: dict = {}
    for r in rows:
        k = (r["group"], r["task_id"])
        cur = last.get(k)
        if cur is None or int(r.get("iteration", 0) or 0) >= int(cur.get("iteration", 0) or 0):
            last[k] = r
    by: dict = defaultdict(lambda: defaultdict(dict))
    for (g, t), r in last.items():
        by[r["split"]][g][t] = r.get("score") or 0.0

    print(f"\n{'split':14s} {'n':>3s}   A      B      C      C-B     B-A")
    order = ["adaptability", "time", "ambiguity", "execution", "search"]
    for s in sorted(by, key=lambda x: (order.index(x) if x in order else 99, x)):
        d = by[s]
        n = len(d.get("C_gpr", {}) or d.get("A_baseline", {}))

        def mean(g):
            v = list(d.get(g, {}).values())
            return sum(v) / len(v) if v else float("nan")

        a, b, c = mean("A_baseline"), mean("B_evomem"), mean("C_gpr")
        print(f"{s:14s} {n:3d}  {a:.3f}  {b:.3f}  {c:.3f}  {c-b:+.3f}  {b-a:+.3f}")
    print("\n(dynamic splits = adaptability/time/ambiguity — where the thesis "
          "predicts C-B > 0; execution/search are the static controls)")


if __name__ == "__main__":
    main()
