#!/usr/bin/env python3
"""Ablation driver (paper tab:ablation) — stage-cumulative curation + an
equal-budget control, WITHOUT a second experiment harness.

Design (decided 2026-07-02): LoCoMo + GAIA2, n=100, ITER_CHAIN=3, the PRIMARY
backbone, same tasks as the main table. Arms B (raw store) and C (full method)
are REUSED from the main sweep at experiments_results/latest/<model>/ — only
three new arms run here:

  C_refine        refinement only            (C_USE_CRITIC=0, C_USE_ENRICH=0)
  C_refine_critic refinement + critic        (C_USE_CRITIC=1, C_USE_ENRICH=0)
  ctrl_reprompt   equal-budget, no memory    (REPROMPT_CONTROL=1: the baseline
                  answer gets REPROMPT_CALLS=2 generic self-refinement calls,
                  spending C's write-time budget with no store)

Each arm is one subprocess invocation of latest_runner.py — the ONE harness —
parameterized by env (ARMS / RESULTS_BASE / BENCHMARKS / C_USE_* /
REPROMPT_CONTROL). No duplicated loop, so this file cannot drift from the main
pipeline (the disease that killed the old eval.py copy). Results land in
experiments_results/ablation/<arm>/<model>/<benchmark>/trace.jsonl; trace group
keys keep the harness names (C_gpr / A_baseline) and are relabeled by arm
directory during aggregation.

Run (server):
  ABLATION_BENCHMARKS=locomo,gaia2 TASK_LIMIT=100 ITER_CHAIN=3 \
    CODEBUDDY_MODEL=hy3-preview python scripts/latest/ablation_runner.py
Aggregate only (no runs):
  python scripts/latest/ablation_runner.py --report
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL = os.environ.get("CODEBUDDY_MODEL", "hy3-preview")
BENCHES = [b.strip() for b in
           os.environ.get("ABLATION_BENCHMARKS", "locomo,gaia2").split(",") if b.strip()]

# arm name -> (env overrides, which harness arm slot it runs, trace group key)
ARMS = {
    "C_refine":        ({"ARMS": "C", "C_USE_CRITIC": "0", "C_USE_ENRICH": "0"},
                        "C", "C_gpr"),
    "C_refine_critic": ({"ARMS": "C", "C_USE_CRITIC": "1", "C_USE_ENRICH": "0"},
                        "C", "C_gpr"),
    "C_no_wc":         ({"ARMS": "C", "W_C_DISABLED": "1"},
                        "C", "C_gpr"),
    "C_small_inject":  ({"ARMS": "C", "C_INJECT_BUDGET_CH": "500"},
                        "C", "C_gpr"),
    "ctrl_reprompt":   ({"ARMS": "A", "REPROMPT_CONTROL": "1",
                         "REPROMPT_CALLS": os.environ.get("REPROMPT_CALLS", "2")},
                        "A", "A_baseline"),
}


def _run_arm(arm: str, overrides: dict) -> int:
    env = os.environ.copy()
    env.update(overrides)
    env["RESULTS_BASE"] = f"ablation/{arm}"
    env["BENCHMARKS"] = ",".join(BENCHES)
    env.setdefault("RESUME", "1")            # crash-safe: rerun only what's missing
    env.setdefault("ITER_CHAIN", "3")
    print(f"\n=== [ablation] arm {arm} on {BENCHES} (model={MODEL}) ===", flush=True)
    return subprocess.call([sys.executable, "-u",
                            str(PROJECT_ROOT / "scripts/latest/latest_runner.py")],
                           env=env, cwd=str(PROJECT_ROOT))


def _final_iter_mean(trace: Path) -> tuple[float, float, int]:
    """Mean (score, em) over each task's LAST traced iteration (the paper's
    main-table methodology), plus n."""
    last: dict = {}
    if not trace.exists():
        return 0.0, 0.0, 0
    for line in open(trace):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tid = r.get("task_id", "?")
        cur = last.get(tid)
        if cur is None or int(r.get("iteration", 0) or 0) >= int(cur.get("iteration", 0) or 0):
            last[tid] = r
    rows = list(last.values())
    if not rows:
        return 0.0, 0.0, 0
    n = len(rows)
    return (sum(r.get("score") or 0.0 for r in rows) / n,
            sum(r.get("em") or 0.0 for r in rows) / n, n)


def report() -> None:
    """tab:ablation markdown: main-sweep B and C + the three ablation arms."""
    main_dir = PROJECT_ROOT / "experiments_results/latest" / MODEL
    rows = []

    def _arm_row(label, trace_path, group):
        # filter the trace to one group, then final-iteration mean
        tmp = []
        if trace_path.exists():
            for line in open(trace_path):
                line = line.strip()
                if line and json.loads(line).get("group") == group:
                    tmp.append(line)
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(tmp))
            p = Path(f.name)
        s, e, n = _final_iter_mean(p)
        p.unlink(missing_ok=True)
        return s, e, n

    print(f"\n## tab:ablation — model={MODEL}, benchmarks={BENCHES}\n")
    print("| arm | " + " | ".join(f"{b} score (em, n)" for b in BENCHES) + " |")
    print("|---|" + "---|" * len(BENCHES))
    plan = ([("B (raw store, main sweep)", main_dir, "B_evomem"),
             ("+ refinement", None, "C_refine"),
             ("+ critic", None, "C_refine_critic"),
             ("+ enrichment (= C, main sweep)", main_dir, "C_gpr"),
             ("C without w_c (pure similarity)", None, "C_no_wc"),
             ("C with 500-char injection budget", None, "C_small_inject"),
             ("Reprompt (equal budget, no memory)", None, "ctrl_reprompt")])
    for label, base, key in plan:
        cells = []
        for b in BENCHES:
            if base is not None:                      # reused main-sweep arm
                s, e, n = _arm_row(label, base / b / "trace.jsonl", key)
            else:                                     # ablation arm directory
                arm_dir = PROJECT_ROOT / "experiments_results/ablation" / key / MODEL
                group = ARMS[key][2]
                s, e, n = _arm_row(label, arm_dir / b / "trace.jsonl", group)
            cells.append(f"{s:.3f} ({e:.2f}, n={n})" if n else "--")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\nGate: every arm needs n>0 and (for memory arms) injected n>0 "
          "(scripts/latest/breakdown.py <arm dir>).")


def main() -> None:
    if "--report" not in sys.argv:
        failures = []
        for arm, (overrides, _slot, _g) in ARMS.items():
            rc = _run_arm(arm, overrides)
            if rc != 0:
                failures.append((arm, rc))
                print(f"  [ablation] arm {arm} exited rc={rc} — continuing", flush=True)
        if failures:
            print(f"\n[ablation] arms with nonzero exit: {failures}", flush=True)
    report()


if __name__ == "__main__":
    main()
