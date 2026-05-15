#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor v5 void-case experiment progress.

Usage:
    python scripts/check_v5_status.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = PROJECT_ROOT / "experiments" / "paper_v5_stdout.log"
RESULTS = PROJECT_ROOT / "experiments" / "paper_v5_void_results.json"
SKILL_BANKS = PROJECT_ROOT / "experiments" / "paper_v5_skill_banks"

BENCHMARKS = ["hotpotqa", "2wikimultihopqa", "musique",
              "triviaqa", "gsm8k", "longmemeval", "locomo"]


def get_pid() -> int | None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_paper_v5_void.py"],
            text=True,
        ).strip()
        if not out:
            return None
        for pid in out.split("\n"):
            pid = pid.strip()
            if not pid:
                continue
            cmd_path = f"/proc/{pid}/cmdline"
            if not os.path.exists(cmd_path):
                continue
            with open(cmd_path) as f:
                cmd = f.read().replace("\x00", " ")
            if "python" in cmd and "run_paper_v5_void" in cmd:
                return int(pid)
        return None
    except subprocess.CalledProcessError:
        return None


def get_runtime(pid: int) -> str:
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "etime="], text=True,
        ).strip()
        return out
    except subprocess.CalledProcessError:
        return "n/a"


def parse_progress(log_path: Path) -> dict:
    if not log_path.exists():
        return {"error": "no log yet"}
    with open(log_path) as f:
        lines = f.readlines()

    state = {
        "current_benchmark": None,
        "current_phase": None,
        "skill_progress": None,
        "completed_benchmarks": [],
        "total_lines": len(lines),
        "errors": [],
    }
    last_smoke = ""
    for line in lines:
        if "Benchmark:" in line:
            m = re.search(r"Benchmark: (\S+)", line)
            if m:
                state["current_benchmark"] = m.group(1)
                state["current_phase"] = "induction"
                state["skill_progress"] = None
        if re.search(r"Inducing (\d+)/(\d+)", line):
            m = re.search(r"Inducing (\d+)/(\d+)", line)
            state["skill_progress"] = (int(m.group(1)), int(m.group(2)))
        if "Built " in line and "skills" in line:
            state["current_phase"] = "evaluation"
        if "→ Running" in line:
            m = re.search(r"Running (\S+)", line)
            if m:
                state["current_phase"] = f"eval[{m.group(1)}]"
        if "checkpoint →" in line:
            if state["current_benchmark"]:
                if state["current_benchmark"] not in state["completed_benchmarks"]:
                    state["completed_benchmarks"].append(state["current_benchmark"])
        if " ERROR " in line or "FATAL" in line:
            state["errors"].append(line.strip()[:200])
    return state


def parse_results(results_path: Path) -> dict:
    if not results_path.exists():
        return {"status": "no results yet"}
    try:
        with open(results_path) as f:
            data = json.load(f)
    except Exception as exc:
        return {"error": f"failed to parse: {exc}"}

    main = data.get("main_experiment", {})
    summary = {
        "elapsed_seconds": data.get("meta", {}).get("elapsed_seconds", 0),
        "completed_benchmarks": [],
    }
    for bench in BENCHMARKS:
        bd = main.get(bench)
        if not bd or "error" in bd:
            continue
        primary = bd.get("primary_metric", "em")
        row = {"benchmark": bench, "metric": primary}
        for m in data["meta"]["methods"]:
            md = bd["methods"].get(m)
            if not md:
                continue
            score = md["primary"]["avg_em"] if primary == "em" else md["primary"]["avg_f1"]
            row[m] = f"{score:.1%}"
            if "+void" in m:
                row[f"{m}_voidrate"] = f"{md['primary']['void_rate']:.0%}"
        summary["completed_benchmarks"].append(row)
    return summary


def estimate_token_usage(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    pat = re.compile(r"cumulative: \d+ calls, (\d+) tokens")
    last = None
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                last = int(m.group(1))
    return last


def main():
    print("=" * 70)
    print("SkillCurator v5 Void-Case Experiment — Status Monitor")
    print("=" * 70)

    pid = get_pid()
    if pid:
        print(f"\n✓ Running   PID={pid}   uptime={get_runtime(pid)}")
    else:
        print("\n⚠ Process not running")

    progress = parse_progress(LOG)
    print(f"\nLog file: {LOG}")
    if progress.get("error"):
        print(f"  {progress['error']}")
    else:
        print(f"  total log lines: {progress['total_lines']}")
        print(f"  current benchmark: {progress['current_benchmark']}")
        print(f"  current phase: {progress['current_phase']}")
        if progress["skill_progress"]:
            cur, tot = progress["skill_progress"]
            print(f"  skill induction: {cur}/{tot} ({cur/tot:.0%})")
        print(f"  completed benchmarks: {progress['completed_benchmarks']}")
        if progress["errors"]:
            print(f"  ⚠ errors: {len(progress['errors'])}")
            for e in progress["errors"][-3:]:
                print(f"     {e}")

    tokens = estimate_token_usage(LOG)
    if tokens is not None:
        budget = 5_000_000
        print(f"\nToken usage: {tokens:,} / {budget:,} ({tokens/budget:.1%})")

    print(f"\nSkill banks dir: {SKILL_BANKS}")
    if SKILL_BANKS.exists():
        banks = sorted(SKILL_BANKS.glob("*.json"))
        for b in banks:
            try:
                with open(b) as f:
                    n = len(json.load(f))
                print(f"  ✓ {b.name}: {n} skills")
            except Exception:
                print(f"  ⚠ {b.name}: unreadable")

    print(f"\nResults file: {RESULTS}")
    summary = parse_results(RESULTS)
    if summary.get("error") or summary.get("status"):
        print(f"  {summary.get('error') or summary.get('status')}")
    else:
        elapsed = summary.get("elapsed_seconds", 0)
        print(f"  elapsed: {elapsed/60:.1f} min")
        print(f"  benchmarks with results: {len(summary['completed_benchmarks'])}")
        for row in summary["completed_benchmarks"]:
            metric_label = row["metric"].upper()
            cells = []
            for k, v in row.items():
                if k in ("benchmark", "metric") or k.endswith("_voidrate"):
                    continue
                vr = row.get(f"{k}_voidrate", "")
                cells.append(f"{k}={v}" + (f" (vr={vr})" if vr else ""))
            print(f"    {row['benchmark']:20s} [{metric_label}]  " + "  ".join(cells))

    print("\n" + "=" * 70)
    print("Tail of stdout log:")
    print("=" * 70)
    if LOG.exists():
        with open(LOG) as f:
            tail = f.readlines()[-15:]
        for line in tail:
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()
