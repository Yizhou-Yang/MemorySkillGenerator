#!/usr/bin/env python3
"""Convert GAIA2 HuggingFace parquet datasets into the local `gaia2-cli` directory
layout expected by benchmarks/loader._load_gaia2 / are_integration.load_gaia2_tasks_from_cli_dir.

The HF dataset (meta-agents-research-environments/gaia2-cli) stores each scenario
as a parquet row: {scenario_id, scenario(json)}. This script explodes every
scenario into the per-task directory tree:

    <out>/<split>/<scenario_id>/
        environment/scenario.json      # full ARE scenario (apps+events+metadata)
        tests/oracle_task.txt          # USER event -> agent task text
        tests/oracle_events.json       # AGENT OracleEvents (ground-truth actions)
        tests/oracle_answer.txt        # best-effort final answer (ENV/last state)
        task_metadata.json             # split + scenario_id
        instruction.md                 # system prompt (generic)

Usage:
    python convert_gaia2_hf_to_cli.py --src <gaia2-cli HF dir> --out <gaia2-cli cli dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def _user_task_text(scenario: dict) -> str:
    for e in scenario.get("events", []):
        if e.get("event_type") == "USER":
            args = (e.get("action") or {}).get("args") or []
            for a in args:
                if a.get("name") == "content":
                    return str(a.get("value", "")).strip()
    return ""


def _oracle_events(scenario: dict) -> list[dict]:
    return [e for e in scenario.get("events", [])
            if e.get("event_type") == "AGENT" and e.get("class_name") == "OracleEvent"]


def _best_answer(scenario: dict) -> str:
    for e in reversed(scenario.get("events", [])):
        if e.get("event_type") == "ENV":
            args = (e.get("action") or {}).get("args") or []
            for a in args:
                if a.get("name") in ("content", "result", "value"):
                    return str(a.get("value", "")).strip()
    return ""


INSTRUCTION_MD = (
    "You are a helpful AI assistant with access to a set of applications "
    "(email, calendar, file system, messaging, etc.). Complete the user's task "
    "by calling the available tools. The environment state evolves as you act."
)


def convert(src: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    data_dir = src / "data"
    if not data_dir.exists():
        data_dir = src
    for pq in sorted(data_dir.glob("*-test-*.parquet")):
        split = pq.name.split("-test-")[0]
        df = pd.read_parquet(pq)
        for _, row in df.iterrows():
            sid = row["scenario_id"]
            try:
                scenario = json.loads(row["scenario"])
            except (json.JSONDecodeError, TypeError):
                continue
            task_dir = out / split / sid
            (task_dir / "environment").mkdir(parents=True, exist_ok=True)
            (task_dir / "tests").mkdir(parents=True, exist_ok=True)
            (task_dir / "environment" / "scenario.json").write_text(
                json.dumps(scenario, indent=2, ensure_ascii=False))
            task_text = _user_task_text(scenario)
            (task_dir / "tests" / "oracle_task.txt").write_text(task_text)
            oevents = _oracle_events(scenario)
            (task_dir / "tests" / "oracle_events.json").write_text(
                json.dumps(oevents, indent=2, ensure_ascii=False))
            (task_dir / "tests" / "oracle_answer.txt").write_text(
                _best_answer(scenario))
            (task_dir / "task_metadata.json").write_text(json.dumps({
                "config": split,
                "difficulty": "unknown",
                "source_id": sid,
                "scenario_id": sid,
            }, indent=2))
            (task_dir / "instruction.md").write_text(INSTRUCTION_MD)
            total += 1
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                   help="HF gaia2-cli dir (contains data/*.parquet)")
    ap.add_argument("--out", required=True,
                   help="output gaia2-cli cli dir (per-task tree)")
    args = ap.parse_args()
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    n = convert(src, out)
    print(f"[convert] wrote {n} gaia2 tasks -> {out}")


if __name__ == "__main__":
    main()
