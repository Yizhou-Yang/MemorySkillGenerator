#!/usr/bin/env python3
"""Drive official harbor Terminus-2 runs as our A/B/C iteration chains
(HARBOR_TB2_PLAN §3) and convert results into our trace.jsonl schema.

Per arm: run the SAME task set ITER_CHAIN times; between iterations, parse
harbor's per-task rewards, mem.record(task, result, score) (record-after-eval),
persist the store (pickle) so the NEXT iteration's CuratedTerminus injects it.

Usage (server, after Gate 1 smoke of HARBOR_TB2_PLAN §1):
  python scripts/latest/tb2_harbor_bridge.py --arm B --iters 3 \
      --model openai/hy3-preview-ioa --n-tasks 88
Trace lands in experiments_results/harbor_tb2/<model-slug>/terminal_bench_2/
(kept separate from the simplified-loop results until Gate 4 signs off).

VERIFY ON SERVER: the `harbor run` agent-path flag (`-a module:Class` vs
`--agent-import-path`) and the results.json schema of the pinned version —
_parse_results() tries the common layouts and fails loudly otherwise.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

GROUP_KEY = {"A": "A_baseline", "B": "B_evomem", "C": "C_gpr"}


def _task_key(instruction: str) -> str:
    return "tb2h_" + hashlib.sha1((instruction or "").encode()).hexdigest()[:12]


def _code_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _mk_memory(arm: str, benchmark: str = "terminal_bench_2"):
    if arm == "A":
        return None
    from scripts.latest.evomem_bridge import BenchmarkMemory, CuratedMemory
    return BenchmarkMemory(benchmark, "B") if arm == "B" else CuratedMemory(benchmark)


def _parse_results(run_dir: Path) -> list[dict]:
    """Best-effort parse of harbor run output into
    [{task_id, task_name, instruction, reward, response}]. Tries the common
    layouts; extend after pinning the harbor version (plan §0)."""
    out = []
    candidates = (list(run_dir.rglob("results.json"))
                  + list(run_dir.rglob("result.json")))
    for rj in candidates:
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        rows = (data.get("results") if isinstance(data, dict) else data) or []
        if isinstance(rows, dict):
            rows = [dict(v, task_name=k) if isinstance(v, dict) else
                    {"task_name": k, "reward": v} for k, v in rows.items()]
        for r in rows:
            if not isinstance(r, dict):
                continue
            name = str(r.get("task_name") or r.get("task_id") or r.get("name") or "")
            reward = r.get("reward", r.get("score",
                     r.get("resolved", r.get("is_resolved"))))
            if reward is None and "parser_results" in r:
                pr = r["parser_results"] or {}
                vals = [v for v in pr.values() if isinstance(v, (int, float, bool))]
                reward = (sum(map(float, vals)) / len(vals)) if vals else None
            if not name or reward is None:
                continue
            out.append({
                "task_id": name, "task_name": name,
                "instruction": str(r.get("instruction") or r.get("task_description") or ""),
                "reward": float(bool(reward)) if isinstance(reward, bool) else float(reward),
                "response": str(r.get("final_response") or r.get("agent_output") or "")[:2000],
            })
    if not out:
        raise RuntimeError(
            f"no parsable results under {run_dir} — inspect the run layout and "
            "extend _parse_results() (VERIFY marker, plan §3)")
    return out


async def _record_all(mem, rows: list[dict]) -> None:
    for r in rows:
        task = {"task_id": _task_key(r["instruction"]) if r["instruction"] else r["task_id"],
                "description": r["instruction"] or r["task_name"]}
        await mem.record(task, {"response": r["response"] or f"reward={r['reward']:.2f}"},
                         score=r["reward"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C"], required=True)
    ap.add_argument("--iters", type=int, default=int(os.environ.get("ITER_CHAIN", "3")))
    ap.add_argument("--model", default=os.environ.get("TB2_MODEL", "openai/hy3-preview-ioa"))
    ap.add_argument("--dataset", default="terminal-bench/terminal-bench-2")
    ap.add_argument("--n-tasks", type=int, default=0, help="0 = all")
    ap.add_argument("--agent-flag", default="-a",
                    help="harbor's agent flag (VERIFY: -a vs --agent-import-path)")
    args = ap.parse_args()

    slug = re.sub(r"[^A-Za-z0-9._-]", "_", args.model.split("/")[-1])
    out_root = PROJECT_ROOT / "experiments_results/harbor_tb2" / slug
    trace_dir = out_root / "terminal_bench_2"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = trace_dir / "trace.jsonl"
    state = out_root / f"mem_{args.arm}.pkl"
    rev = _code_rev()

    mem = _mk_memory(args.arm)
    import asyncio
    for it in range(args.iters):
        run_dir = out_root / f"runs_{args.arm}_iter{it}"
        env = os.environ.copy()
        env["TB2_ARM"] = args.arm
        env["TB2_MEM_STATE"] = str(state)
        agent = ("terminus-2" if args.arm == "A"
                 else "scripts.latest.tb2_harbor_agent:CuratedTerminus")
        cmd = ["harbor", "run", "-d", args.dataset, args.agent_flag, agent,
               "-m", args.model, "-k", "1", "--output-dir", str(run_dir)]
        if args.n_tasks:
            cmd += ["--n-tasks", str(args.n_tasks)]
        print(f"[bridge] arm={args.arm} iter={it}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT))
        if rc != 0:
            print(f"[bridge] harbor exited rc={rc} — stopping arm", flush=True)
            break
        rows = _parse_results(run_dir)
        with open(trace, "a") as f:
            for r in rows:
                f.write(json.dumps({
                    "benchmark": "terminal_bench_2", "group": GROUP_KEY[args.arm],
                    "phase": "test", "task_id": r["task_id"],
                    "task_desc": (r["instruction"] or r["task_name"])[:500],
                    "score": r["reward"], "em": 1.0 if r["reward"] >= 1.0 else 0.0,
                    "response": r["response"], "error": "",
                    "iteration": it, "iter_total": args.iters,
                    "method": "harbor_terminus2", "code_rev": rev,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }) + "\n")
        print(f"[bridge] iter {it}: {len(rows)} tasks, "
              f"mean reward {sum(r['reward'] for r in rows)/len(rows):.3f}", flush=True)
        if mem is not None:
            asyncio.run(_record_all(mem, rows))
            with open(state, "wb") as f:
                pickle.dump(mem, f)     # next iteration's agent injects this
    print(f"[bridge] done. trace: {trace}", flush=True)


if __name__ == "__main__":
    main()
