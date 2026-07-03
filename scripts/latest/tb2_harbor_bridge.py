#!/usr/bin/env python3
"""Drive official harbor Terminus-2 runs as our A/B/C iteration chains
(HARBOR_TB2_PLAN §3) and convert results into our trace.jsonl schema.

Per arm: run the SAME task set ITER_CHAIN times; between iterations, parse
harbor's per-task rewards, mem.record(task, result, score) (record-after-eval),
persist the store (pickle) so the NEXT iteration's CuratedTerminus injects it.

Usage (server, after Gate 1 smoke of HARBOR_TB2_PLAN §1):
  # Start the proxy first:
  nohup /root/.conda/envs/skillforge/bin/python scripts/latest/codebuddy_oai_proxy.py &

  # Then run the bridge:
  OPENAI_API_BASE=http://localhost:8741/v1 OPENAI_API_KEY=dummy \
  /root/.conda/envs/harbor312/bin/python scripts/latest/tb2_harbor_bridge.py \
      --arm B --iters 3 --model openai/hy3-preview --n-tasks 80

Trace lands in experiments_results/harbor_tb2/<model-slug>/terminal_bench_2/
(kept separate from the simplified-loop results until Gate 4 signs off).

NOTE: Uses `terminal-bench run` CLI (not `harbor run`). The dataset is at
.datasets/terminal-bench-2 (downloaded via `terminal-bench datasets download`).
For arms B/C, uses --agent-import-path with CuratedTerminus subclass.
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


def _agent_transcript(run_dir: Path, task_id: str) -> str:
    """Best-effort read of the agent's own output for one task (commands +
    final pane), used as the B/C patch content. Layouts vary by terminal-bench
    version — we try the common files under the task's dir; VERIFY on the
    pinned version that at least one matches (else patches fall back to a
    one-line summary and memory arms carry little signal)."""
    task_dirs = [d for d in run_dir.rglob(task_id) if d.is_dir()]
    parts: list[str] = []
    for td in task_dirs[:2]:
        for pat in ("**/commands.txt", "**/post-agent*.txt", "**/post-agent*.pane",
                    "**/agent-logs/**/*.log", "**/episode*.json"):
            for fp in sorted(td.glob(pat))[:3]:
                try:
                    s = fp.read_text(errors="replace").strip()
                except Exception:
                    continue
                if s:
                    parts.append(s[-1500:])
            if parts:
                break
        if parts:
            break
    return ("\n".join(parts))[:2000]


def _parse_results(run_dir: Path) -> list[dict]:
    """Parse terminal-bench v0.2.18 results.json into
    [{task_id, task_name, instruction, reward, response}].

    The results.json schema (v0.2.18):
      {
        "results": [{
          "task_id": str,
          "instruction": str,
          "is_resolved": bool,
          "failure_mode": str,
          "parser_results": {"test_name": "passed"|"failed"},
          ...
        }],
        "accuracy": float,
        "n_resolved": int,
      }
    """
    out = []
    # Find the top-level results.json (not per-task ones)
    # newest first: a rerun/retry may leave multiple results.json in the tree
    candidates = sorted(run_dir.rglob("results.json"),
                        key=lambda q: q.stat().st_mtime, reverse=True)
    for rj in candidates:
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        rows = data.get("results", [])
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict):
                continue
            task_id = str(r.get("task_id") or r.get("task_name") or "")
            if not task_id:
                continue
            # Primary: is_resolved boolean
            resolved = r.get("is_resolved")
            if resolved is not None:
                reward = 1.0 if resolved else 0.0
            else:
                # Fallback: parser_results
                pr = r.get("parser_results") or {}
                if pr:
                    passed = sum(1 for v in pr.values() if v == "passed")
                    total = len(pr)
                    reward = passed / total if total > 0 else 0.0
                else:
                    reward = r.get("reward", r.get("score", 0.0))
                    if isinstance(reward, bool):
                        reward = float(reward)
            out.append({
                "task_id": task_id,
                "task_name": task_id,
                "instruction": str(r.get("instruction") or "")[:2000],
                "reward": float(reward),
                # patch content for B/C = the agent's actual transcript (what it
                # DID), not the failure_mode label — an injected "unset" teaches
                # nothing. Falls back to a resolved-summary when no log is found.
                "response": _agent_transcript(run_dir, task_id) or
                            f"resolved={bool(reward >= 1.0)}; failure_mode="
                            f"{r.get('failure_mode') or 'n/a'}",
                "failure_mode": str(r.get("failure_mode") or "")[:200],
            })
        if out:
            break  # Use the first valid results.json found
    if not out:
        raise RuntimeError(
            f"no parsable results under {run_dir} — inspect the run layout and "
            "extend _parse_results()")
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
    ap.add_argument("--model", default=os.environ.get("TB2_MODEL", "openai/hy3-preview"))
    ap.add_argument("--dataset-path",
                    default=str(PROJECT_ROOT / ".datasets" / "terminal-bench-2"),
                    help="Path to downloaded terminal-bench-core dataset")
    ap.add_argument("--n-tasks", type=int,
                    default=int(os.environ.get("TB2_N_TASKS", "50")),
                    help="tasks per iteration (default 50; 0 = all ~88 — raw output is ~5MB/task/iter, keep runs_* out of git)")
    ap.add_argument("--n-concurrent", type=int, default=4)
    ap.add_argument("--task-ids", nargs="*", default=None,
                    help="Specific task IDs to run (default: all)")
    args = ap.parse_args()

    slug = re.sub(r"[^A-Za-z0-9._-]", "_", args.model.split("/")[-1])
    out_root = PROJECT_ROOT / "experiments_results/harbor_tb2" / slug
    trace_dir = out_root / "terminal_bench_2"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = trace_dir / "trace.jsonl"
    state = out_root / f"mem_{args.arm}.pkl"
    rev = _code_rev()

    mem = _mk_memory(args.arm)
    if mem is not None:                       # fail fast, not at iter-0's end
        pickle.loads(pickle.dumps(mem))
        print(f"[bridge] arm {args.arm} store pickle round-trip ok", flush=True)
    import asyncio
    tb_bin = "/root/.conda/envs/harbor312/bin/terminal-bench"
    for it in range(args.iters):
        run_dir = out_root / f"runs_{args.arm}_iter{it}"
        env = os.environ.copy()
        env["TB2_ARM"] = args.arm
        env["TB2_MEM_STATE"] = str(state)
        # Arm A uses vanilla terminus-2; B/C use our memory-prefix subclass
        if args.arm == "A":
            cmd = [tb_bin, "run", "-a", "terminus-2"]
        else:
            cmd = [tb_bin, "run",
                   "--agent-import-path",
                   "scripts.latest.tb2_harbor_agent:CuratedTerminus"]
        cmd += ["-m", args.model,
                "-p", args.dataset_path,
                "--output-path", str(run_dir),
                "--n-attempts", "1",
                "--n-concurrent", str(args.n_concurrent),
                "--no-rebuild"]
        if args.n_tasks:
            cmd += ["--n-tasks", str(args.n_tasks)]
        if args.task_ids:
            for tid in args.task_ids:
                cmd += ["-t", tid]
        print(f"[bridge] arm={args.arm} iter={it}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT))
        if rc != 0:
            print(f"[bridge] terminal-bench exited rc={rc} — stopping arm", flush=True)
            break
        rows = _parse_results(run_dir)
        _empty = sum(1 for r in rows if not r["response"]
                     or r["response"].startswith("resolved="))
        if rows and _empty / len(rows) > 0.5:
            print(f"[bridge] WARNING: {_empty}/{len(rows)} tasks have no agent "
                  f"transcript — _agent_transcript() glob patterns likely do not "
                  f"match this terminal-bench version's layout; B/C patch content "
                  f"is degraded to one-line summaries (VERIFY marker, plan §3)",
                  flush=True)
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
