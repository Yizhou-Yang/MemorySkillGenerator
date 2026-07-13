#!/usr/bin/env python3
"""Drive official tau2-bench runs as our A/B/C iteration chains and convert the
results into our trace.jsonl schema — the Docker-free replacement for the
terminal-bench bridge (tb2_harbor_bridge.py, now under scripts/latest/obsolete/).

Per arm: run the SAME task set ITER_CHAIN times; between iterations, parse tau2's
per-simulation rewards, mem.record(scenario, transcript, reward) (record-after-
eval), persist the store (pickle) so the NEXT iteration's CuratedTau2Agent
injects it. Arms A/B/C map to no_mem / raw_patch / curated_patch, identical to
every other benchmark.

Why tau2 (vs terminal-bench): no Docker (pure-Python simulated retail/airline/
telecom domains, `uv sync`), an order of magnitude cheaper per task, dynamic
(a user simulator drives each dialog), and the current authoritative tool-agent
benchmark. Its known weakness — inconsistent policy adherence / pass^k collapse
— is exactly what an accumulated patch memory targets.

Usage (server, after a Gate-1 smoke — see VERIFY notes):
  # point tau2's LLM at our OpenAI-compatible endpoint (CodeBuddy proxy / vLLM):
  export OPENAI_API_BASE=http://localhost:8741/v1 OPENAI_API_KEY=dummy
  # run each arm (B/C need our repo importable inside tau2's process):
  PYTHONPATH=$PWD TAU2_BIN=tau2 python scripts/latest/tau2_bridge.py \
      --arm B --iters 3 --model openai/hy3 --domain airline --n-tasks 30

Trace lands in experiments_results/<RESULTS_BASE>/<model-slug>/tau2/trace.jsonl
(alongside gaia/gaia2/locomo so gate + pooling pick it up like any benchmark).

VERIFY ON SERVER (Gate 1) — two tau2-version-specific surfaces, both isolated:
  1. _tau2_cmd(): the exact `tau2 run` flags (run `tau2 run --help` on the pinned
     version — flag spellings for domain / agent-llm / user-llm / num-trials /
     num-tasks / save-to / the custom-agent import path vary by release).
  2. _parse_results(): the results JSON schema (simulations[].reward_info.reward,
     the message list, the opening user turn). Defensive fallbacks included; if
     nothing parses it raises with the run path so you can extend it once.
Everything else (arms, iteration chaining, record/persist, trace schema, the
scenario chain key) mirrors the TB2 bridge and needs no per-run tweaking.
"""
from __future__ import annotations

import argparse
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

GROUP_KEY = {"A": "no_mem", "B": "raw_patch", "C": "curated_patch"}  # canonical (arms.py)


def _task_key(user_text: str) -> str:
    """Stable per-scenario id from the opening user message. MUST match
    tau2_agent._task_key so the bridge's record() and the agent's inject() land
    on the same chain across iterations."""
    return "tau2_" + hashlib.sha1((user_text or "").encode()).hexdigest()[:12]


def _code_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _mk_memory(arm: str, benchmark: str = "tau2"):
    if arm == "A":
        return None
    from scripts.latest.evomem_bridge import BenchmarkMemory, CuratedMemory
    return BenchmarkMemory(benchmark, "B") if arm == "B" else CuratedMemory(benchmark)


def _opening_user(messages: list) -> str:
    """The scenario key material: the user turn(s), oldest first (the opener is
    deterministic per scenario, so its hash chains a scenario's iterations)."""
    parts = [m.get("content") for m in messages
             if isinstance(m, dict) and m.get("role") == "user"
             and isinstance(m.get("content"), str) and m.get("content").strip()]
    return "\n".join(parts)[:2000]


def _transcript(messages: list) -> str:
    """B/C patch content = what the agent actually DID: its assistant turns and
    tool calls (not a reward label — an injected label teaches nothing). Falls
    back to a one-line reward summary in _parse_results when empty."""
    out: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("assistant", "tool"):
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            out.append(f"{role}: {c.strip()}")
        for tc in (m.get("tool_calls") or []):
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name, args = fn.get("name"), fn.get("arguments")
            if name:
                out.append(f"tool_call: {name}({str(args)[:200]})")
    return ("\n".join(out))[:2000]


def _parse_results(save_to: Path) -> list[dict]:
    """Parse a tau2 results JSON into
    [{task_id, task_name, instruction, reward, response, failure_mode}].

    VERIFY on the pinned version. Best-known schema:
      {"simulations": [{
          "task_id": str, "trial": int,
          "reward_info": {"reward": 0.0|1.0, ...},
          "messages": [{"role","content","tool_calls"}, ...],
          "termination_reason": str}], ...}
    Fallbacks: results/tasks for the list, reward for reward_info.reward,
    trajectory/conversation for messages."""
    # tau2 `--save-to X.json` may write X.json as a FILE, or (version-dependent)
    # create X.json as a DIRECTORY holding results.json — resolve to the json.
    if save_to.is_dir():
        cands = sorted(save_to.rglob("results.json"),
                       key=lambda q: q.stat().st_mtime, reverse=True) \
            or sorted(save_to.glob("*.json"))
        if not cands:
            raise RuntimeError(
                f"{save_to} is a directory with no results.json — inspect layout")
        save_to = cands[0]
    try:
        data = json.loads(save_to.read_text())
    except Exception as e:
        raise RuntimeError(f"cannot read tau2 results {save_to}: {e}")
    sims = None
    if isinstance(data, dict):
        for k in ("simulations", "results", "tasks", "runs"):
            if isinstance(data.get(k), list):
                sims = data[k]
                break
    elif isinstance(data, list):
        sims = data
    if sims is None:
        raise RuntimeError(
            f"no simulation list in {save_to} (looked for simulations/results/"
            "tasks) — inspect the layout and extend _parse_results()")
    out, skipped_infra = [], 0
    for i, s in enumerate(sims):
        if not isinstance(s, dict):
            continue
        messages = (s.get("messages") or s.get("trajectory")
                    or s.get("conversation") or [])
        term = str(s.get("termination_reason") or "")
        # A sim where the agent NEVER ran (harness/infra crash, no transcript) is
        # not a task result: dropping it stops a broken run from masquerading as an
        # all-zero no_mem baseline (which would silently inflate B/C gains). Real
        # attempts that scored 0 keep their messages, so they are retained.
        if not messages and term in ("infrastructure_error", "error", ""):
            skipped_infra += 1
            continue
        ri = s.get("reward_info") or {}
        reward = ri.get("reward") if isinstance(ri, dict) else None
        if reward is None:
            reward = s.get("reward", s.get("score", 0.0))
        if isinstance(reward, bool):
            reward = float(reward)
        instruction = _opening_user(messages) or str(s.get("instruction") or "")
        task_id = str(s.get("task_id") or s.get("id") or f"tau2_task_{i}")
        trial = s.get("trial", s.get("trial_id", ""))
        out.append({
            "task_id": f"{task_id}#{trial}" if trial != "" else task_id,
            "task_name": task_id,
            "instruction": instruction[:2000],
            "reward": float(reward),
            "response": _transcript(messages)
                        or f"reward={float(reward):.2f}; "
                           f"term={s.get('termination_reason') or 'n/a'}",
            "failure_mode": str(s.get("termination_reason") or "")[:200],
        })
    if skipped_infra:
        print(f"[tau2] WARNING: dropped {skipped_infra}/{len(sims)} sims that never "
              f"ran (infrastructure_error, empty transcript) in {save_to}", flush=True)
    if not out:
        raise RuntimeError(
            f"no usable sims in {save_to}: {skipped_infra}/{len(sims)} were "
            "infrastructure_error with no transcript — the run is BROKEN, rerun it "
            "(do not treat as an all-zero baseline)")
    return out


async def _record_all(mem, rows: list[dict]) -> None:
    for r in rows:
        key = _task_key(r["instruction"]) if r["instruction"] else r["task_name"]
        task = {"task_id": key, "description": r["instruction"] or r["task_name"],
                "metadata": {"chain_id": key}}
        await mem.record(task, {"response": r["response"] or f"reward={r['reward']:.2f}"},
                         score=r["reward"])


def _tau2_cmd(tau2_bin: str, arm: str, model: str, domain: str, n_tasks: int,
              num_trials: int, concurrency: int, save_to: Path,
              task_ids: list[str] | None,
              launcher_path: str | None = None) -> list[str]:
    """Build the `tau2 run` invocation. VERIFY the flag spellings on the pinned
    version (`tau2 run --help`); kept in one place so a mismatch is a one-line
    fix. Agent + user share our endpoint model (LiteLLM reads OPENAI_API_*).

    For arms B/C we use tau2_launcher.py which registers CuratedTau2Agent
    in tau2's global registry BEFORE the CLI parses --agent. tau2 CLI does NOT
    support --agent-import-path, so we inject the custom agent at the launcher
    level instead."""
    # Arm A uses the official tau2 CLI via python -m; arms B/C use the launcher
    # that first registers curated_tau2_agent and then delegates to tau2.cli.main()
    if arm != "A" and launcher_path:
        # Launcher registers CuratedTau2Agent THEN runs tau2.cli.main()
        bin_cmd = [sys.executable, launcher_path]
        agent_name = "curated_tau2_agent"
    else:
        # Use python -m tau2.cli (conda env has the deps; PYTHONPATH has tau2 src)
        bin_cmd = [sys.executable, "-m", "tau2.cli"]
        agent_name = "llm_agent"

    cmd = bin_cmd + ["run",
           "--domain", domain,
           "--agent", agent_name,
           "--agent-llm", model,
           "--user-llm", model,
           "--num-trials", str(num_trials),
           "--max-concurrency", str(concurrency),
           "--auto-resume",
           "--save-to", str(save_to)]
    if n_tasks:
        cmd += ["--num-tasks", str(n_tasks)]
    for t in (task_ids or []):
        cmd += ["--task-id", t]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C"], required=True)
    ap.add_argument("--iters", type=int, default=int(os.environ.get("ITER_CHAIN", "3")))
    ap.add_argument("--model", default=os.environ.get("TAU2_MODEL", "openai/hy3"))
    ap.add_argument("--domain", default=os.environ.get("TAU2_DOMAIN", "airline"),
                    help="tau2 domain: airline | retail | telecom (airline/retail "
                         "are the light ones)")
    ap.add_argument("--n-tasks", type=int,
                    default=int(os.environ.get("TAU2_N_TASKS", "30")),
                    help="tasks per iteration (0 = all in the domain)")
    ap.add_argument("--num-trials", type=int,
                    default=int(os.environ.get("TAU2_NUM_TRIALS", "2")),
                    help="attempts per task (pass^k; mirrors TB2's --n-attempts 2)")
    ap.add_argument("--n-concurrent", type=int,
                    default=int(os.environ.get("TAU2_CONCURRENCY", "4")))
    ap.add_argument("--task-ids", nargs="*", default=None,
                    help="specific tau2 task ids (default: first --n-tasks)")
    args = ap.parse_args()

    tau2_bin = os.environ.get("TAU2_BIN", "tau2")
    launcher_path = str(PROJECT_ROOT / "scripts" / "latest" / "tau2_launcher.py") if args.arm != "A" else None
    results_base = os.environ.get("RESULTS_BASE", "latest_evolving")
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", args.model.split("/")[-1]).lower()
    out_root = PROJECT_ROOT / "experiments_results" / results_base / slug
    trace_dir = out_root / "tau2"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = trace_dir / "trace.jsonl"
    state = out_root / f"tau2_mem_{args.arm}.pkl"
    rev = _code_rev()

    mem = _mk_memory(args.arm)
    if mem is not None:                       # fail fast, not at iter-0's end
        pickle.loads(pickle.dumps(mem))
        print(f"[tau2] arm {args.arm} store pickle round-trip ok", flush=True)
    import asyncio
    # tau2's litellm needs OPENAI_API_BASE + OPENAI_API_KEY (NOT openrouter vars).
    # The parent process may only have OPENROUTER_* set — translate them here so
    # the tau2 subprocess always gets a valid endpoint + key.
    os.environ.setdefault("OPENAI_API_BASE",
                          os.environ.get("OPENROUTER_BASE_URL", "http://localhost:8000/v1"))
    os.environ.setdefault("OPENAI_API_KEY",
                          os.environ.get("OPENROUTER_API_KEY", "EMPTY"))
    for it in range(args.iters):
        save_to = out_root / f"tau2_{args.domain}_{args.arm}_iter{it}.json"
        env = os.environ.copy()
        env["TAU2_ARM"] = args.arm
        env["TAU2_MEM_STATE"] = str(state)
        # Make both our repo AND tau2-bench importable (launcher needs tau2.*,
        # the custom agent needs our src). tau2-bench lives beside us in Ceph.
        tau2_root = str(PROJECT_ROOT.parent / "tau2-bench")
        extra_paths = [str(PROJECT_ROOT)]
        if os.path.isdir(tau2_root):
            extra_paths.append(tau2_root)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(extra_paths +
            ([existing] if existing else []))
        cmd = _tau2_cmd(tau2_bin, args.arm, args.model, args.domain, args.n_tasks,
                        args.num_trials, args.n_concurrent, save_to, args.task_ids,
                        launcher_path)
        print(f"[tau2] arm={args.arm} iter={it}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT))
        if rc != 0:
            print(f"[tau2] tau2 exited rc={rc} — stopping arm", flush=True)
            break
        rows = _parse_results(save_to)
        _empty = sum(1 for r in rows if not r["response"]
                     or r["response"].startswith("reward="))
        if rows and _empty / len(rows) > 0.5:
            print(f"[tau2] WARNING: {_empty}/{len(rows)} sims have no agent "
                  "transcript — _transcript()/_parse_results() schema likely does "
                  "not match this tau2 version; B/C patch content is degraded to "
                  "one-line summaries (VERIFY marker)", flush=True)
        with open(trace, "a") as f:
            for r in rows:
                f.write(json.dumps({
                    "benchmark": "tau2", "group": GROUP_KEY[args.arm],
                    "phase": "test", "task_id": r["task_id"],
                    "task_desc": (r["instruction"] or r["task_name"])[:500],
                    "score": r["reward"], "em": 1.0 if r["reward"] >= 1.0 else 0.0,
                    "response": r["response"],
                    # termination_reason straight from tau2: env-level deaths
                    # (user-sim timeout, max-steps) must be visible per-arm so the
                    # paired analysis can confirm all arms died on the SAME tasks.
                    "failure_mode": r.get("failure_mode", ""),
                    "fb_mode": "env",  # tau2 feedback = final DB-state / action checks
                    "error": "",
                    "iteration": it, "iter_total": args.iters,
                    "method": "tau2_llm_agent", "code_rev": rev,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }) + "\n")
        print(f"[tau2] iter {it}: {len(rows)} sims, "
              f"mean reward {sum(r['reward'] for r in rows)/len(rows):.3f}", flush=True)
        if mem is not None:
            asyncio.run(_record_all(mem, rows))
            with open(state, "wb") as f:
                pickle.dump(mem, f)     # next iteration's agent injects this
            try:
                _n_entries = len(mem)
                _sz = state.stat().st_size
                print(f"[tau2] state saved: {_n_entries} entries, "
                      f"{_sz/1e3:.0f} KB -> {state.name}", flush=True)
                if it >= 1 and _n_entries == 0:
                    print("[tau2] WARNING: store EMPTY after recording "
                          f"iter {it} — memory arm is running blind", flush=True)
            except Exception:
                pass
    print(f"[tau2] done. trace: {trace}", flush=True)


if __name__ == "__main__":
    main()
