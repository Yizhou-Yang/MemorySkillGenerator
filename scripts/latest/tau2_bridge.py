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
import asyncio
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time

# tau2's reward comes from environment checks (final DB state, action match),
# not from the model grading itself — the one natively env-grounded benchmark.
# Declare that BEFORE evomem_bridge is imported, or its provenance defaults to
# whatever ITER_FEEDBACK happens to be in the shell.
os.environ.setdefault("ITER_FEEDBACK", "env")

try:
    from scripts.latest.atomic_io import atomic_pickle_dump as _atomic_pickle_dump
except Exception:  # direct-script invocation without the package on sys.path
    from atomic_io import atomic_pickle_dump as _atomic_pickle_dump
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Recorded on every trace row (mirrors latest_runner):
try:
    from scripts.latest.llm_client import critic_model_id as _cmid
    _CRITIC_ID = _cmid()
except Exception:
    _CRITIC_ID = (os.environ.get("CRITIC_MODEL")
                  or os.environ.get("CODEBUDDY_MODEL") or "?")
_C_META_ON = os.environ.get("C_META", "0") == "1"
_C_POLICY = (os.environ.get("C_POLICY")
             or ("meta" if _C_META_ON else "judgment")).strip().lower()
if _C_POLICY not in ("judgment", "meta", "guarded"):
    _C_POLICY = "judgment"

GROUP_KEY = {"A": "no_mem", "B": "raw_patch", "C": "curated_patch",  # canonical (arms.py)
             "mem0": "mem0",
             "amem": "amem"}   # external framework baselines (tab:external)


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
    if arm in ("mem0", "amem"):   # external framework baselines, same interface as B/C
        from scripts.latest.baseline_memories import make_external_memory
        return make_external_memory(arm, benchmark)
    from scripts.latest.evomem_bridge import BenchmarkMemory, CuratedMemory
    return BenchmarkMemory(benchmark, "B") if arm == "B" else CuratedMemory(benchmark)


def _opening_user(messages: list) -> str:
    """The scenario key material: the OPENING user turn alone.

    It used to join every user turn, which made the hash move with the dialogue.
    The agent keys inject() on the opener by itself, so record() wrote to one chain
    and inject() read another -- the lookup missed on every task and fell through to
    global retrieval, serving the curated arm other dialogues' fragments. Keep this
    identical to tau2_agent._user_texts's first return value.
    """
    parts = [m.get("content") for m in messages
             if isinstance(m, dict) and m.get("role") == "user"
             and isinstance(m.get("content"), str) and m.get("content").strip()]
    return parts[0][:2000] if parts else ""


def _opening_user_key(messages: list) -> str:
    """Alias kept explicit at the call site: this string is hashed into a chain id."""
    return _opening_user(messages)


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
    # tau2-bench 0.2.x appends ".json" to --save-to itself, so the file lands at
    # "<save_to>.json" (i.e. foo.json.json when we pass foo.json). Read whichever
    # exists.
    if not save_to.exists():
        _alt = save_to.with_name(save_to.name + ".json")
        if _alt.exists():
            save_to = _alt
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
        # tau2 grades by the final database state, so a sim that calls no tool
        # leaves the DB untouched and collects the same reward as one that did
        # the work. Inaction scores zero rather than inheriting that reward;
        # the count stays on the row so the substitution is auditable.
        _n_calls = sum(len(m.get("tool_calls") or []) for m in messages)
        if _n_calls == 0:
            reward = 0.0
        out.append({
            "task_id": f"{task_id}#{trial}" if trial != "" else task_id,
            "task_name": task_id,
            "instruction": instruction[:2000],
            "reward": float(reward),
            "response": _transcript(messages)
                        or f"reward={float(reward):.2f}; "
                           f"term={s.get('termination_reason') or 'n/a'}",
            "failure_mode": str(s.get("termination_reason") or "")[:200],
            # tau2 scores a run by the final database state, so a run that calls
            # no tool leaves the DB untouched and collects the same reward as one
            # that did the work correctly. The mean alone cannot tell those apart;
            # carry the count so an inert arm is visible.
            "tool_calls": _n_calls,
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
    # Each record may call the curation LLM. One dead streaming connection used
    # to stall the whole sweep here (2026-07-31: 45 min at 2s CPU between
    # iterations); a record now gets a hard wall-clock budget and a hung one is
    # skipped loudly -- it costs one memory entry, never the arm.
    _skipped = 0
    for r in rows:
        key = _task_key(r["instruction"]) if r["instruction"] else r["task_name"]
        # The chain is keyed on the STABLE task id, not on the instruction hash.
        # tau2 instructions are generated by the user simulator and differ every
        # run, so hashing them put 240 entries into 238 chains -- one chain per
        # entry. Everything that reasons per chain then goes inert: the repair
        # gate needs two attempts on one chain to compare and could evaluate
        # exactly 1 of 238, dead-chain silence never sees a chain history, and
        # version lineage has nothing to supersede. Retrieval was unaffected
        # because C_SEMANTIC_FALLBACK serves the similarity pool, which is why
        # this hid behind a working injection rate.
        chain = str(r.get("task_id") or key)
        task = {"task_id": key, "description": r["instruction"] or r["task_name"],
                "metadata": {"chain_id": chain}}
        try:
            await asyncio.wait_for(
                mem.record(task,
                           {"response": r["response"] or f"reward={r['reward']:.2f}"},
                           score=r["reward"]),
                timeout=float(os.environ.get("TAU2_RECORD_TIMEOUT_S", "240")))
        except asyncio.TimeoutError:
            _skipped += 1
            print(f"[tau2] record TIMED OUT for {key!r} -- skipped "
                  f"({_skipped} so far); the arm continues", flush=True)
    if _skipped:
        print(f"[tau2] WARNING: {_skipped}/{len(rows)} records skipped on "
              f"wall-clock timeout; memory is that much thinner this iteration",
              flush=True)


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
           "--save-to", str(save_to)]
    # --auto-resume only exists on the pinned internal tau2 build; the PyPI
    # tau2-bench (0.2.x) CLI rejects it with rc=2 before running anything.
    # Opt in via TAU2_AUTO_RESUME=1 only where the flag is known to exist —
    # and remember resume poisons reruns unless artifacts are purged first.
    if os.environ.get("TAU2_AUTO_RESUME", "0") == "1":
        cmd.insert(-2, "--auto-resume")
    if n_tasks:
        cmd += ["--num-tasks", str(n_tasks)]
    for t in (task_ids or []):
        cmd += ["--task-id", t]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C", "mem0", "amem"], required=True)
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
    # Support comma-separated domains (e.g. airline,retail); tau2 run only
    # accepts a single --domain, so loop across them sequentially.
    _arm_failed = False
    domains = [d.strip() for d in args.domain.split(",") if d.strip()]
    for domain in domains:
        for it in range(args.iters):
            save_to = out_root / f"tau2_{domain}_{args.arm}_iter{it}.json"
            # Purge stale artifacts BEFORE launching: tau2-bench 0.2.x prompts
            # interactively ("overwrite? [y/N]") when its save file exists, and
            # under nohup stdin is closed -> EOFError kills the whole arm. It
            # writes to <save_to>.json (double suffix), so clear both spellings.
            for _stale in (save_to, save_to.with_name(save_to.name + ".json")):
                try:
                    if _stale.exists() and _stale.is_file():
                        _stale.unlink()
                except Exception:
                    pass
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
            cmd = _tau2_cmd(tau2_bin, args.arm, args.model, domain, args.n_tasks,
                            args.num_trials, args.n_concurrent, save_to, args.task_ids,
                            launcher_path)
            # External-framework stores live on disk behind a client that file-locks
            # its path (local qdrant): release ours so the tau2 subprocess — which
            # unpickles the memory and opens the same path for inject() — can take it.
            if mem is not None and hasattr(mem, "release"):
                mem.release()
            print(f"[tau2] arm={args.arm} iter={it}: {' '.join(cmd)}", flush=True)
            rc = subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT))
            if rc != 0:
                # Stopping the arm is right, but the process used to exit 0
                # afterwards, so the launcher recorded a crashed arm as a finished
                # one: arm A died on iteration 0 and the wrapper logged rc=0.
                print(f"[tau2] tau2 exited rc={rc} — stopping arm", flush=True)
                _arm_failed = True
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
                        "tool_calls": r.get("tool_calls", 0),
                        "fb_mode": "env",  # tau2 feedback = final DB-state / action checks
                        "error": "",
                        "iteration": it, "iter_total": args.iters,
                        "method": "tau2_llm_agent", "code_rev": rev,
                        # Same policy/provenance fields the QA runner logs, so a
                        # tau2 C arm is classifiable (and gate-able) too: which
                        # model judged the entries, which curation policy ran,
                        # and that the score is env-grounded rather than
                        # self-assessed.
                        "critic_model": _CRITIC_ID, "c_meta": _C_META_ON,
                        "c_policy": _C_POLICY,
                        "score_provenance": "env",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }) + "\n")
            print(f"[tau2] iter {it}: {len(rows)} sims, "
                  f"mean reward {sum(r['reward'] for r in rows)/len(rows):.3f}", flush=True)
            if mem is not None:
                asyncio.run(_record_all(mem, rows))
                # Atomic: a reclaim mid-write would otherwise truncate the store
                # to nothing and the next iteration would start from an empty
                # memory while still reporting as arm B/C (see atomic_io).
                _atomic_pickle_dump(mem, state)   # next iteration's agent injects this
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
    if _arm_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
