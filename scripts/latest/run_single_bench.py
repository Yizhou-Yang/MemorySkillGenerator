#!/usr/bin/env python3
import os
"""Single-benchmark robust runner with per-task timeouts and crash resilience.

Usage:
  # GAIA all groups (A+B+C):
  nohup python scripts/latest/run_single_bench.py --benchmark gaia --groups all \
    > experiments_results/latest/run_gaia.log 2>&1 &

  # GAIA2 C group only:
  nohup python scripts/latest/run_single_bench.py --benchmark gaia2 --groups C \
    > experiments_results/latest/run_gaia2_c.log 2>&1 &

  # LoCoMo B+C groups only:
  nohup python scripts/latest/run_single_bench.py --benchmark locomo --groups B,C \
    > experiments_results/latest/run_locomo_bc.log 2>&1 &

  # Terminal-Bench-2 all groups (Docker-based, slow):
  nohup python scripts/latest/run_single_bench.py --benchmark terminal_bench_2 --groups all \
    > experiments_results/latest/run_tb2.log 2>&1 &
"""
import asyncio, json, os, sys, time, argparse, signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'hy3-preview-ioa'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')

MODEL = "hy3-preview-ioa"
CONCURRENCY = 5
RESULTS_DIR = "experiments_results/latest"
PER_TASK_TIMEOUT = 1800  # 30 min max per task

# Import runner infrastructure
from scripts.latest.gaia_runner import run_gaia_task, run_gaia_task_controlled
from scripts.latest.gaia2_runner import run_gaia2_task_with_are
from scripts.latest.locomo_runner import run_locomo_task, run_locomo_task_controlled
from scripts.latest.terminal_bench_2_runner import (
    run_terminal_bench_2_task, run_terminal_bench_2_task_controlled,
)
from scripts.latest.latest_runner import (
    evaluate_task,
    GAIA2_EVOARENA_AUG, GAIA2_SKILLFORGE_AUG,
)
from scripts.latest.trace import TraceLogger
from scripts.latest.llm_client import probe_api_available
from benchmarks.loader import BenchmarkLoader

# Scaled up from 30 → 100 so A/B/C deltas can reach significance (at n=30 every
# delta was n.s.). The loader caps at the number of available tasks, so a value
# above a benchmark's pool (e.g. gaia2 has ~50) simply loads all of them.
# Override per run with TASK_LIMIT=<n>.
_TASK_N = int(os.environ.get("TASK_LIMIT", "100"))
TASK_LIMITS = {
    "gaia": _TASK_N, "gaia2": _TASK_N, "terminal_bench_2": _TASK_N, "locomo": _TASK_N,
}

# Transient failures worth retrying (API blips, rate limits). Empty responses
# with no error are also retried (usually a blip that returned nothing).
_TRANSIENT = ("429", "rate_limit", "rate-limit", "timeout", "quota",
              "unavailable", "503", "502", "overloaded")
_MAX_RETRIES = int(os.environ.get("TASK_MAX_RETRIES", "3"))

BASELINE_RUNNER = {
    "gaia": run_gaia_task,
    "gaia2": run_gaia2_task_with_are,
    "terminal_bench_2": run_terminal_bench_2_task,
    "locomo": run_locomo_task,
}
CONTROLLED_RUNNER = {
    "gaia": run_gaia_task_controlled,
    "gaia2": run_gaia2_task_with_are,
    "terminal_bench_2": run_terminal_bench_2_task_controlled,
    "locomo": run_locomo_task_controlled,
}

_trace = TraceLogger(RESULTS_DIR)


def load_existing_traces(benchmark: str) -> list[dict]:
    """Load existing trace entries for a benchmark."""
    trace_path = Path(RESULTS_DIR) / benchmark / "trace.jsonl"
    if not trace_path.exists():
        return []
    traces = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return traces


def save_traces(benchmark: str, traces: list[dict]) -> None:
    """Atomically write all traces."""
    trace_path = Path(RESULTS_DIR) / benchmark / "trace.jsonl"
    os.makedirs(trace_path.parent, exist_ok=True)
    tmp_path = trace_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as f:
        for entry in traces:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp_path, trace_path)


async def run_task_with_timeout(coro, timeout: int = PER_TASK_TIMEOUT,
                                task_id: str = "timeout"):
    """Run a task coroutine with a hard timeout to prevent deadlocks.
    Preserves the real task_id so a timed-out task stays identifiable in the
    trace (and resumable) instead of collapsing to a literal "timeout" id."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return {
            "task_id": task_id, "expected": "", "response": "",
            "error": f"task_timeout_{timeout}s", "time_cost": timeout,
            "_aug_prompt": "", "execution_mode": "timeout"
        }


async def run_single_benchmark(
    benchmark: str,
    tasks: list[dict],
    groups_to_run: set[str],
) -> dict:
    """Run only the specified groups for a benchmark.
    Preserves existing traces for groups NOT being re-run.
    """
    trace_path = Path(RESULTS_DIR) / benchmark / "trace.jsonl"
    os.makedirs(trace_path.parent, exist_ok=True)

    # Load and preserve traces for groups NOT being re-run
    all_traces = load_existing_traces(benchmark)
    preserved = []
    for t in all_traces:
        g = t.get("group", "")
        group_letter = g[0] if g else ""
        if group_letter not in groups_to_run:
            preserved.append(t)

    # Start fresh with preserved traces only
    all_traces = list(preserved)
    save_traces(benchmark, all_traces)
    _trace.clear_benchmark(benchmark)

    print(f"Preserved {len(preserved)} existing traces", flush=True)
    print(f"Groups to run: {sorted(groups_to_run)}", flush=True)
    print(f"Tasks: {len(tasks)}, Concurrency: {CONCURRENCY}", flush=True)

    run_fn_a = BASELINE_RUNNER.get(benchmark)
    run_fn_controlled = CONTROLLED_RUNNER.get(benchmark)

    if not run_fn_a:
        print(f"ERROR: No runner for benchmark '{benchmark}'")
        return {"error": f"no_runner_{benchmark}"}

    is_qa = benchmark in ("locomo",)
    is_gaia = benchmark == "gaia"
    is_gaia2 = benchmark == "gaia2"
    is_tb2 = benchmark == "terminal_bench_2"

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _worker(label: str, group_key: str, build_coro):
        """Run all tasks for one group concurrently (up to CONCURRENCY at a time).
        Uses asyncio.gather with semaphore — avoids asyncio.as_completed deadlocks.
        """
        total = len(tasks)
        evals = [None] * total

        async def _run_one(i: int, task: dict):
            async with sem:
                t_start = time.time()
                # Retry on transient errors AND on empty-response-no-error
                # (an API blip that returned nothing). build_coro(task) is
                # re-invoked each attempt to get a fresh coroutine.
                r = None
                for attempt in range(_MAX_RETRIES + 1):
                    try:
                        r = await run_task_with_timeout(
                            build_coro(task), task_id=task.get("task_id", str(i)))
                    except Exception as e:
                        r = {
                            "task_id": task.get("task_id", str(i)),
                            "response": "", "error": f"crash:{e}",
                            "time_cost": time.time() - t_start,
                            "_aug_prompt": "",
                        }
                    err = str(r.get("error") or "")
                    resp = (r.get("response") or "").strip()
                    transient = any(k in err for k in _TRANSIENT)
                    soft_empty = (not resp) and (not err)
                    if (resp and not transient) or (err and not transient):
                        break
                    if attempt < _MAX_RETRIES and (transient or soft_empty):
                        await asyncio.sleep(8 * (2 ** attempt))
                        continue
                    break

                # Evaluate
                try:
                    ev = await evaluate_task(r, benchmark)
                except Exception as e:
                    ev = {"score": 0.0, "em": 0.0, "method": "eval_error", "error": str(e)[:200]}

                evals[i] = ev

                # Build and append single trace entry atomically
                entry = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "benchmark": benchmark,
                    "group": group_key,
                    "phase": "test",
                    "task_id": r.get("task_id", task.get("task_id", "")),
                    "task_desc": (task.get("description", "") or "")[:500],
                    "augmented_prompt": r.get("_aug_prompt", ""),
                    "response": (r.get("response", "") or "")[:5000],
                    "expected": task.get("expected", r.get("expected", "")),
                    "score": ev.get("score", 0.0),
                    "test_passed": r.get("test_passed", False),
                    "test_output": (r.get("test_output", "") or "")[:2000],
                    "execution_mode": r.get("execution_mode", "?"),
                }

                # Atomic trace append
                trace_path_t = Path(RESULTS_DIR) / benchmark / "trace.jsonl"
                with open(trace_path_t, "a") as tf:
                    tf.write(json.dumps(entry, ensure_ascii=False) + "\n")

                elapsed = time.time() - t_start
                err = r.get("error")
                status = "\u2717" if err else "\u2713"
                tag = r.get("task_id", str(i))
                msg = f"  [{label}] {i+1}/{total} {status} {tag} ({elapsed:.0f}s) EM={ev.get('em',0):.0%}"
                if err:
                    msg += f" ERR: {str(err)[:80]}"
                if is_tb2:
                    msg += f" test_passed={r.get('test_passed', False)}"
                print(msg, flush=True)

        # Run all tasks concurrently
        await asyncio.gather(*[_run_one(i, t) for i, t in enumerate(tasks)])

        return evals

    def _group_key(letter: str) -> str:
        return {"A": "A_baseline", "B": "B_evoarena", "C": "C_skillforge"}[letter]

    all_evals: dict[str, list] = {}

    for letter in sorted(groups_to_run):
        gk = _group_key(letter)
        print(f"\n--- Group {letter} ({gk}) ---", flush=True)

        if letter == "A":
            label = "A"
            if is_gaia:
                evals = await _worker(label, gk,
                    lambda t: run_fn_controlled(t, "", "A", within_task_patch_mode=None))
            else:
                evals = await _worker(label, gk,
                    lambda t: run_fn_a(t, "", "A"))
        elif letter == "B":
            label = "B"
            if is_qa:
                evals = await _worker(label, gk,
                    lambda t: run_fn_controlled(t, "", "B", within_task_patch_mode="evoarena"))
            elif is_gaia:
                evals = await _worker(label, gk,
                    lambda t: run_fn_controlled(t, "", "B", within_task_patch_mode="evoarena"))
            elif is_gaia2:
                evals = await _worker(label, gk,
                    lambda t: run_fn_a(t, GAIA2_EVOARENA_AUG, "B"))
            else:
                evals = await _worker(label, gk,
                    lambda t: run_fn_a(t, "", "B"))
        elif letter == "C":
            label = "C"
            if is_qa:
                evals = await _worker(label, gk,
                    lambda t: run_fn_controlled(t, "", "C", within_task_patch_mode="skillforge"))
            elif is_gaia:
                evals = await _worker(label, gk,
                    lambda t: run_fn_controlled(t, "", "C", within_task_patch_mode="skillforge"))
            elif is_gaia2:
                evals = await _worker(label, gk,
                    lambda t: run_fn_a(t, GAIA2_SKILLFORGE_AUG, "C"))
            else:
                evals = await _worker(label, gk,
                    lambda t: run_fn_a(t, "", "C"))

        all_evals[gk] = evals

    # Compute final report
    print(f"\n{'='*50}")
    print(f"  Results: {benchmark}")
    print(f"{'='*50}")
    for gk in ["A_baseline", "B_evoarena", "C_skillforge"]:
        if gk in all_evals:
            evals = all_evals[gk]
            scores = [e.get("score", 0) for e in evals if e]
            n = len(scores)
            avg = sum(scores) / max(n, 1) if n > 0 else 0.0
            print(f"  {gk[0]}: {n} tasks, avg_score={avg:.1%}")

    # Also compute from trace for groups not re-run
    all_traces = load_existing_traces(benchmark)
    for gk in ["A_baseline", "B_evoarena", "C_skillforge"]:
        if gk not in all_evals:
            scores = [t.get("score", 0) for t in all_traces if t.get("group") == gk]
            n = len(scores)
            if n > 0:
                avg = sum(scores) / n
                print(f"  {gk[0]} (preserved): {n} tasks, avg_score={avg:.1%}")

    # Save report
    report_data = {
        "benchmark": benchmark,
        "model": MODEL,
        "groups_rerun": sorted(groups_to_run),
        "scores": {},
        "trace_count": len(all_traces),
    }
    for gk in ["A_baseline", "B_evoarena", "C_skillforge"]:
        relevant = [t.get("score", 0) for t in all_traces if t.get("group") == gk]
        if relevant:
            report_data["scores"][gk] = sum(relevant) / len(relevant)

    with open(f"{RESULTS_DIR}/{benchmark}/report.json", "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return report_data


async def main():
    parser = argparse.ArgumentParser(description="Run a single benchmark robustly")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--groups", type=str, required=True,
                        help="Groups to run: 'all' or 'A,B,C'")
    args = parser.parse_args()

    benchmark = args.benchmark
    if args.groups.strip().lower() == "all":
        groups_to_run = {"A", "B", "C"}
    else:
        groups_to_run = {g.strip() for g in args.groups.split(",") if g.strip()}

    print(f"=== SkillForge: {benchmark} (groups: {sorted(groups_to_run)}) ===")

    # Probe API
    print("Probing API...", flush=True)
    if not await probe_api_available():
        print("API unavailable. Aborting.")
        return
    print("API responding.", flush=True)

    # Load tasks
    config = {"name": benchmark, "num_samples": TASK_LIMITS[benchmark]}
    if benchmark == "gaia2":
        config["scenario_dir"] = os.environ.get(
            "GAIA2_SCENARIO_DIR",
            "/tmp/harbor-datasets/datasets/gaia2-cli",
        )
    loader = BenchmarkLoader(config)
    tasks = loader.load()[:TASK_LIMITS[benchmark]]
    print(f"Loaded {len(tasks)} tasks", flush=True)

    # Run
    await run_single_benchmark(benchmark, tasks, groups_to_run)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
