#!/usr/bin/env python3
"""Selective re-run: re-run specific benchmark+group combinations,
preserving existing traces for groups not being re-run.

Usage examples:
  # Re-run all groups for GAIA (full re-run)
  python scripts/latest/selective_rerun.py --benchmarks gaia

  # Re-run only C group for GAIA2
  python scripts/latest/selective_rerun.py --benchmarks gaia2 --groups C

  # Re-run B and C groups for LoCoMo
  python scripts/latest/selective_rerun.py --benchmarks locomo --groups B,C

  # Re-run all groups for multiple benchmarks
  python scripts/latest/selective_rerun.py --benchmarks gaia,gaia2,terminal_bench_2 --groups all

Background execution:
  nohup /root/.conda/envs/skillforge/bin/python scripts/latest/selective_rerun.py \
    --benchmarks gaia,gaia2,locomo,terminal_bench_2 \
  --group-map "gaia:all;gaia2:C;locomo:B,C;terminal_bench_2:all" \
    > experiments_results/latest/selective_rerun_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""
import asyncio, json, os, sys, time, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'deepseek-v4-pro'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')

MODEL = "deepseek-v4-pro"
CONCURRENCY = 5
RESULTS_DIR = "experiments_results/latest"

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
from scripts.latest.eval import compute_partial_results_from_trace
from benchmarks.loader import BenchmarkLoader

TASK_LIMITS = {
    "gaia": 30, "gaia2": 30, "terminal_bench_2": 30, "locomo": 30,
}

# Dispatch tables (mirror latest_runner.py)
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


def parse_groups(arg_str: str, benchmark: str) -> set[str]:
    """Parse group specification. 'all' means A,B,C."""
    if arg_str.strip().lower() == "all":
        return {"A", "B", "C"}
    return {g.strip() for g in arg_str.split(",") if g.strip()}


async def run_selected_groups(
    benchmark: str,
    tasks: list[dict],
    groups_to_run: set[str],
) -> None:
    """Run only the specified groups for a benchmark.
    Preserves existing traces for groups not being re-run.
    """
    trace_path = Path(RESULTS_DIR) / benchmark / "trace.jsonl"

    # Read and preserve existing traces for groups NOT being re-run
    preserved_traces: list[dict] = []
    if trace_path.exists():
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                group = entry.get("group", "")
                # Map group_key to group letter for comparison
                group_letter = group[0] if group else ""
                if group_letter not in groups_to_run:
                    preserved_traces.append(entry)

        print(f"  Preserved {len(preserved_traces)} traces (groups not being re-run)")

    # Clear trace and rewrite preserved traces
    if trace_path.exists():
        trace_path.unlink()
    _trace.clear_benchmark(benchmark)

    # Rewrite preserved traces
    if preserved_traces:
        with open(trace_path, "w") as f:
            for entry in preserved_traces:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*70}")
    print(f"  Selective Re-run: {benchmark} (model: {MODEL})")
    print(f"  Groups to run: {sorted(groups_to_run)}")
    print(f"  Preserved: {len(preserved_traces)} traces")
    print(f"{'='*70}")

    os.makedirs(f"{RESULTS_DIR}/{benchmark}", exist_ok=True)

    run_fn_a = BASELINE_RUNNER.get(benchmark)
    run_fn_controlled = CONTROLLED_RUNNER.get(benchmark)

    if not run_fn_a:
        print(f"  ERROR: No runner for benchmark '{benchmark}'")
        return

    # Determine QA vs agentic
    is_qa = benchmark in ("locomo",)
    is_gaia = benchmark == "gaia"
    is_gaia2 = benchmark == "gaia2"

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _run_group(label: str, group_key: str, tasks: list[dict], build_coro):
        total = len(tasks)
        results = [None] * total
        evals = [None] * total

        async def _wrap(i: int, task: dict):
            async with sem:
                r = await build_coro(task)
            ev = await evaluate_task(r, benchmark)
            expected = task.get("expected", r.get("expected", ""))
            aug = r.get("_aug_prompt", "")
            _trace.log(
                benchmark=benchmark, group=group_key, phase="test",
                task_id=r.get("task_id", task.get("task_id", "")),
                task_desc=task.get("description", ""),
                augmented_prompt=aug,
                response=r.get("response", ""), expected=expected,
                score=ev.get("score", 0.0),
            )
            tag = r.get("task_id", str(i))
            err = r.get("error")
            status = "\u2717" if err else "\u2713"
            msg = f"    [{label}] {i+1}/{total} {status} {tag} ({r.get('time_cost',0):.0f}s) EM={ev.get('em',0):.0%}"
            if err:
                msg += f" ERR: {str(err)[:80]}"
            print(msg, flush=True)
            return i, r, ev

        for coro in asyncio.as_completed([_wrap(i, t) for i, t in enumerate(tasks)]):
            i, r, ev = await coro
            results[i] = r
            evals[i] = ev
        return results, evals

    def _group_key(letter: str) -> str:
        if letter == "A":
            return "A_baseline"
        elif letter == "B":
            return "B_evoarena"
        else:
            return "C_skillforge"

    # Run each requested group
    for group_letter in ["A", "B", "C"]:
        if group_letter not in groups_to_run:
            continue

        gk = _group_key(group_letter)

        if group_letter == "A":
            print(f"    [A] Baseline (no augmentation)...", flush=True)
            if is_gaia:
                await _run_group("A", gk, tasks,
                    lambda t: run_fn_controlled(t, "", "A", within_task_patch_mode=None))
            else:
                await _run_group("A", gk, tasks,
                    lambda t: run_fn_a(t, "", "A"))

        elif group_letter == "B":
            if is_qa:
                print(f"    [B] Self-Consistency (majority vote, 3 samples)...", flush=True)
                await _run_group("B", gk, tasks,
                    lambda t: run_fn_controlled(t, "", "B", within_task_patch_mode="evoarena"))
            elif is_gaia:
                print(f"    [B] EvoArena EvoMem (within-task patch memory)...", flush=True)
                await _run_group("B", gk, tasks,
                    lambda t: run_fn_controlled(t, "", "B", within_task_patch_mode="evoarena"))
            elif is_gaia2:
                print(f"    [B] EvoArena Self-Correction Protocol...", flush=True)
                await _run_group("B", gk, tasks,
                    lambda t: run_fn_a(t, GAIA2_EVOARENA_AUG, "B"))
            else:
                print(f"    [B] Baseline (repeat)...", flush=True)
                await _run_group("B", gk, tasks,
                    lambda t: run_fn_a(t, "", "B"))

        elif group_letter == "C":
            if is_qa:
                print(f"    [C] Evidence-Weighted Self-Consistency...", flush=True)
                await _run_group("C", gk, tasks,
                    lambda t: run_fn_controlled(t, "", "C", within_task_patch_mode="skillforge"))
            elif is_gaia:
                print(f"    [C] SkillForge (B + failure-aware patch routing)...", flush=True)
                await _run_group("C", gk, tasks,
                    lambda t: run_fn_controlled(t, "", "C", within_task_patch_mode="skillforge"))
            elif is_gaia2:
                print(f"    [C] SkillForge Plan-First Architecture...", flush=True)
                await _run_group("C", gk, tasks,
                    lambda t: run_fn_a(t, GAIA2_SKILLFORGE_AUG, "C"))
            else:
                print(f"    [C] Baseline (repeat)...", flush=True)
                await _run_group("C", gk, tasks,
                    lambda t: run_fn_a(t, "", "C"))

    # Compute and print report from ALL traces (preserved + new)
    report = compute_partial_results_from_trace(benchmark, RESULTS_DIR)
    print(f"\n  Results ({benchmark}, model={MODEL}):")
    if isinstance(report, dict) and "scores" in report:
        scores = report["scores"]
        for gk in ["A_baseline", "B_evoarena", "C_skillforge"]:
            if gk in scores:
                val = scores[gk]
                if isinstance(val, (int, float)):
                    em_pct = val * 100.0
                    print(f"    {gk.split('_')[0]} ({gk}): EM={em_pct:.1f}%")
                else:
                    print(f"    {gk.split('_')[0]} ({gk}): {val}")
    elif isinstance(report, dict) and "error" in report:
        print(f"    ERROR: {report['error']}")
    else:
        print(f"    Report: {report}")

    # Save report
    with open(f"{RESULTS_DIR}/{benchmark}/report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


async def main():
    parser = argparse.ArgumentParser(description="Selective benchmark re-run")
    parser.add_argument(
        "--benchmarks", type=str, required=True,
        help="Comma-separated benchmark names (gaia,gaia2,locomo,terminal_bench_2)",
    )
    parser.add_argument(
        "--group-map", type=str, required=True,
        help="Groups per benchmark, semicolon-separated. Format: 'gaia:all;gaia2:C;locomo:B,C'",
    )
    args = parser.parse_args()

    bench_names = [b.strip() for b in args.benchmarks.split(",")]

    # Parse group-map (semicolon-separated items, comma-separated groups within
    # each item, e.g. "gaia:all;gaia2:C;locomo:B,C;terminal_bench_2:all")
    group_map: dict[str, set[str]] = {}
    for item in args.group_map.split(";"):
        item = item.strip()
        if ":" not in item:
            continue
        bench, groups = item.split(":", 1)
        bench = bench.strip()
        group_map[bench] = parse_groups(groups, bench)

    print("=" * 70)
    print("  SkillForge Selective Re-run")
    print(f"  Model: {MODEL}")
    print(f"  Benchmarks: {bench_names}")
    for bm in bench_names:
        groups = group_map.get(bm, {"A", "B", "C"})
        print(f"    {bm}: re-run groups {sorted(groups)}")
    print("=" * 70)

    # Probe API
    print("\n  Probing API availability...", flush=True)
    if not await probe_api_available():
        print("  API is NOT available. Aborting.", flush=True)
        return
    print("  API is responding.", flush=True)

    # Load tasks for each benchmark
    all_tasks: dict[str, list[dict]] = {}
    for bn in bench_names:
        config = {"name": bn, "num_samples": TASK_LIMITS[bn]}
        if bn == "gaia2":
            config["scenario_dir"] = "/tmp/harbor-datasets/datasets/gaia2-cli"
        loader = BenchmarkLoader(config)
        tasks = loader.load()[:TASK_LIMITS[bn]]
        all_tasks[bn] = tasks
        print(f"  Loaded {bn}: {len(tasks)} tasks")

    # Run benchmarks serially
    for bn in bench_names:
        groups = group_map.get(bn, {"A", "B", "C"})
        print(f"\n{'#'*70}")
        print(f"  Starting: {bn} (groups: {sorted(groups)})")
        print(f"{'#'*70}")
        try:
            await run_selected_groups(bn, all_tasks[bn], groups)
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {bn}: {e}")
            traceback.print_exc()

    # Print final summary
    print(f"\n\n{'='*70}")
    print(f"  SELECTIVE RE-RUN COMPLETE")
    print(f"{'='*70}")
    for bn in bench_names:
        report = compute_partial_results_from_trace(bn, RESULTS_DIR)
        if isinstance(report, dict) and "scores" in report:
            scores = report["scores"]
            parts = []
            for gk in ["A_baseline", "B_evoarena", "C_skillforge"]:
                if gk in scores:
                    val = scores[gk]
                    if isinstance(val, (int, float)):
                        parts.append(f"{gk[0]}={val*100:.1f}%")
            print(f"  {bn}: {', '.join(parts)}")
        else:
            print(f"  {bn}: {report}")


if __name__ == "__main__":
    asyncio.run(main())
