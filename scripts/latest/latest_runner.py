#!/usr/bin/env python3
"""SkillForge Latest — Latest Experiment Runner"""
import asyncio
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'deepseek-v4-pro'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')

from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage, ToolUseBlock

from latest import (SkillForgeLatest, ExperienceLibrary, Experience,
                build_augmented_prompt, ai_review_experience,
                cross_agent_evaluate_skill)
from benchmarks.loader import BenchmarkLoader
from latest.eval.gaia2_judge import evaluate_gaia2 as _gaia2_official_judge

# ─── Extracted module imports ─────────────────────────────────────────────
from scripts.latest.trace import TraceLogger, APIUnavailableError
from scripts.latest.llm_client import (
    probe_api_available, _check_api_error,
    save_checkpoint, load_checkpoint, clear_checkpoint,
    llm_review_fn,
    _query_sync, _llm_call, _query_notool_sync, _llm_call_notool,
    _llm_short_call, llm_extract_answer, llm_judge_answer,
    llm_critic_skill_quality,
)
from scripts.latest.eval import (
    normalize_answer, exact_match, evaluate_task,
    compute_partial_results_from_trace,
)

MODEL = "deepseek-v4-pro"
CONCURRENCY = 15
TASK_TIMEOUT_QA = 180
TASK_TIMEOUT_AGENT = 300
QUALITY_THRESHOLD = 5

RESULTS_DIR = str(PROJECT_ROOT / "experiments_results" / "latest")

# ─── Trace Logger (for human review of prompts & responses) ───────────────
TASK_LIMITS = {"gaia": 165, "gaia2": 50, "swebench_dynamic": 30}

CHECKPOINT_FILE = str(PROJECT_ROOT / "experiments_results" / "latest" / "_checkpoint.json")

# ─── Trace logger instance ────────────────────────────────────────────────
_trace = TraceLogger(RESULTS_DIR)

# ─── GAIA2 ARE runner (real tool calling) ─────────────────────────────────

# ─── GAIA2 ARE runner (real tool calling) ─────────────────────────────────
from scripts.latest.gaeia_runner import run_gaia_task, run_gaia_task_controlled
from scripts.latest.gaeia2_runner import run_gaia2_task_with_are

# ─── Evaluation ───────────────────────────────────────────────────────────

async def evaluate_task(result: dict, benchmark: str, use_llm_judge: bool = True) -> dict:
    """Primary metric:
       - gaia2: soft recall (action sequence matching)
       - swebench_dynamic: pass@1 (patch correctness via LLM judge)
       - gaia: Exact Match
    """
    if benchmark == "gaia2":
        # Official GAIA2 CLI judge logic: count gate + LLM action matching +
        # config-aware dual mode + alias normalization.
        oracle_events = result.get("expected", [])
        event_log = result.get("event_log", [])
        response = (result.get("response") or "").strip()
        oracle_answer = result.get("oracle_answer", "")
        task_desc = result.get("description", "")
        config = (result.get("metadata") or {}).get("config", "execution")

        if not oracle_events and not oracle_answer:
            return {"score": 0.0, "em": 0.0, "method": "gaia2_no_oracle"}
        if not event_log and not response:
            return {"score": 0.0, "em": 0.0, "method": "gaia2_no_actions"}

        # Build the LLM call function for the judge (system_prompt, user_prompt) -> str
        async def _judge_llm_call(system_prompt: str, user_prompt: str) -> str:
            """Adapter: call our LLM infrastructure with system+user prompt."""
            try:
                r = await _llm_call_notool(system_prompt, user_prompt, timeout=60)
                return (r.get("text") or "").strip()
            except Exception as e:
                print(f"[GAIA2 judge] LLM call failed: {e}")
                return ""

        try:
            judge_result = await _gaia2_official_judge(
                _judge_llm_call,
                config=config,
                task=task_desc,
                oracle_events=oracle_events,
                oracle_answer=oracle_answer,
                event_log=event_log,
                agent_response=response,
            )
            return judge_result
        except Exception as e:
            print(f"[GAIA2 judge] Official judge failed: {e}")
            return {"score": 0.0, "em": 0.0, "method": "gaia2_judge_error",
                    "error": str(e)[:200]}

    if benchmark == "swebench_dynamic":
        # SWE-bench: use LLM judge to assess if the response addresses the issue
        response = (result.get("response") or "").strip()
        raw_expected = result.get("expected", "")
        expected = str(raw_expected).strip() if not isinstance(raw_expected, list) else ", ".join(raw_expected)
        if not response:
            return {"score": 0.0, "em": 0.0, "method": "swebench_empty"}
        # Check if response contains code changes (patch, code block, or file edits)
        has_code = ("diff" in response or "---" in response or "+++" in response
                    or "patch" in response.lower() or "```" in response
                    or "def " in response or "class " in response
                    or "import " in response or "fix" in response.lower())
        if not has_code:
            return {"score": 0.0, "em": 0.0, "method": "swebench_no_code"}
        # Use LLM judge for quality assessment
        if use_llm_judge:
            judge_prompt = (
                f"Evaluate if this response correctly addresses the software issue.\n\n"
                f"Issue description: {expected}\n\n"
                f"Agent response (code changes): {response}\n\n"
                f"Score 0.0 to 1.0: Does the response identify the correct file/function "
                f"and propose a logically sound fix? "
                f"0.0=completely wrong, 0.3=identifies area but wrong fix, "
                f"0.5=partial fix, 0.7=mostly correct, 1.0=fully correct.\n"
                f"Output ONLY a number:"
            )
            out = await _llm_short_call(judge_prompt, max_turns=1, timeout=30)
            m = re.search(r'(\d+\.?\d*)', out)
            score = 0.0
            if m:
                try:
                    score = min(1.0, max(0.0, float(m.group(1))))
                except ValueError:
                    score = 0.0
            return {"score": score, "em": 1.0 if score >= 0.7 else 0.0,
                    "llm_judge": score, "method": "swebench_llm_judge"}
        return {"score": 0.5, "em": 0.0, "method": "swebench_has_code"}

    expected = (result.get("expected") or "").strip()
    response = (result.get("response") or "").strip()
    if not expected or not response:
        return {"score": 0.0, "em": 0.0, "method": "empty"}

    extracted = await llm_extract_answer(response, result.get("task_id", ""))
    em = exact_match(extracted or response, expected)

    llm_score = 0.0
    if use_llm_judge and em < 1.0:
        llm_score = await llm_judge_answer(extracted or response, expected, result.get("task_id", ""))

    return {
        "score": em if em > 0 else (llm_score if llm_score >= 0.8 else 0.0),
        "em": em,
        "llm_judge": llm_score,
        "extracted_answer": (extracted or "")[:200],
        "method": "exact_match",
    }

# ??? Benchmark runner ?????????????????????????????????????????????????????

async def run_benchmark(benchmark: str, tasks: list) -> dict:
    """Run A/B/C testing on ALL tasks (no train/test split)."""
    print(f"\n{'='*70}")
    print(f"  Benchmark: {benchmark} (model: {MODEL})")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Metric: Exact Match")
    print(f"{'='*70}")

    os.makedirs(f"{RESULTS_DIR}/{benchmark}", exist_ok=True)

    # Clear stale trace file
    trace_path = Path(RESULTS_DIR) / benchmark / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
        print(f"  ??  Cleared stale trace: {trace_path}")
    if benchmark in _trace._files:
        try:
            _trace._files[benchmark].close()
        except Exception:
            pass
        del _trace._files[benchmark]

    # All tasks go through A/B/C testing directly
    test_tasks = tasks
    concurrency = 3 if benchmark == "gaia2" else CONCURRENCY
    sem = asyncio.Semaphore(concurrency)

    print(f"\n  Testing {len(test_tasks)} tasks ? 3 groups (A/B/C)...")
    print(f"    [A] Baseline (no augmentation)...", flush=True)
    async def run_test_a(i, task):
        async with sem:
            if benchmark == "gaia2":
                return await run_gaia2_task_with_are(task, "", "A")
            if benchmark in ("gaia", "swebench_dynamic"):
                return await run_gaia_task(task, "", "A")
            return {"task_id": task["task_id"], "error": "unknown_benchmark", "score": 0.0}
    results_a = await asyncio.gather(*[run_test_a(i, t) for i, t in enumerate(test_tasks)])

    print(f"    [B] EvoArena EvoMem (within-task self-correction injection)...", flush=True)
    async def run_test_b(i, task):
        async with sem:
            if benchmark == "gaia2":
                return await run_gaia2_task_with_are(task, "", "B",
                                                      within_task_patch_mode="evoarena")
            if benchmark in ("gaia", "swebench_dynamic"):
                return await run_gaia_task_controlled(task, "", "B",
                                                       within_task_patch_mode="evoarena")
            return {"task_id": task["task_id"], "error": "unknown_benchmark", "score": 0.0}
    results_b = await asyncio.gather(*[run_test_b(i, t) for i, t in enumerate(test_tasks)])

    print(f"    [C] EvoArena + SkillForge (failure-aware within-task routing)...", flush=True)
    async def run_test_c(i, task):
        async with sem:
            if benchmark == "gaia2":
                return await run_gaia2_task_with_are(task, "", "C",
                                                      within_task_patch_mode="skillforge")
            if benchmark in ("gaia", "swebench_dynamic"):
                return await run_gaia_task_controlled(task, "", "C",
                                                       within_task_patch_mode="skillforge")
            return {"task_id": task["task_id"], "error": "unknown_benchmark", "score": 0.0}
    results_c = await asyncio.gather(*[run_test_c(i, t) for i, t in enumerate(test_tasks)])

    print(f"\n  Evaluating with EM (LLM-Judge as tie-breaker)...", flush=True)
    eval_tasks = []
    for i in range(len(test_tasks)):
        eval_tasks.append(evaluate_task(results_a[i], benchmark))
        eval_tasks.append(evaluate_task(results_b[i], benchmark))
        eval_tasks.append(evaluate_task(results_c[i], benchmark))
    all_evals = await asyncio.gather(*eval_tasks)

    scores = {"A_baseline": [], "B_evoarena": [], "C_skillforge": []}
    for i in range(len(test_tasks)):
        scores["A_baseline"].append(all_evals[i * 3])
        scores["B_evoarena"].append(all_evals[i * 3 + 1])
        scores["C_skillforge"].append(all_evals[i * 3 + 2])

    # Trace logging for test phase (all 3 groups)
    for i, task in enumerate(test_tasks):
        task_id = task["task_id"]
        task_desc = task.get("description", "")
        expected = task.get("expected", results_a[i].get("expected", ""))
        _trace.log(
            benchmark=benchmark, group="A_baseline", phase="test",
            task_id=task_id, task_desc=task_desc,
            augmented_prompt="",
            response=results_a[i].get("response", ""),
            expected=expected,
            score=all_evals[i * 3].get("score", 0.0),
        )
        aug_b = results_b[i].get("_aug_prompt", "")
        _trace.log(
            benchmark=benchmark, group="B_evoarena", phase="test",
            task_id=task_id, task_desc=task_desc,
            augmented_prompt=aug_b,
            response=results_b[i].get("response", ""),
            expected=expected,
            score=all_evals[i * 3 + 1].get("score", 0.0),
        )
        aug_c = results_c[i].get("_aug_prompt", "")
        _trace.log(
            benchmark=benchmark, group="C_skillforge", phase="test",
            task_id=task_id, task_desc=task_desc,
            augmented_prompt=aug_c,
            response=results_c[i].get("response", ""),
            expected=expected,
            score=all_evals[i * 3 + 2].get("score", 0.0),
        )

    report = {}
    for group, evals in scores.items():
        valid = [e["score"] for e in evals if e.get("score") is not None]
        ems = [e.get("em", 0.0) for e in evals]
        report[group] = {
            "avg_score": sum(valid) / len(valid) if valid else 0.0,
            "em": sum(ems) / len(ems) if ems else 0.0,
            "n": len(valid),
        }

    metric_name = "EM"
    print(f"\n  Results ({benchmark}, model={MODEL}):")
    print(f"    A (Baseline):               {metric_name}={report['A_baseline']['em']:.1%}")
    print(f"    B (EvoArena EvoMem):        {metric_name}={report['B_evoarena']['em']:.1%}")
    print(f"    C (EvoArena + SkillForge):  {metric_name}={report['C_skillforge']['em']:.1%}")
    delta_ac = report['C_skillforge']['em'] - report['A_baseline']['em']
    delta_bc = report['C_skillforge']['em'] - report['B_evoarena']['em']
    print(f"    ?(C-A): {delta_ac:+.1%} | ?(C-B): {delta_bc:+.1%}")

    full_report = {
        "benchmark": benchmark, "model": MODEL,
        "metric": metric_name,
        "n_test": len(test_tasks),
        "design": [
            "evoarena_evomem_within_task_patch_memory",
            "failure_aware_attention_routing",
            "cross_agent_critic_gating",
            "exact_match_metrics",
        ],
        "results": report,
        "delta_skillforge_vs_baseline": delta_ac,
        "delta_skillforge_vs_evoarena": delta_bc,
    }

    if benchmark == "gaia2":
        config_scores = {}
        for i, task in enumerate(test_tasks):
            config = (task.get("metadata") or {}).get("config", "unknown")
            if config not in config_scores:
                config_scores[config] = {"A_baseline": [], "B_evoarena": [], "C_skillforge": []}
            config_scores[config]["A_baseline"].append(all_evals[i * 3])
            config_scores[config]["B_evoarena"].append(all_evals[i * 3 + 1])
            config_scores[config]["C_skillforge"].append(all_evals[i * 3 + 2])

        per_config_report = {}
        for config, groups in sorted(config_scores.items()):
            per_config_report[config] = {}
            for group, evals in groups.items():
                ems = [e.get("em", 0.0) for e in evals]
                step_scores = [e.get("score", 0.0) for e in evals]
                per_config_report[config][group] = {
                    "pass_at_1": sum(ems) / len(ems) if ems else 0.0,
                    "step_score": sum(step_scores) / len(step_scores) if step_scores else 0.0,
                    "n": len(evals),
                }
        full_report["per_config"] = per_config_report

        print(f"\n  GAIA2 Per-Config Breakdown (Group A Baseline):")
        print(f"    {'Config':<15} {'pass@1':>8} {'step_score':>12} {'n':>4}")
        print(f"    {'-'*42}")
        total_pass = 0
        total_step = 0.0
        total_n = 0
        for config, groups in sorted(per_config_report.items()):
            a = groups["A_baseline"]
            print(f"    {config:<15} {a['pass_at_1']:>7.1%} {a['step_score']:>11.3f} {a['n']:>4}")
            total_pass += int(a['pass_at_1'] * a['n'])
            total_step += a['step_score'] * a['n']
            total_n += a['n']
        if total_n > 0:
            print(f"    {'-'*42}")
            print(f"    {'TOTAL':<15} {total_pass/total_n:>7.1%} {total_step/total_n:>11.3f} {total_n:>4}")

        print(f"\n  GAIA2 Per-Config Breakdown (Group C EvoArena+SkillForge):")
        print(f"    {'Config':<15} {'pass@1':>8} {'step_score':>12} {'n':>4}")
        print(f"    {'-'*42}")
        total_pass = 0
        total_step = 0.0
        total_n = 0
        for config, groups in sorted(per_config_report.items()):
            c = groups["C_skillforge"]
            print(f"    {config:<15} {c['pass_at_1']:>7.1%} {c['step_score']:>11.3f} {c['n']:>4}")
            total_pass += int(c['pass_at_1'] * c['n'])
            total_step += c['step_score'] * c['n']
            total_n += c['n']
        if total_n > 0:
            print(f"    {'-'*42}")
            print(f"    {'TOTAL':<15} {total_pass/total_n:>7.1%} {total_step/total_n:>11.3f} {total_n:>4}")

    if benchmark == "gaia":
        level_scores = {}
        for i, task in enumerate(test_tasks):
            level = (task.get("metadata") or {}).get("level", "unknown")
            if level not in level_scores:
                level_scores[level] = {"A_baseline": [], "B_evoarena": [], "C_skillforge": []}
            level_scores[level]["A_baseline"].append(all_evals[i * 3])
            level_scores[level]["B_evoarena"].append(all_evals[i * 3 + 1])
            level_scores[level]["C_skillforge"].append(all_evals[i * 3 + 2])

        per_level_report = {}
        for level, groups in sorted(level_scores.items()):
            per_level_report[level] = {}
            for group, evals in groups.items():
                ems = [e.get("em", 0.0) for e in evals]
                per_level_report[level][group] = {
                    "score": sum(ems) / len(ems) if ems else 0.0,
                    "n": len(evals),
                }
        full_report["per_level"] = per_level_report

        print(f"\n  GAIA Per-Level Breakdown (Official Leaderboard Format):")
        print(f"    {'Level':<10} {'Baseline':>10} {'EvoArena+SkillForge':>20} {'n':>4}")
        print(f"    {'-'*46}")
        for level in sorted(per_level_report.keys()):
            a = per_level_report[level]["A_baseline"]
            c = per_level_report[level]["C_skillforge"]
            print(f"    Level {level:<5} {a['score']:>9.1%} {c['score']:>19.1%} {a['n']:>4}")
        all_a = [e.get("em", 0.0) for e in all_evals[::3]]
        all_c = [e.get("em", 0.0) for e in all_evals[2::3]]
        avg_a = sum(all_a) / len(all_a) if all_a else 0.0
        avg_c = sum(all_c) / len(all_c) if all_c else 0.0
        print(f"    {'-'*40}")
        print(f"    {'Average':<10} {avg_a:>9.1%} {avg_c:>11.1%} {len(test_tasks):>4}")

    with open(f"{RESULTS_DIR}/{benchmark}/report.json", "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    return full_report


async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("??????????????????????????????????????????????????????????????????????")
    print("?  SkillForge Latest ? Main Orchestrator (v4 ? no training phase)   ?")
    print("?  A/B/C testing ? EM metrics ? Per-benchmark sub-runners          ?")
    print(f"?  Model: {MODEL:<22} | Concurrency: {CONCURRENCY:<3}              ?")
    print("??????????????????????????????????????????????????????????????????????")

    print("\n  ? Probing API availability...", flush=True)
    api_ok = await probe_api_available()
    if not api_ok:
        print("  ? DeepSeek V4 Pro API is NOT available. Aborting.", flush=True)
        print("  ? Re-run this script when API is back online.", flush=True)
        return
    print("  ? API is responding.", flush=True)

    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    completed_benchmarks = checkpoint.get("completed_benchmarks", {})
    if completed_benchmarks:
        print(f"\n  ? Resuming from checkpoint: {list(completed_benchmarks.keys())} already done.", flush=True)

    BENCHMARKS_TO_RUN = ["gaia", "gaia2", "swebench_dynamic"]
    print(f"\n  Loading benchmarks: {BENCHMARKS_TO_RUN}...")
    benchmarks = {}
    for name in BENCHMARKS_TO_RUN:
        config = {"name": name, "num_samples": TASK_LIMITS[name]}
        if name == "gaia2":
            config["scenario_dir"] = "/tmp/harbor-datasets/datasets/gaia2-cli"
        loader = BenchmarkLoader(config)
        tasks = loader.load()[:TASK_LIMITS[name]]
        benchmarks[name] = tasks
        print(f"    {name}: {len(tasks)} tasks")

    print(f"\n  Total: {sum(len(t) for t in benchmarks.values())} tasks")

    all_reports = dict(completed_benchmarks)
    paused = False

    for name, tasks in benchmarks.items():
        if name in completed_benchmarks:
            print(f"\n  SKIP {name}: already completed (from checkpoint)")
            continue
        if not tasks:
            print(f"\n  SKIP {name}: no tasks")
            continue

        api_ok = await probe_api_available()
        if not api_ok:
            print(f"\n  ??  API unavailable before starting {name}. Pausing experiment.", flush=True)
            save_checkpoint({"completed_benchmarks": all_reports, "paused_at": name,
                             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, CHECKPOINT_FILE)
            paused = True
            break

        try:
            all_reports[name] = await run_benchmark(name, tasks)
        except APIUnavailableError as e:
            print(f"\n  ??  API became unavailable during {name}: {e}", flush=True)
            print(f"  ? Saving checkpoint and pausing...", flush=True)
            save_checkpoint({"completed_benchmarks": all_reports, "paused_at": name,
                             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "error": str(e)}, CHECKPOINT_FILE)
            paused = True
            break
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {name}: {e}")
            traceback.print_exc()
            partial = _compute_partial_results_from_trace(name)
            if partial:
                all_reports[name] = partial
            else:
                all_reports[name] = {"error": str(e)}

    if paused:
        print(f"\n\n{'?'*70}")
        print(f"  EXPERIMENT PAUSED ? API unavailable")
        print(f"  Completed: {[k for k in all_reports if 'error' not in all_reports.get(k, {})]}")
        print(f"  Re-run this script to resume from checkpoint.")
        print(f"{'?'*70}")
    else:
        clear_checkpoint(CHECKPOINT_FILE)
        print(f"\n\n{'?'*70}")
        print(f"  ALL BENCHMARKS COMPLETE")
        print(f"{'?'*70}")
        for name, report in all_reports.items():
            if isinstance(report, dict) and "results" in report:
                r = report["results"]
                print(f"  {name:>20}: A={r['A_baseline']['em']:.1%}, B={r['B_evoarena']['em']:.1%}, C={r['C_skillforge']['em']:.1%}")
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
