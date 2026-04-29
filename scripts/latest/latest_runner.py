#!/usr/bin/env python3
"""SkillForge Latest ? Main Orchestrator (v5 ? 5 primary benchmarks, EvoArena injection)."""
import asyncio
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
# Backbone model for this run. The wrapper scripts/latest/run_all_models.sh sets
# CODEBUDDY_MODEL per model and invokes this script once per model; setdefault
# keeps HY3-preview (in-house) as the default when run directly.
os.environ.setdefault('CODEBUDDY_MODEL', 'hy3-preview-ioa')
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')

# ── Cap CPU thread fan-out BEFORE importing torch / sentence-transformers ──
# The embedding model runs on CPU; by default each encode() grabs every core, so
# K parallel tasks spawn K×cores threads and thrash the machine. Pin intra-op
# threads to 1 (override with EMBED_NUM_THREADS) so total threads ≈ concurrency,
# not concurrency×cores. This must run before the heavy imports below.
_EMBED_THREADS = os.environ.get("EMBED_NUM_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _EMBED_THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from benchmarks.loader import BenchmarkLoader
from latest.eval.gaia2_judge import evaluate_gaia2 as _gaia2_official_judge

from time import perf_counter as _perf

from scripts.latest.trace import TraceLogger, APIUnavailableError
from scripts.latest.profiling import (
    start_task_profile, read_task_profile, summarize,
)
from scripts.latest.llm_client import (
    probe_api_available, _check_api_error,
    save_checkpoint, load_checkpoint, clear_checkpoint,
    _llm_call, _llm_call_notool, _llm_short_call,
    llm_extract_answer, llm_judge_answer,
)
from scripts.latest.eval import (
    normalize_answer, exact_match,
    compute_partial_results_from_trace,
)

# --- Sub-runners (per-benchmark EvoArena-style within-agent injection) ---
from scripts.latest.gaia_runner import run_gaia_task, run_gaia_task_controlled
from scripts.latest.gaia2_runner import run_gaia2_task_with_are
from scripts.latest.terminal_bench_2_runner import run_terminal_bench_2_task, run_terminal_bench_2_task_controlled
from scripts.latest.locomo_runner import run_locomo_task, run_locomo_task_controlled
from scripts.latest.evomem_bridge import BenchmarkMemory, solve_with_memory

MODEL = os.environ.get("CODEBUDDY_MODEL", "deepseek-v4-pro")


def _auto_slots() -> int:
    """Pick a global heavy-task cap from the machine. Memory-aware when psutil
    is available, else a conservative cpu-based default. The point is that this
    bounds PEAK resource use, so expanding the test set only lengthens the queue
    — it does not raise memory/CPU pressure."""
    cpu = os.cpu_count() or 4
    slots = max(1, min(cpu - 1, 8))
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        per_task_gb = float(os.environ.get("PER_TASK_GB", "1.2"))
        slots = max(1, min(slots, int(avail_gb / per_task_gb)))
    except Exception:
        pass
    return slots


# Concurrency model: a SINGLE global semaphore bounds total concurrent heavy
# tasks across ALL benchmarks (decoupled from how many benchmarks are in flight),
# so parallelism no longer multiplies (old 2×N model OOM'd). Docker-backed
# benchmarks take an extra, tighter sub-cap because containers are memory-hungry.
GLOBAL_TASK_SLOTS = int(os.environ.get("TASK_CONCURRENCY", "0")) or _auto_slots()
DOCKER_TASK_SLOTS = int(os.environ.get("DOCKER_CONCURRENCY", "2"))
BENCHMARK_CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "6"))  # benchmarks may interleave
DOCKER_BENCHMARKS = {"terminal_bench_2"}
CONCURRENCY = GLOBAL_TASK_SLOTS  # kept for the startup banner

# Loop-bound semaphores, created inside main() (asyncio objects must bind to the
# running loop). Shared by every benchmark's task runner.
_SEMS: dict = {}

# Per-task retry on transient (rate-limit / timeout) failures.
TASK_MAX_RETRIES = int(os.environ.get("TASK_MAX_RETRIES", "3"))
TASK_RETRY_BASE_DELAY = float(os.environ.get("TASK_RETRY_BASE_DELAY", "8"))
# RESUME=1 keeps existing trace.jsonl and skips already-completed (group, task_id)
# pairs instead of wiping and restarting from scratch.
RESUME = os.environ.get("RESUME", "0") == "1"

_TRANSIENT_MARKERS = (
    "429", "rate_limit", "rate-limit", "timeout", "quota", "quota_exceeded",
    "unavailable", "APIUnavailable", "503", "502", "overloaded",
)

# Per-model results so the 7 backbones do not overwrite each other:
#   experiments_results/latest/<model>/<benchmark>/{trace.jsonl,report.json}
_MODEL_SLUG = re.sub(r"[^A-Za-z0-9._-]", "_",
                     os.environ.get("CODEBUDDY_MODEL", "hy3-preview-ioa"))
RESULTS_DIR = str(PROJECT_ROOT / "experiments_results" / "latest" / _MODEL_SLUG)

# --- Primary Benchmarks ---
# Scaled to 100/benchmark (override with TASK_LIMIT=<n>) so A/B/C deltas can
# reach significance; the loader caps at each benchmark's available pool.
# terminal_bench_2 runs via Docker directly using the SDK as the LLM backend.
_TASK_N = int(os.environ.get("TASK_LIMIT", "100"))
TASK_LIMITS = {
    "gaia": _TASK_N,
    "gaia2": _TASK_N,
    "terminal_bench_2": _TASK_N,
    "locomo": _TASK_N,
}

CHECKPOINT_FILE = str(Path(RESULTS_DIR) / "_checkpoint.json")
_trace = TraceLogger(RESULTS_DIR)

# ── GAIA2 System Prompt Augmentations (B/C differentiation) ──────────────
# Since GAIA2 uses a manual ARE tool loop (similar to GAIA controlled runner),
# we inject metacognitive guidance as experience_section for B and C groups.
# This differentiates B (EvoArena-style self-correction) from C (SkillForge
# precision refinement) without requiring SDK-level hooks.

GAIA2_EVOARENA_AUG = """
## Self-Correction & Verification Protocol

Apply these principles during your multi-step execution:

1. **Action Verification**: After each tool call, verify the result is what you
   expected. If a command returns an unexpected output, immediately diagnose
   WHY and correct before the next action — don't stack errors.

2. **Plan Before Execute**: Before running a tool, state your hypothesis: what
   you expect to find/achieve. After execution, compare result to hypothesis.
   If mismatch, revise the plan.

3. **Error Recovery**: When a tool returns an error or unexpected result:
   (a) Identify the root cause (wrong params, wrong tool, wrong assumption),
   (b) Formulate a corrected approach,
   (c) Execute immediately — don't re-try the same failing approach.

4. **Progressive Disclosure**: If a query returns too much data, refine it.
   If a search returns nothing, try a BROADER term — don't paginate empty results.

5. **Final Verification**: Before calling ALL_DONE, silently verify each
   sub-goal was achieved. Cross-check outputs against the original task.
"""

GAIA2_SKILLFORGE_AUG = """
## Plan-First Architecture (SkillForge)

You operate in a structured PLAN (internal) → EXECUTE → VERIFY cycle.
This is NOT the same as the Self-Correction Protocol — it's a fundamentally
different workflow.

### Phase 1: PLAN (internal mental planning)

Before your first tool call, think about:
- What information do you need?
- Which tool will you use to get it?
- What does success look like for each sub-goal?

Keep your plan internal. Do NOT output the plan as text — go straight to
your first NEXT_OP tool call.

### Phase 2: EXECUTE (one sub-goal at a time)

Execute sequentially. After each tool call:
- Was the output what you expected per your plan?
- Does this change any later sub-goals?
- If a sub-goal fails, pause and re-plan — don't blindly continue.

### Phase 3: VERIFY (after each sub-goal and before finishing)

After each sub-goal, explicitly check:
- Did I get the information I planned to get?
- Is the information consistent with what I know?
- Does the answer match the task's required format?

After ALL sub-goals:
- Re-read the original task description
- Verify your final answer satisfies ALL requirements
- Check formatting: no extra text, no commentary

### Key Differences from Self-Correction Protocol

- PLAN is proactive (before execution), not reactive (after errors)
- VERIFY is systematic (every sub-goal), not ad-hoc (only on failure)
- The cycle repeats: internal PLAN → EXECUTE → VERIFY → re-PLAN if needed
"""


# --- Resilience helpers (retry + resume) ---

def _is_transient(err: str) -> bool:
    err = (err or "")
    return any(k in err for k in _TRANSIENT_MARKERS)


async def _build_with_retry(build_coro, task: dict) -> dict:
    """Run a single task builder, retrying with exponential backoff on transient
    (rate-limit / timeout / overload) failures. Non-transient errors and successful
    results are returned immediately. The whole point is that one flaky API call no
    longer kills a whole benchmark group."""
    last = None
    for attempt in range(TASK_MAX_RETRIES + 1):
        try:
            r = await build_coro(task)
        except Exception as e:
            r = {"task_id": task.get("task_id", ""), "response": "",
                 "error": f"{type(e).__name__}: {str(e)[:200]}"}
        last = r
        err = str(r.get("error") or "")
        resp = (r.get("response") or "").strip()
        transient = _is_transient(err)
        # An empty response with no error is almost always an API blip that
        # returned nothing — treat it as a soft, retryable failure (previously it
        # fell through `if not err` and was returned unretried, scoring 0).
        soft_empty = (not resp) and (not err)
        # Return immediately on a clean success or a HARD (non-retryable) error
        # (e.g. docker_pull_failed — retrying won't help).
        if (resp and not transient) or (err and not transient):
            return r
        if attempt < TASK_MAX_RETRIES and (transient or soft_empty):
            delay = TASK_RETRY_BASE_DELAY * (2 ** attempt)
            reason = err[:60] or "empty_response"
            print(f"      ↻ retry {attempt+1}/{TASK_MAX_RETRIES} "
                  f"{task.get('task_id','')} in {delay:.0f}s ({reason})", flush=True)
            await asyncio.sleep(delay)
            continue
        return r
    return last or {"task_id": task.get("task_id", ""), "response": "",
                    "error": "retry_exhausted"}


def _load_done_map(trace_path: Path) -> dict:
    """Read an existing trace.jsonl into {(group, task_id): record} for resume.
    Later records win, so a re-run of a task overrides an earlier partial."""
    done: dict = {}
    if not trace_path.exists():
        return done
    try:
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[(rec.get("group", ""), rec.get("task_id", ""))] = rec
    except Exception as e:
        print(f"  [resume] failed to parse {trace_path}: {e}")
    return done


# --- Evaluation ---

async def evaluate_task(result: dict, benchmark: str, use_llm_judge: bool = True) -> dict:
    """Primary metric per benchmark:
       - gaia2: GAIA2 official judge (action sequence + gate matching)
       - terminal_bench_2: exact match on command output
       - gaia / locomo: exact match with LLM-Judge tie-breaker
    """
    if benchmark == "gaia2":
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

        async def _judge_llm_call(system_prompt: str, user_prompt: str) -> str:
            try:
                r = await _llm_call_notool(system_prompt, user_prompt, timeout=60)
                return (r.get("text") or "").strip()
            except Exception as e:
                print(f"[GAIA2 judge] LLM call failed: {e}")
                return ""

        try:
            judge_result = await _gaia2_official_judge(
                _judge_llm_call, config=config, task=task_desc,
                oracle_events=oracle_events, oracle_answer=oracle_answer,
                event_log=event_log, agent_response=response,
            )
            return judge_result
        except Exception as e:
            print(f"[GAIA2 judge] Official judge failed: {e}")
            return {"score": 0.0, "em": 0.0, "method": "gaia2_judge_error", "error": str(e)[:200]}

    if benchmark == "terminal_bench_2":
        # Docker-based eval: parse pytest output for partial credit.
        # Official TB2 uses pass@1 (binary), but we compute a finer-grained
        # score from pytest pass/fail ratios to reward partial progress.
        test_passed = result.get("test_passed", False)
        test_output = result.get("test_output", "")
        if test_passed:
            return {"score": 1.0, "em": 1.0, "method": "docker_pytest_pass"}

        # Parse pytest summary line: "X failed, Y passed" or "Y passed, X failed"
        import re as _re
        passed = 0
        failed = 0
        _tb2_match = _re.search(
            r'(\d+)\s+passed.*?(\d+)\s+failed|'
            r'(\d+)\s+failed.*?(\d+)\s+passed',
            test_output
        )
        if _tb2_match:
            if _tb2_match.group(1) and _tb2_match.group(2):
                # "X passed, Y failed"
                passed = int(_tb2_match.group(1))
                failed = int(_tb2_match.group(2))
            elif _tb2_match.group(3) and _tb2_match.group(4):
                # "X failed, Y passed"
                failed = int(_tb2_match.group(3))
                passed = int(_tb2_match.group(4))

        total = passed + failed
        if total > 0:
            partial_score = passed / total
            em = 1.0 if partial_score >= 1.0 else 0.0
            return {
                "score": partial_score,
                "em": em,
                "method": f"pytest_partial_{passed}_{failed}",
                "pytest_passed": passed,
                "pytest_failed": failed,
                "pytest_total": total,
            }

        # No machine-checkable test evidence → score 0. We do NOT credit the
        # agent just because its prose contains "passed": that gave false
        # positives (e.g. a prompt-only run that never executed any test scoring
        # 1.0). A task with no test_output simply did not run its tests.
        if not (test_output or "").strip():
            return {"score": 0.0, "em": 0.0, "method": "no_test_evidence"}
        return {"score": 0.0, "em": 0.0, "method": "docker_pytest_fail",
                "test_output": (test_output or "")[:500]}

    # gaia, locomo: exact match with LLM-Judge tie-breaker
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


# --- Benchmark runner ---

async def run_benchmark(benchmark: str, tasks: list) -> dict:
    """Run A/B/C testing on ALL tasks (no train/test split)."""
    print(f"\n{'='*70}")
    print(f"  Benchmark: {benchmark} (model: {MODEL})")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Metric: Exact Match")
    print(f"{'='*70}")

    os.makedirs(f"{RESULTS_DIR}/{benchmark}", exist_ok=True)

    trace_path = Path(RESULTS_DIR) / benchmark / "trace.jsonl"
    done_map: dict = {}
    if trace_path.exists():
        if RESUME:
            done_map = _load_done_map(trace_path)
            print(f"  RESUME: {len(done_map)} completed (group,task) pairs will be skipped")
        else:
            trace_path.unlink()
            print(f"  Cleared stale trace: {trace_path}")
    _trace.clear_benchmark(benchmark)

    test_tasks = tasks

    # --- Dispatch table (benchmark -> runner functions) ---
    BASELINE_RUNNER = {
        "gaia": run_gaia_task,
        "gaia2": run_gaia2_task_with_are,
        "terminal_bench_2": run_terminal_bench_2_task,
        "locomo": run_locomo_task,
    }
    CONTROLLED_RUNNER = {
        "gaia": run_gaia_task_controlled,  # B/C use controlled runner with within-task EvoMem patches
        "gaia2": run_gaia2_task_with_are,
        "terminal_bench_2": run_terminal_bench_2_task_controlled,
        "locomo": run_locomo_task_controlled,
    }

    run_fn_a = BASELINE_RUNNER.get(benchmark)
    run_fn_controlled = CONTROLLED_RUNNER.get(benchmark)

    # Multiround agentic benchmarks (GAIA, GAIA2, Terminal-Bench):
    #   All A/B/C groups run the same baseline runner — no scaffold injection.
    #   Self-consistency and within-task patches are designed for single-round
    #   QA and would interfere with the agent's multi-turn reasoning workflow.
    #
    # Single-round QA benchmarks (LoCoMo, PersonaMem-v2):
    #   Group B uses self-consistency sampling (3 samples, majority vote).
    #   Group C uses evidence-weighted / persona-consistent voting.
    is_qa = benchmark in ("locomo",)

    if not run_fn_a:
        print(f"  ERROR: No runner for benchmark '{benchmark}'")
        return {"error": f"no_runner_{benchmark}"}

    print(f"\n  Testing {len(test_tasks)} tasks x 3 groups (A/B/C)...")
    # Shared, loop-bound semaphores (created in main): one global cap for all
    # heavy work, plus a tighter sub-cap for Docker-backed benchmarks.
    global_sem = _SEMS["global"]
    docker_sem = _SEMS.get("docker")
    needs_docker = benchmark in DOCKER_BENCHMARKS

    async def _run_group(label: str, group_key: str, tasks: list[dict], build_coro):
        """Run tasks concurrently. Evaluate + trace-log each task as it completes so
        partial results survive crashes. With RESUME=1, already-traced tasks are
        reconstructed from the trace instead of re-run; every fresh task is retried
        with backoff on transient API failures."""
        total = len(tasks)
        results = [None] * total
        evals = [None] * total

        async def _wrap(i: int, task: dict):
            tid = task.get("task_id", "")
            # ── Resume: reconstruct already-completed tasks from trace ──
            prev = done_map.get((group_key, tid))
            if prev is not None:
                r = {"task_id": tid, "response": prev.get("response", ""),
                     "expected": prev.get("expected", ""),
                     "_aug_prompt": prev.get("augmented_prompt", ""),
                     "_sc_responses": prev.get("_sc_responses", []),
                     "_resumed": True}
                ev = {"score": prev.get("score", 0.0), "em": prev.get("em", 0.0),
                      "method": prev.get("method", "resumed")}
                print(f"    [{label}] {i+1}/{total} ⟳ {tid} (resumed EM={ev['em']:.0%})",
                      flush=True)
                return i, r, ev
            # ── Run with retry, bounded by the global (and, for Docker
            #    benchmarks, the tighter container) semaphore.
            #    Profile per-task wall-clock (embed / docker / llm_io). ──
            start_task_profile()
            _t0 = _perf()
            async with global_sem:
                if needs_docker and docker_sem is not None:
                    async with docker_sem:
                        r = await _build_with_retry(build_coro, task)
                else:
                    r = await _build_with_retry(build_coro, task)
            prof = summarize(_perf() - _t0, read_task_profile())
            if isinstance(r, dict):
                r["_prof"] = prof
            # ── Evaluate immediately ──────────────────────────────
            ev = await evaluate_task(r, benchmark)
            # ── Trace-log immediately (with full metric provenance) ──
            expected = task.get("expected", r.get("expected", ""))
            aug = r.get("_aug_prompt", "")
            exec_mode = (r.get("execution_mode")
                         or r.get("_within_task_patch_mode")
                         or "default")
            _meta = task.get("metadata") or {}
            _trace.log(benchmark=benchmark, group=group_key, phase="test",
                       task_id=r.get("task_id", tid),
                       task_desc=task.get("description", ""),
                       augmented_prompt=aug,
                       response=r.get("response", ""), expected=expected,
                       score=ev.get("score", 0.0),
                       extra={"em": ev.get("em", 0.0),
                              "method": ev.get("method", ""),
                              "execution_mode": exec_mode,
                              "error": str(r.get("error") or "")[:200],
                              # --- per-sample fields for the breakdown sub-tables ---
                              # category/type and difficulty (when the benchmark provides them)
                              "category": str(_meta.get("category") or _meta.get("task_type")
                                              or _meta.get("type") or ""),
                              "level": str(_meta.get("level") or _meta.get("difficulty") or ""),
                              # whether a curated patch was injected (for retrieved-vs-not analysis)
                              "patch_injected": bool(aug),
                              "aug_len": len(aug or ""),
                              **{f"prof_{k}": v for k, v in prof.items()}})
            tag = r.get("task_id", str(i))
            err = r.get("error")
            status = "\u2717" if err else "\u2713"
            msg = (f"    [{label}] {i+1}/{total} {status} {tag} "
                   f"({prof['total_s']:.0f}s: embed {prof['embed_s']:.0f} "
                   f"dock {prof['docker_s']:.0f} llm/io {prof['llm_io_s']:.0f}) "
                   f"EM={ev.get('em',0):.0%}")
            if err:
                msg += f" ERR: {str(err)[:80]}"
            print(msg, flush=True)
            return i, r, ev

        for coro in asyncio.as_completed([_wrap(i, t) for i, t in enumerate(tasks)]):
            i, r, ev = await coro
            results[i] = r
            evals[i] = ev
        return results, evals

    # ── Unified A/B/C matching the paper (one mechanism for every benchmark) ──
    #   A  Vanilla : no memory.
    #   B  EvoMem  : cross-task patch memory — retrieve + inject prior patches.
    #   C  GPR     : B's patches + per-patch environment-check (additive superset).
    # B and C each keep their own cross-task patch memory; A keeps none.
    mem_b = BenchmarkMemory(benchmark, "B")
    mem_c = BenchmarkMemory(benchmark, "C")

    print(f"    [A] Vanilla (no memory)...", flush=True)
    results_a, evals_a = await _run_group("A", "A_baseline", test_tasks,
                                          lambda t: run_fn_a(t, "", "A"))

    print(f"    [B] EvoMem (cross-task patch memory)...", flush=True)
    results_b, evals_b = await _run_group("B", "B_evomem", test_tasks,
        lambda t: solve_with_memory(run_fn_a, t, mem_b, "B"))

    print(f"    [C] GPR (EvoMem patches + environment grounding)...", flush=True)
    results_c, evals_c = await _run_group("C", "C_gpr", test_tasks,
        lambda t: solve_with_memory(run_fn_a, t, mem_c, "C"))

    # Build flat all_evals for per_level / per_config breakdowns
    all_evals = []
    for i in range(len(test_tasks)):
        all_evals.append(evals_a[i])
        all_evals.append(evals_b[i])
        all_evals.append(evals_c[i])

    scores = {"A_baseline": [], "B_evomem": [], "C_gpr": []}
    for i in range(len(test_tasks)):
        scores["A_baseline"].append(all_evals[i * 3])
        scores["B_evomem"].append(all_evals[i * 3 + 1])
        scores["C_gpr"].append(all_evals[i * 3 + 2])

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
    print(f"    A (Vanilla):  {metric_name}={report['A_baseline']['em']:.1%}")
    print(f"    B (EvoMem):   {metric_name}={report['B_evomem']['em']:.1%}")
    print(f"    C (GPR):      {metric_name}={report['C_gpr']['em']:.1%}")
    delta_ac = report['C_gpr']['em'] - report['A_baseline']['em']
    delta_bc = report['C_gpr']['em'] - report['B_evomem']['em']
    print(f"    Delta(C-A): {delta_ac:+.1%} | Delta(C-B): {delta_bc:+.1%}")

    # ── Per-task wall-clock breakdown (where time goes → how to scale) ──
    _profs = [r["_prof"] for grp in (results_a, results_b, results_c)
              for r in grp if isinstance(r, dict) and r.get("_prof")]
    _profile_avg = None
    if _profs:
        n = len(_profs)
        avg = {k: sum(p[k] for p in _profs) / n
               for k in ("total_s", "embed_s", "docker_s", "llm_io_s")}
        _profile_avg = {k: round(v, 2) for k, v in avg.items()}
        print(f"    Avg/task: total {avg['total_s']:.1f}s = "
              f"embed {avg['embed_s']:.1f} (CPU) + "
              f"docker {avg['docker_s']:.1f} (mem) + "
              f"llm/io {avg['llm_io_s']:.1f} (net). "
              f"High embed/docker → keep concurrency low; high llm/io → raise it.")

    full_report = {
        "benchmark": benchmark, "model": MODEL, "metric": metric_name,
        "n_test": len(test_tasks),
        "profile_avg_s": _profile_avg,
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
                config_scores[config] = {"A_baseline": [], "B_evomem": [], "C_gpr": []}
            config_scores[config]["A_baseline"].append(all_evals[i * 3])
            config_scores[config]["B_evomem"].append(all_evals[i * 3 + 1])
            config_scores[config]["C_gpr"].append(all_evals[i * 3 + 2])
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

    if benchmark == "gaia":
        level_scores = {}
        for i, task in enumerate(test_tasks):
            level = (task.get("metadata") or {}).get("level", "unknown")
            if level not in level_scores:
                level_scores[level] = {"A_baseline": [], "B_evomem": [], "C_gpr": []}
            level_scores[level]["A_baseline"].append(all_evals[i * 3])
            level_scores[level]["B_evomem"].append(all_evals[i * 3 + 1])
            level_scores[level]["C_gpr"].append(all_evals[i * 3 + 2])
        per_level_report = {}
        for level, groups in sorted(level_scores.items()):
            per_level_report[level] = {}
            for group, evals in groups.items():
                ems = [e.get("em", 0.0) for e in evals]
                per_level_report[level][group] = {
                    "score": sum(ems) / len(ems) if ems else 0.0, "n": len(evals),
                }
        full_report["per_level"] = per_level_report
        print(f"\n  GAIA Per-Level Breakdown:")
        print(f"    {'Level':<10} {'Baseline':>10} {'EvoArena+SkillForge':>20} {'n':>4}")
        print(f"    {'-'*46}")
        for level in sorted(per_level_report.keys()):
            a = per_level_report[level]["A_baseline"]
            c = per_level_report[level]["C_gpr"]
            print(f"    Level {level:<5} {a['score']:>9.1%} {c['score']:>19.1%} {a['n']:>4}")

    with open(f"{RESULTS_DIR}/{benchmark}/report.json", "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    return full_report


# --- Main ---

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Create shared, loop-bound semaphores now that we're inside the event loop.
    _SEMS["global"] = asyncio.Semaphore(GLOBAL_TASK_SLOTS)
    _SEMS["docker"] = asyncio.Semaphore(min(DOCKER_TASK_SLOTS, GLOBAL_TASK_SLOTS))
    print("=" * 70)
    print("  SkillForge Latest ? Main Orchestrator (v5 ? 5 benchmarks)")
    print("  A/B/C testing ? EM metrics ? EvoArena within-agent injection")
    print(f"  Model: {MODEL:<22} | Global slots: {GLOBAL_TASK_SLOTS} "
          f"(docker {min(DOCKER_TASK_SLOTS, GLOBAL_TASK_SLOTS)}) "
          f"| embed threads: {_EMBED_THREADS}")
    print("=" * 70)

    print("\n  Probing API availability...", flush=True)
    api_ok = await probe_api_available()
    if not api_ok:
        print("  DeepSeek V4 Pro API is NOT available. Aborting.", flush=True)
        return
    print("  API is responding.", flush=True)

    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    completed_benchmarks = checkpoint.get("completed_benchmarks", {})
    if completed_benchmarks:
        print(f"\n  Resuming from checkpoint: {list(completed_benchmarks.keys())} already done.", flush=True)

    BENCHMARKS_TO_RUN = [
        "gaia", "gaia2", "terminal_bench_2", "locomo"
    ]
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

    print(f"\n  Total: {sum(len(t) for t in benchmarks.values())} tasks across {len(benchmarks)} benchmarks\n")

    all_reports = dict(completed_benchmarks)
    bench_sem = asyncio.Semaphore(BENCHMARK_CONCURRENCY)

    async def _run_one(name: str, tasks: list):
        async with bench_sem:
            print(f"\n  >>> Concurrent start: {name} ({len(tasks)} tasks)", flush=True)
            try:
                return name, await run_benchmark(name, tasks)
            except APIUnavailableError as e:
                print(f"\n  API unavailable during {name}: {e}", flush=True)
                return name, {"error": f"api_unavailable: {e}"}
            except Exception as e:
                import traceback
                print(f"\n  ERROR on {name}: {e}")
                traceback.print_exc()
                partial = compute_partial_results_from_trace(name, RESULTS_DIR)
                return name, partial if partial else {"error": str(e)}

    # Filter out already-completed and empty benchmarks
    pending = [(n, t) for n, t in benchmarks.items()
               if n not in completed_benchmarks and t]
    skipped = [(n, t) for n, t in benchmarks.items()
               if n in completed_benchmarks or not t]
    for name, _ in skipped:
        if name in completed_benchmarks:
            print(f"  SKIP {name}: already completed (from checkpoint)")
        else:
            print(f"  SKIP {name}: no tasks")

    if not pending:
        print("  All benchmarks already completed. Nothing to run.")
    else:
        coros = [_run_one(name, tasks) for name, tasks in pending]
        results_list = await asyncio.gather(*coros, return_exceptions=True)

        for result in results_list:
            if isinstance(result, Exception):
                print(f"\n  CRITICAL: benchmark gather failed: {result}")
            else:
                name, report = result
                all_reports[name] = report

    clear_checkpoint(CHECKPOINT_FILE)
    print(f"\n\n{'='*70}")
    print(f"  ALL BENCHMARKS COMPLETE")
    print(f"{'='*70}")
    for name, report in all_reports.items():
        if isinstance(report, dict) and "results" in report:
            r = report["results"]
            print(f"  {name:>20}: A={r['A_baseline']['em']:.1%}, B={r['B_evomem']['em']:.1%}, C={r['C_gpr']['em']:.1%}")
        elif isinstance(report, dict) and "error" in report:
            print(f"  {name:>20}: ERROR — {report['error'][:80]}")
        else:
            print(f"  {name:>20}: {report}")
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())