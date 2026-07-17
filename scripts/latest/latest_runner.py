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

# CodeBuddy stays the default (existing path). setdefault (not '=') lets a run
# opt into an alternate OpenAI-compatible backend via LLM_PROVIDER=openrouter in
# the shell or .env without touching the CodeBuddy SDK path.
os.environ.setdefault('LLM_PROVIDER', 'codebuddy')
# Auto-detect local vLLM/OpenAI backend: if an OpenAI-compatible endpoint is
# configured but LLM_PROVIDER still defaults to codebuddy, switch to vllm so
# the OpenAI path initialises correctly at module-import time.
_openai_endpoint = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
                    or os.environ.get("OPENROUTER_BASE_URL"))
if _openai_endpoint and os.environ.get("LLM_PROVIDER") == "codebuddy":
    os.environ["LLM_PROVIDER"] = "vllm"
# Backbone model for this run. The wrapper scripts/latest/run_all_models.sh sets
# CODEBUDDY_MODEL per model and invokes this script once per model; setdefault
# keeps HY3-preview (in-house) as the default when run directly.
os.environ.setdefault('CODEBUDDY_MODEL', 'hy3-preview')
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

# Fingerprint every trace row with the code revision that produced it, so a
# resumed sweep can be audited for rows that predate a bug fix (a stale-row
# resume once kept 300 pre-fix arm-B rows and silently voided a comparison).
try:
    import subprocess as _sp
    _CODE_REV = _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                 cwd=str(PROJECT_ROOT), text=True,
                                 stderr=_sp.DEVNULL).strip()
except Exception:
    _CODE_REV = "unknown"

# --- Sub-runners (per-benchmark EvoArena-style within-agent injection) ---
from scripts.latest.gaia_runner import run_gaia_task, run_gaia_task_controlled
from scripts.latest.gaia2_runner import run_gaia2_task_with_are
from scripts.latest.terminal_bench_2_runner import run_terminal_bench_2_task, run_terminal_bench_2_task_controlled
from scripts.latest.locomo_runner import run_locomo_task, run_locomo_task_controlled
from scripts.latest.evomem_bridge import BenchmarkMemory, CuratedMemory, solve_with_memory
from scripts.latest.arms import norm_group

MODEL = os.environ.get("CODEBUDDY_MODEL", "deepseek-v4-pro").lower()


def _auto_slots() -> int:
    """Pick a global heavy-task cap from the machine. Memory-aware when psutil
    is available, else a conservative cpu-based default. The point is that this
    bounds PEAK resource use, so expanding the test set only lengthens the queue
    — it does not raise memory/CPU pressure.

    Since LLM-agent workloads are I/O-bound (waiting on API), we scale with cores,
    but cap at 30: higher concurrency may put pressure on internal APIs.
    Override with TASK_CONCURRENCY to set a different cap."""
    cpu = os.cpu_count() or 4
    slots = max(1, min(cpu * 4, 30))
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
DOCKER_TASK_SLOTS = int(os.environ.get("DOCKER_CONCURRENCY", "3"))
BENCHMARK_CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "6"))  # benchmarks may interleave
# Iteration chains: run each task ITER_CHAIN times in sequence, threading B/C
# memory across iterations (chain-scoped retrieval). This is the substrate where
# patch memory pays off — feedback across iterations of the SAME task. K=1 is the
# ordinary one-shot run; the main table uses the final iteration (post-memory),
# chain-level accuracy (all iterations correct) is aggregated from the trace.
ITER_CHAIN = max(1, int(os.environ.get("ITER_CHAIN", "3")))
# Evolving chains (ITER_MUTATE=1): iterations k>=1 run a REWORDED variant of the
# task (facts/constraints/gold preserved). Two birds: (a) a frozen repeat only
# rewards memorizing one's own last answer — a variant forces the memory to
# generalize (the thesis: adaptation, not recall); (b) gold-derived feedback
# recorded on v_k and used on variant v_{k+1} is graded-practice, not label
# leakage on the scored instance. Surface rewording only — true environment
# mutation (tool versions, GAIA2 twists) is phase 2, per-benchmark.
ITER_MUTATE = os.environ.get("ITER_MUTATE", "0") == "1"
DOCKER_BENCHMARKS = {"terminal_bench_2"}
CONCURRENCY = GLOBAL_TASK_SLOTS  # kept for the startup banner

# Loop-bound semaphores, created inside main() (asyncio objects must bind to the
# running loop). Shared by every benchmark's task runner.
_SEMS: dict = {}

# Per-task retry on transient (rate-limit / timeout) failures.
TASK_MAX_RETRIES = int(os.environ.get("TASK_MAX_RETRIES", "3"))
TASK_RETRY_BASE_DELAY = float(os.environ.get("TASK_RETRY_BASE_DELAY", "8"))
# Mid-run circuit breaker: the startup probe catches an endpoint that is dead at
# t=0, but one that dies MID-sweep previously let the run keep going for hours —
# llama-33 wrote 770 consecutive APIUnavailable zero-rows. After API_DOWN_LIMIT
# consecutive tasks end with a transient/API-unavailable error (or an empty
# response), abort the whole sweep loudly. Trace rows are flushed per line, so
# finished work is preserved and RESUME=1 reruns the aborted tasks.
API_DOWN_LIMIT = int(os.environ.get("API_DOWN_LIMIT", "12"))
_api_down_streak = 0
# RESUME=1 keeps existing trace.jsonl and skips already-completed (group, task_id)
# pairs instead of wiping and restarting from scratch.
RESUME = os.environ.get("RESUME", "0") == "1"
# Arm selection (used by the ablation driver): run only these arms of A/B/C.
# Default runs all three — the main sweep is unchanged.
_ARMS = set(os.environ.get("ARMS", "A,B,C").replace(" ", "").split(","))
# Arm C's critic. Unset => the critic IS the backbone (self-curation), which is
# what every result before 2026-07-15 used. CRITIC_MODEL=<id> points it at a
# designated model and routes ONLY critic calls there.
try:
    from scripts.latest.llm_client import critic_model_id as _critic_model_id
    from scripts.latest.llm_client import judge_model_id as _judge_model_id
    _CRITIC_MODEL = _critic_model_id()
    _JUDGE_MODEL = _judge_model_id()
except Exception:
    _CRITIC_MODEL = os.environ.get("CRITIC_MODEL") or os.environ.get("CODEBUDDY_MODEL") or "?"
    _JUDGE_MODEL = os.environ.get("JUDGE_MODEL") or os.environ.get("CODEBUDDY_MODEL") or "?"
# Which curation policy arm C ran: judgment (critic score + self-assessment) or
# metadata (measured w_c + version lineage). Different methods, never poolable.
_C_META = os.environ.get("C_META", "0") == "1"
_C_POLICY = (os.environ.get("C_POLICY")
             or ("meta" if _C_META else "judgment")).strip().lower()
if _C_POLICY not in ("judgment", "meta", "guarded"):
    _C_POLICY = "judgment"
# Where a chain's feedback score comes from, and therefore whether everything
# built on it (e.score, the ✓ gate, w_c) is grounded or asserted. "gold"/"env"
# read the benchmark's own scorer; "self" asks the backbone to grade itself.
_SCORE_PROVENANCE = ("self_assessment" if os.environ.get("ITER_FEEDBACK", "gold") == "self"
                     else os.environ.get("ITER_FEEDBACK", "gold"))

_TRANSIENT_MARKERS = (
    "429", "rate_limit", "rate-limit", "timeout", "quota", "quota_exceeded",
    "unavailable", "APIUnavailable", "503", "502", "overloaded",
)

# Per-model results so the 7 backbones do not overwrite each other:
#   experiments_results/latest/<model>/<benchmark>/{trace.jsonl,report.json}
_MODEL_SLUG = re.sub(r"[^A-Za-z0-9._-]", "_",
                     os.environ.get("CODEBUDDY_MODEL", "hy3-preview")).lower()
_RESULTS_BASE = os.environ.get("RESULTS_BASE", "latest")  # ablation arms write elsewhere
RESULTS_DIR = str(PROJECT_ROOT / "experiments_results" / _RESULTS_BASE / _MODEL_SLUG)

# --- Primary Benchmarks ---
# Scaled to 100/benchmark (override with TASK_LIMIT=<n>) so A/B/C deltas can
# reach significance; the loader caps at each benchmark's available pool.
# terminal_bench_2 runs via Docker directly using the SDK as the LLM backend.
_TASK_N = int(os.environ.get("TASK_LIMIT", "100"))
# GAIA2 runs at 200 by default: its 5 config splits (execution/search/
# adaptability/ambiguity/time) mean n=100 leaves only ~20 tasks per split —
# too thin for the per-split (dynamic vs static) analysis the thesis needs;
# 200 gives ~40/split. The loader's stratified sampling is deterministic and
# superset-stable (sorted dirs, first-k per config), so RESUME on an existing
# run extends the task set instead of resampling it.
TASK_LIMITS = {
    "gaia": _TASK_N,
    "gaia2": int(os.environ.get("GAIA2_TASK_LIMIT", str(max(100, _TASK_N)))),
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
    """Retry wrapper + infra accounting. Beyond `_build_with_retry_inner`'s
    per-task retries, this (a) marks a still-empty result as an explicit infra
    error, so it is excluded from stats and rerun on RESUME instead of scoring a
    silent 0 (gaia2 committed 433 such rows), and (b) trips the API-down circuit
    breaker after API_DOWN_LIMIT consecutive infra failures."""
    global _api_down_streak
    r = await _build_with_retry_inner(build_coro, task)
    err = str(r.get("error") or "")
    resp = (r.get("response") or "").strip()
    if not resp and not err and not r.get("actions"):
        # No answer, no actions, no recorded error: the loop never engaged.
        # That is an infra zero, not a task result — mark it as such.
        r["error"] = err = "empty_response_after_retries"
    if _is_transient(err) or err == "empty_response_after_retries":
        _api_down_streak += 1
        if _api_down_streak >= API_DOWN_LIMIT:
            print("\n" + "=" * 72 +
                  f"\n  ✖ ABORT: {_api_down_streak} consecutive tasks failed with "
                  "transient/API-unavailable errors —\n    the endpoint is down. "
                  "Stopping the sweep instead of writing zero-rows.\n    Finished "
                  "work is preserved; RESUME=1 reruns the failed tasks.\n" +
                  "=" * 72, flush=True)
            sys.stdout.flush()
            os._exit(2)
    elif resp or err:
        _api_down_streak = 0
    return r


async def _build_with_retry_inner(build_coro, task: dict) -> dict:
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


# One hung task (a docker wait or an LLM call that never returns) must not pin a
# docker/global slot forever and silently stall its whole benchmark — TB2 sat 30+
# minutes without a trace write behind exactly this. Bound every task by
# wall-clock; on expiry, record an error row and move on. Defaults are generous
# because a legitimate TB2 task can run 40 turns x 180s commands (~2-3h) — the
# bound is a backstop against hangs, not a budget.
_WALL_TIMEOUT_S = int(os.environ.get("TASK_WALL_TIMEOUT_S", "2400"))
_DOCKER_WALL_TIMEOUT_S = int(os.environ.get("DOCKER_WALL_TIMEOUT_S", "10800"))


async def _run_bounded(build_coro, task: dict, needs_docker: bool) -> dict:
    limit = _DOCKER_WALL_TIMEOUT_S if needs_docker else _WALL_TIMEOUT_S
    try:
        return await asyncio.wait_for(_build_with_retry(build_coro, task),
                                      timeout=limit)
    except asyncio.TimeoutError:
        print(f"      ⏱ wall-clock timeout ({limit}s) {task.get('task_id','')}"
              f" — recorded as error; its container may need manual cleanup",
              flush=True)
        return {"task_id": task.get("task_id", ""), "response": "",
                "error": f"wall_clock_timeout_{limit}s"}


# Variants MUST be identical across arms (else C-B compares different task
# texts), across RESUME, and across backbone rows of Table 1 — so they are
# cached in ONE file shared by every model dir of this protocol base.
# MUTATIONS_PATH override: ablation arms (RESULTS_BASE=ablation/<arm>) must
# share the MAIN sweep's variant file, or every arm would mint its own variants
# and cross-arm comparisons stop being paired.
_MUTATIONS_PATH = Path(os.environ.get("MUTATIONS_PATH")
                       or (PROJECT_ROOT / "experiments_results" / _RESULTS_BASE
                           / "mutations.json"))
try:
    _MUTATIONS: dict = json.loads(_MUTATIONS_PATH.read_text())
except Exception:
    _MUTATIONS = {}


def _save_mutations() -> None:
    try:
        _MUTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MUTATIONS_PATH.write_text(json.dumps(_MUTATIONS, ensure_ascii=False))
    except Exception:
        pass


async def _get_variant(task: dict, iteration: int) -> str:
    key = f"{task.get('task_id','')}|{iteration}"
    if key in _MUTATIONS:
        return _MUTATIONS[key]
    desc = await _mutate_task_desc(task, iteration)
    _MUTATIONS[key] = desc
    _save_mutations()
    return desc


async def _mutate_task_desc(task: dict, iteration: int) -> str:
    """Reword the task for iteration k>=1 of an evolving chain. MUST preserve
    every fact, constraint, entity, and the correct answer — only the surface
    form changes. Falls back to the original on any failure."""
    desc = task.get("description", "")
    if not desc.strip():
        return desc
    # Long contexts (LoCoMo: the whole conversation + a question): reword ONLY
    # the question and keep the corpus verbatim — paraphrasing a 6000-char
    # truncation would drop the conversation tail and make the gold answer
    # unreachable. If no question marker is found, skip mutation entirely.
    if len(desc) > 4000:
        head, sep, tail = desc.rpartition("Question:")
        if sep and 0 < len(tail.strip()) <= 2000:
            sub = dict(task, description=tail.strip())
            new_q = await _mutate_task_desc(sub, iteration)
            return head + sep + " " + new_q if new_q != tail.strip() else desc
        return desc
    try:
        out = await _llm_call_notool(
            "You reword task descriptions. Preserve EVERY fact, number, entity, "
            "constraint, and the expected answer exactly; change only wording and "
            "sentence structure. Return ONLY the reworded task, no commentary.",
            f"Reword (variant {iteration}) the following task:\n\n{desc[:6000]}",
            timeout=60)
        new = (out.get("text") or "").strip()
        # sanity: refuse suspicious rewrites (too short / lost most content)
        if new and len(new) >= 0.5 * min(len(desc), 6000):
            return new
    except Exception:
        pass
    return desc


# What flows into MEMORY between iterations (reporting always stays gold):
#   gold  legacy — evaluate_task's score (uses the gold label). On frozen
#         same-task chains this leaks the correctness bit into the scored
#         instance; defensible only with ITER_MUTATE variants (graded practice).
#   env   environment-observable signal only: TB2's in-container test results
#         (the agent can run them itself — Reflexion-style). Other benchmarks
#         have no env signal and fall through to self-assessment.
#   self  deployable self-assessment: one reference-free LLM call estimating
#         P(correct). Never touches the gold label.
ITER_FEEDBACK = os.environ.get("ITER_FEEDBACK", "gold")


async def _feedback_score(task: dict, r: dict, ev: dict, benchmark: str) -> float:
    if ITER_FEEDBACK == "gold":
        return ev.get("score", 0.0)
    if ITER_FEEDBACK == "env" and benchmark == "terminal_bench_2":
        return ev.get("score", 0.0)  # tests run in the container: observable
    try:
        out = await _llm_call_notool(
            "You are a strict self-assessor. Given a task and an attempted "
            "answer, estimate the probability that the answer is correct. "
            "Return ONLY a number between 0 and 1.",
            f"Task:\n{task.get('description', '')[:3000]}\n\n"
            f"Attempt:\n{(r.get('response') or '')[:3000]}\n\n"
            "Probability correct:",
            timeout=45)
        m = re.search(r"(\d*\.?\d+)", out.get("text") or "")
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
    except Exception:
        pass
    return 0.5  # unknown — neutral, neither success nor failure


def _load_done_map(trace_path: Path) -> dict:
    """Read an existing trace.jsonl into {(group, task_id): record} for resume.
    Later records win, so a re-run of a task overrides an earlier partial."""
    done: dict = {}
    skipped_infra = 0
    if not trace_path.exists():
        return done
    try:
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # An infra row (error set, nothing produced) is NOT completed work:
            # counting it as done made RESUME skip — and thereby keep — the 770
            # APIUnavailable zero-rows forever. Leave it out so the task reruns;
            # later records win, so the rerun's row supersedes it downstream.
            if str(rec.get("error") or "").strip() \
                    and not str(rec.get("response") or "").strip():
                skipped_infra += 1
                continue
            done[(norm_group(rec.get("group", "")), rec.get("task_id", ""))] = rec
    except Exception as e:
        print(f"  [resume] failed to parse {trace_path}: {e}")
    if skipped_infra:
        print(f"  [resume] {skipped_infra} infra error-rows in {trace_path.parent.name} "
              "not counted as done — those tasks will rerun")
    return done


def _trace_iter_total(trace_path: Path) -> int:
    """The iter_total recorded in an existing trace (read from its first row), or 1 if
    absent/unreadable. All rows of one run share it, so the first row suffices. Used to
    detect a stale single-pass trace before a chain run."""
    try:
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            return int(json.loads(line).get("iter_total", 1))
    except Exception:
        pass
    return 1


# --- Evaluation ---

def _parse_pytest_counts(test_output: str) -> tuple[int, int]:
    """Return (passed, failed) from pytest output, robust to summary format.

    Handles "N passed", "N failed", "N error", "N passed, M failed" (either
    order), skips/xfails, and falls back to the per-file progress letters
    (".F.sE") when there is no summary banner. `error` and `xfailed` count as
    failures; `skipped`/`warning` are ignored. Sums tokens from the LAST summary
    banner so a "passed" mentioned in a traceback body can't inflate the count."""
    import re as _re
    text = test_output or ""
    passed = failed = 0
    banners = _re.findall(r'={3,}[^\n=]*?\d+\s+(?:passed|failed|error|skipped|xfailed|xpassed)[^\n=]*?={3,}',
                          text)
    target = banners[-1] if banners else text
    for num, kind in _re.findall(r'(\d+)\s+(passed|failed|error|errors|xfailed|xpassed)', target):
        n = int(num)
        if kind in ("passed", "xpassed"):
            passed += n
        elif kind in ("failed", "error", "errors"):
            failed += n
        # xfailed = expected-to-fail: neither passed nor failed (pytest semantics)
    if passed == 0 and failed == 0:
        # Fallback: per-file progress line, e.g. "../tests/test_x.py FFFF [100%]"
        for letters in _re.findall(r'\.py\s+([.FEsxX]{1,})', text):
            passed += letters.count(".")
            failed += letters.count("F") + letters.count("E")
    return passed, failed


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
        # Pre-agent baseline (terminus2 runs the suite once BEFORE the agent):
        # some tasks' tests partially pass on the untouched workspace, so a raw
        # pass ratio credits the agent for work it never did — the n=48 A-arm
        # had pytest_partial_1_5-style scores from runs that executed zero
        # commands. Score only the agent's marginal contribution.
        pre_passed, _pre_failed = _parse_pytest_counts(
            result.get("pre_test_output", "") or "")
        if test_passed:
            method = "docker_pytest_pass"
            if result.get("pre_test_passed"):
                # Tests were green before the agent ran: the pass carries no
                # signal about the agent. Score kept at 1.0 (official binary
                # semantics) but flagged so analysis can exclude the task.
                method += "_pretest_saturated"
            return {"score": 1.0, "em": 1.0, "method": method,
                    "pre_passed": pre_passed}

        # Parse the pytest summary robustly. The old regex required BOTH a
        # "passed" AND a "failed" count on one line, so it scored 0 for the
        # common "=== 4 failed in 0.5s ===" (failed-only) and "=== 4 passed
        # ===" (passed-only, harness missed) summaries. Sum each outcome token
        # independently from the LAST summary line (prefer the "==== ... ===="
        # banner; fall back to the per-file progress letters like ".F.F").
        passed, failed = _parse_pytest_counts(test_output)
        total = passed + failed
        if total > 0:
            gain = max(0, passed - pre_passed)
            denom = total - pre_passed
            if denom <= 0:
                # Everything that passes already passed pre-agent.
                return {"score": 0.0, "em": 0.0,
                        "method": f"pytest_saturated_pre{pre_passed}",
                        "pytest_passed": passed, "pytest_failed": failed,
                        "pytest_total": total, "pre_passed": pre_passed}
            partial_score = gain / denom
            em = 1.0 if partial_score >= 1.0 else 0.0
            method = (f"pytest_partial_{passed}_{failed}"
                      + (f"_pre{pre_passed}" if pre_passed else ""))
            return {
                "score": partial_score,
                "em": em,
                "method": method,
                "pytest_passed": passed,
                "pytest_failed": failed,
                "pytest_total": total,
                "pre_passed": pre_passed,
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
    # Credit the answer if EITHER the extracted span OR the full response matches
    # gold. A wrong-but-nonempty extraction must not shadow a correct full
    # response (the old `extracted or response` did exactly that, e.g. gold
    # "Claude Shannon" present in the response but scored 0). The single-token
    # gold guard inside exact_match (word-boundary + <=20 words) keeps the
    # full-response path from false-positiving on a stray number.
    em = max(exact_match(extracted, expected), exact_match(response, expected))

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
        # A chain run (ITER_CHAIN>1) must not resume from a stale single-pass trace:
        # done_map keys on (group, task_id) only, so an old iter_total=1 A-arm row would
        # be reused and leave A single-pass while B/C run the full chain. If the existing
        # trace was written with a different iter_total, it is from an incompatible run --
        # clear it and start fresh rather than silently poison the A arm.
        stale = RESUME and ITER_CHAIN > 1 and _trace_iter_total(trace_path) != ITER_CHAIN
        if RESUME and not stale:
            done_map = _load_done_map(trace_path)
            print(f"  RESUME: {len(done_map)} completed (group,task) pairs will be skipped")
        else:
            if stale:
                print(f"  [resume] existing trace has iter_total != {ITER_CHAIN}; "
                      "clearing stale trace and starting fresh")
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
    run_fn_a = BASELINE_RUNNER.get(benchmark)

    # A/B/C are unified across every benchmark by the evomem_bridge (see the
    # "Unified A/B/C" block below): A runs the baseline runner with no memory;
    # B/C wrap that SAME baseline with cross-task patch memory (B plain, C
    # effectiveness-weighted + grounded). The old per-benchmark "controlled"
    # runners / self-consistency paths are retired — they made the arms
    # inconsistent across benchmarks and (on the agentic ones) injected nothing.

    if not run_fn_a:
        print(f"  ERROR: No runner for benchmark '{benchmark}'")
        return {"error": f"no_runner_{benchmark}"}

    print(f"\n  Testing {len(test_tasks)} tasks x 3 groups (A/B/C)...")
    # Shared, loop-bound semaphores (created in main): one global cap for all
    # heavy work, plus a tighter sub-cap for Docker-backed benchmarks.
    global_sem = _SEMS["global"]
    docker_sem = _SEMS.get("docker")
    needs_docker = benchmark in DOCKER_BENCHMARKS

    async def _run_group(label: str, group_key: str, tasks: list[dict], build_coro,
                         mem=None):
        """Run tasks concurrently. Evaluate + trace-log each task as it completes so
        partial results survive crashes. With RESUME=1, already-traced tasks are
        reconstructed from the trace instead of re-run; every fresh task is retried
        with backoff on transient API failures.

        `mem` (B/C cross-task memory, None for A): after a task is evaluated its
        patch is recorded into `mem` WITH the task's real score — so C's
        effectiveness-weighted retrieval sees scored patches, and resumed runs
        rebuild memory from the trace instead of starting empty."""
        total = len(tasks)
        results = [None] * total
        evals = [None] * total

        async def _wrap(i: int, task: dict):
            tid = task.get("task_id", "")
            # ── Resume: reconstruct already-completed tasks from trace ──
            prev = done_map.get((group_key, tid))
            # A chain counts as done only if its FINAL iteration was traced. A
            # crash mid-chain would otherwise freeze this task at an early,
            # memory-less iteration forever (biasing B/C down); re-run it fully.
            if (prev is not None and ITER_CHAIN > 1
                    and int(prev.get("iteration", ITER_CHAIN - 1)) < ITER_CHAIN - 1):
                prev = None
            if prev is not None:
                r = {"task_id": tid, "response": prev.get("response", ""),
                     "expected": prev.get("expected", ""),
                     "_aug_prompt": prev.get("augmented_prompt", ""),
                     "_sc_responses": prev.get("_sc_responses", []),
                     "_resumed": True}
                ev = {"score": prev.get("score", 0.0), "em": prev.get("em", 0.0),
                      "method": prev.get("method", "resumed")}
                # Rebuild B/C memory from the resumed result (else a resumed run
                # would retrieve against an empty store).
                if mem is not None:
                    try:
                        _fb = prev.get("fb_score")
                        if _fb is None:
                            _fb = ev.get("score", 0.0) if ITER_FEEDBACK == "gold" else 0.5
                        await mem.record(task, r, float(_fb))
                    except Exception:
                        pass
                print(f"    [{label}] {i+1}/{total} ⟳ {tid} (resumed EM={ev['em']:.0%})",
                      flush=True)
                return i, r, ev
            # ── Iteration chain: run the task ITER_CHAIN times in sequence.
            #    For B/C, each iteration's mem.record() adds a patch to the
            #    chain, and the next iteration's chain-scoped inject() retrieves
            #    it — so memory threads across iterations of the SAME task. K=1
            #    (default) reduces to the ordinary one-shot run. The main table
            #    uses the final iteration (post-memory); the per-iteration trace
            #    rows feed chain-level accuracy. ──
            last_r, last_ev, prof = None, None, {}
            for _it in range(ITER_CHAIN):
                cur_task = task
                if ITER_MUTATE and _it >= 1:
                    _mdesc = await _get_variant(task, _it)
                    if _mdesc != task.get("description", ""):
                        cur_task = dict(task, description=_mdesc)
                start_task_profile()
                _t0 = _perf()
                # Acquire the scarce docker slot BEFORE the global slot. The other
                # order let a docker task hold a global slot while queued for docker,
                # so TB2 could pin all global slots waiting on the 2 docker slots and
                # starve the non-docker benchmarks (LoCoMo/GAIA/GAIA2 got 0 progress).
                # docker-then-global is a consistent lock order, so no deadlock.
                if needs_docker and docker_sem is not None:
                    async with docker_sem:
                        async with global_sem:
                            r = await _run_bounded(build_coro, cur_task, True)
                else:
                    async with global_sem:
                        r = await _run_bounded(build_coro, cur_task, False)
                prof = summarize(_perf() - _t0, read_task_profile())
                if isinstance(r, dict):
                    r["_prof"] = prof
                ev = await evaluate_task(r, benchmark)
                # Record WITH the real score so the NEXT iteration of this chain
                # can retrieve it (and C weights retrieval by it).
                fb = None
                if mem is not None and isinstance(r, dict):
                    try:
                        fb = await _feedback_score(cur_task, r, ev, benchmark)
                        await mem.record(cur_task, r, fb)
                    except Exception:
                        pass
                expected = task.get("expected", r.get("expected", ""))
                aug = r.get("_aug_prompt", "")
                exec_mode = (r.get("execution_mode")
                             or r.get("_within_task_patch_mode") or "default")
                _meta = task.get("metadata") or {}
                _trace.log(benchmark=benchmark, group=group_key, phase="test",
                           task_id=r.get("task_id", tid),
                           task_desc=cur_task.get("description", ""),
                           augmented_prompt=aug,
                           response=r.get("response", ""), expected=expected,
                           score=ev.get("score", 0.0),
                           extra={"em": ev.get("em", 0.0),
                                  "method": ev.get("method", ""),
                                  "execution_mode": exec_mode,
                                  "error": str(r.get("error") or "")[:200],
                                  # category/type and difficulty (when provided)
                                  "category": str(_meta.get("category") or _meta.get("config")
                                                  or _meta.get("task_type")
                                                  or _meta.get("type") or ""),
                                  "level": str(_meta.get("level") or _meta.get("difficulty") or ""),
                                  "patch_injected": bool(aug),
                                  "aug_len": len(aug or ""),
                                  # w_c of the served entries. get_experience_weight
                                  # cold-starts at 1.0 until an entry has >=2 measured
                                  # deltas and a 3-iteration chain gives at most 2, so
                                  # wc_active==0 here means the paper's
                                  # effectiveness-weighted retrieval never actually
                                  # fired and C was ranking on similarity alone.
                                  **({k: v for k, v in (r.get("_wc") or {}).items()}
                                     if isinstance(r.get("_wc"), dict) else {}),
                                  # iteration index along the chain + chain length,
                                  # for chain-level (all-iterations-correct) accuracy.
                                  "iteration": _it,
                                  "iter_total": ITER_CHAIN,
                                  "mutated": bool(ITER_MUTATE and _it >= 1 and cur_task is not task),
                                  "fb_score": fb, "fb_mode": ITER_FEEDBACK,
                                  "code_rev": _CODE_REV,
                                  # Who judged arm C's entries. Self-curation
                                  # (critic == backbone) and cross-curation
                                  # (designated stronger critic) are different
                                  # method configs and must never be pooled as
                                  # one: on GAIA the false-endorsement rate runs
                                  # 36% for DeepSeek-v4-pro vs 90% for
                                  # Llama-3.3-70B, and C-B flips sign with it.
                                  "critic_model": _CRITIC_MODEL,
                                  "c_meta": _C_META,
                                  "c_policy": _C_POLICY,
                                  "score_provenance": _SCORE_PROVENANCE,
                                  "judge_model": _JUDGE_MODEL,
                                  # TB2 loop visibility: how far the agent loop
                                  # got and why it ended — distinguishes "agent
                                  # flailed" from "agent never engaged".
                                  **{k: r[k] for k in
                                     ("turns_used", "end_reason",
                                      "cmds_executed", "pre_test_passed")
                                     if isinstance(r, dict) and k in r},
                                  **{f"prof_{k}": v for k, v in prof.items()}})
                # ── FINAL-ITERATION resampling for the raw-patch arm
                # (PASSK_FINAL=k): after the chain's last iteration, draw k-1
                # extra independent solves of the SAME variant with the SAME
                # frozen memory state (no record between samples), logged as
                # groups raw_patch_passk_s1..s{k-1}; the normal final row is
                # sample 0. This is the fair counterpart to the no-memory
                # pass@k control: B keeps its chain advantage and spends the
                # extra calls on resampling, exactly the budget C spends on
                # curation. Set PASSK_FINAL from the START of the B run —
                # RESUME skips completed chains and will not backfill samples.
                _PKF = int(os.environ.get("PASSK_FINAL", "0"))
                if (_PKF > 1 and group_key == "raw_patch"
                        and _it == ITER_CHAIN - 1):
                    for _s in range(1, _PKF):
                        if needs_docker and docker_sem is not None:
                            async with docker_sem:
                                async with global_sem:
                                    _rs = await _run_bounded(build_coro, cur_task, True)
                        else:
                            async with global_sem:
                                _rs = await _run_bounded(build_coro, cur_task, False)
                        _evs = await evaluate_task(_rs, benchmark)
                        _trace.log(
                            benchmark=benchmark, group=f"raw_patch_passk_s{_s}",
                            phase="test", task_id=_rs.get("task_id", tid),
                            task_desc=cur_task.get("description", ""),
                            augmented_prompt=_rs.get("_aug_prompt", ""),
                            response=_rs.get("response", ""), expected=expected,
                            score=_evs.get("score", 0.0),
                            extra={"em": _evs.get("em", 0.0),
                                   "iteration": _it, "iter_total": ITER_CHAIN,
                                   "sample_idx": _s,
                                   "patch_injected": bool(_rs.get("_aug_prompt")),
                                   "aug_len": len(_rs.get("_aug_prompt") or ""),
                                   **({k: v for k, v in (_rs.get("_wc") or {}).items()}
                                      if isinstance(_rs.get("_wc"), dict) else {}),
                                   "fb_mode": ITER_FEEDBACK,
                                   "code_rev": _CODE_REV,
                                   "critic_model": _CRITIC_MODEL,
                                   "c_meta": _C_META,
                                   "c_policy": _C_POLICY,
                                   "score_provenance": _SCORE_PROVENANCE,
                                   "judge_model": _JUDGE_MODEL})
                last_r, last_ev = r, ev
            r, ev = last_r, last_ev
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
    #   A  Vanilla  : no memory.
    #   B  PatchMem : naive cross-task patch memory — retrieve + inject prior
    #                 task responses verbatim (the \patchmem baseline).
    #   C  Curator  : real curation — refined experiences (LLM reviewer + cross-
    #                 agent critic + forced enrichment), effectiveness-weighted
    #                 retrieval. Injects reusable lessons, not B's raw answers.
    # B and C each keep their own cross-task memory; A keeps none.
    mem_b = BenchmarkMemory(benchmark, "B") if "B" in _ARMS else None
    mem_c = CuratedMemory(benchmark) if "C" in _ARMS else None

    # Equal-budget control (ablation `ctrl_reprompt` arm): wrap the no-memory
    # baseline with m extra self-refinement calls so the A slot spends the same
    # write-time budget as C's refine+critic — separating WHERE the extra compute
    # goes from HOW MUCH there is. Off unless REPROMPT_CONTROL=1.
    if os.environ.get("REPROMPT_CONTROL") == "1":
        _m_reprompt = int(os.environ.get("REPROMPT_CALLS", "2"))
        _run_fn_base = run_fn_a

        async def run_fn_a(task, exp, group, _base=_run_fn_base, _m=_m_reprompt):
            r = await _base(task, exp, group)
            try:
                resp = (r.get("response") or "").strip() if isinstance(r, dict) else ""
                for _ in range(_m):
                    if not resp:
                        break
                    out = await _llm_call_notool(
                        "You critique and improve answers. Return ONLY the improved final answer.",
                        f"Task:\n{task.get('description', '')[:2000]}\n\n"
                        f"Current answer:\n{resp[:2000]}\n\n"
                        "Critique the answer for errors or omissions, then return "
                        "the improved final answer only.",
                        timeout=90)
                    improved = (out.get("text") or "").strip()
                    if improved:
                        resp = improved
                if isinstance(r, dict) and resp:
                    r["response"] = resp
                    r["execution_mode"] = "reprompt_control"
            except Exception:
                pass
            return r

    results_a = results_b = results_c = []
    evals_a = evals_b = evals_c = []
    # pass@k variance/compute control: k INDEPENDENT no-memory samples per
    # task, one group per sample (no_mem_passk_s<i>) so RESUME's (group,task)
    # bookkeeping stays untouched. Single-shot by design — refuse chains.
    _PASSK = int(os.environ.get("PASSK", "0"))
    if _PASSK > 1 and ITER_CHAIN != 1:
        raise SystemExit("PASSK is a single-shot control: run it with "
                         "ITER_CHAIN=1 ARMS=A PASSK=<k> (separate invocation).")
    if "A" in _ARMS and _PASSK > 1:
        for _s in range(_PASSK):
            print(f"    [A·pass@k] no-memory sample {_s+1}/{_PASSK}...", flush=True)
            _r, _e = await _run_group("A", f"no_mem_passk_s{_s}", test_tasks,
                                      lambda t: run_fn_a(t, "", "A"))
            results_a, evals_a = _r, _e     # last sample stands in for reporting
    elif "A" in _ARMS:
        print(f"    [A] Vanilla (no memory)...", flush=True)
        results_a, evals_a = await _run_group("A", "no_mem", test_tasks,
                                              lambda t: run_fn_a(t, "", "A"))
    if "B" in _ARMS:
        print(f"    [B] PatchMem (naive cross-task patch memory)...", flush=True)
        results_b, evals_b = await _run_group("B", "raw_patch", test_tasks,
            lambda t: solve_with_memory(run_fn_a, t, mem_b, "B"), mem=mem_b)
    if "C" in _ARMS:
        print(f"    [C] Curator (refined experiences + effectiveness-weighted retrieval)...", flush=True)
        results_c, evals_c = await _run_group("C", "curated_patch", test_tasks,
            lambda t: solve_with_memory(run_fn_a, t, mem_c, "C"), mem=mem_c)

    # External baseline memory systems (Wen 07/06: 2-3 recent methods with
    # ready modules, run WITHOUT the raw_patch arm, pairing against the
    # existing no_mem rows). EXTERNAL_MEMS="mem0,amem"; each adapter exposes
    # the same inject()/record() interface, so the protocol (mutated chains,
    # self feedback, record-after-eval) is IDENTICAL to B/C.
    _ext_scores = {}
    for _ext in [x.strip().lower() for x in
                 os.environ.get("EXTERNAL_MEMS", "").split(",") if x.strip()]:
        from scripts.latest.baseline_memories import make_external_memory
        _mem_x = make_external_memory(_ext, benchmark)
        print(f"    [{_ext}] external baseline memory...", flush=True)
        _rx, _ex = await _run_group(_ext.upper(), _ext, test_tasks,
            lambda t, _m=_mem_x, _g=_ext: solve_with_memory(run_fn_a, t, _m, _g),
            mem=_mem_x)
        _ext_scores[_ext] = _ex

    # Flat evals + per-group scores for the arms that actually ran.
    scores = {g: e for g, e in (("no_mem", evals_a), ("raw_patch", evals_b),
                                ("curated_patch", evals_c)) if e}
    scores.update({g: e for g, e in _ext_scores.items() if e})
    all_evals = [e for evs in scores.values() for e in evs]

    report = {}
    for group, evals in scores.items():
        valid = [e["score"] for e in evals if e.get("score") is not None]
        ems = [e.get("em", 0.0) for e in evals]
        report[group] = {
            "avg_score": sum(valid) / len(valid) if valid else 0.0,
            "em": sum(ems) / len(ems) if ems else 0.0,
            "n": len(valid),
        }

    # Native metric per benchmark: GAIA2 is scored by soft recall over oracle
    # events (the continuous `avg_score`); GAIA/LoCoMo by exact match (`em`).
    # The stale blanket "EM" label made the GAIA2 `em` column (a meaningless hard
    # threshold on soft recall) look like the headline number. Both fields stay
    # stored; only the label and the printed/delta field follow the native metric.
    metric_name, metric_field = (("soft_recall", "avg_score") if benchmark == "gaia2"
                                 else ("EM", "em"))
    print(f"\n  Results ({benchmark}, model={MODEL}):")
    for _g, _r in report.items():
        print(f"    {_g:12s}: {metric_name}={_r[metric_field]:.1%}")
    if {"no_mem", "raw_patch", "curated_patch"} <= set(report):
        delta_ac = report['curated_patch'][metric_field] - report['no_mem'][metric_field]
        delta_bc = report['curated_patch'][metric_field] - report['raw_patch'][metric_field]
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
            "append_only_manifested_patch_store",
            "additive_curation_refine_critique_enrich",
            "effectiveness_weighted_chain_scoped_retrieval",
        ],
        "results": report,
        "delta_curated_vs_no_mem": delta_ac,   # C - A
        "delta_curated_vs_raw_patch": delta_bc,  # C - B
    }

    if benchmark == "gaia2":
        config_scores = {}
        for i, task in enumerate(test_tasks):
            config = (task.get("metadata") or {}).get("config", "unknown")
            if config not in config_scores:
                config_scores[config] = {"no_mem": [], "raw_patch": [], "curated_patch": []}
            config_scores[config]["no_mem"].append(all_evals[i * 3])
            config_scores[config]["raw_patch"].append(all_evals[i * 3 + 1])
            config_scores[config]["curated_patch"].append(all_evals[i * 3 + 2])
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
                level_scores[level] = {"no_mem": [], "raw_patch": [], "curated_patch": []}
            level_scores[level]["no_mem"].append(all_evals[i * 3])
            level_scores[level]["raw_patch"].append(all_evals[i * 3 + 1])
            level_scores[level]["curated_patch"].append(all_evals[i * 3 + 2])
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
            a = per_level_report[level]["no_mem"]
            c = per_level_report[level]["curated_patch"]
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
    # ── Knob banner: echo EVERY method/protocol knob at launch (Mem0-style
    # config echo). One glance at the log answers "what exactly ran?" — the
    # stale-checkout incident (C rerun on pre-v2 code) would have been caught
    # at launch, not after 128 wasted solves, had this existed then.
    _knobs = {
        "code_rev": _CODE_REV, "critic_model": _CRITIC_MODEL, "c_meta": _C_META,
        "score_provenance": _SCORE_PROVENANCE,
        "ARMS": ",".join(sorted(_ARMS)),
        "ITER_CHAIN": ITER_CHAIN, "ITER_MUTATE": int(ITER_MUTATE),
        "ITER_FEEDBACK": os.environ.get("ITER_FEEDBACK", "gold"),
        "RESUME": int(RESUME), "TASK_LIMIT": _TASK_N,
        "GAIA2_TASK_LIMIT": os.environ.get("GAIA2_TASK_LIMIT", "(default)"),
        "GAIA2_SPLIT_WEIGHTS": os.environ.get("GAIA2_SPLIT_WEIGHTS", "(uniform)"),
        "C_INJECT_BUDGET_CH": os.environ.get("C_INJECT_BUDGET_CH", "900"),
        "C_CRITIC_GATE": os.environ.get("C_CRITIC_GATE", "5"),
        "C_RAW_FALLBACK": os.environ.get("C_RAW_FALLBACK", "1"),
        "C_PAGE_KEEP": os.environ.get("C_PAGE_KEEP", "0"),
        "C_NO_PARTITION": os.environ.get("C_NO_PARTITION", "0"),
        "C_USE_CRITIC": os.environ.get("C_USE_CRITIC", "1"),
        "C_USE_ENRICH": os.environ.get("C_USE_ENRICH", "1"),
        "W_C_DISABLED": os.environ.get("W_C_DISABLED", "0"),
        "REPROMPT_CONTROL": os.environ.get("REPROMPT_CONTROL", "0"),
        "PASSK": os.environ.get("PASSK", "0"),
        "PASSK_FINAL": os.environ.get("PASSK_FINAL", "0"),
        "EXTERNAL_MEMS": os.environ.get("EXTERNAL_MEMS", "(none)"),
        "RESULTS_BASE": _RESULTS_BASE,
    }
    print("  knobs: " + " ".join(f"{k}={v}" for k, v in _knobs.items()))
    print("=" * 70)

    print("\n  Probing API availability...", flush=True)
    # When running with a local vLLM/OpenAI backend (LLM_PROVIDER in
    # vllm/openrouter/openai/oai/openai_compatible), the CodeBuddy SDK probe is
    # irrelevant. Skip it and trust the endpoint directly.
    _llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
    if _llm_provider in ("vllm", "openrouter", "openai", "oai", "openai_compatible"):
        print("  Skipping probe (using local/OpenAI backend: %s)." % _llm_provider, flush=True)
        api_ok = True
    # Detect local endpoint via OPENAI_* vars even when LLM_PROVIDER didn't
    # make it through the env chain (nohup/ceph edge cases).
    elif os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"):
        print("  Skipping probe (OPENAI_API_BASE/OPENAI_BASE_URL detected).", flush=True)
        os.environ["LLM_PROVIDER"] = "vllm"
        api_ok = True
    else:
        api_ok = await probe_api_available()
        if not api_ok:
            print("  DeepSeek V4 Pro API is NOT available. Aborting.", flush=True)
            return
    print("  API is ready.", flush=True)

    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    completed_benchmarks = checkpoint.get("completed_benchmarks", {})
    if completed_benchmarks:
        print(f"\n  Resuming from checkpoint: {list(completed_benchmarks.keys())} already done.", flush=True)

    # terminal_bench_2 RETIRED (2026-07-09): replaced by tau2-bench
    # (scripts/latest/tau2_bridge.py, routed automatically by
    # run_all_benchmarks.sh). The old harbor harness is archived under
    # scripts/latest/obsolete/. The simplified Terminus-2-style loop below stays
    # only as a hard-guarded BENCHMARKS=terminal_bench_2 override and must never
    # feed paper numbers.
    BENCHMARKS_TO_RUN = [
        "gaia", "gaia2", "locomo"
    ]
    if os.environ.get("BENCHMARKS", "").strip():
        BENCHMARKS_TO_RUN = ["gaia", "gaia2", "terminal_bench_2", "locomo"]
    # Benchmark selection (ablation driver / partial reruns): BENCHMARKS=locomo,gaia2
    _bf = os.environ.get("BENCHMARKS", "").strip()
    if _bf:
        _keep = {b.strip() for b in _bf.split(",") if b.strip()}
        # HARD GUARD: TB2 through this runner is the RETIRED simplified loop.
        # An explicit BENCHMARKS list quietly revived it once and burned 2h+801
        # trace rows on a box where docker pulls fail. TB2 is retired entirely;
        # the current agentic benchmark is tau2 (Docker-free).
        if "terminal_bench_2" in _keep and os.environ.get("ALLOW_RETIRED_TB2") != "1":
            raise SystemExit(
                "[latest_runner] REFUSING to run terminal_bench_2 — it is RETIRED.\n"
                "  The agentic benchmark is now tau2 (Docker-free):\n"
                "    bash scripts/latest/run_all_benchmarks.sh <MODEL>\n"
                "    # or: python scripts/latest/tau2_bridge.py --arm A|B|C ...\n"
                "  (the old harbor bridge is archived under scripts/latest/obsolete/;\n"
                "   override for debugging only: ALLOW_RETIRED_TB2=1)")
        BENCHMARKS_TO_RUN = [b for b in BENCHMARKS_TO_RUN if b in _keep]
    print(f"\n  Loading benchmarks: {BENCHMARKS_TO_RUN}...")
    benchmarks = {}
    for name in BENCHMARKS_TO_RUN:
        config = {"name": name, "num_samples": TASK_LIMITS[name]}
        if name == "gaia2":
            config["scenario_dir"] = os.environ.get(
                "GAIA2_SCENARIO_DIR",
                "/tmp/harbor-datasets/datasets/gaia2-cli",
            )
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
            print(f"  {name:>20}: A={r.get('no_mem',{}).get('em',0):.1%}, B={r.get('raw_patch',{}).get('em',0):.1%}, C={r.get('curated_patch',{}).get('em',0):.1%}")
        elif isinstance(report, dict) and "error" in report:
            print(f"  {name:>20}: ERROR — {report['error'][:80]}")
        else:
            print(f"  {name:>20}: {report}")
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())