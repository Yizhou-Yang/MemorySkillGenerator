#!/usr/bin/env python3
"""SkillForge V6 Unified Benchmark Runner — AI-Refined Experience Evolution"""
import asyncio
import json
import os
import sys
import time
import copy
import shutil
from pathlib import Path

sys.path.insert(0, '/data/home/yizhouyang/workspace/SkillForge/src')
sys.path.insert(0, '/data/home/yizhouyang/workspace/SkillForge')

# CODEBUDDY_API_KEY must be set in environment
os.environ['CODEBUDDY_INTERNET_ENVIRONMENT'] = 'ioa'
os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'hy3-preview-ioa'

from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage, ToolUseBlock
from v6 import (SkillForgeV6, ExperienceLibrary, Experience,
                analyze_execution, build_augmented_prompt,
                format_success_experience, format_failure_experience,
                ai_review_experience)

# ─── Config ───────────────────────────────────────────────────────────────
MODEL = "hy3-preview-ioa"
CONCURRENCY = 20
TASK_TIMEOUT = 300  # 5 min per task
RESULTS_DIR = "/data1/benchmarks/unified_v6_results"

# Task limits per benchmark (matched to ~similar runtime)
TASK_LIMITS = {
    "gaia2": 50,       # 25 train + 25 test, ~3min/task with tools
    "locomo": 50,      # 25 train + 25 test, ~15s/task (QA)
    "gaia": 50,        # 25 train + 25 test, ~15s/task (QA)
    "alfworld": 40,    # 20 train + 20 test, ~15s/task (QA via HF)
    "swebench": 20,    # 10 train + 10 test, ~5min/task with Docker
}

# ─── LLM Helper (for AI Review) ──────────────────────────────────────────

def llm_review_fn(prompt: str) -> str:
    """Call hy3-preview-ioa for AI review (single-turn, sync via thread)."""
    import concurrent.futures

    async def _call():
        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=2, cwd="/tmp"
        )
        result = ""
        try:
            async with asyncio.timeout(60):
                async for msg in query(prompt=prompt, options=opt):
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if hasattr(block, 'text') and block.text:
                                result += block.text
                        if result:
                            break
        except Exception:
            pass
        return result

    # Run in a new event loop in a thread (since we're called from sync context
    # inside record_experience which is called from async code)
    def _run_in_thread():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_call())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        return future.result(timeout=90)

# ─── Unified Task Runner ──────────────────────────────────────────────────

async def run_task_generic(task: dict, experience_section: str = "",
                           benchmark: str = "gaia", group: str = "train") -> dict:
    """Run a single task using CodeBuddy SDK agent loop."""
    task_id = task["task_id"]
    description = task["description"]
    expected = task.get("expected", "")

    if benchmark == "gaia2":
        # Gaia2 uses CLI tools via Bash
        return await _run_gaia2_task(task, experience_section, group=group)

    # For LoCoMo, GAIA, ALFWorld (HF versions): single-turn LLM call
    system = "You are a helpful assistant. Answer the question directly and concisely."
    if experience_section:
        system += f"\n\n{experience_section}"

    prompt = f"[System]\n{system}\n\n{description}"

    opt = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions", model=MODEL, max_turns=2, cwd="/tmp"
    )

    result = {"task_id": task_id, "expected": expected, "response": "", "error": None,
              "time_cost": 0, "augmented": bool(experience_section)}
    t0 = time.time()

    try:
        async with asyncio.timeout(TASK_TIMEOUT):
            async for msg in query(prompt=prompt, options=opt):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, 'text') and block.text:
                            if '429' in block.text and '额度' in block.text:
                                result["error"] = "429"
                                break
                            result["response"] += block.text
                    if result["response"] or result["error"]:
                        break
    except TimeoutError:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)[:200]

    result["time_cost"] = time.time() - t0
    return result

async def _run_gaia2_task(task: dict, experience_section: str, group: str = "train") -> dict:
    """Run Gaia2 task (CLI tool calling via CodeBuddy agent)."""
    import subprocess
    scenario_path = task["metadata"]["scenario_path"]
    scenario = json.load(open(scenario_path))
    task_id = task["task_id"]

    # Get task prompt from scenario
    task_prompt = ""
    for e in scenario['events']:
        if e.get('event_type') == 'USER':
            for a in e['action'].get('args', []):
                if a['name'] == 'content':
                    task_prompt = a['value']
                    break

    # Tool list
    app_to_cli = {'Calendar': 'calendar', 'Contacts': 'contacts', 'Emails': 'emails',
                  'Messages': 'messages', 'Chats': 'chats', 'RentAFlat': 'rent-a-flat',
                  'City': 'city', 'Cabs': 'cabs', 'Shopping': 'shopping'}
    tools = []
    for app in scenario.get('apps', []):
        name = app.get('name', '')
        if name in app_to_cli:
            tools.append(f"- `{app_to_cli[name]}` — {name}")
    tool_list = "\n".join(tools)

    # Init state — ISOLATED per group to avoid cross-contamination
    PYTHON = "/data/home/yizhouyang/.workbuddy/binaries/python/versions/3.14.3/bin/python3.14"
    GAIA2_BIN = "/data/home/yizhouyang/.workbuddy/binaries/python/versions/3.14.3/bin"
    cwd = f"/tmp/gaia2_unified/{group}_{task_id}"
    state_dir = os.path.join(cwd, "state")
    # Clean previous state to ensure fresh start
    if os.path.exists(cwd):
        shutil.rmtree(cwd)
    os.makedirs(state_dir, exist_ok=True)

    init_r = subprocess.run(
        f"{PYTHON} -m gaia2_cli.init_cmd --scenario {scenario_path} --state-dir {state_dir}",
        shell=True, capture_output=True, text=True, timeout=30
    )
    if init_r.returncode != 0:
        return {"task_id": task_id, "expected": "", "response": "",
                "error": f"init: {init_r.stderr[:100]}", "time_cost": 0, "augmented": False,
                "group": group, "actions": []}

    system_prompt = f"""# Gaia2 Agent
You are an AI assistant operating inside a Gaia2 environment.

## Available CLI tools:
{tool_list}

Every command MUST be prefixed with `GAIA2_STATE_DIR={state_dir}`.
Run any command with `--help` for details.

## Rules
1. Follow instructions exactly.
2. Be resourceful — use tools to gather missing information.
3. Complete ALL steps of the task.

{experience_section}"""

    full_prompt = f"{system_prompt}\n\n## Your Task\n\n{task_prompt}"

    opt = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions", max_turns=30, model=MODEL, cwd=cwd,
        env={"CODEBUDDY_API_KEY": os.environ["CODEBUDDY_API_KEY"],
             "CODEBUDDY_INTERNET_ENVIRONMENT": "ioa",
             "PATH": f"{GAIA2_BIN}:/usr/local/bin:/usr/bin:/bin",
             "GAIA2_STATE_DIR": state_dir},
    )

    result = {"task_id": task_id, "expected": "", "response": "", "error": None,
              "time_cost": 0, "augmented": bool(experience_section), "actions": [],
              "group": group}
    t0 = time.time()

    try:
        async with asyncio.timeout(600):
            async for msg in query(prompt=full_prompt, options=opt):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            result["actions"].append({"tool": block.name, "input": block.input})
    except TimeoutError:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)[:200]

    result["time_cost"] = time.time() - t0
    result["total_actions"] = len(result["actions"])
    return result

# ─── Benchmark Loaders ────────────────────────────────────────────────────

def load_gaia2_tasks(n: int = 50) -> list[dict]:
    """Load Gaia2 tasks from scenarios_full."""
    scenarios_dir = "/data1/benchmarks/gaia2/scenarios_full"
    tasks = []
    for f in sorted(os.listdir(scenarios_dir))[:n]:
        sp = os.path.join(scenarios_dir, f)
        scenario = json.load(open(sp))
        task_prompt = ""
        for e in scenario['events']:
            if e.get('event_type') == 'USER':
                for a in e['action'].get('args', []):
                    if a['name'] == 'content':
                        task_prompt = a['value']
        tasks.append({
            "task_id": f.replace('.json', ''),
            "description": task_prompt,
            "expected": "",  # evaluated via events.jsonl
            "context": "",
            "metadata": {"scenario_path": sp, "benchmark": "gaia2"},
        })
    return tasks

def load_hf_tasks(benchmark: str, n: int = 50) -> list[dict]:
    """Load tasks from HuggingFace via BenchmarkLoader."""
    from benchmarks.loader import BenchmarkLoader
    loader = BenchmarkLoader({"name": benchmark, "num_samples": n})
    return loader.load()

def load_swebench_tasks(n: int = 20) -> list[dict]:
    """Load SWE-bench Verified tasks (subset with available Docker images)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    except Exception:
        # Fallback: load from local
        ds = json.load(open("/data1/benchmarks/swe-bench/SWE-bench/swebench/resources/swebench-og/django__django/10097/task.json"))
        return []

    # Filter to instances with available Docker images
    available_images = set()
    img_list = "/data1/benchmarks/swe-bench/image_list.txt"
    if os.path.exists(img_list):
        with open(img_list) as f:
            for line in f:
                # Extract instance_id from image name
                # Format: ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest
                parts = line.strip().split(".")
                if len(parts) >= 5:
                    instance_id = ".".join(parts[4:]).replace(":latest", "")
                    available_images.add(instance_id)

    tasks = []
    for row in ds:
        instance_id = row.get("instance_id", "")
        # Check if Docker image is available
        img_key = instance_id.replace("/", "__").replace("-", "_")
        if available_images and img_key not in available_images:
            continue

        problem_statement = row.get("problem_statement", "")
        patch = row.get("patch", "")

        tasks.append({
            "task_id": f"swebench_{instance_id}",
            "description": f"Fix the following issue in the codebase:\n\n{problem_statement[:2000]}",
            "expected": patch[:500],  # Gold patch (for scoring)
            "context": "",
            "metadata": {
                "instance_id": instance_id,
                "repo": row.get("repo", ""),
                "base_commit": row.get("base_commit", ""),
                "benchmark": "swebench",
            },
        })
        if len(tasks) >= n:
            break

    return tasks

# ─── Evaluation ───────────────────────────────────────────────────────────

def evaluate_task(result: dict, benchmark: str) -> dict:
    """Compute score for a single task result."""
    if benchmark == "gaia2":
        return _evaluate_gaia2(result)

    expected = result.get("expected", "").strip().lower()
    response = result.get("response", "").strip().lower()

    if not expected or not response:
        return {"score": 0.0, "method": "empty"}

    # Exact match (substring)
    em = 1.0 if expected in response or response in expected else 0.0

    # Token F1
    exp_tokens = set(expected.split())
    resp_tokens = set(response.split())
    if exp_tokens and resp_tokens:
        precision = len(exp_tokens & resp_tokens) / len(resp_tokens)
        recall = len(exp_tokens & resp_tokens) / len(exp_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        f1 = 0.0

    return {"score": max(em, f1), "em": em, "f1": f1, "method": "token_f1"}

def _evaluate_gaia2(result: dict) -> dict:
    """Evaluate Gaia2 task using events.jsonl (exact action matching)."""
    task_id = result.get("task_id", "")
    group = result.get("group", "train")
    state_dir = f"/tmp/gaia2_unified/{group}_{task_id}/state"
    events_file = os.path.join(state_dir, "events.jsonl")

    # Load agent events
    agent_events = []
    if os.path.exists(events_file):
        with open(events_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        agent_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Load oracle from scenario
    scenario_path = f"/data1/benchmarks/gaia2/scenarios_full/{task_id}.json"
    if not os.path.exists(scenario_path):
        return {"score": 0.0, "method": "no_scenario"}

    scenario = json.load(open(scenario_path))
    oracle_events = [e for e in scenario["events"] if e.get("event_type") == "AGENT"]

    if not oracle_events:
        return {"score": 0.0, "method": "no_oracle"}

    # Exact matching: oracle (app.function + args) vs agent (app + fn + args)
    oracle_details = []
    for e in oracle_events:
        a = e["action"]
        args = {arg["name"]: arg["value"] for arg in a.get("args", [])}
        oracle_details.append({"app": a["app"], "fn": a["function"], "args": args})

    agent_details = []
    for ev in agent_events:
        agent_details.append({"app": ev.get("app", ""), "fn": ev.get("fn", ""), "args": ev.get("args", {})})

    # Soft matching (function-level)
    soft_matched = 0
    used = set()
    for od in oracle_details:
        oracle_sig = f"{od['app']}.{od['fn']}".lower()
        for j, ad in enumerate(agent_details):
            if j not in used:
                agent_sig = f"{ad['app']}.{ad['fn']}".lower()
                if oracle_sig == agent_sig:
                    soft_matched += 1
                    used.add(j)
                    break

    # Exact matching (function + args)
    exact_matched = 0
    used_exact = set()
    for od in oracle_details:
        for j, ad in enumerate(agent_details):
            if j not in used_exact:
                if (od["app"] == ad["app"] and od["fn"] == ad["fn"]
                        and od["args"] == ad["args"]):
                    exact_matched += 1
                    used_exact.add(j)
                    break

    n_oracle = len(oracle_details)
    soft_recall = soft_matched / n_oracle if n_oracle else 0
    exact_recall = exact_matched / n_oracle if n_oracle else 0
    em = 1.0 if exact_matched == n_oracle else 0.0

    return {"score": soft_recall, "em": em, "exact_recall": exact_recall,
            "soft_recall": soft_recall, "method": "gaia2_events"}

# ─── Main Experiment ──────────────────────────────────────────────────────

async def run_benchmark(benchmark: str, tasks: list[dict]) -> dict:
    """Run full ablation on one benchmark: train → collect experiences → test with/without."""
    print(f"\n{'='*70}")
    print(f"  Benchmark: {benchmark}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Train: {len(tasks)//2} | Test: {len(tasks) - len(tasks)//2}")
    print(f"{'='*70}")

    mid = len(tasks) // 2
    train_tasks = tasks[:mid]
    test_tasks = tasks[mid:]

    os.makedirs(f"{RESULTS_DIR}/{benchmark}", exist_ok=True)

    # ─── Phase 1: Train (collect experiences) ─────────────────────────
    print(f"\n  Phase 1: Running {len(train_tasks)} train tasks (collecting experiences)...")
    sf = SkillForgeV6(token_budget=2000)
    sem = asyncio.Semaphore(CONCURRENCY)

    def _extract_oracle_actions(task):
        """Extract oracle actions from Gaia2 scenario or task metadata."""
        if benchmark == "gaia2":
            sp = task.get("metadata", {}).get("scenario_path", "")
            if sp and os.path.exists(sp):
                scenario = json.load(open(sp))
                oracle_events = [e for e in scenario.get("events", []) if e.get("event_type") == "AGENT"]
                oracle = []
                for ev in oracle_events:
                    a = ev["action"]
                    oracle.append({"app": a["app"], "fn": a["function"],
                                   "args": {arg["name"]: arg["value"] for arg in a.get("args", [])}})
                return oracle
        return []

    async def run_train(task):
        async with sem:
            r = await run_task_generic(task, experience_section="", benchmark=benchmark, group="train")
            # Record experience WITH oracle actions from scenario
            oracle = _extract_oracle_actions(task)
            sf.record_experience(
                task_id=r["task_id"],
                task_desc=task["description"][:300],
                agent_actions=r.get("actions", []),
                oracle_actions=oracle,
                token_cost=len(r.get("response", "")) // 4,
                time_cost=r.get("time_cost", 0),
                llm_reviewer=llm_review_fn,  # Actually call AI refine
            )
            return r

    train_results = await asyncio.gather(*[run_train(t) for t in train_tasks])
    train_valid = [r for r in train_results if not r.get("error")]
    print(f"  Train complete: {len(train_valid)}/{len(train_tasks)} successful")
    print(f"  Library: {sf.stats}")

    # Save library
    sf.save(f"{RESULTS_DIR}/{benchmark}/library_after_train.json")

    # ─── Phase 2: Test (3 groups) ─────────────────────────────────────
    print(f"\n  Phase 2: Testing {len(test_tasks)} tasks × 3 groups...")

    # Group A: Baseline
    print(f"    [A] Baseline (no augmentation)...")
    async def run_test_baseline(task):
        async with sem:
            return await run_task_generic(task, experience_section="", benchmark=benchmark, group="A")
    results_a = await asyncio.gather(*[run_test_baseline(t) for t in test_tasks])

    # Group B: Raw injection (no AI refinement — strip ai_refined fields)
    print(f"    [B] Raw experience injection...")
    # Build a raw library copy (without AI-refined fields)
    raw_library = ExperienceLibrary()
    for exp in sf.library.experiences:
        raw_exp = copy.deepcopy(exp)
        # Clear AI-refined fields to force raw format
        raw_exp.failure_taxonomy = {k: v for k, v in raw_exp.failure_taxonomy.items()
                                     if k not in ("ai_refined", "causal_lesson", "avoidance_note",
                                                  "transferability", "generalized_steps",
                                                  "evolution_insight", "quality_score", "evolution_trace")}
        raw_library.record(raw_exp)

    async def run_test_raw(task):
        async with sem:
            aug = build_augmented_prompt(task["description"][:300], raw_library, token_budget=2000)
            return await run_task_generic(task, experience_section=aug, benchmark=benchmark, group="B")
    results_b = await asyncio.gather(*[run_test_raw(t) for t in test_tasks])

    # Group C: AI-refined injection (uses full AI-refined library)
    print(f"    [C] AI-refined experience injection...")
    async def run_test_refined(task):
        async with sem:
            aug = build_augmented_prompt(task["description"][:300], sf.library, token_budget=2000)
            return await run_task_generic(task, experience_section=aug, benchmark=benchmark, group="C")
    results_c = await asyncio.gather(*[run_test_refined(t) for t in test_tasks])

    # ─── Evaluate ─────────────────────────────────────────────────────
    scores = {"A_baseline": [], "B_raw": [], "C_refined": []}
    for i, task in enumerate(test_tasks):
        scores["A_baseline"].append(evaluate_task(results_a[i], benchmark))
        scores["B_raw"].append(evaluate_task(results_b[i], benchmark))
        scores["C_refined"].append(evaluate_task(results_c[i], benchmark))

    # Compute averages
    report = {}
    for group, evals in scores.items():
        valid = [e["score"] for e in evals if e.get("score") is not None]
        avg = sum(valid) / len(valid) if valid else 0
        report[group] = {"avg_score": avg, "n": len(valid)}

    print(f"\n  Results ({benchmark}):")
    print(f"    A (Baseline):    {report['A_baseline']['avg_score']:.1%}")
    print(f"    B (Raw inject):  {report['B_raw']['avg_score']:.1%}")
    print(f"    C (AI-refined):  {report['C_refined']['avg_score']:.1%}")
    delta_bc = report['C_refined']['avg_score'] - report['B_raw']['avg_score']
    delta_ac = report['C_refined']['avg_score'] - report['A_baseline']['avg_score']
    print(f"    Δ(C-A): {delta_ac:+.1%} | Δ(C-B): {delta_bc:+.1%}")

    # Save
    full_report = {
        "benchmark": benchmark,
        "n_train": len(train_tasks), "n_test": len(test_tasks),
        "results": report,
        "delta_refined_vs_baseline": delta_ac,
        "delta_refined_vs_raw": delta_bc,
    }
    with open(f"{RESULTS_DIR}/{benchmark}/report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    return full_report

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  SkillForge V6 Unified Benchmark — AI-Refined Evolution      ║")
    print("║  Benchmarks: Gaia2, LoCoMo, GAIA, ALFWorld, SWE-bench        ║")
    print(f"║  Model: {MODEL} | Concurrency: {CONCURRENCY}                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    all_reports = {}

    # Load benchmarks with task limits
    print("\n  Loading benchmarks...")
    benchmarks = {}

    benchmarks["gaia2"] = load_gaia2_tasks(TASK_LIMITS["gaia2"])
    print(f"    gaia2: {len(benchmarks['gaia2'])} tasks")

    benchmarks["locomo"] = load_hf_tasks("locomo", TASK_LIMITS["locomo"])[:TASK_LIMITS["locomo"]]
    print(f"    locomo: {len(benchmarks['locomo'])} tasks")

    benchmarks["gaia"] = load_hf_tasks("gaia", TASK_LIMITS["gaia"])[:TASK_LIMITS["gaia"]]
    print(f"    gaia: {len(benchmarks['gaia'])} tasks")

    benchmarks["alfworld"] = load_hf_tasks("alfworld", TASK_LIMITS["alfworld"])[:TASK_LIMITS["alfworld"]]
    print(f"    alfworld: {len(benchmarks['alfworld'])} tasks")

    benchmarks["swebench"] = load_swebench_tasks(TASK_LIMITS["swebench"])
    print(f"    swebench: {len(benchmarks['swebench'])} tasks")

    print(f"\n  Total: {sum(len(t) for t in benchmarks.values())} tasks across {len(benchmarks)} benchmarks")

    for name, tasks in benchmarks.items():
        if not tasks:
            print(f"\n  SKIP {name}: no tasks loaded")
            continue
        try:
            report = await run_benchmark(name, tasks)
            all_reports[name] = report
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {name}: {e}")
            traceback.print_exc()
            all_reports[name] = {"error": str(e)}

    # Final summary
    print(f"\n\n{'═'*70}")
    print("  FINAL SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Benchmark':<12} {'Baseline':>10} {'Raw':>10} {'AI-Refined':>12} {'Δ(C-A)':>8} {'Δ(C-B)':>8}")
    print(f"  {'-'*60}")
    for name, r in all_reports.items():
        if "error" in r:
            print(f"  {name:<12} ERROR: {r['error'][:40]}")
        else:
            res = r["results"]
            print(f"  {name:<12} {res['A_baseline']['avg_score']:>9.1%} "
                  f"{res['B_raw']['avg_score']:>9.1%} "
                  f"{res['C_refined']['avg_score']:>11.1%} "
                  f"{r['delta_refined_vs_baseline']:>+7.1%} "
                  f"{r['delta_refined_vs_raw']:>+7.1%}")

    # Save final
    with open(f"{RESULTS_DIR}/final_summary.json", "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS_DIR}/final_summary.json")

if __name__ == "__main__":
    asyncio.run(main())
