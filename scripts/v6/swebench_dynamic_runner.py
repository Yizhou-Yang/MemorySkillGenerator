#!/usr/bin/env python3
"""
SWE-bench Dynamic Runner — SkillForge V6

Agent interacts with a real codebase inside Docker containers:
  read files → search code → edit → run tests → iterate

Each instance: Docker container with repo at base_commit.
Agent uses CodeBuddy SDK (with bash tools) to navigate and fix the bug.
Evaluation: apply agent's patch → run test suite → pass@1.

Train/Test split: first 10 train, next 10 test (× 3 groups).
"""
import asyncio
import json
import os
import sys
import time
import copy
import re
import subprocess

sys.path.insert(0, '/data/home/yizhouyang/workspace/SkillForge/src')
sys.path.insert(0, '/data/home/yizhouyang/workspace/SkillForge')

# CODEBUDDY_API_KEY must be set in environment
os.environ['CODEBUDDY_INTERNET_ENVIRONMENT'] = 'ioa'

from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage, ToolUseBlock
from v6 import (SkillForgeV6, ExperienceLibrary,
                analyze_execution, build_augmented_prompt, ai_review_experience)

MODEL = "hy3-preview-ioa"
CONCURRENCY = 3  # Docker containers + headless cleanup
N_TRAIN = 30
N_TEST = 50
RESULTS_DIR = "/data1/benchmarks/unified_v6_results/swebench_dynamic"
DOCKER_IMAGE_PREFIX = "ghcr.io/epoch-research/swe-bench.eval.x86_64"


# ══════════════════════════════════════════════════════════════════════════
#  Docker helpers
# ══════════════════════════════════════════════════════════════════════════

def instance_to_image(instance_id: str) -> str:
    """Convert instance_id to Docker image name."""
    # astropy__astropy-12907 → ghcr.io/epoch-research/swe-bench.eval.x86_64.astropy__astropy-12907:latest
    return f"{DOCKER_IMAGE_PREFIX}.{instance_id}:latest"


def image_exists(instance_id: str) -> bool:
    """Check if Docker image exists locally."""
    img = instance_to_image(instance_id)
    r = subprocess.run(["docker", "image", "inspect", img], capture_output=True, timeout=10)
    return r.returncode == 0


def start_container(instance_id: str) -> str:
    """Start a Docker container for this instance. Returns container ID."""
    img = instance_to_image(instance_id)
    r = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", f"swe_{instance_id[:30]}_{int(time.time())%10000}",
         "-w", "/testbed", img, "sleep", "3600"],
        capture_output=True, text=True, timeout=30
    )
    if r.returncode != 0:
        raise RuntimeError(f"Docker start failed: {r.stderr[:200]}")
    return r.stdout.strip()


def exec_in_container(container_id: str, cmd: str, timeout: int = 60) -> tuple[str, str, int]:
    """Execute command in container. Returns (stdout, stderr, returncode)."""
    r = subprocess.run(
        ["docker", "exec", container_id, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return r.stdout, r.stderr, r.returncode


def stop_container(container_id: str):
    """Stop and remove container."""
    try:
        subprocess.run(["docker", "kill", container_id], capture_output=True, timeout=30)
    except Exception:
        pass


def get_patch_from_container(container_id: str) -> str:
    """Get git diff (the agent's patch) from the container."""
    stdout, _, _ = exec_in_container(container_id, "cd /testbed && git diff", timeout=30)
    return stdout


# ══════════════════════════════════════════════════════════════════════════
#  Run one SWE-bench instance
# ══════════════════════════════════════════════════════════════════════════

async def run_instance(instance: dict, experience_section: str = "") -> dict:
    """Run one SWE-bench instance: agent fixes bug in Docker container."""
    instance_id = instance["instance_id"]
    problem = instance["problem_statement"]
    repo = instance["repo"]
    
    result = {
        "instance_id": instance_id, "repo": repo,
        "problem": problem[:300], "patch": "", "won": False,
        "steps": 0, "error": None, "time_cost": 0,
        "actions": [],
    }
    
    if not image_exists(instance_id):
        result["error"] = "no_docker_image"
        return result
    
    container_id = None
    t0 = time.time()
    try:
        container_id = start_container(instance_id)
        
        # Get repo structure overview
        stdout, _, _ = exec_in_container(container_id, "cd /testbed && find . -name '*.py' -maxdepth 3 | head -30")
        repo_files = stdout.strip()
        
        system_prompt = f"""You are a software engineer fixing a bug in a Python project.

## Repository: {repo}
## Key files:
{repo_files}

## Bug Report:
{problem[:2000]}

## Instructions:
1. Read relevant source files to understand the codebase
2. Identify the root cause of the bug
3. Make the minimal fix — edit only the necessary files
4. Verify your fix makes sense

Use bash commands to navigate and edit. You are in /testbed/.
Available: cat, grep, find, sed, python, git diff, etc.
Do NOT run the test suite (it's slow). Focus on making a correct patch.
{experience_section}"""

        prompt = f"{system_prompt}\n\nFix this bug. Start by reading the relevant code."

        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=15, cwd="/tmp",
            env={"CODEBUDDY_API_KEY": os.environ["CODEBUDDY_API_KEY"],
                 "CODEBUDDY_INTERNET_ENVIRONMENT": "ioa"},
        )

        # The agent will use Bash tool to exec commands.
        # We intercept tool calls and route them to the Docker container.
        # Actually, CodeBuddy SDK runs commands locally. We need a different approach:
        # Give the agent a script that wraps docker exec.
        
        # Write a wrapper script
        wrapper_dir = f"/tmp/swe_{instance_id}"
        os.makedirs(wrapper_dir, exist_ok=True)
        wrapper_script = f"""#!/bin/bash
docker exec {container_id} bash -c "$@"
"""
        wrapper_path = f"{wrapper_dir}/dexec.sh"
        with open(wrapper_path, "w") as f:
            f.write(wrapper_script)
        os.chmod(wrapper_path, 0o755)

        # Rewrite prompt to use the wrapper
        prompt2 = f"""You are fixing a bug in {repo}. The code is in a Docker container.

## Bug Report:
{problem[:2000]}

## How to interact:
Run commands via: {wrapper_path} "your command here"
Examples:
  {wrapper_path} "cat /testbed/path/to/file.py"
  {wrapper_path} "grep -rn 'function_name' /testbed/src/"
  {wrapper_path} "cd /testbed && git diff"

To edit files, use sed or python:
  {wrapper_path} "sed -i 's/old_line/new_line/' /testbed/path/file.py"

## Key files:
{repo_files}
{experience_section}
Fix the bug. Start by reading the relevant source code."""

        opt2 = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=15,
            cwd=wrapper_dir,
        )

        import concurrent.futures
        import signal
        def _run_agent():
            # Track headless PIDs before call
            import subprocess as _sp
            before = set(_sp.run("pgrep -f 'codebuddy-headless.*max-turns.15'",
                                 shell=True, capture_output=True, text=True).stdout.split())
            
            async def _inner():
                actions = []
                try:
                    async for msg in query(prompt=prompt2, options=opt2):
                        if isinstance(msg, AssistantMessage):
                            for block in msg.content:
                                if isinstance(block, ToolUseBlock):
                                    actions.append({"tool": block.name, "input": str(block.input)[:200]})
                except Exception:
                    pass
                return actions
            
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_inner())
            except Exception:
                result = []
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
            
            # Kill NEW headless processes spawned by this call
            after = set(_sp.run("pgrep -f 'codebuddy-headless.*max-turns.15'",
                                shell=True, capture_output=True, text=True).stdout.split())
            for pid in (after - before):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
            
            return result

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            actions = await loop.run_in_executor(pool, _run_agent)
        
        result["actions"] = actions
        result["steps"] = len(actions)
        
        # Get the patch
        patch = get_patch_from_container(container_id)
        result["patch"] = patch
        
        # Evaluate: apply test_patch and run tests
        if patch.strip():
            result["has_patch"] = True
            # Run the failing tests to see if they pass now
            test_patch = instance.get("test_patch", "")
            fail_to_pass = instance.get("FAIL_TO_PASS", "")
            
            if fail_to_pass:
                # Parse test names
                try:
                    test_names = json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else fail_to_pass
                except Exception:
                    test_names = [fail_to_pass]
                
                # Apply test patch if available
                if test_patch:
                    exec_in_container(container_id, 
                        f"cd /testbed && echo '{test_patch}' | git apply --allow-empty 2>/dev/null || true",
                        timeout=30)
                
                # Run the specific failing tests
                all_pass = True
                for test_name in test_names[:3]:  # Cap at 3 tests
                    stdout, stderr, rc = exec_in_container(
                        container_id,
                        f"cd /testbed && python -m pytest {test_name} -x --timeout=60 2>&1 | tail -5",
                        timeout=120
                    )
                    if rc != 0 or "FAILED" in stdout or "ERROR" in stdout:
                        all_pass = False
                        break
                
                result["won"] = all_pass
                result["score"] = 1.0 if all_pass else 0.0
            else:
                result["won"] = False
                result["score"] = 0.0
        else:
            result["has_patch"] = False
            result["won"] = False
            result["score"] = 0.0
    
    except Exception as e:
        result["error"] = str(e)[:200]
        result["won"] = False
        result["score"] = 0.0
    finally:
        result["time_cost"] = time.time() - t0
        if container_id:
            stop_container(container_id)
    
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  SWE-bench Dynamic — SkillForge V6                           ║")
    print(f"║  Model: {MODEL} | Concurrency: {CONCURRENCY}                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # Load dataset
    print("\n  Loading SWE-bench Verified...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    
    # Filter to instances with available Docker images
    available = []
    for row in ds:
        iid = row["instance_id"]
        if image_exists(iid):
            available.append(dict(row))
        if len(available) >= N_TRAIN + N_TEST:
            break
    
    print(f"  Available instances with Docker images: {len(available)}")
    if len(available) < N_TRAIN + N_TEST:
        print(f"  WARNING: only {len(available)} available, need {N_TRAIN + N_TEST}")
    
    train_instances = available[:N_TRAIN]
    test_instances = available[N_TRAIN:N_TRAIN + N_TEST]
    print(f"  Train: {len(train_instances)} | Test: {len(test_instances)}")

    # ─── Phase 1: Train ───────────────────────────────────────────
    print(f"\n  Phase 1: Train ({len(train_instances)} instances)...", flush=True)
    sf = SkillForgeV6(token_budget=1500)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def train_one(inst):
        async with sem:
            return await run_instance(inst)

    train_results = await asyncio.gather(*[train_one(i) for i in train_instances])

    for i, r in enumerate(train_results):
        s = "\u2713" if r["won"] else "\u2717"
        has_p = "patch" if r.get("has_patch") else "no-patch"
        print(f"    {s} {r['instance_id']:<35} {has_p:<10} steps={r['steps']:>2} "
              f"t={r['time_cost']:.0f}s", flush=True)
        
        sf.record_experience(
            task_id=r["instance_id"],
            task_desc=f"Fix bug in {r['repo']}: {r['problem'][:200]}",
            agent_actions=r["actions"],
            oracle_actions=[{"tool": "patch", "output": r.get("patch", "")[:500]}] if r["won"] else [],
            token_cost=r["steps"] * 200,
            time_cost=r["time_cost"],
            llm_reviewer=None,
        )

    # Passthrough refine
    for exp in sf.library.experiences:
        rv = ai_review_experience(exp, llm_fn=None)
        exp.failure_taxonomy.update({k: rv.get(k, "") for k in
            ("ai_refined","causal_lesson","avoidance_note","transferability",
             "generalized_steps","evolution_insight")})

    sf.save(f"{RESULTS_DIR}/library.json")
    n_won = sum(1 for r in train_results if r["won"])
    n_patch = sum(1 for r in train_results if r.get("has_patch"))
    print(f"\n  Train: {n_won}/{len(train_results)} passed, {n_patch} produced patches", flush=True)
    print(f"  Library: {sf.stats}", flush=True)

    # ─── Phase 2: Test ─────────────────────────────────────────────
    print(f"\n  Phase 2: Test ({len(test_instances)} instances × 3 groups)...", flush=True)

    raw_lib = ExperienceLibrary()
    for exp in sf.library.experiences:
        raw_exp = copy.deepcopy(exp)
        raw_exp.failure_taxonomy = {k: v for k, v in raw_exp.failure_taxonomy.items()
                                     if k not in ("ai_refined","causal_lesson","avoidance_note",
                                                  "transferability","generalized_steps",
                                                  "evolution_insight","quality_score","evolution_trace")}
        raw_lib.record(raw_exp)

    # A: Baseline
    print("    [A] Baseline...", flush=True)
    async def test_a(inst):
        async with sem:
            return await run_instance(inst)
    results_a = await asyncio.gather(*[test_a(i) for i in test_instances])

    # B: Raw
    print("    [B] Raw injection...", flush=True)
    async def test_b(idx, inst):
        async with sem:
            td = f"Fix bug in {inst['repo']}: {inst['problem_statement'][:200]}"
            aug = build_augmented_prompt(td, raw_lib, token_budget=1500,
                                         metadata={"benchmark": "swebench"})
            return await run_instance(inst, experience_section=f"\n## Experience\n{aug}" if aug else "")
    results_b = await asyncio.gather(*[test_b(i, inst) for i, inst in enumerate(test_instances)])

    # C: AI-refined
    print("    [C] AI-refined...", flush=True)
    async def test_c(idx, inst):
        async with sem:
            td = f"Fix bug in {inst['repo']}: {inst['problem_statement'][:200]}"
            aug = build_augmented_prompt(td, sf.library, token_budget=1500,
                                         metadata={"benchmark": "swebench"})
            return await run_instance(inst, experience_section=f"\n## Experience\n{aug}" if aug else "")
    results_c = await asyncio.gather(*[test_c(i, inst) for i, inst in enumerate(test_instances)])

    # ─── Results ───────────────────────────────────────────────────
    sr = lambda rs: sum(1 for r in rs if r["won"]) / len(rs) if rs else 0
    sa, sb, sc = sr(results_a), sr(results_b), sr(results_c)

    print(f"\n{'='*60}")
    print(f"  SWE-bench Dynamic Results (n={len(test_instances)})")
    print(f"{'='*60}")
    print(f"  A (Baseline):    {sa:.1%} ({sum(1 for r in results_a if r['won'])}/{len(results_a)})")
    print(f"  B (Raw inject):  {sb:.1%} ({sum(1 for r in results_b if r['won'])}/{len(results_b)})")
    print(f"  C (AI-refined):  {sc:.1%} ({sum(1 for r in results_c if r['won'])}/{len(results_c)})")
    print(f"  Δ(C-A): {sc-sa:+.1%} | Δ(C-B): {sc-sb:+.1%}")

    # Patch generation rate
    pa = sum(1 for r in results_a if r.get("has_patch")) / len(results_a)
    pb = sum(1 for r in results_b if r.get("has_patch")) / len(results_b)
    pc = sum(1 for r in results_c if r.get("has_patch")) / len(results_c)
    print(f"\n  Patch rate: A={pa:.0%} B={pb:.0%} C={pc:.0%}")

    report = {
        "benchmark": "swebench_dynamic", "n_train": len(train_instances), "n_test": len(test_instances),
        "results": {"A_baseline": {"avg_score": sa, "n": len(results_a)},
                     "B_raw": {"avg_score": sb, "n": len(results_b)},
                     "C_refined": {"avg_score": sc, "n": len(results_c)}},
        "delta_refined_vs_baseline": sc - sa, "delta_refined_vs_raw": sc - sb,
    }
    with open(f"{RESULTS_DIR}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR}/report.json")


if __name__ == "__main__":
    asyncio.run(main())
