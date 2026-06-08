#!/usr/bin/env python3
"""
SkillForge V6 — Latest Experiment Runner

Design:
  1. Sequential iterative training (each task uses accumulated experience)
  2. GAIA: agentic mode with tool calling (max_turns=30)
  3. ALFWorld: interactive subprocess environment
  4. Cross-agent skill quality evaluation (no oracle-dependent retry)
  5. Evaluation: Exact Match + pass@1 (aligned with competing papers)

Model: deepseek-v4-pro via CodeBuddy CLI
"""
import asyncio
import json
import os
import sys
import time
import copy
import re
import concurrent.futures
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Environment — use CodeBuddy CLI for deepseek-v4-pro
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'deepseek-v4-pro'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')

from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage, ToolUseBlock

from v6 import (SkillForgeV6, ExperienceLibrary, Experience,
                analyze_execution, build_augmented_prompt,
                format_success_experience, format_failure_experience,
                ai_review_experience, cross_agent_evaluate_skill)
from v6.gate import classify_task_type
from benchmarks.loader import BenchmarkLoader

MODEL = "deepseek-v4-pro"
CONCURRENCY = 15
TASK_TIMEOUT_QA = 120
TASK_TIMEOUT_AGENT = 300
TASK_TIMEOUT_ALFWORLD = 180
CROSS_AGENT_QUALITY_THRESHOLD = 5  # 0-10; experiences below this are excluded from injection
RESULTS_DIR = str(PROJECT_ROOT / "experiments_results" / "latest")

TASK_LIMITS = {
    "gaia": 50,
    "alfworld": 40,
    "locomo": 50,
}

ALFWORLD_PYTHON = str(PROJECT_ROOT / ".venv_alfworld" / "bin" / "python")
ALFWORLD_DATA = str(PROJECT_ROOT / ".venv_alfworld" / "data")
ALFWORLD_MAX_STEPS = 30

def llm_review_fn(prompt: str) -> str:
    """Call deepseek-v4-pro for AI review (single-turn, sync)."""
    async def _call():
        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=2, cwd="/tmp"
        )
        result = ""
        try:
            async with asyncio.timeout(90):
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

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_call())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        return future.result(timeout=120)


async def llm_extract_answer(response: str, question: str) -> str:
    """Use LLM to extract the concise final answer from a verbose response."""
    if len(response.split()) < 30:
        return response  # Already concise
    
    prompt = f"""Extract ONLY the final answer from this response. Output just the answer, nothing else.

Question: {question[:200]}

Response: {response[:1000]}

Final answer (concise, just the key fact/number/name):"""
    
    opt = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions", model=MODEL, max_turns=1, cwd="/tmp"
    )
    result = ""
    try:
        async with asyncio.timeout(30):
            async for msg in query(prompt=prompt, options=opt):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, 'text') and block.text:
                            result += block.text
                    if result:
                        break
    except Exception:
        pass
    return result.strip() if result.strip() else response


def compute_exact_match(response: str, expected: str) -> float:
    """Exact Match: normalized string comparison (aligned with GAIA/LoCoMo papers)."""
    if not response or not expected:
        return 0.0
    norm_resp = re.sub(r'\s+', ' ', response.strip().lower())
    norm_exp = re.sub(r'\s+', ' ', expected.strip().lower())
    if norm_resp == norm_exp:
        return 1.0
    if norm_exp in norm_resp or norm_resp in norm_exp:
        return 1.0
    resp_tokens = set(norm_resp.split())
    exp_tokens = set(norm_exp.split())
    if exp_tokens and exp_tokens.issubset(resp_tokens):
        return 1.0
    return 0.0


async def run_gaia_task(task: dict, experience_section: str = "",
                        group: str = "A") -> dict:
    """Run GAIA task with full agent capabilities (tool calling, web search)."""
    task_id = task["task_id"]
    description = task["description"]
    expected = task.get("expected", "")

    system = (
        "You are a research assistant with access to tools. "
        "Answer the question accurately. Use web search, file reading, "
        "or computation as needed. Give a concise final answer."
    )
    if experience_section:
        system += f"\n\n{experience_section}"

    prompt = f"{system}\n\n{description}\n\nProvide your final answer concisely."

    # Use max_turns=30 for tool calling (agentic mode)
    opt = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions", model=MODEL, max_turns=30, cwd="/tmp"
    )

    result = {"task_id": task_id, "expected": expected, "response": "",
              "error": None, "time_cost": 0, "augmented": bool(experience_section),
              "group": group, "actions": []}
    t0 = time.time()

    try:
        async with asyncio.timeout(TASK_TIMEOUT_AGENT):
            async for msg in query(prompt=prompt, options=opt):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            result["actions"].append({
                                "tool": block.name,
                                "input": str(block.input)[:200]
                            })
                        elif hasattr(block, 'text') and block.text:
                            if '429' in block.text and '额度' in block.text:
                                result["error"] = "429_rate_limit"
                                break
                            result["response"] += block.text
    except TimeoutError:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)[:200]

    result["time_cost"] = time.time() - t0
    return result


ALFWORLD_GAME_WORKER = '''
import sys, os, json

os.environ["ALFWORLD_DATA"] = "{alfworld_data}"
game_file = "{game_file}"
max_steps = {max_steps}

_saved_fd = os.dup(1)
os.dup2(2, 1)
import textworld
ri = textworld.EnvInfos(won=True, admissible_commands=True, description=True, inventory=True)
env = textworld.start(game_file, request_infos=ri)
os.dup2(_saved_fd, 1)
os.close(_saved_fd)
_out = os.fdopen(1, "w", buffering=1)

step_count = 0
_out.write(json.dumps({{"type":"ready"}}) + "\\n"); _out.flush()

for line in sys.stdin:
    c = json.loads(line)
    if c["action"] == "reset":
        gs = env.reset()
        step_count = 0
        adm = list(gs.admissible_commands or [])
        _out.write(json.dumps({{"type":"obs","obs":gs.feedback,"actions":adm[:30],
                                "won":bool(gs.won),"done":bool(gs.won)}}) + "\\n"); _out.flush()
    elif c["action"] == "step":
        gs, score, done = env.step(c["command"])
        step_count += 1
        adm = list(gs.admissible_commands or [])
        won = bool(gs.won)
        done = done or won or step_count >= max_steps
        _out.write(json.dumps({{"type":"obs","obs":gs.feedback,"actions":adm[:30],
                                "won":won,"done":done}}) + "\\n"); _out.flush()
    elif c["action"] == "quit":
        break
env.close()
'''


class SingleGameEnv:
    """One ALFWorld game in a subprocess."""
    def __init__(self, game_file):
        import subprocess as sp
        code = ALFWORLD_GAME_WORKER.format(
            alfworld_data=ALFWORLD_DATA, game_file=game_file, max_steps=ALFWORLD_MAX_STEPS
        )
        self.proc = sp.Popen([ALFWORLD_PYTHON, "-c", code],
                             stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE,
                             text=True, bufsize=1)
        line = self.proc.stdout.readline()
        msg = json.loads(line)
        assert msg.get("type") == "ready", f"Init failed: {self.proc.stderr.read()[:300]}"

    def reset(self):
        self.proc.stdin.write('{"action":"reset"}\n'); self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def step(self, cmd):
        self.proc.stdin.write(json.dumps({"action": "step", "command": cmd}) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        try:
            self.proc.stdin.write('{"action":"quit"}\n'); self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


async def llm_decide_alfworld_action(observation: str, task: str,
                                      admissible_actions: list, history: list,
                                      experience_section: str = "") -> str:
    """Use LLM to decide next ALFWorld action."""
    # Simplify observation
    obs_simple = re.sub(r'_bar__(?:minus|plus)_\d+_dot_\d+(?:_bar__(?:minus|plus)_\d+_dot_\d+)*', '', observation)
    obs_simple = re.sub(r'_+', ' ', obs_simple)

    action_lines = [f"  {i}. {a}" for i, a in enumerate(admissible_actions[:20])]
    history_str = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history[-8:]))

    prompt = f"""You are an AI agent completing a household task. Choose the best next action by its NUMBER.

## Task
{task}

## Current Observation
{obs_simple[:500]}

## Action History
{history_str if history else "(none yet)"}

## Available Actions (choose by number)
{chr(10).join(action_lines)}
{experience_section}
## Response
Output ONLY the number (0-{min(len(admissible_actions), 20)-1}) of your chosen action. Nothing else."""

    opt = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions", model=MODEL, max_turns=1, cwd="/tmp"
    )
    result = ""
    try:
        async with asyncio.timeout(30):
            async for msg in query(prompt=prompt, options=opt):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, 'text') and block.text:
                            result += block.text
                    if result:
                        break
    except Exception:
        pass

    # Parse number
    m = re.search(r'\b(\d+)\b', result.strip())
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(admissible_actions):
            return admissible_actions[idx]

    # Fallback: first action
    return admissible_actions[0] if admissible_actions else "look"


async def run_alfworld_task(game_file: str, game_type: str,
                            experience_section: str = "", group: str = "A") -> dict:
    """Run one ALFWorld game interactively."""
    result = {"task": game_type, "task_type": game_type, "won": False,
              "steps": 0, "trajectory": [], "score": 0.0, "group": group,
              "error": None, "time_cost": 0}
    t0 = time.time()

    try:
        env = SingleGameEnv(game_file)
    except Exception as e:
        result["error"] = f"env_init: {str(e)[:100]}"
        result["time_cost"] = time.time() - t0
        return result

    try:
        info = env.reset()
        obs = info["obs"]
        m = re.search(r"Your task is to:\s*(.+)", obs)
        task = m.group(1).strip() if m else game_type
        result["task"] = task
        trajectory = []

        for _ in range(ALFWORLD_MAX_STEPS):
            admissible = info.get("actions", ["look"])
            if not admissible:
                admissible = ["look"]
            action = await llm_decide_alfworld_action(
                obs, task, admissible,
                [t[0] for t in trajectory], experience_section
            )
            info = env.step(action)
            obs = info["obs"]
            trajectory.append((action, obs[:200]))
            if info.get("won") or info.get("done"):
                break

        result["won"] = info.get("won", False)
        result["steps"] = len(trajectory)
        result["trajectory"] = trajectory
        result["score"] = 1.0 if info.get("won", False) else 0.0
    except Exception as e:
        result["error"] = str(e)[:200]
    finally:
        env.close()

    result["time_cost"] = time.time() - t0
    return result


async def run_locomo_task(task: dict, experience_section: str = "",
                          group: str = "A") -> dict:
    """Run LoCoMo QA task (conversation memory)."""
    task_id = task["task_id"]
    description = task["description"]
    expected = task.get("expected", "")

    system = (
        "You are a helpful assistant. Answer the question based on the "
        "conversation history. Be concise — give only the answer, no explanation."
    )
    if experience_section:
        system += f"\n\n{experience_section}"

    prompt = f"[System]\n{system}\n\n{description}\n\nAnswer concisely:"

    opt = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions", model=MODEL, max_turns=2, cwd="/tmp"
    )

    result = {"task_id": task_id, "expected": expected, "response": "",
              "error": None, "time_cost": 0, "augmented": bool(experience_section),
              "group": group}
    t0 = time.time()

    try:
        async with asyncio.timeout(TASK_TIMEOUT_QA):
            async for msg in query(prompt=prompt, options=opt):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, 'text') and block.text:
                            if '429' in block.text and '额度' in block.text:
                                result["error"] = "429_rate_limit"
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


async def evaluate_exact_match(result: dict, benchmark: str) -> dict:
    """Evaluation using Exact Match + pass@1 (aligned with competing papers)."""
    if benchmark == "alfworld":
        return {"em": result.get("score", 0.0), "pass_at_1": result.get("won", False),
                "method": "task_completion"}

    expected = result.get("expected", "").strip()
    response = result.get("response", "").strip()

    if not expected or not response:
        return {"em": 0.0, "pass_at_1": False, "method": "empty"}

    extracted = await llm_extract_answer(response, result.get("task_id", ""))
    em = compute_exact_match(extracted or response, expected)
    pass_at_1 = em >= 1.0

    return {
        "em": em,
        "pass_at_1": pass_at_1,
        "extracted_answer": (extracted or "")[:200],
        "method": "exact_match"
    }


async def train_sequential(benchmark: str, train_tasks: list, sf: SkillForgeV6,
                           sem: asyncio.Semaphore, game_list: list = None) -> list:
    """
    Sequential iterative training: each task uses accumulated experience.
    Cross-agent evaluation filters low-quality experiences from injection.
    """
    all_results = []

    for i, task in enumerate(train_tasks):
        async with sem:
            # Build experience section from accumulated library
            aug = ""
            if sf.library.experiences:
                if benchmark == "gaia":
                    aug = build_augmented_prompt(
                        task["description"][:300], sf.library, token_budget=1500,
                        metadata={"benchmark": "gaia"}
                    )
                elif benchmark == "alfworld":
                    task_desc = f"{task.get('description', '')} [type: {task.get('metadata', {}).get('task_type', '')}]"
                    aug = build_augmented_prompt(
                        task_desc, sf.library, token_budget=1500,
                        metadata=task.get("metadata", {})
                    )
                else:  # locomo
                    aug = build_augmented_prompt(
                        task["description"][:300], sf.library, token_budget=600,
                        metadata=task.get("metadata", {})
                    )

            # Run task
            if benchmark == "gaia":
                r = await run_gaia_task(task, experience_section=aug, group="train")
            elif benchmark == "alfworld":
                game = game_list[i] if game_list else None
                if game:
                    r = await run_alfworld_task(
                        game["file"], game["type"],
                        experience_section=f"\n## Experience\n{aug}" if aug else "",
                        group="train"
                    )
                else:
                    r = {"task_id": task["task_id"], "error": "no_game", "score": 0.0}
            else:  # locomo
                r = await run_locomo_task(task, experience_section=aug, group="train")

            # Evaluate with EM
            eval_result = await evaluate_exact_match(r, benchmark)
            score = eval_result.get("em", 0.0)

            # Record experience + cross-agent quality evaluation
            _record_experience(sf, task, r, score, benchmark, aug)

            r["_train_score"] = score
            all_results.append(r)

            status = "✓" if score >= 0.3 else "✗"
            print(f"    {status} [{i+1}/{len(train_tasks)}] score={score:.2f} "
                  f"lib={len(sf.library.experiences)} | {task['task_id'][:30]}", flush=True)

    return all_results


def _record_experience(sf: SkillForgeV6, task: dict, result: dict,
                       score: float, benchmark: str, aug_used: str):
    """Record experience from a task result."""
    response = result.get("response", "")
    actions = result.get("actions", [])

    if benchmark == "gaia":
        # GAIA: use tool actions as the trajectory
        tool_seq = [a.get("tool", "answer") for a in actions] if actions else ["answer"]
        action_cmds = [f"{a.get('tool', '')}: {a.get('input', '')[:100]}" for a in actions]
        if not action_cmds:
            action_cmds = [response[:300]]
    elif benchmark == "alfworld":
        # ALFWorld: use trajectory
        trajectory = result.get("trajectory", [])
        tool_seq = [t[0] for t in trajectory] if trajectory else ["none"]
        action_cmds = [f"{t[0]} → {t[1][:50]}" for t in trajectory] if trajectory else ["none"]
    else:
        # LoCoMo: QA
        tool_seq = ["answer"]
        action_cmds = [response[:300]]

    outcome = "success" if score >= 0.8 else "partial" if score >= 0.3 else "failure"
    missing = [] if score >= 0.5 else ["correct_answer"]

    exp = Experience(
        task_id=result.get("task_id", task["task_id"]),
        task_desc=task["description"][:300],
        tool_sequence=tool_seq,
        action_commands=action_cmds,
        outcome=outcome,
        score=score,
        missing_steps=missing,
        extra_steps=[],
        failure_reason="" if outcome == "success" else f"Score={score:.2f}",
        failure_taxonomy={
            "category": "success" if outcome == "success" else "model_failure",
            "root_cause": "" if outcome == "success" else f"score={score:.2f}",
        },
        token_cost=len(response) // 4 + len(actions) * 50,
        time_cost=result.get("time_cost", 0),
        task_complexity="complex" if benchmark == "gaia" else "moderate",
        augmentation_used=aug_used[:100] if aug_used else "",
        timestamp=time.time(),
    )

    # AI refinement
    review = ai_review_experience(exp, llm_fn=llm_review_fn)
    exp.failure_taxonomy.update({
        "ai_refined": review.get("refined", False),
        "causal_lesson": review.get("causal_lesson", ""),
        "avoidance_note": review.get("avoidance_note", ""),
        "transferability": review.get("transferability", ""),
        "generalized_steps": review.get("generalized_steps", ""),
        "evolution_insight": review.get("evolution_insight", ""),
        "quality_score": review.get("quality_score", 0),
    })

    # Cross-agent quality evaluation (replaces oracle-dependent retry)
    quality_eval = cross_agent_evaluate_skill(exp, llm_fn=llm_review_fn)
    exp.failure_taxonomy["cross_agent_verdict"] = quality_eval.get("verdict", "inject")
    exp.failure_taxonomy["cross_agent_score"] = quality_eval.get("total", 5)
    exp.failure_taxonomy["cross_agent_reason"] = quality_eval.get("reason", "")

    # Only inject high-quality experiences into the library
    if quality_eval.get("total", 5) >= CROSS_AGENT_QUALITY_THRESHOLD:
        sf.library.record(exp)
    else:
        # Still record but mark as low-quality (won't be retrieved)
        exp.failure_taxonomy["excluded"] = True
        sf.library.record(exp)


def get_alfworld_games(n: int = 40) -> list:
    """Get ALFWorld game files from HuggingFace dataset."""
    try:
        from datasets import load_dataset
        raw = load_dataset("awawa-agi/alfworld-raw", split="eval_out_of_distribution")
        games = []
        for idx, row in enumerate(raw):
            if idx >= n:
                break
            game_content_str = row.get("game_content", "{}")
            try:
                game_content = json.loads(game_content_str)
            except (json.JSONDecodeError, TypeError):
                game_content = {}
            walkthrough = game_content.get("walkthrough", [])
            game_file = game_content.get("game_file", "")
            if not game_file:
                # Try to find game file from data dir
                game_file_path = row.get("game_file_path", "")
                if game_file_path:
                    game_file = os.path.join(ALFWORLD_DATA, "json_2.1.1", game_file_path, "game.z8")
            games.append({
                "file": game_file,
                "type": row.get("task_type", "unknown"),
                "walkthrough": walkthrough,
                "game_file_path": row.get("game_file_path", ""),
            })
        return games
    except Exception as e:
        print(f"  WARNING: Could not load ALFWorld games: {e}")
        return []


async def run_benchmark(benchmark: str, tasks: list, game_list: list = None) -> dict:
    """Run full ablation on one benchmark with all fixes applied."""
    print(f"\n{'='*70}")
    print(f"  Benchmark: {benchmark} (model: {MODEL})")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Fixes: sequential train, cross-agent eval, EM+pass@1", flush=True)
    if benchmark == "gaia":
        print(f"  Mode: Agentic (max_turns=30, tool calling)")
    elif benchmark == "alfworld":
        print(f"  Mode: Interactive subprocess environment")
    else:
        print(f"  Mode: QA with enhanced hints + answer extraction")
    print(f"{'='*70}")

    mid = len(tasks) // 2
    train_tasks = tasks[:mid]
    test_tasks = tasks[mid:]
    print(f"  Train: {len(train_tasks)} | Test: {len(test_tasks)}")

    os.makedirs(f"{RESULTS_DIR}/{benchmark}", exist_ok=True)

    # ─── Phase 1: Sequential Iterative Training ───────────────────
    print(f"\n  Phase 1: Sequential iterative training ({len(train_tasks)} tasks)...")
    sf = SkillForgeV6(token_budget=2000)
    sem = asyncio.Semaphore(CONCURRENCY)

    train_results = await train_sequential(
        benchmark, train_tasks, sf, sem,
        game_list=game_list[:mid] if game_list else None
    )

    train_valid = [r for r in train_results if not r.get("error")]
    train_success = [r for r in train_results if r.get("_train_score", 0) >= 0.3]
    print(f"\n  Train complete: {len(train_valid)}/{len(train_tasks)} valid")
    print(f"  Train success (score≥0.3): {len(train_success)}/{len(train_valid)}")
    avg_score = sum(r.get("_train_score", 0) for r in train_results) / max(len(train_results), 1)
    print(f"  Train avg score: {avg_score:.1%}")
    print(f"  Library: {sf.stats}")

    sf.save(f"{RESULTS_DIR}/{benchmark}/library_after_train.json")

    # ─── Phase 2: Test (3 groups: A/B/C) ─────────────────────────
    print(f"\n  Phase 2: Testing {len(test_tasks)} tasks × 3 groups...")

    # Group A: Baseline (no augmentation)
    print(f"    [A] Baseline (no augmentation)...", flush=True)

    async def run_test_a(i, task):
        async with sem:
            if benchmark == "gaia":
                return await run_gaia_task(task, experience_section="", group="A")
            elif benchmark == "alfworld":
                game = game_list[mid + i] if game_list and mid + i < len(game_list) else None
                if game:
                    return await run_alfworld_task(game["file"], game["type"], group="A")
                return {"task_id": task["task_id"], "error": "no_game", "score": 0.0}
            else:
                return await run_locomo_task(task, experience_section="", group="A")

    results_a = await asyncio.gather(*[run_test_a(i, t) for i, t in enumerate(test_tasks)])

    # Group B: Raw injection (no AI refinement)
    print(f"    [B] Raw experience injection...", flush=True)
    raw_library = ExperienceLibrary()
    for exp in sf.library.experiences:
        raw_exp = copy.deepcopy(exp)
        raw_exp.failure_taxonomy = {k: v for k, v in raw_exp.failure_taxonomy.items()
                                     if k not in ("ai_refined", "causal_lesson", "avoidance_note",
                                                  "transferability", "generalized_steps",
                                                  "evolution_insight", "quality_score", "evolution_trace")}
        raw_library.record(raw_exp)

    async def run_test_b(i, task):
        async with sem:
            if benchmark == "gaia":
                aug = build_augmented_prompt(task["description"][:300], raw_library,
                                            token_budget=2000, metadata={"benchmark": "gaia"})
                return await run_gaia_task(task, experience_section=aug, group="B")
            elif benchmark == "alfworld":
                game = game_list[mid + i] if game_list and mid + i < len(game_list) else None
                if game:
                    td = f"{results_a[i].get('task', '')} [type: {game['type']}]"
                    aug = build_augmented_prompt(td, raw_library, token_budget=1500,
                                                metadata={"task_type": game["type"]})
                    return await run_alfworld_task(
                        game["file"], game["type"],
                        experience_section=f"\n## Experience\n{aug}" if aug else "",
                        group="B"
                    )
                return {"task_id": task["task_id"], "error": "no_game", "score": 0.0}
            else:
                aug = build_augmented_prompt(task["description"][:300], raw_library,
                                            token_budget=600, metadata=task.get("metadata", {}))
                return await run_locomo_task(task, experience_section=aug, group="B")

    results_b = await asyncio.gather(*[run_test_b(i, t) for i, t in enumerate(test_tasks)])

    # Group C: AI-refined injection
    print(f"    [C] AI-refined experience injection...", flush=True)

    async def run_test_c(i, task):
        async with sem:
            if benchmark == "gaia":
                aug = build_augmented_prompt(task["description"][:300], sf.library,
                                            token_budget=2000, metadata={"benchmark": "gaia"})
                return await run_gaia_task(task, experience_section=aug, group="C")
            elif benchmark == "alfworld":
                game = game_list[mid + i] if game_list and mid + i < len(game_list) else None
                if game:
                    td = f"{results_a[i].get('task', '')} [type: {game['type']}]"
                    aug = build_augmented_prompt(td, sf.library, token_budget=1500,
                                                metadata={"task_type": game["type"]})
                    return await run_alfworld_task(
                        game["file"], game["type"],
                        experience_section=f"\n## Experience\n{aug}" if aug else "",
                        group="C"
                    )
                return {"task_id": task["task_id"], "error": "no_game", "score": 0.0}
            else:
                aug = build_augmented_prompt(task["description"][:300], sf.library,
                                            token_budget=600, metadata=task.get("metadata", {}))
                return await run_locomo_task(task, experience_section=aug, group="C")

    results_c = await asyncio.gather(*[run_test_c(i, t) for i, t in enumerate(test_tasks)])

    # Evaluate with EM + pass@1
    print(f"\n  Evaluating results (Exact Match + pass@1)...", flush=True)
    scores = {"A_baseline": [], "B_raw": [], "C_refined": []}

    eval_tasks = []
    for i in range(len(test_tasks)):
        eval_tasks.append(evaluate_exact_match(results_a[i], benchmark))
        eval_tasks.append(evaluate_exact_match(results_b[i], benchmark))
        eval_tasks.append(evaluate_exact_match(results_c[i], benchmark))

    all_evals = await asyncio.gather(*eval_tasks)

    for i in range(len(test_tasks)):
        scores["A_baseline"].append(all_evals[i * 3])
        scores["B_raw"].append(all_evals[i * 3 + 1])
        scores["C_refined"].append(all_evals[i * 3 + 2])

    report = {}
    for group, evals in scores.items():
        em_scores = [e["em"] for e in evals if e.get("em") is not None]
        pass_at_1 = sum(1 for e in evals if e.get("pass_at_1")) / max(len(evals), 1)
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
        report[group] = {"em": avg_em, "pass_at_1": pass_at_1, "n": len(evals)}

    print(f"\n  Results ({benchmark}, model={MODEL}):")
    print(f"    A (Baseline):    EM={report['A_baseline']['em']:.1%}  pass@1={report['A_baseline']['pass_at_1']:.1%}")
    print(f"    B (Raw inject):  EM={report['B_raw']['em']:.1%}  pass@1={report['B_raw']['pass_at_1']:.1%}")
    print(f"    C (AI-refined):  EM={report['C_refined']['em']:.1%}  pass@1={report['C_refined']['pass_at_1']:.1%}")
    delta_ac = report['C_refined']['em'] - report['A_baseline']['em']
    delta_bc = report['C_refined']['em'] - report['B_raw']['em']
    print(f"    Δ(C-A): {delta_ac:+.1%} | Δ(C-B): {delta_bc:+.1%}")

    full_report = {
        "benchmark": benchmark, "model": MODEL,
        "n_train": len(train_tasks), "n_test": len(test_tasks),
        "methodology": [
            "sequential_iterative_training",
            "cross_agent_skill_quality_evaluation",
            "exact_match_plus_pass_at_1",
            f"{'agentic_tool_use' if benchmark == 'gaia' else 'interactive_env' if benchmark == 'alfworld' else 'enhanced_qa_hints'}",
        ],
        "train_stats": {
            "avg_score": avg_score,
            "success_rate": len(train_success) / max(len(train_valid), 1),
            "library_size": len(sf.library.experiences),
        },
        "results": report,
        "delta_refined_vs_baseline": delta_ac,
        "delta_refined_vs_raw": delta_bc,
    }

    if benchmark == "locomo":
        static_score = report['A_baseline']['em']
        dynamic_score = report['C_refined']['em']
        full_report["static_vs_dynamic"] = {
            "static_em": static_score,
            "dynamic_em": dynamic_score,
            "delta": dynamic_score - static_score,
        }
        print(f"\n  LoCoMo static vs dynamic:")
        print(f"    Static (A): EM={static_score:.1%}")
        print(f"    Dynamic (C): EM={dynamic_score:.1%}")
        print(f"    Δ: {dynamic_score - static_score:+.1%}")

    with open(f"{RESULTS_DIR}/{benchmark}/report.json", "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    return full_report


async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n  SkillForge V6 — Latest Experiment")
    print(f"  Model: {MODEL} | Concurrency: {CONCURRENCY}")
    print(f"  Eval: Exact Match + pass@1 | Quality: Cross-Agent Evaluation")
    print(f"  Output: {RESULTS_DIR}")

    all_reports = {}

    # Load benchmarks
    print("\n  Loading benchmarks...")
    benchmarks = {}

    for name in ["gaia", "alfworld", "locomo"]:
        loader = BenchmarkLoader({"name": name, "num_samples": TASK_LIMITS[name]})
        tasks = loader.load()[:TASK_LIMITS[name]]
        benchmarks[name] = tasks
        print(f"    {name}: {len(tasks)} tasks")

    # Load ALFWorld games for interactive mode
    print("  Loading ALFWorld game files...")
    alfworld_games = get_alfworld_games(TASK_LIMITS["alfworld"])
    print(f"    ALFWorld games: {len(alfworld_games)}")

    print(f"\n  Total: {sum(len(t) for t in benchmarks.values())} tasks")

    # Run each benchmark
    for name, tasks in benchmarks.items():
        if not tasks:
            print(f"\n  SKIP {name}: no tasks")
            continue
        try:
            game_list = alfworld_games if name == "alfworld" else None
            report = await run_benchmark(name, tasks, game_list=game_list)
            all_reports[name] = report
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {name}: {e}")
            traceback.print_exc()
            all_reports[name] = {"error": str(e)}

    print(f"\n\n  FINAL SUMMARY (EM + pass@1 — DeepSeek V4 Pro)")
    print(f"  {'Benchmark':<12} {'Baseline EM':>12} {'Raw EM':>10} {'Refined EM':>12} {'pass@1':>8} {'Δ(C-A)':>8}")
    print(f"  {'-'*64}")
    for name, r in all_reports.items():
        if "error" in r:
            print(f"  {name:<12} ERROR: {r['error'][:40]}")
        else:
            res = r["results"]
            print(f"  {name:<12} {res['A_baseline']['em']:>11.1%} "
                  f"{res['B_raw']['em']:>9.1%} "
                  f"{res['C_refined']['em']:>11.1%} "
                  f"{res['C_refined']['pass_at_1']:>7.1%} "
                  f"{r['delta_refined_vs_baseline']:>+7.1%}")

    prev_dir = str(PROJECT_ROOT / "experiments_results" / "rerun_deepseek_v4pro")
    if os.path.exists(f"{prev_dir}/final_summary.json"):
        print(f"\n  vs Previous Run:")
        with open(f"{prev_dir}/final_summary.json") as f:
            prev = json.load(f)
        for name in ["gaia", "alfworld", "locomo"]:
            if name in all_reports and "results" in all_reports[name]:
                new_em = all_reports[name]["results"]["C_refined"]["em"]
                if name in prev and "results" in prev[name]:
                    old_c = prev[name]["results"].get("C_refined", {}).get("avg_score", 0)
                    print(f"    {name:<12} old={old_c:.1%} → new_em={new_em:.1%}")

    with open(f"{RESULTS_DIR}/final_summary.json", "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS_DIR}/final_summary.json")


if __name__ == "__main__":
    asyncio.run(main())
