#!/usr/bin/env python3
"""
ALFWorld Interactive Runner — SkillForge V6 with real environment interaction.

Each game runs in its own subprocess with a SINGLE game file,
avoiding the game-switching bug in AlfworldEnv.reset(game_idx).

Architecture:
  main (Python 3.14 + SDK) ←→ per-game subprocess (Python 3.9 + alfworld)
"""
import asyncio
import json
import os
import sys
import time
import copy
import re
import concurrent.futures

sys.path.insert(0, '/data/home/yizhouyang/workspace/SkillForge/src')
sys.path.insert(0, '/data/home/yizhouyang/workspace/SkillForge')

# CODEBUDDY_API_KEY must be set in environment
os.environ['CODEBUDDY_INTERNET_ENVIRONMENT'] = 'ioa'

from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage
from v6 import (SkillForgeV6, ExperienceLibrary,
                analyze_execution, build_augmented_prompt, ai_review_experience)

MODEL = "hy3-preview-ioa"
MAX_STEPS = 30
CONCURRENCY = 5
RESULTS_DIR = "/data1/benchmarks/unified_v6_results/alfworld_interactive"
ALFWORLD_PYTHON = "/data/home/yizhouyang/workspace/SkillForge/.venv_alfworld/bin/python"
ALFWORLD_DATA = "/data/home/yizhouyang/workspace/SkillForge/.venv_alfworld/data"


# ══════════════════════════════════════════════════════════════════════════
#  ALFWorld subprocess: one game file per process
# ══════════════════════════════════════════════════════════════════════════

GAME_WORKER = r'''
import sys, os, json

os.environ["ALFWORLD_DATA"] = "{alfworld_data}"
game_file = "{game_file}"
max_steps = {max_steps}

# Redirect stdout→stderr during init
_saved_fd = os.dup(1)
os.dup2(2, 1)

import textworld
ri = textworld.EnvInfos(won=True, admissible_commands=True, description=True, inventory=True)
env = textworld.start(game_file, request_infos=ri)

# Restore stdout
os.dup2(_saved_fd, 1)
os.close(_saved_fd)
_out = os.fdopen(1, "w", buffering=1)

step_count = 0

_out.write(json.dumps({{"type":"ready"}}) + "\n"); _out.flush()

for line in sys.stdin:
    c = json.loads(line)
    if c["action"] == "reset":
        gs = env.reset()
        step_count = 0
        adm = list(gs.admissible_commands or [])
        _out.write(json.dumps({{"type":"obs","obs":gs.feedback,"actions":adm[:30],
                                "won":bool(gs.won),"done":bool(gs.won)}}) + "\n"); _out.flush()
    elif c["action"] == "step":
        gs, score, done = env.step(c["command"])
        step_count += 1
        adm = list(gs.admissible_commands or [])
        won = bool(gs.won)
        done = done or won or step_count >= max_steps
        _out.write(json.dumps({{"type":"obs","obs":gs.feedback,"actions":adm[:30],
                                "won":won,"done":done}}) + "\n"); _out.flush()
    elif c["action"] == "quit":
        break
env.close()
'''


def get_game_list():
    """Probe ALFWorld to get all game files + types."""
    import subprocess as sp
    code = f'''
import sys, os, json, io
os.environ["ALFWORLD_DATA"] = "{ALFWORLD_DATA}"
sys.path.insert(0, "/data/home/yizhouyang/workspace/SkillForge/src")
_o = sys.stdout; sys.stdout = io.StringIO()
from utils.alfworld_env import AlfworldEnv, task_type_from_gamefile
env = AlfworldEnv(split="valid_unseen", max_steps=30)
sys.stdout = _o
print(json.dumps([{{"file":g,"type":task_type_from_gamefile(g)}} for g in env.list_tasks()]))
'''
    r = sp.run([ALFWORLD_PYTHON, "-c", code], capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout.strip())


class SingleGameEnv:
    """One ALFWorld game in a subprocess."""
    def __init__(self, game_file):
        import subprocess as sp
        code = GAME_WORKER.format(alfworld_data=ALFWORLD_DATA, game_file=game_file, max_steps=MAX_STEPS)
        self.proc = sp.Popen([ALFWORLD_PYTHON, "-c", code],
                             stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE,
                             text=True, bufsize=1)
        line = self.proc.stdout.readline()
        assert json.loads(line).get("type") == "ready", f"Init failed: {self.proc.stderr.read()[:300]}"

    def reset(self):
        self.proc.stdin.write('{"action":"reset"}\n'); self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def step(self, cmd):
        self.proc.stdin.write(json.dumps({"action":"step","command":cmd})+"\n"); self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        try:
            self.proc.stdin.write('{"action":"quit"}\n'); self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ══════════════════════════════════════════════════════════════════════════
#  LLM action decision (runs in thread to isolate event loop)
# ══════════════════════════════════════════════════════════════════════════

def _llm_call_sync(prompt: str) -> str:
    """Run SDK query in a fresh event loop in current thread.
    Explicitly kills spawned headless process after each call to prevent leak.
    """
    import signal
    
    # Track child PIDs before call
    import subprocess as _sp
    before = set(_sp.run("pgrep -f 'codebuddy-headless.*max-turns.1'",
                         shell=True, capture_output=True, text=True).stdout.split())
    
    async def _inner():
        opt = CodeBuddyAgentOptions(permission_mode="bypassPermissions", model=MODEL, max_turns=1, cwd="/tmp")
        result = ""
        try:
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

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_inner())
    except Exception:
        result = ""
    finally:
        try:
            loop.close()
        except Exception:
            pass
    
    # Kill any NEW headless processes spawned by this call
    after = set(_sp.run("pgrep -f 'codebuddy-headless.*max-turns.1'",
                        shell=True, capture_output=True, text=True).stdout.split())
    new_pids = after - before
    for pid in new_pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
    
    return result


def _simplify_action(action: str) -> str:
    """Simplify ALFWorld action by stripping coordinate IDs for display."""
    # "go to cabinet_bar__minus_00_dot_49..." → "go to cabinet 1"
    cleaned = re.sub(r'_bar__(?:minus|plus)_\d+_dot_\d+(?:_bar__(?:minus|plus)_\d+_dot_\d+)*', '', action)
    cleaned = re.sub(r'_+', ' ', cleaned).strip()
    return cleaned


async def llm_decide_action(observation, task, admissible_actions, history, experience_section=""):
    history_str = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history[-10:]))
    
    # Simplify observation (strip coordinate noise)
    obs_simple = re.sub(r'_bar__(?:minus|plus)_\d+_dot_\d+(?:_bar__(?:minus|plus)_\d+_dot_\d+)*', '', observation)
    obs_simple = re.sub(r'_+', ' ', obs_simple)
    
    # Simplify actions for display, with index
    action_lines = []
    for i, a in enumerate(admissible_actions[:20]):
        action_lines.append(f"  {i}. {_simplify_action(a)}")
    
    # Simplify history
    history_simple = "\n".join(f"  {i+1}. {_simplify_action(h)}" for i, h in enumerate(history[-8:]))

    prompt = f"""You are an AI agent completing a household task. Choose the best next action by its NUMBER.

## Task
{task}

## Current Observation
{obs_simple[:500]}

## Action History
{history_simple if history else "(none yet)"}

## Available Actions (choose by number)
{chr(10).join(action_lines)}
{experience_section}
## Response
Output ONLY the number (0-{min(len(admissible_actions), 20)-1}) of your chosen action. Nothing else."""

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            result = await loop.run_in_executor(pool, _llm_call_sync, prompt)
        except Exception:
            result = ""

    # Parse number from response
    result_clean = result.strip()
    # Try to extract a number
    m = re.search(r'\b(\d+)\b', result_clean)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(admissible_actions):
            return admissible_actions[idx]
    
    # Fallback: fuzzy match on simplified text
    result_lower = result_clean.lower()
    best, best_s = (admissible_actions[0] if admissible_actions else "look"), 0
    for a in admissible_actions:
        simple = _simplify_action(a).lower()
        # Check if response contains key words from the action
        s = len(set(simple.split()) & set(result_lower.split()))
        if s > best_s:
            best_s, best = s, a
    return best


# ══════════════════════════════════════════════════════════════════════════
#  Run one game
# ══════════════════════════════════════════════════════════════════════════

async def run_game(game_file, game_type, experience_section=""):
    env = SingleGameEnv(game_file)
    try:
        info = env.reset()
        obs = info["obs"]
        # Extract task from initial observation
        m = re.search(r"Your task is to:\s*(.+)", obs)
        task = m.group(1).strip() if m else game_type
        trajectory = []

        for _ in range(MAX_STEPS):
            admissible = info.get("actions", ["look"])
            if not admissible:
                admissible = ["look"]
            action = await llm_decide_action(obs, task, admissible,
                                              [t[0] for t in trajectory], experience_section)
            info = env.step(action)
            obs = info["obs"]
            trajectory.append((action, obs))
            if info.get("won") or info.get("done"):
                break

        return {
            "task": task, "task_type": game_type,
            "won": info.get("won", False),
            "steps": len(trajectory),
            "trajectory": trajectory,
            "score": 1.0 if info.get("won", False) else 0.0,
        }
    except Exception as e:
        return {"task": game_type, "task_type": game_type, "won": False,
                "steps": 0, "trajectory": [], "score": 0.0, "error": str(e)[:200]}
    finally:
        env.close()


# ══════════════════════════════════════════════════════════════════════════
#  Main experiment
# ══════════════════════════════════════════════════════════════════════════

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  ALFWorld Interactive — SkillForge V6                        ║")
    print(f"║  Model: {MODEL} | Steps: {MAX_STEPS} | Concurrency: {CONCURRENCY}       ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    print("\n  Probing ALFWorld games...", flush=True)
    games = get_game_list()
    total = min(len(games), 40)
    games = games[:total]
    mid = total // 2
    train_games = games[:mid]
    test_games = games[mid:]
    print(f"  Total: {total} | Train: {len(train_games)} | Test: {len(test_games)}")

    # ─── Phase 1: Train ───────────────────────────────────────────
    print(f"\n  Phase 1: Train ({len(train_games)} games, concurrency={CONCURRENCY})...", flush=True)
    sf = SkillForgeV6(token_budget=1500)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def train_one(g):
        async with sem:
            return await run_game(g["file"], g["type"])

    train_results = await asyncio.gather(*[train_one(g) for g in train_games])

    for i, r in enumerate(train_results):
        s = "\u2713" if r["won"] else "\u2717"
        print(f"    {s} {r['task_type']:<12} steps={r['steps']:>2} | {r['task'][:50]}", flush=True)
        agent_actions = [{"command": a, "observation": o[:200]} for a, o in r["trajectory"]]
        oracle_actions = [{"command": a} for a, _ in r["trajectory"]] if r["won"] else []
        sf.record_experience(
            task_id=f"alf_train_{i}", task_desc=f"{r['task']} [type: {r['task_type']}]",
            agent_actions=agent_actions, oracle_actions=oracle_actions,
            token_cost=r["steps"]*100, time_cost=r["steps"]*2.0, llm_reviewer=None,
        )

    # Passthrough refine
    for exp in sf.library.experiences:
        rv = ai_review_experience(exp, llm_fn=None)
        exp.failure_taxonomy.update({k: rv.get(k, "") for k in
            ("ai_refined", "causal_lesson", "avoidance_note", "transferability",
             "generalized_steps", "evolution_insight")})

    sf.save(f"{RESULTS_DIR}/library.json")
    n_won = sum(1 for r in train_results if r["won"])
    print(f"\n  Train: {n_won}/{len(train_results)} won | Library: {sf.stats}", flush=True)

    # ─── Phase 2: Test ─────────────────────────────────────────────
    print(f"\n  Phase 2: Test ({len(test_games)} games × 3 groups)...", flush=True)

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
    async def test_a(g):
        async with sem:
            return await run_game(g["file"], g["type"])
    results_a = await asyncio.gather(*[test_a(g) for g in test_games])

    # B: Raw
    print("    [B] Raw injection...", flush=True)
    async def test_b(i, g):
        async with sem:
            td = f"{results_a[i]['task']} [type: {g['type']}]"
            aug = build_augmented_prompt(td, raw_lib, token_budget=1500, metadata={"task_type": g["type"]})
            return await run_game(g["file"], g["type"],
                                   experience_section=f"\n## Experience\n{aug}" if aug else "")
    results_b = await asyncio.gather(*[test_b(i, g) for i, g in enumerate(test_games)])

    # C: AI-refined
    print("    [C] AI-refined injection...", flush=True)
    async def test_c(i, g):
        async with sem:
            td = f"{results_a[i]['task']} [type: {g['type']}]"
            aug = build_augmented_prompt(td, sf.library, token_budget=1500, metadata={"task_type": g["type"]})
            return await run_game(g["file"], g["type"],
                                   experience_section=f"\n## Experience\n{aug}" if aug else "")
    results_c = await asyncio.gather(*[test_c(i, g) for i, g in enumerate(test_games)])

    # ─── Results ───────────────────────────────────────────────────
    sr = lambda rs: sum(r["won"] for r in rs) / len(rs) if rs else 0
    sa, sb, sc = sr(results_a), sr(results_b), sr(results_c)

    print(f"\n{'='*60}")
    print(f"  ALFWorld Interactive Results (n={len(test_games)})")
    print(f"{'='*60}")
    print(f"  A (Baseline):    {sa:.1%} ({sum(r['won'] for r in results_a)}/{len(results_a)})")
    print(f"  B (Raw inject):  {sb:.1%} ({sum(r['won'] for r in results_b)}/{len(results_b)})")
    print(f"  C (AI-refined):  {sc:.1%} ({sum(r['won'] for r in results_c)}/{len(results_c)})")
    print(f"  Δ(C-A): {sc-sa:+.1%} | Δ(C-B): {sc-sb:+.1%}")

    # Per type
    types = sorted(set(r["task_type"] for r in results_a))
    print(f"\n  Per task-type:")
    for tt in types:
        fa = [r for r in results_a if r["task_type"]==tt]
        fb = [r for r in results_b if r["task_type"]==tt]
        fc = [r for r in results_c if r["task_type"]==tt]
        print(f"    {tt:<12} A={sr(fa):.0%} B={sr(fb):.0%} C={sr(fc):.0%} Δ(C-A)={sr(fc)-sr(fa):+.0%}")

    report = {
        "benchmark": "alfworld_interactive", "n_train": len(train_games), "n_test": len(test_games),
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
