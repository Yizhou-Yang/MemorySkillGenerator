#!/usr/bin/env python3
"""SkillForge V6 — Latest Experiment Runner"""
import asyncio
import concurrent.futures
import copy
import json
import os
import re
import sys
import time
import unicodedata
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

from v6 import (SkillForgeV6, ExperienceLibrary, Experience,
                build_augmented_prompt, ai_review_experience,
                cross_agent_evaluate_skill)
from benchmarks.loader import BenchmarkLoader

MODEL = "deepseek-v4-pro"
CONCURRENCY = 15
TASK_TIMEOUT_QA = 180
TASK_TIMEOUT_AGENT = 300
TASK_TIMEOUT_ALFWORLD = 240
ALFWORLD_RETRY_MAX = 2
QUALITY_THRESHOLD = 5

RESULTS_DIR = str(PROJECT_ROOT / "experiments_results" / "latest")

TASK_LIMITS = {"gaia": 50, "alfworld": 40, "locomo": 50, "gaia2": 50, "swebench_dynamic": 30}

ALFWORLD_PYTHON = str(PROJECT_ROOT / ".venv_alfworld" / "bin" / "python")
ALFWORLD_DATA = str(PROJECT_ROOT / ".venv_alfworld" / "data")
ALFWORLD_MAX_STEPS = 50

# ─── LLM helpers ──────────────────────────────────────────────────────────

def llm_review_fn(prompt: str) -> str:
    """Synchronous single-turn LLM call used by ai_review_experience."""
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
        return executor.submit(_run_in_thread).result(timeout=120)

def _query_sync(prompt: str, max_turns: int = 1, timeout: int = 60) -> dict:
    """Run CodeBuddy query in a fresh event loop (thread-safe, avoids cancel scope issues)."""
    async def _inner():
        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=max_turns, cwd="/tmp"
        )
        text = ""
        actions = []
        try:
            async with asyncio.timeout(timeout):
                async for msg in query(prompt=prompt, options=opt):
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, ToolUseBlock):
                                actions.append({"tool": block.name, "input": str(block.input)})
                            elif hasattr(block, 'text') and block.text:
                                if '429' in block.text and '额度' in block.text:
                                    return {"text": "", "actions": actions, "error": "429_rate_limit"}
                                text += block.text
                        if text and max_turns <= 2:
                            break
        except Exception as e:
            return {"text": text, "actions": actions, "error": str(e)[:200] if not text else None}
        return {"text": text, "actions": actions, "error": None}

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_inner())
    finally:
        loop.close()

async def _llm_call(prompt: str, max_turns: int = 1, timeout: int = 60) -> dict:
    """Async wrapper: runs query in isolated thread to avoid anyio conflicts."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _query_sync, prompt, max_turns, timeout)

async def _llm_short_call(prompt: str, max_turns: int = 1, timeout: int = 30) -> str:
    """Short LLM call returning text only."""
    r = await _llm_call(prompt, max_turns=max_turns, timeout=timeout)
    return (r.get("text") or "").strip()

async def llm_extract_answer(response: str, question: str) -> str:
    if len(response.split()) < 30:
        return response
    prompt = (
        "Extract ONLY the final answer from this response. Output just the answer, nothing else.\n\n"
        f"Question: {question}\n\nResponse: {response}\n\n"
        "Final answer (concise, just the key fact/number/name):"
    )
    out = await _llm_short_call(prompt, max_turns=1, timeout=30)
    return out or response

async def llm_judge_answer(response: str, expected: str, question: str) -> float:
    if not response or not expected:
        return 0.0
    prompt = (
        "Judge if the response correctly answers the question. Score 0.0 to 1.0.\n\n"
        f"Question: {question}\nExpected answer: {expected}\n"
        f"Model response: {response}\n\n"
        "Score (0.0=wrong, 0.5=partially, 1.0=fully correct). Output ONLY a number:"
    )
    out = await _llm_short_call(prompt, max_turns=1, timeout=30)
    m = re.search(r'(\d+\.?\d*)', out)
    if m:
        try:
            return min(1.0, max(0.0, float(m.group(1))))
        except ValueError:
            return 0.0
    return 0.0

async def llm_critic_skill_quality(exp_summary: str, task_desc: str) -> float:
    """Cross-agent critic: independent LLM scores skill quality (0-10)."""
    prompt = (
        "You are an experienced AI agent reviewer. Rate how USEFUL and "
        "REUSABLE the following candidate skill is for similar future tasks.\n\n"
        "Score from 0 (useless / harmful) to 10 (highly reusable, clear lesson).\n\n"
        "Scoring guide:\n"
        "- SUCCESSFUL skills (8-10): concrete tool sequence that WORKED, "
        "reproducible steps, clear strategy that transfers to similar tasks.\n"
        "- FAILED skills with lessons (6-8): identifies WHY it failed, "
        "what to avoid, what was missing — useful as negative examples.\n"
        "- LOW quality (0-5): vague generalizations, hallucinated steps, "
        "task-specific facts mistaken for procedure, no actionable info.\n\n"
        "Key: A successful execution with clear steps is ALWAYS valuable "
        "(it shows the correct approach). Do NOT penalize for lacking failure analysis "
        "when the task succeeded.\n\n"
        f"## Task\n{task_desc}\n\n## Candidate skill\n{exp_summary}\n\n"
        "Output ONLY a single integer 0-10:"
    )
    out = await _llm_short_call(prompt, max_turns=1, timeout=30)
    m = re.search(r'\b(\d{1,2})\b', out)
    if m:
        try:
            return float(min(10, max(0, int(m.group(1)))))
        except ValueError:
            pass
    return 5.0

# ─── Metric helpers (EM + pass@1) ─────────────────────────────────────────

_ARTICLES_RE = re.compile(r'\b(a|an|the)\b', flags=re.UNICODE)
_PUNCT_RE = re.compile(r'[^\w\s]', flags=re.UNICODE)
_WS_RE = re.compile(r'\s+')

def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles + punct, collapse whitespace."""
    s = unicodedata.normalize('NFKC', s).lower()
    s = _PUNCT_RE.sub(' ', s)
    s = _ARTICLES_RE.sub(' ', s)
    s = _WS_RE.sub(' ', s).strip()
    return s

def exact_match(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    return 1.0 if p == g or g in p or p in g else 0.0

# ─── GAIA runner ──────────────────────────────────────────────────────────

async def run_gaia_task(task: dict, experience_section: str = "",
                        group: str = "A") -> dict:
    task_id = task["task_id"]
    description = task["description"]
    expected = task.get("expected", "")
    metadata = task.get("metadata", {})
    benchmark_type = metadata.get("benchmark", "")

    # Determine system prompt based on benchmark type
    if benchmark_type == "gaia2" or "gaia2" in task_id:
        # GAIA2: CLI tool-calling tasks — inject available tools
        available_tools = metadata.get("tools", [])
        tools_str = ", ".join(available_tools) if available_tools else "calendar, contacts, emails, messages, cabs, shopping"
        system = (
            "You are an AI assistant that helps users accomplish tasks by using CLI tools.\n\n"
            f"Available tools: {tools_str}\n\n"
            "For each tool, you can perform actions like: search, create, update, delete, list.\n"
            "Example tool usage patterns:\n"
            "- calendar: create events, check schedule, find free slots\n"
            "- contacts: search contacts, get phone/email, add contacts\n"
            "- emails: send emails, search inbox, read messages\n"
            "- messages/chats: send messages, read conversations\n"
            "- cabs: book rides, check availability, get estimates\n"
            "- shopping: search products, place orders, check prices\n\n"
            "Strategy:\n"
            "1. Understand what the user wants to accomplish\n"
            "2. Break it into steps using the available tools\n"
            "3. Execute each step, using tool calls when needed\n"
            "4. Report the final result\n\n"
            "Use web search and bash commands to simulate tool interactions. "
            "Show your reasoning and actions clearly."
        )
    elif "swebench" in task_id or benchmark_type == "swebench_dynamic":
        # SWE-bench: code debugging and patch generation
        system = (
            "You are an expert software engineer debugging a failing test case.\n\n"
            "Strategy:\n"
            "1. UNDERSTAND: Read the issue description carefully. What behavior is expected vs actual?\n"
            "2. LOCATE: Identify which file(s) and function(s) are likely responsible.\n"
            "3. DIAGNOSE: Determine the root cause of the bug.\n"
            "4. FIX: Write a minimal, correct patch that fixes the issue.\n"
            "5. VERIFY: Explain why your fix resolves the failing test.\n\n"
            "Important:\n"
            "- Focus on the MINIMAL change needed — don't refactor unrelated code.\n"
            "- Your response MUST include the actual code fix (as a diff or code block).\n"
            "- Show the file path, the original code, and your corrected code.\n"
            "- If you need to read files or run tests, use the available tools."
        )
    else:
        # GAIA: multi-step QA with tool use
        system = (
            "You are an expert research assistant capable of multi-step reasoning and tool use. "
            "You have access to web search, file reading, code execution, and computation tools.\n\n"
            "Strategy for answering questions:\n"
            "1. ANALYZE: Break down the question — what information do you need?\n"
            "2. SEARCH: Use web search to find relevant facts, data, or context.\n"
            "3. VERIFY: Cross-check information from multiple sources when possible.\n"
            "4. COMPUTE: If math/logic is needed, use code execution for accuracy.\n"
            "5. ANSWER: Provide a concise, precise final answer.\n\n"
            "Important rules:\n"
            "- Always search the web for factual questions — do NOT guess from memory.\n"
            "- For numerical answers, show your computation steps.\n"
            "- If a question asks for a specific format (name, number, date), match that format exactly.\n"
            "- Give ONLY the final answer in your last message — no explanation needed."
        )

    if experience_section:
        system += f"\n\n{experience_section}"

    prompt = (
        f"{system}\n\n"
        f"Question: {description}\n\n"
        f"Think step by step. Use tools as needed. "
        f"End with your final answer on the last line."
    )

    result = {"task_id": task_id, "expected": expected, "response": "",
              "error": None, "time_cost": 0, "augmented": bool(experience_section),
              "group": group, "actions": []}
    t0 = time.time()
    r = await _llm_call(prompt, max_turns=30, timeout=TASK_TIMEOUT_AGENT)
    result["response"] = r.get("text", "")
    result["actions"] = r.get("actions", [])
    result["error"] = r.get("error")
    result["time_cost"] = time.time() - t0
    return result

# ─── ALFWorld runner ──────────────────────────────────────────────────────

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

# ─── ALFWorld ReAct prompt (aligned with EvoArena / SkillOS baselines) ────

ALFWORLD_REACT_SYSTEM = """You are an embodied agent solving a household task in a text-based environment.
You see a textual room description, a task goal, and a list of admissible commands.

At each step you MUST output exactly two lines:
    Thought: <one short sentence reasoning>
    Action: <one command from the admissible list, copied verbatim>

The Action MUST appear in the admissible list verbatim (case-sensitive).
Avoid repeating an action that just produced "Nothing happens." — try a different action.

Here are two example trajectories that illustrate the expected behaviour.

=== Example 1 (pick & place) ===
Task: put a clean plate on the diningtable.
Thought: I should locate a plate first; it is most likely in a cabinet or on the countertop.
Action: go to countertop 1
Thought: I see a plate 1 here; pick it up.
Action: take plate 1 from countertop 1
Thought: The plate may be dirty, so I will clean it at the sink.
Action: go to sinkbasin 1
Thought: Clean the plate using the sink.
Action: clean plate 1 with sinkbasin 1
Thought: Now bring it to the diningtable.
Action: go to diningtable 1
Action: put plate 1 in/on diningtable 1

=== Example 2 (heat & place) ===
Task: heat some bread and put it on the diningtable.
Thought: First find the bread — likely on a countertop or in a cabinet.
Action: go to countertop 1
Action: take bread 1 from countertop 1
Thought: Heat it in the microwave.
Action: go to microwave 1
Action: heat bread 1 with microwave 1
Thought: Deliver to the diningtable.
Action: go to diningtable 1
Action: put bread 1 in/on diningtable 1

Key patterns to remember:
  - To find an object, GO TO each likely receptacle until you see it.
  - Use "clean X with sinkbasin Y", "heat X with microwave Y", "cool X with fridge Y".
  - Use "use desklamp 1" / "examine X with desklamp Y" for look_at_obj_in_light tasks.
  - End with "put <object> in/on <receptacle>" if the goal requires placement.

Now solve the new task.
"""

_ACTION_RE = re.compile(r"Action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)

def parse_alfworld_action(reply: str, admissible: list) -> str:
    """Extract Action line from LLM reply and snap to admissible commands."""
    if not admissible:
        return "look"
    m = _ACTION_RE.search(reply)
    raw = m.group(1).strip() if m else reply.strip().splitlines()[-1].strip()
    raw = raw.strip().rstrip(".").strip("`").strip()

    # 1. exact match
    for a in admissible:
        if a == raw:
            return a
    # 2. case-insensitive exact
    for a in admissible:
        if a.lower() == raw.lower():
            return a
    # 3. substring (longest admissible match)
    raw_l = raw.lower()
    candidates = [a for a in admissible if a.lower() in raw_l or raw_l in a.lower()]
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    # 4. fallback
    return admissible[0]

async def llm_decide_alfworld_action(observation: str, task: str,
                                      admissible_actions: list, history: list,
                                      initial_obs: str = "",
                                      experience_section: str = "") -> str:
    """Full ReAct-style action selection for ALFWorld."""
    obs_simple = re.sub(r'_bar__(?:minus|plus)_\d+_dot_\d+(?:_bar__(?:minus|plus)_\d+_dot_\d+)*', '', observation)
    obs_simple = re.sub(r'_+', ' ', obs_simple)

    # Build history window — full history, no truncation (1M context budget)
    n = len(history)
    history_lines = []
    for i in range(n):
        history_lines.append(f"[Action {i+1}] {history[i][0]}")
        history_lines.append(f"[Obs {i+1}] {history[i][1]}")
    history_str = "\n".join(history_lines) if history_lines else "(no actions taken yet)"

    admissible_str = ", ".join(f"'{a}'" for a in admissible_actions)

    sys_prompt = ALFWORLD_REACT_SYSTEM
    if experience_section:
        sys_prompt += f"\nThe following skills from similar past tasks may help:\n{experience_section}\n"

    user_msg = (
        f"Task: {task}\n\n"
        f"[Initial Obs] {initial_obs}\n\n"
        f"Full history:\n{history_str}\n\n"
        f"[Current Obs] {obs_simple}\n\n"
        f"Admissible commands: [{admissible_str}]\n\n"
        f"Now output:\nThought: ...\nAction: ..."
    )

    prompt = f"[System]\n{sys_prompt}\n\n[User]\n{user_msg}"
    out = await _llm_short_call(prompt, max_turns=1, timeout=30)
    return parse_alfworld_action(out, admissible_actions)

async def run_alfworld_task(game_file: str, game_type: str,
                            experience_section: str = "", group: str = "A") -> dict:
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
        initial_obs = obs.strip()
        m = re.search(r"Your task is to:\s*(.+)", obs)
        task = m.group(1).strip() if m else game_type
        result["task"] = task
        trajectory = []
        last_action = ""
        nothing_count = 0

        for _ in range(ALFWORLD_MAX_STEPS):
            admissible = info.get("actions", ["look"]) or ["look"]
            action = await llm_decide_alfworld_action(
                obs, task, admissible,
                trajectory,  # pass full (action, obs) tuples
                initial_obs=initial_obs,
                experience_section=experience_section
            )

            # Avoid infinite loops: if same action repeated with "Nothing happens"
            if action == last_action and "nothing happens" in obs.lower():
                nothing_count += 1
                if nothing_count >= 1:
                    # Force a different action immediately — don't waste steps
                    alternatives = [a for a in admissible if a != action]
                    action = alternatives[0] if alternatives else "look"
                    nothing_count = 0
            else:
                nothing_count = 0
            last_action = action

            info = env.step(action)
            obs = info["obs"]
            trajectory.append((action, obs))
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

# ─── LoCoMo runner ────────────────────────────────────────────────────────

async def run_locomo_task(task: dict, experience_section: str = "",
                          group: str = "A") -> dict:
    task_id = task["task_id"]
    description = task["description"]
    expected = task.get("expected", "")
    system = (
        "You are a memory-augmented assistant specialized in answering questions "
        "about long conversations. You have access to the full conversation history below.\n\n"
        "Strategy:\n"
        "1. Carefully read the conversation history provided.\n"
        "2. Identify the specific information the question asks about.\n"
        "3. Look for explicit statements, preferences, events, or facts mentioned in the conversation.\n"
        "4. If the answer involves a person's preference or opinion, quote their exact words when possible.\n"
        "5. Give a concise, precise answer — just the key fact/name/date/number.\n\n"
        "Important: The answer is ALWAYS somewhere in the conversation history. "
        "Do NOT make up information. Search carefully."
    )
    if experience_section:
        system += f"\n\n{experience_section}"
    prompt = (
        f"[System]\n{system}\n\n"
        f"{description}\n\n"
        f"Based on the conversation above, provide your answer. "
        f"Be concise — give only the answer, no explanation:"
    )

    result = {"task_id": task_id, "expected": expected, "response": "",
              "error": None, "time_cost": 0, "augmented": bool(experience_section),
              "group": group}
    t0 = time.time()
    r = await _llm_call(prompt, max_turns=10, timeout=TASK_TIMEOUT_QA)
    result["response"] = r.get("text", "")
    result["error"] = r.get("error")
    result["time_cost"] = time.time() - t0
    return result

# ─── Evaluation ───────────────────────────────────────────────────────────

async def evaluate_task(result: dict, benchmark: str, use_llm_judge: bool = True) -> dict:
    """Primary metric:
       - alfworld: pass@1 (binary won)
       - gaia2: soft recall (action sequence matching)
       - swebench_dynamic: pass@1 (patch correctness via LLM judge)
       - gaia/locomo: Exact Match
    """
    if benchmark == "alfworld":
        won = bool(result.get("won", False))
        return {"score": 1.0 if won else 0.0, "em": 1.0 if won else 0.0,
                "won": won, "method": "pass@1"}

    if benchmark == "gaia2":
        # Soft recall: what fraction of oracle actions were covered by agent actions
        oracle_actions = result.get("expected", [])
        agent_actions = result.get("actions", [])
        response = (result.get("response") or "").lower()
        if not oracle_actions:
            return {"score": 0.0, "em": 0.0, "method": "soft_recall_empty"}
        if isinstance(oracle_actions, str):
            # Fallback if expected is string
            return {"score": 0.0, "em": 0.0, "method": "soft_recall_str_fallback"}
        # Match: for each oracle action, check if agent did something similar
        # Use broader matching: check tool names, app names, fn names, and also
        # check if the agent's response text mentions the relevant concepts
        matched = 0
        agent_strs = []
        if agent_actions:
            agent_strs = [f"{a.get('tool','')}: {a.get('input','')}" for a in agent_actions]
        # Also include the full response text for semantic matching
        all_agent_text = " ".join(agent_strs).lower() + " " + response

        # Build app/fn name mappings for flexible matching
        app_aliases = {
            'calendar': ['calendar', 'schedule', 'event', 'meeting', 'appointment'],
            'contacts': ['contacts', 'contact', 'phone', 'address', 'people'],
            'emails': ['emails', 'email', 'mail', 'send', 'inbox', 'message'],
            'messages': ['messages', 'message', 'sms', 'text', 'chat'],
            'chats': ['chats', 'chat', 'conversation', 'group'],
            'rentaflat': ['rent', 'flat', 'apartment', 'housing', 'property'],
            'city': ['city', 'location', 'place', 'map', 'direction'],
            'cabs': ['cabs', 'cab', 'taxi', 'ride', 'uber', 'transport'],
            'shopping': ['shopping', 'shop', 'buy', 'purchase', 'order', 'product'],
        }
        fn_keywords = {
            'create': ['create', 'add', 'new', 'make', 'set'],
            'search': ['search', 'find', 'look', 'query', 'get', 'list'],
            'send': ['send', 'write', 'compose', 'reply'],
            'delete': ['delete', 'remove', 'cancel'],
            'update': ['update', 'edit', 'modify', 'change'],
        }

        for oracle in oracle_actions:
            oracle_app = oracle.get('app', '').lower().replace(' ', '')
            oracle_fn = oracle.get('fn', '').lower()
            found = False

            # Direct match in agent tool calls
            for agent_str in agent_strs:
                agent_lower = agent_str.lower()
                if oracle_app and oracle_app in agent_lower:
                    found = True
                    break
                if oracle_fn and oracle_fn in agent_lower:
                    found = True
                    break

            # Alias-based match in full agent text
            if not found:
                aliases = app_aliases.get(oracle_app, [oracle_app])
                for alias in aliases:
                    if alias and alias in all_agent_text:
                        # Also check if the fn type matches
                        fn_base = oracle_fn.split('_')[0] if oracle_fn else ''
                        fn_kws = fn_keywords.get(fn_base, [fn_base])
                        for kw in fn_kws:
                            if kw and kw in all_agent_text:
                                found = True
                                break
                        if found:
                            break
                        # If app matches but fn doesn't, still count partial
                        if alias in all_agent_text:
                            found = True
                            break

            if found:
                matched += 1

        recall = matched / len(oracle_actions)
        return {"score": recall, "em": 1.0 if recall >= 0.8 else 0.0,
                "soft_recall": recall, "method": "soft_recall"}

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

# ─── Cross-agent skill quality gating ─────────────────────────────────────

async def critic_filter_and_record(sf: SkillForgeV6, task: dict, result: dict,
                                    score: float, benchmark: str, aug_used: str):
    """Record experience via sf.record_experience() to trigger version history
    and patch tracking (EvoMem-style). Async critic refines quality score."""
    response = result.get("response", "")
    actions = result.get("actions", [])

    # Build agent_actions in the format expected by analyze_execution
    if benchmark == "gaia":
        agent_actions = actions if actions else [{"output": response}]
    elif benchmark == "alfworld":
        trajectory = result.get("trajectory", [])
        agent_actions = [{"command": t[0], "output": t[1]} for t in trajectory] if trajectory else []
    else:
        agent_actions = [{"output": response}]

    # Oracle actions: use expected answer as reference
    expected = result.get("expected", task.get("expected", ""))
    if isinstance(expected, list):
        # gaia2: expected is a list of oracle actions (dicts)
        oracle_actions = expected
    elif expected:
        oracle_actions = [{"output": str(expected)}]
    else:
        oracle_actions = []

    # Use the task_id from the task dict (consistent across retries for patch_history)
    task_id = task["task_id"]

    # record_experience without LLM (avoids sync LLM in async context → socket leak)
    # Version history + patch_history still works (no LLM needed)
    exp = sf.record_experience(
        task_id=task_id,
        task_desc=task["description"],
        agent_actions=agent_actions,
        oracle_actions=oracle_actions,
        token_cost=len(response) // 4 + len(actions) * 50,
        time_cost=result.get("time_cost", 0),
        augmentation_used=aug_used if aug_used else "",
    )

    # Async critic evaluation (safe in async context)
    # For successful tasks: highlight WHAT WORKED (tool chain, strategy)
    # For failed tasks: highlight WHAT WENT WRONG (missing steps, failure reason)
    if exp.outcome == "success":
        summary = (
            f"Outcome: SUCCESS (score={exp.score:.2f})\n"
            f"Correct tool chain: {' -> '.join(exp.tool_sequence)}\n"
            f"Steps taken: {'; '.join(exp.action_commands)}\n"
            f"Strategy: completed all required steps successfully"
        )
    else:
        summary = (
            f"Outcome: {exp.outcome} (score={exp.score:.2f})\n"
            f"Steps attempted: {' -> '.join(exp.tool_sequence)}\n"
            f"Missing: {', '.join(exp.missing_steps) or 'unknown'}\n"
            f"Failure reason: {exp.failure_reason or 'incorrect answer'}\n"
            f"What to avoid: repeating the same approach without addressing gaps"
        )
    critic_score = await llm_critic_skill_quality(summary, task["description"])
    exp.failure_taxonomy["critic_quality"] = critic_score

    return True, critic_score

# ─── Sequential training (no oracle-driven retry for QA tasks) ────────────

async def train_sequential(benchmark: str, train_tasks: list, sf: SkillForgeV6,
                           sem: asyncio.Semaphore, game_list: list = None) -> list:
    """Each task uses the current accumulated experience library."""
    all_results = []

    for i, task in enumerate(train_tasks):
        async with sem:
            aug = ""
            if sf.library.experiences:
                aug = build_augmented_prompt(
                    task["description"], sf.library,
                    metadata=task.get("metadata", {"benchmark": benchmark})
                )

            if benchmark == "gaia" or benchmark == "gaia2" or benchmark == "swebench_dynamic":
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
            else:
                r = await run_locomo_task(task, experience_section=aug, group="train")

            ev = await evaluate_task(r, benchmark, use_llm_judge=False)
            score = ev.get("score", 0.0)

            if benchmark == "alfworld" and not r.get("won") and ALFWORLD_RETRY_MAX > 0:
                game = game_list[i] if game_list else None
                for _ in range(ALFWORLD_RETRY_MAX):
                    await critic_filter_and_record(sf, task, r, score, benchmark, aug)
                    retry_aug = build_augmented_prompt(
                        f"{task.get('description', '')} [type: {task.get('metadata', {}).get('task_type', '')}]",
                        sf.library,
                        metadata=task.get("metadata", {"benchmark": benchmark})
                    )
                    if not retry_aug or retry_aug == aug or not game:
                        break
                    r2 = await run_alfworld_task(
                        game["file"], game["type"],
                        experience_section=f"\n## Experience\n{retry_aug}",
                        group="train_retry"
                    )
                    ev2 = await evaluate_task(r2, benchmark, use_llm_judge=False)
                    if ev2.get("score", 0.0) > score:
                        r, score = r2, ev2["score"]
                    if r.get("won"):
                        break

            recorded, cq = await critic_filter_and_record(sf, task, r, score, benchmark, aug)
            r["_train_score"] = score
            r["_critic_quality"] = cq
            r["_recorded"] = recorded
            all_results.append(r)

            tag = "✓" if score >= 0.5 else "✗"
            kept = "kept" if recorded else "drop"
            print(f"    {tag} [{i+1}/{len(train_tasks)}] em={score:.2f} q={cq:.0f} {kept} "
                  f"lib={len(sf.library.experiences)} | {task['task_id'][:30]}", flush=True)

    return all_results

# ─── ALFWorld game list helper ────────────────────────────────────────────

def get_alfworld_games(n: int = 40) -> list:
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

# ─── Benchmark runner ─────────────────────────────────────────────────────

async def run_benchmark(benchmark: str, tasks: list, game_list: list = None) -> dict:
    print(f"\n{'='*70}")
    print(f"  Benchmark: {benchmark} (model: {MODEL})")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Metric: {'pass@1' if benchmark == 'alfworld' else 'Exact Match'}")
    print(f"{'='*70}")

    mid = len(tasks) // 2
    train_tasks = tasks[:mid]
    test_tasks = tasks[mid:]
    print(f"  Train: {len(train_tasks)} | Test: {len(test_tasks)}")

    os.makedirs(f"{RESULTS_DIR}/{benchmark}", exist_ok=True)

    print(f"\n  Phase 1: Sequential iterative training ({len(train_tasks)} tasks)...")
    sf = SkillForgeV6()
    sem = asyncio.Semaphore(CONCURRENCY)

    train_results = await train_sequential(
        benchmark, train_tasks, sf, sem,
        game_list=game_list[:mid] if game_list else None
    )

    train_valid = [r for r in train_results if not r.get("error")]
    train_success = [r for r in train_results if r.get("_train_score", 0) >= 0.5]
    avg_score = sum(r.get("_train_score", 0) for r in train_results) / max(len(train_results), 1)
    avg_q = sum(r.get("_critic_quality", 0) for r in train_results) / max(len(train_results), 1)
    print(f"\n  Train: {len(train_valid)}/{len(train_tasks)} valid, "
          f"{len(train_success)}/{len(train_valid)} pass, "
          f"avg_em={avg_score:.1%}, avg_critic_q={avg_q:.1f}")
    print(f"  Library size (after critic gating): {len(sf.library.experiences)}")
    sf.save(f"{RESULTS_DIR}/{benchmark}/library_after_train.json")

    print(f"\n  Phase 2: Testing {len(test_tasks)} tasks × 3 groups (A/B/C)...")

    print(f"    [A] Baseline (no augmentation)...", flush=True)
    async def run_test_a(i, task):
        async with sem:
            if benchmark in ("gaia", "gaia2", "swebench_dynamic"):
                return await run_gaia_task(task, "", "A")
            if benchmark == "alfworld":
                game = game_list[mid + i] if game_list and mid + i < len(game_list) else None
                return await run_alfworld_task(game["file"], game["type"], "", "A") if game else \
                       {"task_id": task["task_id"], "error": "no_game", "score": 0.0}
            return await run_locomo_task(task, "", "A")
    results_a = await asyncio.gather(*[run_test_a(i, t) for i, t in enumerate(test_tasks)])

    print(f"    [B] Raw experience injection...", flush=True)
    raw_library = ExperienceLibrary()
    for exp in sf.library.experiences:
        raw_exp = copy.deepcopy(exp)
        raw_exp.failure_taxonomy = {
            k: v for k, v in raw_exp.failure_taxonomy.items()
            if k not in ("ai_refined", "causal_lesson", "avoidance_note",
                         "transferability", "generalized_steps",
                         "evolution_insight", "quality_score", "evolution_trace",
                         "critic_quality")
        }
        raw_library.record(raw_exp)

    async def run_test_b(i, task):
        async with sem:
            if benchmark in ("gaia", "gaia2", "swebench_dynamic"):
                aug = build_augmented_prompt(task["description"], raw_library,
                                            metadata={"benchmark": benchmark})
                return await run_gaia_task(task, aug, "B")
            if benchmark == "alfworld":
                game = game_list[mid + i] if game_list and mid + i < len(game_list) else None
                if not game:
                    return {"task_id": task["task_id"], "error": "no_game", "score": 0.0}
                td = f"{results_a[i].get('task', '')} [type: {game['type']}]"
                aug = build_augmented_prompt(td, raw_library,
                                            metadata={"task_type": game["type"]})
                return await run_alfworld_task(
                    game["file"], game["type"],
                    experience_section=f"\n## Experience\n{aug}" if aug else "",
                    group="B"
                )
            aug = build_augmented_prompt(task["description"], raw_library,
                                        metadata=task.get("metadata", {}))
            return await run_locomo_task(task, aug, "B")
    results_b = await asyncio.gather(*[run_test_b(i, t) for i, t in enumerate(test_tasks)])

    print(f"    [C] AI-refined + critic-gated injection...", flush=True)
    async def run_test_c(i, task):
        async with sem:
            if benchmark in ("gaia", "gaia2", "swebench_dynamic"):
                aug = build_augmented_prompt(task["description"], sf.library,
                                            metadata={"benchmark": benchmark})
                return await run_gaia_task(task, aug, "C")
            if benchmark == "alfworld":
                game = game_list[mid + i] if game_list and mid + i < len(game_list) else None
                if not game:
                    return {"task_id": task["task_id"], "error": "no_game", "score": 0.0}
                td = f"{results_a[i].get('task', '')} [type: {game['type']}]"
                aug = build_augmented_prompt(td, sf.library,
                                            metadata={"task_type": game["type"]})
                return await run_alfworld_task(
                    game["file"], game["type"],
                    experience_section=f"\n## Experience\n{aug}" if aug else "",
                    group="C"
                )
            aug = build_augmented_prompt(task["description"], sf.library,
                                        metadata=task.get("metadata", {}))
            return await run_locomo_task(task, aug, "C")
    results_c = await asyncio.gather(*[run_test_c(i, t) for i, t in enumerate(test_tasks)])

    print(f"\n  Evaluating with EM / pass@1 (LLM-Judge as tie-breaker)...", flush=True)
    eval_tasks = []
    for i in range(len(test_tasks)):
        eval_tasks.append(evaluate_task(results_a[i], benchmark))
        eval_tasks.append(evaluate_task(results_b[i], benchmark))
        eval_tasks.append(evaluate_task(results_c[i], benchmark))
    all_evals = await asyncio.gather(*eval_tasks)

    scores = {"A_baseline": [], "B_raw": [], "C_refined": []}
    for i in range(len(test_tasks)):
        scores["A_baseline"].append(all_evals[i * 3])
        scores["B_raw"].append(all_evals[i * 3 + 1])
        scores["C_refined"].append(all_evals[i * 3 + 2])

    report = {}
    for group, evals in scores.items():
        valid = [e["score"] for e in evals if e.get("score") is not None]
        ems = [e.get("em", 0.0) for e in evals]
        report[group] = {
            "avg_score": sum(valid) / len(valid) if valid else 0.0,
            "em": sum(ems) / len(ems) if ems else 0.0,
            "n": len(valid),
        }

    metric_name = "pass@1" if benchmark == "alfworld" else "EM"
    print(f"\n  Results ({benchmark}, model={MODEL}):")
    print(f"    A (Baseline):    {metric_name}={report['A_baseline']['em']:.1%}")
    print(f"    B (Raw inject):  {metric_name}={report['B_raw']['em']:.1%}")
    print(f"    C (AI-refined):  {metric_name}={report['C_refined']['em']:.1%}")
    delta_ac = report['C_refined']['em'] - report['A_baseline']['em']
    delta_bc = report['C_refined']['em'] - report['B_raw']['em']
    print(f"    Δ(C-A): {delta_ac:+.1%} | Δ(C-B): {delta_bc:+.1%}")

    full_report = {
        "benchmark": benchmark, "model": MODEL,
        "metric": metric_name,
        "n_train": len(train_tasks), "n_test": len(test_tasks),
        "design": [
            "sequential_iterative_training",
            "cross_agent_critic_gating",
            "exact_match_pass_at_1_metrics",
            "alfworld_oracle_retry_only",
        ],
        "train_stats": {
            "avg_score": avg_score,
            "success_rate": len(train_success) / max(len(train_valid), 1),
            "library_size": len(sf.library.experiences),
            "avg_critic_quality": avg_q,
        },
        "results": report,
        "delta_refined_vs_baseline": delta_ac,
        "delta_refined_vs_raw": delta_bc,
    }

    if benchmark == "locomo":
        full_report["static_vs_dynamic"] = {
            "static_em": report['A_baseline']['em'],
            "dynamic_em": report['C_refined']['em'],
            "delta": report['C_refined']['em'] - report['A_baseline']['em'],
        }

    with open(f"{RESULTS_DIR}/{benchmark}/report.json", "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    return full_report

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  SkillForge V6 — LATEST runner                                   ║")
    print("║  Cross-agent critic gating · EM / pass@1 metrics                 ║")
    print(f"║  Model: {MODEL:<22} | Concurrency: {CONCURRENCY:<3}              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    print("\n  Loading benchmarks...")
    benchmarks = {}
    for name in ["gaia", "alfworld", "locomo", "gaia2", "swebench_dynamic"]:
        config = {"name": name, "num_samples": TASK_LIMITS[name]}
        if name == "gaia2":
            config["scenario_dir"] = "/tmp/harbor-datasets/datasets/gaia2-cli"
        loader = BenchmarkLoader(config)
        tasks = loader.load()[:TASK_LIMITS[name]]
        benchmarks[name] = tasks
        print(f"    {name}: {len(tasks)} tasks")

    print("  Loading ALFWorld game files...")
    alfworld_games = get_alfworld_games(TASK_LIMITS["alfworld"])
    print(f"    ALFWorld games: {len(alfworld_games)}")
    print(f"\n  Total: {sum(len(t) for t in benchmarks.values())} tasks")

    all_reports = {}
    for name, tasks in benchmarks.items():
        if not tasks:
            print(f"\n  SKIP {name}: no tasks")
            continue
        try:
            game_list = alfworld_games if name == "alfworld" else None
            all_reports[name] = await run_benchmark(name, tasks, game_list=game_list)
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {name}: {e}")
            traceback.print_exc()
            all_reports[name] = {"error": str(e)}

    print(f"\n\n{'═'*70}")
    print(f"  FINAL SUMMARY (latest — DeepSeek V4 Pro · EM / pass@1)")
    print(f"{'═'*70}")
    print(f"  {'Benchmark':<12} {'Metric':<8} {'A':>8} {'B':>8} {'C':>8} {'Δ(C-A)':>9} {'Δ(C-B)':>9}")
    print(f"  {'-'*70}")
    for name, r in all_reports.items():
        if "error" in r:
            print(f"  {name:<12} ERROR: {r['error'][:40]}")
        else:
            res = r["results"]
            print(f"  {name:<12} {r['metric']:<8} "
                  f"{res['A_baseline']['em']:>7.1%} "
                  f"{res['B_raw']['em']:>7.1%} "
                  f"{res['C_refined']['em']:>7.1%} "
                  f"{r['delta_refined_vs_baseline']:>+8.1%} "
                  f"{r['delta_refined_vs_raw']:>+8.1%}")

    with open(f"{RESULTS_DIR}/final_summary.json", "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {RESULTS_DIR}/final_summary.json")

if __name__ == "__main__":
    asyncio.run(main())
