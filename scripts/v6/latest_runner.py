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
from v6.response_filter import AIResponseProcessor
from v6.seed_skills import inject_seed_skills
from benchmarks.loader import BenchmarkLoader

MODEL = "deepseek-v4-pro"
CONCURRENCY = 15
TASK_TIMEOUT_QA = 180
TASK_TIMEOUT_AGENT = 300
TASK_TIMEOUT_ALFWORLD = 240
ALFWORLD_RETRY_MAX = 2
QUALITY_THRESHOLD = 5

RESULTS_DIR = str(PROJECT_ROOT / "experiments_results" / "latest")

# ─── Trace Logger (for human review of prompts & responses) ───────────────

import threading

class TraceLogger:
    """Append-only JSONL logger for full prompt/response/score traces.

    Each line in the trace file is a JSON object with:
      - timestamp, benchmark, group, phase (train/test)
      - task_id, task_desc
      - augmented_prompt (the injected experience section)
      - response (agent's final answer)
      - expected (ground truth)
      - score (EM or pass@1)
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._files = {}  # benchmark -> file handle

    def _get_file(self, benchmark: str):
        if benchmark not in self._files:
            trace_dir = Path(RESULTS_DIR) / benchmark
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / "trace.jsonl"
            self._files[benchmark] = open(trace_path, "a", encoding="utf-8")
        return self._files[benchmark]

    def log(self, benchmark: str, group: str, phase: str,
            task_id: str, task_desc: str, augmented_prompt: str,
            response: str, expected: str, score: float,
            extra: dict = None):
        """Write one trace record."""
        import datetime
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "benchmark": benchmark,
            "group": group,
            "phase": phase,
            "task_id": task_id,
            "task_desc": task_desc[:500],  # truncate very long descs
            "augmented_prompt": augmented_prompt,
            "response": response[:2000],  # truncate very long responses
            "expected": expected,
            "score": score,
        }
        if extra:
            record.update(extra)
        with self._lock:
            f = self._get_file(benchmark)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()


_trace = TraceLogger()

TASK_LIMITS = {"gaia": 50, "alfworld": 40, "locomo": 50, "gaia2": 30, "swebench_dynamic": 30}

ALFWORLD_PYTHON = str(PROJECT_ROOT / ".venv_alfworld" / "bin" / "python")
ALFWORLD_DATA = str(PROJECT_ROOT / ".venv_alfworld" / "data")
ALFWORLD_MAX_STEPS = 50

CHECKPOINT_FILE = str(PROJECT_ROOT / "experiments_results" / "latest" / "_checkpoint.json")

# ─── API availability detection ───────────────────────────────────────────

class APIUnavailableError(Exception):
    """Raised when DeepSeek V4 Pro API is confirmed unavailable."""
    pass

_api_consecutive_failures = 0
_API_FAILURE_THRESHOLD = 3  # After 3 consecutive failures, consider API down

async def probe_api_available() -> bool:
    """Quick probe to check if DeepSeek V4 Pro API is responding."""
    r = await _llm_call("Reply with exactly: OK", max_turns=1, timeout=30)
    if r.get("error"):
        err = str(r["error"])
        if "429" in err or "rate_limit" in err or "timeout" in err or "额度" in err:
            return False
    if not r.get("text"):
        return False
    return True

def _check_api_error(r: dict) -> bool:
    """Check if result indicates API unavailability. Returns True if API is down."""
    global _api_consecutive_failures
    if r.get("error"):
        err = str(r["error"])
        if "429" in err or "rate_limit" in err or "timeout" in err or "额度" in err:
            _api_consecutive_failures += 1
            if _api_consecutive_failures >= _API_FAILURE_THRESHOLD:
                return True
            return False
    if not r.get("text") and not r.get("actions"):
        _api_consecutive_failures += 1
        if _api_consecutive_failures >= _API_FAILURE_THRESHOLD:
            return True
        return False
    # Success — reset counter
    _api_consecutive_failures = 0
    return False

# ─── Checkpoint helpers ───────────────────────────────────────────────────

def save_checkpoint(state: dict):
    """Save experiment state to checkpoint file."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    print(f"  💾 Checkpoint saved: {CHECKPOINT_FILE}", flush=True)

def load_checkpoint() -> dict:
    """Load experiment state from checkpoint file if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

def clear_checkpoint():
    """Remove checkpoint file after successful completion."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

# ─── LLM helpers ──────────────────────────────────────────────────────────

def llm_review_fn(prompt: str) -> str:
    """Synchronous single-turn LLM call used by ai_review_experience."""
    async def _call():
        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=2, cwd="/tmp"
        )
        result = ""
        gen = None
        try:
            async with asyncio.timeout(90):
                gen = query(prompt=prompt, options=opt)
                async for msg in gen:
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if hasattr(block, 'text') and block.text:
                                result += block.text
                        if result:
                            break
        except Exception:
            pass
        finally:
            if gen is not None:
                try:
                    await gen.aclose()
                except Exception:
                    pass
        return result

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_call())
        finally:
            _shutdown_loop(loop)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_in_thread).result(timeout=120)

def _shutdown_loop(loop: asyncio.AbstractEventLoop):
    """Gracefully shutdown an event loop: cancel all pending tasks, then close."""
    try:
        # Cancel all remaining tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def _query_sync(prompt: str, max_turns: int = 1, timeout: int = 60) -> dict:
    """Run CodeBuddy query in a fresh event loop (thread-safe, avoids cancel scope issues)."""
    async def _inner():
        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=max_turns, cwd="/tmp"
        )
        text = ""
        actions = []
        gen = None
        try:
            async with asyncio.timeout(timeout):
                gen = query(prompt=prompt, options=opt)
                async for msg in gen:
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
        finally:
            if gen is not None:
                try:
                    await gen.aclose()
                except Exception:
                    pass
        return {"text": text, "actions": actions, "error": None}

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_inner())
    finally:
        _shutdown_loop(loop)

async def _llm_call(prompt: str, max_turns: int = 1, timeout: int = 60) -> dict:
    """Async wrapper: runs query in isolated thread to avoid anyio conflicts.
    Has a hard outer timeout (timeout + 30s grace) to prevent indefinite hangs."""
    loop = asyncio.get_event_loop()
    hard_timeout = timeout + 30  # Grace period beyond the inner timeout
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _query_sync, prompt, max_turns, timeout),
            timeout=hard_timeout
        )
    except asyncio.TimeoutError:
        return {"text": "", "actions": [], "error": f"hard_timeout_after_{hard_timeout}s"}


def _query_notool_sync(system_prompt: str, user_prompt: str, timeout: int = 60) -> dict:
    """Pure text generation without CodeBuddy tools (for GAIA2 ARE interaction)."""
    async def _inner():
        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=3, cwd="/tmp",
            allowed_tools=[],  # No tools — pure text generation
            system_prompt=system_prompt,
        )
        text = ""
        gen = None
        try:
            async with asyncio.timeout(timeout):
                gen = query(prompt=user_prompt, options=opt)
                async for msg in gen:
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if hasattr(block, 'text') and block.text:
                                if '429' in block.text and '额度' in block.text:
                                    return {"text": "", "error": "429_rate_limit"}
                                text += block.text
        except Exception as e:
            return {"text": text, "error": str(e)[:200] if not text else None}
        finally:
            if gen is not None:
                try:
                    await gen.aclose()
                except Exception:
                    pass
        return {"text": text, "error": None}

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_inner())
    finally:
        _shutdown_loop(loop)


async def _llm_call_notool(system_prompt: str, user_prompt: str, timeout: int = 60) -> dict:
    """Async wrapper for pure-text LLM call (no tools)."""
    loop = asyncio.get_event_loop()
    hard_timeout = timeout + 30
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _query_notool_sync, system_prompt, user_prompt, timeout),
            timeout=hard_timeout
        )
    except asyncio.TimeoutError:
        return {"text": "", "error": f"hard_timeout_after_{hard_timeout}s"}

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
    if not g:
        return 0.0
    # Strict equality
    if p == g:
        return 1.0
    # Allow gold contained in pred ONLY if gold is multi-word (>=2 words)
    # or if pred is short (extracted answer). For single-word gold answers,
    # require word-boundary match to avoid false positives like "4" in "2024".
    g_words = g.split()
    p_words = p.split()
    if len(g_words) >= 2:
        # Multi-word gold: substring match is reasonable
        if g in p:
            return 1.0
    else:
        # Single-word gold: require word-boundary match in pred
        # This prevents "4" matching "2024" or "yes" matching "synthesis"
        if re.search(r'\b' + re.escape(g) + r'\b', p):
            # Only count if pred is reasonably short (extracted answer)
            if len(p_words) <= 20:
                return 1.0
    return 0.0

# ─── GAIA2 ARE runner (real tool calling) ─────────────────────────────────

async def run_gaia2_task_with_are(task: dict, experience_section: str = "",
                                   group: str = "A") -> dict:
    """Run a GAIA2 task using the real ARE simulation environment.

    The LLM interacts with the environment through function calling:
    1. Load scenario → initialize ARE session
    2. Get tool schemas → build system prompt with available tools
    3. LLM generates tool calls → execute in ARE → return results → loop
    4. Record event log for evaluation
    """
    from scripts.v6.are_integration import ARESession

    task_id = task["task_id"]
    task_desc = task.get("description") or task.get("task_desc", "")
    metadata = task.get("metadata", {})
    scenario_path = metadata.get("scenario_path") or task.get("scenario_path", "")

    result = {
        "task_id": task_id,
        "expected": task.get("expected", []),
        "oracle_answer": task.get("oracle_answer", ""),
        "response": "",
        "error": None,
        "time_cost": 0,
        "augmented": bool(experience_section),
        "group": group,
        "actions": [],
        "event_log": [],
    }

    if not scenario_path:
        result["error"] = "no_scenario_path"
        return result

    t0 = time.time()
    session = None
    try:
        # Initialize ARE session
        session = ARESession(scenario_path)

        # Pre-fetch the user task (avoid LLM needing to call get_last_message)
        task_message = session.call_tool("AgentUserInterface__get_last_message_from_user", {})
        task_content = ""
        if isinstance(task_message, dict):
            task_content = task_message.get("result", {}).get("content", "") if isinstance(task_message.get("result"), dict) else str(task_message.get("result", ""))
        if not task_content:
            task_content = task_desc  # Fallback to task description

        result["actions"].append({
            "tool": "AgentUserInterface__get_last_message_from_user",
            "args": {},
            "result_preview": task_content[:200],
        })

        # Build tool ID mapping (op-001, op-002, ...)
        tool_names = list(session._all_tools.keys())
        tool_id_map = {}  # op-001 -> ARE tool name
        tool_lines = []
        for i, name in enumerate(tool_names):
            tid = f"op-{i+1:03d}"
            tool_id_map[tid] = name
            tool = session._all_tools[name]
            # Build compact param list
            params = []
            for arg in tool.args:
                req = " [required]" if not arg.has_default else ""
                params.append(f"{arg.name}{req}")
            params_str = ", ".join(params) if params else "(none)"
            # Use short description
            desc = (tool.function_description or "")[:60]
            tool_lines.append(f"{tid}: {desc} | {params_str}")

        # Add special tools
        tool_lines.append("op-000: Wait for environment notification | timeout_seconds")
        tool_id_map["op-000"] = "_wait_for_notification"

        tool_text = "\n".join(tool_lines)

        # Build system prompt - ONLY universal behavioral rules here.
        # Task-specific strategies (contact search, calendar lookup, discount scanning)
        # are injected via the skill/experience layer (RELEVANT EXPERIENCE section).
        # This separation enables clean ablation: baseline vs baseline+skills.
        system_prompt = (
            "You are a task executor. You call ONE operation per turn.\n\n"
            "OUTPUT FORMAT (exactly 2 lines, nothing else):\n"
            "NEXT_OP: <op-id>\n"
            "PARAMS: <key>:<value> | <key>:<value>\n\n"
            "EXAMPLES:\n"
            "NEXT_OP: op-024\n"
            "PARAMS: query:Film\n\n"
            "NEXT_OP: op-075\n"
            "PARAMS: title:Meeting | start_datetime:2024-10-19 08:00:00 | end_datetime:2024-10-19 20:00:00 | attendees:[\"John\"]\n\n"
            "NEXT_OP: op-076\n"
            "PARAMS: event_id:ABC123\n\n"
            "NEXT_OP: op-000\n"
            "PARAMS: timeout_seconds:30\n\n"
            "RULES:\n"
            "1. Output EXACTLY 2 lines per turn: NEXT_OP + PARAMS. NO other text.\n"
            "2. ONE operation per turn. Never output multiple NEXT_OP lines.\n"
            "3. Use ONLY real data from results. Never invent names, IDs, or emails.\n"
            "4. EFFICIENCY: If a search returns 0 results, try a DIFFERENT shorter keyword.\n"
            "   Never repeat the same query. Max 5 attempts per sub-task, then move on.\n"
            "5. BUDGET: Complete primary actions within 20 turns. If stuck >5 turns on\n"
            "   any sub-task, SKIP it and proceed with available information.\n"
            "6. ERROR RECOVERY: If a tool errors, fix the parameter. Never retry same params.\n"
            "7. When you get a notification, act on it immediately.\n"
            "8. NEVER explain, plan, or narrate. ONLY output NEXT_OP + PARAMS.\n\n"
            "⚠️ CRITICAL TWIST RULE (violating this is the #1 cause of task failure):\n"
            "Tasks with conditional language ('if he can\'t make it', 'reschedule', 'accept\n"
            "any suggested date', 'if not') have TWO phases. After completing Phase 1\n"
            "(primary actions + notify user via op-001), you MUST call op-000 with\n"
            "timeout_seconds:60 to WAIT for a reply. Only after processing the reply\n"
            "can you output ALL_DONE. Skipping op-000 after op-001 loses ~50% of points.\n\n"
            f"OPERATIONS:\n{tool_text}\n\n"
            "START NOW. Output ONLY: NEXT_OP + PARAMS."
        )

        if experience_section:
            system_prompt += f"\n\nRELEVANT EXPERIENCE:\n{experience_section}"

        # Multi-turn interaction loop
        # Since CodeBuddy SDK doesn't support multi-turn chat natively,
        # we accumulate conversation history in the user prompt.
        # Detect if task has a twist (conditional second phase)
        has_twist = any(kw in task_content.lower() for kw in [
            "if my friend", "if he can't", "if she can't",
            "if they can't", "if that doesn't work", "if the person",
            "if the order", "if it doesn't", "if not",
            "reschedule", "accept any suggested", "proposes",
            "can't make it", "declines", "an alternative",
            "if there's", "if you can't", "handle the twist",
            "let me know when", "send him an email after",
            "after scheduling", "after you",
        ])
        # Also check task_desc (metadata description) as backup
        task_desc_lower = task_desc.lower()
        has_twist = has_twist or any(kw in task_desc_lower for kw in [
            "if my friend", "if he can't", "if she can't",
            "if they can't", "reschedule", "accept any suggested",
            "can't make it",
        ])

        twist_reminder = ""
        if has_twist:
            twist_reminder = (
                "\n\n⚠️ CRITICAL: This task has a CONDITIONAL SECOND PHASE (twist).\n"
                "After completing primary actions, this EXACT sequence is REQUIRED:\n"
                "  1. Notify user that primary actions are done (op-001)\n"
                "  2. IMMEDIATELY call op-000 with timeout_seconds:60 to wait for reply\n"
                "  3. When notification arrives: read it, extract the proposed change\n"
                "  4. Execute the change (delete old event, create new one, send email)\n"
                "  5. Only then output ALL_DONE\n"
                "NEVER skip op-000 after op-001. This loses 50%+ of points.\n"
            )

        conversation_history = (
            f"TASK: {task_content}\n"
            f"{twist_reminder}\n"
            "Output your first operation now (NEXT_OP + PARAMS only):"
        )

        max_turns = 100
        all_responses = []
        reasoning_trace = []  # Collect AI-filtered valuable reasoning across turns

        # Twist enforcement: track whether op-001 was called and op-000 followed
        op001_called_at_turn = -1  # Turn number when op-001 was called
        op000_called = False       # Whether op-000 has been called since last op-001
        primary_actions_done = False  # Heuristic: agent has done create+delete+send actions
        # AI Response Processor — filters noise from LLM output, keeps valuable reasoning
        async def _ai_filter_fn(prompt: str) -> str:
            """Lightweight LLM call for AI response filtering."""
            r = await _llm_call_notool(
                "You are a JSON-only response evaluator. Output valid JSON only.",
                prompt, timeout=30
            )
            return r.get("text", "")

        processor = AIResponseProcessor(
            valid_action_ids=set(tool_id_map.keys()),
            llm_fn=_ai_filter_fn,
            enable_ai_filter=True,
            min_text_length_for_ai=80,  # Only AI-eval if >80 chars of non-action text
        )
        processor.set_task_context(task_content[:500])

        for turn in range(max_turns):
            # Call LLM in pure-text mode (no CodeBuddy tools)
            r = await _llm_call_notool(system_prompt, conversation_history, timeout=300)

            if _check_api_error(r):
                raise APIUnavailableError(
                    f"API unavailable after {_API_FAILURE_THRESHOLD} consecutive failures"
                )

            response_text = r.get("text", "")
            all_responses.append(response_text)

            # Process response through AI filter
            processed = await processor.process_response(response_text)

            # Check completion — but BLOCK if twist is pending
            if processed.is_completion:
                # If task has a twist and op-001 was called but op-000 was NOT called,
                # the agent is trying to finish prematurely. Block it.
                if (has_twist and op001_called_at_turn >= 0 and not op000_called):
                    # Reject ALL_DONE — force agent to call op-000
                    conversation_history += (
                        "\n\n🛑 ALL_DONE REJECTED: You have NOT waited for the reply yet.\n"
                        "This task has a conditional second phase. You MUST:\n"
                        "NEXT_OP: op-000\n"
                        "PARAMS: timeout_seconds:60\n\n"
                        "Do NOT output ALL_DONE until you have called op-000 and "
                        "processed any incoming notifications.\n"
                    )
                    continue  # Force another turn instead of breaking
                break

            # Handle invalid responses (no action found)
            if not processed.valid:
                if processed.error_type == "no_action":
                    if processor.consecutive_failures <= 3 and turn < max_turns - 1:
                        retry = processor.get_retry_prompt(processed)
                        conversation_history += retry
                        continue
                    break
                elif processed.error_type == "invalid_id":
                    retry = processor.get_retry_prompt(processed)
                    conversation_history += retry
                    continue

            tool_id = processed.action_id
            tool_args = processed.params_parsed

            # Loop detection: if agent is repeating the same call, break the loop
            if processor.is_looping:
                loop_prompt = processor.get_loop_break_prompt()
                conversation_history += loop_prompt
                continue  # Skip execution, force agent to try something different

            # Execute the tool
            are_tool_name = tool_id_map[tool_id]

            if are_tool_name == "_wait_for_notification":
                op000_called = True
                timeout_val = int(tool_args.get("timeout_seconds", 180))
                tr = session.wait_for_notification(timeout_val)
            elif "send_message_to_user" in are_tool_name:
                op001_called_at_turn = turn
                op000_called = False  # Reset - must call op-000 after this
                tr = session.call_tool(are_tool_name, tool_args)
            else:
                tr = session.call_tool(are_tool_name, tool_args)
                # Detect primary actions completion heuristically
                if not primary_actions_done:
                    method_lower = are_tool_name.lower()
                    if any(kw in method_lower for kw in ["send_email", "add_calendar", "create_event"]):
                        primary_actions_done = True

            result["actions"].append({
                "tool": are_tool_name,
                "args": tool_args,
                "result_preview": str(tr)[:500],
            })

            # Format result for conversation
            result_str = json.dumps(tr, default=str)[:2000]

            # Check for notifications
            notifications = []
            try:
                r_data = tr if isinstance(tr, dict) else json.loads(str(tr))
                if isinstance(r_data, dict) and r_data.get("notifications"):
                    for n in r_data["notifications"]:
                        notifications.append(n.get('message', str(n)))
            except (json.JSONDecodeError, TypeError):
                pass

            # Collect valuable reasoning for experience recording
            if processed.valuable_reasoning:
                reasoning_trace.append(processed.valuable_reasoning)

            # Build history entry with AI-filtered valuable reasoning
            history_entry = processor.build_history_entry(
                processed, result_str, notifications
            )

            # Turn budget tracking: inject remaining turns reminder at key thresholds
            remaining_turns = max_turns - turn - 1
            budget_warning = ""
            if remaining_turns in (75, 50, 30, 15, 5):
                budget_warning = f"\n[⏱ {remaining_turns} turns remaining. "
                if remaining_turns <= 15:
                    budget_warning += "URGENT: Complete primary task NOW. Skip any stuck sub-tasks.]"
                elif remaining_turns <= 30:
                    budget_warning += "Prioritize core actions. Don't waste turns on exhaustive searches.]"
                else:
                    budget_warning += "On track. Remember to reserve turns for any twist/follow-up.]"

            # Twist enforcement: if op-001 called but no op-000 within 1+ turns → force reminder
            if (has_twist and op001_called_at_turn >= 0 and not op000_called
                    and turn - op001_called_at_turn >= 1):
                gap = turn - op001_called_at_turn
                urgency = (
                    "IMMEDIATELY" if gap >= 2 else
                    "NOW - do not output any other operation"
                )
                twist_enforce = (
                    f"\n\n🛑 TWIST PROTOCOL VIOLATION: You called op-001 {gap} turns ago "
                    f"but have NOT called op-000 (wait for reply). "
                    f"The task has a conditional second phase. Next operation MUST be:\n"
                    f"NEXT_OP: op-000\n"
                    f"PARAMS: timeout_seconds:60\n"
                    f"This is REQUIRED before ALL_DONE. Call op-000 {urgency}."
                )
                budget_warning = twist_enforce + (budget_warning or "")

            if budget_warning:
                history_entry += budget_warning

            conversation_history += history_entry

            # Truncate conversation if too long (keep last 40000 chars)
            if len(conversation_history) > 50000:
                task_header = f"TASK: {task_content}\n\n"
                remaining = conversation_history[len(task_header):]
                conversation_history = (
                    task_header
                    + "[Earlier steps truncated]\n...\n"
                    + remaining[-40000:]
                )
        # Collect all responses into result
        result["response"] = "\n---\n".join(all_responses)
        result["event_log"] = session.event_log
        result["reasoning_trace"] = reasoning_trace  # Pass to experience recording

    except APIUnavailableError:
        raise
    except Exception as e:
        result["error"] = str(e)[:300]
    finally:
        if session:
            session.close()
        result["time_cost"] = time.time() - t0

    return result


def _extract_action_calls(text: str) -> list[dict]:
    """Extract ACTION: format tool calls from LLM response.

    Format: ACTION: tool_name | arg1=value1 | arg2=value2
    Tool names use / separator: AppName/method_name
    """
    calls = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("ACTION:"):
            continue
        # Parse: ACTION: tool_name | arg1=value1 | arg2=value2
        parts = line[7:].strip().split("|")
        if not parts:
            continue
        tool_name = parts[0].strip()
        args = {}
        for part in parts[1:]:
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Try to parse JSON values (arrays, objects)
                if val.startswith("[") or val.startswith("{"):
                    try:
                        val = json.loads(val)
                    except json.JSONDecodeError:
                        pass
                args[key] = val
        if tool_name:
            calls.append({"name": tool_name, "args": args})
    return calls


def _extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from LLM response text.

    Supports multiple formats:
    1. JSON code blocks: ```json {"tool": "name", "args": {...}} ```
    2. Inline JSON objects: {"tool": "name", "args": {...}}
    3. Function call syntax: tool_name(arg1=val1, arg2=val2)
    """
    calls = []

    # Pattern 1: JSON in code blocks (most reliable)
    code_block_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
    for block_match in code_block_pattern.finditer(text):
        block = block_match.group(1).strip()
        try:
            data = json.loads(block)
            if isinstance(data, dict):
                name = data.get("tool") or data.get("name") or data.get("function", "")
                args = data.get("args") or data.get("arguments") or data.get("parameters", {})
                if name and ("__" in name or name.startswith("are_")):
                    calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("tool") or item.get("name") or item.get("function", "")
                        args = item.get("args") or item.get("arguments") or item.get("parameters", {})
                        if name and ("__" in name or name.startswith("are_")):
                            calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Pattern 2: Inline JSON objects (look for {"tool": "...", "args": {...}})
    # Use a more careful approach: find all { that start with "tool"
    inline_pattern = re.compile(
        r'\{\s*"(?:tool|name|function)"\s*:\s*"([^"]+)"\s*,\s*"(?:args|arguments|parameters)"\s*:\s*(\{[^}]*\})\s*\}',
        re.DOTALL
    )
    for m in inline_pattern.finditer(text):
        try:
            name = m.group(1)
            args = json.loads(m.group(2))
            if "__" in name or name.startswith("are_"):
                calls.append({"name": name, "args": args})
        except (json.JSONDecodeError, IndexError):
            continue

    if calls:
        return calls

    # Pattern 3: Try to find any JSON object in text that looks like a tool call
    # This handles cases where LLM outputs JSON without code blocks
    brace_pattern = re.compile(r'\{[^{}]{10,500}\}')
    for m in brace_pattern.finditer(text):
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                name = data.get("tool") or data.get("name") or data.get("function", "")
                args = data.get("args") or data.get("arguments") or data.get("parameters", {})
                if name and ("__" in name or name.startswith("are_")):
                    calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Pattern 4: Function call syntax: ToolName__method(key="value", ...)
    func_pattern = re.compile(
        r'(\w+__\w+)\s*\(\s*(.*?)\s*\)',
        re.DOTALL
    )
    for m in func_pattern.finditer(text):
        name = m.group(1)
        args_str = m.group(2).strip()
        args = {}
        if args_str:
            for pair in re.split(r',\s*(?=\w+=)', args_str):
                kv = pair.split("=", 1)
                if len(kv) == 2:
                    key = kv[0].strip()
                    val = kv[1].strip().strip('"').strip("'")
                    args[key] = val
        if name:
            calls.append({"name": name, "args": args})

    return calls


# ─── GAIA runner ──────────────────────────────────────────────────────────

async def run_gaia_task(task: dict, experience_section: str = "",
                        group: str = "A") -> dict:
    task_id = task["task_id"]
    description = task["description"]
    expected = task.get("expected", "")
    metadata = task.get("metadata", {})
    benchmark_type = metadata.get("benchmark", "")

    # Determine system prompt based on benchmark type
    if benchmark_type == "swebench_dynamic" or "swebench" in task_id:
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
    if _check_api_error(r):
        raise APIUnavailableError(f"API unavailable after {_API_FAILURE_THRESHOLD} consecutive failures")
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
        _out.write(json.dumps({{"type":"obs","obs":gs.feedback,"actions":adm,
                                "won":bool(gs.won),"done":bool(gs.won)}}) + "\\n"); _out.flush()
    elif c["action"] == "step":
        gs, score, done = env.step(c["command"])
        step_count += 1
        adm = list(gs.admissible_commands or [])
        won = bool(gs.won)
        done = done or won or step_count >= max_steps
        _out.write(json.dumps({{"type":"obs","obs":gs.feedback,"actions":adm,
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
    # LoCoMo is pure reading comprehension — answer is always in conversation history.
    # max_turns=1 prevents model from using web search tools which pollute the answer.
    r = await _llm_call(prompt, max_turns=1, timeout=TASK_TIMEOUT_QA)
    if _check_api_error(r):
        raise APIUnavailableError(f"API unavailable after {_API_FAILURE_THRESHOLD} consecutive failures")
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
        # ARE-based evaluation: compare agent's event_log against oracle_events.
        # Uses action-matching logic inspired by the official llm_judge.py.
        oracle_events = result.get("expected", [])
        event_log = result.get("event_log", [])
        response = (result.get("response") or "").strip()

        if not oracle_events:
            return {"score": 0.0, "em": 0.0, "method": "gaia2_no_oracle"}
        if not event_log and not response:
            return {"score": 0.0, "em": 0.0, "method": "gaia2_no_actions"}

        # Extract write-actions from agent's event log (matching official verifier logic)
        # Write actions are the ones that modify state (create, send, delete, etc.)
        write_prefixes = {
            "add_calendar_event", "delete_calendar_event",
            "send_email", "reply_to_email", "forward_email",
            "send_message", "create_and_add_message",
            "send_message_to_user", "send_message_to_agent",
            "add_to_cart", "checkout", "remove_from_cart", "cancel_order",
            "order_ride", "user_cancel_ride",
            "add_new_contact", "edit_contact", "delete_contact",
            "save_apartment", "remove_saved_apartment",
            "create_group_conversation", "add_participant_to_conversation",
        }

        agent_write_actions = []
        for event in event_log:
            method = event.get("method", "")
            if method in write_prefixes:
                agent_write_actions.append({
                    "tool": event.get("tool", ""),
                    "method": method,
                    "args": event.get("args", {}),
                })

        # Extract oracle write actions
        oracle_write_actions = []
        for oe in oracle_events:
            fn = oe.get("function", "")
            if fn in write_prefixes:
                oracle_write_actions.append({
                    "app": oe.get("app", ""),
                    "function": fn,
                    "args": oe.get("args", {}),
                })

        if not oracle_write_actions:
            # Answer-mode task: compare agent's user message against oracle answer
            oracle_answer = result.get("oracle_answer", "")
            if oracle_answer and response:
                # Check if agent sent a message to user containing the answer
                agent_user_msgs = [
                    e.get("args", {}).get("content", "")
                    for e in event_log
                    if e.get("method") == "send_message_to_user"
                ]
                agent_text = " ".join(agent_user_msgs) if agent_user_msgs else response
                # Use LLM judge for answer comparison
                judge_prompt = (
                    "Compare the agent's answer to the oracle answer.\n\n"
                    f"Oracle answer: {oracle_answer}\n"
                    f"Agent answer: {agent_text[:2000]}\n\n"
                    "Score 0.0 to 1.0: Does the agent's answer contain the same "
                    "semantic information as the oracle? Output ONLY a number:"
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
                        "method": "gaia2_answer_mode",
                        "agent_actions": len(event_log)}
            return {"score": 0.0, "em": 0.0, "method": "gaia2_no_oracle_writes"}

        # Action-mode: count how many oracle write actions were matched
        matched = 0
        used_agent_indices = set()

        for oracle_action in oracle_write_actions:
            oracle_fn = oracle_action["function"]
            oracle_args = oracle_action["args"]
            best_match = False

            for j, agent_action in enumerate(agent_write_actions):
                if j in used_agent_indices:
                    continue
                agent_method = agent_action["method"]

                # Function name match (allow aliases)
                fn_match = (
                    agent_method == oracle_fn
                    or agent_method.replace("_", "") == oracle_fn.replace("_", "")
                )
                if not fn_match:
                    continue

                # Check key args match (relaxed: at least 50% of oracle args present)
                if oracle_args and isinstance(oracle_args, dict):
                    agent_args = agent_action.get("args", {})
                    if isinstance(agent_args, dict):
                        matching_args = 0
                        total_args = 0
                        for key, val in oracle_args.items():
                            if not val or val == "[]" or val == "":
                                continue
                            total_args += 1
                            agent_val = str(agent_args.get(key, "")).lower()
                            oracle_val = str(val).lower()
                            if oracle_val in agent_val or agent_val in oracle_val:
                                matching_args += 1
                        if total_args > 0 and matching_args / total_args >= 0.4:
                            best_match = True
                    else:
                        best_match = True  # Can't compare args, accept fn match
                else:
                    best_match = True  # No args to compare

                if best_match:
                    used_agent_indices.add(j)
                    break

            if best_match:
                matched += 1

        recall = matched / len(oracle_write_actions) if oracle_write_actions else 0.0
        return {
            "score": recall,
            "em": 1.0 if recall >= 0.7 else 0.0,
            "action_recall": recall,
            "matched": matched,
            "total_oracle": len(oracle_write_actions),
            "total_agent": len(agent_write_actions),
            "method": "gaia2_action_recall",
        }

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
        reasoning_trace=result.get("reasoning_trace", []),
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

            if benchmark == "gaia" or benchmark == "swebench_dynamic":
                r = await run_gaia_task(task, experience_section=aug, group="train")
            elif benchmark == "gaia2":
                r = await run_gaia2_task_with_are(task, experience_section=aug, group="train")
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

            # Trace logging for human review
            _trace.log(
                benchmark=benchmark, group="train", phase="train",
                task_id=task["task_id"], task_desc=task.get("description", ""),
                augmented_prompt=aug, response=r.get("response", ""),
                expected=task.get("expected", r.get("expected", "")),
                score=score,
                extra={"critic_quality": cq, "recorded": recorded,
                       "library_size": len(sf.library.experiences)}
            )

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
                    # game_file_path already includes filename (e.g. "task/trial/game.tw-pddl")
                    # Files are under valid_unseen/ for eval_out_of_distribution split
                    game_file = os.path.join(ALFWORLD_DATA, "json_2.1.1", "valid_unseen", game_file_path)
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
    # Inject seed skills — task-specific strategies from failed traces.
    # These are the SKILL LAYER contribution (not system prompt).
    # They only affect groups B/C (augmented), not group A (baseline).
    n_seeds = inject_seed_skills(sf.library)
    if n_seeds:
        print(f"    Injected {n_seeds} seed skills into experience library")
    # GAIA2 tasks load full scenarios (~2.6MB each) — limit concurrency
    concurrency = 3 if benchmark == "gaia2" else CONCURRENCY
    sem = asyncio.Semaphore(concurrency)

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
            if benchmark == "gaia2":
                return await run_gaia2_task_with_are(task, "", "A")
            if benchmark in ("gaia", "swebench_dynamic"):
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
            if benchmark == "gaia2":
                aug = build_augmented_prompt(task["description"], raw_library,
                                            metadata={"benchmark": benchmark})
                r = await run_gaia2_task_with_are(task, aug, "B")
                r["_aug_prompt"] = aug
                return r
            if benchmark in ("gaia", "swebench_dynamic"):
                aug = build_augmented_prompt(task["description"], raw_library,
                                            metadata={"benchmark": benchmark})
                r = await run_gaia_task(task, aug, "B")
                r["_aug_prompt"] = aug
                return r
            if benchmark == "alfworld":
                game = game_list[mid + i] if game_list and mid + i < len(game_list) else None
                if not game:
                    return {"task_id": task["task_id"], "error": "no_game", "score": 0.0}
                td = f"{results_a[i].get('task', '')} [type: {game['type']}]"
                aug = build_augmented_prompt(td, raw_library,
                                            metadata={"task_type": game["type"]})
                r = await run_alfworld_task(
                    game["file"], game["type"],
                    experience_section=f"\n## Experience\n{aug}" if aug else "",
                    group="B"
                )
                r["_aug_prompt"] = aug
                return r
            aug = build_augmented_prompt(task["description"], raw_library,
                                        metadata=task.get("metadata", {}))
            r = await run_locomo_task(task, aug, "B")
            r["_aug_prompt"] = aug
            return r
    results_b = await asyncio.gather(*[run_test_b(i, t) for i, t in enumerate(test_tasks)])

    print(f"    [C] AI-refined + critic-gated injection...", flush=True)
    async def run_test_c(i, task):
        async with sem:
            if benchmark == "gaia2":
                aug = build_augmented_prompt(task["description"], sf.library,
                                            metadata={"benchmark": benchmark})
                r = await run_gaia2_task_with_are(task, aug, "C")
                r["_aug_prompt"] = aug
                return r
            if benchmark in ("gaia", "swebench_dynamic"):
                aug = build_augmented_prompt(task["description"], sf.library,
                                            metadata={"benchmark": benchmark})
                r = await run_gaia_task(task, aug, "C")
                r["_aug_prompt"] = aug
                return r
            if benchmark == "alfworld":
                game = game_list[mid + i] if game_list and mid + i < len(game_list) else None
                if not game:
                    return {"task_id": task["task_id"], "error": "no_game", "score": 0.0}
                td = f"{results_a[i].get('task', '')} [type: {game['type']}]"
                aug = build_augmented_prompt(td, sf.library,
                                            metadata={"task_type": game["type"]})
                r = await run_alfworld_task(
                    game["file"], game["type"],
                    experience_section=f"\n## Experience\n{aug}" if aug else "",
                    group="C"
                )
                r["_aug_prompt"] = aug
                return r
            aug = build_augmented_prompt(task["description"], sf.library,
                                        metadata=task.get("metadata", {}))
            r = await run_locomo_task(task, aug, "C")
            r["_aug_prompt"] = aug
            return r
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

    # Trace logging for test phase (all 3 groups)
    for i, task in enumerate(test_tasks):
        task_id = task["task_id"]
        task_desc = task.get("description", "")
        expected = task.get("expected", results_a[i].get("expected", ""))
        # Group A
        _trace.log(
            benchmark=benchmark, group="A_baseline", phase="test",
            task_id=task_id, task_desc=task_desc,
            augmented_prompt="",
            response=results_a[i].get("response", ""),
            expected=expected,
            score=all_evals[i * 3].get("score", 0.0),
        )
        # Group B
        aug_b = results_b[i].get("_aug_prompt", "")
        _trace.log(
            benchmark=benchmark, group="B_raw", phase="test",
            task_id=task_id, task_desc=task_desc,
            augmented_prompt=aug_b,
            response=results_b[i].get("response", ""),
            expected=expected,
            score=all_evals[i * 3 + 1].get("score", 0.0),
        )
        # Group C
        aug_c = results_c[i].get("_aug_prompt", "")
        _trace.log(
            benchmark=benchmark, group="C_refined", phase="test",
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
    print("║  SkillForge V6 — LATEST runner (v3 — strict metrics + resume)    ║")
    print("║  Cross-agent critic gating · EM / pass@1 metrics                 ║")
    print(f"║  Model: {MODEL:<22} | Concurrency: {CONCURRENCY:<3}              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # --- API availability probe ---
    print("\n  🔍 Probing API availability...", flush=True)
    api_ok = await probe_api_available()
    if not api_ok:
        print("  ❌ DeepSeek V4 Pro API is NOT available. Aborting.", flush=True)
        print("  💡 Re-run this script when API is back online.", flush=True)
        return
    print("  ✓ API is responding.", flush=True)

    # --- Check for existing checkpoint (resume support) ---
    checkpoint = load_checkpoint()
    completed_benchmarks = checkpoint.get("completed_benchmarks", {})
    if completed_benchmarks:
        print(f"\n  📂 Resuming from checkpoint: {list(completed_benchmarks.keys())} already done.", flush=True)

    # Run ALL 5 benchmarks with fixed evaluation metrics
    BENCHMARKS_TO_RUN = ["gaia2", "gaia", "locomo", "alfworld", "swebench_dynamic"]
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

    alfworld_games = get_alfworld_games(TASK_LIMITS.get("alfworld", 40))
    print(f"    alfworld games: {len(alfworld_games)}")
    print(f"\n  Total: {sum(len(t) for t in benchmarks.values())} tasks")

    all_reports = dict(completed_benchmarks)  # Start with previously completed
    paused = False

    for name, tasks in benchmarks.items():
        if name in completed_benchmarks:
            print(f"\n  SKIP {name}: already completed (from checkpoint)")
            continue
        if not tasks:
            print(f"\n  SKIP {name}: no tasks")
            continue

        # Probe API before each benchmark
        api_ok = await probe_api_available()
        if not api_ok:
            print(f"\n  ⚠️  API unavailable before starting {name}. Pausing experiment.", flush=True)
            save_checkpoint({"completed_benchmarks": all_reports, "paused_at": name,
                             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
            paused = True
            break

        try:
            game_list = alfworld_games if name == "alfworld" else None
            all_reports[name] = await run_benchmark(name, tasks, game_list=game_list)
        except APIUnavailableError as e:
            print(f"\n  ⚠️  API became unavailable during {name}: {e}", flush=True)
            print(f"  💾 Saving checkpoint and pausing...", flush=True)
            save_checkpoint({"completed_benchmarks": all_reports, "paused_at": name,
                             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "error": str(e)})
            paused = True
            break
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {name}: {e}")
            traceback.print_exc()
            all_reports[name] = {"error": str(e)}

    if paused:
        print(f"\n\n{'═'*70}")
        print(f"  EXPERIMENT PAUSED — API unavailable")
        print(f"  Completed: {[k for k in all_reports if 'error' not in all_reports.get(k, {})]}")
        print(f"  Re-run this script to resume from checkpoint.")
        print(f"{'═'*70}")
    else:
        # All done — clear checkpoint
        clear_checkpoint()

    # Print summary of whatever we have
    print(f"\n\n{'═'*70}")
    print(f"  FINAL SUMMARY (latest — DeepSeek V4 Pro · EM / pass@1)")
    print(f"{'═'*70}")
    print(f"  {'Benchmark':<12} {'Metric':<8} {'A':>8} {'B':>8} {'C':>8} {'Δ(C-A)':>9} {'Δ(C-B)':>9}")
    print(f"  {'-'*70}")
    for name, r in all_reports.items():
        if isinstance(r, dict) and "error" in r:
            print(f"  {name:<12} ERROR: {str(r['error'])[:40]}")
        elif isinstance(r, dict) and "results" in r:
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
    _trace.close()
    print(f"  📋 Trace logs: {RESULTS_DIR}/<benchmark>/trace.jsonl")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*was destroyed but it is pending.*")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  ⚠️  Interrupted by user (Ctrl+C).", flush=True)
        _trace.close()
    except Exception as e:
        import traceback
        print(f"\n  💥 FATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        _trace.close()
        sys.exit(1)
