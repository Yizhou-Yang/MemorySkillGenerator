#!/usr/bin/env python3
"""LLM client helpers for SkillForge experiments.

Provides synchronous and asynchronous wrappers for CodeBuddy Agent SDK queries.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import time

from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage, ToolUseBlock


MODEL = os.environ.get("CODEBUDDY_MODEL", "deepseek-v4-pro")

# Pin the SDK's background / "small-fast" model to the SAME model as the main
# agent. Claude-Agent-SDK-style frameworks (codebuddy included) route auxiliary
# calls — conversation summarization, tool orchestration, quick classification —
# to a separate cheap model that DEFAULTS to a Haiku-class model. That showed up
# as separate Haiku spend in the budget tracker. We force every known small/fast
# model env var to MODEL so the whole pipeline runs on one model only.
# (Unknown names are harmless no-ops; if Haiku spend persists, the SDK uses a
#  different knob — pass it via CodeBuddyAgentOptions instead.)
for _smf in ("CODEBUDDY_SMALL_FAST_MODEL", "CODEBUDDY_BACKGROUND_MODEL",
             "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL",
             "CLAUDE_SMALL_FAST_MODEL"):
    os.environ[_smf] = MODEL

_api_consecutive_failures = 0
_API_FAILURE_THRESHOLD = 3


# ─── API availability detection ───────────────────────────────────────────

async def probe_api_available() -> bool:
    """Quick probe to check if DeepSeek V4 Pro API is responding."""
    r = await _llm_call("Reply with exactly: OK", max_turns=1, timeout=30)
    if r.get("error"):
        err = str(r["error"])
        if "429" in err or "rate_limit" in err or "timeout" in err or "quota_exceeded" in err:
            return False
    if not r.get("text"):
        return False
    return True


def _check_api_error(r: dict) -> bool:
    """Check if result indicates API unavailability. Returns True if API is down.

    Auth failures are FATAL — return True immediately (no retry).
    Rate-limit/timeout failures are TRANSIENT — return True only after threshold.
    """
    global _api_consecutive_failures

    # ── FATAL: authentication failures (no point retrying) ──
    if r.get("error"):
        err = str(r["error"])
        if any(kw in err for kw in ("auth", "Authentication", "login", "unauthorized")):
            _api_consecutive_failures = _API_FAILURE_THRESHOLD  # force immediate
            return True

    text = (r.get("text") or "").lower()
    if "authentication required" in text or "please use /login" in text:
        _api_consecutive_failures = _API_FAILURE_THRESHOLD
        return True

    # ── TRANSIENT: rate-limit / timeout (retry up to threshold) ──
    if r.get("error"):
        err = str(r["error"])
        if any(kw in err for kw in ("429", "rate_limit", "timeout", "quota_exceeded")):
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

def save_checkpoint(state: dict, checkpoint_file: str):
    """Save experiment state to checkpoint file."""
    import os as _os
    _os.makedirs(_os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    print(f"  💾 Checkpoint saved: {checkpoint_file}", flush=True)


def load_checkpoint(checkpoint_file: str) -> dict:
    """Load experiment state from checkpoint file if exists."""
    import os as _os
    if _os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {}


def clear_checkpoint(checkpoint_file: str):
    """Remove checkpoint file after successful completion."""
    import os as _os
    if _os.path.exists(checkpoint_file):
        _os.remove(checkpoint_file)


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
                                if '429' in block.text and 'quota_exceeded' in block.text:
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
    """Pure text generation without CodeBuddy tools (for GAIA2 ARE interaction).

    NOTE: We do NOT set allowed_tools=[] because it triggers an auth bug
    in the CodeBuddy SDK (returns 'Authentication required' for empty tool lists).
    Instead, the ARE system prompt instructs the model to use only NEXT_OP/PARAMS
    format, which prevents tool usage at the instruction level.
    """
    async def _inner():
        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=50, cwd="/tmp",
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
                                if '429' in block.text and 'quota_exceeded' in block.text:
                                    return {"text": "", "error": "429_rate_limit"}
                                if 'authentication required' in block.text.lower() or '/login' in block.text.lower():
                                    return {"text": "", "error": "authentication_required"}
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


def _regex_extract_answer(response: str) -> str | None:
    """Fast regex-based answer extraction (pre-filter before LLM call).

    Uses deterministic regex patterns: if we can extract the answer
    via explicit markers or clean last-line heuristics, we skip the
    LLM call entirely — saving time and API cost.

    Returns extracted answer string, or None if regex extraction fails.
    """
    text = response.strip()
    if not text:
        return None

    # Short single-line responses are already answers
    if len(text) < 200 and '\n' not in text:
        return text

    # Try explicit ANSWER markers (highest confidence)
    answer_markers = [
        re.compile(r'(?:^|\n)(?:\d+\.?\s*)?(?:ANSWER|Final Answer|Answer)[:：]\s*(.+?)(?:\n|$)',
                   re.IGNORECASE | re.MULTILINE),
        re.compile(r'^\*\*Answer\*\*[:：]\s*(.+?)(?:\n|$)',
                   re.IGNORECASE | re.MULTILINE),
    ]
    for marker in answer_markers:
        m = marker.search(text)
        if m:
            answer = m.group(1).strip()
            if answer and len(answer) >= 1:
                return answer

    # Walk lines backwards: skip reasoning lines, return first clean answer line
    reasoning_start = re.compile(
        r'^(?:Let me|I need|I\'ll|I will|Now|Based on|From the|'
        r'The answer|The result|I (?:now )?have|I don\'t|'
        r'However|Therefore|Thus|So|We (?:can|need)|'
        r'According to|Looking at|After |First[,. ]|Second[,. ]|'
        r'To (?:find|answer|get|determine)|What (?:is|are))',
        re.IGNORECASE
    )
    reasoning_content = re.compile(
        r'(?:let me|I need to|I\'ll|break (?:this|it) down|step \d+|'
        r'search for|look up|find the|check the)',
        re.IGNORECASE
    )

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in reversed(lines):
        if reasoning_start.match(line):
            continue
        if reasoning_content.search(line):
            continue
        if re.match(r'^\*\*Step', line):
            continue
        if len(line) < 2:
            continue
        # Clean bold markers
        clean = re.sub(r'\*\*(?:Answer|Result|Final)\*\*[:：]?\s*', '', line).strip()
        if clean:
            return clean

    return None  # Could not extract deterministically


async def llm_extract_answer(response: str, question: str) -> str:
    if len(response.split()) < 30:
        return response

    # Fast path: try regex extraction first (no LLM call needed)
    regex_answer = _regex_extract_answer(response)
    if regex_answer and len(regex_answer) < 500:
        return regex_answer

    # Slow path: use LLM to extract answer from verbose response
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