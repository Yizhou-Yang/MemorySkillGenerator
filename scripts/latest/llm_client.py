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

# The CodeBuddy Agent SDK is Tencent-internal and only present on the gateway.
# Guard the import so the whole pipeline still imports (and runs, via the
# OpenAI-compatible backend below) for external reviewers who don't have it.
try:
    from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage, ToolUseBlock
    _HAS_CODEBUDDY = True
except Exception:  # no CodeBuddy CLI → reviewer/OpenAI-compatible path
    query = CodeBuddyAgentOptions = AssistantMessage = ToolUseBlock = None
    _HAS_CODEBUDDY = False

try:
    from profiling import add_tokens as _prof_add_tokens
except Exception:  # pragma: no cover - profiling is optional plumbing
    def _prof_add_tokens(*_a, **_k):  # type: ignore
        return None


MODEL = os.environ.get("CODEBUDDY_MODEL", "deepseek-v4-pro")

# ─── OpenAI-compatible backend (reproducibility path) ─────────────────────
# When the CodeBuddy SDK is absent, or LLM_PROVIDER asks for it, every LLM call
# routes to a standard OpenAI-compatible chat endpoint instead. ARE's tool
# schemas are already OpenAI-shaped, so GAIA2 native tool-calling works here too.
# Reviewers set: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (or CODEBUDDY_MODEL).
try:
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except Exception:
    _OpenAI = None
    _HAS_OPENAI = False

# Prefer OPENAI_*; fall back to the DEEPSEEK_* names already in .env.example
# (DeepSeek's endpoint is OpenAI-compatible), so the shipped example just works.
_OPENAI_BASE_URL = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
                    or os.environ.get("DEEPSEEK_BASE_URL"))
_OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY")
                   or os.environ.get("DEEPSEEK_API_KEY") or "")
_OPENAI_MODEL = (os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL")
                 or os.environ.get("CODEBUDDY_MODEL") or MODEL)


def use_openai_backend() -> bool:
    """True when LLM calls should use the OpenAI-compatible endpoint.

    Triggered when the CodeBuddy SDK is unavailable (reviewer machine) or when
    LLM_PROVIDER explicitly selects an OpenAI-compatible provider.
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider in ("openai", "openai_compatible", "vllm", "oai"):
        return True
    return not _HAS_CODEBUDDY


def openai_backend_ready() -> bool:
    """True when the OpenAI-compatible path can actually make a call."""
    return _HAS_OPENAI and bool(_OPENAI_BASE_URL or _OPENAI_API_KEY)


def _openai_client(timeout: int = 60):
    kwargs = {"timeout": timeout, "max_retries": 2}
    if _OPENAI_BASE_URL:
        kwargs["base_url"] = _OPENAI_BASE_URL
    kwargs["api_key"] = _OPENAI_API_KEY or "EMPTY"  # many local servers accept any key
    return _OpenAI(**kwargs)


def _record_openai_usage(resp) -> None:
    try:
        u = getattr(resp, "usage", None)
        if u is not None:
            _prof_add_tokens(getattr(u, "prompt_tokens", 0) or 0,
                             getattr(u, "completion_tokens", 0) or 0, calls=1)
    except Exception:
        pass


def _openai_notool_sync(system_prompt: str, user_prompt: str, timeout: int = 60) -> dict:
    """Single-turn OpenAI-compatible chat (no tools)."""
    if not _HAS_OPENAI:
        return {"text": "", "error": "openai_package_not_installed"}
    if not (_OPENAI_BASE_URL or _OPENAI_API_KEY):
        return {"text": "", "error": "openai_endpoint_not_configured"}
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        resp = _openai_client(timeout).chat.completions.create(
            model=_OPENAI_MODEL, messages=messages, timeout=timeout)
        _record_openai_usage(resp)
        return {"text": resp.choices[0].message.content or "", "error": None}
    except Exception as e:
        return {"text": "", "error": str(e)[:200]}


def _openai_sync(prompt: str, max_turns: int = 1, timeout: int = 60) -> dict:
    """OpenAI-compatible replacement for the CodeBuddy tool-agent path.

    The gateway's built-in tools (web search, bash, ...) don't exist here, so the
    model answers directly. Enough for the pipeline to run end-to-end; search-heavy
    benchmarks are correspondingly weaker without a tool provider.
    """
    r = _openai_notool_sync("", prompt, timeout)
    return {"text": r.get("text", ""), "actions": [], "error": r.get("error")}


def openai_tool_chat(messages: list, tools: list | None,
                     timeout: int = 120, model: str | None = None) -> dict:
    """One OpenAI-compatible chat turn with function-calling tools.

    Returns {"assistant_message": <dict to append>, "tool_calls": [{id,name,arguments}],
             "error": str|None}. Used by the GAIA2 native tool-calling loop.
    """
    if not _HAS_OPENAI:
        return {"assistant_message": None, "tool_calls": [], "error": "openai_package_not_installed"}
    try:
        req = {"model": model or _OPENAI_MODEL, "messages": messages, "timeout": timeout}
        if tools:
            req["tools"] = tools
            req["tool_choice"] = "auto"
        resp = _openai_client(timeout).chat.completions.create(**req)
        _record_openai_usage(resp)
        msg = resp.choices[0].message
        assistant = {"role": "assistant", "content": msg.content or ""}
        parsed = []
        if getattr(msg, "tool_calls", None):
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                parsed.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"assistant_message": assistant, "tool_calls": parsed, "error": None}
    except Exception as e:
        return {"assistant_message": None, "tool_calls": [], "error": str(e)[:200]}


def _record_usage(msg) -> None:
    """Best-effort per-task token accounting for the cost table.

    Duck-types any SDK message for a `usage` payload. We look in three places,
    because Claude-Agent-SDK-style frameworks vary: directly on the message
    (AssistantMessage / terminal ResultMessage), on a nested ``.message`` (some
    SDKs wrap the provider message), and on a top-level dict. The payload may use
    Anthropic names (input_tokens / output_tokens, plus cache_* on input),
    OpenAI names (prompt_tokens / completion_tokens), or only a total; we accept
    all three. Importing no message type keeps an SDK that omits one from ever
    breaking the call path. Fully defensive: any failure is swallowed.
    """
    try:
        usage = getattr(msg, "usage", None)
        if usage is None:
            inner = getattr(msg, "message", None)
            usage = getattr(inner, "usage", None) if inner is not None else None
        if usage is None and isinstance(msg, dict):
            usage = msg.get("usage")
        if usage is None:
            return
        get = usage.get if isinstance(usage, dict) else (lambda k, d=0: getattr(usage, k, d))
        in_tok = get("input_tokens", 0) or get("prompt_tokens", 0)
        in_tok += (get("cache_read_input_tokens", 0) or 0) + (get("cache_creation_input_tokens", 0) or 0)
        out_tok = get("output_tokens", 0) or get("completion_tokens", 0)
        if not in_tok and not out_tok:
            out_tok = get("total_tokens", 0)  # coarse fallback when only a total is exposed
        if in_tok or out_tok:
            _prof_add_tokens(in_tok, out_tok, calls=1)
    except Exception:
        pass

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
    if use_openai_backend():
        return _openai_notool_sync("", prompt, timeout=90).get("text", "")

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
                    _record_usage(msg)
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
    if use_openai_backend():
        return _openai_sync(prompt, max_turns, timeout)

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
                    _record_usage(msg)
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
    if use_openai_backend():
        return _openai_notool_sync(system_prompt, user_prompt, timeout)

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
                    _record_usage(msg)
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