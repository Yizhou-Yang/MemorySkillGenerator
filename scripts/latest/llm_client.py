#!/usr/bin/env python3
"""LLM client helpers for SkillForge experiments.

Provides synchronous and asynchronous wrappers for CodeBuddy Agent SDK queries.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import os
import re

# SDK tool-call markup leaks into text blocks (</arg_value:..>, </tool_call:..>,
# <think:..>) and poisons downstream consumers (bash syntax errors, failed
# exact-match). Scrub at the source so EVERY caller gets clean text.
_SDK_MARKUP_RE = re.compile(
    r"</?(?:think|tool_calls?|name|args?|arg_key|arg_value)(?::[A-Za-z0-9_-]+)?>"
)
import time

# The CodeBuddy Agent SDK is Tencent-internal and only present on the gateway; guard
# the import so the pipeline still runs via the OpenAI-compatible backend below.
try:
    from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage, ToolUseBlock
    _HAS_CODEBUDDY = True
except Exception:  # no CodeBuddy CLI → reviewer/OpenAI-compatible path
    query = CodeBuddyAgentOptions = AssistantMessage = ToolUseBlock = None
    _HAS_CODEBUDDY = False

try:
    # Import via the SAME package path the runner uses: a bare `from profiling import`
    # binds a second module copy with its own ContextVar accumulator the runner never
    # reads, silently zeroing every prof_llm_calls/prof_tok_* in the trace.
    from scripts.latest.profiling import add_tokens as _prof_add_tokens
except Exception:  # pragma: no cover - profiling is optional plumbing
    try:
        from profiling import add_tokens as _prof_add_tokens
    except Exception:
        def _prof_add_tokens(*_a, **_k):  # type: ignore
            return None


MODEL = os.environ.get("CODEBUDDY_MODEL", "deepseek-v4-pro").lower()

# ─── OpenAI-compatible backend (reproducibility path) ─────────────────────
# Used when the CodeBuddy SDK is absent or LLM_PROVIDER asks for it.
# Set: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (or CODEBUDDY_MODEL).
try:
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except Exception:
    _OpenAI = None
    _HAS_OPENAI = False

# Prefer OPENAI_*; fall back to the DEEPSEEK_* names in .env.example.
_OPENAI_BASE_URL = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
                    or os.environ.get("DEEPSEEK_BASE_URL"))
_OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY")
                   or os.environ.get("DEEPSEEK_API_KEY") or "")
if (os.environ.get("LLM_PROVIDER") or "").lower() in ("vllm", "openrouter"):
    # Self-host/OpenRouter serves CODEBUDDY_MODEL; a DEEPSEEK_MODEL left in .env must
    # NOT win here — it 404s against a vLLM that only serves the backbone.
    _OPENAI_MODEL = (os.environ.get("OPENAI_MODEL")
                     or os.environ.get("CODEBUDDY_MODEL") or MODEL).lower()
else:
    _OPENAI_MODEL = (os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL")
                     or os.environ.get("CODEBUDDY_MODEL") or MODEL).lower()

# ─── OpenRouter (opt-in, OpenAI-compatible) ───────────────────────────────
# Activated ONLY by LLM_PROVIDER=openrouter. These free models are reasoning models:
# they stream the chain into `reasoning` and the answer into `content`, hence the
# generous max_tokens and the `reasoning` fallback when `content` is empty.
_OPENROUTER_ALIASES = {
    "nemotron-super":   "nvidia/nemotron-3-super-120b-a12b:free",
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b:free",
    "hy3":              "tencent/hy3:free",
    "hy3-preview":      "tencent/hy3:free",
    # Other free-tier models are persistently 429-throttled; pass a full
    # "vendor/x:free" id to try them off-peak.
}
_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "0") or 0)


def _openrouter_model(name: str) -> str:
    n = (name or "").strip().lower()
    if "/" in n:                      # already a full OpenRouter id
        return name
    return _OPENROUTER_ALIASES.get(n, name)


if os.environ.get("LLM_PROVIDER", "").lower() == "openrouter":
    _OPENAI_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    _OPENAI_API_KEY = os.environ.get("OPENROUTER_API_KEY") or _OPENAI_API_KEY
    _OPENAI_MODEL = _openrouter_model(os.environ.get("OPENROUTER_MODEL")
                                      or os.environ.get("CODEBUDDY_MODEL") or MODEL)
    if _MAX_TOKENS <= 0:
        _MAX_TOKENS = int(os.environ.get("OPENROUTER_MAX_TOKENS", "8192"))


def served_model() -> str:
    """The model id actually sent to the endpoint, whatever the run is labelled.

    The results directory and every trace row are keyed on CODEBUDDY_MODEL, but
    the request carries this: with LLM_PROVIDER=vllm an OPENAI_MODEL left in
    .env wins, so exporting CODEBUDDY_MODEL=<backbone> relabels a run without
    changing which model answers. Callers compare the two and refuse to write.
    """
    return (_OPENAI_MODEL or MODEL or "").lower()


def _gen_kwargs() -> dict:
    """Extra generation kwargs for OpenAI-compatible calls; only sets what is configured.

    GEN_TEMPERATURE must be set for ALL arms of a run or none — pairing a greedy arm
    against a sampled one folds the temperature difference into the treatment estimate.
    """
    kw: dict = {}
    if _MAX_TOKENS > 0:
        kw["max_tokens"] = _MAX_TOKENS
    t = os.environ.get("GEN_TEMPERATURE")
    if t not in (None, ""):
        kw["temperature"] = float(t)
    # Reasoning models (HY3 via taiji) otherwise leave `content` empty / emit private
    # tool-call markup that third-party libs can't parse (retry storms).
    # OPENAI_REASONING_EFFORT=no_think makes the endpoint behave as a plain chat model.
    re = os.environ.get("OPENAI_REASONING_EFFORT")
    if re:
        kw["reasoning_effort"] = re
    return kw


def _msg_text(msg) -> str:
    """Final answer text from an OpenAI-compatible message. Reasoning models put the
    chain in `reasoning` and the answer in `content`; fall back to `reasoning` when
    `content` is empty (budget spent reasoning) rather than returning nothing."""
    txt = getattr(msg, "content", None)
    if txt:
        return txt
    return getattr(msg, "reasoning", None) or ""


def use_openai_backend() -> bool:
    """True when LLM calls should use the OpenAI-compatible endpoint (SDK unavailable,
    or LLM_PROVIDER selects an OpenAI-compatible provider)."""
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider in ("openai", "openai_compatible", "vllm", "oai", "openrouter"):
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
        pt = (getattr(u, "prompt_tokens", 0) or 0) if u is not None else 0
        ct = (getattr(u, "completion_tokens", 0) or 0) if u is not None else 0
        # calls=1 unconditionally: a call with no usage payload must still count, else
        # prof_llm_calls=0 conflates "no usage surfaced" with "never called".
        _prof_add_tokens(pt, ct, calls=1)
    except Exception:
        pass


# Some OpenAI-compatible proxies ONLY return SSE and ignore stream=false, so a
# non-streaming create() gets a raw event stream ("'str' object has no attribute
# 'choices'"). Set this to force streaming and reassemble the response ourselves.
OPENAI_FORCE_STREAM = os.environ.get("OPENAI_FORCE_STREAM", "").strip().lower() in ("1", "true", "yes")


class _StreamMsg:
    """Minimal stand-in for a non-streaming message, built from SSE deltas."""
    __slots__ = ("content", "tool_calls")

    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls or None


class _TC:
    __slots__ = ("id", "type", "function")

    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


def _openai_stream_create(client, req: dict):
    """Call create() with stream=True and collect content + tool_calls into an
    object exposing `.choices[0].message` and `.usage`, like a normal response."""
    req = {k: v for k, v in req.items() if k != "timeout"}
    req["stream"] = True
    try:
        req["stream_options"] = {"include_usage": True}
    except Exception:
        pass
    content = ""
    tcs: dict = {}   # index -> {id, name, args}
    finish = None
    usage = None
    for chunk in client.chat.completions.create(**req):
        u = getattr(chunk, "usage", None)
        if u is not None:
            usage = u
        for ch in (getattr(chunk, "choices", None) or []):
            delta = getattr(ch, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "content", None):
                content += delta.content
            for tc in (getattr(delta, "tool_calls", None) or []):
                idx = getattr(tc, "index", 0) or 0
                slot = tcs.setdefault(idx, {"id": None, "name": None, "args": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments
            if getattr(ch, "finish_reason", None):
                finish = ch.finish_reason
    tool_calls = [_TC(v["id"] or f"call_{i}", v["name"] or "", v["args"])
                  for i, v in sorted(tcs.items())] or None
    msg = _StreamMsg(content, tool_calls)
    choice = type("C", (), {"message": msg, "finish_reason": finish})()
    return type("R", (), {"choices": [choice], "usage": usage})()


# Private tool-call markup leaking into a no-tools completion (HY3-style
# "<tool_calls:hash>"): the model tried to call tools that don't exist on this path.
_TOOL_LEAK_RE = re.compile(r"<tool_calls?[:>]", re.I)


def _openai_notool_sync(system_prompt: str, user_prompt: str, timeout: int = 60,
                        model: str | None = None, base_url: str | None = None,
                        api_key: str | None = None) -> dict:
    """Single-turn OpenAI-compatible chat (no tools).

    `model`/`base_url`/`api_key` override the backbone's endpoint (used to route the
    critic at a designated model instead of the backbone itself).
    """
    if not _HAS_OPENAI:
        return {"text": "", "error": "openai_package_not_installed"}
    url = base_url if base_url is not None else _OPENAI_BASE_URL
    key = api_key if api_key is not None else _OPENAI_API_KEY
    if not (url or key):
        return {"text": "", "error": "openai_endpoint_not_configured"}
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        client = (_openai_client(timeout) if base_url is None and api_key is None
                  else _OpenAI(base_url=url or None, api_key=key or "EMPTY", timeout=timeout))
        req = {"model": (model or _OPENAI_MODEL), "messages": messages,
               "timeout": timeout, **_gen_kwargs()}
        resp = (_openai_stream_create(client, req) if OPENAI_FORCE_STREAM
                else client.chat.completions.create(**req))
        _record_openai_usage(resp)
        txt = _msg_text(resp.choices[0].message)
        if txt and _TOOL_LEAK_RE.search(txt):
            # Agentic backbones emit pure tool-call markup on this no-tools path and
            # score 0 while looking healthy. Retry ONCE with an explicit no-tools
            # instruction — only when markup is detected, so clean runs are unchanged.
            retry_msgs = [{"role": "system", "content":
                           "No tools are available in this environment. Do not "
                           "emit tool-call markup. Answer directly, reasoning "
                           "in plain text, and end with the final answer."}] \
                + [m for m in messages if m["role"] != "system"]
            resp = (_openai_stream_create(client, {**req, "messages": retry_msgs})
                    if OPENAI_FORCE_STREAM
                    else client.chat.completions.create(**{**req, "messages": retry_msgs}))
            _record_openai_usage(resp)
            txt2 = _msg_text(resp.choices[0].message)
            if txt2 and not _TOOL_LEAK_RE.search(txt2):
                txt = txt2
        return {"text": txt, "error": None}
    except Exception as e:
        return {"text": "", "error": str(e)[:200]}


# ─── Unified critic routing (hardcoded HY3) ───────────────────────────────
# The critic is ONE model, HY3, for every backbone and baseline — never the backbone
# itself (a self-critic inverts on weak backbones, and a per-backbone critic makes the
# sweep incomparable). Only the *endpoint* is configurable; the model identity is fixed.
CRITIC_MODEL = (os.environ.get("CRITIC_MODEL_ID") or "hy3").strip()
CRITIC_BASE_URL = (os.environ.get("HY3_BASE_URL") or os.environ.get("CRITIC_BASE_URL") or "").strip() or None
CRITIC_API_KEY = (os.environ.get("HY3_API_KEY") or os.environ.get("CRITIC_API_KEY") or "").strip() or None
CRITIC_VIA_SDK = os.environ.get("CRITIC_VIA_SDK", "").strip().lower() in ("1", "true", "yes")


# ─── Designated judge routing ────────────────────────────────────────────
# Unset, llm_judge_answer's tie-breaker runs on the backbone itself (the model being
# evaluated decides whether its answer counts). JUDGE_MODEL pins it to one fixed
# external model for every arm and backbone.
JUDGE_MODEL = (os.environ.get("JUDGE_MODEL") or "").strip().lower() or None
# Route the judge through the CodeBuddy SDK rather than an OpenAI-compatible endpoint
# (deepseek-v4-pro lives behind the internal gateway, not an OpenAI URL). Defaults on
# when a judge is named, no JUDGE_BASE_URL/JUDGE_API_KEY is given, and the SDK exists.
JUDGE_VIA_SDK = (os.environ.get("JUDGE_VIA_SDK", "").strip().lower() in ("1", "true", "yes")) or (
    JUDGE_MODEL is not None
    and not (os.environ.get("JUDGE_BASE_URL") or os.environ.get("JUDGE_API_KEY")))
JUDGE_BASE_URL = (os.environ.get("JUDGE_BASE_URL") or "").strip() or None
JUDGE_API_KEY = (os.environ.get("JUDGE_API_KEY") or "").strip() or None


def judge_model_id() -> str:
    """What judged this run's tie-breaks — recorded on every trace row so runs judged
    by different models can never be silently pooled."""
    return JUDGE_MODEL or _OPENAI_MODEL or MODEL


def critic_model_id() -> str:
    """The critic is always HY3 — recorded on every trace row so the field is a
    constant the gate can assert on."""
    return CRITIC_MODEL


def critic_preflight() -> tuple:
    """One tiny call so a dead critic fails at launch, not silently mid-sweep.

    When the critic endpoint is unreachable the refinement path swallows the
    error and stamps its fallback quality score, so the arm completes with
    error==0 and looks healthy while carrying no critic judgment at all --
    which is the one thing arm C is supposed to measure.
    """
    try:
        out = llm_critic_fn("Reply with exactly: OK")
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"
    if not (out or "").strip():
        return False, "empty response"
    return True, (out or "").strip()[:40]


def llm_critic_fn(prompt: str) -> str:
    """Synchronous LLM call for the CRITIC only. Always HY3 — never the backbone;
    raises if HY3's endpoint is unconfigured rather than falling back to a
    self-critic (the exact failure this pins down)."""
    if not (CRITIC_VIA_SDK and _HAS_CODEBUDDY) and not (CRITIC_BASE_URL or CRITIC_API_KEY):
        raise RuntimeError(
            "critic (HY3) endpoint not configured: set HY3_BASE_URL/HY3_API_KEY "
            "(or CRITIC_VIA_SDK=1). Refusing to fall back to the backbone as its "
            "own critic. Until HY3's API is wired, run with DEFER_CRITIC=1 (A/B only).")
    if CRITIC_VIA_SDK and _HAS_CODEBUDDY:
        # Called from synchronous curation code, so drive the async one-shot to
        # completion on a private loop.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_sdk_judge_call(prompt, CRITIC_MODEL, timeout=90))
        finally:
            _shutdown_loop(loop)
    return _openai_notool_sync("", prompt, timeout=90, model=CRITIC_MODEL,
                               base_url=CRITIC_BASE_URL,
                               api_key=CRITIC_API_KEY).get("text", "")


# ─── Metadata authorship (who writes the curated layer's narratives) ───────
#   critic   (default) — the fixed external critic (HY3) authors them; fail-loud if
#              no HY3 endpoint, never silently fall back to the backbone.
#   backbone — legacy behaviour, kept as the comparison arm.
# Recorded in protocol_hash: changing it makes a NEW arm, old C rows are not poolable.
METADATA_AUTHOR = (os.environ.get("METADATA_AUTHOR") or "critic").strip().lower()
if METADATA_AUTHOR not in ("critic", "backbone"):
    raise RuntimeError(f"METADATA_AUTHOR must be 'critic' or 'backbone', got {METADATA_AUTHOR!r}")


def metadata_author_id() -> str:
    """Who authors the curated layer's narrative metadata — constant per run, stamped
    on trace rows/Experiences so differing authorship can never be silently pooled."""
    return METADATA_AUTHOR


def llm_metadata_fn(prompt: str) -> str:
    """The reviewer pen for ai_review_experience: HY3 under METADATA_AUTHOR=
    critic (fail-loud if unconfigured), the backbone under =backbone."""
    if METADATA_AUTHOR == "critic":
        return llm_critic_fn(prompt)
    return llm_review_fn(prompt)


def _openai_sync(prompt: str, max_turns: int = 1, timeout: int = 60) -> dict:
    """OpenAI-compatible replacement for the CodeBuddy tool-agent path. The gateway's
    built-in tools don't exist here, so the model answers directly — search-heavy
    benchmarks are correspondingly weaker."""
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
        req = {"model": model or _OPENAI_MODEL, "messages": messages, "timeout": timeout,
               **_gen_kwargs()}
        if tools:
            req["tools"] = tools
            req["tool_choice"] = "auto"
        client = _openai_client(timeout)
        resp = (_openai_stream_create(client, req) if OPENAI_FORCE_STREAM
                else client.chat.completions.create(**req))
        _record_openai_usage(resp)
        msg = resp.choices[0].message
        assistant = {"role": "assistant", "content": _msg_text(msg)}
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
    """Best-effort per-task token accounting. Duck-types any SDK message for a `usage`
    payload in three places (message, nested `.message`, dict) under Anthropic, OpenAI
    or total-only field names. Fully defensive: any failure is swallowed."""
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
            # calls=0: the invocation is counted once per query() in the _query_*_sync
            # bodies; SDKs attach usage to several streamed messages per call.
            _prof_add_tokens(in_tok, out_tok, calls=0)
    except Exception:
        pass

# Pin the SDK's background / "small-fast" model to MODEL: these SDKs route auxiliary
# calls (summarization, orchestration) to a separate cheap model by default, so the
# pipeline would otherwise run on two models. Unknown names are harmless no-ops.
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
    Auth failures are FATAL (True immediately); rate-limit/timeout are TRANSIENT
    (True only after _API_FAILURE_THRESHOLD consecutive failures)."""
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
        return _SDK_MARKUP_RE.sub("", result)

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
        # Count the invocation itself: the SDK does not always surface a usage payload,
        # and prof_llm_calls must distinguish "called, no usage" from "never called".
        _prof_add_tokens(0, 0, calls=1)
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
            return {"text": _SDK_MARKUP_RE.sub("", text), "actions": actions, "error": str(e)[:200] if not text else None}
        finally:
            if gen is not None:
                try:
                    await gen.aclose()
                except Exception:
                    pass
        return {"text": _SDK_MARKUP_RE.sub("", text), "actions": actions, "error": None}

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
    # run_in_executor does NOT propagate contextvars (asyncio.to_thread does), so
    # without this copy the profiling accumulator is invisible in the worker thread
    # and every token/call count silently no-ops.
    ctx = contextvars.copy_context()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: ctx.run(_query_sync, prompt, max_turns, timeout)),
            timeout=hard_timeout
        )
    except asyncio.TimeoutError:
        return {"text": "", "actions": [], "error": f"hard_timeout_after_{hard_timeout}s"}


def _query_notool_sync(system_prompt: str, user_prompt: str, timeout: int = 60) -> dict:
    """Pure text generation without CodeBuddy tools (for GAIA2 ARE interaction).

    Do NOT set allowed_tools=[]: the SDK returns 'Authentication required' for empty
    tool lists. Tool use is suppressed at the instruction level by the ARE prompt.
    """
    if use_openai_backend():
        return _openai_notool_sync(system_prompt, user_prompt, timeout)

    async def _inner():
        _prof_add_tokens(0, 0, calls=1)  # count the invocation (see _query_sync)
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
    ctx = contextvars.copy_context()  # propagate the profiling accumulator (see _llm_call)
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: ctx.run(_query_notool_sync, system_prompt, user_prompt, timeout)),
            timeout=hard_timeout
        )
    except asyncio.TimeoutError:
        return {"text": "", "error": f"hard_timeout_after_{hard_timeout}s"}


async def _llm_short_call(prompt: str, max_turns: int = 1, timeout: int = 30) -> str:
    """Short LLM call returning text only."""
    r = await _llm_call(prompt, max_turns=max_turns, timeout=timeout)
    return (r.get("text") or "").strip()


def _regex_extract_answer(response: str) -> str | None:
    """Fast regex answer extraction (pre-filter that skips the LLM call).
    Returns the extracted answer, or None if extraction is not deterministic."""
    text = response.strip()
    if not text:
        return None

    # Short single-line responses are already answers
    if len(text) < 200 and '\n' not in text:
        return text

    # Explicit ANSWER markers (highest confidence)
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
        clean = re.sub(r'\*\*(?:Answer|Result|Final)\*\*[:：]?\s*', '', line).strip()
        if clean:
            return clean

    return None


async def llm_extract_answer(response: str, question: str) -> str:
    if len(response.split()) < 30:
        return response

    regex_answer = _regex_extract_answer(response)
    if regex_answer and len(regex_answer) < 500:
        return regex_answer

    prompt = (
        "Extract ONLY the final answer from this response. Output just the answer, nothing else.\n\n"
        f"Question: {question}\n\nResponse: {response}\n\n"
        "Final answer (concise, just the key fact/number/name):"
    )
    out = await _llm_short_call(prompt, max_turns=1, timeout=30)
    return out or response


async def _sdk_judge_call(prompt: str, judge_model: str, timeout: int = 30) -> str:
    """One-shot CodeBuddy-SDK call whose model is the JUDGE, not the backbone — the
    judge must stay independent of the backbone under evaluation (the confound gate
    G8 exists to stop). Scoring call: no tools, max_turns=1, empty system prompt."""
    if not _HAS_CODEBUDDY:
        return ""
    _prof_add_tokens(0, 0, calls=1)

    async def _drive():
        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=judge_model, max_turns=1, cwd="/tmp")
        text = ""
        gen = query(prompt=prompt, options=opt)
        try:
            async for msg in gen:
                _record_usage(msg)
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text") and block.text:
                            text += block.text
                    if text:
                        break
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass
        return text

    # asyncio.timeout is 3.11+; this pipeline runs on 3.10, where it raises
    # AttributeError that a bare except swallows, collapsing every judge score to 0.
    # wait_for is the 3.10-safe equivalent — do not "modernize" this.
    try:
        return await asyncio.wait_for(_drive(), timeout=timeout)
    except asyncio.TimeoutError:
        return ""


async def llm_judge_answer(response: str, expected: str, question: str) -> float:
    if not response or not expected:
        return 0.0
    prompt = (
        "Judge if the response correctly answers the question. Score 0.0 to 1.0.\n\n"
        f"Question: {question}\nExpected answer: {expected}\n"
        f"Model response: {response}\n\n"
        "Score (0.0=wrong, 0.5=partially, 1.0=fully correct). Output ONLY a number:"
    )
    if JUDGE_MODEL is not None and JUDGE_VIA_SDK and _HAS_CODEBUDDY:
        out = (await _sdk_judge_call(prompt, JUDGE_MODEL, timeout=30)).strip()
    elif JUDGE_MODEL is not None:
        r = await asyncio.to_thread(
            _openai_notool_sync, "", prompt, 30,
            JUDGE_MODEL, JUDGE_BASE_URL, JUDGE_API_KEY)
        out = (r.get("text") or "").strip()
    else:
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