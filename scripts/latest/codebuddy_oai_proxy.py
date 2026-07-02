#!/usr/bin/env python3
"""OpenAI-compatible proxy server wrapping CodeBuddy Agent SDK.

Exposes a /v1/chat/completions endpoint so that terminal-bench's litellm
(inside Terminus-2) can call CodeBuddy as its LLM backend.

Usage:
    # Start the proxy (default port 8741):
    /root/.conda/envs/skillforge/bin/python scripts/latest/codebuddy_oai_proxy.py

    # Then point terminal-bench at it:
    export OPENAI_API_BASE=http://localhost:8741/v1
    export OPENAI_API_KEY=dummy
    terminal-bench run -a terminus-2 -m "openai/hy3-preview-ioa" ...

Architecture:
    Request (OpenAI chat format) -> extract messages -> CodeBuddy SDK query()
    -> collect AssistantMessage text blocks -> return OpenAI response format.

The proxy runs in the skillforge conda env (which has codebuddy_agent_sdk).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "latest"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# CodeBuddy SDK
from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage

# FastAPI / uvicorn
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    sys.exit("ERROR: fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")

# ─── Configuration ─────────────────────────────────────────────────────────

PORT = int(os.environ.get("CODEBUDDY_PROXY_PORT", "8741"))
# Model name reported to clients (cosmetic; CodeBuddy uses its own routing)
DEFAULT_MODEL = os.environ.get("CODEBUDDY_MODEL", "hy3-preview-ioa")
# Max turns for CodeBuddy agent (we want pure text, so keep low)
MAX_TURNS = int(os.environ.get("CODEBUDDY_PROXY_MAX_TURNS", "1"))
# Default timeout per request (seconds)
DEFAULT_TIMEOUT = int(os.environ.get("CODEBUDDY_PROXY_TIMEOUT", "300"))

# Pin small/fast model to same model
for _smf in ("CODEBUDDY_SMALL_FAST_MODEL", "CODEBUDDY_BACKGROUND_MODEL",
             "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
    os.environ.setdefault(_smf, DEFAULT_MODEL)

app = FastAPI(title="CodeBuddy OpenAI Proxy", version="0.1.0")


# ─── Helpers ───────────────────────────────────────────────────────────────

def _messages_to_prompt(messages: list[dict]) -> tuple[str, str]:
    """Convert OpenAI messages array to (system_prompt, user_prompt).

    Terminus-2 sends a system message + user message. We extract them.
    If there are multiple user/assistant turns, we concatenate into the
    user prompt (CodeBuddy SDK is single-turn in our usage).
    """
    system_parts = []
    conversation_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle content blocks (text type)
            content = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            conversation_parts.append(f"User: {content}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content}")

    system_prompt = "\n\n".join(system_parts)
    # If only one user message, use it directly (most common case)
    if len(conversation_parts) == 1 and conversation_parts[0].startswith("User: "):
        user_prompt = conversation_parts[0][6:]  # strip "User: " prefix
    else:
        user_prompt = "\n\n".join(conversation_parts)

    return system_prompt, user_prompt


async def _call_codebuddy(system_prompt: str, user_prompt: str,
                          model: str, timeout: int) -> tuple[str, str | None]:
    """Call CodeBuddy SDK and return (text, error)."""
    opt = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions",
        model=model,
        max_turns=MAX_TURNS,
        cwd="/tmp",
        system_prompt=system_prompt if system_prompt else None,
        # Disable all tools - we want pure text generation
        tools=[],
    )

    text_parts = []
    gen = None
    try:
        async with asyncio.timeout(timeout):
            gen = query(prompt=user_prompt, options=opt)
            async for msg in gen:
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text") and block.text:
                            text_parts.append(block.text)
    except asyncio.TimeoutError:
        if text_parts:
            return "".join(text_parts), None  # partial is better than nothing
        return "", f"timeout_after_{timeout}s"
    except Exception as e:
        if text_parts:
            return "".join(text_parts), None
        return "", str(e)[:300]
    finally:
        if gen is not None:
            try:
                await gen.aclose()
            except Exception:
                pass

    return "".join(text_parts), None


# ─── Endpoints ─────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    body = await request.json()

    messages = body.get("messages", [])
    model = body.get("model", DEFAULT_MODEL)
    timeout = DEFAULT_TIMEOUT

    # Strip provider prefix if present (litellm sends "openai/model-name")
    if "/" in model:
        model = model.split("/", 1)[1]

    system_prompt, user_prompt = _messages_to_prompt(messages)

    if not user_prompt:
        return JSONResponse(status_code=400, content={
            "error": {"message": "No user message found", "type": "invalid_request_error"}
        })

    text, error = await _call_codebuddy(system_prompt, user_prompt, model, timeout)

    if error and not text:
        return JSONResponse(status_code=500, content={
            "error": {"message": f"CodeBuddy SDK error: {error}", "type": "server_error"}
        })

    # Build OpenAI-compatible response
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(user_prompt) // 4,  # rough estimate
            "completion_tokens": len(text) // 4,
            "total_tokens": (len(user_prompt) + len(text)) // 4,
        },
    }

    return JSONResponse(content=response)


@app.get("/v1/models")
async def list_models():
    """List available models (for litellm compatibility)."""
    return JSONResponse(content={
        "object": "list",
        "data": [{
            "id": DEFAULT_MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "codebuddy",
        }],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "model": DEFAULT_MODEL, "port": PORT}


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[CodeBuddy OAI Proxy] Starting on port {PORT}")
    print(f"[CodeBuddy OAI Proxy] Model: {DEFAULT_MODEL}")
    print(f"[CodeBuddy OAI Proxy] Max turns: {MAX_TURNS}")
    print(f"[CodeBuddy OAI Proxy] Timeout: {DEFAULT_TIMEOUT}s")
    print(f"[CodeBuddy OAI Proxy] Endpoint: http://localhost:{PORT}/v1/chat/completions")
    print()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
