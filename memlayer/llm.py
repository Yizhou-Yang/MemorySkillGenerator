"""Standalone LLM routing for memlayer installed OUTSIDE the research harness.

Inside the repo, `scripts.latest.llm_client` is preferred (benchmark routing,
SDK paths, usage accounting); memlayer's bridge falls back here only when that
module is not importable. Semantics mirror the harness exactly:

  * the CRITIC is one fixed external model, configured by endpoint
    (HY3_BASE_URL / HY3_API_KEY, or CRITIC_BASE_URL / CRITIC_API_KEY) and
    FAIL-LOUD when unset — a memory layer must never silently let the acting
    model grade its own work;
  * METADATA_AUTHOR decides who writes the store's narrative metadata:
    "critic" (default) routes the reviewer pen to the critic endpoint,
    "backbone" routes it to the general endpoint (OPENAI_API_BASE /
    OPENAI_API_KEY / OPENAI_MODEL).

Pure stdlib (urllib) on purpose: the SDK's core install stays dependency-light.
"""
from __future__ import annotations

import json
import os
import urllib.request

CRITIC_MODEL = "hy3"
METADATA_AUTHOR = (os.environ.get("METADATA_AUTHOR") or "critic").strip().lower()
if METADATA_AUTHOR not in ("critic", "backbone"):
    raise RuntimeError(
        f"METADATA_AUTHOR must be 'critic' or 'backbone', got {METADATA_AUTHOR!r}")


def metadata_author_id() -> str:
    """Who authors the narrative metadata — constant per process, recorded on
    every Experience so stores mixing authors can never pass for one mechanism."""
    return METADATA_AUTHOR


def _chat(prompt: str, base_url: str, api_key: str, model: str,
          timeout: int = 90) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def llm_critic_fn(prompt: str) -> str:
    """The fixed external critic. Fail-loud when unconfigured — falling back to
    the acting model as its own critic is the failure this layer exists to
    prevent."""
    base = (os.environ.get("HY3_BASE_URL") or os.environ.get("CRITIC_BASE_URL") or "").strip()
    key = (os.environ.get("HY3_API_KEY") or os.environ.get("CRITIC_API_KEY") or "").strip()
    if not base or not key:
        raise RuntimeError(
            "critic endpoint not configured: set HY3_BASE_URL/HY3_API_KEY (or "
            "CRITIC_BASE_URL/CRITIC_API_KEY). Refusing to fall back to the "
            "acting model as its own critic.")
    model = (os.environ.get("CRITIC_MODEL_ID") or CRITIC_MODEL).strip()
    return _chat(prompt, base, key, model)


def llm_review_fn(prompt: str) -> str:
    """General single-turn call on the configured backbone endpoint."""
    base = (os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "").strip()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    model = (os.environ.get("OPENAI_MODEL") or "").strip()
    if not base or not model:
        raise RuntimeError(
            "backbone endpoint not configured: set OPENAI_API_BASE / "
            "OPENAI_API_KEY / OPENAI_MODEL (or pass llm= to MemoryLayer).")
    return _chat(prompt, base, key, model)


def llm_metadata_fn(prompt: str) -> str:
    """The reviewer pen (see METADATA_AUTHOR above)."""
    if METADATA_AUTHOR == "critic":
        return llm_critic_fn(prompt)
    return llm_review_fn(prompt)
