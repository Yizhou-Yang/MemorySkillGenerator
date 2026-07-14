#!/usr/bin/env python3
"""End-to-end smoke test for the OpenRouter backend, through the real llm_client
path (so it exercises the alias mapping, the generous max_tokens, and the
reasoning->content fallback exactly as a benchmark run would).

Usage:
    LLM_PROVIDER=openrouter OPENROUTER_API_KEY=sk-or-... \
        python scripts/latest/openrouter_smoke.py

Leaves the CodeBuddy path untouched: it only runs when LLM_PROVIDER=openrouter.
"""
import os
import sys

os.environ.setdefault("LLM_PROVIDER", "openrouter")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.latest import llm_client as L  # noqa: E402

if not os.environ.get("OPENROUTER_API_KEY"):
    sys.exit("set OPENROUTER_API_KEY first (see .env.example).")
if L._OPENAI_BASE_URL and "openrouter" not in L._OPENAI_BASE_URL:
    sys.exit(f"provider not routed to OpenRouter (base_url={L._OPENAI_BASE_URL}); "
             "run with LLM_PROVIDER=openrouter.")

PROMPT = "Reply with exactly the word READY and nothing else."
MODELS = ["hy3", "nemotron-super"]

print(f"base_url={L._OPENAI_BASE_URL}  max_tokens={L._MAX_TOKENS}\n")
for alias in MODELS:
    L._OPENAI_MODEL = L._openrouter_model(alias)          # per-model override
    r = L._openai_notool_sync("", PROMPT, timeout=120)
    text = (r.get("text") or "").strip()
    tag = "OK   " if text else "EMPTY"
    print(f"[{tag}] {alias:14} -> {L._OPENAI_MODEL:42} "
          f"| {text[:50]!r}  err={r.get('error')}")
