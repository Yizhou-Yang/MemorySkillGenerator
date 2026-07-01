#!/usr/bin/env python3
"""Evaluation and scoring utilities for SkillForge experiments.

Provides standard metrics: Exact Match, pass@1, LLM-based judging,
and the unified evaluate_task() dispatcher.
"""
from __future__ import annotations

import re
import unicodedata

from scripts.latest.llm_client import _llm_call_notool, _llm_short_call, llm_extract_answer, llm_judge_answer

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


async def evaluate_task(result: dict, benchmark: str, use_llm_judge: bool = True) -> dict:
    """Primary metric:
       - alfworld: pass@1 (binary won)
       - gaia2: soft recall (action sequence matching via official judge)
       - swebench_dynamic: pass@1 (patch correctness via LLM judge)
       - gaia/locomo: Exact Match
    """
    if benchmark == "alfworld":
        won = bool(result.get("won", False))
        return {"score": 1.0 if won else 0.0, "em": 1.0 if won else 0.0,
                "won": won, "method": "pass@1"}

    if benchmark == "gaia2":
        # Official GAIA2 CLI judge logic: count gate + LLM action matching +
        # config-aware dual mode + alias normalization.
        from latest.eval.gaia2_judge import evaluate_gaia2 as _gaia2_official_judge

        oracle_events = result.get("expected", [])
        event_log = result.get("event_log", [])
        response = (result.get("response") or "").strip()
        oracle_answer = result.get("oracle_answer", "")
        task_desc = result.get("description", "")
        config = (result.get("metadata") or {}).get("config", "execution")

        if not oracle_events and not oracle_answer:
            return {"score": 0.0, "em": 0.0, "method": "gaia2_no_oracle"}
        if not event_log and not response:
            return {"score": 0.0, "em": 0.0, "method": "gaia2_no_actions"}

        # Build the LLM call function for the judge (system_prompt, user_prompt) -> str
        async def _judge_llm_call(system_prompt: str, user_prompt: str) -> str:
            """Adapter: call our LLM infrastructure with system+user prompt."""
            try:
                r = await _llm_call_notool(system_prompt, user_prompt, timeout=60)
                return (r.get("text") or "").strip()
            except Exception as e:
                print(f"[GAIA2 judge] LLM call failed: {e}")
                return ""

        try:
            judge_result = await _gaia2_official_judge(
                _judge_llm_call,
                config=config,
                task=task_desc,
                oracle_events=oracle_events,
                oracle_answer=oracle_answer,
                event_log=event_log,
                agent_response=response,
            )
            return judge_result
        except Exception as e:
            print(f"[GAIA2 judge] Official judge failed: {e}")
            return {"score": 0.0, "em": 0.0, "method": "gaia2_judge_error",
                    "error": str(e)[:200]}

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
    # Match against BOTH the extracted span and the full response (max): a
    # wrong-but-nonempty extraction must not shadow a correct full response.
    em = max(exact_match(extracted, expected), exact_match(response, expected))

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


def compute_partial_results_from_trace(benchmark: str, results_dir: str) -> dict | None:
    """Attempt to compute partial results from existing trace JSONL file."""
    import json as _json
    import os as _os
    from pathlib import Path

    trace_path = Path(results_dir) / benchmark / "trace.jsonl"
    if not trace_path.exists():
        return None
    try:
        with open(trace_path) as f:
            lines = [_json.loads(l) for l in f if l.strip()]
    except Exception:
        return None
    if not lines:
        return None

    groups = {}
    for l in lines:
        g = l.get("group", "?")
        groups.setdefault(g, [])
        groups[g].append(l)

    # Reconstruct partial A/B/C from trace
    scores_per_group = {}
    all_traces_per_group = {}
    for g, items in groups.items():
        valid_scores = [it["score"] for it in items if "score" in it]
        scores_per_group[g] = sum(valid_scores) / max(len(valid_scores), 1) if valid_scores else 0.0
        all_traces_per_group[g] = items

    return {
        "benchmark": benchmark,
        "scores": scores_per_group,
        "n_traces": len(lines),
        "reconstructed_from_trace": True,
    }