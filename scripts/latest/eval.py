#!/usr/bin/env python3
"""Evaluation and scoring utilities for SkillForge experiments.

Provides answer normalization / Exact Match helpers and the partial-report
reconstruction. The evaluate_task() dispatcher lives in latest_runner.py.
"""
from __future__ import annotations

import re
import unicodedata

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


# NOTE: the task-scoring dispatcher (`evaluate_task`) lives in
# scripts/latest/latest_runner.py — that is the ONLY copy. A second copy here
# drifted (it kept dead alfworld/swebench branches and lacked a
# terminal_bench_2 branch, so TB2 would silently fall into the string-EM path
# and score 0). It was removed; import evaluate_task from latest_runner.


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

    # Reconstruct partial A/B/C from trace. Keep only each task's LAST traced
    # iteration: the paper's main-table methodology (and the normal report path)
    # report the final iteration of a chain — a partial report that averaged
    # every iteration would not be comparable with a completed one.
    scores_per_group = {}
    all_traces_per_group = {}
    for g, items in groups.items():
        last_by_task: dict = {}
        for it in items:
            tid = it.get("task_id", "?")
            cur = last_by_task.get(tid)
            if cur is None or int(it.get("iteration", 0) or 0) >= int(cur.get("iteration", 0) or 0):
                last_by_task[tid] = it
        finals = list(last_by_task.values())
        valid_scores = [it["score"] for it in finals if "score" in it]
        scores_per_group[g] = sum(valid_scores) / max(len(valid_scores), 1) if valid_scores else 0.0
        all_traces_per_group[g] = items

    return {
        "benchmark": benchmark,
        "scores": scores_per_group,
        "n_traces": len(lines),
        "reconstructed_from_trace": True,
    }