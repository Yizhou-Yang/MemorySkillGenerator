"""WebShop trace-based environment wrapper.

Wraps the public `Skyler215/webshop-agent-cot` HuggingFace dataset as a
single-step action-prediction benchmark. This is the "static-trace" proxy
for the full WebShop simulator (princeton-nlp/webshop, ~10 GB ScaNN index)
which is deferred to camera-ready.

Each sample is a (state, gold_action) pair extracted from a successful
agent trajectory:
    state.prompt        — instruction + history + observation + valid_actions
    state.valid_actions — closed list parsed from prompt
    gold_action         — the action the trace agent took at this step
                          (e.g. "click[buy now]", "search[blue hoodie]")

Usage:
    env = WebShopTraceEnv(split="test")
    print(env.num_tasks)              # 2225 single-step decisions
    sample = env.get(idx=0)
    print(sample["instruction"])
    print(sample["valid_actions"])    # ["click[large]", "click[buy now]", ...]
    print(sample["gold_action"])      # "click[buy now]"

    is_match = env.score(prediction="click[buy now]", gold="click[buy now]")

Categories (used for stratified sampling, mirrors ALFWorld task_type):
    - buy        : gold is click[buy now]                   (~10% of samples)
    - search     : gold is search[...]                      (~15%)
    - select     : gold is click[<color/size/option>]       (~10%)
    - navigate   : click[< prev], click[next >], click[back to search]  (~30%)
    - inspect    : click[features], click[description], click[reviews]  (~20%)
    - other      : everything else                          (~15%)
"""

from __future__ import annotations

import re
from typing import Any
from collections import Counter


_ACTION_RE = re.compile(r"(click|search)\[([^\]]+)\]", re.IGNORECASE)
_VALID_ACTIONS_BLOCK_RE = re.compile(
    r"Valid actions:\s*\n((?:- .+\n?)+)", re.MULTILINE
)
_VALID_ACTION_LINE_RE = re.compile(r"^- (.+)$", re.MULTILINE)
_INSTRUCTION_RE = re.compile(r"Instruction:\s*\n(.+?)\n", re.DOTALL)
_OBSERVATION_RE = re.compile(r"Observation:\s*\n(.+?)\n\nValid actions:", re.DOTALL)


def extract_action(text: str) -> str | None:
    """Pull the first 'click[...]' or 'search[...]' token from a string."""
    m = _ACTION_RE.search(text)
    if not m:
        return None
    verb = m.group(1).lower()
    target = m.group(2).strip()
    return f"{verb}[{target}]"


def parse_valid_actions(prompt: str) -> list[str]:
    """Extract the closed list of valid actions from a webshop prompt.

    The prompt always ends with a 'Valid actions:' block of '- click[X]' lines.
    Some prompts truncate the list; we accept whatever lines are present.
    """
    m = _VALID_ACTIONS_BLOCK_RE.search(prompt)
    if not m:
        # Fallback: scan from "Valid actions:" to EOF for "- ..." lines
        idx = prompt.rfind("Valid actions:")
        block = prompt[idx:] if idx >= 0 else prompt
    else:
        block = m.group(1)
    actions = []
    for line in _VALID_ACTION_LINE_RE.findall(block):
        line = line.strip()
        # The dataset sometimes wraps long action lines; tolerate that.
        if line:
            actions.append(line)
    return actions


def parse_instruction(prompt: str) -> str:
    m = _INSTRUCTION_RE.search(prompt)
    if m:
        return m.group(1).strip()
    return ""


def parse_observation(prompt: str) -> str:
    m = _OBSERVATION_RE.search(prompt)
    if m:
        return m.group(1).strip()
    return ""


def task_type_from_action(gold_action: str) -> str:
    """Classify a (gold_action) into one of 6 categories for stratification."""
    if not gold_action:
        return "other"
    a = gold_action.lower()
    if a.startswith("search["):
        return "search"
    if "buy now" in a:
        return "buy"
    if any(t in a for t in ["< prev", "prev", "next >", "back to search"]):
        return "navigate"
    if any(t in a for t in ["features", "description", "reviews"]):
        return "inspect"
    if a.startswith("click[") and a not in (
        "click[< prev]", "click[next >]", "click[back to search]"
    ):
        # Color/size/option pick — anything else inside click[]
        return "select"
    return "other"


class WebShopTraceEnv:
    """Trace-based WebShop wrapper. NOT a simulator — single-step replay only.

    Dataset: `Skyler215/webshop-agent-cot` (HuggingFace, 2225 test samples).
    Each sample is one decision point from a successful agent rollout.
    """

    def __init__(self, split: str = "test", num_samples: int | None = None):
        from datasets import load_dataset  # type: ignore

        self.split = split
        ds = load_dataset("Skyler215/webshop-agent-cot", split=split)
        if num_samples is not None:
            ds = ds.select(range(min(num_samples, len(ds))))

        self._records: list[dict[str, Any]] = []
        for i, row in enumerate(ds):
            prompt = row["prompt"]
            response = row["response"]
            row_id = row.get("id", str(i))
            gold = extract_action(response)
            if gold is None:
                # Skip malformed samples (no parseable action)
                continue
            valid_actions = parse_valid_actions(prompt)
            instruction = parse_instruction(prompt)
            observation = parse_observation(prompt)
            self._records.append({
                "id": row_id,
                "prompt": prompt,
                "instruction": instruction,
                "observation": observation,
                "valid_actions": valid_actions,
                "gold_action": gold,
                "task_type": task_type_from_action(gold),
                "raw_response": response,
            })

        self.num_tasks = len(self._records)

    # ----------------------------------------------------------------- API

    def list_tasks(self) -> list[dict[str, Any]]:
        return list(self._records)

    def get(self, idx: int) -> dict[str, Any]:
        return self._records[idx]

    def task_type(self, idx: int) -> str:
        return self._records[idx]["task_type"]

    def task_type_distribution(self) -> dict[str, int]:
        return dict(Counter(r["task_type"] for r in self._records))

    @staticmethod
    def score(prediction: str | None, gold: str) -> bool:
        """Strict action match (case-insensitive, whitespace-tolerant)."""
        if not prediction:
            return False
        p = prediction.strip().lower()
        g = gold.strip().lower()
        if p == g:
            return True
        # Try to normalise spacing inside brackets
        def _norm(s: str) -> str:
            m = _ACTION_RE.search(s)
            if not m:
                return s.lower().strip()
            return f"{m.group(1).lower()}[{m.group(2).strip().lower()}]"
        return _norm(prediction) == _norm(gold)

    @staticmethod
    def snap_to_valid(prediction: str, valid_actions: list[str]) -> str:
        """Map an LLM reply onto the closed valid_actions set.

        Strategy:
            1. Extract click[..]/search[..] token from reply.
            2. Exact match (case-insensitive) → return canonical form.
            3. Substring match (longest valid_action contained in token).
            4. Otherwise return raw extracted token (will likely score 0).
        """
        if not valid_actions:
            return prediction.strip()
        tok = extract_action(prediction)
        if tok is None:
            tok = prediction.strip().splitlines()[-1].strip()
        # Exact / case-insensitive
        for va in valid_actions:
            if va.lower() == tok.lower():
                return va
        # Substring containment
        tl = tok.lower()
        candidates = [va for va in valid_actions if va.lower() in tl or tl in va.lower()]
        if candidates:
            candidates.sort(key=len, reverse=True)
            return candidates[0]
        return tok
