"""SkillForge V6 — AI-Driven Response Processor & Conversation History Manager.

Framework-level module for cleaning LLM responses in agentic loops.
Combines fast regex extraction for actions with AI-based evaluation
of non-action content (reasoning, plans, analysis) to determine what
should be preserved in conversation history.

Theoretical Grounding (SRDP — Attention Signal Density):
    In the SRDP framework, the conversation history fed back to the LLM at each
    turn is itself a "context window" subject to δ_att degradation. Noise tokens
    in the history dilute attention allocated to genuinely useful information
    (retrieved skills, tool results, task instructions). This module directly
    optimizes the **attention signal density** of the conversation context:

        signal_density = valuable_tokens / total_tokens_in_history

    By AI-filtering each turn's output before appending to history, we:
    1. Reduce δ_att(format_parsing): structured action lines are always preserved
       verbatim, preventing format ambiguity in subsequent turns.
    2. Reduce δ_att(retrieval_dilution): noise removal prevents irrelevant text
       from competing for attention with injected skills.
    3. Preserve δ_att(consistency): valuable reasoning (plans, state tracking)
       is retained to maintain coherent multi-turn decision-making.

    The net effect: higher signal density → LLM allocates more attention to
    injected skills and tool results → lower effective δ_att → tighter gap bound.

Key design principles:
1. Actions (NEXT_OP/PARAMS) are always extracted via regex (fast, deterministic)
2. Non-action content is evaluated by AI to distinguish:
   - Valuable reasoning (plans, analysis, error diagnosis) → KEEP in history
   - Noise (filler, repetition, self-talk) → DISCARD from history
3. Never discard information that could help the agent in future turns
4. The AI evaluator uses the same model (deepseek-v4-pro) for consistency

Usage:
    from v6.response_filter import AIResponseProcessor

    processor = AIResponseProcessor(
        action_pattern=r'NEXT_OP:\\s*(?P<action>op-\\d{3})',
        params_pattern=r'PARAMS:\\s*(?P<params>.*?)(?:\\n|$)',
        llm_fn=my_llm_call,  # async fn(prompt) -> str
    )

    # In your agent loop:
    result = await processor.process_response(raw_llm_response, task_context)
    if result.valid:
        # Execute tool...
        processor.append_result(tool_id, result_str)
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional


# ─── AI Evaluation Prompt ─────────────────────────────────────────────────

RESPONSE_EVAL_PROMPT = """You are a response quality evaluator for an AI agent's output.
The agent is solving a multi-step task by calling tools. Each turn, it outputs text that may contain:
- Valid operations (NEXT_OP/PARAMS lines) — these are always kept
- Reasoning/analysis that helps future decisions — KEEP these
- Noise/filler that adds no value — DISCARD these

Your job: Given the agent's non-action text, determine what is VALUABLE REASONING vs NOISE.

## VALUABLE (keep in history):
- Error analysis: "The search returned empty, I need to try a different keyword"
- Planning: "I need to first cancel the event, then create a new one"
- State tracking: "Found 3 contacts matching, the Film Producer is Åke"
- Conditional logic: "If the friend can't make it, I'll need to reschedule"
- Data extraction: "From the results, Saturday is Oct 19 and there's one appointment"

## NOISE (discard from history):
- Self-narration: "Let me think about this step by step"
- Filler: "I'll now proceed to the next step"
- Repetition of the task description
- Hedging: "I'm not sure but maybe..."
- Meta-commentary about the format or process itself
- Restating what was already said without adding new info

## Input
Task context: {task_context}
Agent's non-action text:
---
{non_action_text}
---

## Output (JSON only)
{{
  "valuable_parts": "The extracted valuable reasoning/analysis to keep (empty string if all noise)",
  "is_all_noise": true/false,
  "confidence": 0.0-1.0
}}"""


@dataclass
class ProcessedResponse:
    """Result of processing a raw LLM response."""

    valid: bool = False
    action_id: str = ""
    params_raw: str = ""
    params_parsed: dict[str, Any] = field(default_factory=dict)
    is_completion: bool = False
    raw_response: str = ""
    clean_action: str = ""  # Only the action lines (NEXT_OP + PARAMS)
    valuable_reasoning: str = ""  # AI-evaluated valuable non-action content
    error_type: str = ""  # "no_action" | "invalid_id" | ""
    ai_filtered: bool = False  # Whether AI evaluation was applied


class AIResponseProcessor:
    """AI-driven response processor for agentic loops.

    Responsibilities:
    1. Extract valid actions from LLM output (regex — fast, deterministic)
    2. AI-evaluate non-action content to identify valuable reasoning
    3. Maintain clean conversation history (actions + valuable reasoning + results)
    4. Generate format-retry prompts when LLM output is malformed
    5. Track retry state and enforce retry limits
    6. Detect and break action loops (reduces wasted turns → lower δ_sem)

    SRDP Theory Connection:
        This processor is the runtime mechanism for controlling δ_att in the
        agent's multi-turn execution. Each turn's conversation history is the
        "context" in p_LLM(a|s, c) — if it's polluted with noise, the LLM's
        attention to injected skills (c) degrades. By maintaining a high
        signal-to-noise ratio in history, we ensure:

        - Injected skills remain in attention-peak positions (head of context)
        - Tool results (ground truth) get full attention weight
        - Agent's own valuable reasoning is preserved for coherent planning
        - Noise is removed before it accumulates and triggers Lost-in-the-Middle

        The loop detection mechanism additionally prevents δ_sem degradation:
        when the agent repeatedly calls the same tool without progress, it's
        burning turns that could be used for productive exploration.
    """

    def __init__(
        self,
        action_pattern: str = r'NEXT_OP:\s*(?P<action>op-\d{3})',
        params_pattern: str = r'PARAMS:\s*(?P<params>.*?)(?:\n|$)',
        completion_signals: list[str] | None = None,
        max_retries: int = 3,
        valid_action_ids: set[str] | None = None,
        llm_fn: Optional[Callable[[str], Awaitable[str]]] = None,
        enable_ai_filter: bool = True,
        min_text_length_for_ai: int = 50,
    ):
        """Initialize the processor.

        Args:
            action_pattern: Regex with named group 'action' to extract the action ID.
            params_pattern: Regex with named group 'params' to extract parameters.
            completion_signals: Strings that signal the agent is done.
            max_retries: Max consecutive format retries before giving up.
            valid_action_ids: If provided, validate extracted action IDs against this set.
            llm_fn: Async function that calls LLM for AI evaluation. Signature: async (prompt) -> str
            enable_ai_filter: Whether to use AI evaluation (True) or skip (False).
            min_text_length_for_ai: Minimum non-action text length to trigger AI eval.
        """
        self._action_re = re.compile(action_pattern)
        self._params_re = re.compile(params_pattern)
        self._completion_signals = completion_signals or ["ALL_DONE"]
        self._max_retries = max_retries
        self._valid_ids = valid_action_ids
        self._llm_fn = llm_fn
        self._enable_ai_filter = enable_ai_filter
        self._min_text_for_ai = min_text_length_for_ai

        # State
        self._history_parts: list[str] = []
        self._consecutive_failures: int = 0
        self._turn_count: int = 0
        self._task_context: str = ""

        # Loop detection: track recent (action_id, params_raw) tuples
        self._recent_actions: list[tuple[str, str]] = []
        self._loop_threshold: int = 3  # Trigger after 3 identical calls
        self._same_tool_threshold: int = 4  # Trigger after 4 calls to same tool (even with different params)
        self._search_tool_ids: set[str] = set()  # Tool IDs that are "search" operations

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def should_give_up(self) -> bool:
        return self._consecutive_failures > self._max_retries

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def set_valid_ids(self, ids: set[str]) -> None:
        self._valid_ids = ids

    def set_task_context(self, context: str) -> None:
        """Set task context for AI evaluation (helps judge relevance)."""
        self._task_context = context[:500]

    def reset(self) -> None:
        self._history_parts = []
        self._consecutive_failures = 0
        self._turn_count = 0
        self._task_context = ""
        self._recent_actions = []

    @property
    def is_looping(self) -> bool:
        """Check if agent is stuck in a loop.

        Detects two patterns:
        1. Exact repetition: same (action_id, params) N times in a row
        2. Same-tool spinning: same action_id called M times in last M+2 turns
           (even with different params — e.g. searching contacts with different queries
           but never finding what's needed)
        """
        if len(self._recent_actions) < self._loop_threshold:
            return False

        # Pattern 1: Exact repetition (3 identical calls)
        last_n = self._recent_actions[-self._loop_threshold:]
        if len(set(last_n)) == 1:
            return True

        # Pattern 2: Same tool spinning (6 calls to same tool in last 8 turns)
        if len(self._recent_actions) >= self._same_tool_threshold:
            last_m = self._recent_actions[-self._same_tool_threshold:]
            tool_ids = [a[0] for a in last_m]
            # If >80% of recent calls are to the same tool, it's spinning
            from collections import Counter
            most_common_tool, count = Counter(tool_ids).most_common(1)[0]
            if count >= self._same_tool_threshold - 1:  # 5 out of 6
                return True

        return False

    def get_loop_break_prompt(self) -> str:
        """Generate a prompt to break the agent out of a detected loop.

        Key insight: DO NOT suggest more searching/browsing — that's what caused
        the loop in the first place. Instead, force the agent to SKIP the stuck
        sub-task and proceed with the rest of the task using available info.
        """
        if not self._recent_actions:
            return ""
        last_action, last_params = self._recent_actions[-1]

        # Analyze what kind of loop it is
        last_m = self._recent_actions[-min(self._same_tool_threshold, len(self._recent_actions)):]
        tool_ids = [a[0] for a in last_m]
        from collections import Counter
        most_common_tool, count = Counter(tool_ids).most_common(1)[0]

        if count >= self._same_tool_threshold - 1:
            # Same-tool spinning — agent is stuck searching
            prompt = (
                f"\n\n⚠️ BUDGET EXCEEDED: You called {most_common_tool} {count} times. "
                "STOP SEARCHING IMMEDIATELY.\n"
                "MANDATORY ACTION — pick ONE:\n"
                "  A) If you found ANY partial info (a name, email, etc.), USE IT NOW to proceed.\n"
                "  B) If you found NOTHING, SKIP this sub-task entirely and move to the next step.\n"
                "  C) Try ONE different short keyword (single word only), then proceed regardless.\n"
                "DO NOT browse/paginate/list contacts anymore. Move forward NOW.\n\n"
                "Your next operation (NEXT_OP + PARAMS only):"
            )
        else:
            # Exact repetition
            prompt = (
                f"\n\n⚠️ LOOP DETECTED: {last_action} called {self._loop_threshold}x with same params. "
                "STOP. Try a COMPLETELY different operation or skip this step.\n\n"
                "Your next operation (NEXT_OP + PARAMS only):"
            )
        # Clear loop history to give agent a fresh start
        self._recent_actions = []
        return prompt

    def set_initial_prompt(self, prompt: str) -> None:
        self._history_parts = [prompt]

    def get_conversation_history(self) -> str:
        return "\n".join(self._history_parts)

    def _extract_non_action_text(self, raw: str) -> str:
        """Extract text that is NOT part of NEXT_OP/PARAMS lines."""
        lines = raw.split("\n")
        non_action_lines = []
        skip_next = False
        for line in lines:
            stripped = line.strip()
            if re.match(r'NEXT_OP:', stripped) or re.match(r'PARAMS:', stripped):
                skip_next = False
                continue
            if skip_next:
                continue
            if stripped:
                non_action_lines.append(stripped)
        return "\n".join(non_action_lines).strip()

    async def _ai_evaluate_content(self, non_action_text: str) -> str:
        """Use AI to evaluate which parts of non-action text are valuable.

        Returns the valuable reasoning to keep, or empty string if all noise.
        """
        if not self._llm_fn:
            # No LLM available — keep everything (safe default)
            return non_action_text

        if not non_action_text or len(non_action_text) < self._min_text_for_ai:
            # Too short to bother with AI — keep it all
            return non_action_text

        prompt = RESPONSE_EVAL_PROMPT.format(
            task_context=self._task_context or "(not provided)",
            non_action_text=non_action_text[:1500],  # Cap input to avoid cost
        )

        try:
            response = await self._llm_fn(prompt)
            # Parse JSON response
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if data.get("is_all_noise", False) and data.get("confidence", 0) > 0.7:
                    return ""  # Discard — AI is confident it's all noise
                valuable = data.get("valuable_parts", "")
                if valuable:
                    return valuable
            # Fallback: if AI response is unparseable, keep original (safe)
            return non_action_text
        except Exception:
            # AI call failed — keep everything (never lose info on error)
            return non_action_text

    async def process_response(self, raw_response: str) -> ProcessedResponse:
        """Process a raw LLM response: extract action + AI-evaluate non-action content.

        This is the core method — it:
        1. Checks for completion signals
        2. Extracts action ID and params via regex
        3. AI-evaluates non-action text to find valuable reasoning
        4. Returns ProcessedResponse with both action and valuable content

        Args:
            raw_response: The raw text output from the LLM.

        Returns:
            ProcessedResponse with parsed action and AI-filtered reasoning.
        """
        self._turn_count += 1

        # Strip known noise patterns
        cleaned = re.sub(r'Tool \S+ not found in agent cli\.', '', raw_response).strip()

        # Check completion signals
        for signal in self._completion_signals:
            if signal in cleaned:
                self._consecutive_failures = 0
                return ProcessedResponse(
                    valid=True,
                    is_completion=True,
                    raw_response=raw_response,
                    clean_action=signal,
                )

        # Extract action ID
        action_match = self._action_re.search(cleaned)
        if not action_match:
            self._consecutive_failures += 1
            return ProcessedResponse(
                valid=False,
                error_type="no_action",
                raw_response=raw_response,
            )

        action_id = action_match.group("action")

        # Validate action ID
        if self._valid_ids and action_id not in self._valid_ids:
            self._consecutive_failures += 1
            return ProcessedResponse(
                valid=False,
                action_id=action_id,
                error_type="invalid_id",
                raw_response=raw_response,
                clean_action=f"NEXT_OP: {action_id}",
            )

        # Extract params
        params_raw = ""
        params_parsed: dict[str, Any] = {}
        params_match = self._params_re.search(cleaned)
        if params_match:
            params_raw = params_match.group("params").strip()
            if params_raw and params_raw != "(none)":
                params_parsed = self._parse_params(params_raw)

        # Build clean action string
        clean_lines = [f"NEXT_OP: {action_id}"]
        if params_raw:
            clean_lines.append(f"PARAMS: {params_raw}")
        clean_action = "\n".join(clean_lines)

        # AI-evaluate non-action content
        valuable_reasoning = ""
        ai_filtered = False
        if self._enable_ai_filter:
            non_action = self._extract_non_action_text(cleaned)
            if non_action:
                valuable_reasoning = await self._ai_evaluate_content(non_action)
                ai_filtered = True

        self._consecutive_failures = 0

        # Track action for loop detection
        self._recent_actions.append((action_id, params_raw))

        return ProcessedResponse(
            valid=True,
            action_id=action_id,
            params_raw=params_raw,
            params_parsed=params_parsed,
            raw_response=raw_response,
            clean_action=clean_action,
            valuable_reasoning=valuable_reasoning,
            ai_filtered=ai_filtered,
        )

    def _parse_params(self, params_str: str) -> dict[str, Any]:
        """Parse parameter string into a dict. Supports pipe-separated key:value format.

        Handles automatic type conversion:
        - JSON arrays/objects: ["a","b"] → list, {"k":"v"} → dict
        - Integers: "0", "10", "-5" → int
        - Floats: "3.14", "-0.5" → float
        - Booleans: "true"/"false" → bool
        - Everything else: remains str
        """
        result: dict[str, Any] = {}
        for part in params_str.split("|"):
            part = part.strip()
            if ":" in part:
                key, val = part.split(":", 1)
                key = key.strip()
                val = val.strip()
                # Try JSON arrays/objects first
                if val.startswith("[") or val.startswith("{"):
                    try:
                        val = json.loads(val)
                    except (ValueError, json.JSONDecodeError):
                        pass
                # Try integer conversion
                elif val.lstrip("-").isdigit():
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                # Try float conversion
                elif self._is_float(val):
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                # Boolean conversion
                elif val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                result[key] = val
        return result

    @staticmethod
    def _is_float(s: str) -> bool:
        """Check if a string represents a float number."""
        try:
            float(s)
            return "." in s or "e" in s.lower()
        except ValueError:
            return False

    def build_history_entry(self, processed: ProcessedResponse, result_str: str,
                            notifications: list[str] | None = None) -> str:
        """Build a conversation history entry from a processed response + tool result.

        Includes:
        - Valuable reasoning (if AI found any)
        - The action that was taken
        - The tool result
        - Any notifications

        This is what gets appended to conversation_history.
        """
        parts = []

        # Include valuable reasoning if present (AI-approved context)
        if processed.valuable_reasoning:
            parts.append(f"[Agent reasoning: {processed.valuable_reasoning}]")

        # Always include the result
        parts.append(f"\n\nRESULT ({processed.action_id}):\n{result_str}")

        if notifications:
            for n in notifications:
                parts.append(f"NOTIFICATION: {n}")

        parts.append("\n\nNext (NEXT_OP + PARAMS only):")
        return "".join(parts)

    def append_to_history(self, entry: str) -> None:
        """Append a pre-built entry to conversation history."""
        self._history_parts.append(entry)

    def append_error(self, error_msg: str) -> None:
        """Append an error message to conversation history."""
        self._history_parts.append(
            f"\n\n{error_msg}\n\nYour next operation (output ONLY NEXT_OP and PARAMS):"
        )

    def get_retry_prompt(self, failed: ProcessedResponse) -> str:
        """Generate a format-retry prompt based on the failure type."""
        if failed.error_type == "no_action":
            prompt = (
                "\n\nFORMAT ERROR. Output EXACTLY:\n"
                "NEXT_OP: op-XXX\n"
                "PARAMS: key:value\n\n"
                "Nothing else. Go:"
            )
        elif failed.error_type == "invalid_id":
            prompt = (
                f"\n\nERROR: Invalid ID '{failed.action_id}'. "
                "Check OPERATIONS list.\n\n"
                "Your next operation (output ONLY NEXT_OP and PARAMS):"
            )
        else:
            prompt = "\n\nPlease output your next operation (NEXT_OP + PARAMS only):"

        self._history_parts.append(prompt)
        return prompt

    def get_history_with_truncation(self, max_chars: int = 80000) -> str:
        """Get conversation history with smart truncation.

        Preserves the initial task prompt and the most recent interactions.

        SRDP Theory: This implements an attention budget constraint. The initial
        prompt (containing injected skills) is always preserved at the HEAD of
        context — the highest-attention position per Lost-in-the-Middle findings.
        Recent interactions are preserved at the TAIL (second-highest attention).
        Middle content (older turns) is truncated first — this is exactly where
        attention is weakest, so information loss is minimized.
        """
        full = "\n".join(self._history_parts)
        if len(full) <= max_chars:
            return full

        if not self._history_parts:
            return ""

        header = self._history_parts[0]
        header_budget = min(len(header), max_chars // 4)
        remaining_budget = max_chars - header_budget

        recent_parts = []
        current_len = 0
        for part in reversed(self._history_parts[1:]):
            if current_len + len(part) > remaining_budget:
                break
            recent_parts.insert(0, part)
            current_len += len(part)

        truncation_notice = "\n\n[... earlier interactions truncated ...]\n"
        return header[:header_budget] + truncation_notice + "\n".join(recent_parts)


# ─── Synchronous wrapper for non-async contexts ──────────────────────────

class SyncResponseProcessor:
    """Synchronous wrapper around AIResponseProcessor for non-async agent loops.

    For agent loops that don't use async, this provides a sync interface
    that skips AI evaluation (uses heuristic fallback instead).
    """

    def __init__(self, **kwargs):
        # Force disable AI filter in sync mode (can't await)
        kwargs["enable_ai_filter"] = False
        kwargs["llm_fn"] = None
        self._inner = AIResponseProcessor(**kwargs)

    def process_response(self, raw_response: str) -> ProcessedResponse:
        """Synchronous process — extracts action only, no AI eval."""
        import asyncio
        # Since AI is disabled, the async method won't actually await anything
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._inner.process_response(raw_response))
        finally:
            loop.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)