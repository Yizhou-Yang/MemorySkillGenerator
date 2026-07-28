#!/usr/bin/env python3
"""SkillForge Latest — Latest Experiment Runner"""
import asyncio
import copy
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ.setdefault('CODEBUDDY_MODEL', 'hy3-preview')
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')

try:
    from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage, ToolUseBlock
except Exception:  # reviewer path (no CodeBuddy CLI) → OpenAI-compatible backend
    query = CodeBuddyAgentOptions = AssistantMessage = ToolUseBlock = None

from latest import (SkillForgeLatest, ExperienceLibrary, Experience,
                build_augmented_prompt, ai_review_experience,
                cross_agent_evaluate_skill)
from latest.safety import CompletionGate, BudgetTracker, NoRepeatGuard
from latest.llm.prompts import build_agent_system_prompt, detect_twist, get_twist_reminder
from latest.injection import format_evoarena_patch_log, format_skillforge_patch_log

# ─── Extracted modules ────────────────────────────────────────────────────
from scripts.latest.trace import TraceLogger, APIUnavailableError
from scripts.latest.llm_client import (
    probe_api_available, _check_api_error, _API_FAILURE_THRESHOLD,
    save_checkpoint, load_checkpoint, clear_checkpoint,
    llm_review_fn,
    _query_sync, _llm_call, _query_notool_sync, _llm_call_notool,
    _llm_short_call, llm_extract_answer, llm_judge_answer,
    llm_critic_skill_quality,
)
from scripts.latest.eval import (
    normalize_answer, exact_match,
    compute_partial_results_from_trace,
)

MODEL = os.environ.get("CODEBUDDY_MODEL", "hy3-preview")  # NEVER hardcode: a constant here silently ran a deepseek sweep on HY3
CONCURRENCY = 5
# Per-task ceilings. A reasoning backbone at high effort spends far longer per
# turn, and a ceiling tuned for a fast model turns into a stream of timeouts that
# the client reports as "API unavailable" -- which then trips the runner's
# consecutive-failure abort, so the benchmark looks like an outage instead of a
# too-short clock. Override per backbone rather than lowering reasoning effort,
# which would change the model being measured.
TASK_TIMEOUT_QA = int(os.environ.get("TASK_TIMEOUT_QA", "180"))
TASK_TIMEOUT_AGENT = int(os.environ.get("TASK_TIMEOUT_AGENT", "300"))
QUALITY_THRESHOLD = 5

RESULTS_DIR = str(PROJECT_ROOT / "experiments_results" / "latest")

# ─── Group B/C system-prompt augmentations (GAIA, GAIA2) ──────────────────
#  A = no augmentation, B = EVOARENA_AUGMENTATION, C = SKILLFORGE_AUGMENTATION.

EVOARENA_AUGMENTATION = """
## Self-Correction & Verification Protocol

Apply these metacognitive principles during your multi-step reasoning:

1. **Continuous Self-Monitoring**: Every 3-4 turns, pause and verify whether your
   current direction is correct. If you detect a contradiction or error in your
   previous reasoning, explicitly say "I need to correct my previous conclusion"
   and explain the correction before proceeding.

2. **Cross-Verification**: Never trust a single source. When you find a factual
   claim, verify it with at least one independent source before accepting it.
   If sources disagree, prefer official/primary sources over secondary ones.

3. **Adaptive Search Strategy**: If a search returns no useful results, immediately
   try a DIFFERENT keyword strategy (broader term, synonym, related concept)
   rather than paginating through empty results or rephrasing the same query.

4. **Budget Awareness**: You have limited turns. If you spend 5+ turns on one
   sub-problem without progress, explicitly note it and move to a different
   approach or sub-problem. You can revisit later if time permits.

5. **Error Recovery**: When you realize you have been pursuing a wrong lead:
   (a) State what was wrong and why,
   (b) Identify the correct direction,
   (c) Pursue the new direction immediately without dwelling on the error.

6. **Final Verification**: Before submitting your answer, verify each component
   of your reasoning chain is correct. If the answer is numerical, double-check
   with Python computation.
"""

SKILLFORGE_AUGMENTATION = EVOARENA_AUGMENTATION + """
## Precision Refinement

Beyond self-correction, apply these refinements:

1. **Failure Diagnosis**: When a search or computation fails, briefly state WHY
   (wrong assumption? wrong tool? wrong query scope?) before attempting again.
   This converts reactive correction into proactive prevention.

2. **Answer Format Calibration**: Before final submission, strip ALL explanatory
   text and output ONLY the exact answer in the requested format. If the question
   asks for "a name", output just the name — no sentences, no commentary.
"""

TASK_LIMITS = {"gaia": 165, "gaia2": 50, "swebench_dynamic": 30}

CHECKPOINT_FILE = str(PROJECT_ROOT / "experiments_results" / "latest" / "_checkpoint.json")

# ─── Trace logger ─────────────────────────────────────────────────────────
_trace = TraceLogger(RESULTS_DIR)

# ─── GAIA runner ──────────────────────────────────────────────────────────

async def run_gaia_task(task: dict, experience_section: str = "",
                        group: str = "A") -> dict:
    task_id = task["task_id"]
    description = task["description"]
    expected = task.get("expected", "")
    metadata = task.get("metadata", {})
    benchmark_type = metadata.get("benchmark", "")

    if benchmark_type == "swebench_dynamic" or "swebench" in task_id:
        system = (
            "You are an expert software engineer debugging a failing test case.\n\n"
            "Strategy:\n"
            "1. UNDERSTAND: Read the issue description carefully. What behavior is expected vs actual?\n"
            "2. LOCATE: Identify which file(s) and function(s) are likely responsible.\n"
            "3. DIAGNOSE: Determine the root cause of the bug.\n"
            "4. FIX: Write a minimal, correct patch that fixes the issue.\n"
            "5. VERIFY: Explain why your fix resolves the failing test.\n\n"
            "Important:\n"
            "- Focus on the MINIMAL change needed — don't refactor unrelated code.\n"
            "- Your response MUST include the actual code fix (as a diff or code block).\n"
            "- Show the file path, the original code, and your corrected code.\n"
            "- If you need to read files or run tests, use the available tools."
        )
    else:
        system = (
            "You are an expert research assistant capable of multi-step reasoning and tool use. "
            "You have access to web search, file reading, code execution, and computation tools.\n\n"
            "Strategy for answering questions:\n"
            "1. ANALYZE: Break down the question — what information do you need?\n"
            "2. SEARCH: Use web search to find relevant facts, data, or context.\n"
            "3. VERIFY: Cross-check information from multiple sources when possible.\n"
            "4. COMPUTE: If math/logic is needed, use code execution for accuracy.\n"
            "5. ANSWER: Provide a concise, precise final answer.\n\n"
            "Important rules:\n"
            "- Always search the web for factual questions — do NOT guess from memory.\n"
            "- For numerical answers, show your computation steps.\n"
            "- If a question asks for a specific format (name, number, date), match that format exactly.\n"
            "- Give ONLY the final answer in your last message — no explanation needed."
        )

    if experience_section:
        system += f"\n\n{experience_section}"

    prompt = (
        f"{system}\n\n"
        f"Question: {description}\n\n"
        f"Think step by step. Use tools as needed. "
        f"End with your final answer on the last line."
    )

    result = {"task_id": task_id, "expected": expected, "response": "",
              "error": None, "time_cost": 0, "augmented": bool(experience_section),
              "group": group, "actions": []}
    t0 = time.time()
    r = await _llm_call(prompt, max_turns=int(os.environ.get("GAIA_MAX_TURNS", "20")), timeout=TASK_TIMEOUT_AGENT)
    if _check_api_error(r):
        raise APIUnavailableError(f"API unavailable after {_API_FAILURE_THRESHOLD} consecutive failures")
    result["response"] = r.get("text", "")
    result["actions"] = r.get("actions", [])
    result["error"] = r.get("error")
    result["time_cost"] = time.time() - t0
    return result


# ─── SelfCorrectionDetector — within-task patch tracking ──────────────────
#  is_error_patch=True (wrong conclusion) vs False (incomplete) selects the
#  injection format downstream: avoidance_note vs procedural template.

import re as _re_sc

SELF_CORRECTION_PATTERNS = [
    r"(?i)(?:wait|hold on|hang on|hmm|oh)\\s*,?\\s*(?:I need to|let me|I should|I'll)\\s*(?:reconsider|rethink|re-evaluate|correct|revise|go back|backtrack)",
    r"(?i)(?:actually|in fact|on second thought|come to think of it|I was wrong|I made a mistake|that's (?:wrong|incorrect|not right))",
    r"(?i)(?:let me|I'll|I should|I need to|I have to)\\s*(?:correct|fix|amend|revise|change|update)\\s*(?:that|this|my|the)",
    r"(?i)(?:that doesn't work|that won't work|this approach fails|this isn't working|that's not going to work)\\s*(?:because|since|as|due to)",
    r"(?i)(?:based on (?:the |this )?new|after re-?checking|upon re-?examination|looking (?:back )?at (?:the |this )?again)",
]

PRIOR_CONCLUSION_PATTERNS = [
    r"(?i)(?:I (?:thought|assumed|believed|expected|figured|was thinking))\\s+(?:that\\s+)?(.+?)(?:\\.|but|however|actually)",
    r"(?i)(?:my (?:initial|previous|earlier|original))\\s+(?:conclusion|assumption|thought|answer|result|finding)\\s+(?:was|is|would be|should be)\\s+(.+?)(?:\\.|but|however)",
    r"(?i)(?:previously|initially|at first|earlier)\\s*(?:,?\\s*I\\s+)?(?:concluded|determined|found|decided|thought|assumed)\\s+(?:that\\s+)?(.+?)(?:\\.|but|however)",
]


class SelfCorrectionDetector:
    """Detects self-correction moments in agent multi-turn reasoning."""

    def __init__(self):
        self._correction_patterns = [_re_sc.compile(p) for p in SELF_CORRECTION_PATTERNS]
        self._prior_patterns = [_re_sc.compile(p) for p in PRIOR_CONCLUSION_PATTERNS]
        self._patch_id = 0

    def detect_correction(self, response_text: str, turn: int) -> dict | None:
        """Detect if this turn contains a self-correction."""
        is_correction = False
        correction_match = None
        for pat in self._correction_patterns:
            m = pat.search(response_text)
            if m:
                is_correction = True
                correction_match = m
                break
        if not is_correction:
            return None

        prior_conclusion = ""
        for pat in self._prior_patterns:
            m = pat.search(response_text)
            if m:
                prior_conclusion = m.group(1).strip()[:300]
                break

        is_error = any(kw in response_text.lower() for kw in [
            "wrong", "incorrect", "mistake", "error", "not right"
        ])

        self._patch_id += 1
        correction_start = max(0, correction_match.start() - 100) if correction_match else 0
        correction_end = min(len(response_text), correction_match.end() + 400) if correction_match else len(response_text)
        correction_snippet = response_text[correction_start:correction_end].strip()

        post_correction_text = response_text[correction_match.end():].strip() if correction_match else ""
        revised_conclusion = post_correction_text[:300].split("\n\n")[0].strip()

        rationale_start = 0
        if correction_match:
            rationale = correction_snippet[100:400].strip()
        else:
            rationale = "self-detected error" if is_error else "self-detected refinement"

        return {
            "patch_id": f"sc_{self._patch_id:04d}",
            "turn": turn,
            # duplicate keys (conclusion/revised_conclusion/revision_rationale)
            # are EvoMem's field names — the injection formatters read those.
            "revised_at_turn": turn,
            "prior_conclusion": prior_conclusion,
            "conclusion": prior_conclusion,
            "correction_snippet": correction_snippet[:500],
            "revised_conclusion": revised_conclusion,
            "revision_rationale": rationale[:300],
            "is_error_patch": is_error,
            "correction_type": "error" if is_error else "refinement",
            "timestamp": time.time(),
        }


# ─── GAIA Controlled Multi-Turn Runner (EvoMem-enabled) ───────────────────

async def run_gaia_task_controlled(task: dict, experience_section: str = "",
                                     group: str = "A",
                                     within_task_patch_mode: str | None = None) -> dict:
    """Run GAIA/SWE-bench with an explicit agent loop (vs run_gaia_task's SDK loop),
    so patches can be detected and injected between turns. within_task_patch_mode:
    None/"evoarena" = plain patch injection, "skillforge" = failure-aware routing.
    """
    from scripts.latest.tools import (
        ManualToolExecutor,
        GAIA_SYSTEM_PROMPT_TEMPLATE,
        SWE_SYSTEM_PROMPT_TEMPLATE,
    )

    task_id = task["task_id"]
    description = task["description"]
    expected = task.get("expected", "")
    metadata = task.get("metadata", {})
    benchmark_type = metadata.get("benchmark", "")
    is_swe = benchmark_type == "swebench_dynamic" or "swebench" in task_id

    result = {
        "task_id": task_id,
        "expected": expected,
        "response": "",
        "error": None,
        "time_cost": 0,
        "augmented": bool(experience_section),
        "group": group,
        "actions": [],
        "event_log": [],
    }
    t0 = time.time()

    executor = ManualToolExecutor(working_dir=f"/tmp/skillforge_gaia/{task_id}")
    tool_text = executor.get_tool_list_text()

    if is_swe:
        system_prompt = SWE_SYSTEM_PROMPT_TEMPLATE + f"\nOPERATIONS:\n{tool_text}\n\nSTART NOW. Output: REASONING + NEXT_OP + PARAMS."
    else:
        system_prompt = GAIA_SYSTEM_PROMPT_TEMPLATE + f"\nOPERATIONS:\n{tool_text}\n\nSTART NOW. Output: REASONING + NEXT_OP + PARAMS."

    if experience_section:
        system_prompt += f"\n\nRELEVANT EXPERIENCE:\n{experience_section}"

    conversation_history = (
        f"TASK: {description}\n\n"
        "Output your first operation now (REASONING + NEXT_OP + PARAMS):"
    )

    detector = SelfCorrectionDetector()
    processor = AIResponseProcessor(
        action_pattern=r'NEXT_OP:\s*(?P<action>op-\d{3})',
        params_pattern=r'PARAMS:\s*(?P<params>.*?)(?:\n|$)',
        completion_signals=["op-000"],
        valid_action_ids={"op-000", "op-001", "op-002", "op-003", "op-004", "op-005"},
        enable_ai_filter=False,
        max_retries=3,
    )
    nr_guard = NoRepeatGuard()
    _gaia_turns = int(os.environ.get("GAIA_MAX_TURNS", "20"))
    budget = BudgetTracker(max_turns=_gaia_turns)

    max_turns = _gaia_turns
    all_responses = []
    within_task_patches: list[dict] = []
    _last_answer: str | None = None

    try:
        for turn in range(max_turns):
            patch_injection = ""
            if within_task_patch_mode and within_task_patches:
                if within_task_patch_mode == "skillforge":
                    patch_injection = format_skillforge_patch_log(within_task_patches)
                else:
                    patch_injection = format_evoarena_patch_log(within_task_patches)

            full_history = conversation_history
            if patch_injection:
                full_history = conversation_history + "\n\n" + patch_injection

            r = await _llm_call_notool(system_prompt, full_history, timeout=240)

            if _check_api_error(r):
                raise APIUnavailableError(
                    f"API unavailable after {_API_FAILURE_THRESHOLD} consecutive failures"
                )

            response_text = r.get("text", "")
            all_responses.append(response_text)

            patch = detector.detect_correction(response_text, turn)
            if patch:
                within_task_patches.append(patch)

            processed = await processor.process_response(response_text)

            # An answer change across turns is a self-correction even without
            # explicit "I was wrong" wording.
            current_answer = None
            if processed.params_raw and "answer:" in processed.params_raw:
                params = processed.params_parsed or {}
                current_answer = params.get("answer") or processed.params_raw.split("answer:", 1)[-1].strip()

            if _last_answer is not None and current_answer is not None:
                if _last_answer != current_answer:
                    detector._patch_id += 1
                    implicit_patch = {
                        "patch_id": f"implicit_{detector._patch_id:04d}",
                        "turn": turn,
                        "revised_at_turn": turn,
                        "prior_conclusion": _last_answer[:300],
                        "conclusion": _last_answer[:300],
                        "correction_snippet": (
                            f"Agent revised answer from '{_last_answer}' "
                            f"to '{current_answer}'"
                        ),
                        "revised_conclusion": current_answer[:300],
                        "revision_rationale": (
                            "Implicit correction: Answer changed after further "
                            "tool use, indicating the agent found new evidence "
                            "that contradicted its initial conclusion."
                        ),
                        "is_error_patch": True,
                        "correction_type": "error",
                        "timestamp": time.time(),
                    }
                    within_task_patches.append(implicit_patch)
                    result["event_log"].append(
                        f"turn_{turn}: implicit_correction "
                        f"'{_last_answer[:50]}' -> '{current_answer[:50]}'"
                    )

            if current_answer is not None:
                _last_answer = current_answer

            # Force-research guard: without it 89% of GAIA tasks answered from
            # memory with zero tool use, so no patches were generated and A=B=C.
            if (processed.is_completion or processed.action_id == "op-000") and not result["actions"]:
                rejection = (
                    "\n\nERROR: You must search the web (op-001) or fetch evidence "
                    "(op-002) at least once before providing an answer. "
                    "Do NOT answer from memory — the evaluation depends on real "
                    "web search results. Use op-001 to search first."
                )
                conversation_history += rejection
                continue

            if processed.is_completion or processed.action_id == "op-000":
                params = processed.params_parsed or {}
                answer = params.get("answer", response_text)
                result["response"] = answer
                break

            if not processed.valid:
                if turn < max_turns - 1:
                    retry_prompt = (
                        "\n\nERROR: Invalid response format. "
                        "You MUST output exactly:\n"
                        "REASONING: <one sentence>\n"
                        "NEXT_OP: op-XXX\n"
                        "PARAMS: key:value\n\n"
                        "Available ops: op-001 (search), op-002 (fetch), "
                        "op-003 (read), op-004 (write), op-005 (exec), op-000 (finish)"
                    )
                    conversation_history += retry_prompt
                    continue
                break

            tool_id = processed.action_id
            tool_args = processed.params_parsed or {}

            if nr_guard.would_repeat(tool_id, tool_args):
                conversation_history += nr_guard.get_warning(tool_id)
                continue
            nr_guard.record(tool_id, tool_args)

            if tool_id == "op-000":
                result["response"] = tool_args.get("answer", response_text)
                break

            tool_result = executor.execute(tool_id, tool_args)
            result["actions"].append({
                "tool": tool_id,
                "args": tool_args,
                "result_preview": str(tool_result)[:500],
            })

            history_entry = (
                f"\n\n--- Turn {turn + 1} Result ---\n"
                f"Operation: {tool_id}\n"
                f"Result: {tool_result}\n"
                f"--- End Turn {turn + 1} ---\n\n"
                "Output your next operation (REASONING + NEXT_OP + PARAMS):"
            )

            budget_hint = budget.get_budget_hint(turn)
            if budget_hint:
                history_entry += budget_hint

            conversation_history += f"\n[{tool_id} executed]\n"
            conversation_history += history_entry

            if len(conversation_history) > 40000:
                task_header = f"TASK: {description}\n\n"
                remaining = conversation_history[len(task_header):]
                conversation_history = (
                    task_header
                    + "[Earlier steps truncated]\n...\n"
                    + remaining[-30000:]
                )

        if not result["response"]:
            result["response"] = "\n---\n".join(all_responses)

        result["event_log"] = within_task_patches

    except APIUnavailableError:
        raise
    except Exception as e:
        result["error"] = str(e)[:300]
    finally:
        result["time_cost"] = time.time() - t0

    return result

# ─── AI Response Processor ────────────────────────────────────────────────

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
    clean_action: str = ""
    valuable_reasoning: str = ""
    error_type: str = ""  # "no_action" | "invalid_id" | ""
    ai_filtered: bool = False


class AIResponseProcessor:
    """Extracts actions from LLM output by regex, AI-filters the non-action text
    down to valuable reasoning, keeps the conversation history clean, retries on
    malformed output, and breaks action loops.
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
        """action_pattern/params_pattern need named groups 'action'/'params'.
        llm_fn is `async (prompt) -> str`, used only when enable_ai_filter.
        """
        self._action_re = re.compile(action_pattern)
        self._params_re = re.compile(params_pattern)
        self._completion_signals = completion_signals or ["ALL_DONE"]
        self._max_retries = max_retries
        self._valid_ids = valid_action_ids
        self._llm_fn = llm_fn
        self._enable_ai_filter = enable_ai_filter
        self._min_text_for_ai = min_text_length_for_ai

        self._history_parts: list[str] = []
        self._consecutive_failures: int = 0
        self._turn_count: int = 0
        self._task_context: str = ""

        # Loop detection over recent (action_id, params_raw) tuples
        self._recent_actions: list[tuple[str, str]] = []
        self._loop_threshold: int = 3
        self._same_tool_threshold: int = 4
        self._search_tool_ids: set[str] = set()

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
        """True on exact repetition of (action_id, params), or on same-tool
        spinning (same action_id dominating recent turns with varying params).
        """
        if len(self._recent_actions) < self._loop_threshold:
            return False

        last_n = self._recent_actions[-self._loop_threshold:]
        if len(set(last_n)) == 1:
            return True

        if len(self._recent_actions) >= self._same_tool_threshold:
            last_m = self._recent_actions[-self._same_tool_threshold:]
            tool_ids = [a[0] for a in last_m]
            from collections import Counter
            most_common_tool, count = Counter(tool_ids).most_common(1)[0]
            if count >= self._same_tool_threshold - 1:
                return True

        return False

    def get_loop_break_prompt(self) -> str:
        """Prompt to break a detected loop. Must NOT suggest more searching or
        browsing — that is what caused the loop; force skipping the stuck subtask.
        """
        if not self._recent_actions:
            return ""
        last_action, last_params = self._recent_actions[-1]

        last_m = self._recent_actions[-min(self._same_tool_threshold, len(self._recent_actions)):]
        tool_ids = [a[0] for a in last_m]
        from collections import Counter
        most_common_tool, count = Counter(tool_ids).most_common(1)[0]

        if count >= self._same_tool_threshold - 1:
            prompt = (
                f"\n\n⚠️ BUDGET EXCEEDED: You called {most_common_tool} {count} times. "
                "STOP SEARCHING IMMEDIATELY.\n"
                "MANDATORY ACTION — pick ONE:\n"
                "  A) If you found ANY partial info (a name, email, etc.), USE IT NOW to proceed.\n"
                "  B) If you found NOTHING, SKIP this sub-task entirely and move to the next step.\n"
                "  C) Try ONE different short keyword (single word only), then proceed regardless.\n"
                "DO NOT browse/paginate/list contacts anymore. Move forward NOW.\n\n"
                "Your next operation (THINK + NEXT_OP + PARAMS):"
            )
        else:
            prompt = (
                f"\n\n⚠️ LOOP DETECTED: {last_action} called {self._loop_threshold}x with same params. "
                "STOP. Try a COMPLETELY different operation or skip this step.\n\n"
                "Your next operation (THINK + NEXT_OP + PARAMS):"
            )
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
        """Returns the valuable reasoning to keep, or "" if it is all noise.
        Every failure path keeps the original text — never lose info on error.
        """
        if not self._llm_fn:
            return non_action_text

        if not non_action_text or len(non_action_text) < self._min_text_for_ai:
            return non_action_text

        prompt = RESPONSE_EVAL_PROMPT.format(
            task_context=self._task_context or "(not provided)",
            non_action_text=non_action_text[:1500],
        )

        try:
            response = await self._llm_fn(prompt)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if data.get("is_all_noise", False) and data.get("confidence", 0) > 0.7:
                    return ""
                valuable = data.get("valuable_parts", "")
                if valuable:
                    return valuable
            return non_action_text
        except Exception:
            return non_action_text

    async def process_response(self, raw_response: str) -> ProcessedResponse:
        """Extract the action from a raw LLM response and AI-evaluate the rest."""
        self._turn_count += 1

        cleaned = re.sub(r'Tool \S+ not found in agent cli\.', '', raw_response).strip()

        for signal in self._completion_signals:
            if signal in cleaned:
                self._consecutive_failures = 0
                return ProcessedResponse(
                    valid=True,
                    is_completion=True,
                    raw_response=raw_response,
                    clean_action=signal,
                )

        action_match = self._action_re.search(cleaned)
        if not action_match:
            self._consecutive_failures += 1
            return ProcessedResponse(
                valid=False,
                error_type="no_action",
                raw_response=raw_response,
            )

        action_id = action_match.group("action")

        if self._valid_ids and action_id not in self._valid_ids:
            self._consecutive_failures += 1
            return ProcessedResponse(
                valid=False,
                action_id=action_id,
                error_type="invalid_id",
                raw_response=raw_response,
                clean_action=f"NEXT_OP: {action_id}",
            )

        params_raw = ""
        params_parsed: dict[str, Any] = {}
        params_match = self._params_re.search(cleaned)
        if params_match:
            params_raw = params_match.group("params").strip()
            if params_raw and params_raw != "(none)":
                params_parsed = self._parse_params(params_raw)

        clean_lines = [f"NEXT_OP: {action_id}"]
        if params_raw:
            clean_lines.append(f"PARAMS: {params_raw}")
        clean_action = "\n".join(clean_lines)

        valuable_reasoning = ""
        ai_filtered = False
        if self._enable_ai_filter:
            non_action = self._extract_non_action_text(cleaned)
            if non_action:
                valuable_reasoning = await self._ai_evaluate_content(non_action)
                ai_filtered = True

        self._consecutive_failures = 0

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
        """Parse a pipe-separated key:value param string into a dict, converting
        JSON arrays/objects, ints, floats and bools; anything else stays str.
        """
        result: dict[str, Any] = {}
        parts = self._smart_split_params(params_str)
        for part in parts:
            part = part.strip()
            if ":" in part:
                key, val = part.split(":", 1)
                key = key.strip()
                val = val.strip()
                if val.startswith("[") or val.startswith("{"):
                    parsed_val = self._try_parse_json_value(val)
                    if parsed_val is not None:
                        val = parsed_val
                elif val.lstrip("-").isdigit():
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                elif self._is_float(val):
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                elif val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                result[key] = val
        return result

    @staticmethod
    def _smart_split_params(params_str: str) -> list[str]:
        """Split params on | but NOT inside [] or {} — otherwise a JSON array
        containing separator-like characters gets torn apart.
        """
        parts = []
        current = []
        depth = 0
        for ch in params_str:
            if ch in ("[", "{"):
                depth += 1
                current.append(ch)
            elif ch in ("]", "}"):
                depth = max(0, depth - 1)
                current.append(ch)
            elif ch == "|" and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    @staticmethod
    def _try_parse_json_value(val: str) -> Any:
        """Parse a JSON value, tolerating smart/curly quotes, single quotes and
        unquoted lists ([a@b.com, c@d.com]). Returns None if nothing parses.
        """
        try:
            return json.loads(val)
        except (ValueError, json.JSONDecodeError):
            pass

        fixed = val.replace("\u201c", '"').replace("\u201d", '"')
        fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
        fixed = fixed.replace("'", '"')
        try:
            return json.loads(fixed)
        except (ValueError, json.JSONDecodeError):
            pass

        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if inner:
                items = []
                for item in inner.split(","):
                    item = item.strip().strip('"').strip("'")
                    if item:
                        items.append(item)
                if items:
                    return items

        return None

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
        """Build the conversation_history entry for a processed response: valuable
        reasoning, the tool result, an error hint, and any notifications.
        """
        parts = []

        if processed.valuable_reasoning:
            parts.append(f"[Agent reasoning: {processed.valuable_reasoning}]")

        parts.append(f"\n\nRESULT ({processed.action_id}):\n{result_str}")

        error_hint = self._detect_tool_error(result_str)
        if error_hint:
            parts.append(error_hint)

        if notifications:
            for n in notifications:
                parts.append(f"\nNOTIFICATION: {n}")

        parts.append("\n\nNext (THINK + NEXT_OP + PARAMS):")
        return "".join(parts)

    @staticmethod
    def _detect_tool_error(result_str: str) -> str:
        """Retry guidance for a tool error, or "" if the result is not an error."""
        try:
            data = json.loads(result_str) if isinstance(result_str, str) else result_str
            if not isinstance(data, dict) or not data.get("error"):
                return ""
            error_msg = str(data["error"])
        except (json.JSONDecodeError, TypeError, ValueError):
            if '"error"' not in result_str:
                return ""
            error_msg = result_str

        if "list[str]" in error_msg or "type list" in error_msg:
            return (
                "\n⚠️ TOOL ERROR: Parameter type mismatch. "
                "Format list parameters as: param_name:[\"value1\",\"value2\"]. "
                "RETRY with corrected format."
            )
        elif "not found" in error_msg.lower():
            return (
                "\n⚠️ TOOL ERROR: Resource not found. "
                "Check the ID/name and retry with correct value."
            )
        elif "required" in error_msg.lower():
            return (
                "\n⚠️ TOOL ERROR: Missing required parameter. "
                "Check which parameters are needed and retry."
            )
        else:
            return f"\n⚠️ TOOL ERROR: {error_msg[:150]}. RETRY with corrected parameters."

    def append_to_history(self, entry: str) -> None:
        """Append a pre-built entry to conversation history."""
        self._history_parts.append(entry)

    def append_error(self, error_msg: str) -> None:
        """Append an error message to conversation history."""
        self._history_parts.append(
            f"\n\n{error_msg}\n\nYour next operation (output NEXT_OP and PARAMS):"
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
            # Tell the agent the VALID RANGE explicitly. gaia2 numbers ops
            # op-001..op-N positionally PER SCENARIO, so the agent often emits
            # out-of-range ids (op-081 when only 76 tools exist). Without the
            # range it just re-hallucinates; with it, it self-corrects.
            valid = sorted(self._valid_ids) if self._valid_ids else []
            rng = f"Valid operations are {valid[0]}–{valid[-1]}. " if valid else ""
            prompt = (
                f"\n\nERROR: '{failed.action_id}' is not a valid operation. "
                f"{rng}Choose an existing op-id from the OPERATIONS list — "
                "do NOT invent or guess IDs.\n\n"
                "Your next operation (output ONLY NEXT_OP and PARAMS):"
            )
        else:
            prompt = "\n\nPlease output your next operation (THINK + NEXT_OP + PARAMS):"

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