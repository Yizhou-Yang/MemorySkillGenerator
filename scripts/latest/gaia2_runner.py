#!/usr/bin/env python3
"""SkillForge Latest — Latest Experiment Runner"""
import asyncio
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'deepseek-v4-pro'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')

from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage, ToolUseBlock

from latest import (SkillForgeLatest, ExperienceLibrary, Experience,
                build_augmented_prompt, ai_review_experience,
                cross_agent_evaluate_skill)
from latest.safety import CompletionGate, BudgetTracker, NoRepeatGuard
from latest.llm.prompts import build_agent_system_prompt, detect_twist, get_twist_reminder
from latest.injection import format_evoarena_patch_log, format_skillforge_patch_log

# ─── Extracted module imports ─────────────────────────────────────────────
from scripts.latest.trace import TraceLogger, APIUnavailableError
from scripts.latest.llm_client import (
    probe_api_available, _check_api_error,
    save_checkpoint, load_checkpoint, clear_checkpoint,
    llm_review_fn,
    _query_sync, _llm_call, _query_notool_sync, _llm_call_notool,
    _llm_short_call, llm_extract_answer, llm_judge_answer,
    llm_critic_skill_quality,
)
from scripts.latest.eval import (
    normalize_answer, exact_match, evaluate_task,
    compute_partial_results_from_trace,
)

MODEL = "deepseek-v4-pro"
CONCURRENCY = 15
TASK_TIMEOUT_QA = 180
TASK_TIMEOUT_AGENT = 300
QUALITY_THRESHOLD = 5

RESULTS_DIR = str(PROJECT_ROOT / "experiments_results" / "latest")

# ─── Trace Logger (for human review of prompts & responses) ───────────────
TASK_LIMITS = {"gaia": 165, "gaia2": 50, "swebench_dynamic": 30}

CHECKPOINT_FILE = str(PROJECT_ROOT / "experiments_results" / "latest" / "_checkpoint.json")

# ─── Trace logger instance ────────────────────────────────────────────────
_trace = TraceLogger(RESULTS_DIR)

# ─── GAIA2 ARE runner (real tool calling) ─────────────────────────────────

# ─── GAIA2 ARE runner (real tool calling) ─────────────────────────────────


async def run_gaia2_task_with_are(task: dict, experience_section: str = "",
                                   group: str = "A") -> dict:
    """Run a GAIA2 task using the real ARE simulation environment.

    The LLM interacts with the environment through function calling:
    1. Load scenario → initialize ARE session
    2. Get tool schemas → build system prompt with available tools
    3. LLM generates tool calls → execute in ARE → return results → loop
    4. Record event log for evaluation
    """
    from scripts.latest.are_integration import ARESession

    task_id = task["task_id"]
    task_desc = task.get("description") or task.get("task_desc", "")
    metadata = task.get("metadata", {})
    scenario_path = metadata.get("scenario_path") or task.get("scenario_path", "")

    result = {
        "task_id": task_id,
        "expected": task.get("expected", []),
        "oracle_answer": task.get("oracle_answer", ""),
        "description": task_desc,
        "metadata": metadata,
        "response": "",
        "error": None,
        "time_cost": 0,
        "augmented": bool(experience_section),
        "group": group,
        "actions": [],
        "event_log": [],
    }

    if not scenario_path:
        result["error"] = "no_scenario_path"
        return result

    t0 = time.time()
    session = None
    try:
        # Initialize ARE session
        session = ARESession(scenario_path)

        # Pre-fetch the user task (avoid LLM needing to call get_last_message)
        task_message = session.call_tool("AgentUserInterface__get_last_message_from_user", {})
        task_content = ""
        if isinstance(task_message, dict):
            task_content = task_message.get("result", {}).get("content", "") if isinstance(task_message.get("result"), dict) else str(task_message.get("result", ""))
        if not task_content:
            task_content = task_desc  # Fallback to task description

        result["actions"].append({
            "tool": "AgentUserInterface__get_last_message_from_user",
            "args": {},
            "result_preview": task_content[:200],
        })

        # Build tool ID mapping (op-001, op-002, ...)
        tool_names = list(session._all_tools.keys())
        tool_id_map = {}  # op-001 -> ARE tool name
        tool_lines = []
        for i, name in enumerate(tool_names):
            tid = f"op-{i+1:03d}"
            tool_id_map[tid] = name
            tool = session._all_tools[name]
            # Build compact param list
            params = []
            for arg in tool.args:
                req = " [required]" if not arg.has_default else ""
                params.append(f"{arg.name}{req}")
            params_str = ", ".join(params) if params else "(none)"
            # Use short description
            desc = (tool.function_description or "")[:60]
            tool_lines.append(f"{tid}: {desc} | {params_str}")

        # Add special tools
        tool_lines.append("op-000: Wait for environment notification | timeout_seconds")
        tool_id_map["op-000"] = "_wait_for_notification"

        tool_text = "\n".join(tool_lines)

        # Build system prompt from src/latest/agent_prompt.py
        system_prompt = build_agent_system_prompt(
            tool_text=tool_text,
            experience_section=experience_section,
        )

        # Multi-turn interaction loop
        # Since CodeBuddy SDK doesn't support multi-turn chat natively,
        # we accumulate conversation history in the user prompt.
        # Detect twist (conditional second phase) using src/latest/agent_prompt
        has_twist = detect_twist(task_content, task_desc)

        twist_reminder = get_twist_reminder() if has_twist else ""

        conversation_history = (
            f"TASK: {task_content}\n"
            f"{twist_reminder}\n"
            "Output your first operation now (NEXT_OP + PARAMS only):"
        )

        max_turns = 50
        all_responses = []
        reasoning_trace = []  # Collect AI-filtered valuable reasoning across turns

        # Framework-level completion gate: blocks premature ALL_DONE for twist tasks
        gate = CompletionGate()
        if has_twist:
            gate.set_condition("notify_user", required=True)
            gate.set_condition("wait_for_reply", required=True)

        # Framework-level budget tracker
        budget = BudgetTracker(max_turns=max_turns)

        # Framework-level idempotency guard: prevents re-executing identical (op, args)
        nr_guard = NoRepeatGuard()

        # Twist enforcement: track whether op-001 was called and op-000 followed
        op001_called_at_turn = -1  # Turn number when op-001 was called
        op000_called = False       # Whether op-000 has been called since last op-001
        primary_actions_done = False  # Heuristic: agent has done create+delete+send actions
        # AI Response Processor — filters noise from LLM output, keeps valuable reasoning
        async def _ai_filter_fn(prompt: str) -> str:
            """Lightweight LLM call for AI response filtering."""
            r = await _llm_call_notool(
                "You are a JSON-only response evaluator. Output valid JSON only.",
                prompt, timeout=30
            )
            return r.get("text", "")

        processor = AIResponseProcessor(
            valid_action_ids=set(tool_id_map.keys()),
            llm_fn=_ai_filter_fn,
            enable_ai_filter=True,
            min_text_length_for_ai=80,  # Only AI-eval if >80 chars of non-action text
        )
        processor.set_task_context(task_content[:500])

        for turn in range(max_turns):
            # Call LLM in pure-text mode (no CodeBuddy tools)
            r = await _llm_call_notool(system_prompt, conversation_history, timeout=300)

            if _check_api_error(r):
                raise APIUnavailableError(
                    f"API unavailable after {_API_FAILURE_THRESHOLD} consecutive failures"
                )

            response_text = r.get("text", "")
            all_responses.append(response_text)

            # Process response through AI filter
            processed = await processor.process_response(response_text)

            # Check completion — use framework CompletionGate
            if processed.is_completion:
                if not gate.can_complete():
                    # Reject ALL_DONE — conditions not met
                    hint = gate.get_rejection_hint()
                    # Add specific guidance for twist tasks
                    if has_twist and not op000_called:
                        hint += (
                            "You MUST call op-000 (wait for reply) before ALL_DONE:\n"
                            "NEXT_OP: op-000\n"
                            "PARAMS: timeout_seconds:60\n"
                        )
                    conversation_history += hint
                    continue  # Force another turn
                break

            # Handle invalid responses (no action found)
            if not processed.valid:
                if processed.error_type == "no_action":
                    if processor.consecutive_failures <= 3 and turn < max_turns - 1:
                        retry = processor.get_retry_prompt(processed)
                        conversation_history += retry
                        continue
                    break
                elif processed.error_type == "invalid_id":
                    retry = processor.get_retry_prompt(processed)
                    conversation_history += retry
                    continue

            tool_id = processed.action_id
            tool_args = processed.params_parsed

            # Loop detection: if agent is repeating the same call, break the loop
            if processor.is_looping:
                loop_count = getattr(processor, '_consecutive_loop_breaks', 0) + 1
                processor._consecutive_loop_breaks = loop_count
                if loop_count >= 3:
                    # Hard limit: agent is stuck in an unbreakable loop.
                    # Force completion to avoid wasting turns and inflating action counts.
                    break
                loop_prompt = processor.get_loop_break_prompt()
                conversation_history += loop_prompt
                continue  # Skip execution, force agent to try something different
            else:
                processor._consecutive_loop_breaks = 0  # Reset on non-loop turn

            # ── Runtime dedup: prevent re-executing identical (tool_id, args) ──
            # Uses NoRepeatGuard from src/latest/response_filter.py (paper core contribution)
            if nr_guard.would_repeat(tool_id, tool_args):
                conversation_history += nr_guard.get_warning(tool_id)
                continue  # Skip execution
            nr_guard.record(tool_id, tool_args)

            # Execute the tool
            are_tool_name = tool_id_map[tool_id]

            if are_tool_name == "_wait_for_notification":
                op000_called = True
                gate.mark_satisfied("wait_for_reply")  # Framework gate: reply waited
                timeout_val = int(tool_args.get("timeout_seconds", 180))
                tr = session.wait_for_notification(timeout_val)
            elif "send_message_to_user" in are_tool_name:
                op001_called_at_turn = turn
                op000_called = False  # Reset - must call op-000 after this
                gate.mark_satisfied("notify_user")  # Framework gate: user notified
                tr = session.call_tool(are_tool_name, tool_args)
            else:
                tr = session.call_tool(are_tool_name, tool_args)
                # Detect primary actions completion heuristically
                if not primary_actions_done:
                    method_lower = are_tool_name.lower()
                    if any(kw in method_lower for kw in ["send_email", "add_calendar", "create_event"]):
                        primary_actions_done = True

            result["actions"].append({
                "tool": are_tool_name,
                "args": tool_args,
                "result_preview": str(tr)[:500],
            })

            # Format result for conversation
            result_str = json.dumps(tr, default=str)[:2000]

            # Check for notifications
            notifications = []
            try:
                r_data = tr if isinstance(tr, dict) else json.loads(str(tr))
                if isinstance(r_data, dict) and r_data.get("notifications"):
                    for n in r_data["notifications"]:
                        notifications.append(n.get('message', str(n)))
            except (json.JSONDecodeError, TypeError):
                pass

            # Collect valuable reasoning for experience recording
            if processed.valuable_reasoning:
                reasoning_trace.append(processed.valuable_reasoning)

            # Build history entry with AI-filtered valuable reasoning
            history_entry = processor.build_history_entry(
                processed, result_str, notifications
            )

            # Framework-level budget tracking
            budget_warning = budget.get_budget_hint(turn)

            # Twist enforcement: if op-001 called but no op-000 within 1+ turns → force reminder
            if (has_twist and op001_called_at_turn >= 0 and not op000_called
                    and turn - op001_called_at_turn >= 1):
                gap = turn - op001_called_at_turn
                urgency = (
                    "IMMEDIATELY" if gap >= 2 else
                    "NOW - do not output any other operation"
                )
                twist_enforce = (
                    f"\n\n🛑 TWIST PROTOCOL VIOLATION: You called op-001 {gap} turns ago "
                    f"but have NOT called op-000 (wait for reply). "
                    f"The task has a conditional second phase. Next operation MUST be:\n"
                    f"NEXT_OP: op-000\n"
                    f"PARAMS: timeout_seconds:60\n"
                    f"This is REQUIRED before ALL_DONE. Call op-000 {urgency}."
                )
                budget_warning = twist_enforce + (budget_warning or "")

            if budget_warning:
                history_entry += budget_warning

            conversation_history += history_entry

            # Truncate conversation if too long (keep last 40000 chars)
            if len(conversation_history) > 50000:
                task_header = f"TASK: {task_content}\n\n"
                remaining = conversation_history[len(task_header):]
                conversation_history = (
                    task_header
                    + "[Earlier steps truncated]\n...\n"
                    + remaining[-40000:]
                )
        # Collect all responses into result
        result["response"] = "\n---\n".join(all_responses)
        result["event_log"] = session.event_log
        result["reasoning_trace"] = reasoning_trace  # Pass to experience recording

    except APIUnavailableError:
        raise
    except Exception as e:
        result["error"] = str(e)[:300]
    finally:
        if session:
            session.close()
        result["time_cost"] = time.time() - t0

    return result

