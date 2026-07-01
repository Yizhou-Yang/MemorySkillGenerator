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
os.environ['CODEBUDDY_MODEL'] = 'hy3-preview-ioa'
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
from scripts.latest.gaia_runner import AIResponseProcessor

# ─── Extracted module imports ─────────────────────────────────────────────
from scripts.latest.trace import TraceLogger, APIUnavailableError
from scripts.latest.llm_client import (
    probe_api_available, _check_api_error, _API_FAILURE_THRESHOLD,
    save_checkpoint, load_checkpoint, clear_checkpoint,
    llm_review_fn,
    _query_sync, _llm_call, _query_notool_sync, _llm_call_notool,
    _llm_short_call, llm_extract_answer, llm_judge_answer,
    llm_critic_skill_quality, _shutdown_loop,
    use_openai_backend, openai_backend_ready, openai_tool_chat,
)
from scripts.latest.eval import (
    normalize_answer, exact_match, evaluate_task,
    compute_partial_results_from_trace,
)

import logging
logger = logging.getLogger("gaia2-runner")

# ─── GAIA2 native tool-calling toggle + SDK MCP detection ─────────────────
# The op-NNN text protocol (below) was a stopgap from when we thought the SDK
# couldn't expose custom tools. It can: the CodeBuddy SDK hosts an in-process
# MCP server (same mechanism as the Claude Agent SDK it mirrors), so the model
# calls ARE's tools natively instead of parsing NEXT_OP/PARAMS lines. That is
# how the official GAIA2/ARE harnesses drive the agent, so raw scores line up
# with the leaderboard far better. Native is the default; set GAIA2_NATIVE_TOOLS=0
# to force the text protocol. If the installed SDK lacks MCP support we detect it
# here and keep the text protocol automatically.
_GAIA2_NATIVE_TOOLS = os.environ.get("GAIA2_NATIVE_TOOLS", "1") == "1"
try:
    from codebuddy_agent_sdk.mcp.types import (
        SdkMcpServer,
        SdkMcpServerOptions,
        SdkMcpToolDefinition as _SdkMcpTool,
        ToolInputSchema,
    )
    _MCP_AVAILABLE = True
except Exception:
    SdkMcpServer = None
    SdkMcpServerOptions = None
    _SdkMcpTool = None
    ToolInputSchema = None
    _MCP_AVAILABLE = False

MODEL = "hy3-preview-ioa"
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

# ─── GAIA2 native tool-calling (SDK in-process MCP server) ────────────────

def _sanitize_tool_name(name: str) -> str:
    """ARE tools are named App__method; collapse to an MCP-safe single token."""
    s = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    while "__" in s:
        s = s.replace("__", "_")
    return (s[:60] or "tool").strip("_") or "tool"


def _make_native_handler(session, are_name: str, result: dict):
    """Async MCP handler that forwards one call to the ARE session.

    The environment side-effect and its event-log entry happen inside
    ``session.call_tool`` regardless of how the SDK reads our return envelope,
    so scoring stays correct even if the result shape is interpreted loosely.
    """
    async def handler(args):
        a = args if isinstance(args, dict) else {}
        try:
            if are_name == "__wait__":
                tr = session.wait_for_notification(int(a.get("timeout_seconds", 60)))
            elif are_name == "__time__":
                tr = session.get_time()
            else:
                tr = session.call_tool(are_name, a)
        except Exception as e:  # a tool error must not kill the agent loop
            tr = {"error": str(e), "type": type(e).__name__}
        result["actions"].append({
            "tool": are_name, "args": a, "result_preview": str(tr)[:500],
        })
        return {"content": [{"type": "text", "text": json.dumps(tr, default=str)[:4000]}]}
    return handler


def _make_tool_def(name, description, input_schema, handler):
    """Construct an SdkMcpToolDefinition from a raw handler."""
    props = input_schema.get("properties", {})
    req = input_schema.get("required", [])
    schema = ToolInputSchema(type="object", properties=props, required=req)
    return _SdkMcpTool(
        name=name, description=description,
        input_schema=schema, handler=handler,
    )


def _build_are_mcp_server(session, result: dict):
    """Expose every ARE tool as a native MCP tool. Returns (server, allowed_tools)."""
    defs = []
    used: set = set()
    for sch in session.get_tool_schemas():
        fn = sch.get("function", {})
        are_name = fn.get("name")
        if not are_name:
            continue
        safe = _sanitize_tool_name(are_name)
        while safe in used:
            safe += "_x"
        used.add(safe)
        params = fn.get("parameters") or {"type": "object", "properties": {}, "required": []}
        defs.append(_make_tool_def(
            safe, (fn.get("description") or are_name)[:1000], params,
            _make_native_handler(session, are_name, result),
        ))

    # Synthetic control tools: wait for follow-ups, read the clock.
    for safe, are_name, desc, params in [
        ("wait_for_notification", "__wait__",
         "Advance simulated time and wait for environment notifications such as a "
         "user's follow-up reply or a scheduled event.",
         {"type": "object",
          "properties": {"timeout_seconds": {
              "type": "integer", "description": "Maximum seconds to wait."}},
          "required": []}),
        ("get_time", "__time__", "Return the current simulation time.",
         {"type": "object", "properties": {}, "required": []}),
    ]:
        if safe in used:
            continue
        used.add(safe)
        defs.append(_make_tool_def(safe, desc, params,
                                   _make_native_handler(session, are_name, result)))

    server = SdkMcpServer(SdkMcpServerOptions(name="are", version="1.0.0", tools=defs))
    server_cfg = {"type": "sdk", "name": "are", "server": server}
    allowed = [f"mcp__are__{d}" for d in used]
    return server_cfg, allowed


def _build_native_system_prompt(experience_section: str, has_twist: bool) -> str:
    lines = [
        "You are an autonomous agent that operates a user's applications (email, "
        "calendar, contacts, messaging, files, and others) through function calls.",
        "Complete the user's request by calling the available tools. Read each tool "
        "result before choosing the next call.",
        "Guidelines:",
        "- Actually perform the requested actions with tools; do not just describe them.",
        "- Inspect state (list, search, read) before you create, modify, or send, so "
        "you act on real ids and values instead of guesses.",
        "- Do only what the task asks. Do not add extra messages or events.",
        "- Deliver your final answer with the appropriate send-message-to-user tool.",
    ]
    if has_twist:
        lines.append(
            "- The user may send a follow-up. After you reply, call "
            "wait_for_notification to receive it, then handle it before you finish.")
    lines.append("- Once the task is fully handled, stop calling tools.")
    prompt = "\n".join(lines)
    if experience_section:
        prompt += "\n\n" + experience_section.strip()
    return prompt


async def _drive_native_query(user_prompt, system_prompt, server, allowed, max_turns, result):
    """Run the SDK agent loop; collect assistant text (tool calls run in handlers)."""
    opt = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions",
        model=MODEL,
        max_turns=max_turns,
        cwd="/tmp",
        mcp_servers={"are": server},
        allowed_tools=allowed,
        system_prompt=system_prompt,
    )
    texts = []
    gen = None
    try:
        async with asyncio.timeout(int(os.environ.get("GAIA2_NATIVE_TIMEOUT", "1200"))):
            gen = query(prompt=user_prompt, options=opt)
            async for msg in gen:
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            continue  # side-effect + logging happen in the handler
                        txt = getattr(block, "text", "")
                        if txt:
                            texts.append(txt)
    except Exception as e:
        result.setdefault("error", str(e)[:300])
    finally:
        if gen is not None:
            try:
                await gen.aclose()
            except Exception:
                pass
    result["response"] = "\n".join(t for t in texts if t)[:8000]
    result["reasoning_trace"] = texts[-20:]


def _gaia2_native_sync(scenario_path, task_desc, experience_section, base_result, max_turns):
    """Synchronous worker: own ARE session + own event loop, native tool-calling.

    Returns the result dict. Sets ``_native_failed=True`` only when the MCP server
    cannot be built, so the caller can fall back to the text protocol without
    having spent a full agent run.
    """
    from scripts.latest.are_integration import ARESession

    result = dict(base_result)
    result["actions"] = []
    result["event_log"] = []
    result["reasoning_trace"] = []
    t0 = time.time()
    session = None
    try:
        session = ARESession(scenario_path)

        tmsg = session.call_tool("AgentUserInterface__get_last_message_from_user", {})
        task_content = ""
        if isinstance(tmsg, dict):
            r = tmsg.get("result")
            task_content = r.get("content", "") if isinstance(r, dict) else str(r or "")
        task_content = task_content or task_desc
        result["actions"].append({
            "tool": "AgentUserInterface__get_last_message_from_user",
            "args": {}, "result_preview": task_content[:200],
        })

        has_twist = detect_twist(task_content, task_desc)

        try:
            server, allowed = _build_are_mcp_server(session, result)
        except Exception as e:
            logger.warning("GAIA2 native MCP build failed (%s); using text protocol", e)
            result["_native_failed"] = True
            return result

        system_prompt = _build_native_system_prompt(experience_section, has_twist)
        user_prompt = (
            f"TASK: {task_content}\n\n"
            "Complete this task using the available tools. When it is fully done, stop."
        )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_drive_native_query(
                user_prompt, system_prompt, server, allowed, max_turns, result))
        finally:
            _shutdown_loop(loop)

        result["event_log"] = session.event_log
    except Exception as e:
        result["error"] = str(e)[:300]
    finally:
        if session:
            session.close()
        result["time_cost"] = time.time() - t0
    return result


def _gaia2_openai_native_sync(scenario_path, task_desc, experience_section, base_result, max_turns):
    """OpenAI-compatible native tool-calling worker (external-reviewer path).

    ARE's get_tool_schemas() is already OpenAI function-calling shaped, so this runs
    a standard chat loop with no CodeBuddy dependency: the model emits tool_calls, we
    execute them against the ARE session, feed the results back, and repeat. Same
    result-dict shape (event_log drives scoring) as the SDK MCP worker.
    """
    from scripts.latest.are_integration import ARESession

    result = dict(base_result)
    result["actions"] = []
    result["event_log"] = []
    result["reasoning_trace"] = []
    t0 = time.time()
    session = None
    try:
        session = ARESession(scenario_path)

        tmsg = session.call_tool("AgentUserInterface__get_last_message_from_user", {})
        task_content = ""
        if isinstance(tmsg, dict):
            r = tmsg.get("result")
            task_content = r.get("content", "") if isinstance(r, dict) else str(r or "")
        task_content = task_content or task_desc
        result["actions"].append({
            "tool": "AgentUserInterface__get_last_message_from_user",
            "args": {}, "result_preview": task_content[:200],
        })

        has_twist = detect_twist(task_content, task_desc)

        # ARE schemas are OpenAI-compatible; add the two synthetic control tools.
        tools = list(session.get_tool_schemas())
        tools.append({"type": "function", "function": {
            "name": "wait_for_notification",
            "description": ("Advance simulated time and wait for environment "
                            "notifications such as a user's follow-up reply."),
            "parameters": {"type": "object", "properties": {
                "timeout_seconds": {"type": "integer",
                                    "description": "Maximum seconds to wait."}},
                "required": []}}})
        tools.append({"type": "function", "function": {
            "name": "get_time",
            "description": "Return the current simulation time.",
            "parameters": {"type": "object", "properties": {}, "required": []}}})

        system_prompt = _build_native_system_prompt(experience_section, has_twist)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"TASK: {task_content}\n\nComplete this task using the available "
                "tools. When it is fully done, stop."},
        ]
        texts = []
        timeout = int(os.environ.get("GAIA2_OPENAI_TIMEOUT", "120"))

        for _turn in range(max_turns):
            r = openai_tool_chat(messages, tools, timeout=timeout)
            if r.get("error"):
                result.setdefault("error", r["error"])
                break
            assistant = r.get("assistant_message") or {"role": "assistant", "content": ""}
            if assistant.get("content"):
                texts.append(assistant["content"])
            messages.append(assistant)
            calls = r.get("tool_calls") or []
            if not calls:
                break  # model produced a final answer with no tool calls
            for tc in calls:
                name = tc["name"]
                args = tc["arguments"] if isinstance(tc.get("arguments"), dict) else {}
                try:
                    if name == "wait_for_notification":
                        tr = session.wait_for_notification(int(args.get("timeout_seconds", 60)))
                    elif name == "get_time":
                        tr = session.get_time()
                    else:
                        tr = session.call_tool(name, args)
                except Exception as e:
                    tr = {"error": str(e), "type": type(e).__name__}
                result["actions"].append({
                    "tool": name, "args": args, "result_preview": str(tr)[:500]})
                messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "content": json.dumps(tr, default=str)[:4000]})

        result["response"] = "\n".join(t for t in texts if t)[:8000]
        result["reasoning_trace"] = texts[-20:]
        result["event_log"] = session.event_log
    except Exception as e:
        result["error"] = str(e)[:300]
    finally:
        if session:
            session.close()
        result["time_cost"] = time.time() - t0
    return result


# ─── GAIA2 ARE runner (text protocol fallback + native dispatch) ──────────


async def run_gaia2_task_with_are(task: dict, experience_section: str = "",
                                   group: str = "A", **kwargs) -> dict:
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

    # ── Native tool-calling path (preferred, aligns with the official harness) ──
    # Give the model ARE's tools as real function-calling tools. Internally this
    # uses an in-process CodeBuddy SDK MCP server; for external reviewers with no
    # CodeBuddy CLI it uses a standard OpenAI-compatible tools= loop (ARE schemas
    # are already OpenAI-shaped). Both run in a worker thread with their own session
    # and event loop. Falls through to the text protocol only if neither native
    # backend is usable or the SDK server cannot be built.
    if _GAIA2_NATIVE_TOOLS:
        worker = None
        if use_openai_backend():
            if openai_backend_ready():
                worker = _gaia2_openai_native_sync
        elif _MCP_AVAILABLE:
            worker = _gaia2_native_sync
        if worker is not None:
            loop = asyncio.get_event_loop()
            native = await loop.run_in_executor(
                None, worker,
                scenario_path, task_desc, experience_section, dict(result),
                int(os.environ.get("GAIA2_MAX_TURNS", "50")),
            )
            if native is not None and not native.pop("_native_failed", False):
                return native

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
            "Start with your first operation (THINK + NEXT_OP + PARAMS):"
        )

        # 25 was too small: multi-app GAIA2 tasks need 6-10 oracle actions plus
        # exploration and a twist follow-up, and the BudgetTracker's own thresholds
        # (75,50,30,15,5) assume a larger budget -- at 25 the "URGENT/FINAL" hints fire
        # in the first half and the agent quits mid-task. Env-tunable.
        max_turns = int(os.environ.get("GAIA2_MAX_TURNS", "50"))
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
            r = await _llm_call_notool(system_prompt, conversation_history, timeout=120)

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
                timeout_val = int(tool_args.get("timeout_seconds", 60))
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
                    f"THINK: Need to wait for user reply before finishing.\n"
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
        # Collect all responses into result. Strip the CodeBuddy SDK's
        # "Tool op-XXX not found in agent cli." noise that leaks in when the
        # model emits a tool call instead of a NEXT_OP line (mirrors gaia_runner).
        import re as _re
        cleaned = [_re.sub(r'Tool \S+ not found in agent cli\.', '', x).strip()
                   for x in all_responses]
        result["response"] = "\n---\n".join(c for c in cleaned if c)
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

