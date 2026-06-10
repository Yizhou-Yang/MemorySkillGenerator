"""ARE Integration for GAIA2 — In-process simulation environment.

Loads a GAIA2 scenario, initializes the ARE Environment, and provides:
  - Tool schema extraction (for LLM function calling)
  - Tool execution (call ARE methods directly)
  - Event logging (for evaluation against oracle)
  - Clock management (advance simulation time)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("are-integration")

# Lazy imports — ARE may not be installed in all environments
_ARE_AVAILABLE = None


def _check_are_available() -> bool:
    global _ARE_AVAILABLE
    if _ARE_AVAILABLE is None:
        try:
            from are.simulation.benchmark.scenario_loader import load_scenario  # noqa: F401
            from are.simulation.environment import Environment  # noqa: F401
            _ARE_AVAILABLE = True
        except ImportError:
            _ARE_AVAILABLE = False
    return _ARE_AVAILABLE


def _smart_serialize(obj: object) -> object:
    """Recursively serialize ARE objects to JSON-safe types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): _smart_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_smart_serialize(item) for item in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return _smart_serialize(asdict(obj))
    if hasattr(obj, "value") and hasattr(obj, "name"):
        return obj.value
    if hasattr(obj, "__dict__"):
        return {
            k: _smart_serialize(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return str(obj)


class ARESession:
    """A single GAIA2 scenario session with tool calling support.

    Usage:
        session = ARESession(scenario_path)
        tools = session.get_tool_schemas()  # For LLM function calling
        result = session.call_tool("Calendar__add_calendar_event", {...})
        session.close()
    """

    def __init__(self, scenario_path: str | Path):
        if not _check_are_available():
            raise ImportError(
                "meta-agents-research-environments not installed. "
                "Run: pip install meta-agents-research-environments==1.2.0"
            )

        from are.simulation.benchmark.scenario_loader import load_scenario
        from are.simulation.environment import (
            Environment, EnvironmentConfig, EnvironmentType
        )
        from are.simulation.notification_system import VerboseNotificationSystem

        scenario_path = Path(scenario_path)
        scenario_json = scenario_path.read_text()
        self._scenario, _ = load_scenario(
            scenario_json, scenario_path.as_posix(), load_completed_events=False
        )
        if self._scenario is None:
            raise ValueError(f"Failed to load scenario from {scenario_path}")

        self._scenario.initialize()

        self._env = Environment(
            config=EnvironmentConfig(
                oracle_mode=False,
                queue_based_loop=False,
                start_time=self._scenario.start_time,
                duration=None,
                exit_when_no_events=False,
                time_increment_in_seconds=self._scenario.time_increment_in_seconds,
            ),
            environment_type=EnvironmentType.CLI,
            notification_system=VerboseNotificationSystem(),
        )

        self._env.run(self._scenario, wait_for_end=False, schedule_events=True)
        self._env.pause()

        # Build tool registry
        self._all_tools: dict[str, Any] = {}
        self._tool_methods: dict[str, Any] = {}
        for app in self._scenario.apps:
            if not hasattr(app, "get_tools"):
                continue
            for tool in app.get_tools():
                public_name = tool._public_name
                self._all_tools[public_name] = tool
                method_name = (
                    public_name.split("__", 1)[1]
                    if "__" in public_name
                    else public_name
                )
                self._tool_methods[public_name] = getattr(
                    tool.class_instance, method_name
                )

        self._event_log: list[dict] = []
        self._last_tool_time: float = time.monotonic()
        self._closed = False

        logger.info(
            "ARESession initialized: %d apps, %d tools",
            len(self._scenario.apps),
            len(self._all_tools),
        )

    @property
    def tool_count(self) -> int:
        return len(self._all_tools)

    @property
    def event_log(self) -> list[dict]:
        return self._event_log

    def get_tool_schemas(self) -> list[dict]:
        """Get OpenAI-compatible function schemas for all ARE tools."""
        schemas = []
        for name, tool in self._all_tools.items():
            params = {"type": "object", "properties": {}, "required": []}
            for arg in tool.args:
                prop: dict[str, Any] = {"description": arg.description or arg.name}
                # Map ARE types to JSON schema types
                if "int" in str(getattr(arg, "type", "")).lower():
                    prop["type"] = "integer"
                elif "float" in str(getattr(arg, "type", "")).lower():
                    prop["type"] = "number"
                elif "bool" in str(getattr(arg, "type", "")).lower():
                    prop["type"] = "boolean"
                else:
                    prop["type"] = "string"
                params["properties"][arg.name] = prop
                if not arg.has_default:
                    params["required"].append(arg.name)

            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.function_description or name,
                    "parameters": params,
                },
            })
        return schemas

    def get_tool_descriptions_text(self, max_tools: int = 120) -> str:
        """Get a compact text description of all tools for system prompt injection."""
        lines = []
        for i, (name, tool) in enumerate(self._all_tools.items()):
            if i >= max_tools:
                lines.append(f"... and {len(self._all_tools) - max_tools} more tools")
                break
            args_desc = []
            for arg in tool.args:
                default_note = f" (default: {arg.default})" if arg.has_default else " [required]"
                args_desc.append(f"    - {arg.name}: {arg.description or 'no desc'}{default_note}")
            args_str = "\n".join(args_desc) if args_desc else "    (no arguments)"
            lines.append(f"- {name}: {tool.function_description or ''}\n{args_str}")
        return "\n".join(lines)

    def get_tool_descriptions_action_format(self, max_tools: int = 120) -> str:
        """Get tool descriptions using / separator format (avoids CodeBuddy tool detection).

        Output format uses AppName/method_name instead of AppName__method_name.
        """
        lines = []
        for i, (name, tool) in enumerate(self._all_tools.items()):
            if i >= max_tools:
                lines.append(f"... and {len(self._all_tools) - max_tools} more tools")
                break
            # Convert AppName__method to AppName/method
            action_name = name.replace("__", "/")
            args_desc = []
            for arg in tool.args:
                required = " [required]" if not arg.has_default else ""
                args_desc.append(f"{arg.name}{required}")
            args_str = ", ".join(args_desc) if args_desc else "(no args)"
            desc = tool.function_description or ""
            lines.append(f"- {action_name}: {desc} | Args: {args_str}")

        # Add special tools
        lines.append("- wait_for_notification: Wait for environment events | Args: timeout_seconds")
        lines.append("- get_time: Get current simulation time | Args: (no args)")
        return "\n".join(lines)

    def _advance_clock(self) -> None:
        """Advance simulation clock by elapsed real time (agent thinking time)."""
        elapsed = time.monotonic() - self._last_tool_time
        offset = min(elapsed, 300.0)
        if offset > 0:
            self._env.time_manager.add_offset(offset)
            self._env.tick()

    def _drain_notifications(self) -> list[dict]:
        """Get any pending notifications from the environment."""
        ns = self._env.notification_system
        if ns is None or ns.message_queue is None:
            return []
        current_time = self._env.time_manager.time()
        dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
        messages = ns.message_queue.get_by_timestamp(dt)
        return [
            {
                "type": msg.message_type.value if hasattr(msg.message_type, "value") else str(msg.message_type),
                "message": msg.message,
                "timestamp": (
                    msg.timestamp.isoformat()
                    if hasattr(msg.timestamp, "isoformat")
                    else str(msg.timestamp)
                ),
            }
            for msg in messages
        ]

    def _maybe_parse_json_arg(self, value: Any) -> Any:
        """Try to parse string arguments that look like JSON arrays/objects."""
        if not isinstance(value, str):
            return value
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return value
        if isinstance(parsed, (list, dict)):
            return parsed
        return value

    def _complete_matching_oracle_event(self, tool_name: str) -> None:
        """Find and complete a matching OracleEvent to advance the dependency chain.

        ARE uses dependency-driven event scheduling: ENV events (e.g., a friend's
        email reply) only fire after their prerequisite AGENT OracleEvents are
        marked as completed (event_time set). In oracle_mode=False, OracleEvents
        are ignored by process_event(), so we must manually mark them when the
        agent performs the matching action.

        IMPORTANT: OracleEvents with dependencies are NOT in the event_queue until
        their dependencies are satisfied. We must search scenario.events (the full
        event graph) to find all matching OracleEvents.
        """
        from are.simulation.types import OracleEvent, EventType

        # Extract app_name and function_name from tool_name (e.g., "Emails__send_email")
        if "__" not in tool_name:
            return
        app_name, func_name = tool_name.split("__", 1)

        # Search ALL scenario events (not just the queue — most OracleEvents
        # with dependencies haven't been placed in the queue yet)
        all_events = list(self._scenario.events) if hasattr(self._scenario, "events") else []

        eq = self._env.event_queue
        current_time = self._env.time_manager.time()

        for event in all_events:
            if not isinstance(event, OracleEvent):
                continue
            if not hasattr(event, "action_desc") or event.action_desc is None:
                continue
            if event.action_desc.app == app_name and event.action_desc.function == func_name:
                # Mark this oracle event as completed (even if already marked by
                # a predecessor's successor scheduling — we still need to process
                # its successors recursively)
                if event.event_time is None or event.event_time == 0.0:
                    event.event_time = current_time

                # Recursively schedule all ready successors
                self._schedule_ready_successors(event, eq, current_time)

                logger.debug(
                    "Completed OracleEvent %s (app=%s, func=%s)",
                    event.event_id[:30], app_name, func_name,
                )
                return

    def _schedule_ready_successors(self, event: Any, eq: Any, current_time: float,
                                   visited: set | None = None) -> None:
        """Recursively schedule all ready successors of an event.

        When a successor becomes ready and is an OracleEvent, we also mark it
        as completed and recurse into its successors. This ensures the entire
        dependency chain is resolved in one pass.
        """
        from are.simulation.types import OracleEvent, EventType

        if visited is None:
            visited = set()
        if id(event) in visited:
            return
        visited.add(id(event))

        for succ in event.successors:
            if id(succ) in visited:
                continue
            if not succ.is_ready():
                continue

            # Set event_time if not already set
            if succ.event_time is None or succ.event_time == 0.0:
                succ.event_time = current_time + 0.001

            if isinstance(succ, OracleEvent):
                # OracleEvent successors won't be processed by tick() in
                # oracle_mode=False, so we must recursively handle them
                self._schedule_ready_successors(succ, eq, current_time, visited)
            else:
                # ENV or other event types — put into queue for tick() to process
                eq.put(succ)

    def call_tool(self, tool_name: str, kwargs: dict) -> dict:
        """Execute an ARE tool and return the result.

        Returns:
            {"result": ..., "notifications": [...], "error": None}
            or {"error": "...", "type": "..."}
        """
        if self._closed:
            return {"error": "Session is closed", "type": "SessionClosed"}

        if tool_name not in self._tool_methods:
            return {"error": f"Unknown tool: {tool_name}", "type": "ToolNotFound"}

        method = self._tool_methods[tool_name]

        # Parse JSON string arguments
        parsed_kwargs = {}
        for key, value in kwargs.items():
            parsed_kwargs[key] = self._maybe_parse_json_arg(value)

        self._advance_clock()
        try:
            result = method(**parsed_kwargs)
        except Exception as e:
            response = {"error": str(e), "type": type(e).__name__}
            self._log_event(tool_name, parsed_kwargs, response)
            self._last_tool_time = time.monotonic()
            return response

        # Advance the oracle dependency chain for this tool call
        self._complete_matching_oracle_event(tool_name)

        # Tick to process triggered events (including newly scheduled ENV events)
        self._env.tick()

        notifications = self._drain_notifications()
        serialized = _smart_serialize(result)
        response: dict[str, Any] = {"result": serialized, "error": None}
        if notifications:
            response["notifications"] = notifications

        self._last_tool_time = time.monotonic()
        self._log_event(tool_name, parsed_kwargs, response)
        return response

    def wait_for_notification(self, timeout_seconds: int = 180) -> dict:
        """Wait for environment notifications (advance time event-by-event)."""
        self._advance_clock()
        timeout_boundary = self._env.time_manager.time() + timeout_seconds
        max_iterations = 500
        iterations = 0

        try:
            while iterations < max_iterations:
                iterations += 1
                next_event_time = None
                if hasattr(self._env, "get_next_event_time"):
                    next_event_time = self._env.get_next_event_time()
                elif hasattr(self._env, "event_queue") and hasattr(self._env.event_queue, "peek_time"):
                    next_event_time = self._env.event_queue.peek_time()

                if next_event_time is not None and next_event_time > timeout_boundary:
                    next_event_time = None

                if next_event_time is None:
                    jump = timeout_boundary - self._env.time_manager.time()
                    if jump > 0:
                        self._env.time_manager.add_offset(jump)
                    self._env.tick()
                    break
                else:
                    jump = next_event_time - self._env.time_manager.time()
                    if jump > 0:
                        self._env.time_manager.add_offset(jump)
                    self._env.tick()
                    # Check if notification arrived
                    ns = self._env.notification_system
                    if ns and ns.message_queue.has_new_messages(
                        datetime.fromtimestamp(self._env.time_manager.time(), tz=timezone.utc)
                    ):
                        break
        except Exception as e:
            logger.warning("wait_for_notification error: %s", e)

        notifications = self._drain_notifications()
        self._last_tool_time = time.monotonic()
        response = {
            "notifications": notifications,
            "sim_time": self._env.time_manager.time(),
            "waited_seconds": timeout_seconds,
            "iterations": iterations,
        }
        self._log_event("are_wait_for_notification", {"timeout_seconds": timeout_seconds}, response)
        return response

    def get_time(self) -> dict:
        """Get current simulation time."""
        return {
            "sim_time": self._env.time_manager.time(),
            "sim_time_passed": self._env.time_manager.time_passed(),
            "is_paused": self._env.time_manager.is_paused,
        }

    def _log_event(self, tool_name: str, args: dict, result: object) -> None:
        """Record tool call in event log."""
        method = tool_name.split("__", 1)[-1] if "__" in tool_name else tool_name
        entry = {
            "tool": tool_name,
            "method": method,
            "args": _smart_serialize(args),
            "result": _smart_serialize(result),
            "sim_time": self._env.time_manager.time(),
        }
        self._event_log.append(entry)

    def close(self) -> None:
        """Shut down the ARE environment."""
        if self._closed:
            return
        self._closed = True
        try:
            self._env.stop()
        except Exception:
            pass
        logger.info("ARESession closed. %d events logged.", len(self._event_log))


def load_gaia2_tasks_from_cli_dir(
    base_dir: str | Path,
    num_samples: int = 50,
) -> list[dict]:
    """Load GAIA2 tasks from the harbor CLI dataset directory.

    Each task directory contains:
      - instruction.md: System prompt for the agent
      - task_metadata.json: Metadata (config, difficulty, apps, etc.)
      - environment/scenario.json: The ARE scenario
      - tests/oracle_events.json: Ground truth actions
      - tests/oracle_task.txt: The user's task text
      - tests/oracle_answer.txt: Expected final answer
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        logger.warning("GAIA2 CLI directory not found: %s", base_dir)
        return []

    task_dirs = sorted(base_dir.iterdir())
    tasks = []

    for task_dir in task_dirs:
        if len(tasks) >= num_samples:
            break
        if not task_dir.is_dir():
            continue

        # Required files
        scenario_path = task_dir / "environment" / "scenario.json"
        oracle_task_path = task_dir / "tests" / "oracle_task.txt"
        oracle_events_path = task_dir / "tests" / "oracle_events.json"
        oracle_answer_path = task_dir / "tests" / "oracle_answer.txt"
        metadata_path = task_dir / "task_metadata.json"
        instruction_path = task_dir / "instruction.md"

        if not scenario_path.exists() or not oracle_task_path.exists():
            continue

        # Read task description
        task_desc = oracle_task_path.read_text().strip()
        if not task_desc:
            continue

        # Read metadata
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        # Read oracle events
        oracle_events = []
        if oracle_events_path.exists():
            try:
                oracle_events = json.loads(oracle_events_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        # Read oracle answer
        oracle_answer = ""
        if oracle_answer_path.exists():
            oracle_answer = oracle_answer_path.read_text().strip()

        # Read instruction
        instruction = ""
        if instruction_path.exists():
            instruction = instruction_path.read_text().strip()

        config = metadata.get("config", "unknown")
        difficulty = metadata.get("difficulty", "unknown")
        source_id = metadata.get("source_id", task_dir.name)

        tasks.append({
            "task_id": f"gaia2_{source_id}",
            "description": task_desc,
            "expected": oracle_events,  # List of oracle action dicts
            "oracle_answer": oracle_answer,
            "context": instruction,
            "metadata": {
                "benchmark": "gaia2",
                "config": config,
                "difficulty": difficulty,
                "scenario_path": str(scenario_path),
                "task_dir": str(task_dir),
                "app_names": metadata.get("app_names", []),
                "top_action_apps": metadata.get("top_action_apps", []),
                "top_action_functions": metadata.get("top_action_functions", []),
                "event_count": metadata.get("event_count", 0),
            },
        })

    logger.info("Loaded %d GAIA2 tasks from %s", len(tasks), base_dir)
    return tasks
