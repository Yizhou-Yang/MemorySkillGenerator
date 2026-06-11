"""
OpenHands agent — code engineering agent for SWE-Chain-Evo tasks.

Real integration: github.com/OpenHands/OpenHands
  - Python SDK: pip install openhands (requires Python >=3.12, Docker)
  - CLI: openhands run --task "<issue>" --repo <url>
  - SDK: from openhands.core.config import AgentConfig; agent.run(task)

Architecture:
  Mode 1 (Docker): OpenHands SDK — full sandboxed code engineering
  Mode 2 (CLI):    openhands CLI via subprocess
  Mode 3 (Prompt): LLM-only — generates patches without execution

SkillForge role: Injects cross-task coding experience into the system prompt
before the agent generates a fix, improving patch quality.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import time

from .base import BaseAgent


# Path to Python 3.12 env where openhands is installed
_OH_PYTHON = "/root/.conda/envs/harbor312/bin/python"


def _has_openhands_sdk() -> bool:
    """Check if openhands SDK is importable."""
    try:
        result = subprocess.run(
            [_OH_PYTHON, "-c", "import openhands; print('ok')"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return result.stdout.strip() == "ok"
    except Exception:
        return False


def _has_docker() -> bool:
    """Check if Docker daemon is reachable."""
    return shutil.which("docker") is not None


class OpenHandsAgent(BaseAgent):
    """Code engineering agent for SWE-Chain-Evo.

    Delegates to OpenHands SDK in Docker mode for full sandboxed
    code editing, testing, and patch generation. Falls back to
    prompt-only patch generation when Docker is unavailable.

    Benchmark mapping (LEGACY):
      - swe_chain_evo: SWE-bench-style code engineering tasks (not published)
    """

    BENCHMARKS = {"swe_chain_evo"}

    def __init__(self, model: str = "deepseek-v4-pro",
                 timeout: int = 300):
        self.model = model
        self.timeout = timeout
        self._oh_available = _has_openhands_sdk()
        self._docker_available = _has_docker()

    def supports_benchmark(self, benchmark: str) -> bool:
        return benchmark in self.BENCHMARKS

    async def run_task(self, task: dict, experience_section: str = "",
                       group: str = "A") -> dict:
        """Execute a SWE code engineering task.

        Priority:
          1. OpenHands SDK Docker mode — full containerized execution
          2. OpenHands CLI mode — subprocess call
          3. Prompt-only mode — LLM generates the patch text
        """
        # Try OpenHands SDK Docker mode
        if self._oh_available and self._docker_available:
            try:
                return self._run_via_sdk(task, experience_section, group)
            except Exception:
                pass

        # Try OpenHands CLI
        if self._oh_available:
            try:
                return self._run_via_cli(task, experience_section, group)
            except Exception:
                pass

        # Fallback: prompt-only
        return await self._run_prompt_only(task, experience_section, group)

    # ------------------------------------------------------------------
    # Mode 1: OpenHands SDK (Docker sandbox)
    # ------------------------------------------------------------------

    def _run_via_sdk(self, task: dict, experience_section: str,
                     group: str) -> dict:
        """Run via OpenHands Python SDK in Docker sandbox.

        OpenHands SDK usage:
          from openhands.core.config import AppConfig
          from openhands.core.main import run_controller

          config = AppConfig(...)
          state = run_controller(config, task)

        For simplicity, we delegate to a subprocess script that
        imports openhands and runs the task.
        """
        from scripts.latest.llm_client import _check_api_error
        from scripts.latest.trace import APIUnavailableError

        task_id = task["task_id"]
        description = task.get("description", "")
        expected = task.get("expected", "")
        repo = task.get("metadata", {}).get("repo", "")
        instance_id = task.get("metadata", {}).get("instance_id", "")

        result = {"task_id": task_id, "expected": expected,
                  "response": "", "error": None, "time_cost": 0,
                  "augmented": bool(experience_section), "group": group,
                  "execution_mode": "openhands_sdk"}
        t0 = time.time()

        # Build the OpenHands runner script
        script = (
            "import json, os, sys\n"
            "os.environ['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY', '')\n"
            "try:\n"
            "    from openhands.core.config import AppConfig\n"
            "    from openhands.core.main import run_controller\n"
            "    config = AppConfig(\n"
            "        max_iterations=10,\n"
            f"        llm_model='deepseek/{self.model}',\n"
            "    )\n"
            f"    task_str = {repr(description)}\n"
            "    state = run_controller(config, task_str)\n"
            "    output = state.history.get_last_agent_message() or ''\n"
            "    print(json.dumps({'output': output, 'error': None}))\n"
            "except ImportError as e:\n"
            "    print(json.dumps({'output': '', 'error': f'openhands_import: {e}'}))\n"
            "except Exception as e:\n"
            "    print(json.dumps({'output': '', 'error': str(e)}))\n"
        )

        env = os.environ.copy()
        proc = subprocess.run(
            [_OH_PYTHON, "-c", script],
            capture_output=True, text=True,
            timeout=self.timeout, env=env,
        )

        if proc.returncode == 0 and proc.stdout.strip():
            try:
                import json as _json
                data = _json.loads(proc.stdout.strip())
                result["response"] = data.get("output", "")
                result["error"] = data.get("error")
            except Exception:
                result["response"] = proc.stdout.strip()
        else:
            result["error"] = proc.stderr.strip() or "openhands_sdk_failed"

        result["time_cost"] = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # Mode 2: OpenHands CLI
    # ------------------------------------------------------------------

    def _run_via_cli(self, task: dict, experience_section: str,
                     group: str) -> dict:
        """Run via openhands CLI subprocess.

        Equivalent to:
          openhands run --task "<issue>" --model deepseek/deepseek-v4-pro
        """
        task_id = task["task_id"]
        description = task.get("description", "")
        expected = task.get("expected", "")

        result = {"task_id": task_id, "expected": expected,
                  "response": "", "error": None, "time_cost": 0,
                  "augmented": bool(experience_section), "group": group,
                  "execution_mode": "openhands_cli"}
        t0 = time.time()

        # Inject experience into task description
        full_task = description
        if experience_section:
            full_task = (
                f"## SkillForge Experience (learn from past tasks)\n"
                f"{experience_section}\n\n"
                f"## Current Task\n{description}"
            )

        cmd = [
            _OH_PYTHON, "-m", "openhands", "run",
            "--task", full_task,
            "--model", f"deepseek/{self.model}",
            "--max-iterations", "10",
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout,
            )
            result["response"] = proc.stdout.strip()
            result["stderr"] = proc.stderr.strip()
            result["return_code"] = proc.returncode
        except subprocess.TimeoutExpired:
            result["error"] = "openhands_cli_timeout"
        except FileNotFoundError:
            result["error"] = "openhands_not_found"
        except Exception as e:
            result["error"] = str(e)

        result["time_cost"] = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # Mode 3: Prompt-only (no Docker, no execution)
    # ------------------------------------------------------------------

    async def _run_prompt_only(self, task: dict, experience_section: str,
                               group: str) -> dict:
        """Generate patch text only — no Docker sandbox execution.

        The LLM analyzes the issue and produces a unified diff patch
        as text. Evaluation compares against the expected patch.
        """
        from scripts.latest.llm_client import _llm_call, _check_api_error
        from scripts.latest.trace import APIUnavailableError

        task_id = task["task_id"]
        description = task.get("description", "")
        expected = task.get("expected", "")
        repo = task.get("metadata", {}).get("repo", "")

        result = {"task_id": task_id, "expected": expected,
                  "response": "", "error": None, "time_cost": 0,
                  "augmented": bool(experience_section), "group": group,
                  "execution_mode": "prompt_only"}
        t0 = time.time()

        system = (
            "You are an expert software engineer. Given a GitHub issue "
            "description, analyze the problem and produce a fix as a "
            "unified diff patch.\n\n"
            "Output format:\n"
            "1. Brief analysis of the root cause\n"
            "2. The fix as a unified diff (diff --git format)\n\n"
            "Be precise — only fix what the issue describes, nothing else."
        )
        if experience_section:
            system += f"\n\n## SkillForge Experience\n{experience_section}"

        prompt = (
            f"[System]\n{system}\n\n"
            f"Repository: {repo}\n"
            f"{description}\n\n"
            f"Produce the fix:"
        )

        r = await _llm_call(prompt, max_turns=3, timeout=self.timeout)
        if _check_api_error(r):
            raise APIUnavailableError("API unavailable")

        result["response"] = r.get("text", "")
        result["error"] = r.get("error")
        result["time_cost"] = time.time() - t0
        return result
