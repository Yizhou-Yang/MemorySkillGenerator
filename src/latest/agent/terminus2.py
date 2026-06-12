"""
Terminus 2 agent — Harbor-based terminal execution for Terminal-Bench-Evo and GAIA2.

Real integration: harbor-framework/terminal-bench/agents/terminus_2
  - Harbor CLI: harbor run --dataset terminal-bench@2.0 --agent terminus-2 --model <model>
  - Requires: Docker, harbor Python package (>=3.12), DEEPSEEK_API_KEY

Architecture:
  Mode 1 (Docker): Harbor subprocess — full agentic execution in container
  Mode 2 (CLI):    Direct shell subprocess — runs commands locally (no isolation)
  Mode 3 (Prompt): LLM-only — generates commands without execution (no Docker)

SkillForge role: Injects cross-task experience into the system prompt before
the agent executes, regardless of which mode is active.
"""
from __future__ import annotations
import json as _json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from .base import BaseAgent


# Path to Python 3.12 env where harbor is installed
_HARBOR_PYTHON = "/root/.conda/envs/harbor312/bin/python"


def _has_harbor() -> bool:
    """Check if harbor CLI is available."""
    return shutil.which("harbor") is not None or (
        os.path.exists(_HARBOR_PYTHON)
        and subprocess.run(
            [_HARBOR_PYTHON, "-c", "import harbor; print('ok')"],
            capture_output=True, text=True, timeout=10
        ).stdout.strip() == "ok"
    )


def _has_docker() -> bool:
    """Check if Docker daemon is reachable."""
    return shutil.which("docker") is not None


class Terminus2Agent(BaseAgent):
    """Terminal command agent for Terminal-Bench-Evo and GAIA2.

    Delegates to Harbor framework in Docker mode, falls back to
    local shell execution or prompt-only generation when Docker
    is unavailable.

    Benchmark mapping:
      - terminal_bench_2: Terminal-Bench 2.0 tasks via Harbor
      - gaia2: GAIA CLI tasks executed in a container
    """

    BENCHMARKS = {"terminal_bench_2", "gaia2"}

    def __init__(self, model: str = "deepseek-v4-pro",
                 timeout: int = 300,
                 sandbox_dir: str = "/tmp/terminus2_sandbox"):
        self.model = model
        self.timeout = timeout
        self.sandbox_dir = sandbox_dir
        self._harbor_available = _has_harbor()
        self._docker_available = _has_docker()

    def supports_benchmark(self, benchmark: str) -> bool:
        return benchmark in self.BENCHMARKS

    async def run_task(self, task: dict, experience_section: str = "",
                       group: str = "A") -> dict:
        """Execute a terminal/CLI task with SkillForge experience injection.

        Priority:
          1. Harbor Docker mode — full isolation, real Terminus 2
          2. Local shell mode — runs commands directly (no isolation)
          3. Prompt-only mode — LLM generates command text only
        """
        # Try Harbor Docker mode first
        if self._harbor_available and self._docker_available:
            try:
                result = self._run_via_harbor(task, experience_section, group)
                # Check if harbor produced usable output
                if (result.get("response") and not result.get("error") and
                        result.get("return_code") == 0):
                    return result
                # Harbor ran but produced no output — fall through
            except Exception as e:
                # Fall through to next mode
                pass

        # Try local shell execution
        try:
            return await self._run_via_shell(task, experience_section, group)
        except Exception:
            pass

        # Final fallback: prompt-only
        return await self._run_prompt_only(task, experience_section, group)

    # ------------------------------------------------------------------
    # Mode 1: Harbor Docker execution
    # ------------------------------------------------------------------

    def _run_via_harbor(self, task: dict, experience_section: str,
                        group: str) -> dict:
        """Run task through Harbor framework in Docker container.

        Harbor CLI equivalent:
          harbor run --dataset terminal-bench@2.0 --agent terminus-2 \
            --model deepseek/deepseek-v4-pro --n 1

        For per-task execution, we write the task as a Harbor-compatible
        task directory and run harbor against it.
        """
        from scripts.latest.llm_client import _check_api_error
        from scripts.latest.trace import APIUnavailableError

        task_id = task["task_id"]
        expected = task.get("expected", "")
        instruction = task.get("description", "")

        result = {"task_id": task_id, "expected": expected,
                  "response": "", "error": None, "time_cost": 0,
                  "augmented": bool(experience_section), "group": group,
                  "execution_mode": "harbor_docker"}
        t0 = time.time()

        # Write task to a temp harbor task directory
        task_dir = tempfile.mkdtemp(prefix="harbor_task_")
        try:
            # Write Dockerfile (minimal Ubuntu with bash)
            dockerfile = os.path.join(task_dir, "Dockerfile")
            with open(dockerfile, "w") as f:
                f.write(
                    "FROM ubuntu:22.04\n"
                    "RUN apt-get update && apt-get install -y python3 curl git\n"
                )

            # Write task prompt
            prompt_file = os.path.join(task_dir, "prompt.txt")
            prompt_text = (
                f"## Task\n{instruction}\n\n"
                f"## Expected Output Format\nProvide the exact output.\n"
            )
            if experience_section:
                prompt_text = (
                    f"## SkillForge Experience\n{experience_section}\n\n"
                    + prompt_text
                )
            with open(prompt_file, "w") as f:
                f.write(prompt_text)

            # Write check script (evaluates correctness)
            check_script = os.path.join(task_dir, "check.py")
            with open(check_script, "w") as f:
                f.write(
                    "import sys\n"
                    "output = sys.stdin.read().strip()\n"
                    f"expected = {_json.dumps(expected)}\n"
                    "print(str(output == expected).lower())\n"
                )

            # Run via harbor
            env = os.environ.copy()
            env["DEEPSEEK_API_KEY"] = env.get("DEEPSEEK_API_KEY", "")
            harbor_bin = os.path.join(os.path.dirname(_HARBOR_PYTHON), "harbor")
            if not os.path.exists(harbor_bin):
                harbor_bin = shutil.which("harbor") or "harbor"
            cmd = [
                harbor_bin, "run",
                "--path", task_dir,
                "--agent", "terminus-2",
                "--model", f"deepseek/{self.model}",
                "--n", "1",
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, cwd=task_dir, env=env,
            )
            result["response"] = proc.stdout.strip()
            result["stderr"] = proc.stderr.strip()
            result["return_code"] = proc.returncode
        except subprocess.TimeoutExpired:
            result["error"] = "harbor_timeout"
        except FileNotFoundError:
            result["error"] = "harbor_not_found"
        except Exception as e:
            result["error"] = f"harbor_error: {e}"
        finally:
            # Cleanup
            try:
                shutil.rmtree(task_dir, ignore_errors=True)
            except Exception:
                pass

        result["time_cost"] = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # Mode 2: Local shell execution (no Docker, no isolation)
    # ------------------------------------------------------------------

    async def _run_via_shell(self, task: dict, experience_section: str,
                       group: str) -> dict:
        """Run task by executing commands directly on local shell.

        First generates a single bash command via LLM. If the LLM response
        is reasoning text (not a command) or exceeds turn limits, falls
        through to prompt-only mode by raising an exception.

        No Docker isolation — use with trusted tasks only.
        """
        from scripts.latest.llm_client import _llm_call, _check_api_error
        from scripts.latest.trace import APIUnavailableError

        task_id = task["task_id"]
        instruction = task.get("description", "")
        expected = task.get("expected", "")

        # Set up files
        files = {}
        try:
            files = _json.loads(task.get("context", "{}"))
        except Exception:
            pass

        os.makedirs(self.sandbox_dir, exist_ok=True)
        for fname, fcontent in files.items():
            fpath = os.path.join(self.sandbox_dir, fname)
            with open(fpath, "w") as f:
                f.write(fcontent)

        result = {"task_id": task_id, "expected": expected,
                  "response": "", "error": None, "time_cost": 0,
                  "augmented": bool(experience_section), "group": group,
                  "execution_mode": "local_shell"}
        t0 = time.time()

        # Generate command via LLM
        system = (
            "You are a terminal command agent. Given a task instruction, "
            "output the exact bash command to accomplish it. "
            "Output ONLY the command, nothing else."
        )
        if experience_section:
            system += f"\n\n## SkillForge Experience\n{experience_section}"

        prompt = (
            f"[System]\n{system}\n\n"
            f"Working directory: {self.sandbox_dir}\n"
            f"Task: {instruction}\n\n"
            f"Command:"
        )

        r = await _llm_call(prompt, max_turns=1, timeout=self.timeout)
        if _check_api_error(r):
            raise APIUnavailableError("API unavailable")

        command = r.get("text", "").strip()

        # Detect unusable responses — fall through to prompt-only mode
        if not command or "Max turns" in command or len(command) < 3:
            raise RuntimeError(f"shell_mode_unusable: {command[:100]}")

        result["response"] = command

        # Execute locally
        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.sandbox_dir,
                capture_output=True, text=True, timeout=30,
            )
            result["stdout"] = proc.stdout.strip()
            result["stderr"] = proc.stderr.strip()
            result["return_code"] = proc.returncode
        except subprocess.TimeoutExpired:
            result["error"] = "command_timeout"
        except Exception as e:
            result["error"] = str(e)

        result["time_cost"] = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # Mode 3: Prompt-only (no execution)
    # ------------------------------------------------------------------

    async def _run_prompt_only(self, task: dict, experience_section: str,
                               group: str) -> dict:
        """Solve task via multi-turn LLM reasoning — no code execution.

        Used when neither Docker nor shell execution is available.
        The LLM reasons through the problem with multiple turns (up to 5),
        then outputs the final answer. The response is compared against
        the expected output string by the benchmark evaluator.

        For Terminal-Bench-2.0 coding tasks, the LLM is expected to:
        1. Analyze the problem requirements
        2. Reason about the solution approach
        3. Produce the correct output/code/answer
        """
        from scripts.latest.llm_client import _llm_call_notool, _check_api_error
        from scripts.latest.trace import APIUnavailableError

        task_id = task["task_id"]
        instruction = task.get("description", "")
        expected = task.get("expected", "")

        result = {"task_id": task_id, "expected": expected,
                  "response": "", "error": None, "time_cost": 0,
                  "augmented": bool(experience_section), "group": group,
                  "execution_mode": "prompt_only"}
        t0 = time.time()

        system = (
            "You are an expert software engineer and systems administrator. "
            "Given a technical task, solve it by reasoning step by step "
            "and producing the exact correct output.\n\n"
            "For coding tasks: write the complete solution with all necessary code.\n"
            "For terminal tasks: produce the exact output the correct command would generate.\n"
            "For data analysis: show your work and give the precise answer.\n\n"
            "Important: Output ONLY the final result on the last line. "
            "The last line of your response will be compared exactly against the expected answer."
        )
        if experience_section:
            system += f"\n\n## SkillForge Experience\n{experience_section}"

        user_prompt = (
            f"Task: {instruction}\n\n"
            f"Solve this task. Show your reasoning, then output the final answer on the last line."
        )

        r = await _llm_call_notool(system, user_prompt, timeout=self.timeout)
        if _check_api_error(r):
            raise APIUnavailableError("API unavailable")

        result["response"] = r.get("text", "").strip()
        result["error"] = r.get("error")
        result["time_cost"] = time.time() - t0
        return result