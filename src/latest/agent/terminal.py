"""
Terminal agent — legacy ARE simulation wrapper for GAIA2.

GAIA2 now delegates to Terminus2Agent (Harbor containers).
This class is kept for backward compatibility and as a
simplified prompt-only alternative.

Architecture:
  GAIA2 primary path:     Terminus2Agent (Harbor Docker mode)
  GAIA2 fallback path:    TerminalAgent (ARE simulation)
"""
from __future__ import annotations
import time
from .base import BaseAgent


class TerminalAgent(BaseAgent):
    """Legacy ARE simulation agent for GAIA2.

    Now that GAIA2 maps to Terminus2Agent in the runner, this
    class is retained for backward compatibility. It wraps the
    ARE simulation environment (run_gaia2_task_with_are) as a
    prompt-only alternative when Docker is unavailable.
    """

    BENCHMARKS = {"gaia2"}

    def __init__(self, model: str = "deepseek-v4-pro",
                 concurrency: int = 15, timeout: int = 300):
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout

    def supports_benchmark(self, benchmark: str) -> bool:
        return benchmark in self.BENCHMARKS

    async def run_task(self, task: dict, experience_section: str = "",
                       group: str = "A") -> dict:
        """Delegate to the existing GAIA2 ARE runner."""
        from scripts.latest.llm_client import _check_api_error
        from scripts.latest.trace import APIUnavailableError

        # Use the existing ARE runner (imported lazily to avoid circular deps)
        from scripts.latest.latest_runner import run_gaia2_task_with_are
        result = await run_gaia2_task_with_are(task, experience_section, group)
        return result
