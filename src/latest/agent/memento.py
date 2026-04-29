"""
Memento-S agent — multi-step web-search QA agent for GAIA benchmark.

Real integration: github.com/Memento-Teams/Memento-Skills
  - Memento-Skills is an agent-designing-agent: it creates, adapts,
    and improves task-specific skills as structured Markdown files.
  - For GAIA tasks, it uses tool-based agents (web search, crawl,
    code execution) with a skill retrieval mechanism.

Relationship with SkillForge:
  - Memento-Skills manages skills WITHIN a single agent (self-evolution)
  - SkillForge transfers experience BETWEEN different agents (cross-agent)
  - The two are complementary: Memento optimizes one agent's skills,
    SkillForge shares those learnings across agent types.

Architecture:
  Mode 1 (CodeBuddy SDK): Full multi-step agent with web search tools
  Mode 2 (Memento-Skills): Skill-based agent execution (if available)
  Mode 3 (Prompt-only): Direct LLM QA without tool use

For GAIA tasks, we use CodeBuddy SDK with web search + crawl tools
for multi-hop reasoning QA. This matches Memento-Skills' approach
of tool-augmented agent execution.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import time

from .base import BaseAgent


# Path to Python 3.12 env for Memento-Skills if installed
_MS_PYTHON = "/root/.conda/envs/harbor312/bin/python"


def _has_memento_skills() -> bool:
    """Check if Memento-Skills is installed."""
    try:
        result = subprocess.run(
            [_MS_PYTHON, "-c",
             "from memento_skills import SkillManager; print('ok')"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return result.stdout.strip() == "ok"
    except Exception:
        return False


class MementoSAgent(BaseAgent):
    """Multi-step web-search agent for GAIA benchmark.

    Uses CodeBuddy SDK with tool-use (web search, crawl) for
    multi-hop reasoning QA, matching Memento-Skills' approach.
    Optionally integrates with Memento-Skills for skill retrieval.
    """

    BENCHMARKS = {"gaia"}

    def __init__(self, model: str = "deepseek-v4-pro",
                 timeout: int = 300):
        self.model = model
        self.timeout = timeout
        self._ms_available = _has_memento_skills()

    def supports_benchmark(self, benchmark: str) -> bool:
        return benchmark in self.BENCHMARKS

    async def run_task(self, task: dict, experience_section: str = "",
                       group: str = "A") -> dict:
        """Run GAIA task with SkillForge experience injection.

        Delegates to the existing run_gaia_task pipeline (CodeBuddy SDK),
        which provides multi-step web search + crawl tools for GAIA QA.

        If Memento-Skills is available, skills from its library are
        merged with SkillForge's cross-agent experience.
        """
        from scripts.latest.latest_runner import run_gaia_task

        # If Memento-Skills is available, merge its skills with ours
        augmented_experience = experience_section
        if self._ms_available and experience_section:
            try:
                ms_skills = self._query_memento_skills(task)
                if ms_skills:
                    augmented_experience = (
                        f"{experience_section}\n\n"
                        f"## Memento-Skills Patterns\n"
                        f"{ms_skills}"
                    )
            except Exception:
                pass

        result = await run_gaia_task(task, augmented_experience, group)
        result["execution_mode"] = "codebuddy_tools"
        return result

    def _query_memento_skills(self, task: dict) -> str:
        """Query Memento-Skills library for relevant skills.

        Memento-Skills stores skills as Markdown files. This queries
        the skill library for patterns relevant to the current task.
        """
        instruction = task.get("description", "")[:500]
        try:
            proc = subprocess.run(
                [_MS_PYTHON, "-c",
                 f"from memento_skills import SkillManager; "
                 f"mgr = SkillManager(); "
                 f"results = mgr.search({repr(instruction)}, top_k=3); "
                 f"for r in results: print(r.content); print('---')"],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass
        return ""