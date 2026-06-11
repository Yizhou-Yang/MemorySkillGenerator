#!/usr/bin/env python3
"""SkillForge Latest — LoCoMo Runner (A-Mem Agent, conversation memory QA)."""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'deepseek-v4-pro'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')


async def run_locomo_task(task: dict, experience_section: str = "",
                           group: str = "A") -> dict:
    """Run LoCoMo task — Group A baseline (no within-task injection)."""
    return await _run_with_agent(task, experience_section, group, within_task_patch_mode=None)


async def run_locomo_task_controlled(task: dict, experience_section: str = "",
                                      group: str = "A",
                                      within_task_patch_mode: str | None = None) -> dict:
    """Run LoCoMo task — Groups B/C (EvoArena within-task injection).

    Args:
        within_task_patch_mode:
            "evoarena" -> B group: plain EvoMem within-task patch injection
            "skillforge" -> C group: failure-aware SkillForge patch routing
    """
    return await _run_with_agent(task, experience_section, group, within_task_patch_mode)


async def _run_with_agent(task: dict, experience_section: str, group: str,
                          within_task_patch_mode: str | None) -> dict:
    """Core execution via AmemAgent with EvoMem injection and locomo-specific QA optimization.

    LoCoMo optimization (vs. generic QA):
    - Temporal reasoning: locomo has temporal questions requiring timeline awareness.
      The agent extracts time references from questions and prioritizes temporally-anchored notes.
    - Multi-hop retrieval: questions may require combining info from multiple conversation turns.
      The agent uses higher retrieve_k (15 vs default 10) for multi-hop queries.
    - Evidence citation: answer must be traceable to a specific conversation segment.
      The agent includes evidence spans in retrieval output for verifiability.
    """
    from src.latest.agent.amem import AmemAgent
    from src.latest.injection import format_within_task_patches

    task_id = task["task_id"]

    # Build within-task patches for B/C groups
    aug = experience_section or ""
    if within_task_patch_mode and group in ("B", "C"):
        patches = format_within_task_patches(
            task.get("description", ""),
            mode=within_task_patch_mode,
            task_id=task_id,
        )
        if patches:
            aug = f"{aug}\n\n{patches}" if aug else patches

    agent = AmemAgent(retrieve_k=15)  # Higher k for locomo multi-hop retrieval
    result = await agent.run_task(task, experience_section=aug, group=group)
    result["_aug_prompt"] = aug
    return result
