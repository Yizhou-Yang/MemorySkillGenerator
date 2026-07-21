"""SkillForge Latest — compatibility re-exports.

The memory core (SkillForgeLatest + experience/gate/analysis/refine/injection/
vgr) moved to the pip-installable `memlayer` package; this package keeps the
old import surface working and still owns the harness-only sub-packages
(agent/, eval/, llm/, safety/, base.py). Sibling modules (src.latest.experience
etc.) are alias shims onto their memlayer counterparts, so classes pickled
under the old paths keep loading.
"""
from __future__ import annotations

from memlayer.forge import (  # noqa: F401
    SkillForgeLatest,
    Experience, ExperienceLibrary, FailureTaxonomy,
    assess_task_complexity, should_augment, classify_task_type,
    build_augmented_prompt, build_skillforge_prompt,
    format_evoarena_patch_log, format_skillforge_patch_log,
    format_success_experience, format_failure_experience,
    format_intermediate_state_patch, format_within_task_patches,
    analyze_execution, classify_failure,
    ai_review_experience, cross_agent_evaluate_skill,
    critic_refine_experience, _format_patch_history,
)
