#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Health Experiment — validates the "Agent Skill Bloat to Refactoring"
article's engineering solutions against SkillForge's skill system.

Inspired by: snowsyzheng (2026-05-12) "Agent Skill Bloat to Refactoring"

Core hypothesis: Applying the article's anti-bloat principles to skill
formatting and evolution will improve downstream EM/F1 on held-out tasks.

Experiments:
1. Skill Ordering: critical rules first (primacy effect) vs random order
2. Positive Rewrite: convert negative constraints to positive instructions
3. Skill Compaction: merge redundant skills before evaluation
4. Conflict Detection: identify and resolve conflicting skill rules
5. Structural Format: table format vs prose format for skill presentation
6. Length Budget: enforce max skill length, measure impact

Paper reference values (from systematic benchmark):
  HotpotQA (held-out): EM=70.0%, F1 varies by variant
  LongMemEval: F1=0.247

Key insight from article:
  "Attention has fixed bandwidth. The more content you stuff into a skill,
   the less attention each instruction gets. Repeating a rule 5 times
   = stealing attention quota from other rules."
"""
from __future__ import annotations

import json
import string
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.utils.prompt_health import PromptHealthMonitor, check_prompt_health, format_health_report
from src.models import Skill
from src.memory.compressor import create_compressor
from src.skill_induction.factory import create_inducer
from src.trajectory.collector import TrajectoryCollector
from src.models import TransformVariant

# Metrics (same as systematic benchmark)

def compute_em(prediction: str, ground_truth: str) -> float:
    def normalize(s):
        s = s.lower().strip()
        for article in ['a ', 'an ', 'the ']:
            if s.startswith(article):
                s = s[len(article):]
        s = s.translate(str.maketrans('', '', string.punctuation))
        return s.strip()
    return 1.0 if normalize(ground_truth) in normalize(prediction) else 0.0

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    if not ground_truth.strip():
        return 1.0 if not prediction.strip() else 0.0
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(gt_tokens) & Counter(pred_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

# Skill Formatting Strategies (Article Solutions 1, 3, 4, 5)

def format_skill_baseline(skill: Skill) -> str:
    """Baseline: standard formatting (current approach)."""
    parts = [f"## Skill: {skill.name}", skill.description, ""]
    if skill.procedure:
        parts.append("**Procedure:**")
        for i, step in enumerate(skill.procedure, 1):
            parts.append(f"{i}. {step}")
        parts.append("")
    if skill.constraints:
        parts.append("**Constraints:**")
        for c in skill.constraints:
            parts.append(f"- {c}")
    if skill.rules:
        parts.append("**Rules:**")
        for r in skill.rules:
            parts.append(f"- {r}")
    return "\n".join(parts)

def format_skill_layered(skill: Skill) -> str:
    """
    Article Solution 1: Layered architecture — position = priority.

    Critical constraints FIRST (primacy effect),
    procedure in the middle,
    output format at the end (recency effect).
    """
    parts = []

    # Layer 1: Critical constraints FIRST (highest attention zone)
    if skill.constraints or skill.rules:
        parts.append("⚠️ CRITICAL RULES (must follow):")
        for c in skill.constraints:
            parts.append(f"  • {c}")
        for r in skill.rules:
            parts.append(f"  • {r}")
        parts.append("")

    # Layer 2: Core methodology
    parts.append(f"## {skill.name}")
    parts.append(skill.description)
    parts.append("")

    # Layer 3: Procedure (middle zone — can tolerate some loss)
    if skill.procedure:
        parts.append("Steps:")
        for i, step in enumerate(skill.procedure, 1):
            parts.append(f"  {i}. {step}")
        parts.append("")

    # Layer 4: Reminder at end (recency effect — sandwich)
    if skill.constraints:
        parts.append(f"Remember: {skill.constraints[0]}")

    return "\n".join(parts)

def format_skill_positive(skill: Skill) -> str:
    """
    Article Solution 3: Positive instructions replace negations.

    Convert "Do not X" → "Instead of X, do Y"
    Convert "Never X" → "Always do Y instead"
    """
    import re

    def rewrite_negative(text: str) -> str:
        """Attempt to rewrite negative instructions as positive ones."""
        # Pattern: "Do not X" → "Focus on Y instead"
        text = re.sub(
            r"(?i)\bdo\s+not\b\s+(.+)",
            lambda m: f"Focus on the opposite of: {m.group(1)}",
            text
        )
        text = re.sub(
            r"(?i)\bnever\b\s+(.+)",
            lambda m: f"Always verify before: {m.group(1)}",
            text
        )
        text = re.sub(
            r"(?i)\bavoid\b\s+(.+)",
            lambda m: f"Prefer alternatives to: {m.group(1)}",
            text
        )
        return text

    parts = [f"## Skill: {skill.name}", skill.description, ""]
    if skill.procedure:
        parts.append("**What to do:**")
        for i, step in enumerate(skill.procedure, 1):
            parts.append(f"{i}. {step}")
        parts.append("")

    # Convert constraints to positive table format
    if skill.constraints:
        parts.append("| Scenario | ✅ Do This |")
        parts.append("|----------|-----------|")
        for c in skill.constraints:
            positive = rewrite_negative(c)
            parts.append(f"| When applicable | {positive} |")

    return "\n".join(parts)

def format_skill_table(skill: Skill) -> str:
    """
    Article Solution 4: Structured table format.

    Tables are easier for LLMs to "scan" than prose.
    """
    parts = [f"## {skill.name}", ""]

    # Description as one-liner
    parts.append(f"**Purpose:** {skill.description}")
    parts.append("")

    # Procedure as numbered table
    if skill.procedure:
        parts.append("| Step | Action |")
        parts.append("|------|--------|")
        for i, step in enumerate(skill.procedure, 1):
            parts.append(f"| {i} | {step} |")
        parts.append("")

    # Constraints as checklist
    if skill.constraints:
        parts.append("**Checklist:** " + " | ".join(f"☐ {c}" for c in skill.constraints))

    return "\n".join(parts)

def format_skill_compact(skill: Skill) -> str:
    """
    Article Solution 6: Externalize — keep only essential rules.

    Remove verbose descriptions, keep only actionable steps.
    Target: < 150 tokens per skill.
    """
    parts = [f"{skill.name}: {skill.description[:80]}"]
    if skill.procedure:
        # Only keep first 4 steps
        for i, step in enumerate(skill.procedure[:4], 1):
            parts.append(f"  {i}. {step[:60]}")
    if skill.constraints:
        parts.append(f"  ⚠️ {skill.constraints[0][:60]}")
    return "\n".join(parts)

# Skill Library Formatting (with ordering strategies)

def format_library_random(skills: list[Skill], formatter) -> str:
    """Random order (baseline)."""
    import random
    shuffled = skills.copy()
    random.shuffle(shuffled)
    return "\n\n---\n\n".join(formatter(s) for s in shuffled)

def format_library_priority(skills: list[Skill], formatter) -> str:
    """Priority order: most constrained/important skills first."""
    # Sort by: number of constraints (more = more critical) + procedure length
    sorted_skills = sorted(
        skills,
        key=lambda s: len(s.constraints) + len(s.rules),
        reverse=True,
    )
    return "\n\n---\n\n".join(formatter(s) for s in sorted_skills)

def format_library_sandwich(skills: list[Skill], formatter) -> str:
    """
    Article Solution 5: Instruction sandwich.

    Most important skill at START and END (primacy + recency).
    Less important skills in the middle.
    """
    if len(skills) <= 2:
        return "\n\n---\n\n".join(formatter(s) for s in skills)

    # Sort by importance
    sorted_skills = sorted(
        skills,
        key=lambda s: len(s.constraints) + len(s.rules),
        reverse=True,
    )

    # Sandwich: most important first, second-most important last
    result = [sorted_skills[0]]  # Most important at start
    result.extend(sorted_skills[2:])  # Middle
    result.append(sorted_skills[1])  # Second-most important at end

    return "\n\n---\n\n".join(formatter(s) for s in result)

# Experiment Runner

def run_formatting_experiment(
    llm_client: LLMClient,
    skills: list[Skill],
    test_tasks: list[dict],
    experiment_name: str,
    formatter,
    library_formatter=format_library_priority,
) -> dict:
    """Run a single formatting experiment on held-out test tasks."""
    logger.info(f"\n  [{experiment_name}] Running with {len(skills)} skills on {len(test_tasks)} tasks...")

    skill_library = library_formatter(skills, formatter)

    # Check prompt health
    report = check_prompt_health(skill_library)

    em_scores = []
    f1_scores = []

    for task in test_tasks:
        desc = task["description"]
        expected = task.get("expected", "")

        messages = [
            {"role": "system", "content": (
                "You have access to a library of learned skills. "
                "Use the most relevant skill(s) to answer the question.\n\n"
                f"=== SKILL LIBRARY ===\n{skill_library}\n=== END ===\n\n"
                "Answer the question directly and concisely. "
                "Give only the answer, no explanation."
            )},
            {"role": "user", "content": desc},
        ]

        try:
            resp = llm_client.chat(messages, temperature=0.3, max_tokens=256)
            em = compute_em(resp, expected)
            f1 = compute_token_f1(resp, expected)
            em_scores.append(em)
            f1_scores.append(f1)
        except Exception as exc:
            logger.error(f"    Task failed: {exc}")
            em_scores.append(0.0)
            f1_scores.append(0.0)

    result = {
        "experiment": experiment_name,
        "avg_em": avg(em_scores),
        "avg_f1": avg(f1_scores),
        "em_scores": em_scores,
        "f1_scores": f1_scores,
        "prompt_health": {
            "score": report.score,
            "lines": report.total_lines,
            "tokens": report.estimated_tokens,
            "issues": len(report.issues),
            "is_healthy": report.is_healthy,
        },
        "library_chars": len(skill_library),
    }

    logger.info(
        f"  [{experiment_name}] EM={result['avg_em']:.1%}, F1={result['avg_f1']:.3f}, "
        f"health={report.score:.2f}, tokens≈{report.estimated_tokens}"
    )
    return result

# Skill Compaction Experiment (Article Solution 8)

def compact_skill_bank(skills: list[Skill], llm_client: LLMClient) -> list[Skill]:
    """
    Apply compaction to a skill bank: merge semantically similar skills.

    This implements the article's "anti-bloat brake" at the skill level:
    - Detect near-duplicate skills (Jaccard similarity > 0.6)
    - Merge them into one more abstract skill
    - Enforce a max skill count budget
    """
    if len(skills) <= 3:
        return skills

    # Compute pairwise similarity
    def skill_similarity(a: Skill, b: Skill) -> float:
        text_a = f"{a.name} {a.description} {' '.join(a.procedure)}".lower()
        text_b = f"{b.name} {b.description} {' '.join(b.procedure)}".lower()
        tokens_a = set(text_a.split())
        tokens_b = set(text_b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    # Greedy merge: find pairs with similarity > threshold
    merged = []
    used = set()
    threshold = 0.4

    for i, skill_a in enumerate(skills):
        if i in used:
            continue
        cluster = [skill_a]
        used.add(i)

        for j in range(i + 1, len(skills)):
            if j in used:
                continue
            if skill_similarity(skill_a, skills[j]) > threshold:
                cluster.append(skills[j])
                used.add(j)

        if len(cluster) == 1:
            merged.append(cluster[0])
        else:
            # Merge cluster into one skill
            merged_skill = Skill(
                name=cluster[0].name,
                description=cluster[0].description,
                procedure=cluster[0].procedure,
                constraints=list(set(
                    c for s in cluster for c in s.constraints
                ))[:3],  # Deduplicate and limit
                facts=list(set(
                    f for s in cluster for f in s.facts
                ))[:3],
                rules=list(set(
                    r for s in cluster for r in s.rules
                ))[:3],
                metadata={"compacted_from": len(cluster)},
            )
            merged.append(merged_skill)

    logger.info(f"  [Compaction] {len(skills)} skills → {len(merged)} skills")
    return merged

# Main Experiment

def main():
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
    log_path = Path("experiments/prompt_health_experiment.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}", level="DEBUG")

    logger.info("=" * 70)
    logger.info("Prompt Health Experiment — Article Solutions Validation")
    logger.info("=" * 70)
    logger.info("Hypothesis: Anti-bloat formatting improves skill-guided EM/F1")
    logger.info("")

    load_env()
    start_time = time.time()
    llm_client = LLMClient({"temperature": 0.7, "max_tokens": 2048, "timeout": 120})

    # API test
    resp = llm_client.chat([{"role": "user", "content": "Say OK"}], temperature=0.0, max_tokens=5)
    logger.info(f"API: '{resp.strip()}' ✅")

    # ---- Phase 1: Induce skills from train tasks ----
    logger.info("\n" + "─" * 50)
    logger.info("Phase 1: Skill Induction (5 train tasks)")
    logger.info("─" * 50)

    from benchmarks.loader import BenchmarkLoader
    loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": 12})
    all_tasks = loader.load()
    train_tasks = all_tasks[:5]
    test_tasks = all_tasks[5:]
    logger.info(f"Tasks: {len(train_tasks)} train, {len(test_tasks)} test")

    collector = TrajectoryCollector(llm_client, {"max_steps": 4})
    compressor = create_compressor("mem0", llm_client, {})

    skills = []
    for idx, task in enumerate(train_tasks):
        logger.info(f"  Train {idx+1}/{len(train_tasks)}: {task['task_id']}")
        try:
            traj = collector.collect(task_id=task["task_id"], task_description=task["description"])
            memory = compressor.compress(traj)
            inducer = create_inducer(TransformVariant.HYBRID_TO_SKILL, llm_client, {})
            skill = inducer.induce(trajectory=traj, memory=memory)
            skills.append(skill)
            logger.info(f"    → '{skill.name}' ({len(skill.procedure)} steps, {len(skill.constraints)} constraints)")
        except Exception as exc:
            logger.error(f"    Failed: {exc}")

    logger.info(f"\nInduced {len(skills)} skills total")

    # ---- Phase 2: Run formatting experiments ----
    logger.info("\n" + "─" * 50)
    logger.info("Phase 2: Formatting Experiments (held-out test)")
    logger.info("─" * 50)

    experiments = []

    # Experiment 1: Baseline (current approach)
    experiments.append(run_formatting_experiment(
        llm_client, skills, test_tasks,
        "baseline_standard",
        format_skill_baseline,
        format_library_priority,
    ))

    # Experiment 2: Layered architecture (Solution 1)
    experiments.append(run_formatting_experiment(
        llm_client, skills, test_tasks,
        "layered_priority_first",
        format_skill_layered,
        format_library_priority,
    ))

    # Experiment 3: Positive instructions (Solution 3)
    experiments.append(run_formatting_experiment(
        llm_client, skills, test_tasks,
        "positive_instructions",
        format_skill_positive,
        format_library_priority,
    ))

    # Experiment 4: Table format (Solution 4)
    experiments.append(run_formatting_experiment(
        llm_client, skills, test_tasks,
        "table_format",
        format_skill_table,
        format_library_priority,
    ))

    # Experiment 5: Compact format (Solution 6)
    experiments.append(run_formatting_experiment(
        llm_client, skills, test_tasks,
        "compact_minimal",
        format_skill_compact,
        format_library_priority,
    ))

    # Experiment 6: Sandwich ordering (Solution 5)
    experiments.append(run_formatting_experiment(
        llm_client, skills, test_tasks,
        "sandwich_ordering",
        format_skill_baseline,
        format_library_sandwich,
    ))

    # Experiment 7: Random ordering (control)
    experiments.append(run_formatting_experiment(
        llm_client, skills, test_tasks,
        "random_ordering",
        format_skill_baseline,
        format_library_random,
    ))

    # Experiment 8: Compacted skill bank (Solution 8)
    compacted_skills = compact_skill_bank(skills, llm_client)
    experiments.append(run_formatting_experiment(
        llm_client, compacted_skills, test_tasks,
        "compacted_bank",
        format_skill_layered,
        format_library_priority,
    ))

    # ---- Phase 3: Results Analysis ----
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS: Prompt Health Experiment")
    logger.info("=" * 70)

    logger.info(f"\n{'Experiment':<25} {'EM':<8} {'F1':<8} {'Health':<8} {'Tokens':<8} {'Issues':<8}")
    logger.info("-" * 65)

    best_em = 0
    best_exp = ""
    for exp in experiments:
        em_str = f"{exp['avg_em']:.1%}"
        f1_str = f"{exp['avg_f1']:.3f}"
        health_str = f"{exp['prompt_health']['score']:.2f}"
        tokens_str = f"{exp['prompt_health']['tokens']}"
        issues_str = f"{exp['prompt_health']['issues']}"
        logger.info(f"{exp['experiment']:<25} {em_str:<8} {f1_str:<8} {health_str:<8} {tokens_str:<8} {issues_str:<8}")
        if exp['avg_em'] > best_em:
            best_em = exp['avg_em']
            best_exp = exp['experiment']

    logger.info(f"\n🏆 Best: {best_exp} (EM={best_em:.1%})")

    # ---- Article Predictions vs Results ----
    logger.info("\n" + "─" * 50)
    logger.info("Article Predictions vs Actual Results")
    logger.info("─" * 50)

    baseline = next(e for e in experiments if e["experiment"] == "baseline_standard")
    predictions = [
        ("Layered > Baseline", "layered_priority_first",
         "Critical rules first gets more attention (primacy effect)"),
        ("Compact < Baseline tokens", "compact_minimal",
         "Fewer tokens = more attention per instruction"),
        ("Table ≈ Baseline EM", "table_format",
         "Structured format easier to scan but same info"),
        ("Random < Priority ordering", "random_ordering",
         "Priority ordering leverages primacy effect"),
    ]

    for prediction, exp_name, reasoning in predictions:
        exp = next((e for e in experiments if e["experiment"] == exp_name), None)
        if exp:
            em_diff = exp["avg_em"] - baseline["avg_em"]
            token_diff = exp["prompt_health"]["tokens"] - baseline["prompt_health"]["tokens"]
            status = "✅" if (
                (">" in prediction and em_diff > 0) or
                ("<" in prediction and (em_diff < 0 or token_diff < 0)) or
                ("≈" in prediction and abs(em_diff) < 0.15)
            ) else "❓"
            logger.info(f"  {status} {prediction}")
            logger.info(f"     Actual: EM diff={em_diff:+.1%}, token diff={token_diff:+d}")
            logger.info(f"     Reasoning: {reasoning}")

    # ---- Token Usage ----
    elapsed = time.time() - start_time
    stats = llm_client.stats
    logger.info(f"\n💰 Token Usage:")
    logger.info(f"  API calls: {stats['total_calls']}")
    logger.info(f"  Total tokens: {stats['total_tokens']:,}")
    logger.info(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save results
    output_path = Path("experiments/prompt_health_experiment_results.json")
    output = {
        "experiments": experiments,
        "skills_induced": len(skills),
        "skills_compacted": len(compacted_skills),
        "test_tasks": len(test_tasks),
        "elapsed_seconds": elapsed,
        "total_tokens": stats["total_tokens"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\n  Results saved to: {output_path}")

if __name__ == "__main__":
    main()
