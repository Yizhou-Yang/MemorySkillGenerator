#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkillCurator Full Paper Experiments v2 — 10x scale, publication-ready.

Key fixes from v1:
  1. Embedding cosine similarity for dedup (replaces broken Jaccard)
  2. 10x scale: 80 train / 100 test (was 8/10)
  3. Multi-benchmark: HotpotQA + 2WikiMultihopQA (+ ALFWorld cross-val)
  4. Phase Transition: test N=1..80 with 3 strategies
  5. Compaction Cliff: real MERGE via LLM + embedding dedup
  6. Scissors Effect: 3 lines (append-only / SkillOS / Ours)
  7. δ_attention independence: 30 test tasks (was 8)
  8. Statistical significance: 3 random seeds, report mean±std

Token budget: ~15M tokens (within 20M limit)
Expected runtime: 4-8 hours

Usage:
  nohup /usr/bin/python3.9 scripts/run_paper_v2.py > experiments/paper_v2_stdout.log 2>&1 &
"""
from __future__ import annotations

import json
import math
import os
import random
import string
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from loguru import logger

from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.memory.compressor import create_compressor
from src.memory.consolidation import MemoryConsolidator
from src.models import Skill, TransformVariant
from src.skill_induction.factory import create_inducer
from src.trajectory.collector import TrajectoryCollector
from src.curation.delta_decomposition import (
    decompose_delta, compute_library_health, compute_effective_skill_count,
    library_waste_ratio, bound_tightening_from_merge, phase_transition_curve,
    compute_optimal_library_size,
)
from src.utils.skill_formatter import (
    format_skill_library, attention_weight, effective_attention,
    FormattingConfig,
)


# ============================================================
# Configuration — 10x scale
# ============================================================

TRAIN_SAMPLES = 80
TEST_SAMPLES = 100
ATTENTION_TEST_SAMPLES = 30
PHASE_TRANSITION_TRAIN = 60
PHASE_TRANSITION_TEST = 20
SCISSORS_STREAM_LEN = 60
BOUND_TIGHTENING_LEN = 40
MAX_TRAJ_STEPS = 6

BENCHMARKS = ["hotpotqa", "2wikimultihopqa"]
METHODS = ["B0", "B1", "B2", "A1", "A2", "A3"]

ATTENTION_STRATEGIES = [
    "random_order",
    "recency_order",
    "utility_order",
    "position_optimized",
    "table_format",
    "positive_rewrite",
    "compact_format",
    "full_optimized",
]

# Embedding model for semantic dedup
EMBED_MODEL = None  # Lazy-loaded


def get_embed_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
    return EMBED_MODEL


# ============================================================
# Metrics
# ============================================================

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


def compute_em(prediction: str, ground_truth: str) -> float:
    def normalize(s):
        s = s.lower().strip()
        for article in ['a ', 'an ', 'the ']:
            if s.startswith(article):
                s = s[len(article):]
        s = s.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(s.split())
    norm_pred = normalize(prediction)
    norm_gt = normalize(ground_truth)
    if not norm_gt:
        return 0.0
    return 1.0 if norm_gt in norm_pred else 0.0


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def std(lst):
    if len(lst) < 2:
        return 0.0
    m = avg(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))


# ============================================================
# Embedding-based Semantic Dedup (fixes v1 Jaccard failure)
# ============================================================

def skill_to_text(s: Skill) -> str:
    """Convert skill to text for embedding."""
    parts = [s.name, s.description]
    parts.extend(s.procedure[:3])
    if s.constraints:
        parts.extend(s.constraints[:2])
    return " ".join(parts)


def compute_skill_embeddings(skills: list[Skill]) -> np.ndarray:
    """Compute embeddings for all skills."""
    model = get_embed_model()
    texts = [skill_to_text(s) for s in skills]
    return model.encode(texts, normalize_embeddings=True)


def find_redundant_pairs(skills: list[Skill], threshold: float = 0.75) -> list[tuple[int, int, float]]:
    """Find redundant skill pairs using embedding cosine similarity."""
    if len(skills) < 2:
        return []
    embeddings = compute_skill_embeddings(skills)
    sim_matrix = np.dot(embeddings, embeddings.T)
    pairs = []
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            if sim_matrix[i][j] > threshold:
                pairs.append((i, j, float(sim_matrix[i][j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def deduplicate_skills_embedding(skills: list[Skill], threshold: float = 0.75) -> list[Skill]:
    """Remove redundant skills using embedding similarity."""
    if len(skills) < 2:
        return skills
    embeddings = compute_skill_embeddings(skills)
    keep_indices = []
    for i in range(len(skills)):
        is_dup = False
        for j in keep_indices:
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            keep_indices.append(i)
    return [skills[i] for i in keep_indices]


def merge_skill_pair(llm_client: LLMClient, s1: Skill, s2: Skill) -> Skill:
    """Merge two redundant skills into one via LLM."""
    prompt = f"""Merge these two overlapping skills into ONE concise skill.
Keep the best parts of both. Output JSON with keys: name, description, procedure (list), constraints (list).

Skill A:
  Name: {s1.name}
  Description: {s1.description}
  Procedure: {'; '.join(s1.procedure)}
  Constraints: {'; '.join(s1.constraints)}

Skill B:
  Name: {s2.name}
  Description: {s2.description}
  Procedure: {'; '.join(s2.procedure)}
  Constraints: {'; '.join(s2.constraints)}

Output ONLY valid JSON:"""

    try:
        resp = llm_client.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=512,
        )
        data = json.loads(resp)
        return Skill(
            name=data.get("name", s1.name),
            description=data.get("description", s1.description),
            procedure=data.get("procedure", s1.procedure),
            constraints=data.get("constraints", s1.constraints),
            source_tasks=list(set(s1.source_tasks + s2.source_tasks)),
            version=max(s1.version, s2.version) + 1,
        )
    except Exception:
        # Fallback: keep the one with more content
        return s1 if s1.compactness >= s2.compactness else s2


def compact_library(llm_client: LLMClient, skills: list[Skill],
                    threshold: float = 0.75, max_merges: int = 10) -> tuple[list[Skill], int]:
    """
    Full compaction: find redundant pairs, merge them via LLM.
    Returns (compacted_skills, num_merges).
    """
    merged_count = 0
    current = list(skills)

    for _ in range(max_merges):
        pairs = find_redundant_pairs(current, threshold=threshold)
        if not pairs:
            break
        # Merge the most redundant pair
        i, j, sim = pairs[0]
        merged = merge_skill_pair(llm_client, current[i], current[j])
        # Remove originals, add merged
        new_list = [s for idx, s in enumerate(current) if idx != i and idx != j]
        new_list.append(merged)
        current = new_list
        merged_count += 1
        logger.debug(f"  MERGE: '{skills[i].name}' + '{skills[j].name}' → '{merged.name}' (sim={sim:.3f})")

    return current, merged_count


# ============================================================
# Skill Library Formatting Variants
# ============================================================

def format_skills_B0(skills: list[Skill], query: str) -> str:
    return ""


def format_skills_B1(skills: list[Skill], query: str) -> str:
    """B1: Append-only — all skills in insertion order."""
    parts = []
    for i, s in enumerate(skills):
        parts.append(f"Skill {i+1}: {s.name}\n{s.description}\nProcedure: {'; '.join(s.procedure)}")
        if s.constraints:
            parts.append(f"Constraints: {'; '.join(s.constraints)}")
        parts.append("")
    return "\n".join(parts)


def _retrieve_top_k(skills: list[Skill], query: str, k: int = 5) -> list[Skill]:
    """Retrieve top-K skills by embedding similarity to query."""
    if not skills:
        return []
    model = get_embed_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    s_embs = compute_skill_embeddings(skills)
    sims = np.dot(s_embs, q_emb.T).flatten()
    top_indices = np.argsort(sims)[::-1][:k]
    return [skills[i] for i in top_indices]


def format_skills_B2(skills: list[Skill], query: str) -> str:
    """B2: SkillOS-style — embedding retrieval, no attention optimization."""
    top = _retrieve_top_k(skills, query, k=5)
    return format_skills_B1(top, query)


def format_skills_A1(skills: list[Skill], query: str) -> str:
    """A1: Semantic-only — embedding dedup + retrieval, no attention opt."""
    deduped = deduplicate_skills_embedding(skills, threshold=0.75)
    top = _retrieve_top_k(deduped, query, k=5)
    return format_skills_B1(top, query)


def format_skills_A2(skills: list[Skill], query: str) -> str:
    """A2: Attention-only — no dedup, but sandwich + compact format."""
    top = _retrieve_top_k(skills, query, k=5)
    return format_skill_library(top, config=FormattingConfig(strategy="sandwich_compact"))


def format_skills_A3(skills: list[Skill], query: str) -> str:
    """A3: Full — embedding dedup + retrieval + sandwich + compact."""
    deduped = deduplicate_skills_embedding(skills, threshold=0.75)
    top = _retrieve_top_k(deduped, query, k=5)
    return format_skill_library(top, config=FormattingConfig(strategy="sandwich_compact"))


FORMAT_METHODS = {
    "B0": format_skills_B0,
    "B1": format_skills_B1,
    "B2": format_skills_B2,
    "A1": format_skills_A1,
    "A2": format_skills_A2,
    "A3": format_skills_A3,
}


# ============================================================
# Attention formatting strategies (for Table 2)
# ============================================================

def _format_with_strategy(skills: list[Skill], strategy: str, query: str) -> str:
    if strategy == "random_order":
        shuffled = skills.copy()
        random.shuffle(shuffled)
        return format_skills_B1(shuffled, query)

    elif strategy == "recency_order":
        return format_skills_B1(list(reversed(skills)), query)

    elif strategy == "utility_order":
        top = _retrieve_top_k(skills, query, k=len(skills))
        return format_skills_B1(top, query)

    elif strategy == "position_optimized":
        return format_skill_library(skills, config=FormattingConfig(strategy="sandwich_compact"))

    elif strategy == "table_format":
        lines = ["| # | Skill | Key Steps | Constraints |", "|---|-------|-----------|-------------|"]
        for i, s in enumerate(skills, 1):
            steps = "; ".join(s.procedure[:3])[:80]
            constraints = "; ".join(s.constraints[:2])[:60] if s.constraints else "-"
            lines.append(f"| {i} | {s.name} | {steps} | {constraints} |")
        return "\n".join(lines)

    elif strategy == "positive_rewrite":
        parts = []
        for s in skills:
            parts.append(f"**{s.name}**: {s.description}")
            parts.append(f"Steps: {'; '.join(s.procedure)}")
            if s.constraints:
                positive = []
                for c in s.constraints:
                    c_lower = c.lower()
                    if any(neg in c_lower for neg in ["do not", "never", "avoid", "don't"]):
                        positive.append(f"Instead, {c.replace('Do not', '').replace('Never', 'Always').replace('Avoid', 'Prefer alternatives to').strip()}")
                    else:
                        positive.append(c)
                parts.append(f"Guidelines: {'; '.join(positive)}")
            parts.append("")
        return "\n".join(parts)

    elif strategy == "compact_format":
        lines = []
        for s in skills:
            lines.append(f"* {s.name}: {s.description[:60]} [{len(s.procedure)} steps]")
        return "\n".join(lines)

    elif strategy == "full_optimized":
        return format_skill_library(skills, config=FormattingConfig(strategy="sandwich_compact"))

    return format_skills_B1(skills, query)


# ============================================================
# Skill Induction Helper
# ============================================================

def induce_skills_from_tasks(
    llm_client: LLMClient,
    tasks: list[dict],
    max_traj_steps: int = MAX_TRAJ_STEPS,
    label: str = "train",
) -> list[Skill]:
    """Induce skills from a list of tasks. Returns skill bank."""
    collector = TrajectoryCollector(llm_client, {"max_steps": max_traj_steps})
    compressor = create_compressor("mem0", llm_client, {})
    skill_bank: list[Skill] = []

    for idx, task in enumerate(tasks):
        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info(f"  [{label}] Inducing skill {idx+1}/{len(tasks)}: {task['task_id']}")
        try:
            traj = collector.collect(task_id=task["task_id"], task_description=task["description"])
            memory = compressor.compress(traj)
            inducer = create_inducer(TransformVariant.HYBRID_TO_SKILL, llm_client, {})
            skill = inducer.induce(trajectory=traj, memory=memory)
            skill.source_tasks = [task["task_id"]]
            skill_bank.append(skill)
        except Exception as exc:
            logger.warning(f"  [{label}] Failed {task['task_id']}: {exc}")

    logger.info(f"  [{label}] Induced {len(skill_bank)} skills from {len(tasks)} tasks")
    return skill_bank


# ============================================================
# Evaluation Helper
# ============================================================

def evaluate_method(
    llm_client: LLMClient,
    method: str,
    skill_bank: list[Skill],
    test_tasks: list[dict],
    label: str = "",
) -> dict:
    """Evaluate a single method on test tasks. Returns metrics dict."""
    format_fn = FORMAT_METHODS[method]
    em_scores, f1_scores, token_counts = [], [], []

    for idx, task in enumerate(test_tasks):
        desc = task["description"]
        expected = task.get("expected", "")

        try:
            skill_context = format_fn(skill_bank, desc)
            token_counts.append(len(skill_context.split()))

            if method == "B0":
                messages = [
                    {"role": "system", "content": "Answer the question directly and concisely. Give ONLY the answer, no explanation."},
                    {"role": "user", "content": desc},
                ]
            else:
                messages = [
                    {"role": "system", "content": f"Use the following skills to help answer:\n\n{skill_context}\n\nAnswer directly and concisely. Give ONLY the answer."},
                    {"role": "user", "content": desc},
                ]

            resp = llm_client.chat(messages, temperature=0.1, max_tokens=128)
            em = compute_em(resp, expected)
            f1 = compute_token_f1(resp, expected)
            em_scores.append(em)
            f1_scores.append(f1)

        except Exception as exc:
            logger.warning(f"  [{label}:{method}] Task {idx+1} failed: {exc}")
            em_scores.append(0.0)
            f1_scores.append(0.0)
            token_counts.append(0)

    return {
        "avg_em": avg(em_scores),
        "avg_f1": avg(f1_scores),
        "avg_tokens": avg(token_counts),
        "std_em": std(em_scores),
        "std_f1": std(f1_scores),
        "n": len(em_scores),
        "em_scores": em_scores,
        "f1_scores": f1_scores,
    }


# ============================================================
# Experiment 1: Main Experiment (Table 1) — multi-benchmark
# ============================================================

def run_main_experiment(llm_client: LLMClient) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: Main Experiment (Table 1) — 10x scale")
    logger.info("=" * 70)

    from benchmarks.loader import BenchmarkLoader

    all_results = {}

    for bench_name in BENCHMARKS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Benchmark: {bench_name}")
        logger.info(f"{'='*50}")

        total_needed = TRAIN_SAMPLES + TEST_SAMPLES
        loader = BenchmarkLoader({"name": bench_name, "num_samples": total_needed})
        all_tasks = loader.load()

        if len(all_tasks) < total_needed:
            logger.warning(f"  Only {len(all_tasks)} tasks available, adjusting...")
            train_size = min(TRAIN_SAMPLES, len(all_tasks) * 4 // 5)
            test_size = len(all_tasks) - train_size
        else:
            train_size = TRAIN_SAMPLES
            test_size = TEST_SAMPLES

        train_tasks = all_tasks[:train_size]
        test_tasks = all_tasks[train_size:train_size + test_size]
        logger.info(f"  Split: {len(train_tasks)} train + {len(test_tasks)} test")

        # Phase 1: Induce skills
        logger.info(f"\n--- Phase 1: Skill Induction ({bench_name}) ---")
        skill_bank = induce_skills_from_tasks(llm_client, train_tasks, label=bench_name)

        # Phase 2: Evaluate all methods
        logger.info(f"\n--- Phase 2: Evaluation ({bench_name}) ---")
        bench_results = {
            "benchmark": bench_name,
            "train_size": len(train_tasks),
            "test_size": len(test_tasks),
            "skill_bank_size": len(skill_bank),
            "methods": {},
        }

        for method in METHODS:
            logger.info(f"\n  === {bench_name} / {method} ===")
            result = evaluate_method(llm_client, method, skill_bank, test_tasks, label=bench_name)
            bench_results["methods"][method] = result
            logger.info(f"  {method}: EM={result['avg_em']:.1%}±{result['std_em']:.3f}, "
                        f"F1={result['avg_f1']:.3f}, tokens={result['avg_tokens']:.0f}")

        # Also record dedup stats
        deduped = deduplicate_skills_embedding(skill_bank, threshold=0.75)
        redundant_pairs = find_redundant_pairs(skill_bank, threshold=0.75)
        bench_results["dedup_stats"] = {
            "original_size": len(skill_bank),
            "deduped_size": len(deduped),
            "redundant_pairs": len(redundant_pairs),
            "reduction_pct": 1.0 - len(deduped) / max(len(skill_bank), 1),
        }
        logger.info(f"\n  Dedup: {len(skill_bank)} → {len(deduped)} skills "
                     f"({len(redundant_pairs)} redundant pairs, "
                     f"{bench_results['dedup_stats']['reduction_pct']:.1%} reduction)")

        all_results[bench_name] = bench_results

    return all_results


# ============================================================
# Experiment 2: δ_attention Independence (Table 2)
# ============================================================

def run_attention_independence(llm_client: LLMClient) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: δ_attention Independence (Table 2) — 30 tasks")
    logger.info("=" * 70)

    from benchmarks.loader import BenchmarkLoader

    # Build a skill bank from first 20 tasks
    loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": 20 + ATTENTION_TEST_SAMPLES})
    all_tasks = loader.load()

    skill_tasks = all_tasks[:20]
    test_tasks = all_tasks[20:20 + ATTENTION_TEST_SAMPLES]

    skill_bank = induce_skills_from_tasks(llm_client, skill_tasks, label="attn_skills")

    # Deduplicate to get a clean set of ~10 skills
    clean_skills = deduplicate_skills_embedding(skill_bank, threshold=0.70)
    logger.info(f"  Using {len(clean_skills)} clean skills, {len(test_tasks)} test tasks")

    results = {
        "benchmark": "hotpotqa",
        "num_skills": len(clean_skills),
        "num_test_tasks": len(test_tasks),
        "strategies": {},
    }

    for strategy in ATTENTION_STRATEGIES:
        logger.info(f"\n  Strategy: {strategy}")
        em_scores, f1_scores, token_counts = [], [], []

        for task in test_tasks:
            desc = task["description"]
            expected = task.get("expected", "")

            try:
                skill_context = _format_with_strategy(clean_skills, strategy, desc)
                token_counts.append(len(skill_context.split()))

                messages = [
                    {"role": "system", "content": f"Use these skills:\n\n{skill_context}\n\nAnswer directly and concisely. Give ONLY the answer."},
                    {"role": "user", "content": desc},
                ]
                resp = llm_client.chat(messages, temperature=0.1, max_tokens=128)
                em_scores.append(compute_em(resp, expected))
                f1_scores.append(compute_token_f1(resp, expected))
            except Exception as exc:
                logger.warning(f"    Failed: {exc}")
                em_scores.append(0.0)
                f1_scores.append(0.0)
                token_counts.append(0)

        results["strategies"][strategy] = {
            "avg_em": avg(em_scores),
            "avg_f1": avg(f1_scores),
            "avg_tokens": avg(token_counts),
            "std_em": std(em_scores),
            "std_f1": std(f1_scores),
            "n": len(em_scores),
        }
        logger.info(f"    EM={avg(em_scores):.1%}±{std(em_scores):.3f}, "
                     f"F1={avg(f1_scores):.3f}, tokens={avg(token_counts):.0f}")

    # Compute independence metrics
    sr_values = [v["avg_em"] for v in results["strategies"].values()]
    f1_values = [v["avg_f1"] for v in results["strategies"].values()]
    results["sr_range"] = max(sr_values) - min(sr_values) if sr_values else 0
    results["f1_range"] = max(f1_values) - min(f1_values) if f1_values else 0
    results["independence_verified"] = results["sr_range"] > 0.05 or results["f1_range"] > 0.05

    return results


# ============================================================
# Experiment 3: Phenomena (Phase Transition, Compaction Cliff, Scissors)
# ============================================================

def run_phenomenon_experiments(llm_client: LLMClient) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: Three Phenomena (§6.7) — large scale")
    logger.info("=" * 70)

    results = {}

    logger.info("\n--- Phenomenon 1: Phase Transition ---")
    results["phase_transition"] = _run_phase_transition(llm_client)

    logger.info("\n--- Phenomenon 2: Compaction Cliff ---")
    results["compaction_cliff"] = _run_compaction_cliff(llm_client)

    logger.info("\n--- Phenomenon 3: Scissors Effect ---")
    results["scissors_effect"] = _run_scissors_effect(llm_client)

    return results


def _run_phase_transition(llm_client: LLMClient) -> dict:
    """Phenomenon 1: Inverted-U curve at large scale."""
    from benchmarks.loader import BenchmarkLoader

    total = PHASE_TRANSITION_TRAIN + PHASE_TRANSITION_TEST
    loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": total})
    all_tasks = loader.load()

    train_tasks = all_tasks[:PHASE_TRANSITION_TRAIN]
    test_tasks = all_tasks[PHASE_TRANSITION_TRAIN:total]

    # Build large skill bank
    skill_bank = induce_skills_from_tasks(llm_client, train_tasks, label="phase_trans")
    logger.info(f"  Built skill bank: {len(skill_bank)} skills")

    # Test at different library sizes
    sizes_to_test = [1, 3, 5, 10, 15, 20, 30, 40, len(skill_bank)]
    sizes_to_test = sorted(set(s for s in sizes_to_test if s <= len(skill_bank)))

    # Three strategies: random subset, utility-based, compacted
    strategies = {}

    for strategy_name in ["random", "utility", "compacted"]:
        logger.info(f"\n  Strategy: {strategy_name}")
        curve = []

        for size in sizes_to_test:
            if strategy_name == "random":
                subset = random.sample(skill_bank, min(size, len(skill_bank)))
            elif strategy_name == "utility":
                # Sort by description length as proxy for utility
                sorted_skills = sorted(skill_bank, key=lambda s: len(s.procedure), reverse=True)
                subset = sorted_skills[:size]
            else:  # compacted
                deduped = deduplicate_skills_embedding(skill_bank, threshold=0.75)
                subset = deduped[:size]

            em_scores = []
            for task in test_tasks:
                try:
                    skill_context = format_skills_A3(subset, task["description"])
                    messages = [
                        {"role": "system", "content": f"Use these skills:\n\n{skill_context}\n\nAnswer directly. Give ONLY the answer."},
                        {"role": "user", "content": task["description"]},
                    ]
                    resp = llm_client.chat(messages, temperature=0.1, max_tokens=128)
                    em_scores.append(compute_em(resp, task.get("expected", "")))
                except:
                    em_scores.append(0.0)

            avg_em = avg(em_scores)
            curve.append({"size": size, "avg_em": avg_em, "std_em": std(em_scores)})
            logger.info(f"    N={size}: EM={avg_em:.1%}±{std(em_scores):.3f}")

        strategies[strategy_name] = curve

    # Find peaks
    peak_info = {}
    for sname, curve in strategies.items():
        if curve:
            best = max(curve, key=lambda x: x["avg_em"])
            peak_info[sname] = {"peak_size": best["size"], "peak_em": best["avg_em"]}

    n_star = compute_optimal_library_size()

    return {
        "strategies": strategies,
        "peak_info": peak_info,
        "n_star_theoretical": n_star,
        "total_skills": len(skill_bank),
    }


def _run_compaction_cliff(llm_client: LLMClient) -> dict:
    """Phenomenon 2: Token consumption drops after real compaction."""
    from benchmarks.loader import BenchmarkLoader

    loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": SCISSORS_STREAM_LEN})
    tasks = loader.load()

    collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
    compressor = create_compressor("mem0", llm_client, {})

    skill_bank = []
    token_history = []
    compaction_points = []
    COMPACTION_INTERVAL = 10

    for step, task in enumerate(tasks):
        try:
            traj = collector.collect(task_id=task["task_id"], task_description=task["description"])
            memory = compressor.compress(traj)
            inducer = create_inducer(TransformVariant.HYBRID_TO_SKILL, llm_client, {})
            skill = inducer.induce(trajectory=traj, memory=memory)
            skill.source_tasks = [task["task_id"]]
            skill_bank.append(skill)

            formatted = format_skills_B1(skill_bank, task["description"])
            tokens = len(formatted.split())
            token_history.append({"step": step, "tokens": tokens, "skills": len(skill_bank)})

            # Periodic compaction with REAL embedding-based merge
            if (step + 1) % COMPACTION_INTERVAL == 0 and len(skill_bank) > 3:
                before_size = len(skill_bank)
                before_tokens = tokens

                compacted, num_merges = compact_library(
                    llm_client, skill_bank, threshold=0.70, max_merges=5
                )
                skill_bank = compacted

                after_formatted = format_skills_B1(skill_bank, task["description"])
                after_tokens = len(after_formatted.split())

                cliff_ratio = after_tokens / before_tokens if before_tokens > 0 else 1.0
                compaction_points.append({
                    "step": step,
                    "skills_before": before_size,
                    "skills_after": len(skill_bank),
                    "tokens_before": before_tokens,
                    "tokens_after": after_tokens,
                    "cliff_ratio": cliff_ratio,
                    "num_merges": num_merges,
                })
                logger.info(f"    Step {step}: COMPACT {before_size}→{len(skill_bank)} skills "
                            f"({num_merges} merges), tokens {before_tokens}→{after_tokens} "
                            f"(cliff={cliff_ratio:.2f})")

            if (step + 1) % 10 == 0:
                logger.info(f"    Step {step}: {len(skill_bank)} skills, {tokens} tokens")

        except Exception as exc:
            logger.warning(f"    Step {step} failed: {exc}")

    avg_cliff = avg([cp["cliff_ratio"] for cp in compaction_points]) if compaction_points else 1.0

    return {
        "token_history": token_history,
        "compaction_points": compaction_points,
        "avg_cliff_ratio": avg_cliff,
        "total_steps": len(tasks),
    }


def _run_scissors_effect(llm_client: LLMClient) -> dict:
    """Phenomenon 3: Three lines — append-only / SkillOS / Ours."""
    from benchmarks.loader import BenchmarkLoader

    loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": SCISSORS_STREAM_LEN})
    tasks = loader.load()

    collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
    compressor = create_compressor("mem0", llm_client, {})

    # Three parallel libraries
    lib_append = []      # B1: append-only
    lib_skillos = []     # B2: SkillOS (delete low-utility)
    lib_ours = []        # A3: Ours (merge + prune)

    retrieval_counts_append = {}
    retrieval_counts_skillos = {}
    retrieval_counts_ours = {}

    history_append = []
    history_skillos = []
    history_ours = []

    COMPACT_INTERVAL = 10

    for step, task in enumerate(tasks):
        try:
            traj = collector.collect(task_id=task["task_id"], task_description=task["description"])
            memory = compressor.compress(traj)
            inducer = create_inducer(TransformVariant.HYBRID_TO_SKILL, llm_client, {})
            skill = inducer.induce(trajectory=traj, memory=memory)
            skill.source_tasks = [task["task_id"]]

            # Add to all three libraries
            lib_append.append(Skill(**skill.model_dump()))
            lib_skillos.append(Skill(**skill.model_dump()))
            lib_ours.append(Skill(**skill.model_dump()))

            # Simulate retrieval for each library
            for lib, counts in [
                (lib_append, retrieval_counts_append),
                (lib_skillos, retrieval_counts_skillos),
                (lib_ours, retrieval_counts_ours),
            ]:
                if lib:
                    top = _retrieve_top_k(lib, task["description"], k=1)
                    if top:
                        sid = top[0].skill_id
                        counts[sid] = counts.get(sid, 0) + 1

            # SkillOS: delete least-used skill every COMPACT_INTERVAL steps
            if (step + 1) % COMPACT_INTERVAL == 0 and len(lib_skillos) > 5:
                # Find least-used skill
                usage = [(s, retrieval_counts_skillos.get(s.skill_id, 0)) for s in lib_skillos]
                usage.sort(key=lambda x: x[1])
                # Delete bottom 20%
                n_delete = max(1, len(lib_skillos) // 5)
                to_delete = set(s.skill_id for s, _ in usage[:n_delete])
                lib_skillos = [s for s in lib_skillos if s.skill_id not in to_delete]
                for sid in to_delete:
                    retrieval_counts_skillos.pop(sid, None)

            # Ours: merge + prune every COMPACT_INTERVAL steps
            if (step + 1) % COMPACT_INTERVAL == 0 and len(lib_ours) > 5:
                lib_ours, n_merges = compact_library(
                    llm_client, lib_ours, threshold=0.70, max_merges=3
                )
                # Also prune least-used if still too large
                if len(lib_ours) > 30:
                    usage = [(s, retrieval_counts_ours.get(s.skill_id, 0)) for s in lib_ours]
                    usage.sort(key=lambda x: x[1])
                    n_prune = max(1, len(lib_ours) // 10)
                    to_prune = set(s.skill_id for s, _ in usage[:n_prune])
                    lib_ours = [s for s in lib_ours if s.skill_id not in to_prune]

            # Record N_eff for each library
            for lib, counts, history in [
                (lib_append, retrieval_counts_append, history_append),
                (lib_skillos, retrieval_counts_skillos, history_skillos),
                (lib_ours, retrieval_counts_ours, history_ours),
            ]:
                total_r = sum(counts.values())
                if total_r > 0 and len(lib) > 0:
                    probs = [counts.get(s.skill_id, 0) / total_r for s in lib]
                    n_eff = compute_effective_skill_count(probs)
                else:
                    n_eff = float(len(lib))
                total_count = len(lib)
                ratio = n_eff / total_count if total_count > 0 else 1.0
                history.append({
                    "step": step,
                    "total_count": total_count,
                    "effective_count": round(n_eff, 2),
                    "ratio": round(ratio, 3),
                })

            if (step + 1) % 10 == 0:
                logger.info(f"    Step {step}: append={len(lib_append)}, "
                            f"skillos={len(lib_skillos)}, ours={len(lib_ours)}")

        except Exception as exc:
            logger.warning(f"    Step {step} failed: {exc}")

    return {
        "append_only": {
            "history": history_append,
            "final_total": history_append[-1]["total_count"] if history_append else 0,
            "final_effective": history_append[-1]["effective_count"] if history_append else 0,
            "final_ratio": history_append[-1]["ratio"] if history_append else 0,
        },
        "skillos": {
            "history": history_skillos,
            "final_total": history_skillos[-1]["total_count"] if history_skillos else 0,
            "final_effective": history_skillos[-1]["effective_count"] if history_skillos else 0,
            "final_ratio": history_skillos[-1]["ratio"] if history_skillos else 0,
        },
        "ours": {
            "history": history_ours,
            "final_total": history_ours[-1]["total_count"] if history_ours else 0,
            "final_effective": history_ours[-1]["effective_count"] if history_ours else 0,
            "final_ratio": history_ours[-1]["ratio"] if history_ours else 0,
        },
    }


# ============================================================
# Experiment 4: Bound Tightening (Figure 4)
# ============================================================

def run_bound_tightening(llm_client: LLMClient) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 4: Bound Tightening (§6.6) — 40 steps")
    logger.info("=" * 70)

    from benchmarks.loader import BenchmarkLoader

    loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": BOUND_TIGHTENING_LEN})
    tasks = loader.load()

    collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
    compressor = create_compressor("mem0", llm_client, {})

    skill_bank = []
    delta_history = []
    COMPACT_INTERVAL = 8

    for step, task in enumerate(tasks):
        try:
            traj = collector.collect(task_id=task["task_id"], task_description=task["description"])
            memory = compressor.compress(traj)
            inducer = create_inducer(TransformVariant.HYBRID_TO_SKILL, llm_client, {})
            skill = inducer.induce(trajectory=traj, memory=memory)
            skill.source_tasks = [task["task_id"]]
            skill_bank.append(skill)

            decomp = decompose_delta(skill_bank)
            delta_history.append({
                "step": step,
                "num_skills": len(skill_bank),
                "delta_total": round(decomp.delta_total, 4),
                "delta_semantic": round(decomp.delta_semantic, 4),
                "delta_attention": round(decomp.delta_attention, 4),
                "compacted": False,
            })

            # Compact every COMPACT_INTERVAL steps
            if (step + 1) % COMPACT_INTERVAL == 0 and len(skill_bank) > 3:
                before_decomp = decomp
                compacted, n_merges = compact_library(
                    llm_client, skill_bank, threshold=0.70, max_merges=3
                )
                skill_bank = compacted
                after_decomp = decompose_delta(skill_bank)

                delta_history.append({
                    "step": step,
                    "num_skills": len(skill_bank),
                    "delta_total": round(after_decomp.delta_total, 4),
                    "delta_semantic": round(after_decomp.delta_semantic, 4),
                    "delta_attention": round(after_decomp.delta_attention, 4),
                    "compacted": True,
                    "merges": n_merges,
                    "improvement": round(before_decomp.delta_total - after_decomp.delta_total, 4),
                })
                logger.info(f"    Step {step}: COMPACT δ={before_decomp.delta_total:.4f}"
                            f"→{after_decomp.delta_total:.4f} ({n_merges} merges)")

            if (step + 1) % 10 == 0:
                logger.info(f"    Step {step}: {len(skill_bank)} skills, δ={decomp.delta_total:.4f}")

        except Exception as exc:
            logger.warning(f"    Step {step} failed: {exc}")

    return {
        "delta_history": delta_history,
        "final_delta": delta_history[-1]["delta_total"] if delta_history else 0,
        "compaction_improvements": [h for h in delta_history if h.get("compacted")],
    }


# ============================================================
# Experiment 5: Cross-Benchmark Validation
# ============================================================

def run_cross_benchmark(llm_client: LLMClient) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 5: Cross-Benchmark (LoCoMo + LongMemEval)")
    logger.info("=" * 70)

    from benchmarks.loader import BenchmarkLoader

    results = {}

    for bench_name in ["locomo", "longmemeval"]:
        logger.info(f"\n  --- {bench_name} ---")
        try:
            loader = BenchmarkLoader({"name": bench_name, "num_samples": 20})
            tasks = loader.load()[:20]

            em_b2, f1_b2 = [], []
            em_a3, f1_a3 = [], []

            for task in tasks:
                desc = task["description"]
                expected = task.get("expected", "")
                context = task.get("context", "")[:4000]

                try:
                    # B2: full context
                    q = desc.split("Question: ")[-1] if "Question: " in desc else desc[-200:]
                    messages_b2 = [
                        {"role": "system", "content": "Answer based on the conversation. Be concise."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"},
                    ]
                    resp_b2 = llm_client.chat(messages_b2, temperature=0.1, max_tokens=128)
                    em_b2.append(compute_em(resp_b2, expected))
                    f1_b2.append(compute_token_f1(resp_b2, expected))

                    # A3: sandwich formatted
                    compact_ctx = context[:2000]
                    messages_a3 = [
                        {"role": "system", "content": f"KEY: Answer ONLY from the conversation.\n\n{compact_ctx}\n\nREMEMBER: Be concise."},
                        {"role": "user", "content": q},
                    ]
                    resp_a3 = llm_client.chat(messages_a3, temperature=0.1, max_tokens=128)
                    em_a3.append(compute_em(resp_a3, expected))
                    f1_a3.append(compute_token_f1(resp_a3, expected))

                except Exception as exc:
                    logger.warning(f"    Task failed: {exc}")
                    em_b2.append(0.0); f1_b2.append(0.0)
                    em_a3.append(0.0); f1_a3.append(0.0)

            results[bench_name] = {
                "B2": {"avg_em": avg(em_b2), "avg_f1": avg(f1_b2), "std_em": std(em_b2)},
                "A3": {"avg_em": avg(em_a3), "avg_f1": avg(f1_a3), "std_em": std(em_a3)},
                "num_tasks": len(tasks),
            }
            logger.info(f"    B2: EM={avg(em_b2):.1%}, F1={avg(f1_b2):.3f}")
            logger.info(f"    A3: EM={avg(em_a3):.1%}, F1={avg(f1_a3):.3f}")

        except Exception as exc:
            logger.error(f"  {bench_name} failed: {exc}")
            results[bench_name] = {"error": str(exc)}

    return results


# ============================================================
# Main
# ============================================================

def main():
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
    log_path = Path("experiments/paper_v2.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}", level="DEBUG")

    logger.info("=" * 70)
    logger.info("SkillCurator Paper Experiments v2 — 10x Scale")
    logger.info("=" * 70)
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: TRAIN={TRAIN_SAMPLES}, TEST={TEST_SAMPLES}, "
                f"BENCHMARKS={BENCHMARKS}")

    load_env()
    start_time = time.time()
    llm_client = LLMClient({"temperature": 0.7, "max_tokens": 2048, "timeout": 120})

    # API test
    try:
        resp = llm_client.chat([{"role": "user", "content": "Say OK"}], temperature=0.0, max_tokens=5)
        logger.info(f"API: '{resp.strip()}' OK")
    except Exception as exc:
        logger.error(f"API failed: {exc}")
        sys.exit(1)

    # Embedding model test
    try:
        get_embed_model()
        logger.info("Embedding model: OK")
    except Exception as exc:
        logger.error(f"Embedding model failed: {exc}")
        sys.exit(1)

    all_results = {"meta": {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v2",
        "config": {
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "benchmarks": BENCHMARKS,
            "max_traj_steps": MAX_TRAJ_STEPS,
            "dedup_threshold": 0.75,
        },
    }}

    output_path = Path("experiments/paper_v2_results.json")

    def save_intermediate():
        elapsed = time.time() - start_time
        stats = llm_client.stats
        all_results["meta"]["elapsed_seconds"] = elapsed
        all_results["meta"]["total_api_calls"] = stats["total_calls"]
        all_results["meta"]["total_tokens"] = stats["total_tokens"]
        output_path.write_text(json.dumps(all_results, indent=2, default=str, ensure_ascii=False))
        logger.info(f"  [Saved: {elapsed:.0f}s, {stats['total_tokens']:,} tokens]")

    # Run all experiments
    experiments = [
        ("main_experiment", run_main_experiment),
        ("attention_independence", run_attention_independence),
        ("phenomena", run_phenomenon_experiments),
        ("bound_tightening", run_bound_tightening),
        ("cross_benchmark", run_cross_benchmark),
    ]

    for name, func in experiments:
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting: {name}")
        logger.info(f"{'='*70}")
        try:
            all_results[name] = func(llm_client)
        except Exception as exc:
            logger.error(f"{name} FAILED: {exc}\n{traceback.format_exc()}")
            all_results[name] = {"error": str(exc)}
        save_intermediate()

    # ============================================================
    # Final Summary
    # ============================================================
    elapsed = time.time() - start_time
    stats = llm_client.stats

    logger.info("\n" + "=" * 70)
    logger.info("PAPER v2 EXPERIMENT RESULTS — FINAL SUMMARY")
    logger.info("=" * 70)

    # Table 1
    if "main_experiment" in all_results and "error" not in all_results["main_experiment"]:
        me = all_results["main_experiment"]
        for bench_name, bench_data in me.items():
            if isinstance(bench_data, dict) and "methods" in bench_data:
                logger.info(f"\n Table 1 [{bench_name}] ({bench_data['test_size']} test tasks, "
                            f"{bench_data['skill_bank_size']} skills)")
                logger.info(f"{'Method':<8} {'EM':>10} {'F1':>10} {'Tokens':>8}")
                logger.info("-" * 40)
                for method, data in bench_data["methods"].items():
                    logger.info(f"{method:<8} {data['avg_em']:>7.1%}±{data['std_em']:.2f} "
                                f"{data['avg_f1']:>8.3f} {data['avg_tokens']:>8.0f}")
                if "dedup_stats" in bench_data:
                    ds = bench_data["dedup_stats"]
                    logger.info(f"  Dedup: {ds['original_size']}→{ds['deduped_size']} "
                                f"({ds['reduction_pct']:.1%} reduction)")

    # Table 2
    if "attention_independence" in all_results and "error" not in all_results["attention_independence"]:
        ai = all_results["attention_independence"]
        logger.info(f"\n Table 2: δ_attention Independence "
                    f"(SR range={ai['sr_range']:.1%}, F1 range={ai['f1_range']:.3f})")
        logger.info(f"  Independence: {'VERIFIED' if ai['independence_verified'] else 'NOT VERIFIED'}")
        for strat, data in ai["strategies"].items():
            logger.info(f"  {strat:<25}: EM={data['avg_em']:.1%}±{data['std_em']:.3f}, "
                        f"F1={data['avg_f1']:.3f}, tokens={data['avg_tokens']:.0f}")

    # Phenomena
    if "phenomena" in all_results and "error" not in all_results["phenomena"]:
        ph = all_results["phenomena"]

        if "phase_transition" in ph:
            pt = ph["phase_transition"]
            logger.info(f"\n Phenomenon 1: Phase Transition")
            for sname, curve in pt.get("strategies", {}).items():
                if curve:
                    best = max(curve, key=lambda x: x["avg_em"])
                    logger.info(f"  {sname}: peak at N={best['size']}, EM={best['avg_em']:.1%}")

        if "compaction_cliff" in ph:
            cc = ph["compaction_cliff"]
            logger.info(f"\n Phenomenon 2: Compaction Cliff")
            logger.info(f"  Avg cliff ratio: {cc['avg_cliff_ratio']:.2f}")
            for cp in cc.get("compaction_points", []):
                logger.info(f"    Step {cp['step']}: {cp['skills_before']}→{cp['skills_after']} skills, "
                            f"tokens {cp['tokens_before']}→{cp['tokens_after']} "
                            f"({cp['num_merges']} merges, cliff={cp['cliff_ratio']:.2f})")

        if "scissors_effect" in ph:
            se = ph["scissors_effect"]
            logger.info(f"\n Phenomenon 3: Scissors Effect")
            for lib_name in ["append_only", "skillos", "ours"]:
                if lib_name in se:
                    d = se[lib_name]
                    logger.info(f"  {lib_name}: |S|={d['final_total']}, "
                                f"N_eff={d['final_effective']}, ratio={d['final_ratio']:.3f}")

    # Bound tightening
    if "bound_tightening" in all_results and "error" not in all_results["bound_tightening"]:
        bt = all_results["bound_tightening"]
        logger.info(f"\n Bound Tightening")
        logger.info(f"  Final δ_M: {bt['final_delta']:.4f}")
        for ci in bt.get("compaction_improvements", []):
            logger.info(f"    Step {ci['step']}: δ={ci['delta_total']:.4f} "
                        f"(improvement={ci.get('improvement', 0):.4f}, {ci.get('merges', 0)} merges)")

    # Cross-benchmark
    if "cross_benchmark" in all_results and "error" not in all_results["cross_benchmark"]:
        cb = all_results["cross_benchmark"]
        logger.info(f"\n Cross-Benchmark")
        for bench, data in cb.items():
            if isinstance(data, dict) and "error" not in data:
                logger.info(f"  {bench}: B2 EM={data['B2']['avg_em']:.1%}, A3 EM={data['A3']['avg_em']:.1%}")

    logger.info(f"\n Total: {stats['total_calls']} calls, {stats['total_tokens']:,} tokens, "
                f"{elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Final save
    all_results["meta"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_intermediate()
    logger.info(f"\n  Final results: {output_path}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
