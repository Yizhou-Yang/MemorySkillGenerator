#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillCurator Full Paper Experiments v3 — 9 benchmarks, publication-ready."""
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
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from loguru import logger

from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.memory.compressor import create_compressor
from src.models import Skill, TransformVariant
from src.skill_induction.factory import create_inducer
from src.trajectory.collector import TrajectoryCollector
from src.curation.delta_decomposition import (
    decompose_delta, compute_effective_skill_count,
    compute_optimal_library_size,
)
from src.utils.skill_formatter import (
    format_skill_library, FormattingConfig,
)

# 9-Benchmark Configuration (tiered)

BENCHMARK_TIERS = {
    # Tier 1: Full scale — best for skill induction (multi-hop QA)
    "hotpotqa":         {"train": 40, "test": 50, "tier": 1},
    "2wikimultihopqa":  {"train": 40, "test": 50, "tier": 1},
    "musique":          {"train": 40, "test": 50, "tier": 1},
    # Tier 2: Medium scale — cross-domain validation
    "triviaqa":         {"train": 20, "test": 30, "tier": 2},
    "gsm8k":            {"train": 20, "test": 30, "tier": 2},
    "alfworld":         {"train": 20, "test": 30, "tier": 2},
    # Tier 3: Light scale — specialized memory evaluation
    "webshop":          {"train": 10, "test": 20, "tier": 3},
    "locomo":           {"train": 10, "test": 20, "tier": 3},
    "longmemeval":      {"train": 10, "test": 20, "tier": 3},
}

ALL_BENCHMARKS = list(BENCHMARK_TIERS.keys())
METHODS = ["B0", "B1", "B2", "A1", "A2", "A3"]
MAX_TRAJ_STEPS = 6

# Phenomena experiments run on these benchmarks
PHENOMENA_BENCHMARKS = ["hotpotqa", "2wikimultihopqa", "gsm8k"]
PHENOMENA_STREAM_LEN = 40

# δ_attention independence on these benchmarks
ATTENTION_BENCHMARKS = ["hotpotqa", "musique"]
ATTENTION_SKILL_TASKS = 15
ATTENTION_TEST_TASKS = 20

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

# Embedding model
EMBED_MODEL = None

def get_embed_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
    return EMBED_MODEL

# Metrics (same as v2)

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

# Embedding-based Semantic Dedup (from v2)

def skill_to_text(s: Skill) -> str:
    parts = [s.name, s.description]
    parts.extend(s.procedure[:3])
    if s.constraints:
        parts.extend(s.constraints[:2])
    return " ".join(parts)

def compute_skill_embeddings(skills: list[Skill]) -> np.ndarray:
    model = get_embed_model()
    texts = [skill_to_text(s) for s in skills]
    return model.encode(texts, normalize_embeddings=True)

def find_redundant_pairs(skills: list[Skill], threshold: float = 0.75) -> list[tuple[int, int, float]]:
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
        return s1 if s1.compactness >= s2.compactness else s2

def compact_library(llm_client: LLMClient, skills: list[Skill],
                    threshold: float = 0.75, max_merges: int = 10) -> tuple[list[Skill], int]:
    merged_count = 0
    current = list(skills)
    for _ in range(max_merges):
        pairs = find_redundant_pairs(current, threshold=threshold)
        if not pairs:
            break
        i, j, sim = pairs[0]
        merged = merge_skill_pair(llm_client, current[i], current[j])
        new_list = [s for idx, s in enumerate(current) if idx != i and idx != j]
        new_list.append(merged)
        current = new_list
        merged_count += 1
        logger.debug(f"  MERGE: sim={sim:.3f}, {len(current)} skills remain")
    return current, merged_count

# Skill Library Formatting (from v2)

def format_skills_B0(skills: list[Skill], query: str) -> str:
    return ""

def format_skills_B1(skills: list[Skill], query: str) -> str:
    parts = []
    for i, s in enumerate(skills):
        parts.append(f"Skill {i+1}: {s.name}\n{s.description}\nProcedure: {'; '.join(s.procedure)}")
        if s.constraints:
            parts.append(f"Constraints: {'; '.join(s.constraints)}")
        parts.append("")
    return "\n".join(parts)

def _retrieve_top_k(skills: list[Skill], query: str, k: int = 5) -> list[Skill]:
    if not skills:
        return []
    model = get_embed_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    s_embs = compute_skill_embeddings(skills)
    sims = np.dot(s_embs, q_emb.T).flatten()
    top_indices = np.argsort(sims)[::-1][:k]
    return [skills[i] for i in top_indices]

def format_skills_B2(skills: list[Skill], query: str) -> str:
    top = _retrieve_top_k(skills, query, k=5)
    return format_skills_B1(top, query)

def format_skills_A1(skills: list[Skill], query: str) -> str:
    deduped = deduplicate_skills_embedding(skills, threshold=0.75)
    top = _retrieve_top_k(deduped, query, k=5)
    return format_skills_B1(top, query)

def format_skills_A2(skills: list[Skill], query: str) -> str:
    top = _retrieve_top_k(skills, query, k=5)
    return format_skill_library(top, config=FormattingConfig(strategy="sandwich_compact"))

def format_skills_A3(skills: list[Skill], query: str) -> str:
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

# Skill Induction Helper

def induce_skills_from_tasks(
    llm_client: LLMClient,
    tasks: list[dict],
    max_traj_steps: int = MAX_TRAJ_STEPS,
    label: str = "train",
) -> list[Skill]:
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

# Evaluation Helper

def evaluate_method(
    llm_client: LLMClient,
    method: str,
    skill_bank: list[Skill],
    test_tasks: list[dict],
    label: str = "",
) -> dict:
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
            em_scores.append(compute_em(resp, expected))
            f1_scores.append(compute_token_f1(resp, expected))
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

# EXPERIMENT 1: Main Experiment (Table 1) — 9 benchmarks

def run_main_experiment(llm_client: LLMClient) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: Main Experiment (Table 1) — 9 benchmarks")
    logger.info("=" * 70)

    from benchmarks.loader import BenchmarkLoader

    all_results = {}

    for bench_name, bench_cfg in BENCHMARK_TIERS.items():
        train_target = bench_cfg["train"]
        test_target = bench_cfg["test"]
        tier = bench_cfg["tier"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark: {bench_name} (Tier {tier}, train={train_target}, test={test_target})")
        logger.info(f"{'='*60}")

        total_needed = train_target + test_target
        try:
            loader = BenchmarkLoader({"name": bench_name, "num_samples": total_needed})
            all_tasks = loader.load()
        except Exception as exc:
            logger.error(f"  Failed to load {bench_name}: {exc}")
            all_results[bench_name] = {"error": str(exc), "tier": tier}
            continue

        if len(all_tasks) < 5:
            logger.warning(f"  Only {len(all_tasks)} tasks, skipping {bench_name}")
            all_results[bench_name] = {"error": f"Only {len(all_tasks)} tasks", "tier": tier}
            continue

        # Adjust sizes if not enough data
        if len(all_tasks) < total_needed:
            train_size = min(train_target, len(all_tasks) * 2 // 3)
            test_size = len(all_tasks) - train_size
        else:
            train_size = train_target
            test_size = test_target

        train_tasks = all_tasks[:train_size]
        test_tasks = all_tasks[train_size:train_size + test_size]
        logger.info(f"  Split: {len(train_tasks)} train + {len(test_tasks)} test")

        # Phase 1: Skill Induction
        logger.info(f"\n--- Skill Induction ({bench_name}) ---")
        skill_bank = induce_skills_from_tasks(llm_client, train_tasks, label=bench_name)

        # Phase 2: Evaluate all 6 methods
        logger.info(f"\n--- Evaluation ({bench_name}) ---")
        bench_results = {
            "benchmark": bench_name,
            "tier": tier,
            "train_size": len(train_tasks),
            "test_size": len(test_tasks),
            "skill_bank_size": len(skill_bank),
            "methods": {},
        }

        for method in METHODS:
            logger.info(f"  {bench_name}/{method}...")
            result = evaluate_method(llm_client, method, skill_bank, test_tasks, label=bench_name)
            bench_results["methods"][method] = result
            logger.info(f"    {method}: EM={result['avg_em']:.1%}±{result['std_em']:.3f}, "
                        f"F1={result['avg_f1']:.3f}, tokens={result['avg_tokens']:.0f}")

        # Dedup stats
        deduped = deduplicate_skills_embedding(skill_bank, threshold=0.75)
        redundant_pairs = find_redundant_pairs(skill_bank, threshold=0.75)
        bench_results["dedup_stats"] = {
            "original_size": len(skill_bank),
            "deduped_size": len(deduped),
            "redundant_pairs": len(redundant_pairs),
            "reduction_pct": 1.0 - len(deduped) / max(len(skill_bank), 1),
        }
        logger.info(f"  Dedup: {len(skill_bank)}→{len(deduped)} "
                     f"({len(redundant_pairs)} pairs, {bench_results['dedup_stats']['reduction_pct']:.1%} reduction)")

        all_results[bench_name] = bench_results

    return all_results

# EXPERIMENT 2: δ_attention Independence (Table 2) — 2 benchmarks

def run_attention_independence(llm_client: LLMClient) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: δ_attention Independence — 2 benchmarks")
    logger.info("=" * 70)

    from benchmarks.loader import BenchmarkLoader

    all_results = {}

    for bench_name in ATTENTION_BENCHMARKS:
        logger.info(f"\n--- {bench_name} ---")
        total = ATTENTION_SKILL_TASKS + ATTENTION_TEST_TASKS
        loader = BenchmarkLoader({"name": bench_name, "num_samples": total})
        all_tasks = loader.load()

        skill_tasks = all_tasks[:ATTENTION_SKILL_TASKS]
        test_tasks = all_tasks[ATTENTION_SKILL_TASKS:total]

        skill_bank = induce_skills_from_tasks(llm_client, skill_tasks, label=f"attn_{bench_name}")
        clean_skills = deduplicate_skills_embedding(skill_bank, threshold=0.70)
        logger.info(f"  {len(clean_skills)} clean skills, {len(test_tasks)} test tasks")

        bench_results = {
            "benchmark": bench_name,
            "num_skills": len(clean_skills),
            "num_test_tasks": len(test_tasks),
            "strategies": {},
        }

        for strategy in ATTENTION_STRATEGIES:
            logger.info(f"  Strategy: {strategy}")
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

            bench_results["strategies"][strategy] = {
                "avg_em": avg(em_scores),
                "avg_f1": avg(f1_scores),
                "avg_tokens": avg(token_counts),
                "std_em": std(em_scores),
                "std_f1": std(f1_scores),
                "n": len(em_scores),
            }
            logger.info(f"    EM={avg(em_scores):.1%}±{std(em_scores):.3f}, "
                         f"F1={avg(f1_scores):.3f}, tokens={avg(token_counts):.0f}")

        sr_values = [v["avg_em"] for v in bench_results["strategies"].values()]
        f1_values = [v["avg_f1"] for v in bench_results["strategies"].values()]
        bench_results["sr_range"] = max(sr_values) - min(sr_values) if sr_values else 0
        bench_results["f1_range"] = max(f1_values) - min(f1_values) if f1_values else 0
        bench_results["independence_verified"] = bench_results["sr_range"] > 0.05 or bench_results["f1_range"] > 0.05

        all_results[bench_name] = bench_results

    return all_results

# EXPERIMENT 3: Phenomena — 3 benchmarks

def run_phenomenon_experiments(llm_client: LLMClient) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: Three Phenomena — 3 benchmarks")
    logger.info("=" * 70)

    results = {}

    # Phase Transition on 2 benchmarks
    logger.info("\n--- Phenomenon 1: Phase Transition ---")
    results["phase_transition"] = {}
    for bench_name in ["hotpotqa", "2wikimultihopqa"]:
        logger.info(f"\n  [{bench_name}]")
        results["phase_transition"][bench_name] = _run_phase_transition(llm_client, bench_name)

    # Compaction Cliff on 2 benchmarks
    logger.info("\n--- Phenomenon 2: Compaction Cliff ---")
    results["compaction_cliff"] = {}
    for bench_name in ["hotpotqa", "gsm8k"]:
        logger.info(f"\n  [{bench_name}]")
        results["compaction_cliff"][bench_name] = _run_compaction_cliff(llm_client, bench_name)

    # Scissors Effect on 2 benchmarks
    logger.info("\n--- Phenomenon 3: Scissors Effect ---")
    results["scissors_effect"] = {}
    for bench_name in ["hotpotqa", "2wikimultihopqa"]:
        logger.info(f"\n  [{bench_name}]")
        results["scissors_effect"][bench_name] = _run_scissors_effect(llm_client, bench_name)

    return results

def _run_phase_transition(llm_client: LLMClient, bench_name: str) -> dict:
    from benchmarks.loader import BenchmarkLoader

    train_n = 30
    test_n = 15
    total = train_n + test_n
    loader = BenchmarkLoader({"name": bench_name, "num_samples": total})
    all_tasks = loader.load()

    train_tasks = all_tasks[:train_n]
    test_tasks = all_tasks[train_n:total]

    skill_bank = induce_skills_from_tasks(llm_client, train_tasks, label=f"pt_{bench_name}")
    logger.info(f"  Built {len(skill_bank)} skills")

    sizes_to_test = [1, 3, 5, 10, 15, 20, len(skill_bank)]
    sizes_to_test = sorted(set(s for s in sizes_to_test if s <= len(skill_bank)))

    strategies = {}
    for strategy_name in ["random", "utility", "compacted"]:
        logger.info(f"  Strategy: {strategy_name}")
        curve = []
        for size in sizes_to_test:
            if strategy_name == "random":
                subset = random.sample(skill_bank, min(size, len(skill_bank)))
            elif strategy_name == "utility":
                sorted_skills = sorted(skill_bank, key=lambda s: len(s.procedure), reverse=True)
                subset = sorted_skills[:size]
            else:
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
            logger.info(f"    N={size}: EM={avg_em:.1%}")

        strategies[strategy_name] = curve

    peak_info = {}
    for sname, curve in strategies.items():
        if curve:
            best = max(curve, key=lambda x: x["avg_em"])
            peak_info[sname] = {"peak_size": best["size"], "peak_em": best["avg_em"]}

    return {
        "strategies": strategies,
        "peak_info": peak_info,
        "n_star_theoretical": compute_optimal_library_size(),
        "total_skills": len(skill_bank),
    }

def _run_compaction_cliff(llm_client: LLMClient, bench_name: str) -> dict:
    from benchmarks.loader import BenchmarkLoader

    loader = BenchmarkLoader({"name": bench_name, "num_samples": PHENOMENA_STREAM_LEN})
    tasks = loader.load()

    collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
    compressor = create_compressor("mem0", llm_client, {})

    skill_bank = []
    token_history = []
    compaction_points = []
    COMPACTION_INTERVAL = 8

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
                logger.info(f"    Step {step}: COMPACT {before_size}→{len(skill_bank)} "
                            f"({num_merges} merges, cliff={cliff_ratio:.2f})")

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

def _run_scissors_effect(llm_client: LLMClient, bench_name: str) -> dict:
    from benchmarks.loader import BenchmarkLoader

    loader = BenchmarkLoader({"name": bench_name, "num_samples": PHENOMENA_STREAM_LEN})
    tasks = loader.load()

    collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
    compressor = create_compressor("mem0", llm_client, {})

    lib_append = []
    lib_skillos = []
    lib_ours = []
    rc_append, rc_skillos, rc_ours = {}, {}, {}
    hist_append, hist_skillos, hist_ours = [], [], []
    COMPACT_INTERVAL = 8

    for step, task in enumerate(tasks):
        try:
            traj = collector.collect(task_id=task["task_id"], task_description=task["description"])
            memory = compressor.compress(traj)
            inducer = create_inducer(TransformVariant.HYBRID_TO_SKILL, llm_client, {})
            skill = inducer.induce(trajectory=traj, memory=memory)
            skill.source_tasks = [task["task_id"]]

            lib_append.append(Skill(**skill.model_dump()))
            lib_skillos.append(Skill(**skill.model_dump()))
            lib_ours.append(Skill(**skill.model_dump()))

            for lib, counts in [(lib_append, rc_append), (lib_skillos, rc_skillos), (lib_ours, rc_ours)]:
                if lib:
                    top = _retrieve_top_k(lib, task["description"], k=1)
                    if top:
                        sid = top[0].skill_id
                        counts[sid] = counts.get(sid, 0) + 1

            if (step + 1) % COMPACT_INTERVAL == 0 and len(lib_skillos) > 5:
                usage = [(s, rc_skillos.get(s.skill_id, 0)) for s in lib_skillos]
                usage.sort(key=lambda x: x[1])
                n_delete = max(1, len(lib_skillos) // 5)
                to_delete = set(s.skill_id for s, _ in usage[:n_delete])
                lib_skillos = [s for s in lib_skillos if s.skill_id not in to_delete]
                for sid in to_delete:
                    rc_skillos.pop(sid, None)

            if (step + 1) % COMPACT_INTERVAL == 0 and len(lib_ours) > 5:
                lib_ours, n_merges = compact_library(llm_client, lib_ours, threshold=0.70, max_merges=3)
                if len(lib_ours) > 25:
                    usage = [(s, rc_ours.get(s.skill_id, 0)) for s in lib_ours]
                    usage.sort(key=lambda x: x[1])
                    n_prune = max(1, len(lib_ours) // 10)
                    to_prune = set(s.skill_id for s, _ in usage[:n_prune])
                    lib_ours = [s for s in lib_ours if s.skill_id not in to_prune]

            for lib, counts, history in [
                (lib_append, rc_append, hist_append),
                (lib_skillos, rc_skillos, hist_skillos),
                (lib_ours, rc_ours, hist_ours),
            ]:
                total_r = sum(counts.values())
                if total_r > 0 and len(lib) > 0:
                    probs = [counts.get(s.skill_id, 0) / total_r for s in lib]
                    n_eff = compute_effective_skill_count(probs)
                else:
                    n_eff = float(len(lib))
                total_count = len(lib)
                ratio = n_eff / total_count if total_count > 0 else 1.0
                history.append({"step": step, "total_count": total_count,
                                "effective_count": round(n_eff, 2), "ratio": round(ratio, 3)})

            if (step + 1) % 10 == 0:
                logger.info(f"    Step {step}: append={len(lib_append)}, "
                            f"skillos={len(lib_skillos)}, ours={len(lib_ours)}")

        except Exception as exc:
            logger.warning(f"    Step {step} failed: {exc}")

    def _final(hist):
        return {"history": hist,
                "final_total": hist[-1]["total_count"] if hist else 0,
                "final_effective": hist[-1]["effective_count"] if hist else 0,
                "final_ratio": hist[-1]["ratio"] if hist else 0}

    return {"append_only": _final(hist_append),
            "skillos": _final(hist_skillos),
            "ours": _final(hist_ours)}

# EXPERIMENT 4: Bound Tightening — 2 benchmarks

def run_bound_tightening(llm_client: LLMClient) -> dict:
    logger.info("=" * 70)
    logger.info("EXPERIMENT 4: Bound Tightening — 2 benchmarks")
    logger.info("=" * 70)

    from benchmarks.loader import BenchmarkLoader

    all_results = {}

    for bench_name in ["hotpotqa", "gsm8k"]:
        logger.info(f"\n--- {bench_name} ---")
        loader = BenchmarkLoader({"name": bench_name, "num_samples": 30})
        tasks = loader.load()

        collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
        compressor = create_compressor("mem0", llm_client, {})

        skill_bank = []
        delta_history = []
        COMPACT_INTERVAL = 6

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
                    "step": step, "num_skills": len(skill_bank),
                    "delta_total": round(decomp.delta_total, 4),
                    "delta_semantic": round(decomp.delta_semantic, 4),
                    "delta_attention": round(decomp.delta_attention, 4),
                    "compacted": False,
                })

                if (step + 1) % COMPACT_INTERVAL == 0 and len(skill_bank) > 3:
                    before_decomp = decomp
                    compacted, n_merges = compact_library(llm_client, skill_bank, threshold=0.70, max_merges=3)
                    skill_bank = compacted
                    after_decomp = decompose_delta(skill_bank)
                    delta_history.append({
                        "step": step, "num_skills": len(skill_bank),
                        "delta_total": round(after_decomp.delta_total, 4),
                        "delta_semantic": round(after_decomp.delta_semantic, 4),
                        "delta_attention": round(after_decomp.delta_attention, 4),
                        "compacted": True, "merges": n_merges,
                        "improvement": round(before_decomp.delta_total - after_decomp.delta_total, 4),
                    })
                    logger.info(f"    Step {step}: COMPACT δ={before_decomp.delta_total:.4f}"
                                f"→{after_decomp.delta_total:.4f} ({n_merges} merges)")

                if (step + 1) % 10 == 0:
                    logger.info(f"    Step {step}: {len(skill_bank)} skills, δ={decomp.delta_total:.4f}")

            except Exception as exc:
                logger.warning(f"    Step {step} failed: {exc}")

        all_results[bench_name] = {
            "delta_history": delta_history,
            "final_delta": delta_history[-1]["delta_total"] if delta_history else 0,
            "compaction_improvements": [h for h in delta_history if h.get("compacted")],
        }

    return all_results

# Main

def main():
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
    log_path = Path("experiments/paper_v3.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}", level="DEBUG")

    logger.info("=" * 70)
    logger.info("SkillCurator Paper Experiments v3 — 9 Benchmarks")
    logger.info("=" * 70)
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Benchmarks: {ALL_BENCHMARKS}")
    logger.info(f"Tiers: {json.dumps({k: v['tier'] for k, v in BENCHMARK_TIERS.items()})}")

    load_env()
    start_time = time.time()
    llm_client = LLMClient({"temperature": 0.7, "max_tokens": 2048, "timeout": 120})

    # Preflight checks
    try:
        resp = llm_client.chat([{"role": "user", "content": "Say OK"}], temperature=0.0, max_tokens=5)
        logger.info(f"API: '{resp.strip()}' OK")
    except Exception as exc:
        logger.error(f"API failed: {exc}")
        sys.exit(1)

    try:
        get_embed_model()
        logger.info("Embedding model: OK")
    except Exception as exc:
        logger.error(f"Embedding model failed: {exc}")
        sys.exit(1)

    all_results = {"meta": {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v3",
        "config": {
            "benchmark_tiers": BENCHMARK_TIERS,
            "methods": METHODS,
            "max_traj_steps": MAX_TRAJ_STEPS,
            "dedup_threshold": 0.75,
            "phenomena_benchmarks": PHENOMENA_BENCHMARKS,
            "attention_benchmarks": ATTENTION_BENCHMARKS,
        },
    }}

    output_path = Path("experiments/paper_v3_results.json")

    def save_intermediate():
        elapsed = time.time() - start_time
        stats = llm_client.stats
        all_results["meta"]["elapsed_seconds"] = elapsed
        all_results["meta"]["total_api_calls"] = stats["total_calls"]
        all_results["meta"]["total_tokens"] = stats["total_tokens"]
        output_path.write_text(json.dumps(all_results, indent=2, default=str, ensure_ascii=False))
        logger.info(f"  [Saved: {elapsed:.0f}s, {stats['total_tokens']:,} tokens]")

    experiments = [
        ("main_experiment", run_main_experiment),
        ("attention_independence", run_attention_independence),
        ("phenomena", run_phenomenon_experiments),
        ("bound_tightening", run_bound_tightening),
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

    # Final Summary
    elapsed = time.time() - start_time
    stats = llm_client.stats

    logger.info("\n" + "=" * 70)
    logger.info("PAPER v3 — 9 BENCHMARK FINAL SUMMARY")
    logger.info("=" * 70)

    # Table 1: Main results across 9 benchmarks
    if "main_experiment" in all_results and "error" not in all_results["main_experiment"]:
        me = all_results["main_experiment"]
        logger.info(f"\n{'='*80}")
        logger.info(f"TABLE 1: Main Results (6 methods × 9 benchmarks)")
        logger.info(f"{'='*80}")
        logger.info(f"{'Benchmark':<18} {'Tier':>4} {'B0':>7} {'B1':>7} {'B2':>7} {'A1':>7} {'A2':>7} {'A3':>7} {'Dedup':>8}")
        logger.info("-" * 90)
        for bench_name in ALL_BENCHMARKS:
            if bench_name in me and isinstance(me[bench_name], dict) and "methods" in me[bench_name]:
                bd = me[bench_name]
                tier = bd.get("tier", "?")
                row = f"{bench_name:<18} T{tier:>3}"
                for method in METHODS:
                    if method in bd["methods"]:
                        em = bd["methods"][method]["avg_em"]
                        row += f" {em:>6.1%}"
                    else:
                        row += f" {'N/A':>6}"
                ds = bd.get("dedup_stats", {})
                reduction = ds.get("reduction_pct", 0)
                row += f" {reduction:>7.1%}"
                logger.info(row)
            elif bench_name in me:
                logger.info(f"{bench_name:<18} {'ERROR':>7}")

        # Aggregate by tier
        for tier in [1, 2, 3]:
            tier_ems = {m: [] for m in METHODS}
            for bench_name, bd in me.items():
                if isinstance(bd, dict) and bd.get("tier") == tier and "methods" in bd:
                    for m in METHODS:
                        if m in bd["methods"]:
                            tier_ems[m].append(bd["methods"][m]["avg_em"])
            if any(tier_ems[m] for m in METHODS):
                row = f"{'Tier '+str(tier)+' avg':<18} T{tier:>3}"
                for m in METHODS:
                    if tier_ems[m]:
                        row += f" {avg(tier_ems[m]):>6.1%}"
                    else:
                        row += f" {'N/A':>6}"
                row += f" {'':>7}"
                logger.info(row)

        # Overall average
        overall_ems = {m: [] for m in METHODS}
        for bench_name, bd in me.items():
            if isinstance(bd, dict) and "methods" in bd:
                for m in METHODS:
                    if m in bd["methods"]:
                        overall_ems[m].append(bd["methods"][m]["avg_em"])
        row = f"{'OVERALL avg':<18} {'':>4}"
        for m in METHODS:
            if overall_ems[m]:
                row += f" {avg(overall_ems[m]):>6.1%}"
            else:
                row += f" {'N/A':>6}"
        logger.info(row)

    # Table 2: δ_attention independence
    if "attention_independence" in all_results and "error" not in all_results["attention_independence"]:
        ai = all_results["attention_independence"]
        logger.info(f"\n{'='*80}")
        logger.info(f"TABLE 2: δ_attention Independence")
        logger.info(f"{'='*80}")
        for bench_name, bd in ai.items():
            if isinstance(bd, dict) and "strategies" in bd:
                logger.info(f"\n  [{bench_name}] SR range={bd['sr_range']:.1%}, "
                            f"F1 range={bd['f1_range']:.3f}, "
                            f"verified={bd['independence_verified']}")
                for strat, data in bd["strategies"].items():
                    logger.info(f"    {strat:<25}: EM={data['avg_em']:.1%}±{data['std_em']:.3f}, "
                                f"F1={data['avg_f1']:.3f}")

    # Phenomena
    if "phenomena" in all_results and "error" not in all_results["phenomena"]:
        ph = all_results["phenomena"]
        logger.info(f"\n{'='*80}")
        logger.info(f"PHENOMENA RESULTS")
        logger.info(f"{'='*80}")

        if "phase_transition" in ph:
            for bench_name, pt in ph["phase_transition"].items():
                if isinstance(pt, dict) and "peak_info" in pt:
                    logger.info(f"\n  Phase Transition [{bench_name}]:")
                    for sname, info in pt["peak_info"].items():
                        logger.info(f"    {sname}: peak N={info['peak_size']}, EM={info['peak_em']:.1%}")

        if "compaction_cliff" in ph:
            for bench_name, cc in ph["compaction_cliff"].items():
                if isinstance(cc, dict):
                    logger.info(f"\n  Compaction Cliff [{bench_name}]: avg ratio={cc.get('avg_cliff_ratio', 1):.2f}")

        if "scissors_effect" in ph:
            for bench_name, se in ph["scissors_effect"].items():
                if isinstance(se, dict):
                    logger.info(f"\n  Scissors Effect [{bench_name}]:")
                    for lib_name in ["append_only", "skillos", "ours"]:
                        if lib_name in se:
                            d = se[lib_name]
                            logger.info(f"    {lib_name}: |S|={d['final_total']}, "
                                        f"N_eff={d['final_effective']}, ratio={d['final_ratio']:.3f}")

    # Bound tightening
    if "bound_tightening" in all_results and "error" not in all_results["bound_tightening"]:
        bt = all_results["bound_tightening"]
        logger.info(f"\n  Bound Tightening:")
        for bench_name, bd in bt.items():
            if isinstance(bd, dict) and "final_delta" in bd:
                logger.info(f"    [{bench_name}] final δ={bd['final_delta']:.4f}")

    logger.info(f"\n{'='*70}")
    logger.info(f"Total: {stats['total_calls']} calls, {stats['total_tokens']:,} tokens, "
                f"{elapsed:.0f}s ({elapsed/3600:.1f}h)")
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results["meta"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_intermediate()
    logger.info(f"Final results: {output_path}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
