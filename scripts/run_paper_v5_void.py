#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillCurator Paper Experiments v5 — Void-Case (c_∅) Augmentation."""
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml
from loguru import logger

from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.memory.compressor import create_compressor
from src.models import Skill, TransformVariant
from src.skill_induction.factory import create_inducer
from src.trajectory.collector import TrajectoryCollector
from src.utils.skill_formatter import format_skill_library, FormattingConfig
from src.curation.void_case import (
    VoidCaseConfig,
    VoidCaseStats,
    apply_void_case,
    compute_lambda_x,
)

# Configuration loader

CONFIG_PATH = PROJECT_ROOT / "configs" / "paper_v5.yaml"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

CONFIG = load_config()
SEED = CONFIG.get("seed", 42)
random.seed(SEED)
np.random.seed(SEED)

# Embedding model (lazy, shared)

EMBED_MODEL = None

def get_embed_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        EMBED_MODEL = SentenceTransformer(CONFIG["embedding"]["model"])
        logger.info(f"Loaded embedding: {CONFIG['embedding']['model']}")
    return EMBED_MODEL

# Metrics (copied from v4 for self-containment)

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

# Skill embedding helpers (shared with v4)

def skill_to_text(s: Skill) -> str:
    parts = [s.name, s.description]
    parts.extend(s.procedure[:3])
    if s.constraints:
        parts.extend(s.constraints[:2])
    return " ".join(parts)

def compute_skill_embeddings(skills: list[Skill]) -> np.ndarray:
    if not skills:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_embed_model()
    texts = [skill_to_text(s) for s in skills]
    return model.encode(texts, normalize_embeddings=True).astype(np.float32)

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

# Skill induction (same as v4)

def induce_skills_from_tasks(
    llm_client: LLMClient,
    tasks: list[dict],
    max_traj_steps: int = 6,
    label: str = "train",
) -> list[Skill]:
    collector = TrajectoryCollector(llm_client, {"max_steps": max_traj_steps})
    compressor = create_compressor("mem0", llm_client, {})
    skill_bank: list[Skill] = []
    for idx, task in enumerate(tasks):
        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info(f"  [{label}] Inducing {idx+1}/{len(tasks)}: {task['task_id']}")
        try:
            traj = collector.collect(task_id=task["task_id"], task_description=task["description"])
            memory = compressor.compress(traj)
            inducer = create_inducer(TransformVariant.HYBRID_TO_SKILL, llm_client, {})
            skill = inducer.induce(trajectory=traj, memory=memory)
            skill.source_tasks = [task["task_id"]]
            skill_bank.append(skill)
        except Exception as exc:
            logger.warning(f"  [{label}] Failed {task['task_id']}: {exc}")
    logger.info(f"  [{label}] Built {len(skill_bank)} skills from {len(tasks)} tasks")
    return skill_bank

def save_skill_bank(skill_bank: list[Skill], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for s in skill_bank:
        data.append({
            "name": s.name,
            "description": s.description,
            "procedure": list(s.procedure),
            "constraints": list(s.constraints),
            "facts": list(s.facts) if hasattr(s, "facts") else [],
            "rules": list(s.rules) if hasattr(s, "rules") else [],
            "preconditions": list(s.preconditions) if hasattr(s, "preconditions") else [],
            "source_tasks": list(s.source_tasks),
        })
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_skill_bank(path: Path) -> list[Skill] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    skills = []
    for d in data:
        s = Skill(
            name=d.get("name", ""),
            description=d.get("description", ""),
            procedure=d.get("procedure", []),
            constraints=d.get("constraints", []),
            facts=d.get("facts", []),
            rules=d.get("rules", []),
            preconditions=d.get("preconditions", []),
            source_tasks=d.get("source_tasks", []),
            source_variant=TransformVariant.HYBRID_TO_SKILL,
        )
        skills.append(s)
    return skills

# Skill Library Formatting (same as v4 for B0/B2/A1/A3)

def format_skills_B0(skills: list[Skill]) -> str:
    return ""

def format_skills_B1(skills: list[Skill]) -> str:
    parts = []
    for i, s in enumerate(skills):
        parts.append(f"Skill {i+1}: {s.name}\n{s.description}\nProcedure: {'; '.join(s.procedure)}")
        if s.constraints:
            parts.append(f"Constraints: {'; '.join(s.constraints)}")
        parts.append("")
    return "\n".join(parts)

def format_skills_B2(skills: list[Skill]) -> str:
    return format_skills_B1(skills)

def format_skills_A3(skills: list[Skill]) -> str:
    return format_skill_library(skills, config=FormattingConfig(strategy="sandwich_compact"))

def format_skills_A1(skills: list[Skill]) -> str:
    # A1 uses plain B1 formatting after dedup (compaction without attention ops)
    return format_skills_B1(skills)

# Evaluation Core: caches s_max so we can sweep τ_void post-hoc

class V5Evaluator:
    """Evaluates a method on a list of test tasks, caching s_max per task"""

    def __init__(self, llm_client: LLMClient, top_k: int = 5):
        self.llm = llm_client
        self.top_k = top_k

    def _retrieve_with_simcache(
        self, skills: list[Skill], skill_embs: np.ndarray, query: str
    ) -> tuple[list[Skill], float]:
        """Top-k retrieval, also returning s_max for c_∅ gating."""
        if not skills or skill_embs.shape[0] == 0:
            return [], 0.0
        model = get_embed_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        sims = skill_embs @ q_emb  # (N,)
        s_max = float(sims.max())
        top_idx = np.argsort(sims)[::-1][: self.top_k]
        return [skills[i] for i in top_idx], s_max

    @staticmethod
    def _build_messages(skill_block: str, query: str) -> list[dict]:
        if not skill_block:
            return [
                {"role": "system",
                 "content": "Answer the question directly and concisely. Give ONLY the answer, no explanation."},
                {"role": "user", "content": query},
            ]
        return [
            {"role": "system",
             "content": f"Use the following skills to help answer:\n\n{skill_block}\n\nAnswer directly and concisely. Give ONLY the answer."},
            {"role": "user", "content": query},
        ]

    def evaluate(
        self,
        method: str,
        skill_bank: list[Skill],
        skill_embs: np.ndarray,
        test_tasks: list[dict],
        with_void: bool,
        label: str = "",
    ) -> dict:
        """Evaluate a method, optionally with c_∅ post-hoc sweepable."""
        per_task = []
        for idx, task in enumerate(test_tasks):
            desc = task["description"]
            expected = task.get("expected", "")
            try:
                # Step 1: retrieve top-k + compute s_max
                if method == "B0":
                    top_skills, s_max = [], 0.0
                else:
                    top_skills, s_max = self._retrieve_with_simcache(
                        skill_bank, skill_embs, desc
                    )
                    # Apply method-specific dedup BEFORE formatting (A1/A3)
                    if method in ("A1", "A3") and top_skills:
                        top_skills = deduplicate_skills_embedding(
                            top_skills, threshold=0.75
                        )

                # Step 2: format skill block per method
                if method == "B0":
                    skill_block = ""
                elif method == "B2":
                    skill_block = format_skills_B2(top_skills)
                elif method == "A1":
                    skill_block = format_skills_A1(top_skills)
                elif method == "A3":
                    skill_block = format_skills_A3(top_skills)
                else:
                    skill_block = format_skills_B1(top_skills)

                tokens = len(skill_block.split())

                # Step 3: ALWAYS run injection LLM call (or zero-shot for B0)
                msgs_inject = self._build_messages(skill_block, desc)
                pred_inject = self.llm.chat(msgs_inject, temperature=0.1, max_tokens=128)

                # Step 4: if with_void, ALSO run zero-shot baseline call
                pred_void = pred_inject if method == "B0" else None
                if with_void and method != "B0":
                    msgs_void = self._build_messages("", desc)
                    pred_void = self.llm.chat(msgs_void, temperature=0.1, max_tokens=128)

                em_inject = compute_em(pred_inject, expected)
                f1_inject = compute_token_f1(pred_inject, expected)
                em_void = compute_em(pred_void, expected) if pred_void is not None else em_inject
                f1_void = compute_token_f1(pred_void, expected) if pred_void is not None else f1_inject

                per_task.append({
                    "task_id": task.get("task_id", f"t{idx}"),
                    "expected": expected[:100],
                    "s_max": s_max,
                    "tokens_inject": tokens,
                    "pred_inject": pred_inject[:200],
                    "pred_void": pred_void[:200] if pred_void else None,
                    "em_inject": em_inject,
                    "f1_inject": f1_inject,
                    "em_void": em_void,
                    "f1_void": f1_void,
                })

            except Exception as exc:
                logger.warning(f"  [{label}:{method}] task {idx} failed: {exc}")
                per_task.append({
                    "task_id": task.get("task_id", f"t{idx}"),
                    "expected": "",
                    "s_max": 0.0,
                    "tokens_inject": 0,
                    "pred_inject": "",
                    "pred_void": None,
                    "em_inject": 0.0, "f1_inject": 0.0,
                    "em_void": 0.0, "f1_void": 0.0,
                })
        return {"per_task": per_task}

    @staticmethod
    def aggregate_with_tau(per_task: list[dict], tau: float) -> dict:
        """Apply c_∅ gate at threshold τ, aggregate metrics."""
        em_list, f1_list, tok_list = [], [], []
        n_void = 0
        for r in per_task:
            if r["s_max"] >= tau:
                em_list.append(r["em_inject"])
                f1_list.append(r["f1_inject"])
                tok_list.append(r["tokens_inject"])
            else:
                em_list.append(r["em_void"])
                f1_list.append(r["f1_void"])
                tok_list.append(0)  # zero-shot has no skill tokens
                n_void += 1
        return {
            "tau": tau,
            "n": len(em_list),
            "n_void": n_void,
            "void_rate": n_void / max(len(em_list), 1),
            "avg_em": avg(em_list),
            "avg_f1": avg(f1_list),
            "avg_tokens": avg(tok_list),
            "std_em": std(em_list),
            "std_f1": std(f1_list),
        }

# Main pipeline

def run_v5_main(llm_client: LLMClient) -> dict:
    """Run v5 main experiment: 6 methods × 7 benchmarks + τ sweep."""
    benches = CONFIG["benchmarks"]
    methods = CONFIG["methods"]
    sweep_taus = CONFIG["void_case"]["sweep_tau"]
    primary_tau = CONFIG["void_case"]["tau_void"]
    skill_bank_dir = Path(CONFIG["output"]["skill_banks_dir"])
    save_banks = CONFIG["output"].get("save_skill_banks", True)
    top_k = CONFIG["retrieval"]["top_k"]

    from benchmarks.loader import BenchmarkLoader

    evaluator = V5Evaluator(llm_client, top_k=top_k)
    all_results: dict[str, Any] = {"meta": {
        "version": "v5",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "primary_tau": primary_tau,
        "sweep_taus": sweep_taus,
        "benchmarks": list(benches.keys()),
        "methods": methods,
        "seed": SEED,
    }, "main_experiment": {}}

    t0 = time.time()
    total_calls = 0

    for bench_name, bench_cfg in benches.items():
        logger.info(f"\n{'='*70}\nBenchmark: {bench_name}\n{'='*70}")
        train_n = bench_cfg["train"]
        test_n = bench_cfg["test"]
        primary_metric = bench_cfg.get("primary_metric", "em")

        # ----- Load benchmark -----
        try:
            loader = BenchmarkLoader({"name": bench_name, "num_samples": train_n + test_n})
            tasks = loader.load()
        except Exception as exc:
            logger.error(f"  Load failed: {exc}")
            all_results["main_experiment"][bench_name] = {"error": str(exc)}
            continue
        if len(tasks) < 5:
            all_results["main_experiment"][bench_name] = {
                "error": f"insufficient data: {len(tasks)} tasks"
            }
            continue
        if len(tasks) < train_n + test_n:
            train_n_eff = min(train_n, len(tasks) * 2 // 3)
            test_n_eff = len(tasks) - train_n_eff
        else:
            train_n_eff, test_n_eff = train_n, test_n
        train_tasks = tasks[:train_n_eff]
        test_tasks = tasks[train_n_eff:train_n_eff + test_n_eff]
        logger.info(f"  split: {len(train_tasks)} train + {len(test_tasks)} test")

        # ----- Build / load skill bank -----
        bank_path = skill_bank_dir / f"{bench_name}.json"
        skill_bank = load_skill_bank(bank_path) if save_banks else None
        if skill_bank is None:
            skill_bank = induce_skills_from_tasks(
                llm_client, train_tasks, label=bench_name
            )
            if save_banks:
                save_skill_bank(skill_bank, bank_path)
                logger.info(f"  saved skill bank → {bank_path}")
        else:
            logger.info(f"  loaded skill bank from {bank_path} ({len(skill_bank)} skills)")

        skill_embs = compute_skill_embeddings(skill_bank)

        # ----- Evaluate each method -----
        bench_result: dict[str, Any] = {
            "benchmark": bench_name,
            "primary_metric": primary_metric,
            "train_size": len(train_tasks),
            "test_size": len(test_tasks),
            "skill_bank_size": len(skill_bank),
            "methods": {},
        }

        # Map method name → (base_method, with_void)
        method_specs = {
            "B0":      ("B0", False),
            "B2":      ("B2", False),
            "A3":      ("A3", False),
            "B2+void": ("B2", True),
            "A1+void": ("A1", True),
            "A3+void": ("A3", True),
        }

        # Cache per-task results keyed by base_method to avoid duplicate
        # injection-side LLM calls (e.g. A3 and A3+void share inject result).
        # We always call evaluator.evaluate() with with_void=True for the
        # +void variants (since we need both pred_inject and pred_void).
        # For non-void variants, we just compute the inject side.
        per_task_cache: dict[str, list[dict]] = {}

        for full_method in methods:
            base, with_void = method_specs[full_method]
            cache_key = f"{base}_{int(with_void)}"
            if cache_key not in per_task_cache:
                logger.info(f"  → Running {full_method} (base={base}, void={with_void})")
                eval_result = evaluator.evaluate(
                    method=base,
                    skill_bank=skill_bank,
                    skill_embs=skill_embs,
                    test_tasks=test_tasks,
                    with_void=with_void,
                    label=bench_name,
                )
                per_task_cache[cache_key] = eval_result["per_task"]
            per_task = per_task_cache[cache_key]

            # ----- Aggregate at primary τ (or ignore for non-void) -----
            if with_void:
                agg = V5Evaluator.aggregate_with_tau(per_task, primary_tau)
            else:
                # Without void: just use inject results directly
                em_list = [r["em_inject"] for r in per_task]
                f1_list = [r["f1_inject"] for r in per_task]
                tok_list = [r["tokens_inject"] for r in per_task]
                agg = {
                    "tau": None,
                    "n": len(em_list),
                    "n_void": 0,
                    "void_rate": 0.0,
                    "avg_em": avg(em_list),
                    "avg_f1": avg(f1_list),
                    "avg_tokens": avg(tok_list),
                    "std_em": std(em_list),
                    "std_f1": std(f1_list),
                }

            # ----- τ sweep (only for void variants) -----
            sweep = []
            if with_void:
                for tau in sweep_taus:
                    sweep.append(V5Evaluator.aggregate_with_tau(per_task, tau))

            # ----- Per-task records (compact for json size) -----
            per_task_compact = [{
                "s_max": r["s_max"],
                "em_inject": r["em_inject"],
                "f1_inject": r["f1_inject"],
                "em_void": r["em_void"],
                "f1_void": r["f1_void"],
                "tokens_inject": r["tokens_inject"],
            } for r in per_task]

            bench_result["methods"][full_method] = {
                "primary": agg,
                "sweep": sweep,
                "per_task": per_task_compact,
                "primary_tau": primary_tau if with_void else None,
            }
            logger.info(
                f"    {full_method}: EM={agg['avg_em']:.1%} "
                f"F1={agg['avg_f1']:.3f} tokens={agg['avg_tokens']:.0f} "
                f"void_rate={agg['void_rate']:.1%}"
            )

        all_results["main_experiment"][bench_name] = bench_result

        # Persist intermediate result after every benchmark (crash safety)
        intermediate_path = Path(CONFIG["output"]["experiment_dir"]) / "paper_v5_void_results.json"
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        all_results["meta"]["elapsed_seconds"] = time.time() - t0
        with open(intermediate_path, "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"  ✓ checkpoint → {intermediate_path}")

    all_results["meta"]["elapsed_seconds"] = time.time() - t0
    return all_results

# Entry

def main():
    load_env()
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    log_path = Path(CONFIG["output"]["experiment_dir"]) / CONFIG["output"]["log_file"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level=CONFIG["output"]["log_level"], rotation="100 MB")

    logger.info("=" * 70)
    logger.info("SkillCurator Paper Experiments v5 — Void-Case (c_∅)")
    logger.info("=" * 70)
    logger.info(f"  config: {CONFIG_PATH}")
    logger.info(f"  benchmarks: {list(CONFIG['benchmarks'].keys())}")
    logger.info(f"  methods: {CONFIG['methods']}")
    logger.info(f"  primary τ_void: {CONFIG['void_case']['tau_void']}")
    logger.info(f"  τ sweep: {CONFIG['void_case']['sweep_tau']}")

    llm_client = LLMClient(CONFIG["llm"])

    try:
        results = run_v5_main(llm_client)
    except Exception as exc:
        logger.error(f"FATAL: {exc}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Final save
    out_path = Path(CONFIG["output"]["experiment_dir"]) / CONFIG["output"]["results_file"]
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\n✓ Final results → {out_path}")
    logger.info(f"  total elapsed: {results['meta']['elapsed_seconds']:.0f}s")

if __name__ == "__main__":
    main()
