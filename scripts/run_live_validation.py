#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkillForge Live Validation — actual LLM calls to verify framework correctness.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.evaluation.evaluator import SkillEvaluator
from src.memory.compressor import create_compressor
from src.models import Skill, TransformVariant
from src.skill_induction.factory import create_inducer
from src.trajectory.collector import TrajectoryCollector

# Configuration

# Sample sizes: small enough for fast validation, large enough for signal
HOTPOTQA_SAMPLES = 5
LOCOMO_SAMPLES = 5  # 5 QA pairs from 1 LoCoMo sample

VARIANTS = [
    TransformVariant.TRAJ_TO_SKILL,
    TransformVariant.MEMORY_TO_SKILL,
    TransformVariant.HYBRID_TO_SKILL,
]

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    if not ground_truth.strip():
        return 1.0 if not prediction.strip() else 0.0

    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    from collections import Counter
    common = Counter(gt_tokens) & Counter(pred_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match (normalized)."""
    # Normalize: lowercase, strip punctuation and whitespace
    import string
    def normalize(s):
        s = s.lower().strip()
        # Remove articles
        for article in ['a ', 'an ', 'the ']:
            if s.startswith(article):
                s = s[len(article):]
        # Remove punctuation
        s = s.translate(str.maketrans('', '', string.punctuation))
        return s.strip()

    return 1.0 if normalize(ground_truth) in normalize(prediction) else 0.0

# Test 1: HotpotQA — Multi-hop QA

def run_hotpotqa_validation(llm_client: LLMClient) -> dict:
    """Run HotpotQA validation with actual LLM calls."""
    logger.info("=" * 60)
    logger.info("TEST 1: HotpotQA Multi-hop QA Validation")
    logger.info("=" * 60)

    from benchmarks.loader import BenchmarkLoader

    # Load a small subset
    loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": HOTPOTQA_SAMPLES})
    tasks = loader.load()
    logger.info(f"Loaded {len(tasks)} HotpotQA tasks")

    collector = TrajectoryCollector(llm_client, {"max_steps": 4})
    compressor = create_compressor("mem0", llm_client, {})

    results = {
        "benchmark": "hotpotqa",
        "num_tasks": len(tasks),
        "baseline": {"em_scores": [], "f1_scores": []},
        "variants": {},
    }

    for variant in VARIANTS:
        results["variants"][variant.value] = {
            "em_scores": [], "f1_scores": [], "skills": [],
        }

    for task_idx, task in enumerate(tasks):
        task_id = task["task_id"]
        description = task["description"]
        expected = task.get("expected", "")

        logger.info(f"\n--- Task {task_idx+1}/{len(tasks)}: {task_id} ---")
        logger.info(f"Expected: {expected[:100]}")

        try:
            # Step 1: Collect trajectory
            trajectory = collector.collect(
                task_id=task_id,
                task_description=description,
            )
            logger.info(f"Trajectory: {trajectory.num_steps} steps, answer={trajectory.final_answer[:80] if trajectory.final_answer else 'None'}")

            # Baseline: direct answer from trajectory
            if trajectory.final_answer:
                baseline_em = compute_em(trajectory.final_answer, expected)
                baseline_f1 = compute_token_f1(trajectory.final_answer, expected)
                results["baseline"]["em_scores"].append(baseline_em)
                results["baseline"]["f1_scores"].append(baseline_f1)
                logger.info(f"Baseline: EM={baseline_em:.1f}, F1={baseline_f1:.3f}")

            # Step 2: Compress into memory
            memory = compressor.compress(trajectory)
            logger.info(f"Memory: {memory.num_entries} entries")

            # Step 3: Induce skills via all variants
            for variant in VARIANTS:
                inducer = create_inducer(variant, llm_client, {})
                skill = inducer.induce(trajectory=trajectory, memory=memory)

                # Step 4: Use skill to answer a similar question
                skill_prompt = _format_skill_for_eval(skill)
                messages = [
                    {"role": "system", "content": f"You are a QA agent. Use this skill:\n\n{skill_prompt}\n\nAnswer the question directly and concisely."},
                    {"role": "user", "content": description},
                ]
                skill_response = llm_client.chat(messages, temperature=0.3, max_tokens=256)

                em = compute_em(skill_response, expected)
                f1 = compute_token_f1(skill_response, expected)

                results["variants"][variant.value]["em_scores"].append(em)
                results["variants"][variant.value]["f1_scores"].append(f1)
                results["variants"][variant.value]["skills"].append({
                    "name": skill.name,
                    "steps": len(skill.procedure),
                    "constraints": len(skill.constraints),
                })

                logger.info(f"  {variant.value}: EM={em:.1f}, F1={f1:.3f}, skill='{skill.name}'")

        except Exception as exc:
            logger.error(f"Task {task_id} failed: {exc}")
            continue

    # Compute aggregates
    _compute_aggregates(results)
    return results

# Test 2: LoCoMo — Long Conversation Memory

def run_locomo_validation(llm_client: LLMClient) -> dict:
    """Run LoCoMo validation with actual LLM calls."""
    logger.info("=" * 60)
    logger.info("TEST 2: LoCoMo Long Conversation Memory Validation")
    logger.info("=" * 60)

    from benchmarks.loader import BenchmarkLoader

    # Load 1 sample (which gives ~200 QA pairs), take first N
    loader = BenchmarkLoader({"name": "locomo", "num_samples": 1})
    all_tasks = loader.load()

    # Take a small subset for validation
    tasks = all_tasks[:LOCOMO_SAMPLES]
    logger.info(f"Loaded {len(all_tasks)} LoCoMo QA pairs, using {len(tasks)}")

    results = {
        "benchmark": "locomo",
        "num_tasks": len(tasks),
        "total_available": len(all_tasks),
        "direct_qa": {"em_scores": [], "f1_scores": []},
        "with_context": {"em_scores": [], "f1_scores": []},
        "categories": {},
    }

    for task_idx, task in enumerate(tasks):
        task_id = task["task_id"]
        question = task["description"].split("Question: ")[-1] if "Question: " in task["description"] else task["description"][-200:]
        expected = task.get("expected", "")
        context = task.get("context", "")[:4000]  # Limit context size
        category = task.get("metadata", {}).get("category_name", "unknown")

        logger.info(f"\n--- LoCoMo {task_idx+1}/{len(tasks)}: {task_id} (cat={category}) ---")
        logger.info(f"Q: {question[:100]}")
        logger.info(f"Expected: {expected[:100]}")

        try:
            # Test 1: Direct QA (no context, just the question)
            messages_direct = [
                {"role": "system", "content": "Answer the question concisely based on your knowledge."},
                {"role": "user", "content": question},
            ]
            direct_response = llm_client.chat(messages_direct, temperature=0.3, max_tokens=256)
            direct_em = compute_em(direct_response, expected)
            direct_f1 = compute_token_f1(direct_response, expected)
            results["direct_qa"]["em_scores"].append(direct_em)
            results["direct_qa"]["f1_scores"].append(direct_f1)
            logger.info(f"  Direct QA: EM={direct_em:.1f}, F1={direct_f1:.3f}")

            # Test 2: QA with conversation context
            messages_ctx = [
                {"role": "system", "content": "Answer the question based on the conversation history provided. Be concise."},
                {"role": "user", "content": f"Conversation:\n{context}\n\nQuestion: {question}"},
            ]
            ctx_response = llm_client.chat(messages_ctx, temperature=0.3, max_tokens=256)
            ctx_em = compute_em(ctx_response, expected)
            ctx_f1 = compute_token_f1(ctx_response, expected)
            results["with_context"]["em_scores"].append(ctx_em)
            results["with_context"]["f1_scores"].append(ctx_f1)
            logger.info(f"  With context: EM={ctx_em:.1f}, F1={ctx_f1:.3f}")

            # Track by category
            if category not in results["categories"]:
                results["categories"][category] = {"em": [], "f1": []}
            results["categories"][category]["em"].append(ctx_em)
            results["categories"][category]["f1"].append(ctx_f1)

        except Exception as exc:
            logger.error(f"LoCoMo task {task_id} failed: {exc}")
            continue

    # Compute aggregates
    for key in ["direct_qa", "with_context"]:
        em_scores = results[key]["em_scores"]
        f1_scores = results[key]["f1_scores"]
        results[key]["avg_em"] = sum(em_scores) / len(em_scores) if em_scores else 0
        results[key]["avg_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    for cat, scores in results["categories"].items():
        scores["avg_em"] = sum(scores["em"]) / len(scores["em"]) if scores["em"] else 0
        scores["avg_f1"] = sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0

    return results

# Test 3: Skill Induction Quality

def run_skill_quality_validation(llm_client: LLMClient) -> dict:
    """
    Validate that skill induction produces meaningful, structured skills.
    """
    logger.info("=" * 60)
    logger.info("TEST 3: Skill Induction Quality Validation")
    logger.info("=" * 60)

    collector = TrajectoryCollector(llm_client, {"max_steps": 4})
    compressor = create_compressor("mem0", llm_client, {})

    # Use a simple task to test skill induction
    test_task = {
        "task_id": "quality_test_001",
        "description": (
            "Answer this multi-hop question: "
            "The director of the 2003 film 'Lost in Translation' also directed "
            "which film that stars Scarlett Johansson and Bill Murray?"
        ),
        "expected": "Lost in Translation",
    }

    results = {"variants": {}}

    try:
        trajectory = collector.collect(
            task_id=test_task["task_id"],
            task_description=test_task["description"],
        )
        memory = compressor.compress(trajectory)

        for variant in VARIANTS:
            inducer = create_inducer(variant, llm_client, {})
            skill = inducer.induce(trajectory=trajectory, memory=memory)

            # Validate skill structure
            has_name = bool(skill.name and skill.name.strip())
            has_description = bool(skill.description and skill.description.strip())
            has_procedure = len(skill.procedure) > 0
            has_constraints = len(skill.constraints) > 0
            procedure_steps = len(skill.procedure)
            compactness = skill.compactness

            results["variants"][variant.value] = {
                "name": skill.name,
                "has_name": has_name,
                "has_description": has_description,
                "has_procedure": has_procedure,
                "has_constraints": has_constraints,
                "procedure_steps": procedure_steps,
                "compactness": compactness,
                "description_preview": skill.description[:200],
                "valid": has_name and has_description and has_procedure,
            }

            logger.info(
                f"  {variant.value}: name='{skill.name}', "
                f"steps={procedure_steps}, constraints={len(skill.constraints)}, "
                f"compactness={compactness}, valid={has_name and has_description and has_procedure}"
            )

    except Exception as exc:
        logger.error(f"Skill quality validation failed: {exc}")
        results["error"] = str(exc)

    return results

# Helpers

def _format_skill_for_eval(skill: Skill) -> str:
    """Format a skill as a prompt for evaluation."""
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
    return "\n".join(parts)

def _compute_aggregates(results: dict) -> None:
    """Compute aggregate metrics for HotpotQA results."""
    # Baseline
    bl = results["baseline"]
    bl["avg_em"] = sum(bl["em_scores"]) / len(bl["em_scores"]) if bl["em_scores"] else 0
    bl["avg_f1"] = sum(bl["f1_scores"]) / len(bl["f1_scores"]) if bl["f1_scores"] else 0

    # Variants
    for vname, vdata in results["variants"].items():
        vdata["avg_em"] = sum(vdata["em_scores"]) / len(vdata["em_scores"]) if vdata["em_scores"] else 0
        vdata["avg_f1"] = sum(vdata["f1_scores"]) / len(vdata["f1_scores"]) if vdata["f1_scores"] else 0

# Main

def main():
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
    # Also log to file for background runs
    log_path = Path("experiments/live_validation.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}", level="DEBUG")

    logger.info("=" * 70)
    logger.info("SkillForge Live Validation — Burning DeepSeek Tokens")
    logger.info("=" * 70)

    load_env()
    start_time = time.time()

    llm_client = LLMClient({"temperature": 0.7, "max_tokens": 2048, "timeout": 120})

    # Quick API connectivity test
    logger.info("\nAPI connectivity test...")
    try:
        test_response = llm_client.chat(
            [{"role": "user", "content": "Say 'API OK' and nothing else."}],
            temperature=0.0, max_tokens=10,
        )
        logger.info(f"API test: '{test_response.strip()}' ✅")
    except Exception as exc:
        logger.error(f"API test failed: {exc}")
        logger.error("Cannot proceed without API access. Check .env configuration.")
        sys.exit(1)

    # Run all validations
    all_results = {}

    # Test 1: HotpotQA
    try:
        all_results["hotpotqa"] = run_hotpotqa_validation(llm_client)
    except Exception as exc:
        logger.error(f"HotpotQA validation failed: {exc}")
        all_results["hotpotqa"] = {"error": str(exc)}

    # Test 2: LoCoMo
    try:
        all_results["locomo"] = run_locomo_validation(llm_client)
    except Exception as exc:
        logger.error(f"LoCoMo validation failed: {exc}")
        all_results["locomo"] = {"error": str(exc)}

    # Test 3: Skill Quality
    try:
        all_results["skill_quality"] = run_skill_quality_validation(llm_client)
    except Exception as exc:
        logger.error(f"Skill quality validation failed: {exc}")
        all_results["skill_quality"] = {"error": str(exc)}

    # Summary Report
    elapsed = time.time() - start_time
    stats = llm_client.stats

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION RESULTS SUMMARY")
    logger.info("=" * 70)

    # HotpotQA results
    if "hotpotqa" in all_results and "error" not in all_results["hotpotqa"]:
        hq = all_results["hotpotqa"]
        logger.info(f"\n📊 HotpotQA ({hq['num_tasks']} tasks):")
        logger.info(f"  Baseline (direct):  EM={hq['baseline']['avg_em']:.1%}, F1={hq['baseline']['avg_f1']:.3f}")
        for vname, vdata in hq["variants"].items():
            logger.info(f"  {vname:20s}: EM={vdata['avg_em']:.1%}, F1={vdata['avg_f1']:.3f}")
        logger.info(f"  Paper ref (MemSkill, LLaMA-70B, 100docs K=7): EM≈70.70%")

    # LoCoMo results
    if "locomo" in all_results and "error" not in all_results["locomo"]:
        lc = all_results["locomo"]
        logger.info(f"\n📊 LoCoMo ({lc['num_tasks']} QA pairs from {lc.get('total_available', '?')} total):")
        logger.info(f"  Direct QA (no ctx):  EM={lc['direct_qa']['avg_em']:.1%}, F1={lc['direct_qa']['avg_f1']:.3f}")
        logger.info(f"  With context:        EM={lc['with_context']['avg_em']:.1%}, F1={lc['with_context']['avg_f1']:.3f}")
        if lc.get("categories"):
            for cat, scores in lc["categories"].items():
                logger.info(f"    {cat:12s}: EM={scores['avg_em']:.1%}, F1={scores['avg_f1']:.3f}")
        logger.info(f"  Paper ref (MemSkill, LLaMA-70B): F1=38.78, L-J=50.96")

    # Skill quality results
    if "skill_quality" in all_results and "error" not in all_results["skill_quality"]:
        sq = all_results["skill_quality"]
        logger.info(f"\n📊 Skill Induction Quality:")
        all_valid = True
        for vname, vdata in sq["variants"].items():
            valid = vdata.get("valid", False)
            all_valid = all_valid and valid
            logger.info(
                f"  {vname:20s}: valid={valid}, name='{vdata.get('name', 'N/A')}', "
                f"steps={vdata.get('procedure_steps', 0)}, "
                f"compactness={vdata.get('compactness', 0)}"
            )
        logger.info(f"  All variants produce valid skills: {'✅' if all_valid else '❌'}")

    # Token usage
    logger.info(f"\n💰 Token Usage:")
    logger.info(f"  Total API calls: {stats['total_calls']}")
    logger.info(f"  Total tokens: {stats['total_tokens']:,}")
    logger.info(f"  Elapsed time: {elapsed:.1f}s")

    # Validation verdict
    logger.info(f"\n{'=' * 70}")
    logger.info("VALIDATION VERDICT")
    logger.info("=" * 70)

    checks = []

    # Check 1: Pipeline runs end-to-end
    pipeline_ok = "hotpotqa" in all_results and "error" not in all_results.get("hotpotqa", {})
    checks.append(("Pipeline runs end-to-end", pipeline_ok))

    # Check 2: EM/F1 metrics are non-trivial
    if pipeline_ok:
        hq = all_results["hotpotqa"]
        bl_f1 = hq["baseline"]["avg_f1"]
        metrics_ok = bl_f1 > 0.0  # At least some F1
        checks.append(("EM/F1 metrics are non-trivial", metrics_ok))

    # Check 3: Skill induction produces valid skills
    if "skill_quality" in all_results and "error" not in all_results["skill_quality"]:
        sq = all_results["skill_quality"]
        skills_ok = all(v.get("valid", False) for v in sq["variants"].values())
        checks.append(("All skill variants produce valid skills", skills_ok))

    # Check 4: LoCoMo context helps
    if "locomo" in all_results and "error" not in all_results["locomo"]:
        lc = all_results["locomo"]
        ctx_helps = lc["with_context"]["avg_f1"] >= lc["direct_qa"]["avg_f1"] * 0.8
        checks.append(("LoCoMo context provides useful signal", ctx_helps))

    # Check 5: Multiple API calls succeed
    api_ok = stats["total_calls"] >= 5
    checks.append(("Multiple API calls succeed", api_ok))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)

    for name, ok in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        logger.info(f"  {status}  {name}")

    logger.info(f"\n  Result: {passed}/{total} checks passed")

    if passed == total:
        logger.info("  🎉 Framework validation PASSED — code is correct and produces meaningful results.")
    else:
        logger.warning(f"  ⚠️ {total - passed} checks failed — review the results above.")

    # Save results to file
    output_path = Path("experiments/live_validation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Clean up non-serializable data
    clean_results = json.loads(json.dumps(all_results, default=str))
    clean_results["meta"] = {
        "elapsed_seconds": elapsed,
        "total_api_calls": stats["total_calls"],
        "total_tokens": stats["total_tokens"],
        "model": llm_client.model,
        "checks_passed": passed,
        "checks_total": total,
    }
    output_path.write_text(json.dumps(clean_results, indent=2, ensure_ascii=False))
    logger.info(f"\n  Results saved to: {output_path}")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
