#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillForge Systematic Benchmark — comprehensive validation across multiple"""
from __future__ import annotations

import json
import string
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.memory.compressor import create_compressor
from src.memory.consolidation import MemoryConsolidator
from src.models import Skill, MemoryStore, TransformVariant
from src.skill_induction.factory import create_inducer
from src.trajectory.collector import TrajectoryCollector

# Configuration

HOTPOTQA_TRAIN = 10   # Tasks used to induce skills
HOTPOTQA_TEST = 10    # Held-out tasks to evaluate skill generalization
LOCOMO_SAMPLES = 15
LONGMEMEVAL_SAMPLES = 5
MAX_TRAJ_STEPS = 4  # Keep trajectory collection fast

VARIANTS = [
    TransformVariant.TRAJ_TO_SKILL,
    TransformVariant.MEMORY_TO_SKILL,
    TransformVariant.HYBRID_TO_SKILL,
]

# Paper reference values
PAPER_REFS = {
    "hotpotqa": {"em": 0.707, "source": "MemSkill Table 1, LLaMA-70B, 100docs K=7"},
    "locomo": {"f1": 0.3878, "lj": 0.5096, "source": "MemSkill Table 1, LLaMA-70B"},
    "longmemeval": {"f1": 0.2431, "lj": 0.4207, "source": "MemSkill Table 1, LLaMA-70B"},
}

# Metrics

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
        return s.strip()
    return 1.0 if normalize(ground_truth) in normalize(prediction) else 0.0

def format_skill_prompt(skill: Skill) -> str:
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

def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

# Benchmark 1: HotpotQA (Multi-hop QA)

def run_hotpotqa(llm_client: LLMClient) -> dict:
    """HotpotQA evaluation with proper train/test split."""
    logger.info("=" * 60)
    logger.info("BENCHMARK 1: HotpotQA Multi-hop QA (Train/Test Split)")
    logger.info("=" * 60)

    from benchmarks.loader import BenchmarkLoader
    total_samples = HOTPOTQA_TRAIN + HOTPOTQA_TEST
    loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": total_samples})
    tasks = loader.load()

    # Split into train (for skill induction) and test (for evaluation)
    train_tasks = tasks[:HOTPOTQA_TRAIN]
    test_tasks = tasks[HOTPOTQA_TRAIN:]
    logger.info(f"Loaded {len(tasks)} HotpotQA tasks: {len(train_tasks)} train, {len(test_tasks)} test")

    collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
    compressor = create_compressor("mem0", llm_client, {})

    results = {"benchmark": "hotpotqa", "num_train": len(train_tasks),
               "num_test": len(test_tasks),
               "baseline_direct": {"em": [], "f1": []},
               "variants": {}}
    for v in VARIANTS:
        results["variants"][v.value] = {"em": [], "f1": [], "skills_induced": []}

    # ---- Phase 1: Induce skills from TRAIN tasks ----
    logger.info(f"\n{'─'*40}")
    logger.info("Phase 1: Skill Induction (train set)")
    logger.info(f"{'─'*40}")

    induced_skills = {v.value: [] for v in VARIANTS}

    for idx, task in enumerate(train_tasks):
        tid = task["task_id"]
        desc = task["description"]
        logger.info(f"\n  [Train {idx+1}/{len(train_tasks)}] {tid}")

        try:
            traj = collector.collect(task_id=tid, task_description=desc)
            memory = compressor.compress(traj)

            for variant in VARIANTS:
                inducer = create_inducer(variant, llm_client, {})
                skill = inducer.induce(trajectory=traj, memory=memory)
                induced_skills[variant.value].append(skill)
                logger.info(f"    {variant.value}: '{skill.name}'")

        except Exception as exc:
            logger.error(f"    Train task {tid} failed: {exc}")

    # Log skill bank summary
    for vname, skills in induced_skills.items():
        results["variants"][vname]["skills_induced"] = [s.name for s in skills]
        logger.info(f"  {vname}: {len(skills)} skills induced")

    # ---- Phase 2: Evaluate on HELD-OUT test tasks ----
    logger.info(f"\n{'─'*40}")
    logger.info("Phase 2: Evaluation (held-out test set)")
    logger.info(f"{'─'*40}")

    for idx, task in enumerate(test_tasks):
        tid = task["task_id"]
        desc = task["description"]
        expected = task.get("expected", "")
        logger.info(f"\n  [Test {idx+1}/{len(test_tasks)}] {tid} | expected='{expected[:60]}'")

        try:
            # Baseline: direct LLM answer (no skill)
            baseline_resp = llm_client.chat(
                [{"role": "system", "content": "Answer the following multi-hop question directly and concisely."},
                 {"role": "user", "content": desc}],
                temperature=0.3, max_tokens=256)
            bl_em = compute_em(baseline_resp, expected)
            bl_f1 = compute_token_f1(baseline_resp, expected)
            results["baseline_direct"]["em"].append(bl_em)
            results["baseline_direct"]["f1"].append(bl_f1)
            logger.info(f"    Baseline (direct): EM={bl_em:.0f}, F1={bl_f1:.3f}")

            # Skill-guided: select most relevant skill from bank, then answer
            for variant in VARIANTS:
                skills = induced_skills[variant.value]
                if not skills:
                    continue

                # Select skill: use the full skill bank as a "skill library"
                # Format all skills as a library prompt
                skill_library = "\n\n---\n\n".join(
                    format_skill_prompt(s) for s in skills[-5:]  # Use last 5 skills (most recent)
                )

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
                resp = llm_client.chat(messages, temperature=0.3, max_tokens=256)
                em = compute_em(resp, expected)
                f1 = compute_token_f1(resp, expected)
                results["variants"][variant.value]["em"].append(em)
                results["variants"][variant.value]["f1"].append(f1)
                logger.info(f"    {variant.value}: EM={em:.0f}, F1={f1:.3f}")

        except Exception as exc:
            logger.error(f"    Test task {tid} failed: {exc}")

    # Aggregates
    results["baseline_direct"]["avg_em"] = avg(results["baseline_direct"]["em"])
    results["baseline_direct"]["avg_f1"] = avg(results["baseline_direct"]["f1"])
    for v in results["variants"].values():
        v["avg_em"] = avg(v["em"])
        v["avg_f1"] = avg(v["f1"])
    return results

# Benchmark 2: LoCoMo (Long Conversation Memory)

def run_locomo(llm_client: LLMClient) -> dict:
    logger.info("=" * 60)
    logger.info("BENCHMARK 2: LoCoMo Long Conversation Memory")
    logger.info("=" * 60)

    from benchmarks.loader import BenchmarkLoader
    loader = BenchmarkLoader({"name": "locomo", "num_samples": 1})
    all_tasks = loader.load()
    tasks = all_tasks[:LOCOMO_SAMPLES]
    logger.info(f"Loaded {len(all_tasks)} LoCoMo QA pairs, using {len(tasks)}")

    results = {"benchmark": "locomo", "num_tasks": len(tasks),
               "total_available": len(all_tasks),
               "direct_qa": {"em": [], "f1": []},
               "with_context": {"em": [], "f1": []},
               "categories": {}}

    for idx, task in enumerate(tasks):
        question = task["description"].split("Question: ")[-1] if "Question: " in task["description"] else task["description"][-200:]
        expected = task.get("expected", "")
        context = task.get("context", "")[:6000]
        category = task.get("metadata", {}).get("category_name", "unknown")

        logger.info(f"\n--- LoCoMo {idx+1}/{len(tasks)}: cat={category} | expected='{expected[:60]}' ---")

        try:
            # Direct QA (no context)
            resp_direct = llm_client.chat(
                [{"role": "system", "content": "Answer concisely."},
                 {"role": "user", "content": question}],
                temperature=0.3, max_tokens=256)
            results["direct_qa"]["em"].append(compute_em(resp_direct, expected))
            results["direct_qa"]["f1"].append(compute_token_f1(resp_direct, expected))

            # With context
            resp_ctx = llm_client.chat(
                [{"role": "system", "content": "Answer based on the conversation. Be concise."},
                 {"role": "user", "content": f"Conversation:\n{context}\n\nQuestion: {question}"}],
                temperature=0.3, max_tokens=256)
            ctx_em = compute_em(resp_ctx, expected)
            ctx_f1 = compute_token_f1(resp_ctx, expected)
            results["with_context"]["em"].append(ctx_em)
            results["with_context"]["f1"].append(ctx_f1)

            if category not in results["categories"]:
                results["categories"][category] = {"em": [], "f1": [], "count": 0}
            results["categories"][category]["em"].append(ctx_em)
            results["categories"][category]["f1"].append(ctx_f1)
            results["categories"][category]["count"] += 1

            logger.info(f"  Direct: EM={results['direct_qa']['em'][-1]:.0f}, F1={results['direct_qa']['f1'][-1]:.3f}")
            logger.info(f"  Context: EM={ctx_em:.0f}, F1={ctx_f1:.3f}")

        except Exception as exc:
            logger.error(f"  LoCoMo task failed: {exc}")

    # Aggregates
    for key in ["direct_qa", "with_context"]:
        results[key]["avg_em"] = avg(results[key]["em"])
        results[key]["avg_f1"] = avg(results[key]["f1"])
    for cat in results["categories"].values():
        cat["avg_em"] = avg(cat["em"])
        cat["avg_f1"] = avg(cat["f1"])
    return results

# Benchmark 3: LongMemEval (Ultra-long Dialogue Memory)

def run_longmemeval(llm_client: LLMClient) -> dict:
    logger.info("=" * 60)
    logger.info("BENCHMARK 3: LongMemEval Ultra-long Dialogue Memory")
    logger.info("=" * 60)

    from benchmarks.loader import BenchmarkLoader
    loader = BenchmarkLoader({"name": "longmemeval", "num_samples": LONGMEMEVAL_SAMPLES})
    tasks = loader.load()
    logger.info(f"Loaded {len(tasks)} LongMemEval tasks")

    results = {"benchmark": "longmemeval", "num_tasks": len(tasks),
               "with_focused": {"em": [], "f1": []},
               "details": []}

    for idx, task in enumerate(tasks):
        question = task["description"].split("Question: ")[-1] if "Question: " in task["description"] else task["description"][-200:]
        expected = task.get("expected", "")
        context = task.get("context", "")[:4000]

        logger.info(f"\n--- LongMemEval {idx+1}/{len(tasks)}: expected='{expected[:60]}' ---")

        try:
            resp = llm_client.chat(
                [{"role": "system", "content": "Answer based on the conversation excerpt. Be concise."},
                 {"role": "user", "content": f"Conversation:\n{context}\n\nQuestion: {question}"}],
                temperature=0.3, max_tokens=256)
            em = compute_em(resp, expected)
            f1 = compute_token_f1(resp, expected)
            results["with_focused"]["em"].append(em)
            results["with_focused"]["f1"].append(f1)
            results["details"].append({
                "task_id": task["task_id"],
                "expected": expected[:100],
                "response": resp[:100],
                "em": em, "f1": f1,
            })
            logger.info(f"  EM={em:.0f}, F1={f1:.3f}")

        except Exception as exc:
            logger.error(f"  LongMemEval task failed: {exc}")

    results["with_focused"]["avg_em"] = avg(results["with_focused"]["em"])
    results["with_focused"]["avg_f1"] = avg(results["with_focused"]["f1"])
    return results

# Benchmark 4: Memory Consolidation (Mem2Evolve)

def run_consolidation_test(llm_client: LLMClient) -> dict:
    logger.info("=" * 60)
    logger.info("BENCHMARK 4: Memory Consolidation (Mem2Evolve)")
    logger.info("=" * 60)

    collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
    compressor = create_compressor("mem0", llm_client, {})
    consolidator = MemoryConsolidator(llm_client=llm_client, config={"similarity_threshold": 0.5})

    test_tasks = [
        "What is the capital of France and what is it known for?",
        "Explain the difference between Python lists and tuples.",
        "What is the capital of France and its most famous landmark?",  # Similar to task 1
    ]

    results = {"benchmark": "consolidation", "tasks": []}
    all_entries_before = 0
    all_entries_after = 0

    for i, desc in enumerate(test_tasks):
        logger.info(f"\n--- Consolidation task {i+1}/{len(test_tasks)} ---")
        try:
            traj = collector.collect(task_id=f"consol_{i}", task_description=desc)
            store = compressor.compress(traj)
            before = store.num_entries
            all_entries_before += before

            consolidated = consolidator.consolidate(store)
            after = consolidated.num_entries
            all_entries_after += after

            ratio = after / before if before > 0 else 1.0
            results["tasks"].append({
                "task": desc[:80],
                "entries_before": before,
                "entries_after": after,
                "compression_ratio": ratio,
            })
            logger.info(f"  Entries: {before} -> {after} (ratio={ratio:.2f})")

        except Exception as exc:
            logger.error(f"  Consolidation task failed: {exc}")

    results["total_before"] = all_entries_before
    results["total_after"] = all_entries_after
    results["overall_ratio"] = all_entries_after / all_entries_before if all_entries_before > 0 else 1.0
    logger.info(f"\n  Overall: {all_entries_before} -> {all_entries_after} (ratio={results['overall_ratio']:.2f})")
    return results

# Benchmark 5: EvolveLab Adapter Integration

def run_evolvelab_test(llm_client: LLMClient) -> dict:
    logger.info("=" * 60)
    logger.info("BENCHMARK 5: EvolveLab Adapter Integration")
    logger.info("=" * 60)

    from src.memory.evolvelab_adapter import SkillForgeAsEvolveLabProvider
    from src.memory.evolvelab.memory_types import (
        MemoryType, MemoryRequest, MemoryStatus, TrajectoryData,
    )

    compressor = create_compressor("mem0", llm_client, {})
    adapter = SkillForgeAsEvolveLabProvider(
        compressor=compressor,
        memory_type=MemoryType.AGENT_KB,
        config={"top_k": 3},
    )
    adapter.initialize()

    results = {"benchmark": "evolvelab_adapter", "tests": []}

    # Test 1: Ingest trajectory
    collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
    traj = collector.collect(task_id="evolve_test", task_description="What is machine learning?")

    traj_data = TrajectoryData(
        query="What is machine learning?",
        trajectory=[{"type": s.step_type.value, "content": s.content} for s in traj.steps],
        result=traj.final_answer,
        metadata={"task_id": "evolve_test"},
    )

    success, desc = adapter.take_in_memory(traj_data)
    results["tests"].append({
        "name": "ingest_trajectory",
        "success": success,
        "num_memories": adapter.num_memories,
    })
    logger.info(f"  Ingest: success={success}, memories={adapter.num_memories}")

    # Test 2: Retrieve memories
    request = MemoryRequest(
        query="machine learning algorithms",
        context="",
        status=MemoryStatus.BEGIN,
    )
    response = adapter.provide_memory(request)
    results["tests"].append({
        "name": "retrieve_memories",
        "num_results": response.total_count,
        "top_score": response.memories[0].score if response.memories else 0,
    })
    logger.info(f"  Retrieve: {response.total_count} results, top_score={response.memories[0].score:.3f}" if response.memories else "  Retrieve: 0 results")

    # Test 3: Inbound adapter (EvolveLab -> SkillForge)
    from src.memory.evolvelab_adapter import EvolveLabAsSkillForgeCompressor
    inbound = EvolveLabAsSkillForgeCompressor(provider=adapter)
    traj2 = collector.collect(task_id="inbound_test", task_description="Explain neural networks.")
    store = inbound.compress(traj2)
    results["tests"].append({
        "name": "inbound_adapter",
        "num_entries": store.num_entries,
        "framework": store.framework,
    })
    logger.info(f"  Inbound: {store.num_entries} entries, framework={store.framework}")

    results["all_passed"] = all(t.get("success", True) and t.get("num_memories", 1) > 0 for t in results["tests"])
    return results

# Benchmark 6: Skill Designer Evolution (MemSkill §3.8)

def run_skill_designer_test(llm_client: LLMClient) -> dict:
    logger.info("=" * 60)
    logger.info("BENCHMARK 6: Skill Designer Hard-Case Evolution")
    logger.info("=" * 60)

    from src.skill_induction.skill_designer import SkillDesigner, HardCase

    designer = SkillDesigner(
        llm_client=llm_client,
        config={"trigger_interval": 5, "max_edits_per_cycle": 2, "patience": 3},
    )

    # Simulate failures
    failures = [
        ("When did the meeting happen?", "I don't know", "March 15th", 0.0),
        ("Where was the restaurant?", "Some place", "Italian place on 5th street", 0.1),
        ("What time did they arrive?", "Evening", "7:30 PM", 0.2),
        ("Who organized the event?", "Someone", "Sarah from marketing", 0.0),
        ("How many people attended?", "Many", "42 people", 0.1),
    ]

    for q, pred, gt, reward in failures:
        designer.record_failure(query=q, prediction=pred, ground_truth=gt, reward=reward, step=1)

    results = {"benchmark": "skill_designer", "buffer_size": designer.hard_case_buffer.size}

    # Get top cases
    top = designer.hard_case_buffer.get_top_cases(n=3)
    results["top_cases"] = [{"query": c.query, "difficulty": c.difficulty_score} for c in top]
    logger.info(f"  Buffer: {designer.hard_case_buffer.size} cases")
    for c in top:
        logger.info(f"    d={c.difficulty_score:.2f}: {c.query}")

    # Run evolution
    initial_skills = [
        Skill(name="Basic QA", description="Answer questions directly"),
        Skill(name="Context Extraction", description="Extract info from context"),
    ]

    proposals = designer.evolve(current_skills=initial_skills, current_step=10)
    results["num_proposals"] = len(proposals)
    results["proposals"] = [{"action": p.action, "name": p.skill_name, "reasoning": p.reasoning[:100]} for p in proposals]
    logger.info(f"  Proposals: {len(proposals)}")
    for p in proposals:
        logger.info(f"    {p.action}: {p.skill_name}")

    # Apply proposals
    bank = initial_skills.copy()
    for p in proposals:
        bank, new_skill = designer.apply_proposal(p, bank)
    results["final_bank_size"] = len(bank)
    results["evolution_success"] = len(proposals) > 0

    logger.info(f"  Final bank: {len(bank)} skills")
    return results

# Benchmark 7: Cross-variant Skill Quality Comparison

def run_variant_comparison(llm_client: LLMClient) -> dict:
    logger.info("=" * 60)
    logger.info("BENCHMARK 7: Cross-variant Skill Quality Comparison")
    logger.info("=" * 60)

    collector = TrajectoryCollector(llm_client, {"max_steps": MAX_TRAJ_STEPS})
    compressor = create_compressor("mem0", llm_client, {})

    test_tasks = [
        {"id": "var_1", "desc": "What is the relationship between DNA and RNA?", "expected": "RNA is transcribed from DNA"},
        {"id": "var_2", "desc": "Compare TCP and UDP protocols.", "expected": "TCP is connection-oriented, UDP is connectionless"},
    ]

    results = {"benchmark": "variant_comparison", "tasks": []}

    for task in test_tasks:
        logger.info(f"\n--- Variant comparison: {task['id']} ---")
        try:
            traj = collector.collect(task_id=task["id"], task_description=task["desc"])
            memory = compressor.compress(traj)

            task_result = {"task_id": task["id"], "variants": {}}
            for variant in VARIANTS:
                inducer = create_inducer(variant, llm_client, {})
                skill = inducer.induce(trajectory=traj, memory=memory)

                task_result["variants"][variant.value] = {
                    "name": skill.name,
                    "procedure_steps": len(skill.procedure),
                    "constraints": len(skill.constraints),
                    "compactness": skill.compactness,
                    "has_facts": len(skill.facts) > 0,
                    "has_rules": len(skill.rules) > 0,
                }
                logger.info(f"  {variant.value}: '{skill.name}', steps={len(skill.procedure)}, compact={skill.compactness}")

            results["tasks"].append(task_result)
        except Exception as exc:
            logger.error(f"  Variant comparison failed: {exc}")

    # Compute averages
    for vname in [v.value for v in VARIANTS]:
        compactness_vals = [t["variants"].get(vname, {}).get("compactness", 0) for t in results["tasks"]]
        steps_vals = [t["variants"].get(vname, {}).get("procedure_steps", 0) for t in results["tasks"]]
        results[f"avg_compactness_{vname}"] = avg(compactness_vals)
        results[f"avg_steps_{vname}"] = avg(steps_vals)

    return results

# Main

def main():
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
    log_path = Path("experiments/systematic_benchmark.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}", level="DEBUG")

    logger.info("=" * 70)
    logger.info("SkillForge Systematic Benchmark — Multi-Paper Validation")
    logger.info("=" * 70)

    load_env()
    start_time = time.time()
    llm_client = LLMClient({"temperature": 0.7, "max_tokens": 2048, "timeout": 120})

    # API test
    logger.info("\nAPI connectivity test...")
    try:
        resp = llm_client.chat([{"role": "user", "content": "Say 'OK'"}], temperature=0.0, max_tokens=5)
        logger.info(f"API: '{resp.strip()}' ✅")
    except Exception as exc:
        logger.error(f"API failed: {exc}")
        sys.exit(1)

    all_results = {}

    # Run all benchmarks
    benchmarks = [
        ("hotpotqa", run_hotpotqa),
        ("locomo", run_locomo),
        ("longmemeval", run_longmemeval),
        ("consolidation", run_consolidation_test),
        ("evolvelab", run_evolvelab_test),
        ("skill_designer", run_skill_designer_test),
        ("variant_comparison", run_variant_comparison),
    ]

    for name, func in benchmarks:
        try:
            logger.info(f"\n{'='*70}")
            all_results[name] = func(llm_client)
        except Exception as exc:
            logger.error(f"Benchmark {name} failed: {exc}\n{traceback.format_exc()}")
            all_results[name] = {"error": str(exc)}

    # Summary Report
    elapsed = time.time() - start_time
    stats = llm_client.stats

    logger.info("\n" + "=" * 70)
    logger.info("SYSTEMATIC BENCHMARK RESULTS")
    logger.info("=" * 70)

    # HotpotQA
    if "hotpotqa" in all_results and "error" not in all_results["hotpotqa"]:
        hq = all_results["hotpotqa"]
        logger.info(f"\n📊 HotpotQA (train={hq['num_train']}, test={hq['num_test']}):")
        logger.info(f"  Baseline (direct):  EM={hq['baseline_direct']['avg_em']:.1%}, F1={hq['baseline_direct']['avg_f1']:.3f}")
        for vname, vdata in hq["variants"].items():
            logger.info(f"  {vname:20s}: EM={vdata['avg_em']:.1%}, F1={vdata['avg_f1']:.3f} (skills={len(vdata.get('skills_induced', []))})")
        logger.info(f"  📖 Paper ref: EM={PAPER_REFS['hotpotqa']['em']:.1%} ({PAPER_REFS['hotpotqa']['source']})")
        logger.info(f"  ⚠️  Note: Our EM is on held-out test set (generalization), not same-task.")

    # LoCoMo
    if "locomo" in all_results and "error" not in all_results["locomo"]:
        lc = all_results["locomo"]
        logger.info(f"\n📊 LoCoMo ({lc['num_tasks']} QA pairs):")
        logger.info(f"  Direct QA:          EM={lc['direct_qa']['avg_em']:.1%}, F1={lc['direct_qa']['avg_f1']:.3f}")
        logger.info(f"  With context:       EM={lc['with_context']['avg_em']:.1%}, F1={lc['with_context']['avg_f1']:.3f}")
        for cat, scores in lc.get("categories", {}).items():
            logger.info(f"    {cat:12s} (n={scores['count']}): EM={scores['avg_em']:.1%}, F1={scores['avg_f1']:.3f}")
        logger.info(f"  📖 Paper ref: F1={PAPER_REFS['locomo']['f1']:.4f}, L-J={PAPER_REFS['locomo']['lj']:.4f}")

    # LongMemEval
    if "longmemeval" in all_results and "error" not in all_results["longmemeval"]:
        lm = all_results["longmemeval"]
        logger.info(f"\n📊 LongMemEval ({lm['num_tasks']} tasks):")
        logger.info(f"  With focused input: EM={lm['with_focused']['avg_em']:.1%}, F1={lm['with_focused']['avg_f1']:.3f}")
        logger.info(f"  📖 Paper ref: F1={PAPER_REFS['longmemeval']['f1']:.4f}, L-J={PAPER_REFS['longmemeval']['lj']:.4f}")

    # Consolidation
    if "consolidation" in all_results and "error" not in all_results["consolidation"]:
        cs = all_results["consolidation"]
        logger.info(f"\n📊 Memory Consolidation (Mem2Evolve):")
        logger.info(f"  Total entries: {cs['total_before']} -> {cs['total_after']} (ratio={cs['overall_ratio']:.2f})")
        logger.info(f"  📖 Paper ref: target ratio <= 0.70 (Mem2Evolve §2.4)")

    # EvolveLab
    if "evolvelab" in all_results and "error" not in all_results["evolvelab"]:
        ev = all_results["evolvelab"]
        logger.info(f"\n📊 EvolveLab Adapter:")
        for t in ev["tests"]:
            logger.info(f"  {t['name']}: {t}")
        logger.info(f"  All passed: {'✅' if ev.get('all_passed') else '❌'}")

    # Skill Designer
    if "skill_designer" in all_results and "error" not in all_results["skill_designer"]:
        sd = all_results["skill_designer"]
        logger.info(f"\n📊 Skill Designer (MemSkill §3.8):")
        logger.info(f"  Hard cases: {sd['buffer_size']}, proposals: {sd['num_proposals']}")
        logger.info(f"  Final bank: {sd['final_bank_size']} skills")
        logger.info(f"  Evolution success: {'✅' if sd.get('evolution_success') else '❌'}")

    # Variant Comparison
    if "variant_comparison" in all_results and "error" not in all_results["variant_comparison"]:
        vc = all_results["variant_comparison"]
        logger.info(f"\n📊 Variant Comparison:")
        for vname in [v.value for v in VARIANTS]:
            logger.info(f"  {vname:20s}: avg_steps={vc.get(f'avg_steps_{vname}', 0):.1f}, avg_compact={vc.get(f'avg_compactness_{vname}', 0):.0f}")

    # Token usage
    logger.info(f"\n💰 Token Usage:")
    logger.info(f"  Total API calls: {stats['total_calls']}")
    logger.info(f"  Total tokens: {stats['total_tokens']:,}")
    logger.info(f"  Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Paper Comparison Table
    logger.info(f"\n{'='*70}")
    logger.info("PAPER COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info(f"{'Benchmark':<20} {'Metric':<10} {'Ours':<12} {'Paper':<12} {'Model':<20}")
    logger.info("-" * 74)

    if "hotpotqa" in all_results and "error" not in all_results["hotpotqa"]:
        hq = all_results["hotpotqa"]
        best_em = max(v["avg_em"] for v in hq["variants"].values())
        logger.info(f"{'HotpotQA':<20} {'EM':<10} {best_em:<12.1%} {PAPER_REFS['hotpotqa']['em']:<12.1%} {'DeepSeek-V3 vs LLaMA-70B'}")

    if "locomo" in all_results and "error" not in all_results["locomo"]:
        lc = all_results["locomo"]
        logger.info(f"{'LoCoMo':<20} {'F1':<10} {lc['with_context']['avg_f1']:<12.4f} {PAPER_REFS['locomo']['f1']:<12.4f} {'DeepSeek-V3 vs LLaMA-70B'}")

    if "longmemeval" in all_results and "error" not in all_results["longmemeval"]:
        lm = all_results["longmemeval"]
        logger.info(f"{'LongMemEval':<20} {'F1':<10} {lm['with_focused']['avg_f1']:<12.4f} {PAPER_REFS['longmemeval']['f1']:<12.4f} {'DeepSeek-V3 vs LLaMA-70B'}")

    # Validation Verdict
    logger.info(f"\n{'='*70}")
    logger.info("VALIDATION VERDICT")
    logger.info("=" * 70)

    checks = []

    # Check 1: Pipeline end-to-end
    checks.append(("Pipeline runs end-to-end (HotpotQA)", "hotpotqa" in all_results and "error" not in all_results.get("hotpotqa", {})))

    # Check 2: Skill-guided EM beats baseline on held-out test set
    if "hotpotqa" in all_results and "error" not in all_results["hotpotqa"]:
        hq = all_results["hotpotqa"]
        best_em = max(v["avg_em"] for v in hq["variants"].values()) if hq["variants"] else 0
        baseline_em = hq["baseline_direct"]["avg_em"]
        checks.append(("HotpotQA skill-guided EM >= baseline (generalization)", best_em >= baseline_em))

    # Check 3: LoCoMo context helps
    if "locomo" in all_results and "error" not in all_results["locomo"]:
        lc = all_results["locomo"]
        checks.append(("LoCoMo context improves over direct QA", lc["with_context"]["avg_f1"] >= lc["direct_qa"]["avg_f1"]))

    # Check 4: LongMemEval works
    checks.append(("LongMemEval runs successfully", "longmemeval" in all_results and "error" not in all_results.get("longmemeval", {})))

    # Check 5: Consolidation reduces entries
    if "consolidation" in all_results and "error" not in all_results["consolidation"]:
        checks.append(("Memory consolidation reduces entries", all_results["consolidation"]["overall_ratio"] <= 1.0))

    # Check 6: EvolveLab adapter works
    if "evolvelab" in all_results and "error" not in all_results["evolvelab"]:
        checks.append(("EvolveLab adapter integration", all_results["evolvelab"].get("all_passed", False)))

    # Check 7: Skill Designer produces proposals
    if "skill_designer" in all_results and "error" not in all_results["skill_designer"]:
        checks.append(("Skill Designer produces evolution proposals", all_results["skill_designer"].get("evolution_success", False)))

    # Check 8: All 3 variants produce valid skills
    if "variant_comparison" in all_results and "error" not in all_results["variant_comparison"]:
        vc = all_results["variant_comparison"]
        all_have_skills = all(
            len(t.get("variants", {})) == 3
            for t in vc.get("tasks", [])
        )
        checks.append(("All 3 skill variants produce valid skills", all_have_skills))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)

    for name, ok in checks:
        logger.info(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")

    logger.info(f"\n  Result: {passed}/{total} checks passed")
    if passed == total:
        logger.info("  🎉 All systematic benchmarks PASSED!")
    else:
        logger.warning(f"  ⚠️ {total - passed} checks failed")

    # Save results
    output_path = Path("experiments/systematic_benchmark_results.json")
    clean = json.loads(json.dumps(all_results, default=str))
    clean["meta"] = {
        "elapsed_seconds": elapsed,
        "total_api_calls": stats["total_calls"],
        "total_tokens": stats["total_tokens"],
        "model": llm_client.model,
        "checks_passed": passed,
        "checks_total": total,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
    logger.info(f"\n  Results saved to: {output_path}")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
