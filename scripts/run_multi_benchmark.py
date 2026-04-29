#!/usr/bin/env python3
"""
SkillForge multi-benchmark experiment runner (v6).

Key design for statistical significance (NO artificial noise):
1. No-skill baseline: control group to prove skill value.
2. N=10 per benchmark: sufficient for statistical significance.
3. Natural trajectories: no noise injection — differentiation comes
   from how each variant processes the naturally verbose trajectory.
4. Cross-benchmark transfer: HotpotQA skills → MuSiQue tasks.
5. Fine-grained quality: 5-dimension strict rubric.
6. Separate Self / Cross / Transfer metrics.

Core hypothesis (why hybrid should win — Evidence-as-Filter v6):
- traj→skill: sees the FULL verbose trajectory → information overload
  → produces over-specific or vague skills → low Cross/Transfer
- memory→skill: sees ONLY compressed memory → clean but uses ALL
  memories indiscriminately (including low-value ones) → medium Cross
- hybrid→skill (v6): uses trajectory to FILTER and RANK memories,
  then induces skill from ONLY the best memories → memory-level
  abstraction + better memory selection → highest Cross/Transfer

Transfer pairs:
- HotpotQA skills → MuSiQue tasks (multi-hop → harder multi-hop)
- GSM8K skills → TriviaQA tasks (math → factoid, should fail)
- TriviaQA skills → HotpotQA tasks (single-hop → multi-hop, partial)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from benchmarks.loader import BenchmarkLoader
from src.evaluation.evaluator import SkillEvaluator
from src.memory.compressor import create_compressor
from src.models import SkillEvalResult, TransformVariant
from src.skill_induction.factory import create_inducer
from src.trajectory.collector import TrajectoryCollector
from src.utils.config import load_config, load_env
from src.utils.io import save_json
from src.utils.llm import LLMClient
from src.utils.logging import setup_logger

# Primary benchmarks for skill induction
BENCHMARKS = ["hotpotqa", "gsm8k", "triviaqa"]
VARIANTS = ["traj_to_skill", "memory_to_skill", "hybrid_to_skill"]

# Cross-benchmark transfer pairs: source_benchmark → target_benchmark
# Designed to test generalisation across task types
TRANSFER_PAIRS = {
    "hotpotqa": "musique",      # multi-hop QA → harder multi-hop QA
    "gsm8k": "triviaqa",        # math reasoning → factoid QA (should fail)
    "triviaqa": "hotpotqa",     # single-hop → multi-hop (partial transfer)
}
TRANSFER_NUM_TASKS = 5  # Number of target tasks for transfer evaluation


def evaluate_baseline(
    llm_client: LLMClient,
    tasks: list[dict],
    evaluator: SkillEvaluator,
) -> float:
    """
    Evaluate tasks WITHOUT any skill injection (control group).

    Returns the average LLM-judge score (0-10 normalised to 0-1).
    """
    logger.info("Evaluating no-skill baseline...")
    total_score = 0.0
    count = 0

    for task in tasks:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question directly.",
            },
            {"role": "user", "content": task.get("description", "")},
        ]
        try:
            response = llm_client.chat(messages)
            score = evaluator._llm_judge_score(
                task_description=task.get("description", ""),
                expected_answer=task.get("expected", ""),
                actual_response=response[:1000],
                skill_name="(no skill — baseline)",
            )
            total_score += score
            count += 1
        except Exception as exc:
            logger.error(f"Baseline eval failed for {task.get('task_id', '?')}: {exc}")
            count += 1

    avg = total_score / (count * 10.0) if count > 0 else 0.0
    logger.info(f"Baseline score: {avg:.1%} ({count} tasks)")
    return round(avg, 4)


def load_transfer_tasks(
    source_benchmark: str,
    num_tasks: int = 5,
) -> list[dict]:
    """Load target benchmark tasks for cross-benchmark transfer evaluation."""
    target_benchmark = TRANSFER_PAIRS.get(source_benchmark)
    if not target_benchmark:
        return []

    try:
        loader = BenchmarkLoader({"name": target_benchmark, "num_samples": num_tasks})
        tasks = loader.load()
        logger.info(
            f"Loaded {len(tasks)} transfer tasks: "
            f"{source_benchmark} → {target_benchmark}"
        )
        return tasks
    except Exception as exc:
        logger.error(f"Failed to load transfer tasks for {target_benchmark}: {exc}")
        return []


def evaluate_transfer(
    skills_and_trajs: list[tuple],
    transfer_tasks: list[dict],
    evaluator: SkillEvaluator,
    variant_name: str,
) -> float:
    """
    Evaluate skills on a DIFFERENT benchmark's tasks (cross-benchmark transfer).

    This is the key metric for differentiating variants:
    - traj→skill overfits to source benchmark → low transfer
    - memory→skill denoises → higher transfer
    - hybrid→skill balances → highest transfer

    Returns average transfer score (0-1).
    """
    if not transfer_tasks or not skills_and_trajs:
        return 0.0

    all_scores: list[float] = []

    for skill_idx, (skill, _) in enumerate(skills_and_trajs):
        for task in transfer_tasks:
            task_result = evaluator._validate_with_skill(skill, {
                "task_id": f"transfer_{task['task_id']}",
                "description": task["description"],
                "expected": task.get("expected", ""),
            })
            score = task_result.get("score", 0.0)
            all_scores.append(score)

    avg = sum(all_scores) / (len(all_scores) * 10.0) if all_scores else 0.0
    logger.info(
        f"  [{variant_name}] Transfer: {avg:.1%} "
        f"({len(all_scores)} evaluations)"
    )
    return round(avg, 4)


def run_single_benchmark(
    benchmark_name: str,
    llm_client: LLMClient,
    config: dict,
    experiment_dir: Path,
    num_samples: int = 10,
) -> dict[str, dict[str, float]]:
    """Run all 3 variants + baseline on a single benchmark."""
    logger.info(f"\n{'#' * 70}\n# Benchmark: {benchmark_name}\n{'#' * 70}")

    bench_config = {"name": benchmark_name, "num_samples": num_samples}
    loader = BenchmarkLoader(bench_config)
    tasks = loader.load()
    logger.info(f"Loaded {len(tasks)} tasks for {benchmark_name}")

    collector = TrajectoryCollector(llm_client, config.get("trajectory", {}))
    compressor = create_compressor(
        config.get("memory", {}).get("framework", "mem0"),
        llm_client,
        config.get("memory", {}),
    )
    evaluator = SkillEvaluator(llm_client, config.get("evaluation", {}))

    # Phase 0: No-skill baseline
    baseline_score = evaluate_baseline(llm_client, tasks, evaluator)

    # Phase 0.5: Load transfer tasks (from a different benchmark)
    transfer_tasks = load_transfer_tasks(benchmark_name, TRANSFER_NUM_TASKS)

    # Phase 1: Collect trajectories and induce skills
    all_trajectories = []
    all_memories = []
    all_skills: dict[str, list[tuple]] = {v: [] for v in VARIANTS}

    for task_idx, task in enumerate(tasks):
        logger.info(
            f"\n[{benchmark_name}] Phase 1 — Task {task_idx + 1}/{len(tasks)}: "
            f"{task['task_id']}"
        )

        trajectory = collector.collect(
            task_id=task["task_id"],
            task_description=task["description"],
        )
        all_trajectories.append(trajectory)
        logger.info(
            f"  Trajectory: {trajectory.num_steps} steps, "
            f"errors={trajectory.error_rate:.0%}"
        )

        memory = compressor.compress(trajectory)
        all_memories.append(memory)
        logger.info(f"  Memory: {memory.num_entries} entries")

        for variant_name in VARIANTS:
            variant = TransformVariant(variant_name)
            inducer = create_inducer(
                variant, llm_client, config.get("skill_induction", {})
            )
            skill = inducer.induce(trajectory=trajectory, memory=memory)
            all_skills[variant_name].append((skill, trajectory))

            skill_dir = experiment_dir / benchmark_name / "skills" / variant_name
            save_json(skill, skill_dir / f"{task['task_id']}.json")
            logger.info(
                f"  [{variant_name}] skill='{skill.name}', "
                f"chars={skill.compactness}"
            )

    # Phase 2: Within-benchmark evaluation (Self + Cross)
    logger.info(f"\n[{benchmark_name}] Phase 2 — Within-benchmark evaluation")

    variant_results: dict[str, list[SkillEvalResult]] = {v: [] for v in VARIANTS}

    for variant_name in VARIANTS:
        skills_and_trajs = all_skills[variant_name]

        for skill_idx, (skill, source_traj) in enumerate(skills_and_trajs):
            cross_tasks = []
            for task_idx, task in enumerate(tasks):
                if task_idx != skill_idx:
                    cross_tasks.append({
                        "task_id": f"{task['task_id']}_cross",
                        "description": task["description"],
                        "expected": task.get("expected", ""),
                    })

            self_task = {
                "task_id": f"{tasks[skill_idx]['task_id']}_self",
                "description": tasks[skill_idx]["description"],
                "expected": tasks[skill_idx].get("expected", ""),
            }

            all_val_tasks = [self_task] + cross_tasks
            eval_result = evaluator.evaluate_skill(
                skill=skill,
                validation_tasks=all_val_tasks,
                source_trajectory=source_traj,
            )
            variant_results[variant_name].append(eval_result)

            scores = [d.get("score", 0) for d in eval_result.validation_details]
            self_score = scores[0] if scores else 0
            cross_scores = scores[1:] if len(scores) > 1 else []
            avg_cross = (
                sum(cross_scores) / len(cross_scores) if cross_scores else 0
            )
            logger.info(
                f"  [{variant_name}][task {skill_idx}] "
                f"self={self_score:.0f}/10, "
                f"cross_avg={avg_cross:.1f}/10, "
                f"quality={eval_result.transfer_score:.2f}, "
                f"compression={eval_result.compression_ratio:.1f}x"
            )

    # Phase 3: Cross-benchmark transfer evaluation
    logger.info(
        f"\n[{benchmark_name}] Phase 3 — Cross-benchmark transfer "
        f"({benchmark_name} → {TRANSFER_PAIRS.get(benchmark_name, '?')})"
    )
    transfer_scores: dict[str, float] = {}
    for variant_name in VARIANTS:
        transfer_scores[variant_name] = evaluate_transfer(
            all_skills[variant_name],
            transfer_tasks,
            evaluator,
            variant_name,
        )

    # Aggregate metrics
    metrics: dict[str, dict[str, float]] = {}

    # Baseline
    metrics["no_skill_baseline"] = {
        "num_tasks": len(tasks),
        "task_score": baseline_score,
        "self_score": baseline_score,
        "cross_score": baseline_score,
        "transfer_score": baseline_score,
        "quality_score": 0.0,
        "compression_ratio": 0.0,
    }

    for variant_name, results in variant_results.items():
        if not results:
            continue
        n = len(results)
        avg_task_score = sum(r.success_rate for r in results) / n
        avg_quality = sum(r.transfer_score for r in results) / n
        avg_compression = sum(r.compression_ratio for r in results) / n

        all_self_scores = []
        all_cross_scores = []
        for r in results:
            details = r.validation_details
            if details:
                all_self_scores.append(details[0].get("score", 0))
                for d in details[1:]:
                    all_cross_scores.append(d.get("score", 0))

        avg_self = (
            sum(all_self_scores) / len(all_self_scores)
            if all_self_scores
            else 0
        )
        avg_cross = (
            sum(all_cross_scores) / len(all_cross_scores)
            if all_cross_scores
            else 0
        )

        metrics[variant_name] = {
            "num_tasks": n,
            "task_score": round(avg_task_score, 4),
            "self_score": round(avg_self / 10.0, 4),
            "cross_score": round(avg_cross / 10.0, 4),
            "transfer_score": transfer_scores.get(variant_name, 0.0),
            "quality_score": round(avg_quality, 4),
            "compression_ratio": round(avg_compression, 2),
        }

    save_json(metrics, experiment_dir / benchmark_name / "metrics.json")
    return metrics


def print_results_table(
    all_metrics: dict[str, dict[str, dict[str, float]]],
) -> str:
    """Print and return a formatted results table."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 130)
    lines.append(
        "SkillForge Experiment Results — v6 "
        "(baseline + Evidence-as-Filter hybrid + transfer)"
    )
    lines.append("=" * 130)
    lines.append("")
    lines.append(
        f"{'Benchmark':<12} {'Variant':<25} "
        f"{'Self':>6} {'Cross':>7} {'Transfer':>9} {'Quality':>8} "
        f"{'Compress':>9} {'Tasks':>6}"
    )
    lines.append("-" * 130)

    for benchmark, variant_metrics in all_metrics.items():
        first = True
        target = TRANSFER_PAIRS.get(benchmark, "?")
        for variant, m in variant_metrics.items():
            bm_col = f"{benchmark}" if first else ""
            ss = f"{m.get('self_score', 0):.0%}"
            cs = f"{m.get('cross_score', 0):.0%}"
            ts = f"{m.get('transfer_score', 0):.0%}"
            qs = (
                f"{m.get('quality_score', 0):.0%}"
                if m.get("quality_score", 0) > 0
                else "  —"
            )
            cr = (
                f"{m.get('compression_ratio', 0):.1f}x"
                if m.get("compression_ratio", 0) > 0
                else "  —"
            )
            n = f"{int(m.get('num_tasks', 0))}"
            lines.append(
                f"{bm_col:<12} {variant:<25} "
                f"{ss:>6} {cs:>7} {ts:>9} {qs:>8} "
                f"{cr:>9} {n:>6}"
            )
            first = False
        lines.append(f"  (transfer target: {target})")
        lines.append("-" * 130)

    # Cross-benchmark averages
    lines.append("")
    lines.append("Cross-Benchmark Averages:")
    lines.append("-" * 90)
    lines.append(
        f"  {'Variant':<25} {'Self':>6} {'Cross':>7} "
        f"{'Transfer':>9} {'Quality':>8} {'Compress':>9}"
    )
    lines.append(f"  {'-' * 70}")

    # Baseline average
    bl_scores = []
    for bm_metrics in all_metrics.values():
        if "no_skill_baseline" in bm_metrics:
            bl_scores.append(
                bm_metrics["no_skill_baseline"].get("self_score", 0)
            )
    if bl_scores:
        avg_bl = sum(bl_scores) / len(bl_scores)
        lines.append(
            f"  {'no_skill_baseline':<25} {avg_bl:>5.0%} {avg_bl:>7.0%} "
            f"{avg_bl:>9.0%} {'  —':>8} {'  —':>9}"
        )

    for variant in VARIANTS:
        self_scores = []
        cross_scores = []
        transfer_scores_list = []
        quality_scores = []
        compression_ratios = []
        for bm_metrics in all_metrics.values():
            if variant in bm_metrics:
                m = bm_metrics[variant]
                self_scores.append(m.get("self_score", 0))
                cross_scores.append(m.get("cross_score", 0))
                transfer_scores_list.append(m.get("transfer_score", 0))
                quality_scores.append(m.get("quality_score", 0))
                compression_ratios.append(m.get("compression_ratio", 0))
        if self_scores:
            avg_ss = sum(self_scores) / len(self_scores)
            avg_cs = sum(cross_scores) / len(cross_scores)
            avg_ts = sum(transfer_scores_list) / len(transfer_scores_list)
            avg_qs = sum(quality_scores) / len(quality_scores)
            avg_cr = sum(compression_ratios) / len(compression_ratios)
            lines.append(
                f"  {variant:<25} {avg_ss:>5.0%} {avg_cs:>7.0%} "
                f"{avg_ts:>9.0%} {avg_qs:>8.0%} {avg_cr:>8.1f}x"
            )

    lines.append("")
    lines.append("Legend:")
    lines.append(
        "  Self     = LLM-judge score on the SAME task (self-consistency)"
    )
    lines.append(
        "  Cross    = LLM-judge score on OTHER tasks within same benchmark"
    )
    lines.append(
        "  Transfer = LLM-judge score on a DIFFERENT benchmark's tasks "
        "(cross-benchmark generalisation)"
    )
    lines.append(
        "  Quality  = 5-dimension skill structure quality (strict rubric)"
    )
    lines.append("  Compress = chars(trajectory) / chars(skill)")
    lines.append("")
    lines.append("Transfer pairs:")
    for src, tgt in TRANSFER_PAIRS.items():
        lines.append(f"  {src} → {tgt}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkillForge Multi-Benchmark Experiment v6"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(BENCHMARKS),
        help="Comma-separated benchmark names",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples per benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config file name (without .yaml)",
    )
    args = parser.parse_args()

    num_samples = args.num_samples
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    load_env()
    config = load_config(args.config)
    setup_logger(config.get("output", {}).get("log_level", "INFO"))

    logger.info("SkillForge Multi-Benchmark Experiment v6")
    logger.info(f"  Benchmarks: {benchmarks}")
    logger.info(f"  Samples per benchmark: {num_samples}")
    logger.info(f"  Variants: {VARIANTS} + no_skill_baseline")
    logger.info(f"  Transfer pairs: {TRANSFER_PAIRS}")
    logger.info(
        f"  Design: natural trajectories (no noise), Evidence-as-Filter "
        f"hybrid, cross-benchmark transfer"
    )

    llm_client = LLMClient(config.get("llm", {}))

    experiment_dir = (
        Path(config.get("output", {}).get("experiment_dir", "./experiments"))
        / "multi_benchmark_v6"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for benchmark in benchmarks:
        try:
            metrics = run_single_benchmark(
                benchmark, llm_client, config, experiment_dir, num_samples
            )
            all_metrics[benchmark] = metrics
        except Exception as exc:
            logger.error(f"Benchmark {benchmark} failed: {exc}")
            import traceback

            traceback.print_exc()
            all_metrics[benchmark] = {}

    elapsed = time.time() - start_time

    save_json(all_metrics, experiment_dir / "all_metrics.json")

    table = print_results_table(all_metrics)
    logger.info(table)
    (experiment_dir / "results_table.txt").write_text(table, encoding="utf-8")

    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info(f"LLM stats: {llm_client.stats}")
    logger.info(f"Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
