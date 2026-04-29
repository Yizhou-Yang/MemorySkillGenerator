#!/usr/bin/env python3
"""
SkillForge MVP experiment runner.

Executes the full MVP pipeline:
1. Load benchmark tasks.
2. Collect trajectories.
3. Compress into structured memory.
4. Induce skills via all three variants.
5. Evaluate and compare results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the project root to the Python path
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
from src.utils.io import save_json, save_jsonl
from src.utils.llm import LLMClient
from src.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillForge MVP Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="mvp_locomo",
        help="Config file name (without .yaml extension)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: print config and exit without running the experiment",
    )
    args = parser.parse_args()

    # ===== Initialisation =====
    load_env()
    config = load_config(args.config)
    setup_logger(config.get("output", {}).get("log_level", "INFO"))

    logger.info(f"SkillForge MVP experiment started: config={args.config}")
    logger.info(f"Config:\n{json.dumps(config, indent=2, ensure_ascii=False)}")

    if args.dry_run:
        logger.info("Dry-run mode — exiting")
        return

    # ===== Initialise components =====
    llm_config = config.get("llm", {})
    llm_client = LLMClient(llm_config)

    collector = TrajectoryCollector(llm_client, config.get("trajectory", {}))
    compressor = create_compressor(
        config.get("memory", {}).get("framework", "mem0"),
        llm_client,
        config.get("memory", {}),
    )
    evaluator = SkillEvaluator(llm_client, config.get("evaluation", {}))

    # ===== Experiment output directory =====
    experiment_dir = Path(
        config.get("output", {}).get("experiment_dir", "./experiments")
    )
    experiment_dir = experiment_dir / args.config
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # ===== Load benchmark tasks =====
    benchmark_loader = BenchmarkLoader(config.get("benchmark", {}))
    tasks = benchmark_loader.load()
    logger.info(f"Loaded {len(tasks)} tasks")

    # ===== Pipeline execution =====
    all_results: dict[TransformVariant, list[SkillEvalResult]] = {
        TransformVariant.TRAJ_TO_SKILL: [],
        TransformVariant.MEMORY_TO_SKILL: [],
        TransformVariant.HYBRID_TO_SKILL: [],
    }

    for task in tasks:
        logger.info(
            f"\n{'=' * 60}\nProcessing task: {task['task_id']}\n{'=' * 60}"
        )

        # Step 1: Collect trajectory
        trajectory = collector.collect(
            task_id=task["task_id"],
            task_description=task["description"],
        )
        if config.get("output", {}).get("save_trajectories", True):
            save_json(
                trajectory,
                experiment_dir / "trajectories" / f"{task['task_id']}.json",
            )

        # Step 2: Compress into memory
        memory = compressor.compress(trajectory)
        if config.get("output", {}).get("save_memories", True):
            save_json(
                memory,
                experiment_dir / "memories" / f"{task['task_id']}.json",
            )

        # Step 3: Induce skills via all three variants
        variant_names = config.get("skill_induction", {}).get("variants", [])
        for variant_name in variant_names:
            variant = TransformVariant(variant_name)
            inducer = create_inducer(
                variant, llm_client, config.get("skill_induction", {})
            )

            skill = inducer.induce(trajectory=trajectory, memory=memory)

            if config.get("output", {}).get("save_skills", True):
                save_json(
                    skill,
                    experiment_dir / "skills" / variant_name / f"{task['task_id']}.json",
                )

            # Step 4: Evaluate
            validation_tasks = _get_validation_tasks(task)
            eval_result = evaluator.evaluate_skill(
                skill=skill,
                validation_tasks=validation_tasks,
                source_trajectory=trajectory,
            )
            all_results[variant].append(eval_result)

    # ===== Comparison analysis =====
    comparison = evaluator.compare_variants(all_results)
    save_json(comparison, experiment_dir / "comparison.json")

    # Save full results
    for variant, results in all_results.items():
        save_jsonl(results, experiment_dir / "results" / f"{variant.value}.jsonl")

    logger.info(f"\nExperiment complete! Results saved to: {experiment_dir}")
    logger.info(f"LLM call statistics: {llm_client.stats}")


def _get_validation_tasks(task: dict) -> list[dict[str, str]]:
    """
    Get validation tasks (MVP phase: reuse the original task).

    TODO: Generate variant tasks to validate skill generalisation.
    """
    return [
        {
            "task_id": f"{task['task_id']}_val",
            "description": task["description"],
            "expected": task.get("expected", ""),
        }
    ]


if __name__ == "__main__":
    main()
