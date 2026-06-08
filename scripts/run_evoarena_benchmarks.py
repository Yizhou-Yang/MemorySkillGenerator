#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillForge — Skill Evolution Benchmark (EvoArena-style)."""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.utils.config import load_env, load_config
from src.utils.llm import LLMClient
from src.models import Skill, TaskType, TransformVariant
from src.skill_induction.skill_library import SkillLibrary
from src.evomem.patch_recorder import PatchRecorder
from src.evomem.patch_retriever import PatchRetriever
from benchmarks.loader import BenchmarkLoader

# Metrics

def normalize(s: str) -> str:
    s = s.lower().strip()
    for a in ["a ", "an ", "the "]:
        if s.startswith(a):
            s = s[len(a):]
    return s.translate(str.maketrans("", "", string.punctuation)).strip()

def em_score(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

def f1_score(pred: str, gold: str) -> float:
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt:
        return 1.0 if pt == gt else 0.0
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pt), common / len(gt)
    return 2 * p * r / (p + r)

# B0: Zero-shot Baseline

def run_b0(llm: LLMClient, tasks: list[dict], bench: str) -> dict:
    logger.info(f"[B0] {bench}: {len(tasks)} tasks")
    results = []
    for i, t in enumerate(tasks):
        try:
            resp = llm.chat([
                {"role": "system", "content": "Answer concisely and directly. Give only the final answer."},
                {"role": "user", "content": t["description"][:4000]},
            ], temperature=0.0, max_tokens=200)
            results.append({"em": em_score(resp, t["expected"]), "f1": f1_score(resp, t["expected"])})
        except Exception as e:
            results.append({"em": 0.0, "f1": 0.0})
        if (i+1) % 20 == 0:
            logger.info(f"[B0] {i+1}/{len(tasks)} F1={sum(r['f1'] for r in results)/len(results):.3f}")

    return {
        "method": "B0", "benchmark": bench,
        "em": sum(r["em"] for r in results) / len(results),
        "f1": sum(r["f1"] for r in results) / len(results),
        "n": len(results),
    }

# Skill Evolution Engine

class SkillEvolutionEngine:
    """Drives skill evolution through a task stream."""

    def __init__(
        self,
        llm: LLMClient,
        enable_evolution: bool = True,
        enable_patches: bool = True,
    ):
        self.llm = llm
        self.library = SkillLibrary()
        self.enable_evolution = enable_evolution
        self.enable_patches = enable_patches and enable_evolution
        self.recorder = PatchRecorder() if self.enable_patches else None
        self.retriever = PatchRetriever(self.recorder, top_k=3) if self.recorder else None

        # Evolution stats
        self.stats = {
            "skills_created": 0,
            "skills_refined": 0,
            "skills_merged": 0,
            "skills_retired": 0,
            "patches_recorded": 0,
        }

    def process_task_stream(
        self,
        tasks: list[dict],
        benchmark: str,
    ) -> list[dict]:
        """Process a stream of tasks with skill evolution."""
        results = []

        for i, task in enumerate(tasks):
            # --- Step 1: Retrieve relevant skills + patch context ---
            skill_context = self._retrieve_context(task["description"])

            # --- Step 2: Generate answer with skill augmentation ---
            response = self._answer_with_skills(task, skill_context)

            # --- Step 3: Evaluate ---
            em = em_score(response, task["expected"])
            f1 = f1_score(response, task["expected"])
            results.append({"task_id": task["task_id"], "em": em, "f1": f1})

            # --- Step 4: Evolve skills based on outcome ---
            if self.enable_evolution:
                self._evolve(task, response, em, f1, step=i, benchmark=benchmark)

            # Progress logging
            if (i + 1) % 10 == 0:
                avg_f1 = sum(r["f1"] for r in results) / len(results)
                logger.info(
                    f"[Stream] {i+1}/{len(tasks)} | F1={avg_f1:.3f} | "
                    f"lib={self.library.size} | patches={self.recorder.size if self.recorder else 0} | "
                    f"evolved: +{self.stats['skills_created']}c "
                    f"+{self.stats['skills_refined']}r "
                    f"+{self.stats['skills_merged']}m "
                    f"-{self.stats['skills_retired']}d"
                )

        return results

    def _retrieve_context(self, description: str) -> str:
        """Retrieve skill context + patch history."""
        query = description[:500]
        matches = self.library.search(query, top_k=3)

        if not matches:
            return ""

        lines = []
        for skill, score in matches:
            lines.append(f"[Skill: {skill.name} (v{skill.version}, rel={score:.2f})]")
            lines.append(f"  {skill.description[:200]}")
            if skill.procedure:
                lines.append(f"  Steps: {'; '.join(skill.procedure[:3])}")

        skill_ctx = "\n".join(lines)

        # Augment with patch history (version-aware context)
        if self.retriever:
            skill_ctx = self.retriever.augmented_retrieval(query, skill_ctx)

        return skill_ctx

    def _answer_with_skills(self, task: dict, skill_context: str) -> str:
        """Generate answer using skill-augmented context."""
        system = "Answer concisely and directly. Give only the final answer."
        if skill_context:
            system += f"\n\nYour knowledge from past experience:\n{skill_context[:2000]}"

        try:
            return self.llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": task["description"][:4000]},
            ], temperature=0.0, max_tokens=200)
        except Exception:
            return ""

    def _evolve(
        self, task: dict, response: str, em: float, f1: float,
        step: int, benchmark: str,
    ):
        """Evolve skills based on task outcome."""
        task_desc = task["description"][:500]

        if f1 >= 0.5:
            # Success → create or strengthen skill
            self._create_or_strengthen(task, response, step, benchmark)

        elif f1 > 0.1:
            # Partial success → refine existing skill
            self._refine_skill(task, response, f1, step, benchmark)

        else:
            # Failure → check for misleading skills, potentially retire
            self._handle_failure(task, step, benchmark)

    def _create_or_strengthen(self, task: dict, response: str, step: int, benchmark: str):
        """On success: create new skill or merge with existing."""
        desc_short = task["description"][:300]
        existing, sim = self.library.recruit_or_create(desc_short)

        if existing and sim >= 0.4:
            # Merge: existing skill covers similar ground → evolve it
            old_desc = existing.description
            # Extract insight from this success
            insight = self._extract_insight(task, response)
            if insight and insight not in existing.description:
                existing.description = f"{existing.description} | {insight}"[:600]
                existing.version += 1
                existing.source_tasks.append(task["task_id"])
                self.stats["skills_merged"] += 1

                if self.recorder:
                    self.recorder.record_merge(
                        skill_id=existing.skill_id,
                        skill_name=existing.name,
                        content_before=old_desc[:300],
                        content_after=existing.description[:300],
                        merged_with=f"insight from {task['task_id']}",
                        rationale=f"Successful on similar task (sim={sim:.2f}), adding insight",
                        task_id=task["task_id"],
                        benchmark=benchmark,
                        step=step,
                    )
                    self.stats["patches_recorded"] += 1
        else:
            # New skill
            skill = Skill(
                name=self._generate_skill_name(task),
                description=self._extract_insight(task, response) or desc_short[:200],
                procedure=[f"Apply pattern from: {task['task_id'][:30]}"],
                source_tasks=[task["task_id"]],
                source_variant=TransformVariant.HYBRID_TO_SKILL,
                version=1,
                success_rate=1.0,
            )
            self.library.add(skill)
            self.stats["skills_created"] += 1

    def _refine_skill(self, task: dict, response: str, f1: float, step: int, benchmark: str):
        """On partial success: refine the most relevant skill."""
        matches = self.library.search(task["description"][:300], top_k=1)
        if not matches:
            return

        skill, sim = matches[0]
        if sim < 0.2:
            return

        # Refine: add constraint or clarification
        old_desc = skill.description
        refinement = f"Note: on similar tasks, partial match (F1={f1:.2f}) suggests need for more precision."

        if refinement not in skill.description:
            skill.description = f"{skill.description} | {refinement}"[:600]
            skill.version += 1
            self.stats["skills_refined"] += 1

            if self.recorder:
                self.recorder.record_update(
                    skill_id=skill.skill_id,
                    skill_name=skill.name,
                    content_before=old_desc[:300],
                    content_after=skill.description[:300],
                    rationale=f"Partial success (F1={f1:.2f}) — adding precision constraint",
                    task_id=task["task_id"],
                    benchmark=benchmark,
                    step=step,
                )
                self.stats["patches_recorded"] += 1

    def _handle_failure(self, task: dict, step: int, benchmark: str):
        """On failure: check if a skill is consistently misleading."""
        matches = self.library.search(task["description"][:300], top_k=1)
        if not matches:
            return

        skill, sim = matches[0]
        if sim < 0.3:
            return

        # Track failure
        skill.success_rate = max(0, skill.success_rate - 0.2)

        # Retire if success rate drops too low
        if skill.success_rate <= 0.1 and skill.version > 1:
            old_content = f"{skill.name}: {skill.description[:200]}"
            self.library.remove(skill.skill_id)
            self.stats["skills_retired"] += 1

            if self.recorder:
                self.recorder.record_delete(
                    skill_id=skill.skill_id,
                    skill_name=skill.name,
                    content_before=old_content,
                    rationale=f"Retired after repeated failures (success_rate={skill.success_rate:.2f})",
                    task_id=task["task_id"],
                    benchmark=benchmark,
                    step=step,
                )
                self.stats["patches_recorded"] += 1

    def _extract_insight(self, task: dict, response: str) -> str:
        """Extract a reusable insight from a successful task."""
        try:
            resp = self.llm.chat([
                {"role": "system", "content": (
                    "Extract a ONE-SENTENCE reusable insight/pattern from this task+answer pair. "
                    "Focus on the METHOD, not the specific answer. Be concise."
                )},
                {"role": "user", "content": (
                    f"Task: {task['description'][:500]}\n"
                    f"Successful answer: {response[:200]}\n"
                    f"Reusable insight:"
                )},
            ], temperature=0.0, max_tokens=100)
            return resp.strip()[:200]
        except Exception:
            return ""

    def _generate_skill_name(self, task: dict) -> str:
        """Generate a short skill name from task description."""
        try:
            resp = self.llm.chat([
                {"role": "system", "content": "Generate a 3-5 word skill name for this task pattern. Just the name, nothing else."},
                {"role": "user", "content": task["description"][:300]},
            ], temperature=0.0, max_tokens=20)
            return resp.strip()[:50]
        except Exception:
            return f"Skill_{task['task_id'][:8]}"

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaia-samples", type=int, default=50)
    parser.add_argument("--locomo-samples", type=int, default=5)
    args = parser.parse_args()

    load_env()
    config = load_config("evoarena")
    llm = LLMClient(config.get("llm", {}))

    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("SkillForge — Skill EVOLUTION Benchmark")
    logger.info(f"GAIA: {args.gaia_samples} | LoCoMo: {args.locomo_samples}")
    logger.info("Methods: B0 (no skill) | SkillStatic (no evolve) | SkillEvo (full evolve+patch)")
    logger.info("=" * 60)

    all_results = []
    benchmarks_cfg = [("gaia", args.gaia_samples), ("locomo", args.locomo_samples)]

    for bench_name, n_samples in benchmarks_cfg:
        logger.info(f"\n{'='*50}\n  {bench_name.upper()} ({n_samples} samples)\n{'='*50}")

        loader = BenchmarkLoader({"name": bench_name, "num_samples": n_samples})
        try:
            tasks = loader.load()
        except Exception as e:
            logger.error(f"Load failed: {e}")
            continue

        if not tasks:
            continue

        random.seed(42)
        random.shuffle(tasks)
        logger.info(f"Total tasks: {len(tasks)}")

        # --- B0: Zero-shot ---
        r = run_b0(llm, tasks, bench_name)
        all_results.append(r)
        logger.info(f">>> [B0] EM={r['em']:.3f} F1={r['f1']:.3f}")

        # --- SkillStatic: Skills created but NEVER evolved ---
        logger.info(f"\n[SkillStatic] Running task stream (evolution DISABLED)...")
        engine_static = SkillEvolutionEngine(llm, enable_evolution=False, enable_patches=False)
        static_results = engine_static.process_task_stream(tasks, bench_name)
        avg_em = sum(r["em"] for r in static_results) / len(static_results)
        avg_f1 = sum(r["f1"] for r in static_results) / len(static_results)
        all_results.append({
            "method": "SkillStatic", "benchmark": bench_name,
            "em": avg_em, "f1": avg_f1, "n": len(static_results),
            "library_size": engine_static.library.size,
            "evolution_stats": engine_static.stats,
        })
        logger.info(f">>> [SkillStatic] EM={avg_em:.3f} F1={avg_f1:.3f} lib={engine_static.library.size}")

        # --- SkillEvo: Full evolution with patches ---
        logger.info(f"\n[SkillEvo] Running task stream (evolution ENABLED + patches)...")
        engine_evo = SkillEvolutionEngine(llm, enable_evolution=True, enable_patches=True)
        evo_results = engine_evo.process_task_stream(tasks, bench_name)
        avg_em = sum(r["em"] for r in evo_results) / len(evo_results)
        avg_f1 = sum(r["f1"] for r in evo_results) / len(evo_results)
        all_results.append({
            "method": "SkillEvo", "benchmark": bench_name,
            "em": avg_em, "f1": avg_f1, "n": len(evo_results),
            "library_size": engine_evo.library.size,
            "patches": engine_evo.recorder.size if engine_evo.recorder else 0,
            "evolution_stats": engine_evo.stats,
        })
        logger.info(
            f">>> [SkillEvo] EM={avg_em:.3f} F1={avg_f1:.3f} "
            f"lib={engine_evo.library.size} patches={engine_evo.recorder.size if engine_evo.recorder else 0}"
        )
        logger.info(f"    Evolution: {engine_evo.stats}")

    # --- Final Summary ---
    logger.info(f"\n{'='*60}\n  FINAL RESULTS\n{'='*60}")
    for r in all_results:
        evo = r.get("evolution_stats", {})
        evo_str = f" | +{evo.get('skills_created',0)}c +{evo.get('skills_refined',0)}r +{evo.get('skills_merged',0)}m -{evo.get('skills_retired',0)}d" if evo else ""
        logger.info(
            f"  {r['method']:12s} | {r['benchmark']:8s} | "
            f"EM={r['em']:.3f} F1={r['f1']:.3f} | n={r['n']}{evo_str}"
        )

    # Deltas
    for bench in [b[0] for b in benchmarks_cfg]:
        br = [r for r in all_results if r["benchmark"] == bench]
        b0 = next((r for r in br if r["method"] == "B0"), None)
        static = next((r for r in br if r["method"] == "SkillStatic"), None)
        evo = next((r for r in br if r["method"] == "SkillEvo"), None)
        if b0 and static and evo:
            logger.info(f"\n  {bench} gains vs B0:")
            logger.info(f"    SkillStatic: F1 +{(static['f1']-b0['f1'])*100:.1f}pp")
            logger.info(f"    SkillEvo:    F1 +{(evo['f1']-b0['f1'])*100:.1f}pp")
            logger.info(f"    Evolution gain (Evo vs Static): F1 +{(evo['f1']-static['f1'])*100:.1f}pp")

    # Save
    out_path = output_dir / "evoarena_benchmark_results.json"
    out_path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
        "llm_stats": llm.stats,
    }, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")
    logger.info(f"API: {llm.stats}")

if __name__ == "__main__":
    main()
