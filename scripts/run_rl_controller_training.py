#!/usr/bin/env python3
"""RL Controller Training Pipeline — enables the full MemSkill training loop."""
from __future__ import annotations

import json
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.memory.compressor import create_compressor
from src.skill_induction.factory import create_inducer
from src.trajectory.collector import TrajectoryCollector
from src.models import Skill, TransformVariant
from src.rl_controller.controller import (
    ControllerState,
    PPOTransition,
    SkillSelectionController,
)
from src.skill_induction.skill_designer import SkillDesigner

# Embedding Utilities

class TextEmbedder:
    """Lightweight text embedder using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dim: int = 384):
        self.dim = dim
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"[Embedder] Loaded '{model_name}' (dim={self.dim})")
        except Exception as exc:
            logger.warning(f"[Embedder] Failed to load model: {exc}. Using random embeddings.")

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to a normalized embedding vector."""
        if self._model is not None:
            vec = self._model.encode(text, convert_to_numpy=True).astype(np.float32)
        else:
            # Deterministic random embedding based on text hash
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts."""
        if self._model is not None:
            vecs = self._model.encode(texts, convert_to_numpy=True, batch_size=32).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms
        else:
            return np.stack([self.encode(t) for t in texts])

# Metrics

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

# Skill Formatting

def format_skill_prompt(skill: Skill) -> str:
    """Format a skill for inclusion in the LLM prompt."""
    parts = [f"## {skill.name}", skill.description, ""]
    if skill.procedure:
        parts.append("Steps:")
        for i, step in enumerate(skill.procedure, 1):
            parts.append(f"  {i}. {step}")
    if skill.constraints:
        parts.append("Constraints: " + "; ".join(skill.constraints))
    return "\n".join(parts)

def skill_to_text(skill: Skill) -> str:
    """Convert skill to a single text string for embedding."""
    parts = [skill.name, skill.description]
    parts.extend(skill.procedure)
    parts.extend(skill.constraints)
    return " ".join(parts)

# Core Training Loop

def execute_with_skills(
    llm_client: LLMClient,
    question: str,
    selected_skills: list[Skill],
) -> str:
    """Execute a question using the controller-selected skills."""
    if not selected_skills:
        # Fallback: direct answer
        messages = [
            {"role": "system", "content": "Answer the question directly and concisely."},
            {"role": "user", "content": question},
        ]
    else:
        skill_text = "\n\n---\n\n".join(format_skill_prompt(s) for s in selected_skills)
        messages = [
            {"role": "system", "content": (
                "You have access to learned skills. Use the most relevant one(s) "
                "to answer the question.\n\n"
                f"=== SELECTED SKILLS ===\n{skill_text}\n=== END ===\n\n"
                "Answer directly and concisely. Give only the answer."
            )},
            {"role": "user", "content": question},
        ]
    return llm_client.chat(messages, temperature=0.3, max_tokens=256)

def run_training_epoch(
    llm_client: LLMClient,
    controller: SkillSelectionController,
    embedder: TextEmbedder,
    skills: list[Skill],
    tasks: list[dict],
    epoch: int,
    training: bool = True,
    designer: SkillDesigner | None = None,
) -> dict:
    """Run one training epoch through all tasks."""
    em_scores = []
    f1_scores = []
    rewards = []

    for idx, task in enumerate(tasks):
        desc = task["description"]
        expected = task.get("expected", "")

        # Step 1: Create state from query embedding
        query_emb = embedder.encode(desc)
        state = ControllerState(
            embedding=query_emb,
            span_text=desc[:100],
        )

        # Step 2: Controller selects skills
        result = controller.select_skills(state, training=training)

        # Map selected indices back to Skill objects
        selected_skills = []
        for skill_idx in result.selected_indices:
            if skill_idx < len(skills):
                selected_skills.append(skills[skill_idx])

        # Step 3: Execute with selected skills
        try:
            response = execute_with_skills(llm_client, desc, selected_skills)
        except Exception as exc:
            logger.error(f"  Execution failed: {exc}")
            response = ""

        # Step 4: Compute reward
        em = compute_em(response, expected)
        f1 = compute_token_f1(response, expected)
        reward = (em + f1) / 2.0  # Combined reward signal

        em_scores.append(em)
        f1_scores.append(f1)
        rewards.append(reward)

        # Step 5: Record transition (only during training)
        if training:
            value = controller.get_value(state)
            transition = PPOTransition(
                state_embedding=state.embedding,
                selected_indices=result.selected_indices,
                log_prob=result.log_prob,
                reward=reward,
                value=value,
            )
            controller.record_transition(transition)

            # Record hard cases for Skill Designer
            if designer and reward < 0.5:
                designer.record_failure(
                    query=desc[:200],
                    prediction=response[:100],
                    ground_truth=expected[:100],
                    reward=reward,
                    step=controller._current_step,
                )

        if (idx + 1) % 5 == 0 or idx == len(tasks) - 1:
            logger.info(
                f"  Epoch {epoch} | Task {idx+1}/{len(tasks)} | "
                f"EM={avg(em_scores):.1%} F1={avg(f1_scores):.3f} "
                f"reward={avg(rewards):.3f} "
                f"selected={[s.name[:20] for s in selected_skills[:2]]}"
            )

    # PPO update after epoch (if training)
    ppo_stats = {}
    if training and rewards:
        final_reward = avg(rewards)
        controller.compute_advantages(final_reward)
        ppo_stats = controller.ppo_update(epochs=4)
        controller.save_snapshot(final_reward)
        logger.info(
            f"  [PPO] policy_loss={ppo_stats.get('policy_loss', 0):.4f}, "
            f"value_loss={ppo_stats.get('value_loss', 0):.4f}, "
            f"entropy={ppo_stats.get('entropy', 0):.4f}"
        )

    return {
        "epoch": epoch,
        "training": training,
        "avg_em": avg(em_scores),
        "avg_f1": avg(f1_scores),
        "avg_reward": avg(rewards),
        "ppo_stats": ppo_stats,
        "num_tasks": len(tasks),
    }

# Main

def main():
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
    log_path = Path("experiments/rl_controller_training.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}", level="DEBUG")

    logger.info("=" * 70)
    logger.info("RL Controller Training Pipeline")
    logger.info("=" * 70)
    logger.info("Implements full MemSkill §3.2-3.7 training loop:")
    logger.info("  Embed → Controller Select → Execute → Reward → PPO Update")
    logger.info("")

    load_env()
    start_time = time.time()
    llm_client = LLMClient({"temperature": 0.7, "max_tokens": 2048, "timeout": 120})

    # API test
    resp = llm_client.chat([{"role": "user", "content": "Say OK"}], temperature=0.0, max_tokens=5)
    logger.info(f"API: '{resp.strip()}' ✅")

    # ---- Initialize Embedder ----
    logger.info("\n" + "─" * 50)
    logger.info("Initializing Text Embedder...")
    embedder = TextEmbedder()
    embedding_dim = embedder.dim
    logger.info(f"Embedding dimension: {embedding_dim}")

    # ---- Load Benchmark ----
    logger.info("\n" + "─" * 50)
    logger.info("Loading HotpotQA benchmark...")
    from benchmarks.loader import BenchmarkLoader

    NUM_TRAIN_INDUCTION = 8   # Tasks for skill induction
    NUM_TRAIN_RL = 10         # Tasks for RL training
    NUM_TEST = 7              # Held-out evaluation
    TOTAL_NEEDED = NUM_TRAIN_INDUCTION + NUM_TRAIN_RL + NUM_TEST

    loader = BenchmarkLoader({"name": "hotpotqa", "num_samples": TOTAL_NEEDED})
    all_tasks = loader.load()

    induction_tasks = all_tasks[:NUM_TRAIN_INDUCTION]
    rl_train_tasks = all_tasks[NUM_TRAIN_INDUCTION:NUM_TRAIN_INDUCTION + NUM_TRAIN_RL]
    test_tasks = all_tasks[NUM_TRAIN_INDUCTION + NUM_TRAIN_RL:]

    logger.info(f"Tasks: {len(induction_tasks)} induction, {len(rl_train_tasks)} RL-train, {len(test_tasks)} test")

    # ---- Phase 1: Skill Induction ----
    logger.info("\n" + "─" * 50)
    logger.info("Phase 1: Skill Induction")
    logger.info("─" * 50)

    collector = TrajectoryCollector(llm_client, {"max_steps": 4})
    compressor = create_compressor("mem0", llm_client, {})

    skills: list[Skill] = []
    for idx, task in enumerate(induction_tasks):
        logger.info(f"  [{idx+1}/{len(induction_tasks)}] {task['task_id']}")
        try:
            traj = collector.collect(task_id=task["task_id"], task_description=task["description"])
            memory = compressor.compress(traj)
            inducer = create_inducer(TransformVariant.HYBRID_TO_SKILL, llm_client, {})
            skill = inducer.induce(trajectory=traj, memory=memory)
            skills.append(skill)
            logger.info(f"    → '{skill.name}' ({len(skill.procedure)} steps)")
        except Exception as exc:
            logger.error(f"    Failed: {exc}")

    logger.info(f"\nInduced {len(skills)} skills")

    if not skills:
        logger.error("No skills induced, cannot proceed with RL training")
        sys.exit(1)

    # ---- Phase 2: Initialize RL Controller ----
    logger.info("\n" + "─" * 50)
    logger.info("Phase 2: Initialize RL Controller")
    logger.info("─" * 50)

    controller = SkillSelectionController({
        "embedding_dim": embedding_dim,
        "hidden_dim": min(256, embedding_dim),
        "top_k": min(3, len(skills)),
        "tau_0": 0.3,
        "t_explore": 50,
        "clip_epsilon": 0.2,
        "gamma": 0.99,
        "entropy_coeff": 0.01,
        "value_coeff": 0.5,
        "learning_rate": 0.0003,
    })

    # Register all skills with their embeddings
    logger.info("Embedding and registering skills...")
    for skill in skills:
        skill_text = skill_to_text(skill)
        skill_emb = embedder.encode(skill_text)
        controller.register_skill(
            skill_id=skill.skill_id,
            skill_name=skill.name,
            embedding=skill_emb,
            is_new=True,
        )

    logger.info(f"Controller initialized: bank_size={controller.skill_bank_size}, top_k={controller.top_k}")

    # Initialize Skill Designer for hard-case evolution
    designer = SkillDesigner(
        llm_client=llm_client,
        config={"trigger_interval": 50, "max_edits_per_cycle": 2, "patience": 3},
    )

    # ---- Phase 3: Baseline (no training, random controller) ----
    logger.info("\n" + "─" * 50)
    logger.info("Phase 3: Baseline Evaluation (untrained controller)")
    logger.info("─" * 50)

    baseline_result = run_training_epoch(
        llm_client, controller, embedder, skills, test_tasks,
        epoch=0, training=False,
    )
    logger.info(f"  Baseline: EM={baseline_result['avg_em']:.1%}, F1={baseline_result['avg_f1']:.3f}")

    # ---- Phase 4: RL Training Loop ----
    NUM_EPOCHS = 3
    logger.info("\n" + "─" * 50)
    logger.info(f"Phase 4: RL Training ({NUM_EPOCHS} epochs on {len(rl_train_tasks)} tasks)")
    logger.info("─" * 50)

    training_history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"\n  ═══ Epoch {epoch}/{NUM_EPOCHS} ═══")
        epoch_result = run_training_epoch(
            llm_client, controller, embedder, skills, rl_train_tasks,
            epoch=epoch, training=True, designer=designer,
        )
        training_history.append(epoch_result)
        logger.info(
            f"  Epoch {epoch} done: EM={epoch_result['avg_em']:.1%}, "
            f"F1={epoch_result['avg_f1']:.3f}, reward={epoch_result['avg_reward']:.3f}"
        )

    # ---- Phase 5: Post-training Evaluation ----
    logger.info("\n" + "─" * 50)
    logger.info("Phase 5: Post-training Evaluation (trained controller)")
    logger.info("─" * 50)

    trained_result = run_training_epoch(
        llm_client, controller, embedder, skills, test_tasks,
        epoch=NUM_EPOCHS + 1, training=False,
    )
    logger.info(f"  Trained: EM={trained_result['avg_em']:.1%}, F1={trained_result['avg_f1']:.3f}")

    # ---- Phase 6: Ablation — Random Selection (w/o controller) ----
    logger.info("\n" + "─" * 50)
    logger.info("Phase 6: Ablation — Random Selection (w/o controller)")
    logger.info("─" * 50)

    random_em_scores = []
    random_f1_scores = []
    for task in test_tasks:
        desc = task["description"]
        expected = task.get("expected", "")

        # Random skill selection
        random_indices = np.random.choice(len(skills), size=min(3, len(skills)), replace=False)
        random_skills = [skills[i] for i in random_indices]

        try:
            response = execute_with_skills(llm_client, desc, random_skills)
            random_em_scores.append(compute_em(response, expected))
            random_f1_scores.append(compute_token_f1(response, expected))
        except Exception:
            random_em_scores.append(0.0)
            random_f1_scores.append(0.0)

    random_em = avg(random_em_scores)
    random_f1 = avg(random_f1_scores)
    logger.info(f"  Random: EM={random_em:.1%}, F1={random_f1:.3f}")

    # ---- Phase 7: Ablation — No Skills (direct LLM) ----
    logger.info("\n" + "─" * 50)
    logger.info("Phase 7: Ablation — No Skills (direct LLM)")
    logger.info("─" * 50)

    noskill_em_scores = []
    noskill_f1_scores = []
    for task in test_tasks:
        desc = task["description"]
        expected = task.get("expected", "")
        try:
            response = execute_with_skills(llm_client, desc, [])
            noskill_em_scores.append(compute_em(response, expected))
            noskill_f1_scores.append(compute_token_f1(response, expected))
        except Exception:
            noskill_em_scores.append(0.0)
            noskill_f1_scores.append(0.0)

    noskill_em = avg(noskill_em_scores)
    noskill_f1 = avg(noskill_f1_scores)
    logger.info(f"  No-skill: EM={noskill_em:.1%}, F1={noskill_f1:.3f}")

    # ---- Final Results ----
    elapsed = time.time() - start_time
    stats = llm_client.stats

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS: RL Controller Training")
    logger.info("=" * 70)

    logger.info(f"\n{'Method':<30} {'EM':<10} {'F1':<10} {'Δ EM vs baseline':<15}")
    logger.info("-" * 65)
    logger.info(f"{'No skills (direct LLM)':<30} {noskill_em:<10.1%} {noskill_f1:<10.3f} {'—':<15}")
    logger.info(f"{'Random selection (w/o ctrl)':<30} {random_em:<10.1%} {random_f1:<10.3f} {random_em - baseline_result['avg_em']:+.1%}")
    logger.info(f"{'Untrained controller':<30} {baseline_result['avg_em']:<10.1%} {baseline_result['avg_f1']:<10.3f} {'(baseline)':<15}")
    logger.info(f"{'Trained controller (PPO)':<30} {trained_result['avg_em']:<10.1%} {trained_result['avg_f1']:<10.3f} {trained_result['avg_em'] - baseline_result['avg_em']:+.1%}")

    # Paper comparison
    logger.info(f"\n  Paper reference (MemSkill Table 2, LoCoMo L-J):")
    logger.info(f"    Full MemSkill: 50.96")
    logger.info(f"    w/o controller (random): 45.86 (drop 5.10)")
    logger.info(f"    → Controller contributes ~5 points after training")

    controller_delta = trained_result['avg_em'] - random_em
    logger.info(f"\n  Our result:")
    logger.info(f"    Controller advantage: {controller_delta:+.1%} (trained vs random)")
    logger.info(f"    Training improvement: {trained_result['avg_em'] - baseline_result['avg_em']:+.1%} (trained vs untrained)")

    # Training curve
    logger.info(f"\n  Training curve:")
    for h in training_history:
        logger.info(f"    Epoch {h['epoch']}: EM={h['avg_em']:.1%}, F1={h['avg_f1']:.3f}, reward={h['avg_reward']:.3f}")

    logger.info(f"\n💰 Token Usage:")
    logger.info(f"  API calls: {stats['total_calls']}")
    logger.info(f"  Total tokens: {stats['total_tokens']:,}")
    logger.info(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save results
    output_path = Path("experiments/rl_controller_training_results.json")
    output = {
        "baseline_untrained": baseline_result,
        "trained": trained_result,
        "random_ablation": {"avg_em": random_em, "avg_f1": random_f1},
        "noskill_ablation": {"avg_em": noskill_em, "avg_f1": noskill_f1},
        "training_history": training_history,
        "controller_config": controller.config,
        "num_skills": len(skills),
        "embedding_dim": embedding_dim,
        "elapsed_seconds": elapsed,
        "total_tokens": stats["total_tokens"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\n  Results saved to: {output_path}")

if __name__ == "__main__":
    main()
