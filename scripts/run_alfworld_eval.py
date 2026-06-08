#!/usr/bin/env python3
"""
ALFWorld evaluation runner — Path B (real TextWorld engine + ReAct loop).

Pipeline:
    1. Build / load skill bank (induce_alfworld_skills.py output).
    2. For each method ∈ {B0, B2, A3, A3+Plan-C}:
         - For each task in stratified test sample:
              run ReAct loop (max_steps), collect SR/steps/tokens.
    3. Apply Plan C: per-bench τ_b = quantile_q*(train_s_max), q* via 5-fold CV.
    4. Aggregate metrics, write JSON.

Standardized with the v5 paper:
    - Methods: B0 (no skills), B2 (SkillOS-style retrieve top-3 raw),
               A3 (full curator: merged + reformatted skills),
               A3+Plan-C (A3 + SRDP void-case gating, τ from train s_max quantile).
    - Metric: Success Rate (SR, binary 0/1) — replaces EM for procedural tasks.
    - Split: valid_unseen (134 games), stratified sample of 50 across 6 task types.

Usage:
    source .venv_alfworld/bin/activate
    python scripts/run_alfworld_eval.py \
        --skill-bank experiments/alfworld_skills/a3_bank.json \
        --methods B0 B2 A3 A3+PlanC \
        --n-test 50 --max-steps 50 --seed 42 \
        --output experiments/alfworld_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# Make project importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from loguru import logger

from src.curation.void_case import calibrate_tau_quantile, cv_select_quantile
from src.models import Skill
from src.utils.alfworld_env import AlfworldEnv, task_type_from_gamefile
from src.utils.config import load_env  # for .env loading
from src.utils.llm import LLMClient
from src.utils.skill_formatter import FormattingConfig, format_skill_library

# Skill bank IO

def load_skill_bank(path: Path) -> tuple[list[Skill], np.ndarray]:
    """Load a skill bank JSON file into (skills, embeddings).

    JSON schema (produced by induce_alfworld_skills.py):
        {
          "skills": [<Skill.dict()>, ...],
          "embeddings": [[float, ...], ...],   # parallel to skills
          "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
          "method": "A3" | "B2" | ...,
          "induced_from": "valid_seen[:N]"
        }
    """
    if not path.exists():
        raise FileNotFoundError(f"Skill bank not found: {path}")
    blob = json.loads(path.read_text())
    skills = [Skill(**s) for s in blob["skills"]]
    embs = np.array(blob["embeddings"], dtype=np.float32)
    if embs.size == 0:
        return skills, embs
    # L2-normalize for cosine = dot
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return skills, embs

def embed_query(text: str, encoder) -> np.ndarray:
    """Embed a single query string and L2-normalize."""
    v = encoder.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)

# Skill retrieval (cosine top-k + s_max)

def retrieve_skills(
    query_emb: np.ndarray,
    skills: list[Skill],
    skill_embs: np.ndarray,
    top_k: int,
) -> tuple[list[Skill], float]:
    """Cosine top-k retrieval. Returns (selected_skills, s_max)."""
    if not skills or skill_embs.size == 0:
        return [], 0.0
    sims = (skill_embs @ query_emb).flatten()
    s_max = float(sims.max()) if sims.size else 0.0
    idx = np.argsort(sims)[::-1][:top_k]
    return [skills[i] for i in idx], s_max

# ReAct loop

REACT_SYSTEM_BASE = """You are an embodied agent solving a household task in a text-based environment.
You see a textual room description, a task goal, and a list of admissible commands.

At each step you MUST output exactly two lines:
    Thought: <one short sentence reasoning>
    Action: <one command from the admissible list, copied verbatim>

The Action MUST appear in the admissible list verbatim (case-sensitive).
Avoid repeating an action that just produced "Nothing happens." — try a different action.

Here are two example trajectories that illustrate the expected behaviour.

=== Example 1 (pick & place) ===
Task: put a clean plate on the diningtable.
Thought: I should locate a plate first; it is most likely in a cabinet or on the countertop.
Action: go to countertop 1
Thought: I see a plate 1 here; pick it up.
Action: take plate 1 from countertop 1
Thought: The plate may be dirty, so I will clean it at the sink.
Action: go to sinkbasin 1
Thought: Clean the plate using the sink.
Action: clean plate 1 with sinkbasin 1
Thought: Now bring it to the diningtable.
Action: go to diningtable 1
Action: put plate 1 in/on diningtable 1

=== Example 2 (heat & place) ===
Task: heat some bread and put it on the diningtable.
Thought: First find the bread — likely on a countertop or in a cabinet.
Action: go to countertop 1
Action: take bread 1 from countertop 1
Thought: Heat it in the microwave.
Action: go to microwave 1
Action: heat bread 1 with microwave 1
Thought: Deliver to the diningtable.
Action: go to diningtable 1
Action: put bread 1 in/on diningtable 1

Key patterns to remember:
  - To find an object, GO TO each likely receptacle until you see it.
  - Use "clean X with sinkbasin Y", "heat X with microwave Y", "cool X with fridge Y".
  - Use "use desklamp 1" / "examine X with desklamp Y" for look_at_obj_in_light tasks.
  - End with "put <object> in/on <receptacle>" if the goal requires placement.

Now solve the new task.
"""

REACT_SYSTEM_WITH_SKILLS = REACT_SYSTEM_BASE + """
The following procedural skills, learned from similar past tasks, may help.
Use them as guidance when relevant; ignore them when not applicable:

{skill_block}
"""

_ACTION_RE = re.compile(r"Action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)

def parse_action(reply: str, admissible: list[str]) -> str:
    """Extract the Action line from the LLM reply and snap to admissible.

    Strategy:
      1. Match "Action: <verb>" via regex.
      2. If exact match in admissible → use it.
      3. Else: case-insensitive exact
      4. Else: substring (longest admissible match)
      5. Else: fallback
    """
    if not admissible:
        return "look"  # graceful no-op
    m = _ACTION_RE.search(reply)
    raw = m.group(1).strip() if m else reply.strip().splitlines()[-1].strip()
    raw = raw.strip().rstrip(".").strip("`").strip()

    # 1. exact
    for a in admissible:
        if a == raw:
            return a
    # 2. case-insensitive exact
    for a in admissible:
        if a.lower() == raw.lower():
            return a
    # 3. substring (longest admissible match)
    raw_l = raw.lower()
    candidates = [a for a in admissible if a.lower() in raw_l or raw_l in a.lower()]
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    # 4. fallback
    return admissible[0]

def run_react_episode(
    env: AlfworldEnv,
    game_idx: int,
    llm: LLMClient,
    skill_block: str,
    max_steps: int,
    history_keep: int = 8,
    max_admissible_show: int = 40,
) -> dict[str, Any]:
    """Run one ReAct episode in ALFWorld, return per-task result dict."""
    obs, info = env.reset(game_idx=game_idx)
    task = info["task"]
    initial_obs = obs.strip()  # always keep the room description visible
    # history stores observations *after* every step (including the initial one).
    history: list[str] = [obs]
    actions_taken: list[str] = []
    t0 = time.time()

    if skill_block:
        sys_prompt = REACT_SYSTEM_WITH_SKILLS.format(skill_block=skill_block)
    else:
        sys_prompt = REACT_SYSTEM_BASE

    won = False
    done = info["done"]
    last_obs = obs

    for step in range(max_steps):
        if done:
            break
        admissible = info["actions"]
        # Build a window: last (history_keep) (action, obs) pairs.
        # actions_taken[i] led to history[i+1].  We always keep the initial obs.
        n = len(actions_taken)
        start = max(0, n - history_keep)
        history_lines: list[str] = []
        if start > 0:
            history_lines.append(f"[... {start} earlier step(s) elided ...]")
        for i in range(start, n):
            history_lines.append(f"[Action {i+1}] {actions_taken[i]}")
            history_lines.append(f"[Obs {i+1}] {history[i+1].strip()}")
        history_str = "\n".join(history_lines) if history_lines else "(no actions taken yet)"

        admissible_show = admissible[:max_admissible_show]
        admissible_str = ", ".join(f"'{a}'" for a in admissible_show)
        if len(admissible) > max_admissible_show:
            admissible_str += f", ... ({len(admissible) - max_admissible_show} more)"
        user_msg = (
            f"Task: {task}\n\n"
            f"[Initial Obs] {initial_obs}\n\n"
            f"Recent history:\n{history_str}\n\n"
            f"Admissible commands: [{admissible_str}]\n\n"
            f"Now output:\nThought: ...\nAction: ..."
        )
        try:
            reply = llm.chat(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except Exception as e:
            logger.warning(f"[game={game_idx} step={step}] LLM error: {e}")
            reply = f"Action: {admissible[0]}"
        action = parse_action(reply, admissible)
        actions_taken.append(action)
        last_obs, info = env.step(action)
        history.append(last_obs)
        won = info["won"]
        done = info["done"]
        if won:
            break

    elapsed = time.time() - t0
    return {
        "game_idx": game_idx,
        "task": task,
        "won": bool(won),
        "steps": len(actions_taken),
        "elapsed_sec": round(elapsed, 1),
        "actions": actions_taken,
        "final_obs": last_obs[:200],
    }

# Method: skill block construction

def build_skill_block_b0(query: str, **kwargs) -> tuple[str, dict]:
    """B0: no skills."""
    return "", {"method": "B0", "n_skills_used": 0, "s_max": 0.0}

def _format_skills_plain(skills: list[Skill]) -> str:
    """SkillOS-style plain dump: raw skills concatenated, no sandwich/compact.

    Used by B2 so that the only difference vs A3 is bank curation (merged vs raw),
    not formatting. This isolates the curation contribution.
    """
    if not skills:
        return ""
    parts = []
    for i, s in enumerate(skills, 1):
        block = [f"Skill {i}: {s.name}", f"Description: {s.description}"]
        if s.preconditions:
            block.append("Preconditions: " + "; ".join(s.preconditions))
        if s.procedure:
            steps = "\n".join(f"  {j+1}. {st}" for j, st in enumerate(s.procedure))
            block.append("Procedure:\n" + steps)
        if s.constraints:
            block.append("Constraints: " + "; ".join(s.constraints))
        parts.append("\n".join(block))
    return "\n\n".join(parts)

def build_skill_block_b2(
    query: str,
    skills: list[Skill],
    skill_embs: np.ndarray,
    encoder,
    top_k: int = 3,
    **kwargs,
) -> tuple[str, dict]:
    """B2: SkillOS-style — retrieve top-K raw skills, plain dump, no curator."""
    if not skills:
        return "", {"method": "B2", "n_skills_used": 0, "s_max": 0.0}
    q = embed_query(query, encoder)
    selected, s_max = retrieve_skills(q, skills, skill_embs, top_k)
    block = _format_skills_plain(selected)
    return block, {"method": "B2", "n_skills_used": len(selected), "s_max": s_max}

def build_skill_block_a3(
    query: str,
    skills: list[Skill],
    skill_embs: np.ndarray,
    encoder,
    top_k: int = 3,
    **kwargs,
) -> tuple[str, dict]:
    """A3 (Merge-only variant): retrieve top-K from the LLM-merged bank, then
    apply the curator's sandwich+compact attention operators on the prompt block.

    Caveat (transparency): this is the Merge+Format subset of the full v5 A3
    pipeline (which additionally has Prune/Consistency/Rewrite). On ALFWorld
    the bank is small enough that Prune is a no-op, and Consistency/Rewrite
    require a writeback loop we deliberately skip here for cost reasons.
    """
    if not skills:
        return "", {"method": "A3", "n_skills_used": 0, "s_max": 0.0}
    q = embed_query(query, encoder)
    selected, s_max = retrieve_skills(q, skills, skill_embs, top_k)
    block = format_skill_library(selected, FormattingConfig(max_skills_in_prompt=top_k))
    return block, {"method": "A3", "n_skills_used": len(selected), "s_max": s_max}

def build_skill_block_a3_planc(
    query: str,
    skills: list[Skill],
    skill_embs: np.ndarray,
    encoder,
    top_k: int = 3,
    tau_void: float = 0.0,
    **kwargs,
) -> tuple[str, dict]:
    """A3+Plan-C: A3 with SRDP void-case gating.
    If s_max < τ_void → c_∅ (no skills, fall back to LLM zero-shot).
    τ_void must be calibrated on TRAIN-side rollouts only (no test leakage).
    """
    if not skills:
        return "", {"method": "A3+PlanC", "n_skills_used": 0, "s_max": 0.0,
                     "tau": tau_void, "void": True}
    q = embed_query(query, encoder)
    selected, s_max = retrieve_skills(q, skills, skill_embs, top_k)
    if s_max < tau_void:
        return "", {"method": "A3+PlanC", "n_skills_used": 0, "s_max": s_max,
                     "tau": tau_void, "void": True}
    block = format_skill_library(selected, FormattingConfig(max_skills_in_prompt=top_k))
    return block, {"method": "A3+PlanC", "n_skills_used": len(selected),
                    "s_max": s_max, "tau": tau_void, "void": False}

METHOD_BUILDERS = {
    "B0": build_skill_block_b0,
    "B2": build_skill_block_b2,
    "A3": build_skill_block_a3,
    "A3+PlanC": build_skill_block_a3_planc,
}

# Stratified sampling

def stratified_sample(num_games: int, gamefiles: list[str], n: int, seed: int) -> list[int]:
    """Sample n game indices stratified by task type (6 ALFWorld categories).

    Returns indices into the gamefiles list, sorted (for deterministic order).
    """
    rng = np.random.default_rng(seed)
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(gamefiles):
        by_type[task_type_from_gamefile(f)].append(i)

    n_types = len(by_type)
    per_type = max(1, n // n_types)
    selected: list[int] = []
    for t, idxs in sorted(by_type.items()):
        rng.shuffle(idxs)
        selected.extend(idxs[:per_type])
    # Top up if we need more (round-robin from leftovers)
    if len(selected) < n:
        leftovers = [i for i in range(num_games) if i not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    selected = selected[:n]
    selected.sort()
    return selected

# Plan C calibration

def calibrate_planc_tau(
    train_per_task: list[dict],
    q_grid: tuple[float, ...] = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70),
    seeds: tuple[int, ...] = (42, 123, 456, 789, 2024),
    n_folds: int = 5,
) -> dict[str, Any]:
    """Choose τ for ALFWorld via per-bench q-quantile CV.

    train_per_task entries must contain: s_max, sr_inject, sr_void
    where sr_inject = SR with skills (A3 method), sr_void = SR without (B0).

    Returns: {q_star, tau, score_mean, score_std, all_seed_results}
    """
    if not train_per_task:
        return {"q_star": 0.30, "tau": 0.0, "score_mean": 0.0, "score_std": 0.0,
                "n_train": 0}
    s = np.array([t["s_max"] for t in train_per_task], dtype=np.float64)
    ei = np.array([t["sr_inject"] for t in train_per_task], dtype=np.float64)
    ev = np.array([t["sr_void"] for t in train_per_task], dtype=np.float64)

    from collections import Counter
    q_choices = []
    scores = []
    for seed in seeds:
        q_star, score_star, _ = cv_select_quantile(
            s, ei, ev, q_grid=list(q_grid), n_folds=n_folds, seed=seed
        )
        q_choices.append(q_star)
        scores.append(score_star)
    q_mode = Counter(q_choices).most_common(1)[0]
    return {
        "q_star": float(q_mode[0]),
        "q_star_stability": q_mode[1] / len(q_choices),
        "tau": calibrate_tau_quantile(s, q=q_mode[0]),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "all_seed_results": [
            {"seed": int(seeds[i]), "q": float(q_choices[i]), "score": float(scores[i])}
            for i in range(len(seeds))
        ],
        "n_train": int(len(s)),
    }

# Main eval

def run_eval(args) -> dict[str, Any]:
    # Load .env so DEEPSEEK_API_KEY is available
    load_env()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- Build env (single instance, shared across methods) -----
    logger.info(f"[init] Building AlfworldEnv split={args.split} max_steps={args.max_steps}")
    env = AlfworldEnv(split=args.split, max_steps=args.max_steps)
    gamefiles = env.list_tasks()
    logger.info(f"[init] num_games={env.num_games}, sampling n={args.n_test}")
    test_idx = stratified_sample(env.num_games, gamefiles, args.n_test, args.seed)
    logger.info(f"[init] selected indices: first 10 = {test_idx[:10]}, total {len(test_idx)}")

    # Type distribution of the sample
    sample_types = [task_type_from_gamefile(gamefiles[i]) for i in test_idx]
    type_counts = dict((t, sample_types.count(t)) for t in set(sample_types))
    logger.info(f"[init] sample type distribution: {type_counts}")

    # ----- Load LLM and encoder -----
    logger.info(f"[init] Loading LLM client and encoder...")
    llm = LLMClient({"temperature": 0.0, "max_tokens": 200, "timeout": 60})
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ----- Load skill bank(s) -----
    skills_b2: list[Skill] = []
    skill_embs_b2: np.ndarray = np.zeros((0,), dtype=np.float32)
    skills_a3: list[Skill] = []
    skill_embs_a3: np.ndarray = np.zeros((0,), dtype=np.float32)
    if args.skill_bank_b2:
        logger.info(f"[init] Loading B2 skill bank: {args.skill_bank_b2}")
        skills_b2, skill_embs_b2 = load_skill_bank(Path(args.skill_bank_b2))
        logger.info(f"[init] B2 bank: {len(skills_b2)} skills")
    if args.skill_bank_a3:
        logger.info(f"[init] Loading A3 skill bank: {args.skill_bank_a3}")
        skills_a3, skill_embs_a3 = load_skill_bank(Path(args.skill_bank_a3))
        logger.info(f"[init] A3 bank: {len(skills_a3)} skills")

    # ----- Run each method -----
    results: dict[str, Any] = {
        "meta": {
            "split": args.split,
            "n_test": args.n_test,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "test_indices": test_idx,
            "sample_type_distribution": type_counts,
            "model": llm.model,
            "skill_bank_b2": args.skill_bank_b2,
            "skill_bank_a3": args.skill_bank_a3,
        },
        "methods": {},
    }

    method_set = list(args.methods)
    # Eval order: B0 → B2 → A3 → A3+PlanC. Plan-C τ is calibrated from
    # the TRAIN-side calibration JSON (no test leakage).
    desired_order = ["B0", "B2", "A3", "A3+PlanC"]
    method_set.sort(key=lambda m: desired_order.index(m) if m in desired_order else 99)

    # Pre-calibrate Plan-C τ from train_calib.json (induction output).
    planc_tau: float | None = None
    planc_calib: dict[str, Any] = {}
    if "A3+PlanC" in method_set:
        if args.train_calib and Path(args.train_calib).exists():
            blob = json.loads(Path(args.train_calib).read_text())
            recs = blob.get("records", [])
            train_pt = [{"s_max": r["s_max"], "sr_inject": r["sr_inject"],
                          "sr_void": r["sr_void"]} for r in recs]
            if train_pt:
                planc_calib = calibrate_planc_tau(train_pt)
                planc_tau = planc_calib["tau"]
                logger.info(f"[A3+PlanC calib] (TRAIN-only, no leakage) "
                            f"q*={planc_calib['q_star']:.2f} τ={planc_tau:.3f} "
                            f"stability={planc_calib.get('q_star_stability', 0):.0%} "
                            f"CV-score={planc_calib['score_mean']:.3f}±"
                            f"{planc_calib['score_std']:.3f} "
                            f"n_train={planc_calib['n_train']}")
            else:
                logger.warning("[A3+PlanC] train_calib.records is empty; τ=0 (degenerates to A3).")
                planc_tau = 0.0
                planc_calib = {"q_star": 0.0, "tau": 0.0, "note": "empty train_calib"}
        else:
            logger.warning("[A3+PlanC] --train-calib not provided or missing; "
                           "τ=0 (degenerates to A3). "
                           "Re-run induction with the new code to produce train_calib.json.")
            planc_tau = 0.0
            planc_calib = {"q_star": 0.0, "tau": 0.0,
                            "note": "no train_calib provided"}
        results["meta"]["plan_c_calibration"] = planc_calib

    # State accumulators (kept for diagnostic logs only, NOT used for τ)
    per_task_b0: dict[int, dict] = {}
    per_task_a3: dict[int, dict] = {}

    for method in method_set:
        logger.info(f"\n{'='*70}\n[run] method={method}\n{'='*70}")
        if method not in METHOD_BUILDERS:
            logger.warning(f"Unknown method {method}, skipping.")
            continue

        # Pick the right bank
        if method == "B2":
            sk, embs = skills_b2, skill_embs_b2
        elif method in ("A3", "A3+PlanC"):
            sk, embs = skills_a3, skill_embs_a3
        else:
            sk, embs = [], np.zeros((0,), dtype=np.float32)

        # (Plan-C τ has been pre-calibrated from train data above.)

        per_task: list[dict] = []
        builder = METHOD_BUILDERS[method]
        method_t0 = time.time()
        method_token_start = llm._total_tokens

        for k, gi in enumerate(test_idx):
            # Need the task description before calling builder, so do an env.reset.
            obs, info = env.reset(game_idx=gi)
            task = info["task"]

            # Build skill block
            kwargs = {"top_k": args.top_k}
            if method == "A3+PlanC":
                kwargs["tau_void"] = planc_tau or 0.0
            try:
                skill_block, sb_meta = builder(
                    query=task,
                    skills=sk,
                    skill_embs=embs,
                    encoder=encoder,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"[task {gi}] builder error: {e}")
                skill_block, sb_meta = "", {"method": method, "error": str(e)}

            # Run ReAct (re-reset inside the function)
            tokens_before = llm._total_tokens
            res = run_react_episode(
                env=env, game_idx=gi, llm=llm,
                skill_block=skill_block, max_steps=args.max_steps,
            )
            tokens_after = llm._total_tokens
            res["tokens"] = tokens_after - tokens_before
            res["meta"] = sb_meta
            per_task.append(res)

            if method == "B0":
                per_task_b0[gi] = res
            if method == "A3":
                per_task_a3[gi] = res

            logger.info(f"  [{method}] task {k+1}/{len(test_idx)} "
                        f"(idx={gi}, type={task_type_from_gamefile(gamefiles[gi])}): "
                        f"won={res['won']} steps={res['steps']} tok={res['tokens']}"
                        + (f" s_max={sb_meta.get('s_max', 0.0):.3f}" if method != 'B0' else "")
                        + (' VOID' if sb_meta.get('void') else ''))

            # Periodic checkpoint
            if (k + 1) % 10 == 0:
                _flush_partial(out_path, results, method, per_task)

        method_elapsed = time.time() - method_t0
        method_tokens = llm._total_tokens - method_token_start
        sr = float(np.mean([t["won"] for t in per_task])) if per_task else 0.0
        avg_steps = float(np.mean([t["steps"] for t in per_task])) if per_task else 0.0
        avg_tok = float(np.mean([t["tokens"] for t in per_task])) if per_task else 0.0
        n_won = sum(1 for t in per_task if t["won"])
        per_type_sr: dict[str, float] = defaultdict(list)
        for t in per_task:
            tt = task_type_from_gamefile(gamefiles[t["game_idx"]])
            per_type_sr[tt].append(float(t["won"]))
        per_type_sr_avg = {tt: float(np.mean(v)) for tt, v in per_type_sr.items()}

        results["methods"][method] = {
            "sr": sr,
            "n_won": n_won,
            "n_total": len(per_task),
            "avg_steps": avg_steps,
            "avg_tokens": avg_tok,
            "elapsed_sec": round(method_elapsed, 1),
            "total_tokens": method_tokens,
            "per_type_sr": per_type_sr_avg,
            "per_task": per_task,
        }
        logger.info(f"[done] {method}: SR={sr:.1%} ({n_won}/{len(per_task)}) "
                    f"avg_steps={avg_steps:.1f} avg_tok={avg_tok:.0f} "
                    f"elapsed={method_elapsed/60:.1f}min")

        # Flush after each method
        _flush_partial(out_path, results, method, per_task)

    # Final write
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\n[final] saved to {out_path}")
    return results

def _flush_partial(out_path: Path, results: dict, method: str, per_task: list) -> None:
    """Best-effort partial save — overwrite the same file with current state."""
    snap = dict(results)
    snap["_partial"] = {"method_in_progress": method, "n_done": len(per_task)}
    try:
        out_path.write_text(json.dumps(snap, indent=2, default=str))
    except Exception as e:
        logger.warning(f"partial flush failed: {e}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="valid_unseen", choices=["valid_seen", "valid_unseen"])
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--methods", nargs="+",
                   default=["B0", "B2", "A3", "A3+PlanC"],
                   help="Methods to evaluate.")
    p.add_argument("--skill-bank-b2", type=str, default=None,
                   help="JSON path to B2 (raw) skill bank.")
    p.add_argument("--skill-bank-a3", type=str, default=None,
                   help="JSON path to A3 (curated) skill bank.")
    p.add_argument("--train-calib", type=str, default=None,
                   help="JSON path to train_calib.json from induction "
                        "(used by A3+PlanC for τ calibration; no test leakage).")
    p.add_argument("--output", type=str,
                   default="experiments/alfworld_eval_results.json")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path("experiments").mkdir(parents=True, exist_ok=True)
    logger.add(f"experiments/alfworld_eval_{int(time.time())}.log",
               level="INFO", rotation="50 MB")
    run_eval(args)