#!/usr/bin/env python3
"""ALFWorld skill induction — produce skill banks for B2 (raw) and A3 (curated)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from loguru import logger

from src.models import Skill
from src.utils.alfworld_env import AlfworldEnv, task_type_from_gamefile
from src.utils.config import load_env
from src.utils.llm import LLMClient

# Reuse ReAct primitives from the eval runner
from scripts.run_alfworld_eval import (  # type: ignore[import]
    run_react_episode,
    parse_action,
    REACT_SYSTEM_BASE,
    stratified_sample,
)

# LLM-based skill distillation

DISTILL_SYSTEM = """You are an expert at extracting reusable procedural skills from agent trajectories.

Given a successful trajectory (task description + sequence of actions that led to success),
produce a single concise Skill in strict JSON form:

{
  "name": "<imperative short name, e.g. 'Heat object in microwave then place'>",
  "description": "<1-sentence summary of when this skill applies>",
  "preconditions": ["<bullet>", ...],
  "procedure": ["<step 1>", "<step 2>", ...],   // 4-8 steps, abstracted (use placeholders like {object}, {receptacle})
  "constraints": ["<edge case to avoid>", ...]  // 1-3 items
}

The skill MUST be a generalization (use placeholders, not specific objects from this trajectory).
Output ONLY the JSON, no commentary."""

MERGE_SYSTEM = """You are an expert at distilling redundant procedural skills.

Given several skills that solve similar tasks, merge them into a SINGLE more general skill.
Output strict JSON form:

{
  "name": "<concise general name>",
  "description": "<1-sentence summary covering all input skills>",
  "preconditions": ["<bullet>", ...],
  "procedure": ["<step 1>", ...],   // 4-8 steps, use placeholders like {object} {receptacle}
  "constraints": ["<edge case>", ...]
}

Be more general than any one input but still actionable. Output ONLY the JSON."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(reply: str) -> dict | None:
    m = _JSON_RE.search(reply)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def distill_skill(llm: LLMClient, task: str, actions: list[str], task_type: str) -> Skill | None:
    """Ask LLM to summarize a successful trajectory into a Skill."""
    actions_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions))
    user = (
        f"Task type: {task_type}\n"
        f"Task: {task}\n\n"
        f"Successful action sequence:\n{actions_str}\n\n"
        f"Now extract a reusable skill (JSON only)."
    )
    try:
        reply = llm.chat(
            messages=[
                {"role": "system", "content": DISTILL_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=600,
        )
    except Exception as e:
        logger.warning(f"distill LLM error: {e}")
        return None
    blob = _extract_json(reply)
    if not blob:
        logger.warning(f"distill failed (no JSON) for task={task!r}")
        return None
    try:
        return Skill(
            name=str(blob.get("name", task_type))[:120],
            description=str(blob.get("description", task))[:300],
            preconditions=list(blob.get("preconditions", []))[:5],
            procedure=list(blob.get("procedure", []))[:8],
            constraints=list(blob.get("constraints", []))[:3],
            source_tasks=[task],
        )
    except Exception as e:
        logger.warning(f"Skill construction error: {e}")
        return None

def merge_skills_llm(llm: LLMClient, group: list[Skill], group_label: str) -> Skill | None:
    """Ask LLM to merge a group of skills into a single more general skill."""
    if not group:
        return None
    if len(group) == 1:
        return group[0]
    skills_str = "\n\n".join(
        f"--- Skill {i+1}: {s.name} ---\n"
        f"Description: {s.description}\n"
        f"Procedure:\n" + "\n".join(f"  - {p}" for p in s.procedure) +
        ("\nConstraints:\n" + "\n".join(f"  - {c}" for c in s.constraints) if s.constraints else "")
        for i, s in enumerate(group)
    )
    user = (
        f"Skill group label: {group_label}\n"
        f"Number of skills to merge: {len(group)}\n\n"
        f"{skills_str}\n\n"
        f"Now produce a single merged skill (JSON only)."
    )
    try:
        reply = llm.chat(
            messages=[
                {"role": "system", "content": MERGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=700,
        )
    except Exception as e:
        logger.warning(f"merge LLM error: {e}")
        return group[0]
    blob = _extract_json(reply)
    if not blob:
        logger.warning(f"merge failed (no JSON) for group={group_label}")
        return group[0]
    try:
        merged = Skill(
            name=str(blob.get("name", group_label))[:120],
            description=str(blob.get("description", ""))[:300],
            preconditions=list(blob.get("preconditions", []))[:5],
            procedure=list(blob.get("procedure", []))[:8],
            constraints=list(blob.get("constraints", []))[:3],
            source_tasks=sum([s.source_tasks for s in group], []),
        )
        return merged
    except Exception as e:
        logger.warning(f"merged Skill construction error: {e}")
        return group[0]

# Embedding

def embed_skills(skills: list[Skill], encoder) -> np.ndarray:
    """Embed each skill's 'name + description' via the encoder, L2-normalize."""
    if not skills:
        return np.zeros((0, 384), dtype=np.float32)
    texts = [f"{s.name}. {s.description}" for s in skills]
    embs = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embs / norms).astype(np.float32)

def save_bank(skills: list[Skill], embs: np.ndarray, path: Path,
              method: str, induced_from: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "method": method,
        "induced_from": induced_from,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "n_skills": len(skills),
        "skills": [s.model_dump(mode="json") for s in skills],
        "embeddings": embs.tolist() if embs.size else [],
    }
    path.write_text(json.dumps(blob, indent=2, default=str))
    logger.info(f"[save] {method} bank → {path} ({len(skills)} skills)")

# Induction main

def induce(args) -> None:
    load_env()

    logger.info(f"[init] AlfworldEnv split={args.split} max_steps={args.max_steps}")
    env = AlfworldEnv(split=args.split, max_steps=args.max_steps)
    gamefiles = env.list_tasks()
    train_idx = stratified_sample(env.num_games, gamefiles, args.n_train, args.seed)
    logger.info(f"[init] num_games={env.num_games}, sampling n_train={args.n_train}")
    logger.info(f"[init] selected first 10 = {train_idx[:10]}, total {len(train_idx)}")

    type_counts = defaultdict(int)
    for gi in train_idx:
        type_counts[task_type_from_gamefile(gamefiles[gi])] += 1
    logger.info(f"[init] train type distribution: {dict(type_counts)}")

    llm = LLMClient({"temperature": 0.0, "max_tokens": 200, "timeout": 60})
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ----- Phase 1: split train_idx into induce_set + calib_set (no leakage) -----
    # Keep ~half for skill induction, the other half for τ calibration.
    # This avoids the within-sample bias of running induced skills back on the
    # very tasks they were distilled from.
    rng_split = np.random.default_rng(args.seed + 1)
    shuffled = list(train_idx)
    rng_split.shuffle(shuffled)
    n_induce = max(1, args.n_train // 2)
    induce_set = sorted(shuffled[:n_induce])
    calib_set = sorted(shuffled[n_induce:])
    logger.info(f"[init] split: |induce_set|={len(induce_set)} |calib_set|={len(calib_set)}")

    logger.info(f"\n{'='*70}\n[Phase 1] B0 rollouts on INDUCE set (collect successful trajectories)\n{'='*70}")
    successes: list[dict] = []  # {game_idx, task_type, task, actions} from induce_set
    n_attempted = 0
    n_won = 0
    for k, gi in enumerate(induce_set):
        n_attempted += 1
        # B0 ReAct rollout (no skills)
        res = run_react_episode(
            env=env, game_idx=gi, llm=llm,
            skill_block="", max_steps=args.max_steps,
        )
        tt = task_type_from_gamefile(gamefiles[gi])
        logger.info(f"  [induce {k+1}/{len(induce_set)}] idx={gi} type={tt}: "
                    f"won={res['won']} steps={res['steps']}")
        if res["won"]:
            n_won += 1
            successes.append({
                "game_idx": gi, "task_type": tt,
                "task": res["task"], "actions": res["actions"],
            })

    logger.info(f"\n[Phase 1] Done: {n_won}/{n_attempted} successful trajectories.")

    if not successes:
        logger.error("No successful trajectories — cannot induce skills. "
                     "Try increasing --n-train or --max-steps.")
        sys.exit(2)

    # ----- Phase 2: distill each into a raw Skill (B2 bank) -----
    logger.info(f"\n{'='*70}\n[Phase 2] Distilling raw skills (B2 bank)\n{'='*70}")
    raw_skills: list[Skill] = []
    for i, s in enumerate(successes):
        sk = distill_skill(llm, s["task"], s["actions"], s["task_type"])
        if sk is None:
            continue
        raw_skills.append(sk)
        logger.info(f"  [distill {i+1}/{len(successes)}] {sk.name!r}")

    b2_embs = embed_skills(raw_skills, encoder)
    save_bank(raw_skills, b2_embs, Path(args.output_b2),
              method="B2", induced_from=f"{args.split}[stratified-{args.n_train}]")

    # ----- Phase 3: merge intra-cluster (A3 bank: type-clustered + LLM-merged) -----
    logger.info(f"\n{'='*70}\n[Phase 3] Curating A3 bank (per-type merge)\n{'='*70}")
    by_type: dict[str, list[Skill]] = defaultdict(list)
    for sk, src in zip(raw_skills, successes):
        by_type[src["task_type"]].append(sk)

    a3_skills: list[Skill] = []
    for tt, group in sorted(by_type.items()):
        merged = merge_skills_llm(llm, group, group_label=tt)
        if merged:
            a3_skills.append(merged)
            logger.info(f"  [merge {tt}] {len(group)} skills → 1: {merged.name!r}")

    a3_embs = embed_skills(a3_skills, encoder)
    save_bank(a3_skills, a3_embs, Path(args.output_a3),
              method="A3", induced_from=f"{args.split}[stratified-{args.n_train}]")

    # ----- Phase 4: collect CALIB-set A3 + B0 rollouts for Plan-C τ calibration -----
    # Critical: the calib_set is DISJOINT from induce_set, so the induced bank
    # has not seen these tasks. This is what makes τ a held-out estimate.
    logger.info(f"\n{'='*70}\n[Phase 4] CALIB-set rollouts for τ calibration ({len(calib_set)} tasks)\n{'='*70}")
    from scripts.run_alfworld_eval import build_skill_block_a3
    calib_records: list[dict] = []
    for k, gi in enumerate(calib_set):
        tt = task_type_from_gamefile(gamefiles[gi])
        # B0 rollout on this calib task
        b0_res = run_react_episode(
            env=env, game_idx=gi, llm=llm,
            skill_block="", max_steps=args.max_steps,
        )
        # A3 rollout on the same task
        block, sb_meta = build_skill_block_a3(
            query=b0_res["task"], skills=a3_skills, skill_embs=a3_embs,
            encoder=encoder, top_k=3,
        )
        a3_res = run_react_episode(
            env=env, game_idx=gi, llm=llm,
            skill_block=block, max_steps=args.max_steps,
        )
        rec = {
            "game_idx": gi,
            "task_type": tt,
            "task": b0_res["task"],
            "s_max": float(sb_meta.get("s_max", 0.0)),
            "sr_void": float(b0_res["won"]),
            "sr_inject": float(a3_res["won"]),
            "steps_void": b0_res["steps"],
            "steps_inject": a3_res["steps"],
        }
        calib_records.append(rec)
        logger.info(f"  [calib {k+1}/{len(calib_set)}] idx={gi} type={tt}: "
                    f"sr_void={rec['sr_void']:.0f} sr_inject={rec['sr_inject']:.0f} "
                    f"s_max={rec['s_max']:.3f}")

    calib_path = Path(args.output_a3).parent / "train_calib.json"
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    if calib_records:
        n_void_only = sum(1 for c in calib_records if c["sr_void"] and not c["sr_inject"])
        n_inject_only = sum(1 for c in calib_records if c["sr_inject"] and not c["sr_void"])
        n_both = sum(1 for c in calib_records if c["sr_void"] and c["sr_inject"])
        n_neither = sum(1 for c in calib_records if not c["sr_void"] and not c["sr_inject"])
        sr_void_avg = float(np.mean([c["sr_void"] for c in calib_records]))
        sr_inject_avg = float(np.mean([c["sr_inject"] for c in calib_records]))
    else:
        n_void_only = n_inject_only = n_both = n_neither = 0
        sr_void_avg = sr_inject_avg = 0.0
    calib_blob = {
        "split": args.split,
        "n_train": args.n_train,
        "n_induce": len(induce_set),
        "n_calib": len(calib_set),
        "records": calib_records,
        "summary": {
            "n": len(calib_records),
            "sr_void": sr_void_avg,
            "sr_inject": sr_inject_avg,
            "void_only": n_void_only,
            "inject_only": n_inject_only,
            "both": n_both,
            "neither": n_neither,
        },
    }
    calib_path.write_text(json.dumps(calib_blob, indent=2, default=str))
    logger.info(f"  [save] train calibration → {calib_path}")
    logger.info(f"  Summary: void={sr_void_avg:.1%}  "
                f"inject={sr_inject_avg:.1%}  "
                f"both={n_both} void_only={n_void_only} inject_only={n_inject_only}")

    logger.info(f"\n{'='*70}\n[FINAL]\n"
                f"  Successful rollouts: {n_won}/{n_attempted}\n"
                f"  B2 bank: {len(raw_skills)} skills → {args.output_b2}\n"
                f"  A3 bank: {len(a3_skills)} skills → {args.output_a3}\n"
                f"  Total LLM tokens: {llm._total_tokens}\n")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="valid_seen", choices=["train", "valid_seen", "valid_unseen"])
    p.add_argument("--n-train", type=int, default=40,
                   help="Stratified sample size from train split. Half is used for "
                        "induction (B0 success → distill → merge), the other half "
                        "is held out as calib set for Plan-C τ calibration.")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=2024)  # different from eval seed
    p.add_argument("--output-b2", type=str, default="experiments/alfworld_skills/b2_bank.json")
    p.add_argument("--output-a3", type=str, default="experiments/alfworld_skills/a3_bank.json")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path("experiments").mkdir(parents=True, exist_ok=True)
    logger.add(f"experiments/alfworld_induce_{int(time.time())}.log",
               level="INFO", rotation="50 MB")
    induce(args)
