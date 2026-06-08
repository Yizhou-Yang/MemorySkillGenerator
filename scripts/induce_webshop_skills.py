#!/usr/bin/env python3
"""WebShop skill induction — produce skill banks for B2 (raw) and A3 (curated)"""

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
from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.utils.webshop_env import WebShopTraceEnv, task_type_from_action

# Single-step predictor (re-use eval primitives)
from scripts.run_webshop_eval import (  # type: ignore[import]
    predict_action,
    build_skill_block_a3,
)

# Trajectory grouping

_TRAJ_RE = re.compile(r"(.+)_step_(\d+)")

def group_by_trajectory(records: list[dict]) -> dict[str, list[int]]:
    """Group record indices by trajectory id. Returns {traj_id: [idx, idx, ...] sorted by step}."""
    by_traj: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, r in enumerate(records):
        m = _TRAJ_RE.match(r["id"])
        if m:
            by_traj[m.group(1)].append((int(m.group(2)), i))
        else:
            by_traj[r["id"]].append((0, i))
    return {traj: [i for _, i in sorted(items)] for traj, items in by_traj.items()}

# LLM-based skill distillation

DISTILL_SYSTEM = """You are an expert at extracting reusable web-shopping skills from agent trajectories.

Given an instruction and a sequence of (observation_summary, action) pairs that
came from a SUCCESSFUL shopping trajectory, produce a single concise Skill in
strict JSON form:

{
  "name": "<imperative short name, e.g. 'Search & filter products by attribute then buy'>",
  "description": "<1-sentence summary of when this skill applies>",
  "preconditions": ["<bullet>", ...],
  "procedure": ["<step 1>", "<step 2>", ...],   // 4-8 abstracted steps,
                                                // use placeholders like {item}, {attribute}, {price_limit}
  "constraints": ["<edge case>", ...]
}

Use placeholders, NOT specific products from this trajectory.
Output ONLY the JSON, no commentary."""

MERGE_SYSTEM = """You are an expert at distilling redundant web-shopping skills.

Given several skills that handle similar shopping situations, merge them into
a SINGLE more general skill. Output strict JSON form:

{
  "name": "<concise general name>",
  "description": "<1-sentence summary covering all input skills>",
  "preconditions": ["<bullet>", ...],
  "procedure": ["<step 1>", ...],   // 4-8 steps with placeholders
  "constraints": ["<edge case>", ...]
}

Output ONLY the JSON."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json(reply: str) -> dict | None:
    m = _JSON_RE.search(reply)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _summarise_observation(obs: str, max_chars: int = 200) -> str:
    """Compress a webshop observation into a 1-line summary (drop UI fluff)."""
    obs = re.sub(r"\[.*?\]", "", obs)         # strip [button] markers
    obs = re.sub(r"\s+", " ", obs).strip()
    return obs[:max_chars]

def distill_skill(
    llm: LLMClient,
    instruction: str,
    steps: list[tuple[str, str]],
    task_type: str,
) -> Skill | None:
    """Ask LLM to summarise a (instruction, [(obs_summary, action)]) chain."""
    steps_str = "\n".join(
        f"  Step {i+1}: obs=\"{obs}\" → action: {act}"
        for i, (obs, act) in enumerate(steps)
    )
    user = (
        f"Task type: {task_type}\n"
        f"Instruction: {instruction}\n\n"
        f"Successful trajectory steps:\n{steps_str}\n\n"
        f"Now extract a reusable skill (JSON only)."
    )
    try:
        reply = llm.chat(
            messages=[
                {"role": "system", "content": DISTILL_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0, max_tokens=600,
        )
    except Exception as e:
        logger.warning(f"distill LLM error: {e}")
        return None
    blob = _extract_json(reply)
    if not blob:
        return None
    try:
        return Skill(
            name=str(blob.get("name", task_type))[:120],
            description=str(blob.get("description", instruction))[:300],
            preconditions=list(blob.get("preconditions", []))[:5],
            procedure=list(blob.get("procedure", []))[:8],
            constraints=list(blob.get("constraints", []))[:3],
            source_tasks=[instruction],
        )
    except Exception as e:
        logger.warning(f"Skill construction error: {e}")
        return None

def merge_skills_llm(llm: LLMClient, group: list[Skill], group_label: str) -> Skill | None:
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
            temperature=0.0, max_tokens=700,
        )
    except Exception as e:
        logger.warning(f"merge LLM error: {e}")
        return group[0]
    blob = _extract_json(reply)
    if not blob:
        return group[0]
    try:
        return Skill(
            name=str(blob.get("name", group_label))[:120],
            description=str(blob.get("description", ""))[:300],
            preconditions=list(blob.get("preconditions", []))[:5],
            procedure=list(blob.get("procedure", []))[:8],
            constraints=list(blob.get("constraints", []))[:3],
            source_tasks=sum([s.source_tasks for s in group], []),
        )
    except Exception as e:
        logger.warning(f"merged Skill construction error: {e}")
        return group[0]

# Embedding

def embed_skills(skills: list[Skill], encoder) -> np.ndarray:
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

# Trajectory characterisation (use the LAST step's task_type as label)

def trajectory_task_type(records: list[dict], step_idxs: list[int]) -> str:
    """Use the last visible step's task_type to label the trajectory."""
    last = records[step_idxs[-1]]
    return last["task_type"]

# Stratified trajectory sampler

def stratified_sample_trajectories(
    eligible_traj_ids: list[str],
    by_traj: dict[str, list[int]],
    records: list[dict],
    n: int,
    seed: int,
) -> list[str]:
    """Sample n trajectory IDs stratified by trajectory_task_type."""
    rng = np.random.default_rng(seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    for tid in eligible_traj_ids:
        tt = trajectory_task_type(records, by_traj[tid])
        by_type[tt].append(tid)
    n_types = len(by_type)
    if n_types == 0:
        return []
    per_type = max(1, n // n_types)
    selected: list[str] = []
    for tt in sorted(by_type.keys()):
        ids = list(by_type[tt])
        rng.shuffle(ids)
        selected.extend(ids[:per_type])
    if len(selected) < n:
        leftovers = [t for t in eligible_traj_ids if t not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    return selected[:n]

# Induction main

def induce(args) -> None:
    load_env()

    logger.info(f"[init] Loading WebShopTraceEnv split={args.split}")
    env = WebShopTraceEnv(split=args.split)
    records = env.list_tasks()
    logger.info(f"[init] num_tasks={env.num_tasks}, "
                f"type distribution: {env.task_type_distribution()}")

    by_traj = group_by_trajectory(records)
    logger.info(f"[init] num_trajectories={len(by_traj)}, "
                f"steps-per-traj: min={min(len(v) for v in by_traj.values())} "
                f"max={max(len(v) for v in by_traj.values())} "
                f"mean={np.mean([len(v) for v in by_traj.values()]):.2f}")

    eligible = [tid for tid, idxs in by_traj.items() if len(idxs) >= args.min_traj_steps]
    logger.info(f"[init] eligible trajectories (≥{args.min_traj_steps} steps): "
                f"{len(eligible)} / {len(by_traj)}")
    if len(eligible) < args.n_train:
        logger.warning(f"Only {len(eligible)} eligible; reducing n_train to that.")
        args.n_train = max(2, len(eligible))

    selected_traj = stratified_sample_trajectories(
        eligible, by_traj, records, args.n_train, args.seed,
    )
    logger.info(f"[init] selected {len(selected_traj)} trajectories: {selected_traj[:10]}...")

    # Split into induce_set / calib_set (50/50, disjoint)
    rng_split = np.random.default_rng(args.seed + 1)
    shuffled = list(selected_traj)
    rng_split.shuffle(shuffled)
    n_induce = max(1, args.n_train // 2)
    induce_set = shuffled[:n_induce]
    calib_set = shuffled[n_induce:]
    logger.info(f"[init] split: |induce_set|={len(induce_set)} |calib_set|={len(calib_set)}")

    # Track all sample IDs touched (so eval can exclude them)
    induced_ids: list[str] = []
    for tid in induce_set + calib_set:
        for idx in by_traj[tid]:
            induced_ids.append(records[idx]["id"])

    llm = LLMClient({"temperature": 0.0, "max_tokens": 200, "timeout": 60})
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ----- Phase 1: distill raw skills from induce_set -----
    logger.info(f"\n{'='*70}\n[Phase 1] Distilling raw skills (B2 bank)\n{'='*70}")
    raw_skills: list[Skill] = []
    for k, tid in enumerate(induce_set):
        idxs = by_traj[tid]
        # The trajectory's instruction (use the first visible step)
        first = records[idxs[0]]
        instruction = first["instruction"]
        steps: list[tuple[str, str]] = []
        for idx in idxs:
            r = records[idx]
            obs_sum = _summarise_observation(r["observation"])
            steps.append((obs_sum, r["gold_action"]))
        tt = trajectory_task_type(records, idxs)
        sk = distill_skill(llm, instruction, steps, tt)
        if sk:
            raw_skills.append(sk)
            logger.info(f"  [induce {k+1}/{len(induce_set)}] traj={tid} "
                        f"steps={len(steps)} type={tt}: {sk.name!r}")

    b2_embs = embed_skills(raw_skills, encoder)
    save_bank(raw_skills, b2_embs, Path(args.output_b2),
              method="B2",
              induced_from=f"webshop_trace[{args.split}, n_traj={len(induce_set)}]")

    # ----- Phase 2: merge per task_type → A3 bank -----
    logger.info(f"\n{'='*70}\n[Phase 2] Curating A3 bank (per-type merge)\n{'='*70}")
    by_type: dict[str, list[Skill]] = defaultdict(list)
    for sk, tid in zip(raw_skills, induce_set):
        tt = trajectory_task_type(records, by_traj[tid])
        by_type[tt].append(sk)
    a3_skills: list[Skill] = []
    for tt, group in sorted(by_type.items()):
        merged = merge_skills_llm(llm, group, group_label=tt)
        if merged:
            a3_skills.append(merged)
            logger.info(f"  [merge {tt}] {len(group)} skills → 1: {merged.name!r}")
    a3_embs = embed_skills(a3_skills, encoder)
    save_bank(a3_skills, a3_embs, Path(args.output_a3),
              method="A3",
              induced_from=f"webshop_trace[{args.split}, n_traj={len(induce_set)}]")

    # ----- Phase 3: CALIB rollouts (B0 + A3 single-step on calib_set) -----
    logger.info(f"\n{'='*70}\n[Phase 3] CALIB-set rollouts ({len(calib_set)} trajectories)\n{'='*70}")
    calib_records: list[dict] = []
    for k, tid in enumerate(calib_set):
        idxs = by_traj[tid]
        # Use the LAST visible step (most informative — usually buy/select decision)
        idx = idxs[-1]
        sample = records[idx]
        # B0: no skill block
        b0_pred, _ = predict_action(llm=llm, sample=sample, skill_block="")
        b0_correct = WebShopTraceEnv.score(b0_pred, sample["gold_action"])
        # A3: skill block + s_max
        block, sb_meta = build_skill_block_a3(
            query=sample["instruction"], skills=a3_skills, skill_embs=a3_embs,
            encoder=encoder, top_k=3,
        )
        a3_pred, _ = predict_action(llm=llm, sample=sample, skill_block=block)
        a3_correct = WebShopTraceEnv.score(a3_pred, sample["gold_action"])

        rec = {
            "trajectory_id": tid,
            "sample_id": sample["id"],
            "task_type": sample["task_type"],
            "instruction": sample["instruction"],
            "gold": sample["gold_action"],
            "s_max": float(sb_meta.get("s_max", 0.0)),
            "sr_void": float(b0_correct),
            "sr_inject": float(a3_correct),
            "pred_b0": b0_pred,
            "pred_a3": a3_pred,
        }
        calib_records.append(rec)
        logger.info(
            f"  [calib {k+1}/{len(calib_set)}] tid={tid} type={sample['task_type']}: "
            f"B0={'✓' if b0_correct else '✗'} A3={'✓' if a3_correct else '✗'} "
            f"s_max={rec['s_max']:.3f}"
        )

    # ---- Save calib + induced_ids ----
    calib_path = Path(args.output_a3).parent / "train_calib.json"
    if calib_records:
        sr_void_avg = float(np.mean([c["sr_void"] for c in calib_records]))
        sr_inject_avg = float(np.mean([c["sr_inject"] for c in calib_records]))
        n_void_only = sum(1 for c in calib_records if c["sr_void"] and not c["sr_inject"])
        n_inject_only = sum(1 for c in calib_records if c["sr_inject"] and not c["sr_void"])
        n_both = sum(1 for c in calib_records if c["sr_void"] and c["sr_inject"])
        n_neither = sum(1 for c in calib_records if not c["sr_void"] and not c["sr_inject"])
    else:
        sr_void_avg = sr_inject_avg = 0.0
        n_void_only = n_inject_only = n_both = n_neither = 0
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
    logger.info(f"  [save] calib → {calib_path}")
    logger.info(f"  Summary: void={sr_void_avg:.1%}  inject={sr_inject_avg:.1%}  "
                f"both={n_both} void_only={n_void_only} inject_only={n_inject_only}")

    ids_path = Path(args.output_a3).parent / "induced_ids.json"
    ids_path.write_text(json.dumps(induced_ids, indent=2))
    logger.info(f"  [save] induced_ids ({len(induced_ids)}) → {ids_path}")

    logger.info(f"\n{'='*70}\n[FINAL]\n"
                f"  B2 bank: {len(raw_skills)} skills → {args.output_b2}\n"
                f"  A3 bank: {len(a3_skills)} skills → {args.output_a3}\n"
                f"  Calib  : {len(calib_records)} records → {calib_path}\n"
                f"  Excl.  : {len(induced_ids)} ids → {ids_path}\n"
                f"  Total LLM tokens: {llm._total_tokens}\n")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--n-train", type=int, default=30,
                   help="Total trajectories sampled. Half for distillation, half for calib.")
    p.add_argument("--min-traj-steps", type=int, default=3,
                   help="Only consider trajectories with ≥ this many visible steps.")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--output-b2", type=str,
                   default="experiments/webshop_skills/b2_bank.json")
    p.add_argument("--output-a3", type=str,
                   default="experiments/webshop_skills/a3_bank.json")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path("experiments").mkdir(parents=True, exist_ok=True)
    logger.add(f"experiments/webshop_induce_{int(time.time())}.log",
               level="INFO", rotation="50 MB")
    induce(args)
