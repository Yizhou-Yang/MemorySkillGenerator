#!/usr/bin/env python3
"""
WebShop evaluation runner — trace-based single-step action prediction.

Pipeline:
    1. Load the public webshop trace dataset (Skyler215/webshop-agent-cot).
       Each row is a (state, gold_action) decision point from a successful
       trajectory. We do NOT run the full simulator (deferred to camera-ready).
    2. Build / load skill bank (induce_webshop_skills.py output).
    3. For each method ∈ {B0, B2, A3, A3+Plan-C}:
         - For each test task:
             query the LLM once given (instruction+history+obs+valid_actions)
             plus optional skill_block, snap reply onto valid_actions, score.
    4. Apply Plan C: τ_b = quantile_q*(train_s_max), q* via 5-fold CV on a
       calibration set produced by induce_webshop_skills.py.
    5. Write JSON with per-task records, per-type breakdown, Buy-Match SR.

Metrics:
    - Action-Match Accuracy: prediction == gold (case-insensitive, normalised)
    - Buy-Match Accuracy   : same metric restricted to gold==click[buy now]
                              (proxy for task-success rate; 'right product picked')
    - Per-type accuracy    : 6 categories from task_type_from_action()

Usage:
    python scripts/run_webshop_eval.py \
        --skill-bank-b2 experiments/webshop_skills/b2_bank.json \
        --skill-bank-a3 experiments/webshop_skills/a3_bank.json \
        --train-calib experiments/webshop_skills/train_calib.json \
        --methods B0 B2 A3 A3+PlanC \
        --n-test 100 --seed 42 \
        --output experiments/webshop_eval_results.json
"""

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

from src.curation.void_case import calibrate_tau_quantile, cv_select_quantile
from src.models import Skill
from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.utils.skill_formatter import FormattingConfig, format_skill_library
from src.utils.webshop_env import WebShopTraceEnv


# ============================================================
# Skill bank IO  (mirrors run_alfworld_eval.load_skill_bank)
# ============================================================

def load_skill_bank(path: Path) -> tuple[list[Skill], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Skill bank not found: {path}")
    blob = json.loads(path.read_text())
    skills = [Skill(**s) for s in blob["skills"]]
    embs = np.array(blob["embeddings"], dtype=np.float32)
    if embs.size == 0:
        return skills, embs
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return skills, embs


def embed_query(text: str, encoder) -> np.ndarray:
    v = encoder.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def retrieve_skills(
    query_emb: np.ndarray,
    skills: list[Skill],
    skill_embs: np.ndarray,
    top_k: int,
) -> tuple[list[Skill], float]:
    if not skills or skill_embs.size == 0:
        return [], 0.0
    sims = (skill_embs @ query_emb).flatten()
    s_max = float(sims.max()) if sims.size else 0.0
    idx = np.argsort(sims)[::-1][:top_k]
    return [skills[i] for i in idx], s_max


# ============================================================
# Single-step decision prompt
# ============================================================

PROMPT_SYSTEM_BASE = """You are a web-shopping agent solving a customer's request.

You see:
  - Instruction: what the customer wants.
  - History   : a brief log of earlier actions and their results.
  - Observation: the current page rendered as text.
  - Valid actions: a closed list — your reply MUST be exactly one item from it.

Output exactly two lines:
    Thought: <one short sentence>
    Action: <one valid_action, copied verbatim from the Valid actions list>

The Action MUST appear in the Valid actions list verbatim (case-insensitive).
Examples of valid actions: click[buy now], click[< prev], click[red], search[blue hoodie]."""


PROMPT_SYSTEM_WITH_SKILLS = PROMPT_SYSTEM_BASE + """

The following procedural skills, learned from past successful trajectories,
may help. Use them as guidance; ignore them when not applicable:

{skill_block}
"""


_ACTION_RE = re.compile(r"Action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def parse_action(reply: str) -> str:
    """Pull the 'Action: ...' line out of an LLM reply (raw text, no snapping)."""
    m = _ACTION_RE.search(reply)
    if m:
        return m.group(1).strip().rstrip(".").strip("`").strip()
    # Fallback: return last non-empty line
    for ln in reversed(reply.strip().splitlines()):
        if ln.strip():
            return ln.strip()
    return ""


def predict_action(
    llm: LLMClient,
    sample: dict[str, Any],
    skill_block: str,
    max_admissible_show: int = 60,
) -> tuple[str, int]:
    """Single LLM call; returns (snapped_prediction, tokens_consumed)."""
    if skill_block:
        sys_prompt = PROMPT_SYSTEM_WITH_SKILLS.format(skill_block=skill_block)
    else:
        sys_prompt = PROMPT_SYSTEM_BASE

    valid_actions = sample["valid_actions"][:max_admissible_show]
    valid_str = "\n".join(f"- {a}" for a in valid_actions)
    if len(sample["valid_actions"]) > max_admissible_show:
        valid_str += f"\n- ... ({len(sample['valid_actions']) - max_admissible_show} more)"

    user_msg = (
        f"Instruction: {sample['instruction']}\n\n"
        f"Observation:\n{sample['observation'][:1500]}\n\n"
        f"Valid actions:\n{valid_str}\n\n"
        f"Now output:\nThought: ...\nAction: ..."
    )

    tokens_before = llm._total_tokens
    try:
        reply = llm.chat(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=120,
        )
    except Exception as e:
        logger.warning(f"[task {sample['id']}] LLM error: {e}")
        reply = f"Action: {sample['valid_actions'][0] if sample['valid_actions'] else ''}"
    tokens_used = llm._total_tokens - tokens_before

    raw = parse_action(reply)
    snapped = WebShopTraceEnv.snap_to_valid(raw, sample["valid_actions"])
    return snapped, tokens_used


# ============================================================
# Method-specific skill block builders (mirrors run_alfworld_eval)
# ============================================================

def build_skill_block_b0(query: str, **kwargs) -> tuple[str, dict]:
    return "", {"method": "B0", "n_skills_used": 0, "s_max": 0.0}


def _format_skills_plain(skills: list[Skill]) -> str:
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
    query: str, skills: list[Skill], skill_embs: np.ndarray,
    encoder, top_k: int = 3, **kwargs,
) -> tuple[str, dict]:
    if not skills:
        return "", {"method": "B2", "n_skills_used": 0, "s_max": 0.0}
    q = embed_query(query, encoder)
    selected, s_max = retrieve_skills(q, skills, skill_embs, top_k)
    return _format_skills_plain(selected), {
        "method": "B2", "n_skills_used": len(selected), "s_max": s_max,
    }


def build_skill_block_a3(
    query: str, skills: list[Skill], skill_embs: np.ndarray,
    encoder, top_k: int = 3, **kwargs,
) -> tuple[str, dict]:
    if not skills:
        return "", {"method": "A3", "n_skills_used": 0, "s_max": 0.0}
    q = embed_query(query, encoder)
    selected, s_max = retrieve_skills(q, skills, skill_embs, top_k)
    block = format_skill_library(selected, FormattingConfig(max_skills_in_prompt=top_k))
    return block, {"method": "A3", "n_skills_used": len(selected), "s_max": s_max}


def build_skill_block_a3_planc(
    query: str, skills: list[Skill], skill_embs: np.ndarray,
    encoder, top_k: int = 3, tau_void: float = 0.0, **kwargs,
) -> tuple[str, dict]:
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


# ============================================================
# Stratified sampling on (id, task_type)
# ============================================================

def stratified_sample_records(
    records: list[dict], n: int, seed: int,
    exclude_ids: set[str] | None = None,
) -> list[int]:
    """Sample n indices stratified by task_type. Returns sorted list of indices."""
    exclude_ids = exclude_ids or set()
    rng = np.random.default_rng(seed)

    by_type: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r["id"] in exclude_ids:
            continue
        by_type[r["task_type"]].append(i)

    n_types = len(by_type)
    if n_types == 0:
        return []
    per_type = max(1, n // n_types)
    selected: list[int] = []
    for t in sorted(by_type.keys()):
        idxs = list(by_type[t])
        rng.shuffle(idxs)
        selected.extend(idxs[:per_type])
    if len(selected) < n:
        leftovers = [
            i for i, r in enumerate(records)
            if i not in selected and r["id"] not in exclude_ids
        ]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    selected = selected[:n]
    selected.sort()
    return selected


# ============================================================
# Plan-C calibration  (re-uses CV machinery from void_case)
# ============================================================

def calibrate_planc_tau(
    train_per_task: list[dict],
    q_grid: tuple[float, ...] = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70),
    seeds: tuple[int, ...] = (42, 123, 456, 789, 2024),
    n_folds: int = 5,
) -> dict[str, Any]:
    if not train_per_task:
        return {"q_star": 0.30, "tau": 0.0, "score_mean": 0.0, "score_std": 0.0,
                "n_train": 0}
    s = np.array([t["s_max"] for t in train_per_task], dtype=np.float64)
    ei = np.array([t["sr_inject"] for t in train_per_task], dtype=np.float64)
    ev = np.array([t["sr_void"] for t in train_per_task], dtype=np.float64)

    from collections import Counter
    q_choices, scores = [], []
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


# ============================================================
# Main eval
# ============================================================

def run_eval(args) -> dict[str, Any]:
    load_env()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- Load env -----
    logger.info(f"[init] Loading WebShopTraceEnv split={args.split}")
    env = WebShopTraceEnv(split=args.split)
    records = env.list_tasks()
    logger.info(f"[init] num_tasks={env.num_tasks}, type distribution: "
                f"{env.task_type_distribution()}")

    # Exclude trajectories used in induction (to prevent leakage)
    exclude_ids: set[str] = set()
    if args.exclude_ids_file and Path(args.exclude_ids_file).exists():
        exclude_ids = set(json.loads(Path(args.exclude_ids_file).read_text()))
        logger.info(f"[init] Excluding {len(exclude_ids)} ids used in induction")

    test_idx = stratified_sample_records(records, args.n_test, args.seed, exclude_ids)
    logger.info(f"[init] selected n={len(test_idx)} test indices "
                f"(first 10: {test_idx[:10]})")
    sample_types = [records[i]["task_type"] for i in test_idx]
    type_counts = dict((t, sample_types.count(t)) for t in set(sample_types))
    logger.info(f"[init] sample type distribution: {type_counts}")

    # ----- LLM + encoder -----
    llm = LLMClient({"temperature": 0.0, "max_tokens": 120, "timeout": 60})
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ----- Skill banks -----
    skills_b2, embs_b2 = [], np.zeros((0,), dtype=np.float32)
    skills_a3, embs_a3 = [], np.zeros((0,), dtype=np.float32)
    if args.skill_bank_b2:
        skills_b2, embs_b2 = load_skill_bank(Path(args.skill_bank_b2))
        logger.info(f"[init] B2 bank: {len(skills_b2)} skills")
    if args.skill_bank_a3:
        skills_a3, embs_a3 = load_skill_bank(Path(args.skill_bank_a3))
        logger.info(f"[init] A3 bank: {len(skills_a3)} skills")

    # ----- Plan-C τ calibration (TRAIN-only) -----
    planc_tau: float | None = None
    planc_calib: dict[str, Any] = {}
    method_set = list(args.methods)
    desired_order = ["B0", "B2", "A3", "A3+PlanC"]
    method_set.sort(key=lambda m: desired_order.index(m) if m in desired_order else 99)
    if "A3+PlanC" in method_set:
        if args.train_calib and Path(args.train_calib).exists():
            blob = json.loads(Path(args.train_calib).read_text())
            recs = blob.get("records", [])
            train_pt = [{"s_max": r["s_max"], "sr_inject": r["sr_inject"],
                          "sr_void": r["sr_void"]} for r in recs]
            if train_pt:
                planc_calib = calibrate_planc_tau(train_pt)
                planc_tau = planc_calib["tau"]
                logger.info(f"[A3+PlanC calib] q*={planc_calib['q_star']:.2f} "
                            f"τ={planc_tau:.3f} "
                            f"stability={planc_calib.get('q_star_stability',0):.0%} "
                            f"CV={planc_calib['score_mean']:.3f}±"
                            f"{planc_calib['score_std']:.3f} "
                            f"n_train={planc_calib['n_train']}")
            else:
                planc_tau = 0.0
                planc_calib = {"q_star": 0.0, "tau": 0.0, "note": "empty train_calib"}
                logger.warning("[A3+PlanC] empty train_calib; τ=0")
        else:
            planc_tau = 0.0
            planc_calib = {"q_star": 0.0, "tau": 0.0, "note": "no train_calib"}
            logger.warning("[A3+PlanC] no train_calib provided; τ=0")

    results: dict[str, Any] = {
        "meta": {
            "benchmark": "webshop_trace",
            "split": args.split,
            "n_test": args.n_test,
            "actual_n_test": len(test_idx),
            "seed": args.seed,
            "test_indices": test_idx,
            "test_ids": [records[i]["id"] for i in test_idx],
            "type_counts": type_counts,
            "model": llm.model,
            "skill_bank_b2": args.skill_bank_b2,
            "skill_bank_a3": args.skill_bank_a3,
            "exclude_ids_count": len(exclude_ids),
            "plan_c_calibration": planc_calib if "A3+PlanC" in method_set else None,
        },
        "methods": {},
    }

    # ----- Run each method -----
    for method in method_set:
        logger.info(f"\n{'='*70}\n[run] method={method}\n{'='*70}")
        if method not in METHOD_BUILDERS:
            logger.warning(f"Unknown method {method}, skipping.")
            continue
        if method == "B2":
            sk, embs = skills_b2, embs_b2
        elif method in ("A3", "A3+PlanC"):
            sk, embs = skills_a3, embs_a3
        else:
            sk, embs = [], np.zeros((0,), dtype=np.float32)

        builder = METHOD_BUILDERS[method]
        per_task: list[dict] = []
        method_t0 = time.time()
        method_token_start = llm._total_tokens

        for k, idx in enumerate(test_idx):
            sample = records[idx]
            query = sample["instruction"]

            kwargs = {"top_k": args.top_k}
            if method == "A3+PlanC":
                kwargs["tau_void"] = planc_tau or 0.0
            try:
                skill_block, sb_meta = builder(
                    query=query, skills=sk, skill_embs=embs,
                    encoder=encoder, **kwargs,
                )
            except Exception as e:
                logger.warning(f"[{method} task {idx}] builder error: {e}")
                skill_block, sb_meta = "", {"method": method, "error": str(e)}

            t0 = time.time()
            prediction, tokens = predict_action(
                llm=llm, sample=sample, skill_block=skill_block,
            )
            elapsed = time.time() - t0
            correct = WebShopTraceEnv.score(prediction, sample["gold_action"])

            per_task.append({
                "task_idx": idx,
                "id": sample["id"],
                "task_type": sample["task_type"],
                "instruction": sample["instruction"],
                "gold": sample["gold_action"],
                "prediction": prediction,
                "correct": bool(correct),
                "tokens": tokens,
                "elapsed_sec": round(elapsed, 2),
                "meta": sb_meta,
            })

            logger.info(
                f"  [{method}] {k+1}/{len(test_idx)} id={sample['id']} "
                f"type={sample['task_type']}: "
                f"{'✓' if correct else '✗'} "
                f"pred={prediction!r} gold={sample['gold_action']!r} tok={tokens}"
                + (f" s_max={sb_meta.get('s_max',0):.3f}" if method != 'B0' else "")
                + (' VOID' if sb_meta.get('void') else '')
            )

            if (k + 1) % 20 == 0:
                _flush_partial(out_path, results, method, per_task)

        method_elapsed = time.time() - method_t0
        method_tokens = llm._total_tokens - method_token_start

        # ---- Aggregate ----
        n_total = len(per_task)
        n_correct = sum(1 for t in per_task if t["correct"])
        acc = n_correct / max(n_total, 1)
        avg_tok = float(np.mean([t["tokens"] for t in per_task])) if per_task else 0.0

        # Per-type
        per_type_acc: dict[str, float] = {}
        per_type_n: dict[str, int] = {}
        type_buckets: dict[str, list[bool]] = defaultdict(list)
        for t in per_task:
            type_buckets[t["task_type"]].append(t["correct"])
        for tt, vals in type_buckets.items():
            per_type_acc[tt] = float(np.mean(vals))
            per_type_n[tt] = len(vals)

        # Buy-Match (proxy SR)
        buy_correct = sum(1 for t in per_task if t["task_type"] == "buy" and t["correct"])
        buy_total = sum(1 for t in per_task if t["task_type"] == "buy")
        buy_acc = buy_correct / max(buy_total, 1) if buy_total else None

        results["methods"][method] = {
            "accuracy": acc,
            "n_correct": n_correct,
            "n_total": n_total,
            "buy_match_accuracy": buy_acc,
            "buy_match_n": buy_total,
            "per_type_accuracy": per_type_acc,
            "per_type_n": per_type_n,
            "avg_tokens": avg_tok,
            "elapsed_sec": round(method_elapsed, 1),
            "total_tokens": method_tokens,
            "per_task": per_task,
        }
        logger.info(
            f"[done] {method}: acc={acc:.1%} ({n_correct}/{n_total}) "
            f"buy_match={'-' if buy_acc is None else f'{buy_acc:.1%}'} "
            f"({buy_correct}/{buy_total}) "
            f"avg_tok={avg_tok:.0f} elapsed={method_elapsed/60:.1f}min"
        )
        _flush_partial(out_path, results, method, per_task)

    # ---- Final ----
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\n[final] saved to {out_path}")
    return results


def _flush_partial(out_path: Path, results: dict, method: str, per_task: list) -> None:
    snap = dict(results)
    snap["_partial"] = {"method_in_progress": method, "n_done": len(per_task)}
    try:
        out_path.write_text(json.dumps(snap, indent=2, default=str))
    except Exception as e:
        logger.warning(f"partial flush failed: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--methods", nargs="+",
                   default=["B0", "B2", "A3", "A3+PlanC"])
    p.add_argument("--skill-bank-b2", type=str, default=None)
    p.add_argument("--skill-bank-a3", type=str, default=None)
    p.add_argument("--train-calib", type=str, default=None,
                   help="JSON path to train_calib.json from induction.")
    p.add_argument("--exclude-ids-file", type=str, default=None,
                   help="JSON list of trace ids to exclude from test sampling "
                        "(prevents leakage from induction set).")
    p.add_argument("--output", type=str,
                   default="experiments/webshop_eval_results.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path("experiments").mkdir(parents=True, exist_ok=True)
    logger.add(f"experiments/webshop_eval_{int(time.time())}.log",
               level="INFO", rotation="50 MB")
    run_eval(args)
