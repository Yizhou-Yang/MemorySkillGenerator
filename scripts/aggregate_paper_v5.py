#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate v5 void-case results into the paper Table 1 + sensitivity sweep.

Inputs:
    experiments/paper_v5_void_results.json

Outputs:
    paper/figures/table1_v5_void.md         — full per-benchmark comparison
    paper/figures/figure_void_sweep.png     — τ_void sensitivity per benchmark
    paper/figures/figure_void_sweep.pdf
    paper/figures/table_smax_distribution.md — per-benchmark s_max stats
    experiments/paper_v5_aggregated.json    — clean dict for downstream use
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = PROJECT_ROOT / "experiments" / "paper_v5_void_results.json"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def load_results() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def get_primary_metric_score(method_data: dict, primary_metric: str) -> float:
    """Get EM or F1 from a method's primary aggregate."""
    p = method_data["primary"]
    return p["avg_em"] if primary_metric == "em" else p["avg_f1"]


def build_table1(results: dict) -> str:
    """Build markdown Table 1: 6 methods × 7 benchmarks main metric."""
    main = results["main_experiment"]
    methods = results["meta"]["methods"]
    benches = list(main.keys())

    lines = []
    lines.append("# Table 1 — Main Results (Paper v5)")
    lines.append("")
    lines.append(f"τ_void (primary) = {results['meta']['primary_tau']}")
    lines.append("")

    # Header
    header = ["Benchmark", "Metric"] + methods
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    # Per-benchmark rows
    for bench in benches:
        bd = main[bench]
        if "error" in bd:
            lines.append(f"| {bench} | — | _error: {bd['error']}_ | | | | | |")
            continue
        primary = bd["primary_metric"].upper()
        row_vals = [bench, primary]
        for m in methods:
            md = bd["methods"].get(m)
            if not md:
                row_vals.append("—")
                continue
            score = get_primary_metric_score(md, bd["primary_metric"])
            row_vals.append(f"{score:.1%}")
        lines.append("| " + " | ".join(row_vals) + " |")

    # Mean row (across benchmarks)
    lines.append("|||||||||")
    mean_row = ["**Mean**", "—"]
    for m in methods:
        scores = []
        for bench in benches:
            bd = main[bench]
            if "error" in bd or m not in bd["methods"]:
                continue
            scores.append(get_primary_metric_score(bd["methods"][m], bd["primary_metric"]))
        if scores:
            mean_row.append(f"**{np.mean(scores):.1%}**")
        else:
            mean_row.append("—")
    lines.append("| " + " | ".join(mean_row) + " |")

    # Token cost row
    lines.append("")
    lines.append("## Token Cost (avg per task)")
    lines.append("")
    header = ["Benchmark"] + methods
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for bench in benches:
        bd = main[bench]
        if "error" in bd:
            continue
        row = [bench]
        for m in methods:
            md = bd["methods"].get(m)
            if not md:
                row.append("—")
                continue
            row.append(f"{md['primary']['avg_tokens']:.0f}")
        lines.append("| " + " | ".join(row) + " |")

    # Void rate row
    lines.append("")
    lines.append("## Void Rate (% of tasks routed to c_∅)")
    lines.append("")
    void_methods = [m for m in methods if "+void" in m]
    header = ["Benchmark"] + void_methods
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for bench in benches:
        bd = main[bench]
        if "error" in bd:
            continue
        row = [bench]
        for m in void_methods:
            md = bd["methods"].get(m)
            if not md:
                row.append("—")
                continue
            row.append(f"{md['primary']['void_rate']:.1%}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def plot_tau_sweep(results: dict, out_path: Path) -> None:
    """Sensitivity sweep: avg metric vs τ for A3+void per benchmark."""
    main = results["main_experiment"]
    sweep_taus = results["meta"]["sweep_taus"]

    benches = [b for b in main if "error" not in main[b]]
    if not benches:
        print("[plot_tau_sweep] no valid benchmarks, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")

    for i, bench in enumerate(benches):
        bd = main[bench]
        a3v = bd["methods"].get("A3+void")
        if not a3v or not a3v.get("sweep"):
            continue
        primary = bd["primary_metric"]
        scores = [s["avg_em"] if primary == "em" else s["avg_f1"]
                  for s in a3v["sweep"]]
        ax.plot(sweep_taus, scores, "o-", color=cmap(i), label=bench, linewidth=2,
                markersize=6)

    ax.set_xlabel(r"$\tau_{\text{void}}$ (similarity threshold)", fontsize=12)
    ax.set_ylabel("Primary metric (EM or F1)", fontsize=12)
    ax.set_title(r"A3+$c_\emptyset$ sensitivity to $\tau_{\text{void}}$",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, ncol=2)
    ax.axvline(x=results["meta"]["primary_tau"], color="red", linestyle="--",
               alpha=0.5, label=f"primary τ={results['meta']['primary_tau']}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()
    print(f"[plot_tau_sweep] saved → {out_path}")


def build_smax_table(results: dict) -> str:
    """Per-benchmark s_max distribution (helps choose τ_void)."""
    main = results["main_experiment"]
    lines = ["# s_max Distribution (per benchmark, A3 retrieval)", ""]
    lines.append("| Benchmark | n | mean | p10 | p25 | p50 | p75 | p90 | max |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for bench, bd in main.items():
        if "error" in bd:
            continue
        a3v = bd["methods"].get("A3+void") or bd["methods"].get("A3")
        if not a3v:
            continue
        s_maxes = [r["s_max"] for r in a3v["per_task"]]
        if not s_maxes:
            continue
        arr = np.array(s_maxes)
        lines.append(
            f"| {bench} | {len(arr)} | {arr.mean():.3f} | "
            f"{np.percentile(arr,10):.3f} | {np.percentile(arr,25):.3f} | "
            f"{np.percentile(arr,50):.3f} | {np.percentile(arr,75):.3f} | "
            f"{np.percentile(arr,90):.3f} | {arr.max():.3f} |"
        )
    return "\n".join(lines)


def calibrate_tau_lobo(results: dict) -> dict:
    """Leave-one-benchmark-out cross-validation for τ_void.

    For each benchmark b:
      1. On all OTHER benchmarks, find τ* maximizing mean primary metric.
      2. Apply τ* on b → record (b, τ*, score).
    Report mean cross-validated score.
    """
    main = results["main_experiment"]
    sweep_taus = results["meta"]["sweep_taus"]
    benches = [b for b in main if "error" not in main[b]
               and main[b]["methods"].get("A3+void")
               and main[b]["methods"]["A3+void"].get("sweep")]

    if len(benches) < 2:
        return {"error": "insufficient benchmarks for LOBO-CV"}

    lobo_results = []
    for held_out in benches:
        # Average score across other benches at each τ
        tau_scores = {tau: [] for tau in sweep_taus}
        for b in benches:
            if b == held_out:
                continue
            bd = main[b]
            primary = bd["primary_metric"]
            sweep = bd["methods"]["A3+void"]["sweep"]
            for s in sweep:
                tau_scores[s["tau"]].append(
                    s["avg_em"] if primary == "em" else s["avg_f1"]
                )
        avg_scores = {tau: np.mean(v) if v else 0.0 for tau, v in tau_scores.items()}
        # Pick best τ on training (other) benches
        best_tau = max(avg_scores, key=avg_scores.get)
        # Apply on held-out
        ho_bd = main[held_out]
        primary = ho_bd["primary_metric"]
        ho_sweep = ho_bd["methods"]["A3+void"]["sweep"]
        ho_score = next(
            (s["avg_em"] if primary == "em" else s["avg_f1"]
             for s in ho_sweep if abs(s["tau"] - best_tau) < 1e-6),
            None,
        )
        lobo_results.append({
            "held_out": held_out,
            "best_tau_on_others": best_tau,
            "score_on_held_out": ho_score,
            "training_avg_at_best": avg_scores[best_tau],
        })

    cv_score = np.mean([r["score_on_held_out"] for r in lobo_results
                        if r["score_on_held_out"] is not None])
    # Most-frequent best τ as recommended single-value
    from collections import Counter
    tau_counter = Counter(r["best_tau_on_others"] for r in lobo_results)
    recommended_tau = tau_counter.most_common(1)[0][0]
    return {
        "lobo_per_bench": lobo_results,
        "cv_score": float(cv_score),
        "recommended_tau": recommended_tau,
        "tau_vote_distribution": dict(tau_counter),
    }


def main():
    if not RESULTS_PATH.exists():
        print(f"❌ {RESULTS_PATH} not found. Run scripts/run_paper_v5_void.py first.")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    # Table 1
    tbl1 = build_table1(results)
    table1_path = FIGURES_DIR / "table1_v5_void.md"
    table1_path.write_text(tbl1)
    print(f"✓ Table 1 → {table1_path}")
    print()
    print(tbl1)
    print()

    # τ sweep figure
    plot_tau_sweep(results, FIGURES_DIR / "figure_void_sweep.png")

    # s_max distribution
    smax_tbl = build_smax_table(results)
    smax_path = FIGURES_DIR / "table_smax_distribution.md"
    smax_path.write_text(smax_tbl)
    print(f"✓ s_max table → {smax_path}")
    print()
    print(smax_tbl)
    print()

    # LOBO-CV calibration
    lobo = calibrate_tau_lobo(results)
    print("=" * 60)
    print("LOBO Cross-Validation")
    print("=" * 60)
    print(json.dumps(lobo, indent=2, default=str))

    # Aggregate output
    agg = {
        "table1_md": tbl1,
        "smax_table_md": smax_tbl,
        "lobo_calibration": lobo,
        "primary_tau_paper": results["meta"]["primary_tau"],
    }
    out = EXPERIMENTS_DIR / "paper_v5_aggregated.json"
    with open(out, "w") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ Aggregated → {out}")


if __name__ == "__main__":
    main()
