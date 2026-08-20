#!/usr/bin/env python3
"""Paired statistics for every Table-1 cell (paper: tab:paired).

The main table reports point estimates. This computes, per benchmark-backbone
cell and per comparison, the paired per-task delta at the final iteration, a
bootstrap CI over tasks, a Wilcoxon signed-rank p, and a Holm correction across
the whole family of comparisons.

Pairing is by task id. Where PROVENANCE records the two arms as coming from
different runs, the per-task delta at the final iteration confounds the memory
effect with whatever separated the two runs. Iteration 0 measures exactly that
confound: the store is empty and nothing is injected, so the arms are the same
system and any gap between them is run-level drift. We therefore report a
difference in differences,

    did(t) = [hi_final(t) - lo_final(t)] - [hi_iter0(t) - lo_iter0(t)],

which cancels the run-level offset task by task, and we test on that. The
final-iteration delta is kept alongside it because it is what the main table
prints. A cell whose iteration-0 gap is already large is flagged: its printed
lead is mostly drift, and did is the number to read.

  python experiments_results/_paper/paired_stats.py            # table + json
  python experiments_results/_paper/paired_stats.py --tex      # LaTeX body
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TAB = ROOT / "experiments_results" / "by_table" / "tab1"
PROV = ROOT / "experiments_results" / "by_table" / "PROVENANCE.tsv"
COMPARISONS = [("curatormem", "patched"), ("curatormem", "raw"),
               ("curatormem", "mem0"), ("curatormem", "amem")]
BOOT, SEED = 10000, 20260820


def _scores_by_iter(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """(final-iteration, iteration-0) task_id -> score; errored rows dropped."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    its = [r.get("iteration") for r in rows if r.get("iteration") is not None]
    last = max(its) if its else None
    fin, zero = {}, {}
    for r in rows:
        if r.get("error") or r.get("score") is None:
            continue
        tid = r.get("task_id")
        if not tid:
            continue
        it = r.get("iteration")
        if last is not None and it == last:
            fin[str(tid)] = float(r["score"])
        if it == 0:
            zero[str(tid)] = float(r["score"])
    return fin, zero


def _provenance() -> dict[tuple[str, str, str], str]:
    if not PROV.exists():
        return {}
    with open(PROV) as fh:
        return {(r["benchmark"], r["backbone"], r["arm"]): r["source_run"]
                for r in csv.DictReader(fh, delimiter="\t")}


def _wilcoxon_p(d: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank via normal approximation with tie
    correction. Zeros are dropped (Wilcoxon's own convention)."""
    nz = [x for x in d if x != 0.0]
    n = len(nz)
    if n < 6:                      # normal approximation is not trustworthy here
        return None
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w = sum(r for r, x in zip(ranks, nz) if x > 0)
    mu = n * (n + 1) / 4.0
    tie = {}
    for r in ranks:
        tie[r] = tie.get(r, 0) + 1
    corr = sum(t ** 3 - t for t in tie.values()) / 48.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - corr)
    if sd == 0:
        return None
    z = (w - mu) / sd
    return max(0.0, min(1.0, math.erfc(abs(z) / math.sqrt(2))))


def _boot_ci(d: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(d)
    means = []
    for _ in range(BOOT):
        means.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT)]


def _holm(items: list[dict]) -> None:
    """Holm-Bonferroni across every paired comparison in the family."""
    testable = [r for r in items if r.get("p") is not None]
    testable.sort(key=lambda r: r["p"])
    m = len(testable)
    prev = 0.0
    for i, r in enumerate(testable):
        adj = min(1.0, (m - i) * r["p"])
        adj = max(adj, prev)                     # Holm is monotone
        prev = adj
        r["p_holm"] = adj
        r["sig_holm"] = adj < 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", action="store_true")
    a = ap.parse_args()

    # Backbone order follows the main table (HY3, DeepSeek, GPT-5.5) rather than
    # the filesystem's alphabetical order, so a reader can line the two up.
    BB_ORDER = {"hy3": 0, "deepseek-v4": 1, "gpt-5.5": 2}
    prov, rng, out = _provenance(), random.Random(SEED), []
    for bench_dir in sorted(p for p in TAB.iterdir() if p.is_dir()):
        for bb_dir in sorted((p for p in bench_dir.iterdir() if p.is_dir()),
                             key=lambda d: (BB_ORDER.get(d.name, 99), d.name)):
            bench, bb = bench_dir.name, bb_dir.name
            arms = {f.stem: _scores_by_iter(f) for f in bb_dir.glob("*.jsonl")}
            for hi, lo in COMPARISONS:
                if hi not in arms or lo not in arms:
                    continue
                hi_fin, hi_zero = arms[hi]
                lo_fin, lo_zero = arms[lo]
                shared = sorted(set(hi_fin) & set(lo_fin))
                if len(shared) < 5:
                    continue
                d = [hi_fin[t] - lo_fin[t] for t in shared]
                mean = sum(d) / len(d)
                same_run = (prov.get((bench, bb, hi)) == prov.get((bench, bb, lo))
                            and prov.get((bench, bb, hi)) is not None)

                # Iteration 0 is the memory-free condition: any gap there is drift.
                z = sorted(set(hi_zero) & set(lo_zero) & set(shared))
                did = [(hi_fin[t] - lo_fin[t]) - (hi_zero[t] - lo_zero[t]) for t in z]
                z_gap = (sum(hi_zero[t] - lo_zero[t] for t in z) / len(z)) if z else None

                lo_ci, hi_ci = _boot_ci(d, rng)
                r = {"benchmark": bench, "backbone": bb, "cmp": f"{hi}-{lo}",
                     "n": len(shared), "delta": round(100 * mean, 2),
                     "ci_lo": round(100 * lo_ci, 2), "ci_hi": round(100 * hi_ci, 2),
                     "wins": sum(1 for x in d if x > 0),
                     "losses": sum(1 for x in d if x < 0),
                     "same_run": bool(same_run),
                     "src_hi": prov.get((bench, bb, hi), "?"),
                     "src_lo": prov.get((bench, bb, lo), "?"),
                     "n_did": len(did),
                     "iter0_gap": None if z_gap is None else round(100 * z_gap, 2)}
                if did:
                    dl, dh = _boot_ci(did, rng)
                    r.update({"did": round(100 * sum(did) / len(did), 2),
                              "did_ci_lo": round(100 * dl, 2),
                              "did_ci_hi": round(100 * dh, 2)})
                    # The drift-corrected quantity is what carries the claim, so
                    # it is what we test -- for same-run and cross-run alike.
                    r["p"] = _wilcoxon_p(did)
                else:
                    r.update({"did": None, "did_ci_lo": None, "did_ci_hi": None})
                    r["p"] = _wilcoxon_p(d) if same_run else None
                out.append(r)

    _holm(out)
    dest = ROOT / "experiments_results" / "_paper" / "paired_stats.json"
    dest.write_text(json.dumps(out, indent=2))

    if a.tex:
        for r in out:
            ph = r.get("p_holm")
            p = ("$<$0.001" if ph is not None and ph < 0.001
                 else f"{ph:.3f}" if ph is not None else "--")
            tag = "" if r["same_run"] else r"$^{\dagger}$"
            did = "--" if r["did"] is None else f"{r['did']:+.2f}"
            ci = ("--" if r["did"] is None
                  else f"[{r['did_ci_lo']:+.2f}, {r['did_ci_hi']:+.2f}]")
            print(f"{r['benchmark']} & {r['backbone']}{tag} & {r['cmp']} & {r['n']} & "
                  f"{r['delta']:+.2f} & {did} & {ci} & {p}\\\\")
    else:
        hdr = (f"{'bench':7s} {'backbone':12s} {'comparison':22s} {'n':>4s} "
               f"{'final':>7s} {'iter0':>7s} {'DiD':>7s} {'DiD 95% CI':>18s} "
               f"{'p(Holm)':>9s}  source")
        print(hdr); print("-" * len(hdr))
        for r in out:
            ph = r.get("p_holm")
            ps = "--" if ph is None else f"{ph:.4f}"
            star = "*" if r.get("sig_holm") else " "
            z = "--" if r["iter0_gap"] is None else f"{r['iter0_gap']:+7.2f}"
            did = "--" if r["did"] is None else f"{r['did']:+7.2f}"
            ci = ("--" if r["did"] is None
                  else f"[{r['did_ci_lo']:+6.2f},{r['did_ci_hi']:+6.2f}]")
            print(f"{r['benchmark']:7s} {r['backbone']:12s} {r['cmp']:22s} "
                  f"{r['n']:4d} {r['delta']:+7.2f} {z:>7s} {did:>7s} {ci:>18s} "
                  f"{ps:>9s}{star} {'same run' if r['same_run'] else 'rerun'}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
