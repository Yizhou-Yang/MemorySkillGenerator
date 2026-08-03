#!/usr/bin/env python3
"""Recompute every number the paper prints, from the file it came from.

Run it from the repo root:  python3 experiments_results/_paper/verify.py

A cell nobody can trace back to a file is a cell nobody can defend in review, so
each entry below names its source exactly: file, arm, iteration. FAIL means the
number moved; MISSING means the file is gone and that cell is now unbacked.

Cell means: mean over the tasks that arm completed, replicates averaged first
(tau2 runs each task twice). Delta in the paper is the plain difference of two
printed cells, so it needs nothing beyond this.
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cell(rel, group, iteration, field):
    """One table cell, or None if its source file is gone."""
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    per = collections.defaultdict(list)
    for line in open(p, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("error") or d.get("group") != group or d.get("iteration") != iteration:
            continue
        v = d.get(field)
        if v is not None:
            per[d.get("task_id")].append(float(v))
    if not per:
        return None
    vals = [sum(v) / len(v) for v in per.values()]
    return round(100 * sum(vals) / len(vals), 2), len(vals)


# (paper location, printed value, file, arm, iteration, field)
# Cells are read from the distilled copies in _paper/cells: a full trace also
# carries every prompt and response (hy3g2fix's is 143 MB, past GitHub's limit),
# none of which a mean of per-task scores needs. Regenerate them with mkcells.py
# after a rerun. The full traces stay on the box.
T = "experiments_results/_paper/cells/"
CELLS = [
    # -- Table 1, main results. Final iteration is 2 throughout.
    ("tab:main GAIA/HY3/em      Raw",        19.79, T+"gaia_hy3.jsonl",       "no_mem",        2, "em"),
    ("tab:main GAIA/HY3/em      Patched",    19.19, T+"gaia_hy3.jsonl",       "raw_patch",     2, "em"),
    ("tab:main GAIA/HY3/em      A-Mem",      24.49, T+"gaia_hy3.jsonl",       "amem",          2, "em"),
    ("tab:main GAIA/HY3/em      Mem0",       20.00, T+"gaia_hy3.jsonl",       "mem0",          2, "em"),
    ("tab:main GAIA/HY3/em      CuratorMem", 26.80, T+"gaia_hy3_curated.jsonl",    "curated_patch", 2, "em"),
    ("tab:main GAIA/HY3/judge   Raw",        25.83, T+"gaia_hy3.jsonl",       "no_mem",        2, "score"),
    ("tab:main GAIA/HY3/judge   Patched",    24.24, T+"gaia_hy3.jsonl",       "raw_patch",     2, "score"),
    ("tab:main GAIA/HY3/judge   A-Mem",      27.45, T+"gaia_hy3.jsonl",       "amem",          2, "score"),
    ("tab:main GAIA/HY3/judge   Mem0",       22.00, T+"gaia_hy3.jsonl",       "mem0",          2, "score"),
    ("tab:main GAIA/HY3/judge   CuratorMem", 30.93, T+"gaia_hy3_curated.jsonl",    "curated_patch", 2, "score"),

    ("tab:main GAIA/G55/em      Raw",        25.51, T+"gaia_gpt55.jsonl", "no_mem",        2, "em"),
    ("tab:main GAIA/G55/em      Patched",    23.00, T+"gaia_gpt55.jsonl", "raw_patch",     2, "em"),
    ("tab:main GAIA/G55/em      A-Mem",      23.00, T+"gaia_gpt55.jsonl", "amem",          2, "em"),
    ("tab:main GAIA/G55/em      Mem0",       27.00, T+"gaia_gpt55.jsonl", "mem0",          2, "em"),
    ("tab:main GAIA/G55/em      CuratorMem", 32.32, T+"gaia_gpt55.jsonl", "curated_patch", 2, "em"),
    ("tab:main GAIA/G55/judge   Raw",        32.65, T+"gaia_gpt55.jsonl", "no_mem",        2, "score"),
    ("tab:main GAIA/G55/judge   Patched",    30.80, T+"gaia_gpt55.jsonl", "raw_patch",     2, "score"),
    ("tab:main GAIA/G55/judge   A-Mem",      28.99, T+"gaia_gpt55.jsonl", "amem",          2, "score"),
    ("tab:main GAIA/G55/judge   Mem0",       34.00, T+"gaia_gpt55.jsonl", "mem0",          2, "score"),
    ("tab:main GAIA/G55/judge   CuratorMem", 39.39, T+"gaia_gpt55.jsonl", "curated_patch", 2, "score"),

    ("tab:main GAIA2/HY3        Raw",        33.48, T+"gaia2_hy3.jsonl",       "no_mem",        2, "score"),
    ("tab:main GAIA2/HY3        Patched",    31.35, T+"gaia2_hy3.jsonl",       "raw_patch",     2, "score"),
    ("tab:main GAIA2/HY3        A-Mem",      29.01, T+"gaia2_hy3.jsonl",       "amem",          2, "score"),
    ("tab:main GAIA2/HY3        Mem0",       33.54, T+"gaia2_hy3.jsonl",       "mem0",          2, "score"),
    # The record-fix rerun: empty final messages used to drop the chain's own
    # success, which starved 9 of 11 eligible chains and held this cell at 34.32.
    ("tab:main GAIA2/HY3        CuratorMem", 42.00, T+"gaia2_hy3_curated.jsonl",     "curated_patch", 2, "score"),

    ("tab:main tau2/HY3         Raw",        68.75, T+"tau2_hy3.jsonl",       "no_mem",        2, "score"),
    ("tab:main tau2/HY3         Patched",    72.50, T+"tau2_hy3.jsonl",       "raw_patch",     2, "score"),
    ("tab:main tau2/HY3         A-Mem",      71.25, T+"tau2_hy3.jsonl",       "amem",          2, "score"),
    ("tab:main tau2/HY3         Mem0",       73.75, T+"tau2_hy3.jsonl",       "mem0",          2, "score"),
    ("tab:main tau2/HY3         CuratorMem", 78.75, T+"tau2_hy3.jsonl",       "curated_patch", 2, "score"),

    # -- Figure 2a, read-time budget. L=900 is the shared default, so it is the
    #    same run as the GAIA/HY3 judge CuratorMem cell.
    ("fig:dose L=500",   24.80, T+"dose500_hy3.jsonl",  "curated_patch", 2, "score"),
    ("fig:dose L=900",   30.93, T+"gaia_hy3_curated.jsonl",   "curated_patch", 2, "score"),
    ("fig:dose L=inf",   26.60, T+"dose0_hy3.jsonl",    "curated_patch", 2, "score"),
]

# -- Table 2 is not a trace: the native-protocol harness writes its own files.
#    Kept here so one command still checks the whole paper.
NATIVE = [
    ("tab:locomonative answerer=HY3     Raw/Patched/Mem0/CuratorMem",
     "experiments_results/locomo_native/hy3/SUMMARY.txt", "ALL                 10.1    33.2"),
    ("tab:locomonative answerer=HY3     A-Mem (the amem rerun; 8.0 in SUMMARY.txt is the buggy pass)",
     "experiments_results/locomo_native/hy3_amem/results.jsonl", None),
    ("tab:locomonative answerer=GPT-5.5 all five",
     "experiments_results/locomo_native/FINAL_COMPARISON.txt", "ALL                 15.7    46.3    57.1    61.2    64.0"),
]

ok = miss = bad = 0
for label, want, rel, group, it, field in CELLS:
    got = cell(rel, group, it, field)
    if got is None:
        print("MISSING  %-42s want %6.2f   <- %s [%s]" % (label, want, rel, group))
        miss += 1
    elif abs(got[0] - want) > 0.005:
        print("FAIL     %-42s want %6.2f  got %6.2f (n=%d)  <- %s [%s]"
              % (label, want, got[0], got[1], rel, group))
        bad += 1
    else:
        print("ok       %-42s %6.2f  n=%-4d %s [%s]" % (label, want, got[1], rel, group))
        ok += 1

print()
for label, rel, needle in NATIVE:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print("MISSING  %s  <- %s" % (label, rel))
        miss += 1
    elif needle and needle not in open(p, errors="replace").read():
        print("FAIL     %s  <- %s (row not found)" % (label, rel))
        bad += 1
    else:
        print("ok       %s  <- %s" % (label, rel))
        ok += 1

print("\n%d ok, %d failed, %d missing" % (ok, bad, miss))
sys.exit(1 if (bad or miss) else 0)
