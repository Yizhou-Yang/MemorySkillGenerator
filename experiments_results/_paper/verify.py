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
# Table-shaped layout: one directory per table row, one file per arm, so a
# row's five arms sit together and PROVENANCE.tsv says which run each came
# from. The old run-shaped paths hid that GAIA/DeepSeek takes four arms from
# one sweep and its curated arm from another.
PAPER = "experiments_results/by_table/tab1/"
CELLS = [
    # -- Table 1, main results. Final iteration is 2 throughout.
    ("tab:main GAIA/HY3/em      Raw", 19.79, PAPER+"gaia/hy3/raw.jsonl", "no_mem", 2, "em"),
    ("tab:main GAIA/HY3/em      Patched", 19.19, PAPER+"gaia/hy3/patched.jsonl", "raw_patch", 2, "em"),
    ("tab:main GAIA/HY3/em      A-Mem", 24.49, PAPER+"gaia/hy3/amem.jsonl", "amem", 2, "em"),
    ("tab:main GAIA/HY3/em      Mem0", 20.00, PAPER+"gaia/hy3/mem0.jsonl", "mem0", 2, "em"),
    ("tab:main GAIA/HY3/em      CuratorMem", 26.80, PAPER+"gaia/hy3/curatormem.jsonl", "curated_patch", 2, "em"),
    ("tab:main GAIA/HY3/judge   Raw", 25.83, PAPER+"gaia/hy3/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main GAIA/HY3/judge   Patched", 24.24, PAPER+"gaia/hy3/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main GAIA/HY3/judge   A-Mem", 27.45, PAPER+"gaia/hy3/amem.jsonl", "amem", 2, "score"),
    ("tab:main GAIA/HY3/judge   Mem0", 22.00, PAPER+"gaia/hy3/mem0.jsonl", "mem0", 2, "score"),
    ("tab:main GAIA/HY3/judge   CuratorMem", 30.93, PAPER+"gaia/hy3/curatormem.jsonl", "curated_patch", 2, "score"),

    ("tab:main GAIA/G55/em      Raw", 25.51, PAPER+"gaia/gpt-5.5/raw.jsonl", "no_mem", 2, "em"),
    ("tab:main GAIA/G55/em      Patched", 23.00, PAPER+"gaia/gpt-5.5/patched.jsonl", "raw_patch", 2, "em"),
    ("tab:main GAIA/G55/em      A-Mem", 23.00, PAPER+"gaia/gpt-5.5/amem.jsonl", "amem", 2, "em"),
    ("tab:main GAIA/G55/em      Mem0", 27.00, PAPER+"gaia/gpt-5.5/mem0.jsonl", "mem0", 2, "em"),
    ("tab:main GAIA/G55/em      CuratorMem", 32.32, PAPER+"gaia/gpt-5.5/curatormem.jsonl", "curated_patch", 2, "em"),
    ("tab:main GAIA/G55/judge   Raw", 32.65, PAPER+"gaia/gpt-5.5/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main GAIA/G55/judge   Patched", 30.80, PAPER+"gaia/gpt-5.5/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main GAIA/G55/judge   A-Mem", 28.99, PAPER+"gaia/gpt-5.5/amem.jsonl", "amem", 2, "score"),
    ("tab:main GAIA/G55/judge   Mem0", 34.00, PAPER+"gaia/gpt-5.5/mem0.jsonl", "mem0", 2, "score"),
    ("tab:main GAIA/G55/judge   CuratorMem", 39.39, PAPER+"gaia/gpt-5.5/curatormem.jsonl", "curated_patch", 2, "score"),

    ("tab:main GAIA2/HY3 Raw", 33.48, PAPER+"gaia2/hy3/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main GAIA2/HY3 Patched", 31.35, PAPER+"gaia2/hy3/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main GAIA2/HY3 A-Mem", 29.01, PAPER+"gaia2/hy3/amem.jsonl", "amem", 2, "score"),
    ("tab:main GAIA2/HY3 Mem0", 33.54, PAPER+"gaia2/hy3/mem0.jsonl", "mem0", 2, "score"),
    # The record-fix rerun: empty final messages used to drop the chain's own
    # success, which starved 9 of 11 eligible chains and held this cell at 34.32.
    ("tab:main GAIA2/HY3 CuratorMem", 42.00, PAPER+"gaia2/hy3/curatormem.jsonl", "curated_patch", 2, "score"),

    ("tab:main GAIA2/G55 Raw", 31.57, PAPER+"gaia2/gpt-5.5/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main GAIA2/G55 Patched", 41.01, PAPER+"gaia2/gpt-5.5/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main GAIA2/G55 A-Mem", 19.96, PAPER+"gaia2/gpt-5.5/amem.jsonl", "amem", 2, "score"),
    ("tab:main GAIA2/G55 Mem0", 37.18, PAPER+"gaia2/gpt-5.5/mem0.jsonl", "mem0", 2, "score"),
    # The coverage rerun: the endorsement key plus the measured dead-chain rule took
    # arm C from serving 19% of chains to 92%, and the cell from 33.73 to 41.99.
    ("tab:main GAIA2/G55 CuratorMem", 41.99, PAPER+"gaia2/gpt-5.5/curatormem.jsonl", "curated_patch", 2, "score"),

    # Rebuilt 2026-08-11 after the source directory was destroyed. The sweep runs
    # C/B/mem0/amem only, so the memory-free arm is not in it and the table shows
    # that cell as pending rather than carrying the old, now-unbacked 68.75.
    # Raw and CuratorMem come from a PAIRED rerun -- both arms in one sweep, so
    # the comparison between them carries no run-level drift. It was needed: the
    # memory-free arm measured 74.68 and then 67.50 under an identical
    # configuration, a 7.2-point swing, and the earlier reading had it winning
    # the row. Under the pairing it comes last. Patched, A-Mem and Mem0 are
    # still from the original sweep, so only the Raw/CuratorMem comparison is
    # paired; all five sit inside 5 points, which is this cell's noise floor.
    ("tab:main tau2/HY3 Raw", 67.50, PAPER+"tau2/hy3/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main tau2/HY3 Patched", 72.50, PAPER+"tau2/hy3/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main tau2/HY3 A-Mem", 71.25, PAPER+"tau2/hy3/amem.jsonl", "amem", 2, "score"),
    ("tab:main tau2/HY3 Mem0", 68.75, PAPER+"tau2/hy3/mem0.jsonl", "mem0", 2, "score"),
        ("tab:main tau2/HY3         CuratorMem", 77.50, PAPER+"tau2/hy3/curatormem.jsonl", "curated_patch", 2, "score"),

    # -- tau2 GPT-5.5 and DeepSeek. Both measured after the three tau2 repairs
    #    (injection reaches the prompt, record and inject agree on the chain key,
    #    a prior dialogue is never replayed verbatim), so they supersede anything
    #    from before 2026-08-06. The GPT-5.5 A-Mem cell is the RERUN: the first
    #    one exited cleanly while A-Mem never initialised, 1770 concurrent
    #    SentenceTransformer loads all landing on meta tensors.
    ("tab:main tau2/G55 Raw", 68.75, PAPER+"tau2/gpt-5.5/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main tau2/G55 Patched", 86.25, PAPER+"tau2/gpt-5.5/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main tau2/G55 A-Mem", 79.75, PAPER+"tau2/gpt-5.5/amem.jsonl", "amem", 2, "score"),
    ("tab:main tau2/G55 Mem0", 76.25, PAPER+"tau2/gpt-5.5/mem0.jsonl", "mem0", 2, "score"),
    # The printed curated cell is the rerun carrying C_SEMANTIC_FALLBACK. Without
    # it the arm injected on under 25 of 80 simulations -- it was mostly running
    # memory-free and scored 85.00 doing so.
    ("tab:main tau2/G55 CuratorMem", 86.25, PAPER+"tau2/gpt-5.5/curatormem.jsonl", "curated_patch", 2, "score"),

    ("tab:main tau2/DS Raw", 76.25, PAPER+"tau2/deepseek-v4/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main tau2/DS Patched", 77.50, PAPER+"tau2/deepseek-v4/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main tau2/DS A-Mem", 73.75, PAPER+"tau2/deepseek-v4/amem.jsonl", "amem", 2, "score"),
    ("tab:main tau2/DS Mem0", 82.50, PAPER+"tau2/deepseek-v4/mem0.jsonl", "mem0", 2, "score"),
    ("tab:main tau2/DS          CuratorMem", 86.25, PAPER+"tau2/deepseek-v4/curatormem.jsonl", "curated_patch", 2, "score"),

    # -- The DeepSeek rows. These were printed in the table for weeks without a
    #    single entry here, which is precisely the hole this file exists to close:
    #    a number nobody can trace is a number nobody can defend. The CuratorMem
    #    cells of GAIA and GAIA2 are deliberately absent -- the table shows those
    #    as being re-measured on the official endpoint, so there is no printed
    #    value to check yet. Add them when the reruns land.
    ("tab:main GAIA/DS/em      Raw", 53.00, PAPER+"gaia/deepseek-v4/raw.jsonl", "no_mem", 2, "em"),
    ("tab:main GAIA/DS/em      Patched", 51.00, PAPER+"gaia/deepseek-v4/patched.jsonl", "raw_patch", 2, "em"),
    ("tab:main GAIA/DS/em      A-Mem", 53.00, PAPER+"gaia/deepseek-v4/amem.jsonl", "amem", 2, "em"),
    ("tab:main GAIA/DS/em      Mem0", 61.00, PAPER+"gaia/deepseek-v4/mem0.jsonl", "mem0", 2, "em"),
    ("tab:main GAIA/DS/judge   Raw", 56.00, PAPER+"gaia/deepseek-v4/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main GAIA/DS/judge   Patched", 56.00, PAPER+"gaia/deepseek-v4/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main GAIA/DS/judge   A-Mem", 55.80, PAPER+"gaia/deepseek-v4/amem.jsonl", "amem", 2, "score"),
    ("tab:main GAIA/DS/judge   Mem0", 64.80, PAPER+"gaia/deepseek-v4/mem0.jsonl", "mem0", 2, "score"),

    # -- The DeepSeek curated cells, after the answer-first fix (a275a448). The
    #    spoilage mechanism was measured case by case: every spoiled block
    #    carried a checkmark but not the answer, because verbatim_outcome keeps
    #    resp[:400] (the head) while GAIA answers live at the tail. With the
    #    answer leading the block: rescued 23 / spoiled 7 against mem0's 19/8,
    #    and the row flips. GAIA2 stays pending -- flips improved (13/5) but
    #    partial-credit mass still trails mem0.
    ("tab:main GAIA/DS/em      CuratorMem", 63.00, PAPER+"gaia/deepseek-v4/curatormem.jsonl", "curated_patch", 2, "em"),
    ("tab:main GAIA/DS/judge   CuratorMem", 67.00, PAPER+"gaia/deepseek-v4/curatormem.jsonl", "curated_patch", 2, "score"),

    ("tab:main GAIA2/DS Raw", 29.91, PAPER+"gaia2/deepseek-v4/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main GAIA2/DS Patched", 36.72, PAPER+"gaia2/deepseek-v4/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main GAIA2/DS A-Mem", 33.65, PAPER+"gaia2/deepseek-v4/amem.jsonl", "amem", 2, "score"),
    ("tab:main GAIA2/DS Mem0", 37.09, PAPER+"gaia2/deepseek-v4/mem0.jsonl", "mem0", 2, "score"),
    ("tab:main GAIA2/DS        CuratorMem", 53.42, PAPER+"gaia2/deepseek-v4/curatormem.jsonl", "curated_patch", 2, "score"),

    ("tab:main LoCoMo/DS Raw", 22.40, PAPER+"locomo/deepseek-v4/raw.jsonl", "no_mem", 2, "score"),
    ("tab:main LoCoMo/DS Patched", 28.40, PAPER+"locomo/deepseek-v4/patched.jsonl", "raw_patch", 2, "score"),
    ("tab:main LoCoMo/DS A-Mem", 32.40, PAPER+"locomo/deepseek-v4/amem.jsonl", "amem", 2, "score"),
    ("tab:main LoCoMo/DS Mem0", 21.80, PAPER+"locomo/deepseek-v4/mem0.jsonl", "mem0", 2, "score"),
    ("tab:main LoCoMo/DS CuratorMem", 37.00, PAPER+"locomo/deepseek-v4/curatormem.jsonl", "curated_patch", 2, "score"),

    # -- Figure 2a, read-time budget. Re-run 2026-08-06 after the lineage-footer
    #    fix: `room = _C_INJECT_BUDGET + 100 - ...` read the uncapped setting (0)
    #    as a literal zero and dropped the version-lineage footer from the L=inf
    #    arm entirely, so the old bars ranked by footer presence rather than by
    #    budget. All three come from one revision; the earlier cells are retired.
    # The read-time budget figure is no longer in the paper -- no fig:dose label,
    # none of 28.47 / 29.09 / 32.99 appears in main.tex. This file checks numbers
    # the paper prints, so its three entries are retired. The snapshots stay in
    # cells/ as the only surviving record of that sweep; the run directories were
    # never committed and are gone.
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

# -- Table 1's Llama-3.2-3B rows and all of Appendix A. The weak-backbone harness
#    writes one row per (system, conversation, question); a cell is the plain mean
#    over its 497 rows, no per-task replicates to average.
#
#    Grader: gpt-5.6-sol graded ours/amem/nomem, then degraded to ~90% 503s and
#    was replaced by gpt-5.5 for mem0/raw. The three sol-graded arms were re-graded
#    from their stored predictions into results_judge55.jsonl so a row is not
#    compared across two graders; that file wins wherever it exists.
WEAK_DIR = "experiments_results/locomo_weak/"
WEAK = [
    # (model, system, J, F1)
    ("llama-3.2-1b", "nomem",  4.83,  5.23),
    ("llama-3.2-1b", "raw",   30.99, 20.15),
    ("llama-3.2-1b", "amem",  18.91, 11.80),
    ("llama-3.2-1b", "mem0",  40.44, 24.11),
    ("llama-3.2-1b", "ours",  38.23, 26.40),
    ("llama-3.2-3b", "nomem",  7.24,  8.88),
    ("llama-3.2-3b", "raw",   40.04, 33.90),
    ("llama-3.2-3b", "amem",  23.14, 14.86),
    ("llama-3.2-3b", "mem0",  48.89, 40.79),
    ("llama-3.2-3b", "ours",  47.89, 41.14),
    ("qwen2.5-1.5b", "nomem",  7.24, 10.74),
    ("qwen2.5-1.5b", "raw",   35.61, 32.02),
    ("qwen2.5-1.5b", "amem",  20.12, 21.20),
    ("qwen2.5-1.5b", "mem0",  46.08, 38.31),
    ("qwen2.5-1.5b", "ours",  40.85, 38.97),
]


def weak_rows(model):
    """Rows for one backbone, with re-graded J substituted where it exists."""
    d = os.path.join(ROOT, WEAK_DIR, model)
    base = os.path.join(d, "results.jsonl")
    if not os.path.exists(base):
        return None
    regraded = {}
    rj = os.path.join(d, "results_judge55.jsonl")
    if os.path.exists(rj):
        for line in open(rj, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            regraded[(r["system"], r["conv"], r["q"])] = r["J"]
    out = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for line in open(base, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        k = (r["system"], r["conv"], r["q"])
        a = out[r["system"]]
        a[0] += regraded.get(k, r["J"])
        a[1] += r["F1"]
        a[2] += 1
    return out


# -- Table 1's LoCoMo rows. The native-protocol harness writes one row per
#    (system, conversation, question) and no trace, and it splits one backbone's
#    arms across directories, so each cell names its own file. Recomputed here
#    rather than grepped out of SUMMARY.txt, which prints to one decimal.
NAT = "experiments_results/locomo_native/"
NATIVE_CELLS = [
    ("tab:main LoCoMo/HY3       Raw",        10.06,  9.46, NAT+"hy3/results.jsonl",      "nomem"),
    ("tab:main LoCoMo/HY3       Patched",    33.20, 14.50, NAT+"hy3/results.jsonl",      "raw"),
    # the A-Mem rerun; hy3/results.jsonl still holds the pass that hit the recall bug
    ("tab:main LoCoMo/HY3       A-Mem",      30.38, 13.11, NAT+"hy3_amem/results.jsonl", "amem"),
    ("tab:main LoCoMo/HY3       Mem0",       49.90, 25.80, NAT+"hy3/results.jsonl",      "mem0"),
    ("tab:main LoCoMo/HY3       CuratorMem", 52.52, 28.04, NAT+"hy3/results.jsonl",      "ours"),
    ("tab:main LoCoMo/G55       Raw",        15.69, 16.95, NAT+"staged/results.jsonl",   "nomem"),
    ("tab:main LoCoMo/G55       Patched",    46.28, 41.21, NAT+"raw_arm/results.jsonl",  "raw"),
    ("tab:main LoCoMo/G55       A-Mem",      57.14, 50.58, NAT+"staged/results.jsonl",   "amem"),
    ("tab:main LoCoMo/G55       Mem0",       61.17, 51.22, NAT+"staged/results.jsonl",   "mem0"),
    ("tab:main LoCoMo/G55       CuratorMem", 63.98, 53.45, NAT+"ours_sdk/results.jsonl", "ours"),
]

print()
for label, want_j, want_f1, rel, system in NATIVE_CELLS:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print("MISSING  %s  <- %s" % (label, rel)); miss += 1; continue
    j = f1 = 0.0; n = 0
    for line in open(path, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("system") != system:
            continue
        j += r["J"]; f1 += r["F1"]; n += 1
    if not n:
        print("MISSING  %s  <- %s [%s absent]" % (label, rel, system)); miss += 1; continue
    got_j, got_f1 = round(100 * j / n, 2), round(100 * f1 / n, 2)
    if abs(got_j - want_j) > 0.005 or abs(got_f1 - want_f1) > 0.005:
        print("FAIL     %s  want J=%.2f F1=%.2f  got J=%.2f F1=%.2f (n=%d)"
              % (label, want_j, want_f1, got_j, got_f1, n)); bad += 1
    else:
        print("ok       %s  J=%-6.2f F1=%-6.2f n=%d" % (label, got_j, got_f1, n)); ok += 1


print()
_cache = {}
for model, system, want_j, want_f1 in WEAK:
    if model not in _cache:
        _cache[model] = weak_rows(model)
    agg = _cache[model]
    label = "tab:weak %-13s %-6s" % (model, system)
    if agg is None or system not in agg:
        print("MISSING  %s  <- %s%s" % (label, WEAK_DIR, model))
        miss += 1
        continue
    j, f1, n = agg[system]
    got_j, got_f1 = round(100 * j / n, 2), round(100 * f1 / n, 2)
    if abs(got_j - want_j) > 0.005 or abs(got_f1 - want_f1) > 0.005:
        print("FAIL     %s  want J=%.2f F1=%.2f  got J=%.2f F1=%.2f (n=%d)"
              % (label, want_j, want_f1, got_j, got_f1, n))
        bad += 1
    else:
        print("ok       %s  J=%-6.2f F1=%-6.2f n=%d" % (label, got_j, got_f1, n))
        ok += 1



# ── tab:gaia2strict: benchmark-native pass@1 beside the paper's soft recall ──
# Recomputed as the share of tasks scoring a full 1.0, which is what the
# official all-or-nothing rule awards. Same files, same iteration as tab:main.
def strict_rate(rel, group, iteration=2):
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
        v = d.get("score")
        if v is not None:
            per[d.get("task_id")].append(float(v))
    if not per:
        return None
    vals = [sum(v) / len(v) for v in per.values()]
    return round(100 * sum(1 for v in vals if v >= 1.0 - 1e-9) / len(vals), 2), len(vals)


STRICT = [
    ("HY3", "hy3", [("Raw", "raw", "no_mem", 11.00), ("Patched", "patched", "raw_patch", 5.05),
                    ("A-Mem", "amem", "amem", 5.00), ("Mem0", "mem0", "mem0", 9.00),
                    ("CuratorMem", "curatormem", "curated_patch", 9.00)]),
    ("GPT-5.5", "gpt-5.5", [("Raw", "raw", "no_mem", 13.13), ("Patched", "patched", "raw_patch", 18.18),
                            ("A-Mem", "amem", "amem", 9.00), ("Mem0", "mem0", "mem0", 18.00),
                            ("CuratorMem", "curatormem", "curated_patch", 12.00)]),
    ("DeepSeek-v4", "deepseek-v4", [("Raw", "raw", "no_mem", 9.00), ("Patched", "patched", "raw_patch", 16.00),
                                    ("A-Mem", "amem", "amem", 13.00), ("Mem0", "mem0", "mem0", 17.00),
                                    ("CuratorMem", "curatormem", "curated_patch", 16.00)]),
]
print()
for disp, slug, arms in STRICT:
    for name, fname, group, want in arms:
        label = "tab:gaia2strict %-12s %-11s" % (disp, name)
        got = strict_rate(PAPER + "gaia2/%s/%s.jsonl" % (slug, fname), group)
        if got is None:
            print("MISSING  %s" % label); miss += 1
            continue
        val, n = got
        if abs(val - want) > 0.005:
            print("FAIL     %s  want %.2f got %.2f (n=%d)" % (label, want, val, n)); bad += 1
        else:
            print("ok       %s  %.2f (n=%d)" % (label, val, n)); ok += 1

print("\n%d ok, %d failed, %d missing" % (ok, bad, miss))
sys.exit(1 if (bad or miss) else 0)
