"""Distil each cited trace down to the columns a table cell is computed from.

A full trace carries the augmented prompt and every response, which is why
hy3g2fix's is 143 MB and cannot be pushed. None of that is needed to check a
number: a cell is a mean of per-task scores. Keeping the distilled copy in the
repo means the paper stays verifiable from a clone, with the full traces left on
the box for anyone re-deriving something else.
"""
import json, os, sys

ROOT = "/data/workspace/MemorySkillGenerator/experiments_results"
OUT = os.path.join(ROOT, "_paper", "cells")
os.makedirs(OUT, exist_ok=True)

SRC = [
    ("hy3fix/hy3/gaia",      "gaia_hy3.jsonl"),
    ("hy3guard3/hy3/gaia",   "gaia_hy3_curated.jsonl"),
    ("gpt55full/gpt-5.5/gaia", "gaia_gpt55.jsonl"),
    ("hy3fix/hy3/gaia2",     "gaia2_hy3.jsonl"),
    ("hy3g2fix/hy3/gaia2",   "gaia2_hy3_curated.jsonl"),
    ("hy3tau2/hy3/tau2",     "tau2_hy3.jsonl"),
    ("hy3dose500/hy3/gaia",  "dose500_hy3.jsonl"),
    ("hy3dose0/hy3/gaia",    "dose0_hy3.jsonl"),
]
KEEP = ("task_id", "group", "iteration", "score", "em", "error")

total = 0
for rel, name in SRC:
    p = os.path.join(ROOT, rel, "trace.jsonl")
    if not os.path.exists(p):
        print("  absent: %s" % rel); continue
    n = 0
    with open(os.path.join(OUT, name), "w") as w:
        for line in open(p, errors="replace"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            w.write(json.dumps({k: d[k] for k in KEEP if k in d},
                               separators=(",", ":")) + "\n")
            n += 1
    sz = os.path.getsize(os.path.join(OUT, name))
    total += sz
    print("  %-24s %6d rows  %6.1f KB   <- %s" % (name, n, sz / 1024.0, rel))
print("total %.1f KB" % (total / 1024.0))
