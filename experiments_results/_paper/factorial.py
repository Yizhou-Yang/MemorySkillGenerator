#!/usr/bin/env python3
"""Decompose the CuratorMem-minus-Patched gap into the components that differ.

Patched and CuratorMem differ in more than curation: the relevance floor, the
ranking key, and the rendering all change together, so a raw C-vs-B gap cannot
be attributed to any one of them. Each foil below restores exactly one of those
to Patched's setting and leaves the rest of CuratorMem intact, so the drop from
C_full to a foil is that component's contribution.

  C_theta05    overlap floor 0.05 (Patched's) instead of 0.08
  C_ranksim    similarity order instead of the lineage tuple
  C_rawrender  gate-selected entries served verbatim, as Patched renders them

Effects are drift-corrected: iteration 0 is the empty-store condition, so a gap
there is the run rather than the memory, and it is subtracted per task.
"""
import collections
import json
import os
import random
import sys

ROOT = os.environ.get("FACT_ROOT", "experiments_results/fact2")
BENCH = os.environ.get("FACT_BENCH", "gaia2")
BACKBONE = os.environ.get("FACT_BACKBONE", "hy3")
FOILS = [("C_theta05", "relevance floor"),
         ("C_ranksim", "lineage ordering"),
         ("C_rawrender", "curated rendering")]


def load(arm):
    p = os.path.join(ROOT, arm, BACKBONE, BENCH, "trace.jsonl")
    if not os.path.exists(p):
        return None
    rows = []
    for line in open(p, errors="replace"):
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except ValueError: pass
    rows = [r for r in rows if not r.get("error") and r.get("score") is not None]
    if not rows:
        return None
    per = collections.defaultdict(dict)
    for r in rows:
        per[r["iteration"]][r["task_id"]] = float(r["score"])
    last = max(per)
    return {"first": per.get(0, {}), "last": per[last], "k": last,
            "n": len(per[last])}


def boot(v, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(v)
    s = sorted(sum(rng.choice(v) for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def did(a, b):
    """b minus a, per task, with each run's iteration-0 offset removed."""
    sh = sorted(set(a["last"]) & set(b["last"]) & set(a["first"]) & set(b["first"]))
    if len(sh) < 20:
        return None
    d = [100 * ((b["last"][t] - a["last"][t]) - (b["first"][t] - a["first"][t]))
         for t in sh]
    lo, hi = boot(d)
    return {"n": len(d), "mean": sum(d) / len(d), "lo": lo, "hi": hi,
            "rescued": sum(1 for t in sh if b["last"][t] > .5 >= a["last"][t]),
            "spoiled": sum(1 for t in sh if a["last"][t] > .5 >= b["last"][t])}


def line(label, r):
    if r is None:
        print("  %-34s (incomplete)" % label); return
    star = " *" if (r["lo"] > 0) == (r["hi"] > 0) else ""
    print("  %-34s n=%3d  %+6.2f  [%+6.2f,%+6.2f]  +%d/-%d%s"
          % (label, r["n"], r["mean"], r["lo"], r["hi"],
             r["rescued"], r["spoiled"], star))


def main() -> int:
    arms = {a: load(a) for a in ["B", "C_full"] + [f for f, _ in FOILS]}
    if not arms.get("B") or not arms.get("C_full"):
        print("need B and C_full; have %s"
              % ", ".join(a for a, v in arms.items() if v), file=sys.stderr)
        return 1
    print("%s / %s, drift-corrected, * = interval excludes zero\n" % (BENCH, BACKBONE))
    line("C_full - Patched (whole gap)", did(arms["B"], arms["C_full"]))
    print()
    for foil, what in FOILS:
        if arms.get(foil):
            line("C_full - %s  [%s]" % (foil, what), did(arms[foil], arms["C_full"]))
        else:
            print("  %-34s (not run yet)" % ("C_full - " + foil))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
