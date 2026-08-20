#!/usr/bin/env python3
"""Run-to-run variation of the memory-free arm, from the ten-iteration chain.

The main tables give one run per cell at temperature 0, which says nothing about
how much a number moves when the same thing is run again. Arm A never reads the
store, so its ten iterations are ten repeats of one condition -- differing only
in the protocol's own paraphrase of each task and in whatever the endpoint does
not reproduce. That makes them an upper bound on pure seed variance.
"""
import collections
import json
import statistics as st
import sys

PATH = (sys.argv[1] if len(sys.argv) > 1
        else "experiments_results/longchain/A/hy3/gaia/trace.jsonl")
GEN = "deepseek-v4-flash"


def main() -> int:
    rows = []
    for line in open(PATH, errors="replace"):
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except ValueError: pass
    rows = [r for r in rows
            if r.get("critic_model") == GEN and not r.get("error")
            and r.get("score") is not None]
    per = collections.defaultdict(dict)
    for r in rows:
        per[r["iteration"]][r["task_id"]] = float(r["score"])
    if len(per) < 3:
        print("need at least three repeats, found %d" % len(per), file=sys.stderr)
        return 1
    means = [100 * sum(v.values()) / len(v) for _, v in sorted(per.items())]
    shared = set.intersection(*[set(v) for v in per.values()])
    flips, sds = [], []
    for t in shared:
        vals = [per[k][t] for k in sorted(per)]
        flips.append(sum(1 for a, b in zip(vals, vals[1:]) if (a > .5) != (b > .5)))
        sds.append(100 * st.pstdev(vals))
    stable = sum(1 for f in flips if f == 0)
    print("repeats            %d" % len(per))
    print("arm mean           %.2f  sd %.2f  range %.2f"
          % (st.mean(means), st.stdev(means), max(means) - min(means)))
    print("tasks in all       %d" % len(shared))
    print("never flipped      %d (%.0f%%)" % (stable, 100 * stable / len(shared)))
    print("median per-task sd %.2f" % st.median(sds))
    print("per-iteration      %s" % " ".join("%.1f" % m for m in means))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
