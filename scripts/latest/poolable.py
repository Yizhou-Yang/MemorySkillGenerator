#!/usr/bin/env python3
"""Decide whether rows from different runs may be pooled.

Runs are poolable when nothing that could move a score differs between them --
being separate runs is not itself a reason to keep them apart, and treating it
as one would make RESUME useless. What actually has to match:

  protocol_hash   the knobs the runner already hashes (arms, budgets, policy...)
  hot path        code that can change a score, diffed between the two code_revs
  served model    the model that answered, not the directory it was filed under
  task set        the paired unit; pooling is over the intersection

Usage:
  python scripts/latest/poolable.py <trace.jsonl> <trace.jsonl> [more...]
  python scripts/latest/poolable.py --dir experiments_results/latest_evolving/hy3/gaia
"""
from __future__ import annotations
import argparse, collections, json, os, subprocess, sys

# Code whose change can move a score. Analysis-only modules are excluded on
# purpose: rerunning breakdown.py does not invalidate a trace.
HOT = ("scripts/latest/latest_runner.py", "scripts/latest/eval.py",
       "scripts/latest/gaia_runner.py", "scripts/latest/gaia2_runner.py",
       "scripts/latest/locomo_runner.py", "scripts/latest/tau2_bridge.py",
       "scripts/latest/tau2_agent.py", "scripts/latest/baseline_memories.py",
       "scripts/latest/llm_client.py", "benchmarks/loader.py",
       "memlayer/", "src/latest/")


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def facets(rows):
    return {
        "protocol_hash": collections.Counter(str(r.get("protocol_hash")) for r in rows),
        "code_rev": collections.Counter(str(r.get("code_rev"))[:8] for r in rows),
        "judge": collections.Counter(str(r.get("judge_model")) for r in rows),
        "critic": collections.Counter(str(r.get("critic_model")) for r in rows),
        "arms": collections.Counter(str(r.get("group")) for r in rows),
    }


def hot_diff(rev_a, rev_b):
    """Files on the hot path that differ between two revisions."""
    if rev_a == rev_b:
        return []
    try:
        out = subprocess.run(["git", "diff", "--name-only", rev_a, rev_b, "--"] + list(HOT),
                             capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return [f"(diff failed: {exc})"]
    if out.returncode != 0:
        return [f"(diff failed: {out.stderr.strip()[:80]})"]
    return [f for f in out.stdout.split("\n") if f.strip()]


def report(paths):
    groups = {p: load(p) for p in paths}
    groups = {p: r for p, r in groups.items() if r}
    if not groups:
        print("no rows found")
        return 1

    print(f"{'source':52} {'n':>6} {'revs':>6} {'proto':>10}")
    for p, rows in groups.items():
        f = facets(rows)
        print(f"{p[-52:]:52} {len(rows):6d} {len(f['code_rev']):6d} "
              f"{list(f['protocol_hash'])[0][:10] if len(f['protocol_hash'])==1 else 'MIXED':>10}")

    verdicts = []
    # every revision seen anywhere, pairwise against the most common one
    all_revs = collections.Counter()
    for rows in groups.values():
        all_revs.update(str(r.get("code_rev"))[:8] for r in rows)
    base = all_revs.most_common(1)[0][0]
    print(f"\nbaseline revision: {base}")
    for rev, n in all_revs.most_common():
        if rev == base:
            continue
        d = hot_diff(base, rev)
        if d:
            verdicts.append(f"code: {rev} differs from {base} on {len(d)} hot file(s): "
                            + ", ".join(d[:4]) + (" ..." if len(d) > 4 else ""))
        else:
            print(f"  {rev} ({n} rows): hot path identical to {base} -> poolable")

    protos = set()
    judges = set()
    critics = set()
    for rows in groups.values():
        f = facets(rows)
        protos |= set(f["protocol_hash"])
        judges |= set(f["judge"])
        critics |= set(f["critic"])
    if len(protos) > 1:
        verdicts.append(f"protocol_hash differs: {sorted(p[:10] for p in protos)}")
    if len(judges) > 1:
        verdicts.append(f"judge differs: {sorted(judges)}")
    if len(critics) > 1:
        verdicts.append(f"critic differs: {sorted(critics)}")

    # paired unit: which tasks are common to every arm
    per_arm = collections.defaultdict(set)
    for rows in groups.values():
        for r in rows:
            per_arm[r.get("group")].add(r.get("task_id"))
    if len(per_arm) > 1:
        common = set.intersection(*per_arm.values())
        union = set.union(*per_arm.values())
        if common != union:
            verdicts.append(f"task sets differ: {len(union)-len(common)} of {len(union)} "
                            f"tasks missing from some arm; pair over the {len(common)} common ones")

    print()
    if verdicts:
        print("NOT POOLABLE:")
        for v in verdicts:
            print("  -", v)
        return 1
    print("POOLABLE: nothing that can move a score differs.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--dir", help="check one benchmark directory's own trace")
    a = ap.parse_args()
    paths = list(a.paths)
    if a.dir:
        paths.append(os.path.join(a.dir, "trace.jsonl"))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        ap.error("no existing trace files given")
    sys.exit(report(paths))


if __name__ == "__main__":
    main()
