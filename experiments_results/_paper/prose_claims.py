#!/usr/bin/env python3
"""Recompute the numbers that appear in appendix prose rather than in a table.

A figure quoted in a sentence has no table to keep it honest, so each one here is
regenerated from the traces and compared against what the paper says. These are
the claims added late: the silence split, the benchmark's own nondeterminism, and
the ten-repeat spread.
"""
import collections
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(ROOT, "experiments_results", "by_table", "tab1")
GEN = "deepseek-v4-flash"


def _rows(path, group=None):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try: r = json.loads(line)
        except ValueError: continue
        if r.get("error") or r.get("score") is None:
            continue
        if group and r.get("group") != group:
            continue
        out.append(r)
    return out


def by_iter(path, group):
    per = collections.defaultdict(dict)
    for r in _rows(path, group):
        per[r["iteration"]][r["task_id"]] = float(r["score"])
    return per


def silence_split():
    """GAIA/DeepSeek-v4: arms restricted to the chains CuratorMem spoke on."""
    src = os.path.join(ROOT, "experiments_results", "dscb", "deepseek-v4-pro",
                       "gaia", "trace.jsonl")
    if not os.path.exists(src):
        return None
    per = collections.defaultdict(dict)
    for r in _rows(src):
        per[(r.get("group"), r.get("iteration"))][r["task_id"]] = r
    last = max(i for _, i in per)
    C, P, A = per[("curated_patch", last)], per[("raw_patch", last)], per[("no_mem", last)]
    sh = sorted(set(C) & set(P) & set(A))
    spoke = [t for t in sh if (C[t].get("aug_len") or 0) > 0]
    silent = [t for t in sh if (C[t].get("aug_len") or 0) == 0]
    m = lambda ts, D: 100 * sum(float(D[t]["score"]) for t in ts) / len(ts)
    return {"spoke_n": len(spoke), "silent_n": len(silent),
            "spoke": {"C": m(spoke, C), "P": m(spoke, P), "A": m(spoke, A)},
            "silent": {"C": m(silent, C), "P": m(silent, P), "A": m(silent, A)}}


def noise_floor(bench, bb):
    """Raw vs Patched at iteration 0 -- the same system, so this is the benchmark."""
    P = by_iter(os.path.join(TAB, bench, bb, "patched.jsonl"), "raw_patch")
    A = by_iter(os.path.join(TAB, bench, bb, "raw.jsonl"), "no_mem")
    sh = sorted(set(P.get(0, {})) & set(A.get(0, {})))
    if len(sh) < 20:
        return None
    return st.pstdev([100 * (P[0][t] - A[0][t]) for t in sh])


def repeats():
    """Ten iterations of the memory-free arm: same condition, ten times."""
    p = os.path.join(ROOT, "experiments_results", "longchain", "A", "hy3",
                     "gaia", "trace.jsonl")
    if not os.path.exists(p):
        return None
    per = collections.defaultdict(dict)
    for r in _rows(p):
        if r.get("critic_model") == GEN:
            per[r["iteration"]][r["task_id"]] = float(r["score"])
    if len(per) < 3:
        return None
    means = [100 * sum(v.values()) / len(v) for _, v in sorted(per.items())]
    shared = set.intersection(*[set(v) for v in per.values()])
    stable = sum(1 for t in shared
                 if len({per[k][t] > .5 for k in per}) == 1)
    return {"n": len(per), "mean": st.mean(means), "sd": st.stdev(means),
            "range": max(means) - min(means), "tasks": len(shared), "stable": stable}


CLAIMS = []


def check(label, got, want, tol=0.05):
    ok = got is not None and abs(got - want) <= tol
    CLAIMS.append(ok)
    print("%-4s %-46s paper %-8s data %s"
          % ("ok" if ok else "FAIL", label, want,
             "%.2f" % got if got is not None else "(missing)"))


def main():
    s = silence_split()
    if s:
        check("silence split: chains spoken on", s["spoke_n"], 64)
        check("  spoke, CuratorMem", s["spoke"]["C"], 82.50)
        check("  spoke, Patched", s["spoke"]["P"], 78.12)
        check("  spoke, Raw", s["spoke"]["A"], 75.00)
        check("silence split: chains silent on", s["silent_n"], 36)
        check("  silent, CuratorMem", s["silent"]["C"], 30.56)
        check("  silent, Raw", s["silent"]["A"], 22.22)
        check("  silent, Patched", s["silent"]["P"], 16.67)
    check("noise floor GAIA2/HY3", noise_floor("gaia2", "hy3"), 10.6)
    check("noise floor tau2/HY3", noise_floor("tau2", "hy3"), 46.1)
    r = repeats()
    if r:
        check("ten repeats: range", r["range"], 11.99)
        check("ten repeats: mean", r["mean"], 25.72)
        check("ten repeats: sd", r["sd"], 3.90)
        check("ten repeats: tasks in all", r["tasks"], 37)
        check("ten repeats: never flipped", r["stable"], 26)
    bad = CLAIMS.count(False)
    print("\n%d ok, %d failed" % (CLAIMS.count(True), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
