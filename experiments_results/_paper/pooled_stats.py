"""Pool the paired comparison across backbones, stratified.

Each task is paired within its own backbone -- the same task, same run, two
arms -- so pooling adds independent strata rather than mixing conditions. It is
the same estimand as the per-cell rows, measured on three times the tasks.

DiD removes each run's iteration-0 offset before pooling, so a backbone whose
run sat at a different water level does not tilt the pooled mean.
"""
import json, os, random, statistics as st, sys

PAPER = "experiments_results/by_table/tab1/"
ARMS = {"raw": "Raw", "patched": "Patched", "amem": "A-Mem", "mem0": "Mem0"}
GRP = {"raw": "no_mem", "patched": "raw_patch", "amem": "amem",
       "mem0": "mem0", "curatormem": "curated_patch"}


def scores(path, group):
    if not os.path.exists(path):
        return {}
    out = {}
    rows = []
    for l in open(path, errors="replace"):
        l = l.strip()
        if l:
            try: rows.append(json.loads(l))
            except ValueError: pass
    its = [r.get("iteration") for r in rows if r.get("iteration") is not None]
    if not its:
        return {}
    last = max(its)
    for it in (0, last):
        d = {}
        for r in rows:
            if (r.get("iteration") == it and r.get("group") == group
                    and not r.get("error") and r.get("score") is not None):
                d[r["task_id"]] = float(r["score"])
        out[it] = d
    out["last"] = last
    return out


def boot(v, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(v)
    s = sorted(sum(rng.choice(v) for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def wilcoxon(d):
    d = [x for x in d if x != 0]
    n = len(d)
    if n < 6: return 1.0
    ranks = {}
    for i, (a, _) in enumerate(sorted(enumerate(d), key=lambda t: abs(t[1])), 1):
        ranks[a] = i
    wp = sum(ranks[i] for i, x in enumerate(d) if x > 0)
    mu, sd = n * (n + 1) / 4, (n * (n + 1) * (2 * n + 1) / 24) ** .5
    if sd == 0: return 1.0
    z = abs(wp - mu) / sd
    return max(min(2 * (1 - 0.5 * (1 + __import__("math").erf(z / 2 ** .5))), 1.0), 0.0)


for bench in ("gaia", "gaia2", "locomo", "tau2"):
    bdir = os.path.join(PAPER, bench)
    if not os.path.isdir(bdir):
        continue
    backbones = sorted(d for d in os.listdir(bdir) if os.path.isdir(os.path.join(bdir, d)))
    print("\n=== %s (backbones: %s) ===" % (bench, ", ".join(backbones)))
    for arm, label in ARMS.items():
        pooled, strata = [], []
        for bb in backbones:
            C = scores(os.path.join(bdir, bb, "curatormem.jsonl"), GRP["curatormem"])
            X = scores(os.path.join(bdir, bb, arm + ".jsonl"), GRP[arm])
            if not C or not X:
                continue
            lc, lx = C["last"], X["last"]
            sh = set(C[lc]) & set(X[lx]) & set(C[0]) & set(X[0])
            if len(sh) < 20:
                continue
            d = [100 * ((C[lc][t] - X[lx][t]) - (C[0][t] - X[0][t])) for t in sorted(sh)]
            pooled += d
            strata.append((bb, len(d), sum(d) / len(d)))
        if len(strata) < 2:
            continue
        lo, hi = boot(pooled)
        p = wilcoxon(pooled)
        print("  C - %-8s pooled n=%3d  DiD %+6.2f  CI[%+6.2f,%+6.2f]  p=%.4f"
              % (label, len(pooled), sum(pooled) / len(pooled), lo, hi, p))
        for bb, n, m in strata:
            print("        %-14s n=%3d  %+6.2f" % (bb, n, m))
