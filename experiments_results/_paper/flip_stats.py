"""Paired flip counts with an exact interval -- the estimator that matches the data.

On a binary-scored benchmark most tasks are unchanged between two arms, so the
median shift is zero by construction and a Hodges-Lehmann interval is pinned to
[0,0] however lopsided the changes are. What carries the signal is the
DISCORDANT pairs: tasks the curated store rescued against tasks it spoiled.
That is McNemar's setting, and the exact binomial on the discordant pairs gives
both the test and an interval that cannot disagree with it.
"""
import collections, json, math, os

PAPER = "experiments_results/by_table/tab1/"
GRP = {"raw": "no_mem", "patched": "raw_patch", "amem": "amem",
       "mem0": "mem0", "curatormem": "curated_patch"}
ARMS = {"raw": "Raw", "patched": "Patched", "amem": "A-Mem", "mem0": "Mem0"}
THRESH = 0.5


def scores(path, group):
    if not os.path.exists(path):
        return {}
    rows = []
    for l in open(path, errors="replace"):
        l = l.strip()
        if l:
            try: rows.append(json.loads(l))
            except ValueError: pass
    per = collections.defaultdict(dict)
    for r in rows:
        if not r.get("error") and r.get("score") is not None:
            per[r["iteration"]][r["task_id"]] = float(r["score"])
    if not per:
        return {}
    return {0: per.get(0, {}), "last": per[max(per)]}


def binom_cdf(k, n, p=0.5):
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k + 1))


def exact_p(a, b):
    """Two-sided exact McNemar on discordant pairs."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * binom_cdf(k, n))


def prop_ci(a, n, level=0.95):
    """Clopper-Pearson interval for a/n."""
    if n == 0:
        return 0.0, 1.0
    alpha = 1 - level
    lo, hi = 0.0, 1.0
    if a > 0:
        l, h = 0.0, 1.0
        for _ in range(200):
            m = (l + h) / 2
            if 1 - binom_cdf(a - 1, n, m) < alpha / 2: l = m
            else: h = m
        lo = (l + h) / 2
    if a < n:
        l, h = 0.0, 1.0
        for _ in range(200):
            m = (l + h) / 2
            if binom_cdf(a, n, m) > alpha / 2: l = m
            else: h = m
        hi = (l + h) / 2
    return lo, hi


rows_out = []
for bench in ("gaia", "gaia2", "locomo", "tau2"):
    bdir = os.path.join(PAPER, bench)
    if not os.path.isdir(bdir):
        continue
    bbs = sorted(d for d in os.listdir(bdir) if os.path.isdir(os.path.join(bdir, d)))
    for arm, label in ARMS.items():
        resc = spoil = tot = 0
        for bb in bbs:
            C = scores(os.path.join(bdir, bb, "curatormem.jsonl"), GRP["curatormem"])
            X = scores(os.path.join(bdir, bb, arm + ".jsonl"), GRP[arm])
            if not C or not X:
                continue
            sh = sorted(set(C["last"]) & set(X["last"]))
            if len(sh) < 20:
                continue
            tot += len(sh)
            for t in sh:
                c, x = C["last"][t] > THRESH, X["last"][t] > THRESH
                if c and not x: resc += 1
                elif x and not c: spoil += 1
        if tot < 40:
            continue
        n = resc + spoil
        p = exact_p(resc, spoil)
        lo, hi = prop_ci(resc, n) if n else (0, 1)
        net = 100 * (resc - spoil) / tot
        rows_out.append((bench, label, tot, resc, spoil, net,
                         resc / n if n else 0.0, lo, hi, p))
        flag = "" if (lo > 0.5) or (hi < 0.5) else "  spans 50%"
        print("%-6s C-%-8s tasks=%3d  rescued=%3d spoiled=%3d  net=%+5.1fpp  "
              "share=%.2f [%.2f,%.2f]  p=%.4f%s"
              % (bench, label, tot, resc, spoil, net, resc / n if n else 0, lo, hi, p, flag))

if os.environ.get("FLIP_JSON"):
    json.dump([{"bench": b, "arm": a, "tasks": t, "rescued": r, "spoiled": s_,
                "net": net, "share": sh, "lo": lo, "hi": hi, "p": p}
               for b, a, t, r, s_, net, sh, lo, hi, p in rows_out],
              open(os.environ["FLIP_JSON"], "w"), indent=1)
    print("wrote", os.environ["FLIP_JSON"])
