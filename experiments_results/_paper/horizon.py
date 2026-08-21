"""Accuracy vs evidence depth on LoCoMo native: the long-horizon question,
answered from runs that already exist.

The native protocol ingests a conversation session by session (19-35 sessions)
and then answers ~200 questions from retrieval alone, so the store is a
long-horizon accumulation and every question carries an exact annotation of
where its evidence lies. Depth = how many sessions before the conversation's
end the (deepest) evidence sits. If a memory layer only handles recent context,
its accuracy falls with depth; holding flat at depth is what "long-running
accumulating memory" actually requires.
"""
import collections, json, os

NAT = "experiments_results/locomo_native/"
SOURCES = {
    "hy3": [(NAT + "hy3/results.jsonl", ("nomem", "raw", "mem0", "ours")),
            (NAT + "hy3_amem/results.jsonl", ("amem",))],
    "gpt-5.5": [(NAT + "staged/results.jsonl", ("nomem", "amem", "mem0")),
                (NAT + "raw_arm/results.jsonl", ("raw",)),
                (NAT + "ours_sdk/results.jsonl", ("ours",))],
}
ORDER = ["nomem", "raw", "amem", "mem0", "ours"]
LBL = {"nomem": "No memory", "raw": "Raw store", "amem": "A-Mem",
       "mem0": "Mem0", "ours": "CuratorMem"}


def evidence_map():
    from datasets import load_dataset
    ds = load_dataset("KhangPTT373/locomo_preprocess", split="test")
    out = {}
    for i in range(len(ds)):
        r = ds[i]
        parse = lambda x: json.loads(x) if isinstance(x, str) else x
        qs, evs = parse(r["questions"]), parse(r["evidences"])
        n_sess = len(parse(r["sessions"]))
        for q, ev in zip(qs, evs):
            sess = [e[0] for e in ev if isinstance(e, (list, tuple)) and e]
            if not sess:
                continue
            out[("locomo_%d" % i, q.strip())] = {
                "depth": n_sess - max(sess),          # sessions before the end
                "span": len(set(sess)) > 1,           # multi-session evidence
            }
    return out


def main():
    ev = evidence_map()
    for backbone, files in SOURCES.items():
        rows = []
        for path, systems in files:
            if not os.path.exists(path):
                continue
            for line in open(path, errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try: r = json.loads(line)
                except ValueError: continue
                if r.get("system") in systems:
                    rows.append(r)
        if not rows:
            continue
        matched = unmatched = 0
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        span = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            key = (r.get("conv"), (r.get("q") or "").strip())
            info = ev.get(key)
            if info is None:
                unmatched += 1
                continue
            matched += 1
            d = info["depth"]
            b = "recent (0-5)" if d <= 5 else ("mid (6-12)" if d <= 12 else "deep (13+)")
            per[r["system"]][b].append(int(r.get("J") or 0))
            span[r["system"]]["multi" if info["span"] else "single"].append(int(r.get("J") or 0))
        print("\n=== %s (matched %d, unmatched %d) ===" % (backbone, matched, unmatched))
        buckets = ["recent (0-5)", "mid (6-12)", "deep (13+)"]
        print("%-11s %s   %s" % ("system", "  ".join("%14s" % b for b in buckets),
                                 "single-hop  multi-session"))
        for s in ORDER:
            if s not in per:
                continue
            cells = ["J=%5.1f n=%3d" % (100 * sum(v) / len(v), len(v))
                     if (v := per[s].get(b)) else "%14s" % "-" for b in buckets]
            sp = ["%5.1f" % (100 * sum(v) / len(v)) if (v := span[s].get(k)) else "  -"
                  for k in ("single", "multi")]
            print("%-11s %s   %s        %s" % (LBL[s], "  ".join(cells), sp[0], sp[1]))


main()
