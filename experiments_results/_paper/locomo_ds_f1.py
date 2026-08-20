"""token-F1 for the LoCoMo/DeepSeek cell, using the same f1() the native
protocol uses, so all three LoCoMo rows report one metric instead of two."""
import collections, json, os, re, sys


def f1(pred, gold):                      # verbatim from locomo_native.f1
    def toks(s): return re.findall(r"[a-z0-9]+", s.lower())
    p, g = toks(pred), toks(gold)
    if not p or not g: return 0.0
    common = collections.Counter(p) & collections.Counter(g)
    ov = sum(common.values())
    if ov == 0: return 0.0
    prec, rec = ov / len(p), ov / len(g)
    return 2 * prec * rec / (prec + rec)


GROUPS = {"no_mem": "Raw", "raw_patch": "Patched", "amem": "A-Mem",
          "mem0": "Mem0", "curated_patch": "CuratorMem"}
path = sys.argv[1]
rows = []
for line in open(path, errors="replace"):
    line = line.strip()
    if line:
        try: rows.append(json.loads(line))
        except ValueError: pass
its = [r.get("iteration") for r in rows if r.get("iteration") is not None]
last = max(its)
print("%-11s %8s %8s %8s %5s" % ("arm", "judge", "em", "token-F1", "n"))
for g, name in GROUPS.items():
    per = collections.defaultdict(lambda: {"j": [], "e": [], "f": []})
    for r in rows:
        if r.get("iteration") != last or r.get("error") or str(r.get("group")) != g:
            continue
        t = r.get("task_id")
        if r.get("score") is not None: per[t]["j"].append(float(r["score"]))
        if r.get("em") is not None: per[t]["e"].append(float(r["em"]))
        gold = r.get("expected")
        gold = gold if isinstance(gold, str) else (gold[0] if isinstance(gold, list) and gold else "")
        resp = r.get("response") or ""
        if gold:
            per[t]["f"].append(f1(str(resp), str(gold)))
    if not per: continue
    agg = lambda k: [sum(v[k]) / len(v[k]) for v in per.values() if v[k]]
    j, e, fs = agg("j"), agg("e"), agg("f")
    print("%-11s %8.2f %8.2f %8s %5d" % (
        name, 100 * sum(j) / len(j) if j else -1, 100 * sum(e) / len(e) if e else -1,
        ("%.2f" % (100 * sum(fs) / len(fs))) if fs else "n/a", len(j)))
