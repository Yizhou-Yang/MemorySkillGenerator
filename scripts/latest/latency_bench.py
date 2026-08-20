#!/usr/bin/env python3
"""End-to-end read-path latency, per memory layer (paper: tab:latency).

The manifest microbenchmark (tab:manifest) times the store's own lookup with no
model in the loop. That answers "does the append-only design scale" but not
"what does an agent wait for", which is the question a deployment asks and the
one reviewers asked us. This measures the whole inject() an agent actually
blocks on -- query cleaning, embedding, retrieval, ranking, rendering -- for
CuratorMem, the raw patch store, A-Mem and Mem0 on the SAME queries against the
stores those arms wrote during the paper's runs.

No LLM calls: inject() is a read. A-Mem and Mem0 embed locally
(all-MiniLM-L6-v2), which is part of their read path and stays in the number.

Stores are loaded from the pickles the tau2 runs left behind, so every arm is
timed against the store IT built, at the size it really reached. A freshly
constructed (empty) store would make every layer look equally fast and measure
nothing.

  python scripts/latest/latency_bench.py \
      --store-dir experiments_results/_archive/2026-07-31/latest_evolving/hy3 \
      --benchmark tau2 --n 200

Writes JSON to experiments_results/_paper/latency/<model>_<benchmark>.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _pct(xs: list[float], q: float) -> float:
    """Nearest-rank percentile; xs must be sorted."""
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, int(round(q / 100.0 * len(xs) + 0.5)) - 1))
    return xs[k]


def _summary(ms: list[float]) -> dict:
    s = sorted(ms)
    return {"n": len(s),
            "mean": round(statistics.fmean(s), 3) if s else None,
            "p50": round(_pct(s, 50), 3), "p95": round(_pct(s, 95), 3),
            "p99": round(_pct(s, 99), 3),
            "min": round(s[0], 3) if s else None,
            "max": round(s[-1], 3) if s else None}


_PICKLE = {"curatormem": "tau2_mem_C.pkl", "patched": "tau2_mem_B.pkl",
           "amem": "tau2_mem_amem.pkl", "mem0": "tau2_mem_mem0.pkl"}


def _load_store(arm: str, store_dir: Path):
    """The arm's memory as the run left it. None when that arm has no pickle."""
    import pickle
    f = store_dir / _PICKLE[arm]
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def _store_size(mem) -> dict:
    """Entry and chain counts, so latency can be read against store size."""
    out = {}
    for attr in ("_chain_entries", "_per_chain", "_mems"):
        d = getattr(mem, attr, None)
        if isinstance(d, dict) and d:
            out["chains"] = len(d)
            try:
                out["entries"] = sum(len(v) for v in d.values())
            except TypeError:
                pass
            break
    inner = getattr(mem, "_mem", None)
    n = getattr(inner, "_n", None) or getattr(mem, "_n", None)
    if n is not None:
        out.setdefault("entries", int(n))
    return out


def _tasks_for(mem, limit: int) -> list[dict]:
    """Queries drawn from the arm's OWN store.

    Chain ids are per-run content hashes, so the arms' key spaces are disjoint:
    querying every arm with the curated store's ids made three arms miss every
    partition and report the cost of an empty lookup as if it were their read
    latency. Each arm is therefore queried with tasks it actually holds, which
    is the read it would really perform. Query text comes from the same
    benchmark either way, so the comparison stays like-for-like even though the
    id lists differ.
    """
    out, seen = [], set()

    def push(tid, desc, chain):
        if not tid or not desc or tid in seen:
            return
        seen.add(tid)
        out.append({"task_id": str(tid), "description": str(desc)[:4000],
                    "metadata": {"chain_id": chain}})

    ce = getattr(mem, "_chain_entries", None)          # CuratorMem
    if isinstance(ce, dict) and ce:
        for chain, entries in ce.items():
            for e in entries:
                push(getattr(e, "task_id", ""), getattr(e, "task_desc", ""), chain)
                if len(out) >= limit:
                    return out
        return out

    inner = getattr(mem, "_mem", None)                 # PatchedMemory
    if inner is not None and getattr(inner, "_patches", None):
        for p in inner._patches:
            push(getattr(p, "chain_id", ""), getattr(p, "evidence", "")
                 or getattr(p, "content", ""), getattr(p, "chain_id", ""))
            if len(out) >= limit:
                return out
        return out

    bench = getattr(mem, "benchmark", "")

    # A-Mem: _per_chain is keyed by the FULL namespace ("{bench}:{chain}"), but
    # inject() rebuilds that namespace from the task, so handing it the stored
    # key verbatim produces "{bench}:{bench}:{chain}" and matches nothing. Strip
    # the prefix, and take the query from the note itself so it is the real task
    # text rather than an id.
    pc = getattr(mem, "_per_chain", None)
    if isinstance(pc, dict) and pc:
        notes = getattr(mem, "_dehydrated_notes", None) or []
        by_id = {}
        for n in notes:
            if isinstance(n, dict) and n.get("id"):
                by_id[n["id"]] = str(n.get("content") or "")
        for ns, ids in pc.items():
            chain = ns.split(":", 1)[1] if ns.startswith(bench + ":") else ns
            text = ""
            for i in (ids or []):
                text = by_id.get(i, "")
                if text:
                    break
            # notes are stored as "Task: <description>\n..." -- recover the task
            if text.startswith("Task:"):
                text = text[len("Task:"):].strip()
            text = text.split("\n")[0].strip()
            push(chain, text or chain, chain)
            if len(out) >= limit:
                return out
        return out

    ms = getattr(mem, "_mems", None)                   # Mem0 (per-chain stores)
    if isinstance(ms, dict) and ms:
        for chain in ms:
            push(chain, chain, chain)
            if len(out) >= limit:
                return out
        return out

    # Mem0 keeps state on disk (one qdrant dir per chain) and drops its client
    # dict across a pickle, so the live chain list has to come from the store
    # root. Directory name is "mem0_" + slug(_ns) and _ns is "{bench}:{chain}",
    # so the chain id is what follows "mem0_{bench}_". The query text comes from
    # the memories mem0 itself stored, so the search is a real one.
    if type(mem).__name__ == "Mem0Memory":
        try:
            from scripts.latest.baseline_memories import _STORE_ROOT, _ns
        except Exception:
            return out
        pre = "mem0_%s_" % bench
        for d in sorted(q.name for q in Path(_STORE_ROOT).glob(pre + "*")):
            chain = d[len(pre):]
            text = ""
            try:
                ns = _ns(bench, {"task_id": chain, "metadata": {"chain_id": chain}})
                got = mem._mem_for(ns).get_all(filters={"user_id": ns}, limit=1)
                items = got.get("results", got) if isinstance(got, dict) else got
                for it in (items or []):
                    if isinstance(it, dict):
                        text = str(it.get("memory") or it.get("text") or "")
                        if text:
                            break
            except Exception:
                text = ""
            if not text:
                continue          # empty namespace: nothing to query it with
            push(chain, text, chain)
            if len(out) >= limit:
                return out
    return out


def _bench(name: str, mem, tasks: list[dict], warmup: int) -> dict:
    """Time inject() per task. Warmup absorbs lazy client/model construction,
    which is a startup cost, not a per-read one."""
    for t in tasks[:warmup]:
        try:
            mem.inject(t)
        except Exception:
            pass
    lat, served, errs = [], 0, 0
    for t in tasks:
        gc.collect()
        t0 = time.perf_counter()
        try:
            out = mem.inject(t)
        except Exception:
            errs += 1
            continue
        lat.append((time.perf_counter() - t0) * 1000.0)
        served += 1 if (out or "").strip() else 0
    r = _summary(lat)
    r.update({"arm": name, "served": served, "errors": errs,
              "served_rate": round(served / len(lat), 3) if lat else None})
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CODEBUDDY_MODEL", "hy3"))
    ap.add_argument("--benchmark", default="gaia2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--arms", default="curatormem,amem,mem0")
    ap.add_argument("--store-dir", default="",
                    help="directory holding tau2_mem_*.pkl from a real run")
    a = ap.parse_args()

    os.environ.setdefault("CODEBUDDY_MODEL", a.model)
    store_dir = Path(a.store_dir) if a.store_dir else None
    if store_dir and not store_dir.is_absolute():
        store_dir = PROJECT_ROOT / store_dir
    if not store_dir or not store_dir.is_dir():
        print("[latency] --store-dir is required and must hold tau2_mem_*.pkl; "
              "an empty store measures nothing", file=sys.stderr)
        return 1

    print(f"[latency] {a.benchmark} / {a.model}: stores from {store_dir}", flush=True)

    want = [x.strip() for x in a.arms.split(",") if x.strip()]
    rows = []
    for arm in want:
        if arm not in _PICKLE:
            print(f"[latency] unknown arm {arm}", file=sys.stderr)
            continue
        try:
            mem = _load_store(arm, store_dir)
        except BaseException as e:                # SystemExit from missing deps
            print(f"[latency] {arm}: store unreadable ({e})", flush=True)
            continue
        if mem is None:
            print(f"[latency] {arm}: no {_PICKLE[arm]} in {store_dir}", flush=True)
            continue
        tasks = _tasks_for(mem, a.n)
        if not tasks:
            # An arm whose store is empty cannot be timed; reporting its
            # empty-lookup cost as a read latency would be the wrong number.
            print(f"[latency] {arm}: store holds nothing queryable "
                  f"({_store_size(mem)}) -- skipped", flush=True)
            rows.append({"arm": arm, "skipped": "empty store",
                         "store": _store_size(mem)})
            continue
        r = _bench(arm, mem, tasks, a.warmup)
        r["store"] = _store_size(mem)
        rows.append(r)
        print(f"  {arm:12s} p50={r['p50']:>9.3f}ms  p95={r['p95']:>9.3f}ms  "
              f"served={r['served']}/{r['n']}  err={r['errors']}  "
              f"store={r['store']}", flush=True)
        if not r["served"]:
            print(f"  {'':12s} WARNING: served 0 blocks -- the numbers above time "
                  f"a lookup that found nothing, not a real read", flush=True)
        rel = getattr(mem, "release", None)
        if callable(rel):
            try:
                rel()
            except Exception:
                pass
        del mem
        gc.collect()

    out = PROJECT_ROOT / "experiments_results" / "_paper" / "latency"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.model}_{a.benchmark}.json"
    f.write_text(json.dumps(
        {"model": a.model, "benchmark": a.benchmark,
         "store_dir": str(store_dir),
         "note": "inject() wall time; no LLM in the loop; stores as written by "
                 "the paper's runs",
         "rows": rows}, indent=2))
    print(f"[latency] wrote {f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
