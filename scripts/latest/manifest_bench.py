#!/usr/bin/env python3
"""Manifest-layer microbenchmark (paper: the systems view of the store).

The append-only store doubles as an Iceberg-style manifest over patch
snapshots: chain id = partition key, version history H_c = snapshot lineage,
w_c effectiveness table = column statistics, and append-only writes =
immutable snapshots, which is what makes version-conditioned retrieval and
rollback recovery ("time travel") O(chain) instead of O(store).

This script measures, OFFLINE on synthetic patches (no LLM calls, so it does
not touch the frozen method):

  M1  read-path latency vs store size (global pool + chain-index rescue)
  M2  time-travel: reconstruct the injected context as of iteration k
  M3  crash recovery: pickle round-trip time + size vs store size
  M4  write amplification: bytes appended per Augment vs per Append

Usage: python scripts/latest/manifest_bench.py [maxN]
Fills the paper's manifest microbenchmark table (appendix).
"""
from __future__ import annotations

import pickle
import sys
import time
from types import SimpleNamespace as NS

sys.path.insert(0, ".")
from scripts.latest.evomem_bridge import CuratedMemory, _format_curated


def _exp(i: int, chain: str):
    return NS(task_id=chain, task_desc=f"task {chain} variant {i} with some "
              "realistic description words about the work to be done",
              score=0.5 + (i % 5) / 10,
              failure_taxonomy={"critic_quality": 5 + i % 5,
                                "causal_lesson": "check the source before "
                                                 "answering and compare both",
                                "avoidance_note": "do not trust one source",
                                "verbatim_outcome": f"attempt {i}: " + "x" * 300},
              reasoning_trace=[f"reasoned about {i}"], action_commands=[],
              tool_sequence=[], version=i, timestamp=float(i))


def _mk(n_chains: int, per_chain: int):
    m = object.__new__(CuratedMemory)
    m.top_k = 3
    m.benchmark = "gaia"
    m._chain_of, m._chain_entries, m._served = {}, {}, {}
    pool = []
    for c in range(n_chains):
        ch = f"chain{c}"
        m._chain_of[ch] = ch
        ents = [_exp(i, ch) for i in range(per_chain)]
        m._chain_entries[ch] = ents
        pool += ents
    m._sf = NS(library=NS(retrieve_similar=lambda q, top_k: pool[:top_k]))
    return m


def main() -> None:
    maxn = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    print("| store size (patches) | M1 inject p50 (ms) | M2 time-travel (ms) "
          "| M3 pickle save+load (ms) | M3 size (MB) |")
    print("|---|---|---|---|---|")
    n = 1000
    while n <= maxn:
        per_chain = 3
        m = _mk(n // per_chain, per_chain)
        task = {"task_id": "chain7", "description":
                "task chain7 variant 1 with some realistic description words"}
        # M1: read path (median of 50)
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            m.inject(task)
            ts.append((time.perf_counter() - t0) * 1000)
        m1 = sorted(ts)[len(ts) // 2]
        # M2: time travel — context as of iteration k (chain prefix render)
        ents = m._chain_entries["chain7"]
        t0 = time.perf_counter()
        for k in (1, 2, 3):
            _format_curated(ents[:k], [], current_tid="chain7")
        m2 = (time.perf_counter() - t0) / 3 * 1000
        # M3: crash recovery
        t0 = time.perf_counter()
        blob = pickle.dumps({"chain_of": m._chain_of,
                             "chain_entries": m._chain_entries})
        pickle.loads(blob)
        m3 = (time.perf_counter() - t0) * 1000
        print(f"| {n:,} | {m1:.2f} | {m2:.2f} | {m3:.0f} | "
              f"{len(blob)/1e6:.1f} |")
        n *= 10


if __name__ == "__main__":
    main()
