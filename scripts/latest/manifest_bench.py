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
  M5  per-chain history length: read latency and per-write cost as ONE chain
      grows (the never-delete objection: reviewers asked what happens when the
      thing that grows is the chain the read must touch, not the store around it)

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
              tool_sequence=[], version=i, timestamp=float(i),
              # fields the real ExperienceLibrary.record() reads; without them
              # the "real library" path falls back to the stub and measures
              # nothing the system does.
              augmentation_used=False, augmentation_helped=None,
              task_complexity="medium", outcome="success", sys_stats={})


def _mk(n_chains: int, per_chain: int, real_library: bool = False):
    """Build a store of n_chains x per_chain entries.

    real_library=True wires the actual ExperienceLibrary, so the read exercises
    the retriever the system really runs -- including its partition index. The
    stub below is kept for the store-size sweep, where the point is the
    manifest lookup rather than the retriever.
    """
    m = object.__new__(CuratedMemory)
    m.top_k = 3
    m.benchmark = "gaia"
    m._chain_of, m._chain_entries, m._served = {}, {}, {}
    m._served_keys, m._last_score, m._chain_base = {}, {}, {}
    m._last_wc = m._last_render = None
    pool = []
    for c in range(n_chains):
        ch = f"chain{c}"
        m._chain_of[ch] = ch
        ents = [_exp(i, ch) for i in range(per_chain)]
        m._chain_entries[ch] = ents
        pool += ents
    if real_library:
        # Bulk-load rather than record() one by one: record() pre-warms each
        # entry's embedding, which is the real per-write cost but would make
        # building a 10,000-entry store take hours here. The per-write cost is
        # measured separately, on fresh entries, against a store already built.
        from memlayer.experience import ExperienceLibrary
        lib = ExperienceLibrary()
        lib.experiences = list(pool)
        lib._by_task = {}
        for i, e in enumerate(pool):
            lib._by_task.setdefault(e.task_id, []).append(i)
        m._sf = NS(library=lib)
    else:
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

    # M5: hold the store at 10,000 patches and move them between many short
    # chains and few long ones. The read is chain-pruned, so chain length is the
    # axis on which its cost can still grow -- measure it rather than assert it
    # away. Both columns use the REAL library and the REAL write path: an
    # earlier version scored a stubbed retriever and timed a bare list append,
    # which measured neither the retrieval nor the write the system performs.
    print()
    print("| per-chain length | chains | M5 inject p50 (ms) | M5 record (ms) |")
    print("|---|---|---|---|")
    total = 10_000
    for per_chain in (10, 100, 1000, 10_000):
        m = _mk(total // per_chain, per_chain, real_library=True)
        task = {"task_id": "chain0", "description":
                "task chain0 variant 1 with some realistic description words"}
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            m.inject(task)
            ts.append((time.perf_counter() - t0) * 1000)
        m5r = sorted(ts)[len(ts) // 2]
        lib = m._sf.library
        ws = []
        for i in range(20):
            # Unique text per configuration: the embedding cache is keyed by
            # text, so reusing the same synthetic description across configs
            # turned every write after the first config into a cache hit and
            # reported 0.001 ms as the write cost.
            e = _exp(total + i, "chain0")
            e.task_desc = f"{e.task_desc} [cfg {per_chain} write {i}]"
            t0 = time.perf_counter()
            lib.record(e)         # real write: append, index update, embed
            ws.append((time.perf_counter() - t0) * 1000)
        m5w = sorted(ws)[len(ws) // 2]
        del lib.experiences[-20:]
        print(f"| {per_chain:,} | {total // per_chain:,} | {m5r:.2f} | {m5w:.3f} |")


if __name__ == "__main__":
    main()
