#!/usr/bin/env python3
"""memlayer — the curated patch-memory engine behind this repo, packaged as a
standalone memory-layer SDK (Mem0-style surface, different guarantees).

    from memlayer import MemoryLayer

    mem = MemoryLayer(llm=my_llm_fn)          # llm optional: without it the
                                              # layer is pure append-only
    mem.record("tried X, worked because Y", chain_id="deploy-42",
               task="upgrade the catalog API", score=0.9)
    block = mem.inject("how do I upgrade the catalog API?",
                       chain_id="deploy-42")  # rendered, dose-budgeted block
    mem.manifest()                            # partitions / lineage / stats
    mem.time_travel("deploy-42", as_of=1)     # context as of iteration 1
    mem.save("store.pkl"); MemoryLayer.load("store.pkl")

What it guarantees that a compacting memory layer does not:
  * append-only writes — record() and additive curation only; no delete,
    merge, or rewrite, so any earlier regime stays recoverable;
  * a manifest over the store — chains are partitions, version history is
    lineage, effectiveness weights are per-entry statistics; reads are
    partition-pruned (O(chain), flat as the store grows);
  * dose-budgeted rendering — the injected block is capped (default 900
    chars, C_INJECT_BUDGET_CH) and degrades to raw entries rather than to
    silence when curated channels come up empty.

This is a THIN facade over the experiment code (scripts/latest/
evomem_bridge.CuratedMemory): same read/write paths the paper's arms run,
no forked logic. Env knobs (C_INJECT_BUDGET_CH, C_CRITIC_GATE,
C_RAW_FALLBACK, C_PAGE_KEEP, ...) apply unchanged.
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = ["MemoryLayer"]
__version__ = "0.1.0"


def _task(chain_id: str, task: str, task_id: Optional[str]) -> dict:
    return {"task_id": task_id or chain_id, "description": task or "",
            "metadata": {"chain_id": chain_id}}


class MemoryLayer:
    """Curated, manifested, append-only memory with a small imperative API."""

    def __init__(self, llm: Optional[Callable] = None,
                 critic: Optional[Callable] = None,
                 domain: str = "default", top_k: int = 3,
                 use_critic: bool = True, use_enrich: bool = True) -> None:
        from scripts.latest.evomem_bridge import CuratedMemory
        self._mem = CuratedMemory(domain, top_k=top_k,
                                  use_critic=use_critic and critic is not None
                                  or use_critic and llm is not None,
                                  use_enrich=use_enrich)
        # LLM backends are injectable: `llm` refines, `critic` reviews
        # (defaults to `llm`). With neither, curation degrades to a pure
        # append-only patch layer — recording and retrieval still work.
        if llm is not None or critic is not None:
            self._mem._llm = critic or llm
        elif getattr(self._mem, "_llm", None) is None:
            self._mem._llm = None
        self.domain = domain

    # ── write path ──────────────────────────────────────────────────────
    def record(self, content: str, *, chain_id: str, task: str = "",
               task_id: Optional[str] = None, score: Optional[float] = None,
               metadata: Optional[dict] = None) -> None:
        """Append one attempt/observation to the chain (never overwrites).
        `score` is YOUR assessment in [0,1] (self-, env- or gold-derived —
        the layer does not care, but records it honestly)."""
        t = _task(chain_id, task, task_id)
        if metadata:
            t["metadata"].update(metadata)
        coro = self._mem.record(t, {"response": content}, score)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return
        loop.create_task(coro)          # fire-and-forget inside a live loop

    async def arecord(self, content: str, *, chain_id: str, task: str = "",
                      task_id: Optional[str] = None,
                      score: Optional[float] = None,
                      metadata: Optional[dict] = None) -> None:
        t = _task(chain_id, task, task_id)
        if metadata:
            t["metadata"].update(metadata)
        await self._mem.record(t, {"response": content}, score)

    # ── read path ───────────────────────────────────────────────────────
    def inject(self, query: str, *, chain_id: str,
               task_id: Optional[str] = None) -> str:
        """The rendered, dose-budgeted context block for this chain (empty
        string when the chain has no history — never invented content)."""
        return self._mem.inject(_task(chain_id, query, task_id))

    def search(self, query: str, *, chain_id: str, k: int = 3) -> list[dict]:
        """Raw entries (metadata view), newest first — no rendering."""
        ents = list(self._mem._chain_entries.get(chain_id, []))[-k:][::-1]
        out = []
        for e in ents:
            tax = getattr(e, "failure_taxonomy", None) or {}
            out.append({"task_id": e.task_id,
                        "version": getattr(e, "version", None),
                        "score": getattr(e, "score", None),
                        "critic_quality": tax.get("critic_quality"),
                        "lesson": tax.get("causal_lesson", ""),
                        "outcome": (tax.get("verbatim_outcome") or "")[:200]})
        return out

    # ── manifest layer ──────────────────────────────────────────────────
    def manifest(self, chain_id: Optional[str] = None) -> dict:
        """Partitions, lineage depth, and per-entry statistics — metadata
        over the same append-only history, Iceberg-style."""
        chains = self._mem._chain_entries
        if chain_id is not None:
            ents = chains.get(chain_id, [])
            return {"chain": chain_id, "entries": len(ents),
                    "versions": [getattr(e, "version", None) for e in ents]}
        return {"chains": len(chains),
                "entries": sum(len(v) for v in chains.values()),
                "per_chain": {c: len(v) for c, v in chains.items()}}

    def time_travel(self, chain_id: str, as_of: int) -> str:
        """Rendered context exactly as an agent would have seen it after the
        first `as_of` entries of the chain — immutable history makes this a
        query, not a reconstruction effort."""
        from scripts.latest.evomem_bridge import _format_curated
        ents = self._mem._chain_entries.get(chain_id, [])[:as_of]
        return _format_curated(ents, [], current_tid=chain_id)

    # ── persistence ─────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._mem, f)

    @classmethod
    def load(cls, path: str | Path, llm: Optional[Callable] = None) -> "MemoryLayer":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj._mem = pickle.load(f)
        if llm is not None:
            obj._mem._llm = llm
        obj.domain = getattr(obj._mem, "benchmark", "default")
        return obj

    def __len__(self) -> int:
        return sum(len(v) for v in self._mem._chain_entries.values())
