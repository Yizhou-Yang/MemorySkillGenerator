#!/usr/bin/env python3
"""External baseline memory systems as drop-in arms (Wen 07/06).

Each adapter wraps a published memory system's OWN module behind the same
inject()/record() interface as BenchmarkMemory/CuratedMemory, so the
protocol around it (mutated chains, self-assessed feedback, record after
evaluation, chain scoping via a per-chain namespace) is IDENTICAL to the
B/C arms — what differs is only the memory system itself. Run via:

    EXTERNAL_MEMS=mem0,amem ARMS=A RESULTS_BASE=latest_evolving ... latest_runner.py

(ARMS=A keeps no_mem pairing rows fresh if absent; the raw_patch arm is NOT
rerun for these comparisons per the advisor's design.)

Install notes (server):
    mem0:  pip install mem0ai         — configured to talk to the local
           CodeBuddy OAI proxy (OPENAI_API_BASE / localhost:8741) for both
           LLM and embeddings; vector store defaults to an on-disk qdrant
           under experiments_results/_extmem/.
    amem:  pip install -e <A-Mem repo> (WujiangXu/AgenticMemory) or set
           AMEM_PATH=/path/to/repo — uses its AgenticMemorySystem class.

Adapters fail FAST with an instructive message if the module is missing:
a silently-degraded baseline would be worse than no baseline.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_SLUG = re.sub(r"[^A-Za-z0-9._-]", "_",
                     os.environ.get("CODEBUDDY_MODEL", "hy3")).lower()
# per-model roots: two backbones on one box must not share vector stores
_STORE_ROOT = PROJECT_ROOT / "experiments_results" / "_extmem" / _MODEL_SLUG


def _chain_id(task: dict) -> str:
    meta = task.get("metadata") or {}
    return (task.get("chain_id") or meta.get("chain_id")
            or meta.get("chain") or task.get("task_id", ""))


def _ns(benchmark: str, task: dict) -> str:
    """Chain-scoped namespace: same discipline as B/C (memory is same-task /
    same-session feedback, not cross-task transfer)."""
    return f"{benchmark}:{_chain_id(task)}"


class Mem0Memory:
    """Mem0 (mem0.ai OSS) as a baseline arm, using ITS OWN add/search."""

    def __init__(self, benchmark: str, top_k: int = 3) -> None:
        self.benchmark = benchmark
        self.top_k = top_k
        try:
            from mem0 import Memory
        except ImportError as e:
            raise SystemExit(
                "[baseline:mem0] pip install mem0ai (and ensure the OAI proxy "
                "on OPENAI_API_BASE serves /chat/completions and /embeddings): "
                f"{e}")
        base = os.environ.get("OPENAI_API_BASE", "http://localhost:8741/v1")
        key = os.environ.get("OPENAI_API_KEY", "dummy")
        model = os.environ.get("CODEBUDDY_MODEL", "hy3").lower()
        cfg = {
            "llm": {"provider": "openai",
                    "config": {"model": model, "api_key": key,
                               "openai_base_url": base}},
            # LOCAL embedder by default: the chat proxy does not serve
            # /embeddings, and this matches the embedding stack the B/C arms
            # use, so no arm gets a better encoder.
            "embedder": {"provider": "huggingface",
                         "config": {"model": os.environ.get(
                             "MEM0_EMBED_MODEL",
                             "sentence-transformers/all-MiniLM-L6-v2")}},
            "vector_store": {"provider": "qdrant",
                             "config": {"path": str(_STORE_ROOT / "mem0"),
                                        "on_disk": True}},
        }
        override = os.environ.get("MEM0_CONFIG_JSON")
        if override:
            import json
            cfg = json.loads(override)
        self._m = Memory.from_config(cfg)
        try:
            import mem0 as _m0
            print(f"[baseline:mem0] version={getattr(_m0, '__version__', '?')} "
                  f"store={_STORE_ROOT / 'mem0'}", flush=True)
        except Exception:
            pass
        self._add_failures = 0

    def inject(self, task: dict) -> str:
        query = task.get("description", "")[:2000]
        try:
            hits = self._m.search(query, user_id=_ns(self.benchmark, task),
                                  limit=self.top_k)
        except Exception as e:
            print(f"[baseline:mem0] search failed: {e}", flush=True)
            return ""
        items = hits.get("results", hits) if isinstance(hits, dict) else hits
        lines = [f"- {h.get('memory') or h.get('text') or ''}".strip()
                 for h in (items or []) if isinstance(h, dict)]
        lines = [l for l in lines if len(l) > 2][: self.top_k]
        if not lines:
            return ""
        return "## Memories from earlier attempts (mem0)\n\n" + "\n".join(lines)

    async def record(self, task: dict, result: dict, score=None) -> None:
        resp = (result.get("response") or "").strip()
        if not resp:
            return
        try:
            # mem0.add runs its own LLM extraction — blocking; keep it off
            # the event loop or it throttles every concurrent task.
            await asyncio.to_thread(
                self._m.add,
                [{"role": "user", "content": task.get("description", "")[:4000]},
                 {"role": "assistant", "content": resp[:4000]}],
                user_id=_ns(self.benchmark, task))
        except Exception as e:
            self._add_failures += 1
            if self._add_failures <= 3 or self._add_failures % 50 == 0:
                print(f"[baseline:mem0] add failed ({self._add_failures}): {e}",
                      flush=True)

    def __len__(self) -> int:  # parity with other arms' logging
        return 0


class AMemMemory:
    """A-Mem (Agentic Memory, Xu et al. 2025) as a baseline arm, using its
    AgenticMemorySystem (note construction + link generation + evolution)."""

    def __init__(self, benchmark: str, top_k: int = 3) -> None:
        self.benchmark = benchmark
        self.top_k = top_k
        amem_path = os.environ.get("AMEM_PATH")
        if amem_path:
            import sys
            sys.path.insert(0, amem_path)
        try:
            from agentic_memory.memory_system import AgenticMemorySystem
        except ImportError:
            try:
                from memory_system import AgenticMemorySystem  # repo root layout
            except ImportError as e:
                raise SystemExit(
                    "[baseline:amem] A-Mem module not importable — pip install "
                    "-e the WujiangXu/AgenticMemory repo or set AMEM_PATH: "
                    f"{e}")
        model = os.environ.get("CODEBUDDY_MODEL", "hy3").lower()
        # A-Mem's ctor signature varies across revisions; try the documented
        # form first, degrade to defaults rather than guessing kwargs.
        try:
            self._sys = AgenticMemorySystem(
                model_name=os.environ.get("AMEM_EMBED_MODEL", "all-MiniLM-L6-v2"),
                llm_backend="openai", llm_model=model)
        except TypeError:
            self._sys = AgenticMemorySystem()
        self._per_chain: dict[str, list[str]] = {}

    def inject(self, task: dict) -> str:
        query = task.get("description", "")[:2000]
        ns = _ns(self.benchmark, task)
        try:
            hits = self._sys.search_agentic(query, k=self.top_k * 4) \
                if hasattr(self._sys, "search_agentic") \
                else self._sys.search(query, k=self.top_k * 4)
        except Exception as e:
            print(f"[baseline:amem] search failed: {e}", flush=True)
            return ""
        lines = []
        chain_ids = set(self._per_chain.get(ns, []))
        for h in hits or []:
            hid = h.get("id") if isinstance(h, dict) else None
            text = (h.get("content") or h.get("context") or "") \
                if isinstance(h, dict) else str(h)
            # chain scoping: A-Mem's store is global; keep the protocol equal
            # to B/C by serving only this chain's notes.
            if chain_ids and hid is not None and hid not in chain_ids:
                continue
            text = text.strip()
            if text:
                lines.append(f"- {text[:400]}")
            if len(lines) >= self.top_k:
                break
        if not lines:
            return ""
        return "## Memories from earlier attempts (A-Mem)\n\n" + "\n".join(lines)

    async def record(self, task: dict, result: dict, score=None) -> None:
        resp = (result.get("response") or "").strip()
        if not resp:
            return
        ns = _ns(self.benchmark, task)
        try:
            note = (f"Task: {task.get('description', '')[:1500]}\n"
                    f"Attempt: {resp[:2500]}")
            _fn = self._sys.add_note if hasattr(self._sys, "add_note") \
                else self._sys.create_memory
            nid = await asyncio.to_thread(_fn, note)   # LLM link-gen inside
            self._per_chain.setdefault(ns, []).append(nid)
        except Exception as e:
            self._add_failures = getattr(self, "_add_failures", 0) + 1
            if self._add_failures <= 3 or self._add_failures % 50 == 0:
                print(f"[baseline:amem] add failed ({self._add_failures}): {e}",
                      flush=True)

    def __len__(self) -> int:
        return sum(len(v) for v in self._per_chain.values())


class MemoryOSMemory:
    """MemoryOS (BAI-LAB, 2025) as a baseline arm — short/mid/long-term
    hierarchical memory behind its own module (pip install memoryos), pointed
    at the local OAI proxy."""

    def __init__(self, benchmark: str, top_k: int = 3) -> None:
        self.benchmark = benchmark
        self.top_k = top_k
        try:
            from memoryos import Memoryos
        except ImportError as e:
            raise SystemExit(
                "[baseline:memoryos] pip install memoryos (BAI-LAB/MemoryOS) "
                f"or swap the third baseline: {e}")
        self._cls = Memoryos
        self._per_ns: dict[str, object] = {}
        self._base = os.environ.get("OPENAI_API_BASE", "http://localhost:8741/v1")
        self._key = os.environ.get("OPENAI_API_KEY", "dummy")
        self._model = os.environ.get("CODEBUDDY_MODEL", "hy3").lower()

    def _sys_for(self, ns: str):
        if ns not in self._per_ns:
            safe = ns.replace(":", "_").replace("/", "_")
            try:
                self._per_ns[ns] = self._cls(
                    user_id=safe, openai_api_key=self._key,
                    openai_base_url=self._base,
                    data_storage_path=str(_STORE_ROOT / "memoryos" / safe),
                    llm_model=self._model)
            except TypeError:
                self._per_ns[ns] = self._cls(user_id=safe,
                                             openai_api_key=self._key)
        return self._per_ns[ns]

    def inject(self, task: dict) -> str:
        ns = _ns(self.benchmark, task)
        if ns not in self._per_ns:
            return ""      # nothing recorded for this chain yet
        sys_ = self._per_ns[ns]
        query = task.get("description", "")[:2000]
        ctx = None
        for meth in ("retrieve_context", "get_retrieval_context", "retrieve",
                     "search"):
            fn = getattr(sys_, meth, None)
            if fn is None:
                continue
            try:
                ctx = fn(query)
                break
            except TypeError:
                try:
                    ctx = fn(query, self.top_k)
                    break
                except Exception:
                    continue
            except Exception as e:
                print(f"[baseline:memoryos] {meth} failed: {e}", flush=True)
                return ""
        if not ctx:
            return ""
        text = ctx if isinstance(ctx, str) else str(ctx)
        text = text.strip()[:1200]
        if len(text) < 3:
            return ""
        return "## Memories from earlier attempts (MemoryOS)\n\n" + text

    async def record(self, task: dict, result: dict, score=None) -> None:
        resp = (result.get("response") or "").strip()
        if not resp:
            return
        ns = _ns(self.benchmark, task)
        try:
            await asyncio.to_thread(
                self._sys_for(ns).add_memory,
                user_input=task.get("description", "")[:3000],
                agent_response=resp[:3000])
        except Exception as e:
            self._add_failures = getattr(self, "_add_failures", 0) + 1
            if self._add_failures <= 3 or self._add_failures % 50 == 0:
                print(f"[baseline:memoryos] add failed ({self._add_failures}): "
                      f"{e}", flush=True)

    def __len__(self) -> int:
        return len(self._per_ns)


_REGISTRY = {"mem0": Mem0Memory, "amem": AMemMemory, "memoryos": MemoryOSMemory}


def make_external_memory(name: str, benchmark: str):
    if name not in _REGISTRY:
        raise SystemExit(f"[baseline] unknown external memory '{name}' — "
                         f"available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](benchmark)
