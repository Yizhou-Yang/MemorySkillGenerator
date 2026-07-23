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


def _log_version(name: str, module) -> None:
    """Print AND persist the baseline module's version — the paper pins
    versions in the appendix, and this file is where those pins come from
    (captured at run time, not transcribed by hand)."""
    ver = getattr(module, "__version__", "unknown")
    print(f"[baseline:{name}] version={ver}", flush=True)
    try:
        _STORE_ROOT.mkdir(parents=True, exist_ok=True)
        with open(_STORE_ROOT / "versions.txt", "a") as f:
            f.write(f"{name}=={ver}\n")
    except Exception:
        pass


class Mem0Memory:
    """Mem0 (mem0.ai OSS) as a baseline arm, using ITS OWN add/search.

    Picklable by design: tau2 runs the agent in a subprocess that receives the
    memory as a pickle (TAU2_MEM_STATE), so the live clients (qdrant holds a
    file lock; the OpenAI client holds sockets) are built lazily on first use
    and dropped from __getstate__. release() frees the on-disk qdrant lock so
    the bridge process and the tau2 subprocess can take turns on one store.
    """

    # Local (embedded) qdrant is not safe under concurrent access from multiple
    # threads of one client, and tau2 runs sims concurrently — serialize all
    # store I/O. Class-level: locks are unpicklable, instances must not own one.
    _IO_LOCK = __import__("threading").Lock()

    def __init__(self, benchmark: str, top_k: int = 3) -> None:
        self.benchmark = benchmark
        self.top_k = top_k
        try:
            from mem0 import Memory  # noqa: F401 — fail fast before a sweep launches
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
            # embedding_model_dims MUST match the local embedder (all-MiniLM-L6-v2
            # is 384). Without it mem0 builds the qdrant collection at its OpenAI
            # default (1536) and every add fails "shapes (0,1536) and (384,)".
            # Path is per-BENCHMARK: local qdrant file-locks its directory, and
            # two benchmarks (or the tau2 bridge + its subprocess) each holding a
            # client on one shared path would deadlock on that lock.
            "vector_store": {"provider": "qdrant",
                             "config": {"path": str(_STORE_ROOT / f"mem0_{benchmark}"),
                                        "on_disk": True,
                                        "embedding_model_dims": int(
                                            os.environ.get("MEM0_EMBED_DIMS", "384"))}},
        }
        override = os.environ.get("MEM0_CONFIG_JSON")
        if override:
            import json
            cfg = json.loads(override)
        self._cfg = cfg
        self._m = None                      # built lazily (see _mem)
        self._add_failures = 0

    def _mem(self):
        if self._m is None:
            from mem0 import Memory
            self._m = Memory.from_config(self._cfg)
            try:
                import mem0 as _m0
                _log_version("mem0ai", _m0)
            except Exception:
                pass
        return self._m

    def release(self) -> None:
        """Drop the live clients (frees the local-qdrant file lock) so another
        process — the tau2 subprocess, or the bridge after it — can open the
        same on-disk store. State lives on disk; nothing is lost."""
        m, self._m = self._m, None
        if m is not None:
            try:
                m.vector_store.client.close()   # qdrant releases the lock now,
            except Exception:                   # not at some later GC
                pass

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_m"] = None                      # clients never cross a pickle
        return d

    def inject(self, task: dict) -> str:
        query = task.get("description", "")[:2000]
        try:
            # mem0 >=2.x moved entity identity out of top-level kwargs: search()
            # takes filters={"user_id": ...}, not user_id=... (add() still takes
            # the top-level kwarg).
            with self._IO_LOCK:
                hits = self._mem().search(query,
                                          filters={"user_id": _ns(self.benchmark, task)},
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
            def _add():
                with self._IO_LOCK:
                    self._mem().add(
                        [{"role": "user", "content": task.get("description", "")[:4000]},
                         {"role": "assistant", "content": resp[:4000]}],
                        user_id=_ns(self.benchmark, task))
            await asyncio.to_thread(_add)
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
        self._sys = self._build_sys()
        try:
            import agentic_memory as _am
            _log_version("agentic-memory", _am)
        except Exception:
            print("[baseline:amem] version=repo (pin the commit hash in the "
                  "appendix)", flush=True)
        self._per_chain: dict[str, list[str]] = {}

    @staticmethod
    def _build_sys():
        amem_path = os.environ.get("AMEM_PATH")
        if amem_path:
            import sys
            if amem_path not in sys.path:
                sys.path.insert(0, amem_path)
        try:
            from agentic_memory.memory_system import AgenticMemorySystem
        except ImportError:
            try:
                from memory_system import AgenticMemorySystem  # repo root layout
            except ImportError:
                try:
                    # 2025-07 WujiangXu/AgenticMemory layout: the class lives
                    # in memory_layer.py at the repo root.
                    from memory_layer import AgenticMemorySystem
                except ImportError as e:
                    raise SystemExit(
                        "[baseline:amem] A-Mem module not importable — pip install "
                        "-e the WujiangXu/AgenticMemory repo or set AMEM_PATH: "
                        f"{e}")
        model = os.environ.get("CODEBUDDY_MODEL", "hy3").lower()
        # A-Mem's ctor signature varies across revisions; try the documented
        # form first, degrade to defaults rather than guessing kwargs.
        try:
            return AgenticMemorySystem(
                model_name=os.environ.get("AMEM_EMBED_MODEL", "all-MiniLM-L6-v2"),
                llm_backend="openai", llm_model=model,
                api_key=os.environ.get("OPENAI_API_KEY"),
                api_base=os.environ.get("OPENAI_API_BASE"))
        except TypeError:
            try:
                return AgenticMemorySystem(
                    model_name=os.environ.get("AMEM_EMBED_MODEL", "all-MiniLM-L6-v2"),
                    llm_backend="openai", llm_model=model)
            except TypeError:
                return AgenticMemorySystem()

    # ── pickle support (tau2 bridge hands the store between processes) ──
    # The live system holds a SentenceTransformer + an LLM client (both
    # unpicklable). Dehydrate to the raw MemoryNote fields; rehydrate lazily
    # by rebuilding the system, re-inserting the notes with ALL fields set
    # (skips A-Mem's LLM analysis) and one consolidate_memories() pass
    # (local re-embedding only — zero LLM calls).
    _NOTE_FIELDS = ("content", "id", "keywords", "links", "importance_score",
                    "retrieval_count", "timestamp", "last_accessed", "context",
                    "evolution_history", "category", "tags")

    def __getstate__(self):
        d = dict(self.__dict__)
        sys_ = d.pop("_sys", None)
        notes = []
        for n in (getattr(sys_, "memories", {}) or {}).values():
            notes.append({f: getattr(n, f, None) for f in self._NOTE_FIELDS})
        d["_dehydrated_notes"] = notes
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._sys = None            # rebuilt lazily by _system()

    def _system(self):
        if getattr(self, "_sys", None) is None:
            self._sys = self._build_sys()
            notes = getattr(self, "_dehydrated_notes", None) or []
            if notes:
                try:
                    from memory_layer import MemoryNote
                except ImportError:
                    MemoryNote = None
                for f in notes:
                    try:
                        if MemoryNote is None:
                            break
                        kw = {k: v for k, v in f.items() if v is not None}
                        n = MemoryNote(**kw)   # all fields given -> no LLM
                        self._sys.memories[n.id] = n
                    except Exception:
                        pass
                try:
                    self._sys.consolidate_memories()
                except Exception:
                    pass
                self._dehydrated_notes = []
        return self._sys

    def inject(self, task: dict) -> str:
        query = task.get("description", "")[:2000]
        ns = _ns(self.benchmark, task)
        try:
            _sys = self._system()
            for _meth in ("search_agentic", "search",
                          "find_related_memories_raw", "find_related_memories"):
                if hasattr(_sys, _meth):
                    hits = getattr(_sys, _meth)(query, k=self.top_k * 4)
                    break
            else:
                hits = []
        except Exception as e:
            print(f"[baseline:amem] search failed: {e}", flush=True)
            return ""
        lines = []
        chain_ids = set(self._per_chain.get(ns, []))
        for h in hits or []:
            if isinstance(h, dict):
                hid = h.get("id")
                text = h.get("content") or h.get("context") or ""
            elif hasattr(h, "content"):        # MemoryNote object (2025-07 layout)
                hid = getattr(h, "id", None)
                text = getattr(h, "content", "") or ""
            else:
                hid, text = None, str(h)
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
            _sys = self._system()
            _fn = _sys.add_note if hasattr(_sys, "add_note") \
                else _sys.create_memory
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
        try:
            import memoryos as _mos
            _log_version("memoryos", _mos)
        except Exception:
            pass
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
