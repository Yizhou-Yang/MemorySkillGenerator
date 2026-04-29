"""Runner-level EvoMem / GPR bridge — the paper's A/B/C, made uniform.

This replaces the per-benchmark grab-bag of B/C mechanisms (prompt blobs,
self-consistency, within-task patches) with ONE consistent definition that
matches the paper:

  A  Vanilla : no memory.
  B  EvoMem  : accumulate a cross-task patch history; retrieve relevant patches
               for the current task and inject them (plain).
  C  GPR     : inject the SAME patches as B (strict superset) plus an
               environment-check annotation per patch — Grounded Patch
               Resolution. At the runner level the env-check is UNKNOWN for
               non-executable benchmarks; for executable ones (Terminal-Bench)
               the agent grounds internally via the probe. C is always a
               superset of B's context, so C can never lose B's information.

Each group keeps its OWN PatchMemory (B and C are independent conditions).
Memory accumulates across the tasks of a chain (grouped by `chain_id`; if a
benchmark has no chain structure, each task is its own singleton chain and the
patch history stays empty — i.e. EvoMem/GPR only differentiate when there is
cross-task memory to transfer, which is the honest behaviour).

NOTE: this generic patch memory is intentionally agent-agnostic. The paper's
headline numbers come from agent-specific EvoMem instantiations (Memento tips,
A-Mem notes, Terminus2 chain memory) and the live GPR grounding in
vgr_experiment.py; this bridge gives latest_runner a consistent, paper-aligned
A/B/C surface across all benchmarks.
"""
from __future__ import annotations

import threading
from typing import Optional

from scripts.latest._vgr import (
    Patch, PatchMemory, GroundedPatch,
    render_patches_plain, render_patches_grounded,
)


def _chain_id(task: dict) -> str:
    meta = task.get("metadata") or {}
    return (task.get("chain_id") or meta.get("chain_id")
            or meta.get("chain") or task.get("task_id", ""))


def _key(task: dict) -> str:
    meta = task.get("metadata") or {}
    return str(meta.get("category") or meta.get("config")
               or meta.get("level") or "general")


class BenchmarkMemory:
    """Per-group cross-task patch memory for one benchmark run."""

    def __init__(self, benchmark: str, mode: str, top_k: int = 3) -> None:
        self.benchmark = benchmark
        self.mode = mode                  # "B" (EvoMem) or "C" (GPR)
        self.top_k = top_k
        self._mem = PatchMemory()
        self._n = 0
        self._lock = threading.Lock()

    # ── read side: build the injected context for a task ──
    def inject(self, task: dict) -> str:
        cand = self._mem.retrieve(task.get("description", task.get("task_id", "")),
                                  top_k=self.top_k, chain_id=_chain_id(task))
        if not cand:
            return ""
        if self.mode == "B":
            return render_patches_plain(cand)
        # C / GPR: additive superset — same patches, annotated for grounding.
        # Runner-level env-check is UNKNOWN; executable agents ground internally.
        return render_patches_grounded([GroundedPatch(p) for p in cand])

    # ── write side: record a patch from a finished task ──
    def record(self, task: dict, result: dict) -> None:
        resp = (result.get("response") or "").strip()
        if not resp:
            return
        with self._lock:
            self._n += 1
            n = self._n
        summary = resp.splitlines()[0][:120] if resp else ""
        self._mem.add(Patch(
            patch_id=f"{self.benchmark}_{self.mode}_{n}",
            chain_id=_chain_id(task),
            version=n,
            key=_key(task),
            summary=summary,
            content_before="",
            content_after=resp[:400],
            rationale="solved earlier task in this chain",
            evidence=(task.get("description", "") or "")[:200],
        ))

    def __len__(self) -> int:
        return len(self._mem)


async def solve_with_memory(run_fn, task: dict, mem: BenchmarkMemory,
                            group: str) -> dict:
    """Inject memory → run the baseline agent → record a patch. Used for B and C.
    `run_fn` is the benchmark's baseline runner: run_fn(task, experience, group)."""
    injected = mem.inject(task)
    r = await run_fn(task, injected, group)
    if isinstance(r, dict):
        r.setdefault("_aug_prompt", injected)
        r["_aug_prompt"] = injected
        try:
            mem.record(task, r)
        except Exception:
            pass
    return r
