"""CuratedTerminus — official Terminus-2 with a memory-prefix (HARBOR_TB2_PLAN §2).

Subclass, not fork: the ONLY change is prepending the arm's injected memory
block to the task instruction. Arm/state come from env (set by
tb2_harbor_bridge.py, which also does record-after-eval and persists the store
between iterations):

  TB2_ARM        A | B | C            (A = no prefix, byte-identical Terminus-2)
  TB2_MEM_STATE  path to the pickled memory store (read-only here)

VERIFY ON SERVER (Gate 1 of the plan): the exact Terminus2 import path and the
perform_task signature of the pinned terminal-bench version. The override below
is signature-agnostic (prefixes the first str argument), so minor drift is
tolerated, but confirm on the smoke run that the prefix actually lands in the
prompt (grep the agent transcript for '## Curated prior attempts').
"""
from __future__ import annotations

import hashlib
import os
import pickle

_TERMINUS_PATHS = (
    ("terminal_bench.agents.terminus_2", "Terminus2"),
    ("terminal_bench.agents.terminus", "Terminus2"),
    ("harbor.agents.terminus_2", "Terminus2"),
)


def _load_terminus():
    err = []
    for mod, cls in _TERMINUS_PATHS:
        try:
            m = __import__(mod, fromlist=[cls])
            return getattr(m, cls)
        except Exception as e:  # noqa: BLE001 — collect and report all paths
            err.append(f"{mod}.{cls}: {type(e).__name__}: {e}")
    raise ImportError(
        "Terminus2 not importable — install terminal-bench in this env "
        "(HARBOR_TB2_PLAN §0). Tried:\n  " + "\n  ".join(err))


def _task_key(instruction: str) -> str:
    """Stable per-task id derived from the instruction, so the same task's
    iterations share a chain (ITER_CHAIN semantics) across harbor invocations."""
    return "tb2h_" + hashlib.sha1((instruction or "").encode()).hexdigest()[:12]


def _inject_block(instruction: str) -> str:
    arm = os.environ.get("TB2_ARM", "A").upper()
    state = os.environ.get("TB2_MEM_STATE", "")
    if arm == "A" or not state or not os.path.exists(state):
        return ""
    try:
        with open(state, "rb") as f:
            mem = pickle.load(f)          # BenchmarkMemory (B) or CuratedMemory (C)
        task = {"task_id": _task_key(instruction), "description": instruction}
        return mem.inject(task) or ""
    except Exception as e:  # never break the official agent over memory I/O
        print(f"[CuratedTerminus] inject skipped: {type(e).__name__}: {e}", flush=True)
        return ""


_Terminus2 = None


def _base():
    global _Terminus2
    if _Terminus2 is None:
        _Terminus2 = _load_terminus()
    return _Terminus2


try:
    _BaseAgent = _load_terminus()
except ImportError:
    _BaseAgent = object  # keeps this module importable where terminal-bench is
                         # absent (local tooling); using the class then fails
                         # loudly in _inject-time super() call, by design.


class CuratedTerminus(_BaseAgent):
    """Terminus2 subclass with a memory-prefix. harbor requires a real class
    (not a factory) for --agent-import-path; base resolves at import when
    terminal-bench is installed, else degrades to `object` for importability."""

    @staticmethod
    def name() -> str:  # harbor registry display name
        return "curated-terminus"

    def perform_task(self, *a, **kw):
        a = list(a)
        for i, v in enumerate(a):           # first str arg = the instruction
            if isinstance(v, str):
                block = _inject_block(v)
                if block:
                    a[i] = f"{block}\n\n---\n\n{v}"
                break
        else:
            for k, v in kw.items():
                if isinstance(v, str) and len(v) > 40:
                    block = _inject_block(v)
                    if block:
                        kw[k] = f"{block}\n\n---\n\n{v}"
                    break
        return super().perform_task(*a, **kw)
