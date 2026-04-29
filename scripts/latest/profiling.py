"""Per-task wall-clock profiler — always on, no flags.

Answers "where did this task's time go?" so you can decide how to scale:
  - embed  : CPU-bound (SentenceTransformer.encode). High → concurrency hurts.
  - docker : container exec. High → memory-bound, keep DOCKER_CONCURRENCY low.
  - llm_io : everything else (LLM/network/orchestration) = total − embed − docker.
             Network-bound → concurrency HELPS here.

Implementation: a contextvar holds a per-task accumulator. Each asyncio task
copies the current context when created, so per-task totals stay isolated even
under concurrency. Everything here is stdlib and defensive — `timed()` never
raises and is a no-op when no task profile is active, so instrumentation can
never break a run.
"""
from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager

_task_prof: contextvars.ContextVar = contextvars.ContextVar("task_prof", default=None)


def start_task_profile() -> dict:
    """Begin profiling the current task; returns the fresh accumulator."""
    d = {"embed": 0.0, "docker": 0.0}
    _task_prof.set(d)
    return d


def read_task_profile() -> dict | None:
    """Return the current task's accumulator (or None if not profiling)."""
    return _task_prof.get()


@contextmanager
def timed(category: str):
    """Accumulate wall-clock for `category` into the current task profile.
    No-op (but still runs the body) when no profile is active."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        try:
            d = _task_prof.get()
            if d is not None:
                d[category] = d.get(category, 0.0) + (time.perf_counter() - t0)
        except Exception:
            pass


def summarize(total_s: float, prof: dict | None) -> dict:
    """Turn a raw accumulator + total task time into a reportable breakdown.
    llm_io is the remainder (LLM + network + orchestration)."""
    prof = prof or {}
    embed = round(prof.get("embed", 0.0), 2)
    docker = round(prof.get("docker", 0.0), 2)
    llm_io = round(max(0.0, total_s - embed - docker), 2)
    return {"total_s": round(total_s, 2), "embed_s": embed,
            "docker_s": docker, "llm_io_s": llm_io}
