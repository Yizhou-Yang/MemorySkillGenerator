"""A long-running episode must not starve the short calls that score it.

Both used to share asyncio's default executor (32 threads): at
TASK_CONCURRENCY=24 the episodes held most of it for minutes, the judge calls
queued behind them, and a judge call that waited past its timeout returned ""
-- which `"[[Match]]" in ""` scored as a non-match, collapsing the task's recall
with no error recorded. These check the two fixes: episodes run in their own
pool, and an empty verdict is counted rather than silently treated as a
mismatch.
"""
import asyncio
import os
import sys
import threading
import time

sys.path.insert(0, ".")
sys.path.insert(0, "src")

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# ── 1. episodes get a pool of their own, sized to the task concurrency ──
os.environ["TASK_CONCURRENCY"] = "24"
from scripts.latest.gaia2_runner import _episode_pool

pool = _episode_pool()
check("episode pool is separate from the default executor",
      pool is not None and pool is not asyncio.get_event_loop_policy().new_event_loop()._default_executor
      if hasattr(asyncio.get_event_loop_policy().new_event_loop(), "_default_executor") else pool is not None)
check("episode pool sized to TASK_CONCURRENCY", pool._max_workers >= 24,
      f"max_workers={pool._max_workers}")
check("episode pool is a singleton", _episode_pool() is pool)


# ── 2. long episodes in that pool leave the default executor free ──
async def scenario():
    """Occupy the episode pool with slow work, then time a short default-pool
    call. Before the fix both landed in the same 32-slot pool."""
    loop = asyncio.get_running_loop()
    started = threading.Event()

    def slow_episode():
        started.set()
        time.sleep(1.5)
        return "episode"

    eps = [loop.run_in_executor(_episode_pool(), slow_episode) for _ in range(24)]
    started.wait(timeout=5)
    t0 = time.perf_counter()
    await loop.run_in_executor(None, lambda: time.sleep(0.01))
    short_latency = time.perf_counter() - t0
    await asyncio.gather(*eps)
    return short_latency


latency = asyncio.run(scenario())
check("a short call is not blocked by 24 busy episodes", latency < 0.5,
      f"waited {latency*1000:.0f}ms")


# ── 3. an empty judge verdict is counted, not read as a mismatch ──
from latest.eval.gaia2_judge import _judge_action_pair, judge_empty_count, NormalizedAction


def mk(tool, val=1):
    return NormalizedAction(canonical_tool=tool, original_tool=tool,
                            args={"a": val}, index=0)


# Identical args short-circuit to "Exact normalized argument match" without ever
# calling the judge, so the pair must differ for the judge path to run at all.
ORACLE, AGENT = mk("App__send", 1), mk("App__send", 2)


async def empty_judge(_sys, _user):
    return ""


async def matching_judge(_sys, _user):
    return "reasoning ... [[Match]]"


before = judge_empty_count()
m, reason = asyncio.run(_judge_action_pair(empty_judge, task="t", oracle_action=ORACLE, agent_action=AGENT))
check("empty verdict does not count as a match", m is False)
check("empty verdict is counted", judge_empty_count() == before + 1,
      f"{before} -> {judge_empty_count()}")
check("empty verdict is labelled", reason == "__judge_unavailable__", reason)

before2 = judge_empty_count()
m2, _ = asyncio.run(_judge_action_pair(matching_judge, task="t", oracle_action=ORACLE, agent_action=AGENT))
check("a real match still matches", m2 is True)
check("a real verdict is not counted as empty", judge_empty_count() == before2)

print("\n" + ("POOL ISOLATION VERIFIED" if ok else "CHECKS FAILED"))


def test_pool_isolation():
    """pytest entry point; the checks above run at import and record into `ok`."""
    assert ok


if __name__ == "__main__":
    sys.exit(0 if ok else 1)
