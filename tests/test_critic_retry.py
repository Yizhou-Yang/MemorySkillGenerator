"""The critic's default verdict is total=5, which is the inject threshold, so a
gateway outage is indistinguishable from a real mediocre grade. These check that
failures are retried, then counted and flagged."""
import os, sys, time
os.environ["C_CRITIC_TRIES"] = "3"
os.environ["C_CRITIC_BACKOFF_S"] = "0"       # keep the test instant
sys.path.insert(0, ".")
from memlayer.refine import cross_agent_evaluate_skill, critic_failure_count
from memlayer.experience import Experience

def mkexp():
    e = Experience.__new__(Experience)
    e.task_desc, e.outcome = "book a flight", "success"
    e.action_commands, e.reasoning_trace = ["search", "book"], ["looked it up"]
    e.failure_taxonomy = {"causal_lesson": "check dates", "generalized_steps": "search then book"}
    return e

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

# 1. always fails -> retried C_CRITIC_TRIES times, flagged, counted
calls = {"n": 0}
def always_fail(_):
    calls["n"] += 1
    raise RuntimeError("HTTP Error 503: Service Unavailable")
before = critic_failure_count()
v = cross_agent_evaluate_skill(mkexp(), llm_fn=always_fail)
check("retried exactly C_CRITIC_TRIES times", calls["n"] == 3, f"calls={calls['n']}")
check("failure is flagged", v.get("critic_failed") is True)
check("failure is counted", critic_failure_count() == before + 1)
check("default is still the inject threshold", v["total"] == 5)
check("reason names the outage", "unavailable" in v.get("reason", ""), v.get("reason", ""))

# 2. flaky: succeeds on the 3rd try -> real verdict, not flagged, not counted
calls2 = {"n": 0}
def flaky(_):
    calls2["n"] += 1
    if calls2["n"] < 3:
        raise RuntimeError("HTTP Error 503")
    return '{"total": 8, "actionability": 3, "generalizability": 3, "correctness": 2, "novelty": 0}'
before2 = critic_failure_count()
v2 = cross_agent_evaluate_skill(mkexp(), llm_fn=flaky)
check("recovers on a later try", v2["total"] == 8, f"total={v2['total']}")
check("recovery is not flagged", v2.get("critic_failed") is False)
check("recovery is not counted", critic_failure_count() == before2)
check("verdict derived from score", v2.get("verdict") == "inject")

# 3. empty/garbage response is treated as a failure, not as a score
calls3 = {"n": 0}
def empties(_):
    calls3["n"] += 1
    return ""
v3 = cross_agent_evaluate_skill(mkexp(), llm_fn=empties)
check("empty response retried too", calls3["n"] == 3, f"calls={calls3['n']}")
check("empty response flagged as failure", v3.get("critic_failed") is True)

print("\n" + ("CRITIC RETRY VERIFIED" if ok else "CHECKS FAILED"))
sys.exit(0 if ok else 1)
