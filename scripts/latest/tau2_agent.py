"""CuratedTau2Agent — tau2-bench LLMAgent with a memory-prefix (the analog of
tb2_harbor_agent.CuratedTerminus for the terminal benchmark).

Subclass, not fork: the ONLY change is prepending the arm's injected memory
block to the agent's SYSTEM message before each LLM call. Arm and store come
from env (set by tau2_bridge.py, which also does record-after-eval and persists
the store between iterations):

  TAU2_ARM        A | B | C     (A = no prefix, byte-identical tau2 LLMAgent)
  TAU2_MEM_STATE  path to the pickled memory store (read-only here)

DESIGN NOTE — the retrieval chain is the SCENARIO, keyed on the opening user
message, NOT a task instruction the agent is handed up front. In tau2 the agent
never sees the task goal directly (the user simulator holds it); it only sees the
conversation. But the user's OPENING message is deterministic per scenario, so
hash(opening user message) is a stable chain id across iterations — exactly the
role hash(instruction) plays for TB2. We retrieve on the user text seen so far
(content-based, like every other benchmark) and inject once the first user turn
is visible. This targets tau2's known failure mode: inconsistent policy
adherence / pass^k collapse — precisely what an accumulated "what the policy
actually required / what I got wrong before" memory is for.

VERIFY ON SERVER (Gate 1, analog of HARBOR_TB2_PLAN §1): (a) the exact LLMAgent
import path, and (b) the name of the method that receives the message list and
calls the LLM in the pinned tau2-bench version. The override below defaults to
`generate_next_message`; if the pinned version names it differently
(`get_response` / `respond` / `act` / `step`), add that name to _GEN_METHODS.
The injection mutates the passed message list in place (prefixes the system
message), so it is signature-agnostic once the right method is wrapped. Confirm
on the smoke run that the block lands (grep a saved trajectory /
`--save-to` json for '## Curated prior attempts').
"""
from __future__ import annotations

import hashlib
import os
import pickle

# Candidate import paths for tau2's default LLM agent (VERIFY on the pinned
# version — collect-and-report all, like tb2_harbor_agent._TERMINUS_PATHS).
_LLMAGENT_PATHS = (
    ("tau2.agent.llm_agent", "LLMAgent"),
    ("tau2.agents.llm_agent", "LLMAgent"),
    ("tau2.agent.agent", "LLMAgent"),
    ("tau2.agent", "LLMAgent"),
)

# Method(s) that receive the running message list and call the LLM. The first
# one that exists on the base class is wrapped. VERIFY / extend on the server.
_GEN_METHODS = ("generate_next_message", "get_response", "respond", "act", "step")

# Marker the bridge greps for to confirm injection actually reached the prompt.
_MARKERS = ("## Curated prior attempts", "raw memory", "## What to avoid")


def _task_key(user_text: str) -> str:
    """Stable per-scenario id derived from the opening user message, so a
    scenario's iterations share a chain across tau2 invocations. MUST match
    tau2_bridge._task_key so record (bridge) and inject (here) hit the same
    chain."""
    return "tau2_" + hashlib.sha1((user_text or "").encode()).hexdigest()[:12]


def _load_llmagent():
    err = []
    for mod, cls in _LLMAGENT_PATHS:
        try:
            m = __import__(mod, fromlist=[cls])
            return getattr(m, cls)
        except Exception as e:  # noqa: BLE001 — collect and report all paths
            err.append(f"{mod}.{cls}: {type(e).__name__}: {e}")
    raise ImportError(
        "tau2 LLMAgent not importable — install tau2-bench in this env "
        "(`uv sync` in sierra-research/tau2-bench). Tried:\n  " + "\n  ".join(err))


def _inject_block(user_text: str) -> str:
    """Arm-gated memory block for the current scenario (read-only on the store)."""
    arm = os.environ.get("TAU2_ARM", "A").upper()
    state = os.environ.get("TAU2_MEM_STATE", "")
    if arm == "A" or not state or not os.path.exists(state) or not user_text:
        return ""
    try:
        with open(state, "rb") as f:
            mem = pickle.load(f)          # BenchmarkMemory (B) or CuratedMemory (C)
        task = {"task_id": _task_key(user_text), "description": user_text,
                "metadata": {"chain_id": _task_key(user_text)}}
        return mem.inject(task) or ""
    except Exception as e:  # never break the official agent over memory I/O
        print(f"[CuratedTau2Agent] inject skipped: {type(e).__name__}: {e}", flush=True)
        return ""


def _first_user_text(messages: list) -> str:
    """The scenario key material: the user turns seen so far, oldest first. The
    opening user message alone is enough for a stable chain id; later user turns
    enrich the retrieval query without changing the key (we key on the whole
    concatenation, which is dominated by and prefixed with the opener)."""
    parts = [m.get("content") for m in messages
             if isinstance(m, dict) and m.get("role") == "user"
             and isinstance(m.get("content"), str) and m.get("content").strip()]
    return "\n".join(parts)[:2000]


def _prefix_system(messages: list, block: str) -> None:
    """Prepend `block` to the system message in place (insert one if absent).
    Idempotent: skips if a memory marker is already present (the same agent
    instance may generate several turns for one scenario)."""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system" \
                and isinstance(m.get("content"), str):
            if not any(mk in m["content"] for mk in _MARKERS):
                m["content"] = f"{block}\n\n---\n\n{m['content']}"
            return
    messages.insert(0, {"role": "system", "content": block})


try:
    _BaseAgent = _load_llmagent()
except ImportError:
    _BaseAgent = object  # keeps this module importable where tau2 is absent
                         # (local tooling); using the class then fails loudly in
                         # the super() call, by design.


class CuratedTau2Agent(_BaseAgent):
    """tau2 LLMAgent subclass with a memory-prefix. tau2's custom-agent hook
    wants a real class (resolved by import path); base resolves at import when
    tau2 is installed, else degrades to `object` for importability."""

    @staticmethod
    def name() -> str:  # tau2 registry display name
        return "curated-tau2-agent"

    def _inject(self, args, kwargs) -> None:
        # Find the running message list (first list of role-tagged dicts) among
        # positional or keyword args — signature-agnostic across tau2 versions.
        msgs = None
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, list) and v and isinstance(v[0], dict) and "role" in v[0]:
                msgs = v
                break
        if msgs is None:
            return
        block = _inject_block(_first_user_text(msgs))
        if block:
            _prefix_system(msgs, block)


def _make_wrapper(method_name: str):
    def _wrapped(self, *a, **kw):
        try:
            self._inject(a, kw)
        except Exception as e:  # never break the official agent over injection
            print(f"[CuratedTau2Agent] {method_name} inject skipped: "
                  f"{type(e).__name__}: {e}", flush=True)
        return getattr(super(CuratedTau2Agent, self), method_name)(*a, **kw)
    _wrapped.__name__ = method_name
    return _wrapped


# Wrap whichever generate method the pinned base actually defines. Defining an
# override for a name the base lacks would shadow nothing useful and could break
# super() — so we bind overrides ONLY for methods present on the base class.
if _BaseAgent is not object:
    for _mname in _GEN_METHODS:
        if hasattr(_BaseAgent, _mname):
            setattr(CuratedTau2Agent, _mname, _make_wrapper(_mname))
