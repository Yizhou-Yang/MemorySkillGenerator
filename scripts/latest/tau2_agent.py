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


_MEM_CACHE: dict = {}   # single entry: {"key": (path, mtime), "mem": obj}


def _load_mem(state: str):
    """Unpickle the store once per (path, mtime), not once per inject call.

    The pickle only changes between tau2 runs (the bridge records after the
    subprocess exits), so within a run the cache always hits. For external
    stores (mem0) this is correctness, not just speed: each unpickled copy
    lazily opens its own client on a file-locked path (local qdrant), and a
    second copy would deadlock against the first.
    """
    key = (state, os.path.getmtime(state))
    if _MEM_CACHE.get("key") != key:
        with open(state, "rb") as f:
            _MEM_CACHE.update(key=key, mem=pickle.load(f))
    return _MEM_CACHE["mem"]


# Ceiling on what any arm may put in a prompt. Blocks grow with the store, and
# on a small-context backbone an oversized one kills the arm instead of just
# crowding the prompt.
_BLOCK_MAX_CH = int(os.environ.get("MEM_BLOCK_MAX_CH", "10000"))

_INJ_COUNT = [0]


def _msg_role_content(m):
    """(role, content) from a pydantic Message or a plain dict."""
    if isinstance(m, dict):
        return m.get("role"), m.get("content")
    return getattr(m, "role", None), getattr(m, "content", None)


def _state_user_texts(state, incoming):
    """(opening user message, all user turns) from an LLMAgentState.

    The wrapper runs before the base method appends `incoming`, so on the first
    turn state.messages is empty and the opener IS the incoming message.
    """
    parts = []
    for m in list(getattr(state, "messages", None) or []):
        role, content = _msg_role_content(m)
        if role == "user" and isinstance(content, str) and content.strip():
            parts.append(content)
    role, content = _msg_role_content(incoming)
    if role == "user" and isinstance(content, str) and content.strip():
        parts.append(content)
    if not parts:
        return "", ""
    return parts[0][:2000], "\n".join(parts)[:2000]


def _prefix_state_system(state, block: str) -> None:
    """Prepend `block` to the state's system message, idempotently."""
    sys_msgs = getattr(state, "system_messages", None)
    if sys_msgs is None:
        return
    for m in sys_msgs:
        role, content = _msg_role_content(m)
        if role == "system" and isinstance(content, str):
            if any(mk in content for mk in _MARKERS):
                return
            new = f"{block}\n\n---\n\n{content}"
            if isinstance(m, dict):
                m["content"] = new
            else:
                try:
                    object.__setattr__(m, "content", new)
                except Exception:
                    m.content = new
            _INJ_COUNT[0] += 1
            if _INJ_COUNT[0] == 1 or _INJ_COUNT[0] % 25 == 0:
                print(f"[CuratedTau2Agent] injected {_INJ_COUNT[0]} blocks "
                      f"(latest {len(block)} chars)", flush=True)
            return
    # no system message to extend: add one carrying just the block
    if sys_msgs and not isinstance(sys_msgs[0], dict):
        try:
            sys_msgs.insert(0, type(sys_msgs[0])(role="system", content=block))
            return
        except Exception:
            pass
    sys_msgs.insert(0, {"role": "system", "content": block})


def _inject_block(opening_text: str, query_text: str) -> str:
    """Arm-gated memory block for the current scenario (read-only on the store).

    The chain key comes from the OPENING user message alone. It used to be
    hashed from the concatenation of every user turn so far, which changes the
    sha1 on every turn of the dialogue: record() (bridge, keyed on the opener)
    wrote to one chain while inject() read from a different one each turn, so
    the store filled with single-entry orphan chains and the curated arm
    retrieved other dialogues' fragments -- rewards fell iteration over
    iteration while the uncurated arm, injecting verbatim text, held flat. The
    full concatenation survives only as the retrieval QUERY, where growing with
    the dialogue is what you want and no key stability is required."""
    arm = os.environ.get("TAU2_ARM", "A").upper()
    state = os.environ.get("TAU2_MEM_STATE", "")
    if arm == "A" or not state or not os.path.exists(state) or not opening_text:
        return ""
    try:
        mem = _load_mem(state)        # BenchmarkMemory (B) / CuratedMemory (C) / external (mem0)
        key = _task_key(opening_text)
        task = {"task_id": key, "description": query_text or opening_text,
                "metadata": {"chain_id": key}}
        _blk = mem.inject(task) or ""
        if len(_blk) > _BLOCK_MAX_CH:
            _blk = _blk[:_BLOCK_MAX_CH] + "\n[block truncated]"
        return _blk
    except Exception as e:  # never break the official agent over memory I/O
        print(f"[CuratedTau2Agent] inject skipped: {type(e).__name__}: {e}", flush=True)
        return ""


def _user_texts(messages: list) -> tuple[str, str]:
    """(opening user message, all user turns so far). The opener is the chain
    key -- it never changes across the dialogue, so it MUST be hashed alone; a
    hash is not prefix-stable, so hashing the growing concatenation changes the
    chain id on every turn. The concatenation is retrieval query material only."""
    parts = [m.get("content") for m in messages
             if isinstance(m, dict) and m.get("role") == "user"
             and isinstance(m.get("content"), str) and m.get("content").strip()]
    if not parts:
        return "", ""
    return parts[0][:2000], "\n".join(parts)[:2000]


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
        if msgs is not None:
            block = _inject_block(*_user_texts(msgs))
            if block:
                _prefix_system(msgs, block)
            return
        # The pinned tau2 calls generate_next_message(message, state) and never
        # passes raw dicts, so the search above finds nothing. Read the transcript
        # off the state instead and prefix the state's system message.
        state = incoming = None
        for v in list(args) + list(kwargs.values()):
            if hasattr(v, "messages") and hasattr(v, "system_messages"):
                state = v
            elif _msg_role_content(v)[0] is not None:
                incoming = v
        if state is None:
            return
        block = _inject_block(*_state_user_texts(state, incoming))
        if block:
            _prefix_state_system(state, block)


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


def create_curated_tau2_agent(tools, domain_policy, **kwargs):
    """Factory function for CuratedTau2Agent — matches tau2's create_llm_agent
    signature so it can be registered via registry.register_agent_factory()."""
    return CuratedTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
