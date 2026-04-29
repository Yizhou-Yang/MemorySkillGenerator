"""Grounded Patch Resolution (GPR) — Method C on top of EvoMem patch memory.

Design principle: **additive, never subtractive.** EvoMem (Method B) injects the
top-k retrieved patches and lets the agent guess which one is valid for the
current environment version. Method C keeps *all* of B's patches and *adds*
ground-truth evidence from the live environment:

  C's prompt context  ==  B's patch block  (verbatim)
                          + per-patch verification annotations (probe results)
                          + (optional) a repair hint after a failed action.

Because C's context is a strict superset of B's, and the added evidence is
ground truth from the environment (not a new LLM judge), C dominates B when
grounding is informative and ties otherwise. There is no "filter out unverified
patches" step — that would lose information and could underperform B.

Two ingredients give this real teeth:
  1. Self-verifying patches: each patch carries an *applicability predicate*
     (an executable probe + expected signal) synthesized at write time, so
     read-time grounding is cheap and reliable.
  2. Verify-and-repair: act, let the environment verify the outcome, and if the
     result contradicts a grounded-current state, repair before the error
     propagates down the chain.

The module is dependency-light (stdlib only) so the engine and its tests run
without the LLM/Docker stack. Verifiers are pluggable: the environment probe
verifier lives in scripts/latest/terminal_verifier.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Protocol


# ─────────────────────────────────────────────────────────────────────────────
#  Patch schema (EvoMem fields + self-verifying applicability predicate)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Patch:
    """A versioned memory patch. The first six fields mirror EvoMem's schema
    (p_t = (tau, C-, C+, r, z, e)); the last three are Method C's additive
    *applicability predicate* — how to test, at decision time, whether this
    patch's world-state currently holds."""
    patch_id: str
    chain_id: str
    version: int                      # tau: which version/step produced it
    key: str                          # the aspect this patch governs (e.g. "output_path", "branch")
    summary: str                      # z: one-line semantic summary
    content_before: str               # C-: state before the update
    content_after: str                # C+: state after the update
    rationale: str = ""               # r: why it changed
    evidence: str = ""                # e: triggering context
    # ── self-verifying applicability predicate (Method C) ──
    probe: str = ""                   # executable check, e.g. a shell command
    expected_signal: str = ""         # observation substring meaning "this patch applies now"
    is_negative: bool = False         # "avoid this" patch (apply only if its failure state is confirmed)

    def _doc(self) -> str:
        return " ".join([self.key, self.summary, self.content_before,
                         self.content_after, self.rationale])


@dataclass
class VerificationResult:
    """Outcome of grounding one patch against the live environment."""
    applies: Optional[bool]           # True / False / None (probe inconclusive or absent)
    confidence: float                 # 1.0 for direct environment observation
    observation: str = ""             # truncated probe output
    probe: str = ""
    method: str = "env_probe"

    @property
    def label(self) -> str:
        if self.applies is True:
            return "HOLDS"
        if self.applies is False:
            return "DOES-NOT-HOLD"
        return "UNKNOWN"


@dataclass
class GroundedPatch:
    """A retrieved patch paired with its (optional) verification result.
    Note: grounding NEVER drops a patch — every retrieved patch is carried
    forward, annotated with whatever the environment said about it."""
    patch: Patch
    verification: Optional[VerificationResult] = None


class Verifier(Protocol):
    """Pluggable environment verifier. The environment-probe implementation
    lives in scripts/latest/terminal_verifier.py; tests use a fake exec_fn."""
    async def verify(self, patch: Patch, context: dict) -> VerificationResult: ...


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-task patch memory (this is what makes the experiment match the paper:
#  patches accumulate ACROSS tasks in a chain, not within a single task)
# ─────────────────────────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _lexical_sim(query: str, doc: str) -> float:
    a, b = _tokens(query), _tokens(doc)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class PatchMemory:
    """Append-only patch history shared across the tasks of an evolution chain.
    Retrieval mirrors EvoMem: lexical relevance to the current task, recency as
    the tie-breaker. The same retrieval feeds both B and C; the difference is
    purely what happens AFTER retrieval (B injects, C grounds-then-injects)."""

    def __init__(self) -> None:
        self._patches: list[Patch] = []

    def __len__(self) -> int:
        return len(self._patches)

    @property
    def patches(self) -> list[Patch]:
        return list(self._patches)

    def add(self, patch: Patch) -> None:
        self._patches.append(patch)

    def retrieve(self, query: str, top_k: int = 5,
                 chain_id: Optional[str] = None) -> list[Patch]:
        pool = [p for p in self._patches
                if chain_id is None or p.chain_id == chain_id]
        scored = [(p, _lexical_sim(query, p._doc()), p.version)
                  for p in pool]
        # relevance desc, then recency (version) desc
        scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
        return [p for p, s, _ in scored[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
#  Grounding (additive): annotate every retrieved patch with env truth
# ─────────────────────────────────────────────────────────────────────────────

async def ground(candidates: list[Patch], verifier: Verifier,
                 context: dict) -> list[GroundedPatch]:
    """Run each candidate patch's applicability probe against the environment
    and attach the result. CRITICAL: returns one GroundedPatch per candidate —
    nothing is filtered out. The agent always sees the full patch set; grounding
    only *adds* a truth label per patch."""
    grounded: list[GroundedPatch] = []
    for p in candidates:
        ver: Optional[VerificationResult] = None
        if p.probe:
            try:
                ver = await verifier.verify(p, context)
            except Exception as e:  # a flaky probe must never drop the patch
                ver = VerificationResult(applies=None, confidence=0.0,
                                         observation=f"probe_error: {str(e)[:120]}",
                                         probe=p.probe, method="env_probe_error")
        grounded.append(GroundedPatch(patch=p, verification=ver))
    return grounded


# ─────────────────────────────────────────────────────────────────────────────
#  Rendering — B's injection vs C's injection (C is a superset of B)
# ─────────────────────────────────────────────────────────────────────────────

def _patch_line(p: Patch) -> str:
    neg = " [AVOID]" if p.is_negative else ""
    return (f"- [{p.key} @v{p.version}]{neg} {p.summary}: "
            f"{p.content_before} -> {p.content_after}"
            + (f" (because {p.rationale})" if p.rationale else ""))


def render_patches_plain(patches: list[Patch]) -> str:
    """Method B injection: the retrieved patches, nothing else."""
    if not patches:
        return ""
    body = "\n".join(_patch_line(p) for p in patches)
    return ("## Patch history (memory)\n"
            "These record what changed in earlier versions and why:\n" + body)


def render_patches_grounded(grounded: list[GroundedPatch]) -> str:
    """Method C injection: the SAME patch lines as B (verbatim), each followed
    by a ground-truth annotation from the live environment. By construction this
    contains render_patches_plain([g.patch ...]) line-for-line, plus extra
    evidence — so C's context is a strict superset of B's."""
    if not grounded:
        return ""
    lines: list[str] = []
    for g in grounded:
        lines.append(_patch_line(g.patch))          # identical to B's line
        v = g.verification
        if v is not None:
            obs = (v.observation or "").strip().replace("\n", " ")
            if len(obs) > 160:
                obs = obs[:157] + "..."
            lines.append(f"    [ENV-CHECK: {v.label}] probe=`{v.probe}`"
                         + (f" observed: {obs}" if obs else ""))
    return ("## Patch history (memory) + live environment checks\n"
            "Each patch is annotated with whether its assumption HOLDS in the\n"
            "current environment, verified by probing. Trust the ENV-CHECK over\n"
            "the patch when they disagree:\n" + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Verify-and-repair (the loop; ablatable separately from grounding)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionOutcome:
    """Result of executing the agent's action, as judged by the environment
    (e.g. the task's own tests). This is the trustworthy verifier — the same
    signal the benchmark already uses — not a new LLM judge."""
    passed: bool
    observation: str = ""


def repair_hint(outcome: ActionOutcome,
                grounded: list[GroundedPatch]) -> Optional[str]:
    """If the action failed AND the environment confirmed a current-version
    state (a patch whose probe says HOLDS), surface that discrepancy as a
    corrective hint for the next attempt. Returns None when there is nothing
    grounded to repair toward (so repair never fires blindly)."""
    if outcome.passed:
        return None
    confirmed = [g for g in grounded
                 if g.verification and g.verification.applies is True
                 and not g.patch.is_negative]
    if not confirmed:
        return None
    lines = ["## Repair: your last action failed the environment check.",
             f"Failure observation: {outcome.observation[:200]}",
             "The environment confirmed these current-version states — your "
             "next attempt MUST be consistent with them:"]
    for g in confirmed:
        probe = g.verification.probe if g.verification else ""
        lines.append(f"- {g.patch.key}: now `{g.patch.content_after}`"
                     + (f" (probe `{probe}` confirmed)" if probe else ""))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  High-level resolver tying it together (used by the live experiment backend)
# ─────────────────────────────────────────────────────────────────────────────

# A "solver" runs the agent for one attempt given the task and an injected
# context string, and returns (answer, ActionOutcome). Supplied by the backend.
Solver = Callable[[dict, str], Awaitable[tuple[str, ActionOutcome]]]


@dataclass
class ResolveConfig:
    mode: str = "C"                   # "A" base | "B" patch-memory | "C" grounded
    enable_grounding: bool = True     # C: annotate patches with env probes
    enable_repair: bool = True        # C: verify-and-repair loop
    max_repair_rounds: int = 1
    top_k: int = 5


@dataclass
class ResolveTrace:
    answer: str
    outcome: ActionOutcome
    injected: str
    n_patches: int
    n_grounded_true: int
    repair_rounds: int
    mode: str


async def resolve_and_solve(task: dict, memory: PatchMemory, solver: Solver,
                            verifier: Optional[Verifier],
                            cfg: ResolveConfig) -> ResolveTrace:
    """One task of the chain protocol.

    A: solve with no memory.
    B: retrieve patches, inject them (plain), solve once.
    C: retrieve patches, GROUND them (additive), inject grounded block, solve;
       then optionally verify-and-repair using the environment outcome.
    """
    query = task.get("description", task.get("task_id", ""))
    chain_id = task.get("chain_id")

    if cfg.mode == "A":
        ans, outcome = await solver(task, "")
        return ResolveTrace(ans, outcome, "", 0, 0, 0, "A")

    candidates = memory.retrieve(query, top_k=cfg.top_k, chain_id=chain_id)

    if cfg.mode == "B":
        injected = render_patches_plain(candidates)
        ans, outcome = await solver(task, injected)
        return ResolveTrace(ans, outcome, injected, len(candidates), 0, 0, "B")

    # ── Method C ──
    if cfg.enable_grounding and verifier is not None:
        grounded = await ground(candidates, verifier, {"task": task})
        injected = render_patches_grounded(grounded)
    else:
        grounded = [GroundedPatch(p) for p in candidates]
        injected = render_patches_plain(candidates)

    ans, outcome = await solver(task, injected)
    rounds = 0
    if cfg.enable_repair:
        while (not outcome.passed) and rounds < cfg.max_repair_rounds:
            hint = repair_hint(outcome, grounded)
            if not hint:
                break
            rounds += 1
            ans, outcome = await solver(task, injected + "\n\n" + hint)

    n_true = sum(1 for g in grounded
                 if g.verification and g.verification.applies is True)
    return ResolveTrace(ans, outcome, injected, len(candidates), n_true,
                        rounds, "C")
