"""Runner-level A/B/C memory bridge — the paper's three arms, made uniform.

  A  Vanilla  : no memory.
  B  PatchMem : naive cross-task patch memory — accumulate a patch per finished
                task (its raw response), retrieve the lexically-relevant ones and
                inject them verbatim. This is the \\patchmem baseline.
  C  Curator  : the real curation pipeline (src.latest.SkillForgeLatest). Each
                finished task becomes an Experience that is REFINED by an LLM
                reviewer (causal lesson, generalized steps, avoidance note,
                transferability) and scored by a cross-agent critic that forces
                enrichment of weak entries (never discards). Retrieval is
                effectiveness-weighted (score = sim x w_c). So C injects refined,
                reusable lessons — not B's raw answers.

Both B and C use GLOBAL cross-task retrieval when a benchmark has no chain
structure (GAIA/GAIA2/Terminal-Bench), and stay scoped to a real shared chain
(LoCoMo sessions). The patch is recorded by the runner AFTER evaluation, with
the task's real score, so C's effectiveness weighting and the critic see the
true outcome.
"""
from __future__ import annotations

import asyncio
import re
import threading

from scripts.latest._vgr import Patch, PatchMemory, render_patches_plain

# Pollution guard for global cross-task retrieval: a patch must share at least
# this much CONTENT (stopword-filtered Jaccard) with the task, else it's
# unrelated noise and is not injected. Content-filtered (not raw token overlap)
# because raw overlap counts stopwords — "of/the/in" alone clears a naive floor.
_SIM_FLOOR = 0.05
_STOP = frozenset(
    "the a an to and or in on at for of with is are was were be been that this it "
    "i me my we our you your he she they them his her its as by from into over "
    "what which who when where why how do does did done has have had will would can "
    "could should may might must not no yes if then than so such".split())


def _content_overlap(query: str, doc: str) -> float:
    """Stopword-filtered Jaccard between a task and a patch document."""
    qa = {t for t in re.findall(r"[a-z0-9]+", (query or "").lower())
          if t not in _STOP and len(t) > 1}
    db = {t for t in re.findall(r"[a-z0-9]+", (doc or "").lower())
          if t not in _STOP and len(t) > 1}
    if not qa or not db:
        return 0.0
    return len(qa & db) / len(qa | db)


def _chain_id(task: dict) -> str:
    meta = task.get("metadata") or {}
    return (task.get("chain_id") or meta.get("chain_id")
            or meta.get("chain") or task.get("task_id", ""))


def _key(task: dict) -> str:
    meta = task.get("metadata") or {}
    return str(meta.get("category") or meta.get("config")
               or meta.get("level") or "general")


def _actions_from_result(result: dict) -> list[dict]:
    """Best-effort agent action trace for analyze_execution. Uses the runner's
    structured `actions` (GAIA2 ARE) when present, else a single synthetic
    action carrying the final answer so the experience has content."""
    acts = result.get("actions")
    if isinstance(acts, list) and acts:
        return acts
    resp = (result.get("response") or "")
    return [{"tool": "final_answer", "output": resp[:200]}] if resp else []


def _oracle_from_task(task: dict) -> list[dict]:
    """Oracle action trace when the benchmark provides one (GAIA2 events);
    empty otherwise — the real score is supplied separately to record()."""
    exp = task.get("expected")
    if isinstance(exp, list):
        return exp
    oa = task.get("oracle_actions")
    return oa if isinstance(oa, list) else []


# ── C (CuratedMemory) helpers — curate CONCRETE experiences ──────────────────
# C's relevance gate is stricter than B's: precision matters more than recall
# because an injected-but-irrelevant "lesson" actively misled the agent (the
# first real run had C lose to B because retrieval was boilerplate-polluted and
# the refined steps were [PLACEHOLDER] templates).
_C_SIM_FLOOR = 0.08


def _core_task(desc: str) -> str:
    """Strip benchmark boilerplate so similarity reflects the actual question,
    not the shared wrapper. Every GAIA task starts 'Answer the following
    question accurately. Question: ...' — without stripping it, every task looks
    similar to every other and retrieval returns random experiences."""
    d = (desc or "").strip()
    parts = re.split(r"(?i)\bquestion\s*:\s*", d)
    core = parts[-1] if len(parts) > 1 else d
    return core.strip()


def _is_weak_lesson(lesson: str) -> bool:
    """A refined lesson with no actionable content — skip it rather than inject
    an empty or tautological 'Key strategy:'."""
    l = (lesson or "").strip().lower()
    if len(l) < 15:
        return True
    weak = ("it worked", "completed all steps", "completed successfully", "success",
            "the task was completed", "no specific", "not applicable", "n/a", "none")
    return any(l == w or l.startswith(w) for w in weak)


def _concrete_approach(exp) -> str:
    """The CONCRETE thing that worked (agent reasoning / commands) — never the
    [PLACEHOLDER]-templated generalized_steps, which carry no usable specifics."""
    rt = getattr(exp, "reasoning_trace", None)
    if rt:
        s = " ".join(str(x) for x in rt).strip()
        if s:
            return s[:500]
    cmds = getattr(exp, "action_commands", None)
    if cmds:
        return " ".join(str(c) for c in cmds).strip()[:500]
    return ""


def _format_curated(successes: list, failures: list = ()) -> str:
    """Inject concrete, relevance-gated, de-duplicated successful approaches plus
    the refined lesson WHEN genuinely useful; and, for prior attempts that
    FAILED on this chain, the refined avoidance note (what to not repeat) — so C
    still guides the agent on a hard chain whose earlier iterations all failed,
    instead of going silent. No empty fields, no [PLACEHOLDER] templates."""
    blocks, seen = [], set()
    for e in successes:
        concrete = _concrete_approach(e)
        key = (concrete[:80] or _core_task(e.task_desc)[:80]).lower()
        if not concrete or key in seen:
            continue
        seen.add(key)
        parts = [f"[✓ Similar task solved (reliability {getattr(e, 'score', 0.0):.0%})]",
                 f"Task: {_core_task(e.task_desc)[:200]}",
                 f"What worked: {concrete}"]
        lesson = ((e.failure_taxonomy or {}).get("causal_lesson") or "").strip()
        if lesson and not _is_weak_lesson(lesson):
            parts.append(f"Lesson: {lesson}")
        blocks.append("\n".join(parts))
    fail_blocks = []
    for e in failures:
        tax = e.failure_taxonomy or {}
        note = (tax.get("avoidance_note") or tax.get("causal_lesson") or "").strip()
        if not note or _is_weak_lesson(note):
            continue
        key = ("avoid:" + note[:80]).lower()
        if key in seen:
            continue
        seen.add(key)
        fail_blocks.append(f"[✗ Earlier attempt fell short]\n"
                           f"Task: {_core_task(e.task_desc)[:200]}\nAvoid: {note}")
    if not blocks and not fail_blocks:
        return ""
    out = ""
    if blocks:
        out += "## Relevant past solutions (curated from similar solved tasks)\n\n" + "\n\n".join(blocks)
    if fail_blocks:
        out += ("\n\n" if out else "") + "## What to avoid (from earlier attempts)\n\n" + "\n\n".join(fail_blocks)
    return out


class BenchmarkMemory:
    """B — naive cross-task patch memory (the \\patchmem baseline)."""

    def __init__(self, benchmark: str, mode: str = "B", top_k: int = 3) -> None:
        self.benchmark = benchmark
        self.mode = mode
        self.top_k = top_k
        self._mem = PatchMemory()
        self._n = 0
        self._lock = threading.Lock()

    def inject(self, task: dict) -> str:
        query = task.get("description", task.get("task_id", ""))
        # CHAIN-SCOPED retrieval: patch memory is feedback across iterations of
        # the SAME task (a chain), not transfer between unrelated tasks. We scope
        # to the chain (metadata chain_id, e.g. a LoCoMo session, else the
        # task_id so a task's later iterations see its earlier ones). On a
        # single-pass run of independent tasks nothing is in-chain yet, so B
        # honestly injects nothing there — the value appears under iteration
        # chains (ITER_CHAIN>1).
        cand = self._mem.retrieve(query, top_k=self.top_k * 4,
                                  chain_id=_chain_id(task))
        cand = [p for p in cand if _content_overlap(query, p._doc()) >= _SIM_FLOOR]
        if not cand:
            return ""
        return render_patches_plain(cand[:self.top_k])

    async def record(self, task: dict, result: dict, score: float | None = None) -> None:
        """Append a raw patch from a finished task (B injects verbatim, no
        refinement). Async for a uniform interface with CuratedMemory."""
        resp = (result.get("response") or "").strip()
        if not resp:
            return
        with self._lock:
            self._n += 1
            n = self._n
        self._mem.add(Patch(
            patch_id=f"{self.benchmark}_B_{n}",
            chain_id=_chain_id(task),
            version=n, key=_key(task),
            summary=resp.splitlines()[0][:120],
            content_before="", content_after=resp[:400],
            rationale="prior task in this benchmark",
            evidence=(task.get("description", "") or "")[:200],
            is_negative=(score is not None and float(score) < 0.5),
        ))

    def __len__(self) -> int:
        return len(self._mem)


class CuratedMemory:
    """C — the real CuratorMem pipeline wrapping src.latest.SkillForgeLatest.

    inject(): effectiveness-weighted retrieval of REFINED experiences
        (get_augmentation -> build_augmented_prompt -> retrieve_similar with
        score = sim x w_c). LLM-free and fast.
    record(): analyze -> LLM refine (causal lesson / generalized steps) ->
        cross-agent critic -> forced enrichment -> library. The refine + critic
        are blocking LLM calls, so they run in a worker thread (asyncio.to_thread)
        to keep the event loop free. Recorded with the task's real score."""

    def __init__(self, benchmark: str, top_k: int = 3,
                 use_critic: bool = True, use_enrich: bool = True) -> None:
        self.benchmark = benchmark
        self.top_k = top_k
        # Curation-stage toggles (for the ablation): refinement is always on;
        # use_critic adds the cross-agent critic score; use_enrich adds the forced
        # enrichment of weak patches. Full \method{} = both True.
        self.use_critic = use_critic
        self.use_enrich = use_enrich
        # Lazy imports: keep module import light (B doesn't need src.latest deps
        # or the LLM client / SDK). Resolved once, on first C construction.
        from src.latest import SkillForgeLatest
        from scripts.latest.llm_client import llm_review_fn
        self._sf = SkillForgeLatest()
        self._llm = llm_review_fn
        # exp task_id -> its chain id, so retrieval can be scoped to the chain
        # (patch memory = same-task iterations, not cross-task transfer).
        self._chain_of: dict[str, str] = {}

    def inject(self, task: dict) -> str:
        # Retrieve on the CLEANED question (boilerplate stripped) so similarity
        # reflects real content, effectiveness-weighted via retrieve_similar.
        core = _core_task(task.get("description", task.get("task_id", "")))
        if not core:
            return ""
        chain = _chain_id(task)
        try:
            # No outcome filter here: we want both prior successes (to copy) and
            # prior failures (to avoid) from this chain.
            cands = self._sf.library.retrieve_similar(core, top_k=self.top_k * 6)
        except Exception:
            return ""
        # CHAIN-SCOPED: keep only experiences from the SAME chain (this task's
        # earlier iterations, or same LoCoMo session) — patch memory is not
        # cross-task transfer. Then a relevance gate as extra hygiene within
        # multi-task chains. On a single-pass independent-task run nothing is
        # in-chain, so C honestly injects nothing (value appears under chains).
        cands = [e for e in cands if self._chain_of.get(e.task_id) == chain]
        cands = [e for e in cands
                 if _content_overlap(core, _core_task(e.task_desc)) >= _C_SIM_FLOOR]
        succ = [e for e in cands if getattr(e, "outcome", "") == "success"][:self.top_k]
        fail = [e for e in cands if getattr(e, "outcome", "") in ("failure", "partial")]
        fail = fail[:max(1, self.top_k - 1)]
        return _format_curated(succ, fail)

    async def record(self, task: dict, result: dict, score: float | None = None) -> None:
        resp = (result.get("response") or "").strip()
        if not resp:
            return
        actions = _actions_from_result(result)
        oracle = _oracle_from_task(task)
        rtrace = result.get("reasoning_trace") or [resp[:1000]]
        # Store the cleaned question as task_desc so retrieval similarity (and the
        # refiner's prompt) see the real question, not the shared boilerplate.
        core = _core_task(task.get("description", "")) or task.get("description", "")
        try:
            # record_experience is sync and makes 2-3 blocking LLM calls
            # (refine + critic [+ enrich]); offload so the loop stays responsive.
            await asyncio.to_thread(
                self._sf.record_experience,
                task.get("task_id", ""), core,
                actions, oracle,
                augmentation_used=result.get("_aug_prompt", ""),
                reasoning_trace=rtrace,
                score=(None if score is None else float(score)),
                llm_reviewer=self._llm,
                critic_fn=(self._llm if self.use_critic else None),
                enrich=self.use_enrich,
            )
            # Remember which chain this experience belongs to (for chain-scoped
            # retrieval). Same-task iterations share a task_id; LoCoMo shares a
            # session chain_id.
            self._chain_of[task.get("task_id", "")] = _chain_id(task)
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._sf.library.experiences)


async def solve_with_memory(run_fn, task: dict, mem, group: str) -> dict:
    """Inject memory → run the baseline agent. Used for B and C. The patch is
    recorded by the runner AFTER evaluation (mem.record(task, r, score)), once
    the real score is known. `run_fn(task, experience, group)` is the
    benchmark's baseline runner."""
    injected = mem.inject(task)
    r = await run_fn(task, injected, group)
    if isinstance(r, dict):
        r["_aug_prompt"] = injected
    return r
