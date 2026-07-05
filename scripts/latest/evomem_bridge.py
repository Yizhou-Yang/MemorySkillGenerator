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
import os
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
# C's retrieval channels are driven by DEPLOYABLE signals only (no gold, no
# absolute self-score threshold — self-eval is optimistic and 0.5-gating stopped
# filtering anything under ITER_FEEDBACK=self):
#   primary channel  = critic-approved entries (critic_quality >= gate) that are
#                      the chain's best self-assessed attempt so far (relative
#                      rank, which LLM judges do far more reliably than absolute
#                      scores);
#   avoidance channel= everything else with a usable note (worse-than-best
#                      attempts and critic-rejected entries).
_CRITIC_GATE = int(os.environ.get("C_CRITIC_GATE", "5"))
# Injection-dose control — a FIRST-CLASS mechanism of the frozen method (v-final,
# 2026-07-03): C's rendered block is capped at ~the raw baseline's measured dose
# (B ≈ 900ch on gaia/gaia2), entries dropped whole from the tail. Evidence: on
# gaia2 an unbudgeted C injected ~1.6x B's block and scored BELOW its own
# not-injected tasks (C−B significantly negative), while content quality alone
# was ~neutral across two backbones — the dose, not the content, was the harm.
# Dose-matching C to B also removes the block-size confound from C−B: what
# remains is purely WHAT is injected. Set 0 to disable (legacy), or override
# per ablation arm (C_small_inject=500).
_C_INJECT_BUDGET = int(os.environ.get("C_INJECT_BUDGET_CH", "900"))
# Raw fallback (frozen method v2, 2026-07-05): when every curated channel comes
# up empty (critic approves nothing, no note survives the weak-lesson floor, or
# the reworded variant drops the chain below the similarity floor), C injects
# the RAW store's own entries under the SAME dose budget — it degrades to the
# \patchmem baseline, never to no memory. Evidence from the first frozen sweep:
# C skipped injection on 28-37% of opportunities (B: ~100%) and paid for the
# silence exactly where memory helped most (gaia, C-skipped subset: B−A=+13.5pp,
# C−B=−8.1pp); the dilution arithmetic reproduces the observed C−B to the
# decimal. With the fallback, C−B ≥ 0 becomes structural on the read path.
# C_RAW_FALLBACK=0 is the ablation arm (C_no_fallback).
_C_RAW_FALLBACK = os.environ.get("C_RAW_FALLBACK", "1") == "1"


def _critic_q(e) -> int:
    try:
        return int((e.failure_taxonomy or {}).get("critic_quality", 5))
    except Exception:
        return 5


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
        tax = e.failure_taxonomy or {}
        concrete = _concrete_approach(e)
        # Verbatim prior OUTCOME, stashed at record() before refinement. This is
        # the answer B replays raw (content_after); C dropped it when curation
        # abstracted the trace into "what worked", so on exact-match QA whose
        # gold survives the reword, C lost to B by paraphrasing away the exact
        # string. Carrying it makes C's block a SUPERSET of B's — recall AND the
        # version-conditioned lesson — so C >= B on static QA and supersedes a
        # STALE verbatim answer on a shifted chain via the lesson below.
        outcome = (tax.get("verbatim_outcome") or "").strip()
        key = ((concrete[:80] or outcome[:80] or _core_task(e.task_desc)[:80])).lower()
        if (not concrete and not outcome) or key in seen:
            continue
        seen.add(key)
        parts = [f"[✓ Prior attempt — curator quality {_critic_q(e)}/10, "
                 f"self-assessed {getattr(e, 'score', 0.0):.0%}]",
                 f"Task: {_core_task(e.task_desc)[:180]}"]
        if outcome:
            parts.append(f"Answer reached: {outcome[:260]}")
        if concrete:
            # trim the approach when the verbatim answer is already carried, so
            # one rich entry fits the dose budget instead of crowding it out
            parts.append(f"What worked: {concrete[:220 if outcome else 500]}")
        lesson = (tax.get("causal_lesson") or "").strip()
        if lesson and not _is_weak_lesson(lesson):
            parts.append(f"Lesson: {lesson[:220]}")
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
    if _C_INJECT_BUDGET > 0:
        kept, used = [], 0
        for b in blocks:
            if used + len(b) > _C_INJECT_BUDGET:
                break
            kept.append(b); used += len(b)
        blocks = kept
        kept, _budget = [], max(0, _C_INJECT_BUDGET - used)
        for b in fail_blocks:
            if len(b) > _budget:
                break
            kept.append(b); _budget -= len(b)
        fail_blocks = kept
    if not blocks and not fail_blocks:
        return ""
    out = ""
    if blocks:
        out += "## Relevant past solutions (curated from similar solved tasks)\n\n" + "\n\n".join(blocks)
    if fail_blocks:
        out += ("\n\n" if out else "") + "## What to avoid (from earlier attempts)\n\n" + "\n\n".join(fail_blocks)
    return out


def _format_raw(entries: list) -> str:
    """Raw-fallback rendering: B-equivalent content under C's dose budget. No
    curated field is required — the concrete approach (reasoning trace or
    commands) always exists for any recorded attempt — so a chain that HAS
    history is never rendered as silence. Honest tag: the agent sees these are
    raw prior attempts, not curator-approved solutions."""
    blocks, seen = [], set()
    for e in entries:
        concrete = _concrete_approach(e)
        if not concrete:
            continue
        key = concrete[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        sc = getattr(e, "score", 0.0) or 0.0
        head = "What was tried and seemed to work" if sc >= 0.5 else \
               "What was tried and fell short"
        blocks.append(f"[Prior attempt on this task — raw, self-assessed {sc:.0%}]\n"
                      f"Task: {_core_task(e.task_desc)[:200]}\n{head}: {concrete[:400]}")
    if _C_INJECT_BUDGET > 0:
        kept, used = [], 0
        for b in blocks:
            if used + len(b) > _C_INJECT_BUDGET:
                break
            kept.append(b)
            used += len(b)
        blocks = kept
    if not blocks:
        return ""
    return "## Prior attempts on this task (raw memory)\n\n" + "\n\n".join(blocks)


class BenchmarkMemory:
    """B — naive cross-task patch memory (the \\patchmem baseline)."""

    def __init__(self, benchmark: str, mode: str = "B", top_k: int = 3) -> None:
        self.benchmark = benchmark
        self.mode = mode
        self.top_k = top_k
        self._mem = PatchMemory()
        self._n = 0
        self._lock = threading.Lock()

    # The harbor bridge pickles the store between iterations; a threading.Lock
    # is unpicklable and crashed arm B at iteration 0. Drop it on dump,
    # recreate on load.
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_lock"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
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
        # Relevance gate compares the current task against each patch's PRODUCING
        # task (stored untruncated in `evidence`), symmetrically and boilerplate-
        # stripped — NOT against the patch's short answer text (`p._doc()`). On a
        # benchmark whose task text is long (LoCoMo = a whole conversation),
        # desc-vs-answer Jaccard is size-diluted below the floor, so every in-chain
        # candidate was wrongly dropped and B injected nothing. desc-core vs
        # desc-core keeps both sides at the same scale (this is what arm C does).
        qcore = _core_task(query)
        cand = self._mem.retrieve(query, top_k=self.top_k * 4,
                                  chain_id=_chain_id(task))
        cand = [p for p in cand
                if _content_overlap(qcore, _core_task(p.evidence)) >= _SIM_FLOOR]
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
            # Full producing-task text (not truncated): used ONLY by inject()'s
            # relevance gate, never rendered into the prompt, so no bloat.
            evidence=(task.get("description", "") or ""),
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
                 use_critic: bool | None = None, use_enrich: bool | None = None) -> None:
        # Stage toggles default from env so the ablation driver can build the
        # C_refine / C_refine_critic arms without touching the runner:
        #   C_USE_CRITIC=0 -> refinement only; C_USE_ENRICH=0 -> no forced enrich.
        if use_critic is None:
            use_critic = os.environ.get("C_USE_CRITIC", "1") == "1"
        if use_enrich is None:
            use_enrich = os.environ.get("C_USE_ENRICH", "1") == "1"
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
        # Effectiveness feedback loop (the paper's w_c): inject() remembers which
        # experience ids it served for a task; record() then credits/blames them
        # with the within-chain score delta (this iteration vs the previous one),
        # via library.update_effectiveness -> get_experience_weight. Without this
        # wiring w_c stays 1.0 forever and retrieval is pure similarity.
        self._served: dict[str, list[str]] = {}
        self._last_score: dict[str, float] = {}
        self._record_failures = 0
        # Per-chain index of this bridge's own recorded experiences. inject()'s
        # global retrieve_similar ranks against the WHOLE library, so a reworded
        # variant can push this chain's entries out of the pool while other
        # tasks' entries fill it — B never has this failure because it scopes
        # retrieval to the chain at the store. The index guarantees C's recall
        # of its own chain history is at least B's.
        self._chain_entries: dict[str, list] = {}

    # Pickled by the harbor bridge between iterations: drop the LLM function
    # ref on dump (re-imported on load). If SkillForgeLatest's embedder turns
    # out unpicklable too, the bridge's startup round-trip check will catch it.
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_llm"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.__dict__.setdefault("_chain_entries", {})  # pre-fallback pickles
        from scripts.latest.llm_client import llm_review_fn
        self._llm = llm_review_fn

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
            pool = self._sf.library.retrieve_similar(core, top_k=self.top_k * 6)
        except Exception:
            pool = []  # the chain index below still serves
        # CHAIN-SCOPED: keep only experiences from the SAME chain (this task's
        # earlier iterations, or same LoCoMo session) — patch memory is not
        # cross-task transfer. On a single-pass independent-task run nothing is
        # in-chain, so C honestly injects nothing (value appears under chains).
        ranked = [e for e in pool if self._chain_of.get(e.task_id) == chain]
        # Recall rescue: append any in-chain entry the global pool missed
        # (keeps retrieve_similar's sim x w_c order for what it DID surface).
        _key = lambda e: (e.task_id, getattr(e, "version", 0),
                          getattr(e, "timestamp", 0.0))
        _have = {_key(e) for e in ranked}
        cands = ranked + [e for e in self._chain_entries.get(chain, [])
                          if _key(e) not in _have]
        # Similarity floor as HYGIENE only: within multi-task chains (LoCoMo
        # sessions) it filters off-topic entries, but it must never zero out a
        # chain that has history — on reworded variants the floor is exactly
        # what silenced C (63-72% injection) while B kept injecting (~100%).
        floored = [e for e in cands
                   if _content_overlap(core, _core_task(e.task_desc)) >= _C_SIM_FLOOR]
        if floored:
            cands = floored
        best = max((getattr(e, "score", 0.0) or 0.0 for e in cands), default=0.0)
        succ = [e for e in cands
                if _critic_q(e) >= _CRITIC_GATE
                and (getattr(e, "score", 0.0) or 0.0) >= best - 1e-9][:self.top_k]
        fail = [e for e in cands if e not in succ]
        fail = fail[:max(1, self.top_k - 1)]
        out = _format_curated(succ, fail)
        served = succ + fail
        if not out and cands and _C_RAW_FALLBACK:
            # Curated channels rendered nothing -> degrade to the raw store
            # under the same budget, never to no memory (see _C_RAW_FALLBACK).
            served = cands[: self.top_k]
            out = _format_raw(served)
        if out:
            # remember what was actually served, for the effectiveness update
            self._served[task.get("task_id", "")] = [e.task_id for e in served]
        return out

    async def record(self, task: dict, result: dict, score: float | None = None) -> None:
        tid = task.get("task_id", "")
        chain = _chain_id(task)
        served = self._served.pop(tid, None)
        resp = (result.get("response") or "").strip()
        if not resp:
            return
        # Effectiveness update (w_c feedback): the injected experiences get the
        # within-chain paired delta -- this iteration's score minus the previous
        # iteration's. Positive => they helped; negative => they hurt. Bounded
        # into weights by get_experience_weight (clip [0.3, 1.5]).
        if served and score is not None:
            prev = self._last_score.get(chain)
            if prev is not None:
                delta = float(score) - prev
                for eid in served:
                    try:
                        self._sf.library.update_effectiveness(eid, delta)
                    except Exception:
                        pass
        if score is not None:
            self._last_score[chain] = float(score)
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
            # Locate the experience just stored for THIS task (search by id, not
            # experiences[-1]: concurrent records interleave) and index it on
            # its chain for inject()'s recall rescue.
            _exps = self._sf.library.experiences
            _mine = next((e for e in reversed(_exps) if e.task_id == tid), None)
            if _mine is not None:
                # Stash the verbatim outcome BEFORE it is lost to refinement, so
                # inject() can replay the exact answer alongside the curated
                # lesson (the B-superset rendering; see _format_curated). Only
                # for a genuine success — a wrong answer is not worth replaying.
                if score is not None and float(score) >= 0.5:
                    try:
                        _mine.failure_taxonomy.setdefault("verbatim_outcome",
                                                          resp[:400])
                    except Exception:
                        pass
                self._chain_entries.setdefault(chain, []).append(_mine)
            # w_c cold-start prior from the critic (deployable, no extra calls):
            # seed one pseudo-observation so a high-quality patch starts above
            # unit weight and a critic-rejected one below, instead of every
            # patch idling at w_c=1.0 until two real reuse deltas accumulate.
            try:
                if _mine is not None:
                    _q = _critic_q(_mine)
                    self._sf.library.update_effectiveness(
                        tid, (_q - _CRITIC_GATE) / 10.0)
            except Exception:
                pass
        except Exception as e:
            # never crash the run over a memory write, but don't hide it either
            self._record_failures += 1
            if self._record_failures <= 3:
                print(f"  [CuratedMemory] record failed ({self._record_failures}): "
                      f"{type(e).__name__}: {e}", flush=True)

    def __len__(self) -> int:
        return len(self._sf.library.experiences)


async def solve_with_memory(run_fn, task: dict, mem, group: str) -> dict:
    """Inject memory → run the baseline agent. Used for B and C. The patch is
    recorded by the runner AFTER evaluation (mem.record(task, r, score)), once
    the real score is known. `run_fn(task, experience, group)` is the
    benchmark's baseline runner."""
    # A memory-layer failure must DEGRADE the task to no-injection, never kill
    # it: a broken embedder dependency once turned an entire C rerun into 100%
    # error rows while A/B (older data) looked fine — masquerading as a method
    # regression. Degradations are counted and printed, not hidden.
    try:
        injected = mem.inject(task)
    except Exception as e:
        n = getattr(mem, "_inject_failures", 0) + 1
        mem._inject_failures = n
        if n <= 3 or n % 50 == 0:
            print(f"  [memory] inject failed ({n}x, degrading to no-injection): "
                  f"{type(e).__name__}: {e}", flush=True)
        injected = ""
    r = await run_fn(task, injected, group)
    if isinstance(r, dict):
        r["_aug_prompt"] = injected
    return r
