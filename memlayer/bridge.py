"""Runner-level A/B/C memory bridge — the paper's three arms, made uniform.

  A Vanilla: no memory. B PatchMem: raw per-task patches injected verbatim.
  C Curator: LLM-refined + critic-scored experiences, retrieval weighted sim x w_c.

The runner records the patch AFTER evaluation with the task's real score, so the
critic and effectiveness weighting see the true outcome.
"""
from __future__ import annotations

import asyncio
import os
import re
import threading

from .vgr import Patch, PatchMemory, render_patches_plain

# Minimum stopword-filtered Jaccard for a patch to be injected. Stopwords must
# stay filtered: raw token overlap clears a naive floor on "of/the/in" alone.
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
    """Best-effort agent action trace for analyze_execution: the runner's
    structured `actions` when present, else one synthetic final-answer action."""
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
# Stricter than B's floor on purpose: an injected-but-irrelevant "lesson"
# actively misleads the agent, so precision beats recall here.
_C_SIM_FLOOR = 0.08
_CRITIC_GATE = int(os.environ.get("C_CRITIC_GATE", "5"))
_C_META = os.environ.get("C_META", "0") == "1"
_SCORE_PROVENANCE = ("self_assessment" if os.environ.get("ITER_FEEDBACK", "gold") == "self"
                     else os.environ.get("ITER_FEEDBACK", "gold"))
# w_c may only be read when its deltas came from a real scorer; a self-assessed
# delta is an opinion and must never be laundered into evidence.
_WC_IS_GROUNDED = _SCORE_PROVENANCE in ("gold", "env")
# How arm C decides what to endorse:
#   judgment — critic score + actor self-assessment gate the ✓ (legacy).
#   meta     — selection/rendering use only store metadata; nothing endorsed.
#   guarded  — a ✓ needs a SECOND independent key (grounded score or EXTERNAL
#              critic); ungated entries render neutrally rather than dropping.
_C_POLICY = (os.environ.get("C_POLICY")
             or ("meta" if _C_META else "judgment")).strip().lower()
if _C_POLICY not in ("judgment", "meta", "guarded"):
    _C_POLICY = "judgment"
_C_META = _C_POLICY == "meta"
# A critic equal to the acting backbone is self-judgment in a second hat, not a
# second key — external requires CRITIC_MODEL set AND different.
# CRITIC_MODEL_ID is what llm_client and the runners set; accept both so the
# second key is not silently absent because two names drifted apart.
_CRITIC_RAW = (os.environ.get("CRITIC_MODEL")
               or os.environ.get("CRITIC_MODEL_ID") or "").strip().lower()
_BACKBONE_ID = (os.environ.get("CODEBUDDY_MODEL") or "").strip().lower()
_CRITIC_IS_EXTERNAL = bool(_CRITIC_RAW) and _CRITIC_RAW != _BACKBONE_ID
# Injection dose cap in chars — a first-class mechanism, not a safety valve: an
# unbudgeted C block (~1.6x B's) measurably HURT accuracy, and dose-matching C to
# B is what removes the block-size confound from C−B. 0 disables.
_C_INJECT_BUDGET = int(os.environ.get("C_INJECT_BUDGET_CH", "900"))


# Ablation arm C_weak_compact (default off): show retrieval only the newest n
# chain entries. Read dynamically so the driver can set it per-arm without
# reimport.
def _page_keep() -> int:
    return int(os.environ.get("C_PAGE_KEEP", "0"))


# Ablation arm C_no_partition (default off): drop chain pruning and the
# chain-index recall rescue, serving the flat global pool. Read dynamically so
# the driver can set it per-arm without reimport.
def _no_partition() -> bool:
    return os.environ.get("C_NO_PARTITION", "0") == "1"


# When every curated channel renders empty, C falls back to raw store entries
# under the same dose budget — it degrades to the \patchmem baseline, never to
# no memory. C_RAW_FALLBACK=0 is the ablation arm.
_C_RAW_FALLBACK = os.environ.get("C_RAW_FALLBACK", "1") == "1"
# Curation-as-repair: chains whose previous attempt passed the threshold are
# served B's raw rendering untouched.
_C_REPAIR_MODE = os.environ.get("C_REPAIR_MODE", "0") == "1"
_C_REPAIR_THRESH = float(os.environ.get("C_REPAIR_THRESH", "0.6"))
# What decides "the chain is succeeding":
#   stability (default) — last two attempts gave the same normalized answer
#              (zero LLM calls, so the floor never depends on a critic).
#   verdict  — the external critic's outcome_verdict.
#   self     — actor self-assessment (failed; kept as the ablation arm).
_C_REPAIR_GATE = (os.environ.get("C_REPAIR_GATE") or "stability").strip().lower()


def _norm_answer(s: str) -> str:
    """Normalize an answer for stability comparison (case/punctuation/whitespace)."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9. ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:200]


_WC_LOOKUP = {"fn": None}          # set by CuratedPatchMemory once the library exists


def _append_supersession(block: str, pool) -> str:
    """Render 'v1 -> v2 superseded' lines from patch_history, newest last.

    Uses only lineage already in the manifest; silent when a chain records none.
    """
    lines = []
    for e in pool:
        hist = getattr(e, "patch_history", None) or []
        for h in hist:
            fv, tv = h.get("from_version"), h.get("to_version")
            if fv is None or tv is None:
                continue
            why = h.get("fixed_missing") or (
                f"score {h.get('score_delta', 0):+.0%}" if h.get("score_delta") else "")
            lines.append(f"  v{fv} -> v{tv}" + (f": {why}" if why else ""))
    if not lines:
        return block
    seen, uniq = set(), []
    for l in lines:
        if l not in seen:
            seen.add(l); uniq.append(l)
    # This runs AFTER _format_curated has spent the whole budget, and the gate
    # hard-fails any C block over BUDGET+130 — so keep the append inside 100 chars.
    header = "\n\nVersion lineage (later supersedes earlier):"
    room = _C_INJECT_BUDGET + 100 - len(block) - len(header)
    kept, used = [], 0
    for l in uniq[:6]:
        if used + len(l) + 1 > room:
            break
        kept.append(l); used += len(l) + 1
    if not kept:
        return block
    return block + header + "\n" + "\n".join(kept)


def _wc_of(e):
    """Measured effectiveness of an entry, or None before the library is wired."""
    fn = _WC_LOOKUP.get("fn")
    if fn is None:
        return None
    try:
        return float(fn(e.task_id))
    except Exception:
        return None


_DEMOTE_K = int(os.environ.get("C_ENDORSE_DEMOTE_K", "2") or 2)


def _grounded_demoted(e) -> bool:
    """Endorsement demotion: True when the K most recent GROUNDED reuse deltas
    (provenance env/gold only) are all negative. Self-assessed deltas can
    neither endorse nor demote. K = C_ENDORSE_DEMOTE_K (default 2); 0 disables."""
    if _DEMOTE_K <= 0:
        return False
    ds = [float(d.get("delta", 0.0) or 0.0)
          for d in (getattr(e, "sys_stats", None) or {}).get("reuse_deltas", [])
          if d.get("provenance") in ("env", "gold")]
    if len(ds) < _DEMOTE_K:
        return False
    return all(d < 0.0 for d in ds[-_DEMOTE_K:])


def _endorse_basis(e):
    """Second key for an endorsement under C_POLICY=guarded, or None.

    ("env", None) = grounded outcome score; ("critic", name) = external critic,
    revocable by _grounded_demoted. Self-assessment alone never endorses.
    """
    if _C_POLICY != "guarded":
        return None
    if _WC_IS_GROUNDED and float(getattr(e, "score", 0.0) or 0.0) >= 0.5:
        return ("env", None)
    if _CRITIC_IS_EXTERNAL and _critic_q(e) >= _CRITIC_GATE \
            and not _grounded_demoted(e):
        return ("critic", _CRITIC_RAW)
    return None


def _critic_q(e) -> int:
    try:
        return int((e.failure_taxonomy or {}).get("critic_quality", 5))
    except Exception:
        return 5


def _core_task(desc: str) -> str:
    """Strip benchmark boilerplate ('...Question: ') so similarity reflects the
    question. Without this every task looks alike and retrieval goes random."""
    d = (desc or "").strip()
    parts = re.split(r"(?i)\bquestion\s*:\s*", d)
    core = parts[-1] if len(parts) > 1 else d
    return core.strip()


def _is_weak_lesson(lesson: str) -> bool:
    """True for a refined lesson with no actionable content (empty/tautological)."""
    l = (lesson or "").strip().lower()
    if len(l) < 15:
        return True
    weak = ("it worked", "completed all steps", "completed successfully", "success",
            "the task was completed", "no specific", "not applicable", "n/a", "none")
    return any(l == w or l.startswith(w) for w in weak)


def _concrete_approach(exp) -> str:
    """The CONCRETE thing that worked (reasoning / commands). Never use
    generalized_steps here — it is [PLACEHOLDER]-templated, with no specifics."""
    rt = getattr(exp, "reasoning_trace", None)
    if rt:
        s = " ".join(str(x) for x in rt).strip()
        if s:
            return s[:500]
    cmds = getattr(exp, "action_commands", None)
    if cmds:
        return " ".join(str(c) for c in cmds).strip()[:500]
    return ""


# Benchmarks scored on WHICH ACTIONS OCCURRED, where the tool sequence is worth
# displacing prose for. Scope by benchmark, never by entry shape: gaia/locomo
# agents also log tool calls but are scored on ANSWER text, and an
# entry-has-actions heuristic there cost ~7pp by dropping the prose.
_ACTION_SCORED = {"gaia2", "terminal_bench_2", "tau2"}


def _format_curated(successes: list, failures: list = (),
                    current_tid: str = "", action_scored: bool = False) -> str:
    """Layered rendering: every prior attempt (verbatim) is the guaranteed base;
    Lesson/Avoid annotations are a bonus layer added only while budget allows.
    Annotations must never displace an attempt — field caps adapt instead."""
    def _payload(e):
        tax = e.failure_taxonomy or {}
        return ((tax.get("verbatim_outcome") or "").strip(),
                _concrete_approach(e))

    n_est = sum(1 for e in list(successes) + list(failures)
                if any(_payload(e)))
    vcap = 400 if n_est <= 1 else (300 if n_est == 2 else 180)
    tcap = 150 if n_est <= 1 else 90
    pairs, seen, n_succ = [], set(), 0   # (lean_block, annotation) in order
    for e in successes:
        tax = e.failure_taxonomy or {}
        outcome, concrete = _payload(e)
        key = ((concrete[:80] or outcome[:80] or _core_task(e.task_desc)[:80])).lower()
        if (not concrete and not outcome) or key in seen:
            continue
        seen.add(key)
        if _C_POLICY == "judgment":
            # Repair mode drops the self-assessment: "self-assessed 2%" under a ✓
            # is two contradictory signals in one line.
            head =(f"[✓ Prior attempt — curator quality {_critic_q(e)}/10]"
                    if _C_REPAIR_MODE else
                    f"[✓ Prior attempt — curator quality {_critic_q(e)}/10, "
                    f"self-assessed {getattr(e, 'score', 0.0):.0%}]")
            parts = [head, f"Task: {_core_task(e.task_desc)[:tcap]}"]
        else:
            # meta/guarded never print the self-assessment; a ✓ needs a second key
            # and the line names it, so the reader can tell measurement from review.
            _ver =int(getattr(e, "version", 1) or 1)
            _basis = _endorse_basis(e)
            if _basis is not None:
                _src = ("environment-verified outcome" if _basis[0] == "env"
                        else f"reviewed by {_basis[1]}")
                head = f"[✓ Prior attempt v{_ver} — {_src}]"
            else:
                head = f"[Prior attempt — store version v{_ver}]"
            parts = [head, f"Task: {_core_task(e.task_desc)[:tcap]}"]
        acts = ([str(c) for c in (getattr(e, "action_commands", None) or [])]
                or [str(s) for s in (getattr(e, "tool_sequence", None) or [])]) \
            if action_scored else []
        if action_scored:
            # gaia2/TB2: scored on which actions occurred — action list is the
            # payload, answer text secondary (v2.3, trend-positive on gaia2).
            if outcome:
                parts.append(f"Answer given then (unverified): {outcome[:160]}")
            if acts:
                parts.append("Actions used: " + " -> ".join(acts)[:200])
            elif concrete:
                parts.append(f"What worked: {concrete[:200]}")
        elif outcome:
            # Answer-scored QA: the raw attempt VERBATIM — the same resp head
            # B replays, hedges and discrepancy notes intact (truncated only
            # to share the dose budget across ALL attempts of the chain).
            parts.append(f"As recorded: {outcome[:vcap]}")
        elif concrete:
            parts.append(f"What worked: {concrete[:vcap]}")
        lesson = (tax.get("causal_lesson") or "").strip()
        # Asserted content follows the same rule as the ✓: under meta it never
        # enters (pure store); under guarded only with a second key — an
        # unchecked self-judge's lesson is the 48%-of-budget filler the dose
        # analysis flagged.
        _lesson_ok = (_C_POLICY == "judgment"
                      or (_C_POLICY == "guarded" and _endorse_basis(e) is not None))
        annot = (f"\nLesson: {lesson[:180]}"
                 if _lesson_ok and lesson and not _is_weak_lesson(lesson) else "")
        pairs.append(["\n".join(parts), annot])
        n_succ += 1
    for e in failures:
        tax = e.failure_taxonomy or {}
        note = (tax.get("avoidance_note") or tax.get("causal_lesson") or "").strip()
        # Same-task entries carry their verbatim answer here too: self-
        # assessment routes many gold-CORRECT attempts into this channel
        # (LoCoMo: 66% false-failure rate), and dropping their answers is how
        # C lost to a raw baseline that replays everything.
        outcome = (tax.get("verbatim_outcome") or "").strip() \
            if e.task_id == current_tid else ""
        weak = not note or _is_weak_lesson(note)
        if weak and not outcome:
            continue
        # dedup by the actual PAYLOAD: outcome first (distinct attempts can
        # share an identical weak note, which must not merge them)
        key = ("avoid:" + (outcome or note)[:80]).lower()
        if key in seen:
            continue
        seen.add(key)
        parts = [f"[✗ Earlier attempt fell short]",
                 f"Task: {_core_task(e.task_desc)[:tcap]}"]
        if outcome:
            parts.append("As recorded (self-assessed doubtful, verify before "
                         f"reuse): {outcome[:min(150, vcap)]}")
            annot = f"\nAvoid: {note[:180]}" if not weak else ""
        else:
            # no verbatim payload: the avoidance note IS the payload
            parts.append(f"Avoid: {note[:180]}")
            annot = ""
        pairs.append(["\n".join(parts), annot])
    if _C_INJECT_BUDGET > 0:
        # layer 1: keep as many LEAN attempt blocks as fit (attempts first,
        # annotations never displace an attempt)
        kept, used = [], 0
        for lean, annot in pairs:
            if used + len(lean) > _C_INJECT_BUDGET:
                break
            kept.append([lean, annot]); used += len(lean)
        if not kept and pairs:
            # never let one oversized block silence the whole channel
            kept = [[pairs[0][0][:_C_INJECT_BUDGET], ""]]
            used = len(kept[0][0])
        # layer 2: upgrade kept blocks with their annotations while budget lasts
        for kb in kept:
            if kb[1] and used + len(kb[1]) <= _C_INJECT_BUDGET:
                kb[0] += kb[1]; used += len(kb[1])
        rendered = [kb[0] for kb in kept]
    else:
        rendered = [lean + annot for lean, annot in pairs]
    k_succ = min(n_succ, len(rendered))
    blocks, fail_blocks = rendered[:k_succ], rendered[k_succ:]
    if not blocks and not fail_blocks:
        return ""
    out = ""
    if blocks:
        # Header must not overclaim: self-assessment mislabels many entries
        # (gaia: 56% of gold-wrong attempts self-assess >=0.5), so these are
        # "prior attempts", not "solved tasks". TB2 transcript checks grep for
        # the stable "## Curated prior attempts" marker.
        out += "## Curated prior attempts (this task's chain)\n\n" + "\n\n".join(blocks)
    if fail_blocks:
        out += ("\n\n" if out else "") + "## What to avoid (from earlier attempts)\n\n" + "\n\n".join(fail_blocks)
    return out


def _format_raw(entries: list, action_scored: bool = False) -> str:
    """Raw-fallback rendering: B-equivalent content under C's dose budget. No
    curated field is required — the concrete approach (reasoning trace or
    commands) always exists for any recorded attempt — so a chain that HAS
    history is never rendered as silence. Honest tag: the agent sees these are
    raw prior attempts, not curator-approved solutions."""
    blocks, seen = [], set()
    for e in entries:
        concrete = _concrete_approach(e)
        acts = ([str(c) for c in (getattr(e, "action_commands", None) or [])]
                or [str(s) for s in (getattr(e, "tool_sequence", None) or [])]) \
            if action_scored else []
        if not concrete and not acts:
            continue
        key = (concrete or " ".join(acts))[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        sc = getattr(e, "score", 0.0) or 0.0
        head = "What was tried and seemed to work" if sc >= 0.5 else \
               "What was tried and fell short"
        parts = [f"[Prior attempt on this task — raw, self-assessed {sc:.0%}]",
                 f"Task: {_core_task(e.task_desc)[:150]}"]
        if acts:   # agentic: the action sequence is the payload (see curated)
            parts.append("Actions used: " + " -> ".join(acts)[:200])
            if concrete:
                parts.append(f"{head}: {concrete[:200]}")
        else:
            parts.append(f"{head}: {concrete[:400]}")
        blocks.append("\n".join(parts))
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
        from .forge import SkillForgeLatest
        # Prefer the harness LLM client (benchmark routing, SDK paths); fall
        # back to memlayer's standalone env-driven client when installed as an
        # SDK outside this repo.
        try:
            from scripts.latest.llm_client import llm_metadata_fn, llm_critic_fn
        except ImportError:
            from .llm import llm_metadata_fn, llm_critic_fn
        self._sf = SkillForgeLatest()
        # The reviewer pen: HY3 under METADATA_AUTHOR=critic (default — the
        # curated layer's narratives are authored by the fixed external critic,
        # not by the backbone reviewing itself), backbone under =backbone.
        self._llm = llm_metadata_fn
        # The critic routes separately (CRITIC_MODEL); unset -> same as _llm.
        self._critic_llm = llm_critic_fn
        # exp task_id -> its chain id, so retrieval can be scoped to the chain
        # (patch memory = same-task iterations, not cross-task transfer).
        self._chain_of: dict[str, str] = {}
        # Effectiveness feedback loop (the paper's w_c): inject() remembers which
        # experience ids it served for a task; record() then credits/blames them
        # with the within-chain score delta (this iteration vs the previous one),
        # via library.update_effectiveness -> get_experience_weight. Without this
        # wiring w_c stays 1.0 forever and retrieval is pure similarity.
        self._served: dict[str, list[str]] = {}
        # (task_id, version) keys of the same served entries — sys_stats
        # bookkeeping needs the exact version, _served's task_ids do not
        # disambiguate between versions of one task.
        self._served_keys: dict[str, list[tuple]] = {}
        self._last_wc: dict | None = None
        try:
            _WC_LOOKUP["fn"] = self._sf.library.get_experience_weight
        except Exception:
            pass
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
        d["_critic_llm"] = None
        d["_last_wc"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.__dict__.setdefault("_chain_entries", {})  # pre-fallback pickles
        self.__dict__.setdefault("_served_keys", {})    # pre-sys_stats pickles
        self.__dict__.setdefault("_chain_iter", {})     # pre-iter0-ref pickles
        self.__dict__.setdefault("_chain_base", {})
        try:
            from scripts.latest.llm_client import llm_metadata_fn, llm_critic_fn
        except ImportError:
            from .llm import llm_metadata_fn, llm_critic_fn
        self._llm = llm_metadata_fn
        self._critic_llm = llm_critic_fn

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
        _key = lambda e: (e.task_id, getattr(e, "version", 0),
                          getattr(e, "timestamp", 0.0))
        if _no_partition():
            # No-partition foil (ablation only, see _no_partition): serve the
            # flat similarity-ranked pool — no chain pruning, no chain-index
            # rescue. Downstream (floor, same-task sort, channels, budget)
            # is identical, so the arm isolates the partition metadata.
            cands = list(pool)
        else:
            # CHAIN-SCOPED: keep only experiences from the SAME chain (this
            # task's earlier iterations, or same LoCoMo session) — patch memory
            # is not cross-task transfer. On a single-pass independent-task run
            # nothing is in-chain, so C honestly injects nothing (value appears
            # under chains).
            ranked = [e for e in pool if self._chain_of.get(e.task_id) == chain]
            # Recall rescue: append any in-chain entry the global pool missed
            # (keeps retrieve_similar's sim x w_c order for what it DID
            # surface).
            _have = {_key(e) for e in ranked}
            cands = ranked + [e for e in self._chain_entries.get(chain, [])
                              if _key(e) not in _have]
        _pk = _page_keep()
        if _pk > 0:
            # weak compaction (ablation only): page all but the newest n
            # chain entries out of the read path; the store keeps everything
            _allowed = {_key(e) for e in
                        self._chain_entries.get(chain, [])[-_pk:]}
            cands = [e for e in cands if _key(e) in _allowed]
        # Similarity floor as HYGIENE only: within multi-task chains (LoCoMo
        # sessions) it filters off-topic entries, but it must never zero out a
        # chain that has history — on reworded variants the floor is exactly
        # what silenced C (63-72% injection) while B kept injecting (~100%).
        floored = [e for e in cands
                   if _content_overlap(core, _core_task(e.task_desc)) >= _C_SIM_FLOOR]
        if floored:
            cands = floored
        # SAME-TASK FIRST: in a session chain (LoCoMo: one chain = a whole
        # session of questions) the current task's own prior iterations are the
        # direct evidence; other questions' entries are auxiliary context.
        # Stable sort keeps the sim x w_c order within each group, and the
        # chain-best anchor uses own attempts when any exist, so another
        # question's high self-score cannot evict this task's own attempt from
        # the success channel — cross-question replay is precisely the raw
        # baseline's measured failure mode on LoCoMo (B−A = −5.6pp).
        tid_now = task.get("task_id", "")
        cands.sort(key=lambda e: 0 if e.task_id == tid_now else 1)
        own = [e for e in cands if e.task_id == tid_now]
        pool = own if own else cands
        if _C_POLICY != "judgment":
            # Rank on grounded metadata only. Primary key is version lineage:
            # the store recorded that a later version superseded an earlier one,
            # and that record does not depend on anyone's opinion of either.
            # Secondary key is execution evidence — a revision that changed the
            # tool sequence did something; one that only reworded did not. w_c
            # breaks remaining ties, but ONLY when its deltas came from the
            # benchmark's scorer; under self-assessment it is an opinion wearing
            # a number's clothing and is skipped rather than laundered. Nothing
            # is dropped for lacking evidence: absence of evidence is not
            # evidence of harm, and dropping would silently shrink coverage on
            # short chains. Both keys survive deployment — neither needs an
            # oracle or a gold score.
            def _grounded_key(e):
                # w_c deliberately absent: under self-assessment it is opinion,
                # and under this protocol (cold start needs >=2 measured deltas,
                # ITER_CHAIN=3 supplies at most 2 with the second landing on the
                # final iteration) it never influences a decision under gold
                # either. The trace's wc_active field documents that inertness;
                # ranking pretends nothing it cannot back.
                ver = int(getattr(e, "version", 1) or 1)
                hist = getattr(e, "patch_history", None) or []
                moved = sum(len(h.get("new_steps") or [])
                            + len(h.get("removed_steps") or []) for h in hist)
                # Tertiary: measured reuse effect, but ONLY deltas whose score
                # came from the environment/gold scorer (sys_stats provenance
                # filter) — the ANALYZE statistic replacing w_c's laundering
                # problem. No grounded deltas -> 0.0: absence of evidence is
                # not evidence of harm.
                deltas = [d.get("delta", 0.0)
                          for d in (getattr(e, "sys_stats", None) or {}).get("reuse_deltas", [])
                          if d.get("provenance") in ("env", "gold")]
                reuse = sum(deltas) / len(deltas) if deltas else 0.0
                return (-ver, -moved, -reuse)
            if _C_POLICY == "guarded":
                # Endorsed-first, then lineage: judgment proposes, but only
                # entries holding a second key wear the ✓.
                scored = sorted(pool, key=lambda e: (
                    0 if _endorse_basis(e) is not None else 1,) + _grounded_key(e))
            else:
                scored = sorted(pool, key=_grounded_key)
            succ = scored[:self.top_k]
        else:
            best = max((getattr(e, "score", 0.0) or 0.0 for e in pool), default=0.0)
            # Endorsement demotion applies here too: a critic-approved entry
            # whose grounded reuse deltas turned negative drops out of the
            # primary channel (it still renders neutrally / as avoidance).
            succ = [e for e in pool
                    if _critic_q(e) >= _CRITIC_GATE
                    and not _grounded_demoted(e)
                    and (getattr(e, "score", 0.0) or 0.0) >= best - 1e-9][:self.top_k]
        fail = [e for e in cands if e not in succ]
        fail = fail[:max(1, self.top_k - 1)]
        _action_scored = self.benchmark in _ACTION_SCORED
        # Curation as REPAIR, not default (C_REPAIR_MODE=1): when the chain's
        # previous attempt looks successful under the protocol's feedback
        # (>= C_REPAIR_THRESH), serve EXACTLY what B would serve — verbatim
        # raw, no ✓, no lesson, no numbers — and keep the curated treatment
        # for failing chains. Flip evidence across both backbones: every C
        # loss vs B came from disturbing a chain B had already solved
        # (spoiled 7/4/4 vs rescued 7/10/3); not touching succeeding chains
        # removes the spoilage channel by construction.
        _repair_raw = False
        if _C_REPAIR_MODE and cands:
            if _C_REPAIR_GATE == "stability":
                # Meta-driven gate: answer stability across the chain's last
                # two attempts (string-level, zero model calls). One attempt =
                # no drift evidence yet -> serve raw (B-identical); intervene
                # only once instability is MEASURED.
                _ents = self._chain_entries.get(_chain_id(task)) or []
                if len(_ents) >= 2:
                    _a1 = _norm_answer((_ents[-1].failure_taxonomy or {})
                                       .get("verbatim_outcome"))
                    _a0 = _norm_answer((_ents[-2].failure_taxonomy or {})
                                       .get("verbatim_outcome"))
                    _repair_raw = bool(_a1) and _a1 == _a0
                else:
                    _repair_raw = True
            elif _C_REPAIR_GATE == "verdict":
                # Gate on the EXTERNAL critic's outcome verdict for the chain's
                # latest entry — never on the actor's self-assessment. The self
                # gate was tried and failed exactly as the provenance principle
                # predicts: hy3's self-assessment passed 75% of chains as
                # "succeeding" where only ~20% were actually correct, so real
                # failures were replayed verbatim and repair never engaged.
                _ents = self._chain_entries.get(_chain_id(task)) or []
                _v = ""
                if _ents:
                    _v = str((_ents[-1].failure_taxonomy or {})
                             .get("critic_outcome_verdict", "")).lower()
                _repair_raw = _v == "correct"
            else:  # legacy self gate (ablation only)
                _prev = self._last_score.get(_chain_id(task))
                _repair_raw = _prev is not None and float(_prev) >= _C_REPAIR_THRESH
        if _repair_raw:
            served = cands[: self.top_k]
            out = _format_raw(served, action_scored=_action_scored)
        else:
            out = _format_curated(succ, fail, current_tid=tid_now,
                                  action_scored=_action_scored)
            if _C_POLICY != "judgment" and out:
                # Supersession from the version lineage. The judgment path picks
                # one attempt and endorses it; the manifest already records which
                # version replaced which, so conflicting attempts can be shown AS
                # a chain and the model adjudicates rather than trusting an
                # unverified ✓.
                out = _append_supersession(out, pool)
            served = succ + fail
        if not out and cands and _C_RAW_FALLBACK:
            # Curated channels rendered nothing -> degrade to the raw store
            # under the same budget, never to no memory (see _C_RAW_FALLBACK).
            # Same-task-first order makes the fallback replay this task's own
            # attempts before any session-mate's.
            served = cands[: self.top_k]
            out = _format_raw(served, action_scored=_action_scored)
        if out:
            # remember what was actually served, for the effectiveness update
            self._served[tid_now] = [e.task_id for e in served]
            self._served_keys[tid_now] = [
                (e.task_id, int(getattr(e, "version", 1) or 1)) for e in served]
            # system-layer usage statistic (the store's ANALYZE): how often this
            # entry was actually served into a prompt. Measured by the harness,
            # not asserted by any model — safe for ranking to read.
            for e in served:
                try:
                    st = getattr(e, "sys_stats", None)
                    if st is None:          # pre-sys_stats pickles
                        st = {}
                        setattr(e, "sys_stats", st)
                    st["inject_count"] = int(st.get("inject_count", 0)) + 1
                except Exception:
                    pass
            # Observability for w_c (the paper's effectiveness weight). Until now
            # w_c only ever moved retrieval ranking inside this object and was
            # never written anywhere, so no trace could show whether it did
            # anything at all. It cold-starts at 1.0 until an entry has >=2
            # measured deltas, and a 3-iteration chain supplies at most 2, so
            # "effectiveness-weighted retrieval" may be inert in practice. The
            # runner copies this onto every row; see _wc_stats.
            try:
                lib = self._sf.library
                ws = [float(lib.get_experience_weight(e.task_id)) for e in served]
                self._last_wc = {
                    "wc_served": [round(w, 3) for w in ws],
                    "wc_mean": round(sum(ws) / len(ws), 3) if ws else None,
                    "wc_active": sum(1 for w in ws if abs(w - 1.0) > 1e-9),
                    "wc_n": len(ws),
                }
            except Exception:
                self._last_wc = None
        return out

    def pop_wc_stats(self) -> dict | None:
        """w_c of the entries served by the last inject(), or None. Read-once so
        a row can never inherit a previous task's weights."""
        w = getattr(self, "_last_wc", None)
        self._last_wc = None
        return w


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
        served_keys = self._served_keys.pop(tid, None) if hasattr(self, "_served_keys") else None
        # Iteration index of this record within its chain (0-based).
        _kk = int(self.__dict__.setdefault("_chain_iter", {}).get(chain, 0))
        # The chain's own first, memory-free attempt is the reference: iter0
        # ran before anything could be injected, in the same run window, so
        # score(k) − score(0) needs no other arm (A and B stay standalone; in
        # deployment iter0 is simply the first attempt before memory exists).
        # The delta inherits the protocol's score provenance — under
        # self-assessment it is recorded for audit but can neither endorse nor
        # demote; under env/gold feedback (tau2, guarded) it carries weight.
        if _kk == 0 and score is not None:
            self.__dict__.setdefault("_chain_base", {})[chain] = float(score)
        _base = (getattr(self, "_chain_base", None) or {}).get(chain)
        if served and score is not None and _base is not None and _kk >= 1:
            _g = float(score) - float(_base)
            for eid in served:
                try:
                    self._sf.library.update_effectiveness(eid, _g)
                except Exception:
                    pass
            if served_keys:
                try:
                    _by_key_g = {(e.task_id, int(getattr(e, "version", 1) or 1)): e
                                 for e in self._sf.library.experiences}
                    for _k2 in served_keys:
                        e2 = _by_key_g.get(_k2)
                        if e2 is None:
                            continue
                        st2 = getattr(e2, "sys_stats", None)
                        if st2 is None:
                            st2 = {}
                            setattr(e2, "sys_stats", st2)
                        st2.setdefault("reuse_deltas", []).append(
                            {"delta": round(_g, 4),
                             "provenance": _SCORE_PROVENANCE,
                             "source": "vs_iter0", "iter": _kk})
                except Exception:
                    pass
        if served and score is not None:
            prev = self._last_score.get(chain)
            if prev is not None:
                delta = float(score) - prev
                for eid in served:
                    try:
                        self._sf.library.update_effectiveness(eid, delta)
                    except Exception:
                        pass
                # system-layer reuse statistic, provenance-tagged: the same
                # within-chain delta w_c consumes, but stored ON the entry with
                # WHERE the score came from. Ranking later reads only env/gold
                # deltas; self-assessed ones stay for audit (never trusted).
                if served_keys:
                    try:
                        _by_key = {(e.task_id, int(getattr(e, "version", 1) or 1)): e
                                   for e in self._sf.library.experiences}
                        for k in served_keys:
                            e = _by_key.get(k)
                            if e is None:
                                continue
                            st = getattr(e, "sys_stats", None)
                            if st is None:
                                st = {}
                                setattr(e, "sys_stats", st)
                            st.setdefault("reuse_deltas", []).append(
                                {"delta": round(delta, 4),
                                 "provenance": _SCORE_PROVENANCE})
                    except Exception:
                        pass
        if score is not None:
            self._last_score[chain] = float(score)
        self.__dict__.setdefault("_chain_iter", {})[chain] = _kk + 1
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
                critic_fn=(self._critic_llm if self.use_critic else None),
                enrich=self.use_enrich,
                score_provenance=_SCORE_PROVENANCE,
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
                # lesson (the B-superset rendering; see _format_curated).
                # UNCONDITIONAL on purpose: the stash key is the SELF-assessed
                # score, and self-assessment is badly miscalibrated exactly
                # where memory pays (LoCoMo: 66% of gold-correct attempts
                # self-assess <0.5, so a success-only stash silently drops the
                # answers B keeps). Measured replay effects justify keeping
                # wrong-looking answers too: replaying a gold-wrong answer is
                # ~neutral (gaia +3.4pp / locomo −1.1pp B−A) while replaying a
                # gold-right one is the payoff (+7 / +13pp); the render tags
                # low-self-score answers honestly instead of hiding them.
                try:
                    _mine.failure_taxonomy.setdefault("verbatim_outcome",
                                                      resp[:400])
                    if score is not None:
                        _mine.failure_taxonomy.setdefault(
                            "outcome_selfscore", float(score))
                except Exception:
                    pass
                self._chain_entries.setdefault(chain, []).append(_mine)
            # Attempt value vs the chain's own memory-free first attempt,
            # stamped on the new entry at its single curation moment: did this
            # attempt beat iteration 0? Same run window, no other arm read.
            try:
                if _base is not None and _mine is not None and score is not None \
                        and _kk >= 1:
                    _st = getattr(_mine, "sys_stats", None)
                    if _st is None:
                        _st = {}
                        setattr(_mine, "sys_stats", _st)
                    _st["baseline_delta"] = round(float(score) - float(_base), 4)
                    _st["baseline_provenance"] = _SCORE_PROVENANCE
            except Exception:
                pass
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
        # Carry w_c of the entries just served, so the trace can show whether
        # effectiveness weighting is live rather than nominally configured.
        try:
            w = mem.pop_wc_stats() if hasattr(mem, "pop_wc_stats") else None
        except Exception:
            w = None
        if w:
            r["_wc"] = w
    return r
