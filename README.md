# MemorySkillGenerator — research harness for CuratorMem

> Agents accumulate their own successive attempts at related problems. Patch-based
> memory suits that: record each change as a patch, keep superseded states as
> history, delete nothing. But a store built to persist is written once and never
> revisited, so it *decays* as it grows — duplicates split the retriever's
> attention, superseded attempts sit beside the ones that replaced them, and a
> later decision is served experience nobody ever checked.
>
> **CuratorMem** curates the store as it accumulates. Its claim is that the hard
> part of curation is not the editing but the **judging**: a curator asked to grade
> its own work endorses its own failures. It therefore keeps evidence and opinion
> apart — an append-only manifest whose *grounded* fields (version lineage,
> execution records, outcome provenance) are computed from the episode rather than
> asserted by the model — and lets curation claim only what that metadata backs:
> an entry is endorsed to the agent only when a **second, independent key** agrees
> (a grounded outcome score, or an external critic), never the actor's
> self-assessment alone.
>
> This repository is the harness that evaluates it, as a controlled **A/B/C
> ablation** across five agent benchmarks and against external memory frameworks.
> The memory layer itself ships as the [`memlayer`](memlayer/README.md) SDK.

---

## The experiment in one screen

Every benchmark is run under three **arms** (the `group` field in each trace row):

| Arm | What it does |
|-----|--------------|
| **A — no memory** | The plain agent. The control. |
| **B — raw patch memory** | Injects the raw record of what worked on earlier iterations of the *same* task, verbatim. |
| **C — CuratorMem** *(method under test)* | B's patches, curated: refined (generalized + causal lesson), enriched rather than dropped when weak, retrieval effectiveness-weighted, an "avoid this" channel from failed attempts — and endorsed to the agent only when a second, independent key backs the claim. |

The question under test is **C vs B** — whether *curating* content helps beyond
*having* it; **B vs A** is the secondary check (memory beats none).

**What the sweep actually found:** pooled across benchmarks and backbones, C−B is
indistinguishable from zero, and the interval is tight around nothing rather than
wide around something. The gain turns out to be a property of the *curator*, not
of curation: it reverses with backbone strength, because arm C instantiates its
critic from the backbone itself, and a backbone too weak to recognize its own
failures endorses them instead. That diagnosis is what motivates the
metadata-grounded policy (curate by the manifest's measured statistics rather
than the curator's self-assessment) and the fixed external critic. Numbers live
in the paper, not here.

**Key mental model:** patch memory is feedback across **iterations of the same
task** (a version chain), *not* cross-task transfer. Retrieval is **chain-scoped**,
so a single pass over independent tasks injects nothing and A=B=C. To exercise
memory you run **iteration chains** (`ITER_CHAIN=K`, each task run K times).

**Benchmarks:** `gaia` (multi-step QA), `gaia2` (agentic tool-use, soft recall),
`locomo` (long-conversation memory), `tau2` (tool-use against a mutating domain),
`terminal_bench_2` (terminal tasks, pytest). GAIA2/tau2/TB-2 are *interactive* — a
stored attempt carries actions, not text — while GAIA/LoCoMo are static QA that
check curation does not hurt.

**External memory frameworks.** A-Mem, Mem0 and MemoryOS are mounted behind the
same record/inject interface as arms B and C, so they run on the same agent,
tasks and budget (`scripts/latest/baseline_memories.py`).

**LoCoMo also runs under its authors' own protocol.** The A/B/C framing re-attempts
one task, which makes memory of prior *answers* irrelevant on single-shot
conversational QA and understates retrieval-centric systems. So LoCoMo is
additionally run the way Mem0/A-Mem evaluate it — ingest session by session,
retrieve top-k per question, answer from the retrieved memories alone, score with
an LLM judge and token-F1 (`scripts/latest/locomo_native.py`). That is a
memory-layer comparison and is reported separately from the A/B/C results.

### ➜ Vetting a run

If your job is to **gatekeep experiment quality**, start here:
**[`experiments_results/EXPERIMENT_QUALITY.md`](experiments_results/EXPERIMENT_QUALITY.md)**
— the gate checklist (completeness → error rate → injection fired → significance →
plausible scores) and the known failure modes. Most runs that look interesting
fail a gate; that document tells you which and what to do.

---

## Quick start

```bash
pip install -r requirements.txt

# Full A/B/C sweep with iteration chains (memory threads across iterations):
ITER_CHAIN=3 bash scripts/latest/run_all_models.sh

# Resume after a crash (keeps finished arms):
RESUME=1 ITER_CHAIN=3 python scripts/latest/latest_runner.py

# Analyze:
python scripts/latest/analyze_results.py experiments_results/latest/<model>   # A/B/C + significance
python scripts/latest/breakdown.py       experiments_results/latest/<model>   # did memory fire? + sub-tables
```

Results land in `experiments_results/latest/<model>/<benchmark>/trace.jsonl`.
See [`experiments_results/README.md`](experiments_results/README.md) for the trace
schema and [`scripts/latest/EXPERIMENT_PLAN.md`](scripts/latest/EXPERIMENT_PLAN.md)
for how a full sweep (models, pre-flight, order of work) is launched.

---

## How the memory works (arm C)

```
   Agent execution (per iteration of a task)
            │  trajectory + score
            ▼
   analysis.py    trajectory → structured experience; failure classification
            │
            ▼
   refine.py      version-conditioned refinement: generalize + causal lesson,
            │     analyze the patch diff-chain, critic scores quality,
            │     low-quality → enriched (never discarded)
            ▼
   experience.py  append-only patch history + retrieval, per-entry
            │     effectiveness weighting
            ▼
   endorsement    grounded metadata (lineage, execution records, outcome
            │     provenance) vs asserted; endorsed only if a second,
            │     independent key agrees — never self-assessment alone
            ▼
   gate.py        applicability / complexity gate → inject or not
            │
            ▼
   injection.py   chain-scoped, dose-budgeted injection into the next
                  iteration's prompt (+ "avoid this" from failed attempts)
```

Arm **B** uses the same substrate but injects the **raw** patch (no refine, no
critic, no enrichment). Arm **A** skips it entirely.

### Design principles

1. **Non-destructive.** The store never compresses, summarizes, or deletes a
   patch — a low-quality one is *enriched* (failure modes, recovery steps,
   preconditions added), not removed. Content is only replaced by a strictly
   richer version.
2. **Dual feedback.** Both successes (what worked) and failures (what to avoid)
   are recorded and injected — a failed patch can teach more than a successful one.
3. **Append-only version history.** Each experience keeps a patch log: score
   deltas, outcome transitions, steps fixed / still missing — so the system learns
   *how* a patch evolved, not just its final state.
4. **Effectiveness-weighted retrieval.** Each experience tracks its historical
   injection effectiveness; ones that hurt are down-weighted, ones that helped are
   up-weighted — a self-correcting retrieval signal.
5. **Evidence separated from opinion.** Manifest fields are either *grounded*
   (computed from the episode: version lineage, execution records, outcome
   provenance) or *asserted* (written by a model). Curation may only claim what
   grounded metadata backs, and an entry is endorsed to the agent only when a
   second, independent key agrees — the mechanism the aggregate result above
   motivates.

---

## Repository layout

```
memlayer/               # the memory layer as an installable SDK (see its README)
├── bridge.py           # facade: record / inject / search / manifest / time_travel
├── experience.py       # append-only patch history + retrieval
├── vgr.py              # chain-scoped patch store (lexical, or semantic via embedder=)
├── refine.py           # version-conditioned refinement + critic + enrichment
├── analysis.py         # trajectory analysis + failure classification
├── injection.py        # chain-scoped, dose-budgeted prompt injection
└── gate.py             # applicability / complexity gating

src/latest/             # experiment-side agents and scorers
├── agent/              # per-benchmark agents (amem, memento, terminus2, ...)
├── eval/               # scorers (e.g. gaia2_judge.py)
├── llm/                # prompt templates
└── safety/             # budget / completion / dedup guards

benchmarks/loader.py    # unified loader → {task_id, description, expected, context, metadata}
scripts/latest/         # runners, sweep driver, external baselines, analysis
configs/                # YAML experiment configs
experiments_results/    # trace.jsonl + frozen snapshots (+ EXPERIMENT_QUALITY.md)
tests/                  # unit + integration tests
```

The memory layer is also usable on its own, independent of this harness:

```python
from memlayer import MemoryLayer
mem = MemoryLayer(llm=my_llm_fn)                      # or llm=None for append-only
mem.record("v2 needs a partition spec", chain_id="deploy-42", task="register table")
block = mem.inject("how do I register the table?", chain_id="deploy-42")
```

See [`memlayer/README.md`](memlayer/README.md) for the API, guarantees and knobs.

## Dependencies

Algorithmic components use established libraries — no hand-rolled similarity/NLP:

| Library | Usage |
|---------|-------|
| `sentence-transformers` | semantic embedding for retrieval (all-MiniLM-L6-v2) |
| `scikit-learn` | TF-IDF + cosine similarity (fallback retrieval) |
| `rapidfuzz` | fuzzy action-sequence alignment |
| `json_repair` | robust JSON extraction from LLM output |

The LLM backend runs against any OpenAI-compatible chat endpoint (see
[`.env.example`](.env.example)); no vendor CLI is required to reproduce a run.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
