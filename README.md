# MemorySkillGenerator

> A research harness for studying **curated patch memory** for LLM agents: does
> *refining and curating* an agent's memory of past patches beat injecting the raw
> patches, and does either beat no memory at all? Evaluated as a controlled
> **A/B/C ablation** across four agent benchmarks.

---

## The experiment in one screen

Every benchmark is run under three **arms** (the `group` field in each trace row):

| Arm | What it does |
|-----|--------------|
| **A — no memory** | The plain agent. The control. |
| **B — raw patch memory** | Injects the raw record of what worked on earlier iterations of the *same* task, verbatim. |
| **C — curated patch memory** *(method under test)* | B's patches, but refined (generalized + causal lesson), critic-scored, low-quality ones enriched not dropped, retrieval effectiveness-weighted, plus an "avoid this" channel from failed attempts. |

The headline claim is **C > B** (curation beats raw patches); **B > A** is the
secondary check (memory beats none).

**Key mental model:** patch memory is feedback across **iterations of the same
task** (a version chain), *not* cross-task transfer. Retrieval is **chain-scoped**,
so a single pass over independent tasks injects nothing and A=B=C. To exercise
memory you run **iteration chains** (`ITER_CHAIN=K`, each task run K times).

**Benchmarks:** `gaia` (multi-step QA), `gaia2` (agentic tool-use, soft recall),
`locomo` (long-conversation memory), `terminal_bench_2` (terminal tasks, pytest).

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
            │     analyze the patch diff-chain, independent critic scores quality,
            │     low-quality → enriched (never discarded)
            ▼
   experience.py  append-only patch history + semantic retrieval
            │     (sentence-transformers, TF-IDF cosine fallback) +
            │     per-experience effectiveness weighting
            ▼
   gate.py        applicability / complexity gate → inject or not
            │
            ▼
   injection.py   chain-scoped, effectiveness-weighted injection into the next
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

---

## Repository layout

```
src/latest/
├── __init__.py         # orchestrator: record → refine → store → inject
├── experience.py       # Experience + append-only patch history + semantic retrieval
├── analysis.py         # trajectory analysis + 4-way failure classification
├── refine.py           # version-conditioned refinement + cross-agent critic + enrichment
├── injection.py        # chain-scoped, effectiveness-weighted prompt injection
├── gate.py             # task-type / complexity gating
├── vgr.py              # chain-scoped patch store
├── agent/              # per-benchmark agents (amem, memento, terminus2, ...)
├── eval/               # scorers (e.g. gaia2_judge.py)
├── llm/                # prompt templates
└── safety/             # budget / completion / dedup guards

benchmarks/loader.py    # unified loader → {task_id, description, expected, context, metadata}
scripts/latest/         # runner, sweep driver, analysis (analyze_results.py, breakdown.py)
configs/                # YAML experiment configs
experiments_results/    # trace.jsonl + frozen snapshots (+ EXPERIMENT_QUALITY.md)
tests/                  # unit + integration tests
```

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
