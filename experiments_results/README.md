# Experiment Results

**Before trusting any number here, read
[`EXPERIMENT_QUALITY.md`](EXPERIMENT_QUALITY.md)** — the gate checklist for
deciding whether a run is valid. Most runs that look interesting fail a gate.

## Arms

Every benchmark is run under three arms (the `group` field in each trace row):

- **A — no memory** (`A_baseline`): the plain agent; the control.
- **B — raw patch memory** (`B_evomem`): injects the raw record of what worked on
  earlier iterations of the *same* task, verbatim. Chain-scoped.
- **C — curated patch memory** (`C_gpr`): the method under test — B's patches, but
  refined (generalized + causal lesson), critic-scored, low-quality ones enriched
  not dropped, retrieval effectiveness-weighted, plus an "avoid this" channel from
  failed attempts.

The headline comparison is **C vs B**; **B vs A** is the secondary check. Trace
keys are legacy identifiers — read them as A/B/C. (Older frozen snapshots use
`B_evoarena`/`C_skillforge`.)

## Layout

```
experiments_results/
├── README.md                      # this file
├── EXPERIMENT_QUALITY.md          # how to vet a run (read first)
├── latest/                        # current (overwritable) runs
│   └── <model>/<benchmark>/       # e.g. hy3-preview-ioa/gaia/
│       └── trace.jsonl            #   one JSON row per (task, arm, iteration)
├── formal/                        # frozen, immutable snapshots (see formal/README.md)
│   └── <date>_<tag>/<benchmark>/
│       ├── trace.jsonl
│       └── report.json            #   aggregated mean score per arm
└── vgr/report.json
```

Benchmarks: `gaia`, `gaia2`, `locomo`, `terminal_bench_2`. Runs only ever write to
`latest/`; `formal/` snapshots are locked (never written by a run).

## `trace.jsonl` schema (one row per task × arm × iteration)

| Field | Meaning |
|-------|---------|
| `benchmark`, `group`, `task_id` | which benchmark, which arm (A/B/C), which task |
| `iteration`, `iter_total` | position in the iteration chain (`ITER_CHAIN=K` ⇒ `iter_total=K`) |
| `score`, `em` | primary score (may be soft/fractional) and binary exact-match |
| `error` | non-empty ⇒ the task failed to run (infra/dataset/API) — **not** a wrong answer |
| `patch_injected`, `aug_len`, `augmented_prompt` | whether memory fired, and what was injected |
| `category`, `level` | task type / difficulty (for the breakdown sub-tables) |
| `execution_mode`, `method`, `phase`, `response`, `expected`, `timestamp` | run metadata + model output vs. gold |
| `prof_total_s`, `prof_embed_s`, `prof_docker_s`, `prof_llm_io_s`, `prof_llm_calls`, `prof_tok_in`, `prof_tok_out` | per-task profiling (wall-clock, tokens, LLM calls) |

`report.json` (frozen snapshots only): `{benchmark, model, groups_rerun, scores:{<arm>:mean}, trace_count}`.

## Running

```bash
# Full run. Each task is run ITER_CHAIN times so memory threads across iterations.
ITER_CHAIN=3 bash scripts/latest/run_all_models.sh

# Single model / smaller set:
ITER_CHAIN=3 TASK_LIMIT=100 python scripts/latest/latest_runner.py

# Resume an interrupted run — keeps trace.jsonl, skips completed (group, task_id,
# iteration), retries only what's missing. Use after any crash so finished arms
# are not lost.
RESUME=1 ITER_CHAIN=3 python scripts/latest/latest_runner.py
```

Throughput knobs (a single global cap on heavy tasks, auto-sized from CPU/RAM —
raising the task set lengthens the queue, it does not raise peak CPU/RAM):

| Env | Meaning | Default |
|-----|---------|---------|
| `TASK_CONCURRENCY` | global heavy-task cap | auto = `min(cpu-1, 8)` |
| `DOCKER_CONCURRENCY` | container cap for `terminal_bench_2` | 2 |
| `BENCH_CONCURRENCY` | benchmarks allowed to interleave | 6 |
| `PER_TASK_GB` | RAM budget per task for auto-sizing | 1.2 |
| `EMBED_NUM_THREADS` | CPU threads per embedding encode | 1 |

## Analyzing

```bash
# A/B/C means, paired McNemar p-value, bootstrap CI, A≤B≤C ordering, completeness:
python scripts/latest/analyze_results.py experiments_results/latest/<model>

# Injection isolation (did memory fire?), by-type / by-difficulty, chain-level acc:
python scripts/latest/breakdown.py experiments_results/latest/<model>
```

A run is complete only when all three arms (A/B/C) are present at the expected `n`
in every `trace.jsonl`. Missing groups mean it crashed mid-benchmark — resume it,
then re-check the gates in [`EXPERIMENT_QUALITY.md`](EXPERIMENT_QUALITY.md).
