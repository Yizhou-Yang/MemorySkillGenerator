# CuratorMem — Experiment Plan (executable)

**Goal.** Produce the paper's tables: a main A/B/C table across backbone models,
plus the mechanism / breakdown sub-tables.

- **Arms:** A = no memory, B = raw `\patchmem` (inject raw patches), C = CuratorMem
  (curated store). Set per benchmark by the existing `latest_runner.py`.
- **Benchmarks (4):** `gaia`, `gaia2`, `locomo`, `terminal_bench_2`.
- **Models (7), cheapest → most expensive** (output price, RMB/MTok, USD×7):

  | # | model id (CodeBuddy) | out price | note |
  |---|---|---|---|
  | 1 | `hy3-preview-ioa` | in-house | **primary**, in-house |
  | 2 | `deepseek-v4-pro` | ¥6 | confirmed |
  | 3 | `minimax-m2.7` | ¥8.4 | **VERIFY id** (not in console list) |
  | 4 | `glm-5.1` | ¥24 | VERIFY id |
  | 5 | `kimi-k2.6` | ¥27 | VERIFY id |
  | 6 | `gemini-3.1-pro` | ¥84 ($12×7) | VERIFY id |
  | 7 | `gpt-5.5` | ¥210 ($30×7) | VERIFY id |
  | – | `claude-4.6-opus` | ¥175 | **left blank** (paper placeholder column) |

  Cheapest first so pipeline bugs surface before the expensive models burn budget.

## Pre-flight (do once before the sweep)

1. **Verify the 5 non-HY3 CodeBuddy model ids** in the console (esp. `minimax-m2.7`,
   guessed). Fix them in `run_all_models.sh` if wrong.
2. **Internet environment:** HY3 uses `CODEBUDDY_INTERNET_ENVIRONMENT=ioa` (internal).
   External models may need a different value — override per model in the wrapper.
3. **Docker up** for `gaia2` and `terminal_bench_2` (Harbor / sandbox); the other two
   are pure text.
4. **Datasets present** (HF cache, `/tmp/harbor-datasets/...` for gaia2-cli).
5. `.env` has CodeBuddy credentials. API probe runs automatically at start.

## Run

```bash
# full sweep, 7 models × 4 benchmarks × A/B/C, ~100 tasks/benchmark
bash scripts/latest/run_all_models.sh
# fewer tasks for a smoke test:
TASK_LIMIT=10 bash scripts/latest/run_all_models.sh
# resume (default RESUME=1): a crash mid-sweep keeps finished models/benchmarks
```

Results land per model:
`experiments_results/latest/<model>/<benchmark>/{trace.jsonl,report.json}`.

### Iteration chains (where patch memory actually pays off)

Patch memory is feedback across **iterations of the same task**, not cross-task
transfer. On a single-pass run of independent tasks (GAIA/GAIA2/TB2) there is no
in-chain history, so B/C honestly inject nothing and A=B=C — retrieval is now
**chain-scoped** (same task_id / LoCoMo session), not global. To exercise memory,
run iteration chains:

```bash
ITER_CHAIN=3 bash scripts/latest/run_all_models.sh   # each task run 3x, memory threads across iterations
```

The main table uses the **final** iteration (post-memory); **chain-level accuracy**
(all iterations correct) is aggregated by `breakdown.py` from the per-iteration
trace rows (`iteration` / `iter_total` fields) → fills `tab:chain`.

## What each trace.jsonl line logs (one per task × arm)

`benchmark, group(A/B/C), task_id, score, em, error, execution_mode`,
profiling `prof_total_s/embed_s/docker_s/llm_io_s`, **and (new)**:
`category` (task type/level), `level` (difficulty), `patch_injected` (bool),
`aug_len`. These let every sub-table be aggregated from one sweep — no re-runs.

## Tables and how to build each (maps to `paper/main.tex`)

| Table (tex label) | What | Aggregate from |
|---|---|---|
| **Table 1** main (`tab:main`) | A/B/C per benchmark × 7 models + avg | `report.json` per model (mean score per arm); `analyze_results.py <model>` for ±std / McNemar / bootstrap CI / A≤B≤C |
| **Table 2** diagnostics (`tab:diag`) | retrieval precision@k (δ_M proxy); clause/row evidence capture, B vs C | retrieval-precision: separate probe over curated vs raw store; evidence capture: store inspection. **needs a small aggregation script** |
| **Table 3** by-task-type (`tab:bytype`) | C−B by `category`, per benchmark | group `trace.jsonl` by `category`, mean score per arm |
| **Figure** isolation | acc where `patch_injected=True` vs `False` | group `trace.jsonl` by `patch_injected` |
| **App.** cost | tokens/calls/latency per arm | `prof_llm_io_s` (+ token usage if the SDK exposes it) |

`analyze_results.py` does mean±std / McNemar / bootstrap CI / A≤B≤C per model
dir. `breakdown.py <results_dir>` reads `trace.jsonl` and groups by `category` /
`level` / `patch_injected` to fill Tables 2–3 + the isolation figure (works on
old traces too: derives injection from a non-empty `augmented_prompt`).

## Pre-run correctness gate (learned from the first sweep)

The first sweep's A/B/C were **identical** on GAIA/GAIA2/TB2 — cross-task memory
never fired (every task scoped to its own unique `task_id` ⇒ singleton chain ⇒
empty retrieval). Fixed in `evomem_bridge.py` (global retrieval when no shared
chain + pollution guard + effectiveness-weighted C). **Before trusting any new
sweep, confirm injection is actually happening:**

```bash
python scripts/latest/breakdown.py experiments_results/latest/<model>
# the "patch-injection isolation" table must show injected n > 0 for B and C on
# GAIA/GAIA2/TB2 — if it prints the "NO patches were injected" warning, the run
# is just sampling noise and must be re-run.
```

## Order of work

1. HY3 smoke test (`TASK_LIMIT=10`) end-to-end → confirm A/B/C all log + tables aggregate.
2. Full HY3 run (primary) → fills Table 1 HY3 row + Tables 2–3.
3. Sweep the remaining 6 models (cheap → expensive) for the Table 1 rows.
