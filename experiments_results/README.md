# Experiment Results

**A/B/C arms (consistent across all benchmarks, matching the paper):**
- **A — Vanilla:** no memory.
- **B — EvoMem:** cross-task patch memory; retrieve relevant prior patches and inject them.
- **C — GPR:** B's patches **plus** a per-patch environment check (Grounded Patch
  Resolution) — a strict superset of B's context. On executable benchmarks the
  agent grounds via probes; the chain experiment lives in
  `scripts/latest/vgr_experiment.py`.

Trace group keys: `A_baseline`, `B_evomem`, `C_gpr`.


Only the **latest** run is kept here. Earlier paper-v4 / paper-v5 / rerun-v2 /
rerun-deepseek_v4pro / unified_v6 result trees have been removed because the
evaluation pipeline changed (oracle-driven retry → cross-agent critic; F1 → EM
/ pass@1).

```
experiments_results/
└── latest/
    ├── final_summary.json          # cross-benchmark headline
    ├── gaia/
    │   ├── library_after_train.json
    │   └── report.json
    ├── alfworld/
    │   ├── library_after_train.json
    │   └── report.json
    └── locomo/
        ├── library_after_train.json
        └── report.json
```

Run with:

```bash
# Full run (wipes existing traces and starts fresh)
python scripts/latest/latest_runner.py

# Resume an interrupted run — keeps trace.jsonl and skips already-completed
# (group, task_id) pairs, retrying only what's missing. Use this after a
# rate-limit crash so you don't lose finished groups.
RESUME=1 python scripts/latest/latest_runner.py

# Tune throughput vs. resources. Concurrency is now a SINGLE global cap on
# heavy tasks across all benchmarks (auto-sized from cpu/mem), not the old
# multiplicative per-benchmark × per-suite model that OOM'd. Env knobs:
#   TASK_CONCURRENCY   global heavy-task cap   (default: auto = min(cpu-1, 8),
#                                               further capped by free RAM if
#                                               psutil is installed)
#   DOCKER_CONCURRENCY container cap for terminal_bench_2   (default 2)
#   PER_TASK_GB        RAM budget per task for auto-sizing   (default 1.2)
#   EMBED_NUM_THREADS  CPU threads per embedding encode      (default 1)
#   BENCH_CONCURRENCY  benchmarks allowed to interleave      (default 6)
TASK_CONCURRENCY=8 DOCKER_CONCURRENCY=2 python scripts/latest/latest_runner.py
```

Why this is faster *and* won't OOM: the embedding model (a 0.6B transformer on
CPU) is loaded **once** and shared (was one copy per A-Mem agent), each encode is
pinned to one thread (was all cores × every parallel task), and total heavy work
is bounded by one memory-aware semaphore — so expanding the test set only
lengthens the queue, it does not raise peak CPU/RAM.

Analyze A/B/C with paired significance (mean EM ± std, McNemar p-value,
bootstrap CI, and an A≤B≤C ordering check):

```bash
python scripts/latest/analyze_results.py
```

A run is only complete when all three groups (A/B/C) have `n=30` in every
`trace.jsonl`. Missing groups mean the run crashed mid-benchmark — resume it.
