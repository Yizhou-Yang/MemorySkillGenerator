# Experiment Runner

The A/B/C ablation runner for `gaia`, `gaia2`, `locomo`, and `terminal_bench_2`.

## Files

| File | Description |
|------|-------------|
| `latest_runner.py` | Main driver. Runs each benchmark under all three arms, with iteration chains (`ITER_CHAIN`) so patch memory has in-chain history to retrieve. Writes `experiments_results/latest/<model>/<benchmark>/trace.jsonl`. |
| `run_all_models.sh` | Sweeps the runner across the model list (resume-safe). |
| `analyze_results.py` | A/B/C means ± std, paired McNemar, bootstrap CI, A≤B≤C ordering, completeness check. |
| `breakdown.py` | Injection-isolation gate (did memory fire?), by-type / by-difficulty, chain-level accuracy. |
| `EXPERIMENT_PLAN.md` | How a full sweep is launched (models, pre-flight, order of work). |
| `<benchmark>_runner.py`, `evomem_bridge.py`, `eval.py`, `tools.py`, `trace.py` | Per-benchmark harnesses, the memory bridge that wires arms B/C, scoring, tools, and trace I/O. |

## Arms

- **A — no memory.** The plain agent; the control.
- **B — raw patch memory.** Injects the raw record of what worked on earlier
  iterations of the *same* task (chain-scoped), verbatim. No cross-task transfer.
- **C — curated patch memory** (the method under test). B's patches, but each is
  refined (generalized + a causal lesson), scored by an independent LLM critic,
  low-quality patches enriched rather than discarded, retrieval effectiveness-
  weighted, and failed attempts add an "avoid this" channel.

The comparison that matters is **C vs B**. Trace `group` keys are legacy
identifiers (`A_baseline` / `B_evomem` / `C_gpr`) — read them as A/B/C.

## Design choices

1. **No oracle-driven retry on QA.** In deployment you cannot tell whether a
   GAIA/LoCoMo answer is correct, so there is no correctness-gated retry. Instead
   an independent LLM critic rates each candidate experience; only sufficiently
   high-quality ones enter (or enrich) the store.
2. **Memory is within-chain.** Patch memory is feedback across *iterations of the
   same task*, not cross-task transfer. Retrieval is chain-scoped, so a single
   pass over independent tasks injects nothing (A=B=C) — use `ITER_CHAIN>1`.
3. **Metrics.** Exact-match for QA (`gaia`, `locomo`), soft recall for `gaia2`,
   pytest pass for `terminal_bench_2`.

## Run

```bash
ITER_CHAIN=3 bash scripts/latest/run_all_models.sh      # full sweep
RESUME=1 ITER_CHAIN=3 python scripts/latest/latest_runner.py   # resume after a crash
```

Results land in `experiments_results/latest/<model>/<benchmark>/`. **Before
trusting any output, run the gates in
[`../../experiments_results/EXPERIMENT_QUALITY.md`](../../experiments_results/EXPERIMENT_QUALITY.md).**
