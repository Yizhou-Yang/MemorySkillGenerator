# V6 Experiment Runners

Experiment scripts for running SkillForge V6 ablation (A/B/C groups) on each benchmark.

## Files

| File | Benchmark | Notes |
|------|-----------|-------|
| `unified_v6_runner.py` | Gaia2, LoCoMo, GAIA (HF), ALFWorld | Main runner; train/test split = first half / second half |
| `swebench_dynamic_runner.py` | SWE-bench Verified | Docker container per instance; agent uses Bash to read/edit/test |
| `alfworld_interactive_runner.py` | ALFWorld | textworld subprocess per game; agent sends text commands |

## Ablation Groups

- **A (Baseline)**: original prompt, no augmentation
- **B (Raw)**: inject experiences without AI refinement (success+failure)
- **C (AI-Refined)**: inject experiences after `refine.py` (version-conditioned LLM refinement)

All three groups evaluate on the **same** test set with isolated state (per-group state directories
for Gaia2, fresh containers for SWE-bench).

## Configuration

```python
MODEL = "hy3-preview-ioa"     # CodeBuddy SDK free internal model
CONCURRENCY = 20               # 3 for SWE-bench (Docker)
TASK_TIMEOUT = 300             # 5 min per task
RESULTS_DIR = "/data1/benchmarks/unified_v6_results"
```

## Environment

```bash
export CODEBUDDY_API_KEY='...'
export CODEBUDDY_INTERNET_ENVIRONMENT='ioa'
```

Results are written to `experiments_results/unified_v6_results/{benchmark}/`.
