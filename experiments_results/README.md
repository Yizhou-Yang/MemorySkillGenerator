# Experiment Results — SkillForge V6

> Raw experiment artifacts for the V6 ablation study (A: Baseline, B: Raw injection, C: AI-Refined injection).

## Layout

```
experiments_results/
├── unified_v6_results/
│   ├── final_summary.json              # 5-benchmark aggregated summary (gaia2/locomo/gaia/alfworld/swebench)
│   ├── rerun_summary.json              # ALFWorld offline rerun
│   ├── gaia2/
│   │   ├── report.json                 # A/B/C scores (n=25 test)
│   │   └── library_after_train.json    # 25 experiences collected from train phase
│   ├── swebench_dynamic/
│   │   ├── report.json                 # A/B/C scores (n=50 test, Docker-based)
│   │   └── library.json                # 30 train experiences
│   ├── locomo/
│   │   ├── report.json                 # A/B/C scores (n=25 test, F1)
│   │   └── library_after_train.json
│   ├── gaia/
│   │   ├── report.json                 # A/B/C scores (n=25 test, EM/F1)
│   │   └── library_after_train.json
│   └── alfworld_interactive/
│       ├── report.json                 # A/B/C scores (n=20 test, all 0% — model limit)
│       └── library.json
├── results_chart.png                   # Ablation visualization
└── results_chart.pdf
```

## Headline Numbers

| Benchmark | A (Baseline) | B (Raw) | C (AI-Refined) | Δ(C-A) |
|-----------|:---:|:---:|:---:|:---:|
| **Gaia2** (Soft Recall) | 41.6% | 38.6% | **45.1%** | **+3.5pp** |
| **SWE-bench** (Patch Rate) | 40.0% | 50.0% | **54.0%** | **+14pp** |
| LoCoMo (F1) | 7.4% | 7.1% | 6.8% | -0.6pp (中性) |
| GAIA HF (EM/F1) | 18.0% | 18.5% | 4.0% | -14pp (待修复) |
| ALFWorld (interactive) | 0% | 0% | 0% | model capability limit |

## Reproducing

Scripts are in `scripts/v6/`:
- `unified_v6_runner.py` — Gaia2/LoCoMo/GAIA/ALFWorld
- `swebench_dynamic_runner.py` — SWE-bench Verified (Docker)
- `alfworld_interactive_runner.py` — ALFWorld via subprocess

Model: `hy3-preview-ioa` (CodeBuddy SDK, free internal model).

## Key Evidence

**B < A in Gaia2** (38.6% vs 41.6%) — Raw injection introduces noise, proving that
the AI-refine step (`refine.py`) is necessary, not a trivial "more context = better"
gain. Without refinement, experiences hurt performance.
