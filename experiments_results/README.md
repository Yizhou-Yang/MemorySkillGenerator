# Experiment Results

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
python scripts/v6/latest_runner.py
```
