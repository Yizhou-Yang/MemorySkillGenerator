# Obsolete — retired benchmark drivers

Files here are **kept for reference only** and are **not** part of any experiment
launcher. Do not wire them back into `run_all_benchmarks.sh` / `run_all_models.sh`.

## Terminal-Bench-2 (retired 2026-07-09)

`tb2_harbor_bridge.py`, `tb2_harbor_agent.py`, `run_tb2_official.sh`,
`HARBOR_TB2_PLAN.md` — the official-harbor Terminus-2 A/B/C driver.

**Why retired:** every task spins its own Docker container (kernel builds, etc.),
so runs are slow and resource-heavy, and CodeBuddy-API latency pushed tasks past
the 2400 s agent timeout (only arm A / iter 0 ever completed cleanly). Replaced by
**tau2-bench** (`scripts/latest/tau2_bridge.py` + `tau2_agent.py`): no Docker,
an order of magnitude cheaper, dynamic (user simulator), and more authoritative.

The **simplified-loop** TB2 runner (`terminal_bench_2_runner.py`) and
`terminal_verifier.py` stay under `scripts/latest/` because core modules still
import them, but `terminal_bench_2` is no longer launched by the experiment
scripts and does not appear in the paper's benchmark set.
