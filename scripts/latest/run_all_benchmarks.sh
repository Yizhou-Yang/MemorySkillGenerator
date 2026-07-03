#!/usr/bin/env bash
# Launch ALL 4 benchmarks for ONE model together, under the publishable
# evolving protocol:
#   gaia / gaia2 / locomo  -> main sweep (latest_runner, skillforge env)
#   terminal_bench_2       -> official harbor Terminus-2 (bridge, harbor312 env)
#
# The two run in different conda envs, so this script backgrounds each. TB2 gets
# its feedback from in-container tests (env-observable) and does not mutate; the
# QA benchmarks use ITER_MUTATE variants + self-assessed feedback. Nothing here
# ever feeds a gold label into memory (reporting stays gold).
#
# Usage:
#   bash scripts/latest/run_all_benchmarks.sh <MODEL>
#   MODEL is the CodeBuddy id (deepseek-v4-pro, hy3-preview, glm-5.1, ...).
# Env overrides:
#   SKILLFORGE_PY / HARBOR_PY   interpreters (default: the two conda envs)
#   GAIA2_SCENARIO_DIR          persistent gaia2-cli path
#   TB2_N_TASKS                 TB2 tasks/iter (default 80, matched across models)
#   TASK_CONCURRENCY            main-sweep global slots (default 12; two models
#                               on one box should each use 12 to stay < the ~24
#                               internal-API ceiling)
#   NO_TB2=1                    skip TB2 (QA-only run)
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

MODEL="${1:?usage: run_all_benchmarks.sh <MODEL>}"
SKILLFORGE_PY="${SKILLFORGE_PY:-/root/.conda/envs/skillforge/bin/python}"
HARBOR_PY="${HARBOR_PY:-/root/.conda/envs/harbor312/bin/python}"
command -v "$SKILLFORGE_PY" >/dev/null 2>&1 || SKILLFORGE_PY="$(command -v python3)"
DATASET="${GAIA2_SCENARIO_DIR:-$REPO/.datasets/gaia2-cli}"
TB2_N="${TB2_N_TASKS:-80}"
CONC="${TASK_CONCURRENCY:-12}"

echo "==> launching all benchmarks for model=$MODEL"

# ── 1) QA sweep: gaia + gaia2 + locomo (evolving protocol) ──
RESULTS_BASE=latest_evolving ITER_MUTATE=1 ITER_FEEDBACK=self ITER_CHAIN=3 \
  BENCHMARKS=gaia,gaia2,locomo GAIA2_SCENARIO_DIR="$DATASET" \
  CODEBUDDY_MODEL="$MODEL" TASK_CONCURRENCY="$CONC" \
  nohup "$SKILLFORGE_PY" -u scripts/latest/latest_runner.py \
  > "run_${MODEL}_qa.log" 2>&1 &
echo "    [qa]  gaia/gaia2/locomo  PID $!  -> run_${MODEL}_qa.log"

# ── 2) TB2 via official harbor Terminus-2 (A/B/C arms) ──
if [ "${NO_TB2:-0}" != "1" ]; then
  if ! curl -sf localhost:8741/v1/models >/dev/null 2>&1; then
    nohup "$SKILLFORGE_PY" scripts/latest/codebuddy_oai_proxy.py > proxy.log 2>&1 &
    echo "    [tb2] started OAI proxy PID $! (waiting 5s)"; sleep 5
  fi
  for ARM in A B C; do
    OPENAI_API_BASE=http://localhost:8741/v1 OPENAI_API_KEY=dummy TB2_N_TASKS="$TB2_N" \
      nohup "$HARBOR_PY" scripts/latest/tb2_harbor_bridge.py \
      --arm "$ARM" --iters 3 --model "openai/$MODEL" --n-tasks "$TB2_N" \
      > "run_${MODEL}_tb2_${ARM}.log" 2>&1 &
    echo "    [tb2] arm $ARM  PID $!  -> run_${MODEL}_tb2_${ARM}.log"
  done
fi

echo "==> all 4 benchmarks launched for $MODEL. Tail: tail -f run_${MODEL}_*.log"
echo "==> gate when done: python scripts/latest/soft_stats.py experiments_results/latest_evolving/$MODEL"
