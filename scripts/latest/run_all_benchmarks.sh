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
#   MODEL is the CodeBuddy id. Paper roster (5 backbones):
#     PRIMARY (free, run now, full ITER_CHAIN=3):
#       glm-5.1 · kimi-2.6 · hunyuan3-preview-ioa   (=hy3, A/B kept, rerun C)
#     TABLE-FILL (paid, ~Aug, LEAN ok):
#       gpt-5.3-codex (~Y1500) · claude-haiku-4.5 (~Y4000)
#   The 3 free primaries alone give ~900 pooled pairs (MDE~3.3pp).
# Env overrides:
#   SKILLFORGE_PY / HARBOR_PY   interpreters (default: the two conda envs)
#   GAIA2_SCENARIO_DIR          persistent gaia2-cli path
#   TB2_N_TASKS                 TB2 tasks/iter (default 50, matched across models)
#   GAIA2_SPLIT_WEIGHTS         per-split counts (default weights the dynamic
#                               splits, adaptability/time, where the thesis
#                               predicts the effect; declared in the paper)
#   TASK_CONCURRENCY            main-sweep global slots (default 12; two models
#                               on one box should each use 12 to stay < the ~24
#                               internal-API ceiling)
#   NO_TB2=1                    skip TB2 (QA-only run)
#   LEAN=1                      budget protocol for expensive models: ITER_CHAIN=2,
#                               gaia2 at 100 tasks (~50% cost; per-split power and
#                               ablation stay on the cheap/free backbones)
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

MODEL="${1:?usage: run_all_benchmarks.sh <MODEL>}"
# Force lowercase: API model names are case-sensitive and only accept lowercase
MODEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
SKILLFORGE_PY="${SKILLFORGE_PY:-/root/.conda/envs/skillforge/bin/python}"
HARBOR_PY="${HARBOR_PY:-/root/.conda/envs/harbor312/bin/python}"
command -v "$SKILLFORGE_PY" >/dev/null 2>&1 || SKILLFORGE_PY="$(command -v python3)"
DATASET="${GAIA2_SCENARIO_DIR:-$REPO/.datasets/gaia2-cli}"
TB2_N="${TB2_N_TASKS:-30}"
CONC="${TASK_CONCURRENCY:-30}"
# Weighted gaia2 sampling (pre-declared): 50% on the dynamic splits where the
# thesis predicts C-B, 25% on ambiguity where memory value is MEASURED largest
# (B-A=+6.8pp on the uniform-200 A/B data; adaptability +0.0, time +1.9).
# Sums to 100; every per-split count is <= the 40/split a previous uniform-200
# run took, so existing A/B rows still pair.
G2W="${GAIA2_SPLIT_WEIGHTS:-adaptability:25,time:25,ambiguity:25,execution:12,search:13}"

echo "==> launching all benchmarks for model=$MODEL"

# ── 1) QA sweep: gaia + gaia2 + locomo (evolving protocol) ──
ITERS=3
if [ "${LEAN:-0}" = "1" ]; then ITERS=2; echo "    [lean] ITER_CHAIN=2"; fi
RESULTS_BASE=latest_evolving ITER_MUTATE=1 ITER_FEEDBACK=self ITER_CHAIN="$ITERS" \
  GAIA2_SPLIT_WEIGHTS="$G2W" \
  BENCHMARKS=gaia,gaia2,locomo GAIA2_SCENARIO_DIR="$DATASET" \
  CODEBUDDY_MODEL="$MODEL" TASK_CONCURRENCY="$CONC" \
  nohup "$SKILLFORGE_PY" -u scripts/latest/latest_runner.py \
  > "run_${MODEL}_qa.log" 2>&1 &
echo "    [qa]  gaia/gaia2/locomo  PID $!  -> run_${MODEL}_qa.log"

# ── 2) TB2 via official harbor Terminus-2 (A/B/C arms, SERIALIZED) ──
# Arms run SEQUENTIALLY inside one background subshell. Parallel arms meant
# 3 arms x DOCKER-slots heavy containers (kernel builds etc.) on one box —
# the container-name collisions and agent timeouts came from exactly that.
# Wall-clock cost is small: each arm still runs its own internal concurrency,
# and B/C depend on their previous iteration anyway.
if [ "${NO_TB2:-0}" != "1" ]; then
  if ! curl -sf localhost:8741/v1/models >/dev/null 2>&1; then
    nohup "$SKILLFORGE_PY" scripts/latest/codebuddy_oai_proxy.py > proxy.log 2>&1 &
    echo "    [tb2] started OAI proxy PID $! (waiting 5s)"; sleep 5
  fi
  (
    for ARM in A B C; do
      OPENAI_API_BASE=http://localhost:8741/v1 OPENAI_API_KEY=dummy TB2_N_TASKS="$TB2_N" \
        "$HARBOR_PY" scripts/latest/tb2_harbor_bridge.py \
        --arm "$ARM" --iters "$ITERS" --model "openai/$MODEL" --n-tasks "$TB2_N" \
        > "run_${MODEL}_tb2_${ARM}.log" 2>&1
      echo "[tb2-serial] arm $ARM finished rc=$? $(date)" >> "run_${MODEL}_tb2_serial.log"
    done
  ) &
  echo "    [tb2] arms A->B->C serialized, PID $!  -> run_${MODEL}_tb2_{A,B,C}.log"
fi

echo "==> all 4 benchmarks launched for $MODEL. Tail: tail -f run_${MODEL}_*.log"
echo "==> gate when done: python scripts/latest/soft_stats.py experiments_results/latest_evolving/$MODEL"
