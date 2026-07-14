#!/usr/bin/env bash
# Launch ALL 4 benchmarks for ONE model together, under the publishable
# evolving protocol:
#   gaia / gaia2 / locomo  -> main sweep (latest_runner, skillforge env)
#   tau2                   -> official tau2-bench (bridge, NO Docker; retires TB2,
#                             which is under scripts/latest/obsolete/ — Docker-heavy)
#
# The QA sweep and tau2 background independently. tau2 gets its feedback from the
# final DB-state / action checks (env-observable) and does not mutate; the QA
# benchmarks use ITER_MUTATE variants + self-assessed feedback. Nothing here ever
# feeds a gold label into memory (reporting stays gold).
#
# Usage:
#   bash scripts/latest/run_all_benchmarks.sh <MODEL>
#   MODEL is the CodeBuddy id. Paper roster (5 backbones):
#     PRIMARY (free, run now, full ITER_CHAIN=3):
#       glm-5.1 · kimi-2.6 · hy3   (=hunyuan3, A/B kept, rerun C)
#     TABLE-FILL (paid, ~Aug, LEAN ok):
#       gpt-5.3-codex (~Y1500) · claude-haiku-4.5 (~Y4000)
#   The 3 free primaries alone give ~900 pooled pairs (MDE~3.3pp).
# Env overrides:
#   SKILLFORGE_PY               main-sweep interpreter (default: skillforge conda)
#   TAU2_PY                     tau2 interpreter (default: SKILLFORGE_PY; install
#                               tau2-bench into that env so the custom agent and
#                               our deps share ONE interpreter)
#   OPENAI_API_BASE             tau2's OpenAI endpoint (self-host vLLM); unset ->
#                               falls back to the CodeBuddy OAI proxy
#   GAIA2_SCENARIO_DIR          persistent gaia2-cli path
#   TAU2_N_TASKS                tau2 tasks/iter/domain (default 30, matched across models)
#   TAU2_DOMAINS                comma list of domains (default airline,retail)
#   GAIA2_SPLIT_WEIGHTS         per-split counts (default weights the dynamic
#                               splits, adaptability/time, where the thesis
#                               predicts the effect; declared in the paper)
#   TASK_CONCURRENCY            main-sweep global slots (default 12; two models
#                               on one box should each use 12 to stay < the ~24
#                               internal-API ceiling)
#   NO_TAU2=1                   skip tau2 (QA-only run)
#   Separate invocations (NOT part of this launcher):
#     external baselines:  EXTERNAL_MEMS=mem0,amem ARMS= RESULTS_BASE=latest_evolving \
#                            ITER_MUTATE=1 ITER_FEEDBACK=self ITER_CHAIN=3 RESUME=1 \
#                            CODEBUDDY_MODEL=<m> python scripts/latest/latest_runner.py
#                          (pairs against existing no_mem rows; raw_patch NOT rerun)
#     pass@k control:      PASSK=3 ITER_CHAIN=1 ARMS=A RESULTS_BASE=latest_evolving \
#                            CODEBUDDY_MODEL=<m> python scripts/latest/latest_runner.py
#                          then: python scripts/latest/passk_report.py <m>
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
command -v "$SKILLFORGE_PY" >/dev/null 2>&1 || SKILLFORGE_PY="$(command -v python3)"
TAU2_PY="${TAU2_PY:-$SKILLFORGE_PY}"   # tau2-bench installed into the same env
DATASET="${GAIA2_SCENARIO_DIR:-$REPO/.datasets/gaia2-cli-loaded}"
TAU2_N="${TAU2_N_TASKS:-30}"
TAU2_DOMAINS="${TAU2_DOMAINS:-airline,retail}"
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
  RESUME=1 \
  GAIA2_SPLIT_WEIGHTS="$G2W" \
  BENCHMARKS=gaia,gaia2,locomo GAIA2_SCENARIO_DIR="$DATASET" \
  CODEBUDDY_MODEL="$MODEL" TASK_CONCURRENCY="$CONC" \
  nohup "$SKILLFORGE_PY" -u scripts/latest/latest_runner.py \
  > "run_${MODEL}_qa.log" 2>&1 &
echo "    [qa]  gaia/gaia2/locomo  PID $!  -> run_${MODEL}_qa.log"

# ── 2) tau2-bench (A/B/C arms, SERIALIZED, NO Docker) ──
# Arms run SEQUENTIALLY inside one background subshell (B/C depend on their
# previous iteration's store anyway); each arm still uses tau2's own internal
# concurrency, so wall-clock cost is small. tau2 speaks OpenAI via LiteLLM:
# self-host sets OPENROUTER_BASE_URL to your vLLM endpoint (port 8000);
# otherwise falls back to CodeBuddy OAI proxy (port 8741).
if [ "${NO_TAU2:-0}" != "1" ]; then
  # Priority: self-hosted vLLM (OPENROUTER_BASE_URL) > explicit OPENAI_API_BASE > CodeBuddy proxy
  if [ -n "${OPENROUTER_BASE_URL:-}" ] && curl -sf localhost:8000/health >/dev/null 2>&1; then
    OAI_BASE="${OPENROUTER_BASE_URL}"
    echo "    [tau2] using self-hosted vLLM: $OAI_BASE"
  elif [ -n "${OPENAI_API_BASE:-}" ]; then
    OAI_BASE="${OPENAI_API_BASE}"
  elif curl -sf localhost:8741/v1/models >/dev/null 2>&1; then
    OAI_BASE="http://localhost:8741/v1"
  else
    nohup "$SKILLFORGE_PY" scripts/latest/codebuddy_oai_proxy.py > proxy.log 2>&1 &
    echo "    [tau2] started OAI proxy PID $! (waiting 5s)"; sleep 5
    OAI_BASE="http://localhost:8741/v1"
  fi
  # tau2_bridge reads OPENROUTER_BASE_URL (preferred) or OPENAI_API_BASE (fallback)
  # Pass both so the bridge picks the right one
  (
    for ARM in A B C; do
      for DOM in ${TAU2_DOMAINS//,/ }; do
        OPENROUTER_BASE_URL="$OAI_BASE" OPENAI_API_BASE="$OAI_BASE" OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
        TAU2_LOCAL_VLLM=1 \
        PYTHONPATH="${TAU2_ROOT:-}:$REPO" TAU2_N_TASKS="$TAU2_N" \
          "$TAU2_PY" scripts/latest/tau2_bridge.py \
          --arm "$ARM" --iters "$ITERS" --model "openai/$MODEL" \
          --domain "$DOM" --n-tasks "$TAU2_N" \
          >> "run_${MODEL}_tau2_${ARM}.log" 2>&1
        echo "[tau2-serial] arm $ARM domain $DOM rc=$? $(date)" >> "run_${MODEL}_tau2_serial.log"
      done
    done
  ) &
  echo "    [tau2] arms A->B->C serialized, PID $!  -> run_${MODEL}_tau2_{A,B,C}.log"
fi

echo "==> all benchmarks launched for $MODEL. Tail: tail -f run_${MODEL}_*.log"
echo "==> gate when done: python scripts/latest/soft_stats.py experiments_results/latest_evolving/$MODEL"
