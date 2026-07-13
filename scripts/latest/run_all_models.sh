#!/usr/bin/env bash
# ============================================================================
# Run the A/B/C experiment across all backbone models, via the CodeBuddy SDK.
#
#   7 models x 4 benchmarks (gaia, gaia2, locomo, tau2) x {A,B,C}.
#
# Each model writes to its own tree so they never overwrite each other:
#   experiments_results/latest/<model>/<benchmark>/{trace.jsonl,report.json}
#
# Usage (from anywhere):
#   bash scripts/latest/run_all_models.sh
#   TASK_LIMIT=100 bash scripts/latest/run_all_models.sh      # tasks per benchmark
#   RESUME=0 bash scripts/latest/run_all_models.sh            # force fresh (default resumes)
#
# All model ids verified via CodeBuddy SDK probe (2026-07-03).
# Claude-Opus-4.6 is intentionally left out (placeholder column in the paper; not run).
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

# CodeBuddy model ids, ordered CHEAPEST -> MOST EXPENSIVE (output price, RMB/MTok,
# USD x7). Running cheap models first surfaces pipeline bugs before the expensive
# ones burn budget. First entry is the in-house primary.
MODELS=(
  "hy3-preview"     # HY3-preview      in-house  (PRIMARY)            -- confirmed
  "deepseek-v4-pro"     # DeepSeek-V4-Pro  out ~¥6                        -- confirmed
  "minimax-m2.7"        # MiniMax-M2.7     out ~¥8.4                      -- confirmed
  "glm-5.1"             # GLM-5.1          out ~¥24                       -- confirmed
  "kimi-k2.6"           # Kimi-K2.6        out ~¥27                       -- confirmed
  "gemini-3.1-pro"      # Gemini-3.1-Pro   out ~¥84   ($12/MTok ×7)       -- confirmed
  "gpt-5.5"             # GPT-5.5          out ~¥210  ($30/MTok ×7)       -- confirmed
  # "claude-4.6-opus"   # Claude-Opus-4.6  out ~¥175  -- left blank (paper placeholder column)
)

# ── OpenRouter roster (opt-in: OPENROUTER=1) ────────────────────────────────
# Runs the free-tier OpenAI-compatible backbones instead of the CodeBuddy ids
# above; leaves the default (CodeBuddy) path untouched. Needs OPENROUTER_API_KEY
# in .env. NOTE: the free tier is heavily rate-limited, so a full 100-task x 3-arm
# x 3-iter grid will throttle (429) and run slowly; fine for smaller TASK_LIMIT.
if [ "${OPENROUTER:-0}" = "1" ]; then
  export LLM_PROVIDER=openrouter
  MODELS=(
    "hy3"             # tencent/hy3:free                       (reasoning)
    "nemotron-super"  # nvidia/nemotron-3-super-120b-a12b:free
  )
  echo "[roster] OpenRouter free-tier: ${MODELS[*]}"
fi

# Resume by default so a crash mid-sweep doesn't lose finished models.
export RESUME="${RESUME:-1}"

# Iteration chains. Patch memory is feedback across iterations of the SAME task, so on
# the independent-task benchmarks (gaia, gaia2, tau2) a single pass leaves
# every chain a singleton: B and C inject nothing and collapse onto A. Default to 3 so
# memory actually threads across a task's own iterations; override with ITER_CHAIN=1
# only when you explicitly want the A-only / no-memory baseline.
export ITER_CHAIN="${ITER_CHAIN:-3}"

for M in "${MODELS[@]}"; do
  echo ""
  echo "########################################################################"
  echo "#  MODEL: ${M}"
  echo "########################################################################"
  # Non-HY3 models may need a different CodeBuddy internet environment than
  # 'ioa'; override CODEBUDDY_INTERNET_ENVIRONMENT here per model if so.
  if CODEBUDDY_MODEL="${M}" python scripts/latest/latest_runner.py; then
    echo "  [done] ${M}"
  else
    echo "  [FAILED] ${M} (rc=$?) -- continuing to next model"
  fi
  # External-baseline pass on the SAME backbone (opt-in: set EXTERNAL_MEMS).
  # Pairs against the existing No-Mem rows; the raw-patch arm is not rerun.
  if [ -n "${EXTERNAL_MEMS:-}" ]; then
    echo "  [external baselines: ${EXTERNAL_MEMS}] ${M}"
    EXTERNAL_MEMS="${EXTERNAL_MEMS}" ARMS= RESULTS_BASE="${RESULTS_BASE:-latest_evolving}" \
      ITER_MUTATE=1 ITER_FEEDBACK=self CODEBUDDY_MODEL="${M}" \
      python scripts/latest/latest_runner.py \
      || echo "  [external FAILED] ${M} -- continuing"
  fi
done

echo ""
echo "Sweep complete. Per-model results under experiments_results/latest/<model>/<benchmark>/"
echo "Aggregate with: python scripts/latest/analyze_results.py experiments_results/latest/<model>"
