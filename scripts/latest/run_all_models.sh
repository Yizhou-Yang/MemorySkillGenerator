#!/usr/bin/env bash
# ============================================================================
# Run the A/B/C experiment across all backbone models, via the CodeBuddy SDK.
#
#   7 models x 4 benchmarks (gaia, gaia2, locomo, terminal_bench_2) x {A,B,C}.
#
# Each model writes to its own tree so they never overwrite each other:
#   experiments_results/latest/<model>/<benchmark>/{trace.jsonl,report.json}
#
# Usage (from anywhere):
#   bash scripts/latest/run_all_models.sh
#   TASK_LIMIT=100 bash scripts/latest/run_all_models.sh      # tasks per benchmark
#   RESUME=0 bash scripts/latest/run_all_models.sh            # force fresh (default resumes)
#
# NOTE: HY3-preview is the in-house headline model. The other CodeBuddy model
# ids below are best-guesses from the console display names — VERIFY them against
# your CodeBuddy console before the real run. In particular, MiniMax was not in
# the model list, so its id is a guess. Claude-Opus-4.6 is intentionally left out
# (placeholder column in the paper; not run).
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

# CodeBuddy model ids, ordered CHEAPEST -> MOST EXPENSIVE (output price, RMB/MTok,
# USD x7). Running cheap models first surfaces pipeline bugs before the expensive
# ones burn budget. First entry is the in-house primary.
MODELS=(
  "hy3-preview-ioa"     # HY3-preview      in-house  (PRIMARY)            -- confirmed
  "deepseek-v4-pro"     # DeepSeek-V4-Pro  out ~Y6                        -- confirmed
  "minimax-m2.7"        # MiniMax-M2.7     out ~Y8.4                      -- VERIFY id (not in console list)
  "glm-5.1"             # GLM-5.1          out ~Y24                       -- VERIFY id
  "kimi-k2.6"           # Kimi-K2.6        out ~Y27                       -- VERIFY id
  "gemini-3.1-pro"      # Gemini-3.1-Pro   out ~Y84   ($12/MTok x7)       -- VERIFY id
  "gpt-5.5"             # GPT-5.5          out ~Y210  ($30/MTok x7)       -- VERIFY id
  # "claude-4.6-opus"   # Claude-Opus-4.6  out ~Y175  -- left blank (paper placeholder column)
)

# Resume by default so a crash mid-sweep doesn't lose finished models.
export RESUME="${RESUME:-1}"

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
done

echo ""
echo "Sweep complete. Per-model results under experiments_results/latest/<model>/<benchmark>/"
echo "Aggregate with: python scripts/latest/analyze_results.py experiments_results/latest/<model>"
