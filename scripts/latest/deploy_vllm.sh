#!/usr/bin/env bash
# ============================================================================
# Self-host a model with vLLM (OpenAI-compatible) for the benchmark pipeline.
# Designed for the GPU-window workflow: weights pre-staged on ceph, served once,
# kept alive for the whole booking. Run this INSIDE the GPU container.
#
#   HY3-INT4 on 2xH200 (default):
#     MODEL_PATH=/apdcephfs/<you>/hy3-int4 bash scripts/latest/deploy_vllm.sh
#   HY3-FP8 on 4xH200:
#     MODEL_PATH=/apdcephfs/<you>/hy3-fp8 TP=4 QUANT=fp8 bash scripts/latest/deploy_vllm.sh
#   A clean 70B (no quant) on 2xH200:
#     MODEL_PATH=/apdcephfs/<you>/qwen3-72b TP=2 QUANT= bash scripts/latest/deploy_vllm.sh
#
# NOTE: DeepSeek-V4-Pro (1.6T / ~862GB) does NOT fit 2 or 4 H200 — needs ~8x.
# Keep it on the API (you already have its data); do NOT self-host it here.
# ============================================================================
set -uo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the ceph dir holding the weights}"
TP="${TP:-2}"                         # tensor-parallel = number of GPUs
QUANT="${QUANT:-awq}"                 # int4 checkpoint format: awq | gptq | compressed-tensors
                                      #   fp8 for an FP8 checkpoint; empty '' for BF16 (no quant)
SERVED_NAME="${SERVED_NAME:-hy3}"     # the id you pass as CODEBUDDY_MODEL to the pipeline
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"           # cap context to save KV cache (benchmarks + our history
                                      #   window never need 256K; 32K is plenty)
GPU_UTIL="${GPU_UTIL:-0.92}"
LOG="${VLLM_LOG:-/tmp/vllm_${SERVED_NAME}.log}"

# The tool-call parser MUST match the format the model actually emits. If it
# does not, vLLM leaves the call in `content` as raw text, the agent loop sees
# no tool_calls, no tool ever runs, and every tool-using benchmark scores 0
# while looking perfectly healthy (real responses, no errors).
#   llama-33 on GAIA2 hit exactly this: 460/511 answered rows were raw
#   "<|python_tag|>Cabs__get_ride_history_length()" text and all 600 rows
#   scored 0.0, because this script hardcoded llama3_json for every model.
# Llama 3.1/3.2/3.3 emit the PYTHONIC form (<|python_tag|>f(a=1)) unless you
# also pass the JSON chat template; llama3_json cannot read that form. Pick the
# parser by model family, override with TOOL_PARSER=<name>, TOOL_PARSER=none to
# serve without native tool calling.
TOOL_PARSER="${TOOL_PARSER:-}"
if [ -z "$TOOL_PARSER" ]; then
  case "$(echo "${SERVED_NAME} ${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')" in
    *llama*|*nemotron*) TOOL_PARSER=pythonic ;;   # <|python_tag|>f(a=1)
    *qwen*)             TOOL_PARSER=hermes ;;     # <tool_call>{...}</tool_call>
    *)                  TOOL_PARSER=llama3_json ;;  # hy3 default, unchanged
  esac
fi
TOOL_ARG=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
[ "$TOOL_PARSER" = "none" ] && TOOL_ARG=()

echo "[deploy] model=$MODEL_PATH tp=$TP quant='${QUANT:-none}' name=$SERVED_NAME port=$PORT maxlen=$MAX_LEN tool_parser=$TOOL_PARSER"
command -v vllm >/dev/null || { echo "vllm not installed: pip install vllm"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "MODEL_PATH not found (is ceph mounted?): $MODEL_PATH"; exit 1; }

QUANT_ARG=(); [ -n "$QUANT" ] && QUANT_ARG=(--quantization "$QUANT")

# Serve in the background, survive an IDE disconnect (nohup). setsid detaches it
# from the shell so closing the terminal doesn't kill it.
setsid nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  "${QUANT_ARG[@]}" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  "${TOOL_ARG[@]}" \
  --enforce-eager \
  --port "$PORT" >"$LOG" 2>&1 &
echo "[deploy] vllm starting, pid=$!  log=$LOG"

# Wait for readiness (loading 148GB from ceph into VRAM can take a few minutes).
echo "[deploy] waiting for the endpoint to come up (tail -f $LOG to watch)..."
for i in $(seq 1 120); do        # up to ~20 min
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[deploy] READY after ~$((i*10))s"
    curl -s "http://localhost:${PORT}/v1/models" | head -c 400; echo
    echo ""
    echo "[deploy] point the pipeline at it:"
    echo "  export LLM_PROVIDER=openrouter"
    echo "  export OPENROUTER_BASE_URL=http://localhost:${PORT}/v1"
    echo "  export OPENROUTER_API_KEY=EMPTY"
    echo "  export CODEBUDDY_MODEL=${SERVED_NAME}"
    exit 0
  fi
  sleep 10
done
echo "[deploy] NOT ready after 20 min — check $LOG (likely OOM or bad --quantization)."
tail -30 "$LOG"
exit 1
