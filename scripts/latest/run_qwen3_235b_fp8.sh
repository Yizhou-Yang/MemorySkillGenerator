#!/usr/bin/env bash
# ============================================================================
# MemorySkillGenerator 全量实验: Qwen3-VL-235B-A22B-Instruct-FP8 on 4×H20
#
# 用法 (在 GPU 机器上,ceph 已挂载):
#   bash scripts/latest/run_qwen3_235b_fp8.sh
#
# 分步执行:
#   STEP=deploy bash scripts/latest/run_qwen3_235b_fp8.sh    # 只启动vLLM
#   STEP=smoke  bash scripts/latest/run_qwen3_235b_fp8.sh    # 2-task smoke test
#   STEP=full   bash scripts/latest/run_qwen3_235b_fp8.sh    # 全量实验
#
# 环境变量覆盖:
#   TASK_LIMIT=100    ITER_CHAIN=3    (默认)
#   TASK_LIMIT=10     ITER_CHAIN=1    (快速验证)
#   TASK_CONCURRENCY=20              (并发数, 默认30)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

# ── 模型配置 ───────────────────────────────────────────────────────────────
MODEL_PATH="/apdcephfs/private_yizhouyang/Qwen3-VL-235B-A22B-Instruct-FP8"
SERVED_NAME="qwen3-235b"
TP=4                           # 4×H20
QUANT="fp8"                    # FP8 checkpoint
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"
PORT="${PORT:-8000}"
VLLM_LOG="/tmp/vllm_${SERVED_NAME}.log"
# Without a tool-call parser vLLM never turns the model's tool syntax into
# tool_calls: it stays in `content` as raw text, no tool runs, and every
# tool-using benchmark (GAIA2/tau2) scores 0 while looking healthy. This is what
# killed the llama-33 GAIA2 sweep (all 600 rows 0.0). Qwen emits the Hermes
# <tool_call>{...}</tool_call> form.
TOOL_PARSER="${TOOL_PARSER:-hermes}"

# ── 实验配置 ───────────────────────────────────────────────────────────────
TASK_LIMIT="${TASK_LIMIT:-100}"
ITER_CHAIN="${ITER_CHAIN:-3}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-30}"
RESULTS_BASE="${RESULTS_BASE:-latest_evolving}"
STEP="${STEP:-all}"

# ── 颜色 ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# ════════════════════════════════════════════════════════════════════════════
# Step 1: Deploy vLLM
# ════════════════════════════════════════════════════════════════════════════
deploy_vllm() {
    log "━━━ Deploy vLLM: ${SERVED_NAME} ━━━"

    [ -d "$MODEL_PATH" ] || { err "MODEL_PATH not found: $MODEL_PATH (ceph mounted?)"; exit 1; }
    du -sh "$MODEL_PATH"

    # Check if GPU available
    if ! command -v nvidia-smi &>/dev/null; then
        err "No GPU detected (nvidia-smi not found). This script must run on a GPU machine."
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -$TP
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -lt "$TP" ]; then
        err "Need ${TP} GPUs, found ${GPU_COUNT}"
        exit 1
    fi

    # Kill any existing vLLM on this port
    if lsof -ti:$PORT &>/dev/null; then
        log "Killing old vLLM on port ${PORT}..."
        kill -9 $(lsof -ti:$PORT) 2>/dev/null || true; sleep 2
    fi

    # Serve
    log "Starting vLLM: MODEL=$MODEL_PATH TP=$TP QUANT=$QUANT NAME=$SERVED_NAME"
    nohup vllm serve "$MODEL_PATH" \
        --served-model-name "$SERVED_NAME" \
        --tensor-parallel-size "$TP" \
        --quantization "$QUANT" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --trust-remote-code \
        --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
        --port "$PORT" \
        --dtype auto \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    log "vLLM PID=$VLLM_PID  log=$VLLM_LOG"

    # Wait for readiness (222GB from ceph → VRAM ~3-5 min)
    log "Waiting for endpoint (loading 222GB into VRAM)..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            log "READY after ~$((i*10))s"
            curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d,indent=2)[:500])"
            return 0
        fi
        sleep 10
    done
    err "vLLM NOT ready after 10 min. Check log:"
    tail -30 "$VLLM_LOG"
    exit 1
}

# ════════════════════════════════════════════════════════════════════════════
# Step 2: Smoke test (2 tasks per benchmark)
# ════════════════════════════════════════════════════════════════════════════
smoke_test() {
    log "━━━ Smoke Test (2 tasks) ━━━"

    export LLM_PROVIDER=openrouter
    export OPENROUTER_BASE_URL="http://localhost:${PORT}/v1"
    export OPENROUTER_API_KEY=EMPTY
    export CODEBUDDY_MODEL="$SERVED_NAME"

    # Verify endpoint responds to chat
    log "Verifying chat endpoint..."
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"$SERVED_NAME"'","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('RESPONSE:', d.get('choices',[{}])[0].get('message',{}).get('content','ERROR')[:200])"
    echo

    log "Running 2-task smoke on gaia..."
    TASK_LIMIT=2 ITER_CHAIN=1 BENCHMARKS=gaia \
        CODEBUDDY_MODEL="$SERVED_NAME" \
        python3 -u scripts/latest/latest_runner.py 2>&1 | tail -10

    if [ -f "experiments_results/${RESULTS_BASE}/${SERVED_NAME}/gaia/trace.jsonl" ]; then
        log "Smoke OK - trace.jsonl found"
        head -1 "experiments_results/${RESULTS_BASE}/${SERVED_NAME}/gaia/trace.jsonl" | python3 -m json.tool 2>/dev/null
    else
        warn "Smoke may have failed - no trace.jsonl found"
    fi
}

# ════════════════════════════════════════════════════════════════════════════
# Step 3: Full experiment sweep
# ════════════════════════════════════════════════════════════════════════════
run_full_sweep() {
    log "━━━ Full Experiment: Qwen3-VL-235B-FP8 ━━━"
    log "   Benchmarks: gaia, gaia2, locomo"
    log "   ITER_CHAIN=${ITER_CHAIN}  TASK_LIMIT=${TASK_LIMIT}  CONCURRENCY=${TASK_CONCURRENCY}"
    log "   Results: experiments_results/${RESULTS_BASE}/${SERVED_NAME}/"

    export LLM_PROVIDER=openrouter
    export OPENROUTER_BASE_URL="http://localhost:${PORT}/v1"
    export OPENROUTER_API_KEY=EMPTY
    export CODEBUDDY_MODEL="$SERVED_NAME"

    START_TIME=$(date +%s)

    # QA benchmarks (gaia, gaia2, locomo) - via latest_runner
    # NOTE: gaia2 requires ARE/Docker
    log "[1/3] Running gaia + locomo (pure text)..."
    TASK_LIMIT="$TASK_LIMIT" ITER_CHAIN="$ITER_CHAIN" \
        BENCHMARKS=gaia,locomo \
        TASK_CONCURRENCY="$TASK_CONCURRENCY" \
        RESULTS_BASE="$RESULTS_BASE" \
        CODEBUDDY_MODEL="$SERVED_NAME" \
        python3 -u scripts/latest/latest_runner.py 2>&1 | tee "run_${SERVED_NAME}_qa.log"
    RC1=$?

    log "[2/3] Running gaia2 (ARE agent)..."
    TASK_LIMIT="$TASK_LIMIT" ITER_CHAIN="$ITER_CHAIN" \
        BENCHMARKS=gaia2 \
        TASK_CONCURRENCY="$TASK_CONCURRENCY" \
        RESULTS_BASE="$RESULTS_BASE" \
        CODEBUDDY_MODEL="$SERVED_NAME" \
        python3 -u scripts/latest/latest_runner.py 2>&1 | tee "run_${SERVED_NAME}_gaia2.log"
    RC2=$?

    ELAPSED=$(( $(date +%s) - START_TIME ))
    log "━━━ Sweep Complete ━━━"
    log "   Elapsed: $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
    log "   RC: gaia/locomo=$RC1  gaia2=$RC2"

    # Gate check
    log "━━━ Quality Gate ━━━"
    python3 scripts/latest/soft_stats.py "experiments_results/${RESULTS_BASE}/${SERVED_NAME}" 2>/dev/null || true

    log "━━━ Results ━━━"
    find "experiments_results/${RESULTS_BASE}/${SERVED_NAME}" -name "trace.jsonl" | while read f; do
        lines=$(wc -l < "$f")
        dir=$(dirname "$f")
        bench=$(basename "$dir")
        echo "  $bench: $lines rows"
    done
}

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
log "MemorySkillGenerator :: Qwen3-VL-235B-FP8 on 4×H20"
log "Step: ${STEP}"
echo

case "$STEP" in
    deploy)
        deploy_vllm
        ;;
    smoke)
        smoke_test
        ;;
    full)
        run_full_sweep
        ;;
    all)
        deploy_vllm
        echo
        smoke_test
        echo
        run_full_sweep
        ;;
    *)
        err "Unknown STEP='$STEP'. Use: deploy | smoke | full | all"
        exit 1
        ;;
esac

log "Done."
