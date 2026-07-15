#!/usr/bin/env bash
# ============================================================================
# MemorySkillGenerator 全量实验一键启动脚本
# 支持 Hy3-FP8 和 Qwen3-VL-235B-FP8 两个 backbone
#
# 前提条件 (GPU机器上):
#   1. ceph 已挂载到 /apdcephfs/private_yizhouyang
#   2. 模型权重已就绪:
#      - /apdcephfs/private_yizhouyang/Hy3-FP8/           (279G, 295B MoE, 21B active)
#      - /apdcephfs/private_yizhouyang/Qwen3-VL-235B-A22B-Instruct-FP8/ (222G, 235B MoE, 22B active)
#      - /apdcephfs/private_yizhouyang/Qwen3-30B-A3B-Instruct-2507/     (57G, 30B MoE, 3B active)
#   3. GPU 环境: P800 4卡(96GB×4=384GB) 或 H20 4卡(96GB×4=384GB)
#   4. vLLM 0.24+ 已安装
#
# 用法:
#   bash run_all_experiments.sh                    # 全量: 3个模型 × 4个benchmark × A/B/C
#   MODEL=hy3 bash run_all_experiments.sh           # 只跑 Hy3
#   MODEL=qwen3-235b bash run_all_experiments.sh    # 只跑 Qwen3-VL-235B
#   MODEL=qwen3-30b bash run_all_experiments.sh     # 只跑 Qwen3-30B
#   STEP=deploy bash run_all_experiments.sh         # 只启动 vLLM
#   STEP=smoke  bash run_all_experiments.sh         # 2-task smoke test
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(dirname "$(dirname "$(readlink -f "$0")")")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 模型配置 ───────────────────────────────────────────────────────────────
declare -A MODEL_PATH MODEL_TP MODEL_QUANT MODEL_VRAM MODEL_ACTIVE
MODEL_PATH[hy3]="/apdcephfs/private_yizhouyang/Hy3-FP8"
MODEL_TP[hy3]=4
MODEL_QUANT[hy3]="fp8"
MODEL_VRAM[hy3]=279
MODEL_ACTIVE[hy3]="295B MoE, 21B active, 256K ctx, PURE TEXT"

MODEL_PATH[qwen3-235b]="/apdcephfs/private_yizhouyang/Qwen3-VL-235B-A22B-Instruct-FP8"
MODEL_TP[qwen3-235b]=4
MODEL_QUANT[qwen3-235b]="fp8"
MODEL_VRAM[qwen3-235b]=222
MODEL_ACTIVE[qwen3-235b]="235B MoE, 22B active, 256K ctx, VISION-LANGUAGE"

MODEL_PATH[qwen3-30b]="/apdcephfs/private_yizhouyang/Qwen3-30B-A3B-Instruct-2507"
MODEL_TP[qwen3-30b]=1
MODEL_QUANT[qwen3-30b]=""
MODEL_VRAM[qwen3-30b]=57
MODEL_ACTIVE[qwen3-30b]="30B MoE, 3B active, 128 experts, PURE TEXT"

# ── 运行参数 ───────────────────────────────────────────────────────────────
MODEL="${MODEL:-all}"             # hy3 | qwen3-235b | qwen3-30b | all
STEP="${STEP:-all}"               # deploy | smoke | full | all
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"
TASK_LIMIT="${TASK_LIMIT:-100}"
ITER_CHAIN="${ITER_CHAIN:-3}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-20}"
RESULTS_BASE="${RESULTS_BASE:-latest_evolving}"
NO_TB2="${NO_TB2:-0}"

# P800 特殊环境变量
if [ "${GPU_PLATFORM:-}" = "p800" ] || [ -e /dev/xpuctrl ]; then
    export PYTORCH_NVML_BASED_CUDA_CHECK=1
    log "P800 XPU platform detected"
fi

# ════════════════════════════════════════════════════════════════════════════
deploy_model() {
    local MODEL_ID="$1"
    local PATH="${MODEL_PATH[$MODEL_ID]}"
    local TP="${MODEL_TP[$MODEL_ID]}"
    local QUANT="${MODEL_QUANT[$MODEL_ID]}"
    local NAME="$MODEL_ID"
    local LOG="/tmp/vllm_${NAME}.log"

    log "━━━ Deploying ${NAME} (${MODEL_ACTIVE[$MODEL_ID]}) ━━━"
    [ -d "$PATH" ] || { err "Model not found: $PATH"; return 1; }
    log "  Path: $PATH ($(du -sh "$PATH" | cut -f1))"
    log "  TP=$TP QUANT=${QUANT:-none}"

    # Kill existing
    lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2

    # vLLM args
    local VLLM_ARGS=(
        "$PATH"
        --served-model-name "$NAME"
        --tensor-parallel-size "$TP"
        --max-model-len "$MAX_LEN"
        --gpu-memory-utilization "$GPU_UTIL"
        --trust-remote-code
        --port "$PORT"
        --dtype auto
    )
    [ -n "$QUANT" ] && VLLM_ARGS+=(--quantization "$QUANT")
    [ "${GPU_PLATFORM:-}" = "p800" ] && VLLM_ARGS+=(--device xpu)

    nohup vllm serve "${VLLM_ARGS[@]}" > "$LOG" 2>&1 &
    log "  vLLM PID=$!  log=$LOG"

    # Wait
    for i in $(seq 1 90); do
        if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            log "  ✅ READY (~$((i*10))s)"
            return 0
        fi
        sleep 10
    done
    err "  ❌ NOT ready after 15min"
    tail -20 "$LOG"
    return 1
}

# ════════════════════════════════════════════════════════════════════════════
smoke_test() {
    local MODEL_ID="$1"
    log "━━━ Smoke: ${MODEL_ID} (2 tasks × gaia) ━━━"

    export LLM_PROVIDER=openrouter
    export OPENROUTER_BASE_URL="http://localhost:${PORT}/v1"
    export OPENROUTER_API_KEY=EMPTY
    export CODEBUDDY_MODEL="$MODEL_ID"

    # Verify chat
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":8}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('Chat OK:', d.get('choices',[{}])[0].get('message',{}).get('content','?')[:50])"

    TASK_LIMIT=2 ITER_CHAIN=1 BENCHMARKS=gaia \
        CODEBUDDY_MODEL="$MODEL_ID" \
        python3 -u scripts/latest/latest_runner.py 2>&1 | tail -5
}

# ════════════════════════════════════════════════════════════════════════════
run_sweep() {
    local MODEL_ID="$1"
    log "━━━ Full Sweep: ${MODEL_ID} ━━━"
    log "  Benchmarks: gaia, gaia2, locomo, terminal_bench_2"
    log "  ITER_CHAIN=${ITER_CHAIN} TASK_LIMIT=${TASK_LIMIT}"
    log "  Results: experiments_results/${RESULTS_BASE}/${MODEL_ID}/"

    export LLM_PROVIDER=openrouter
    export OPENROUTER_BASE_URL="http://localhost:${PORT}/v1"
    export OPENROUTER_API_KEY=EMPTY
    export CODEBUDDY_MODEL="$MODEL_ID"

    local START=$(date +%s)
    local LOG_PREFIX="run_${MODEL_ID}"

    # gaia + locomo (pure text)
    TASK_LIMIT="$TASK_LIMIT" ITER_CHAIN="$ITER_CHAIN" \
        BENCHMARKS=gaia,locomo TASK_CONCURRENCY="$TASK_CONCURRENCY" \
        RESULTS_BASE="$RESULTS_BASE" CODEBUDDY_MODEL="$MODEL_ID" \
        python3 -u scripts/latest/latest_runner.py 2>&1 | tee "${LOG_PREFIX}_qa.log"

    # gaia2 (ARE)
    TASK_LIMIT="$TASK_LIMIT" ITER_CHAIN="$ITER_CHAIN" \
        BENCHMARKS=gaia2 TASK_CONCURRENCY="$TASK_CONCURRENCY" \
        RESULTS_BASE="$RESULTS_BASE" CODEBUDDY_MODEL="$MODEL_ID" \
        python3 -u scripts/latest/latest_runner.py 2>&1 | tee "${LOG_PREFIX}_gaia2.log"

    # terminal_bench_2
    if [ "$NO_TB2" != "1" ]; then
        bash scripts/latest/run_tb2_official.sh "$MODEL_ID" 2>&1 | tee "${LOG_PREFIX}_tb2.log"
    fi

    local ELAPSED=$(( $(date +%s) - START ))
    log "  Done in $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
}

# ════════════════════════════════════════════════════════════════════════════
main() {
    log "MemorySkillGenerator :: Experiment Launcher"
    log "Models: ${MODEL}"

    local MODELS_TO_RUN=()
    case "$MODEL" in
        all) MODELS_TO_RUN=(hy3 qwen3-235b qwen3-30b) ;;
        *)   MODELS_TO_RUN=("$MODEL") ;;
    esac

    for M in "${MODELS_TO_RUN[@]}"; do
        echo ""
        log "══════════════════ ${M} ══════════════════"
        case "$STEP" in
            deploy) deploy_model "$M" ;;
            smoke)  deploy_model "$M" && smoke_test "$M" ;;
            full)   deploy_model "$M" && run_sweep "$M" ;;
            all)    deploy_model "$M" && smoke_test "$M" && run_sweep "$M" ;;
        esac
    done

    log "━━━ All Done ━━━"
    log "Results: experiments_results/${RESULTS_BASE}/"
    find "experiments_results/${RESULTS_BASE}/" -name "trace.jsonl" 2>/dev/null | while read f; do
        echo "  $(dirname $(dirname $f))/$(basename $(dirname $f)): $(wc -l < $f) rows"
    done
}

main
