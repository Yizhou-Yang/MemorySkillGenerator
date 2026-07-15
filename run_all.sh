#!/usr/bin/env bash
# ============================================================================
# MemorySkillGenerator 全量实验脚本 (H20)
#
# 模型阵容:
#   hy3           Hy3-FP8  295B MoE/21B active  FP8  280G  4卡
#   glm-4.5       GLM-4.5-Air                    BF16 206G  2卡
#   gpt-oss       GPT-OSS-120B                   BF16  61G  1卡
#   llama-70b     Meta-Llama-3-70B               BF16 131G  1卡
#
# 用法:
#   bash run_all.sh                   全量(4模型×4benchmark×A/B/C)
#   MODEL=hy3   bash run_all.sh       单模型
#   STEP=deploy bash run_all.sh       只启动vLLM
#   STEP=smoke  bash run_all.sh       2-task smoke
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 模型配置 ──────────────────────────────────────────────
CEPH=/apdcephfs/private_yizhouyang

declare -A M_PATH M_TP M_QUANT M_DESC
M_PATH[hy3]="$CEPH/Hy3-FP8";           M_TP[hy3]=4; M_QUANT[hy3]="fp8"
M_DESC[hy3]="Hy3-FP8 | 295B MoE/21B active | 4×H20"

M_PATH[glm-4.5]="$CEPH/GLM-4.5-Air";   M_TP[glm-4.5]=2; M_QUANT[glm-4.5]=""
M_DESC[glm-4.5]="GLM-4.5-Air | 2×H20"

M_PATH[gpt-oss]="$CEPH/GPT-OSS-120B";  M_TP[gpt-oss]=1; M_QUANT[gpt-oss]=""
M_DESC[gpt-oss]="GPT-OSS-120B | 1×H20"

M_PATH[llama-33]="$CEPH/Llama-3.3-70B-Instruct-FP8"; M_TP[llama-33]=1; M_QUANT[llama-33]="fp8"
M_DESC[llama-33]="Llama-3.3-70B-FP8 | 1×H20"

# ── 参数 ──────────────────────────────────────────────────
MODEL="${MODEL:-all}"
STEP="${STEP:-all}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"
TASK_LIMIT="${TASK_LIMIT:-100}"
ITER_CHAIN="${ITER_CHAIN:-3}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-20}"
RESULTS_BASE="${RESULTS_BASE:-latest_evolving}"

# ═══════════════════════════════════════════════════════════
deploy_model() {
    local M="$1" PATH="${M_PATH[$M]}" TP="${M_TP[$M]}" QUANT="${M_QUANT[$M]}"
    local LOG="/tmp/vllm_${M}.log"
    log "Deploy: ${M_DESC[$M]}"

    [ -d "$PATH" ] || { err "Not found: $PATH"; return 1; }
    log "  $(du -sh "$PATH" | cut -f1) | TP=$TP | QUANT=${QUANT:-none}"

    lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true; sleep 2

    local ARGS=("$PATH" --served-model-name "$M" --tensor-parallel-size "$TP"
                --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL"
                --trust-remote-code --port "$PORT" --dtype auto)
    [ -n "$QUANT" ] && ARGS+=(--quantization "$QUANT")

    nohup vllm serve "${ARGS[@]}" > "$LOG" 2>&1 &
    log "  PID=$!  log=$LOG"

    for i in $(seq 1 90); do
        curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && log "  ✅ READY (~$((i*10))s)" && return 0
        sleep 10
    done
    err "  ❌ Timeout"; tail -20 "$LOG"; return 1
}

smoke_test() {
    local M="$1"
    log "Smoke: $M (2 tasks × gaia)"
    export LLM_PROVIDER=openrouter OPENROUTER_BASE_URL="http://localhost:${PORT}/v1"
    export OPENROUTER_API_KEY=EMPTY CODEBUDDY_MODEL="$M"

    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Chat:', d.get('choices',[{}])[0].get('message',{}).get('content','?'))"

    TASK_LIMIT=2 ITER_CHAIN=1 BENCHMARKS=gaia \
        CODEBUDDY_MODEL="$M" python3 -u scripts/latest/latest_runner.py 2>&1 | tail -3
}

run_sweep() {
    local M="$1"
    log "Sweep: $M (gaia,gaia2,locomo,tb2 | chain=$ITER_CHAIN tasks=$TASK_LIMIT)"
    export LLM_PROVIDER=openrouter OPENROUTER_BASE_URL="http://localhost:${PORT}/v1"
    export OPENROUTER_API_KEY=EMPTY CODEBUDDY_MODEL="$M"
    local START=$(date +%s)

    for bench in gaia,locomo gaia2; do
        TASK_LIMIT="$TASK_LIMIT" ITER_CHAIN="$ITER_CHAIN" \
            BENCHMARKS="$bench" TASK_CONCURRENCY="$TASK_CONCURRENCY" \
            RESULTS_BASE="$RESULTS_BASE" CODEBUDDY_MODEL="$M" \
            python3 -u scripts/latest/latest_runner.py 2>&1 | tee "run_${M}_$(echo $bench | tr ',' '_').log"
    done

    [ "${NO_TB2:-0}" != "1" ] && bash scripts/latest/run_tb2_official.sh "$M" 2>&1 | tee "run_${M}_tb2.log"

    log "  Done: $(( ($(date +%s)-START)/60 ))min"
}

# ═══════════════════════════════════════════════════════════
log "MemorySkillGenerator on H20"
log "Model: $MODEL | Step: $STEP"

MODELS=()
case "$MODEL" in
    all) MODELS=(hy3 glm-4.5 gpt-oss llama-33) ;;
    *)   MODELS=("$MODEL") ;;
esac

for M in "${MODELS[@]}"; do
    echo ""; log "═══ $M ($(du -sh "${M_PATH[$M]}" 2>/dev/null | cut -f1)) ═══"
    case "$STEP" in
        deploy) deploy_model "$M" ;;
        smoke)  deploy_model "$M" && smoke_test "$M" ;;
        full)   deploy_model "$M" && run_sweep "$M" ;;
        all)    deploy_model "$M" && smoke_test "$M" && run_sweep "$M" ;;
    esac
done

log "All done. Results: experiments_results/${RESULTS_BASE}/"
find "experiments_results/${RESULTS_BASE}/" -name "trace.jsonl" 2>/dev/null | while read f; do
    echo "  $(dirname $(dirname $f))/$(basename $(dirname $f)): $(wc -l < $f) rows"
done
