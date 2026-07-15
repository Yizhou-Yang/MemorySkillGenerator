#!/usr/bin/env bash
# ============================================================================
# autopilot — 无人值守跑实验队列,不依赖任何 SSH 会话或 AI 助手存活。
#
# 它做的事:按队列逐项跑 sweep -> 健康检查 -> gate -> 只在 gate 通过时 push。
# 已完成的项直接跳过(幂等),所以可以被 cron 反复拉起来自愈。
#
# 用法(机器上直接跑):
#   setsid nohup bash scripts/latest/autopilot.sh > /dev/null 2>&1 &
# 自愈(推荐,加进 crontab,每 10 分钟确认它活着):
#   */10 * * * * pgrep -f autopilot.sh >/dev/null || (cd <repo> && setsid nohup bash scripts/latest/autopilot.sh >/dev/null 2>&1 &)
#
# 日志写在 ceph 上($LOG),容器回收也不丢。
# ============================================================================
set -u

REPO=/apdcephfs/private_yizhouyang/MemorySkillGenerator
PY=${AUTOPILOT_PY:-/apdcephfs_hzlf/share_1227201/yizhouyang/conda_envs/llm/bin/python}
# 权重走共享 ceph(本地 xfs,6.5GB/s);个人 ceph 是 ceph-fuse 只有 56MB/s,加载会慢 120 倍
WEIGHTS=${AUTOPILOT_WEIGHTS:-/apdcephfs_hzlf/share_1227201/models/Llama-3.3-70B-Instruct-FP8}
LOG=$REPO/autopilot.log
LOCK=/tmp/autopilot.lock

# 绝不写别人的共享 env
case "$PY" in *samzxge*) echo "FATAL: PY 指向 samzxge,拒绝" >&2; exit 1;; esac

exec >>"$LOG" 2>&1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 单实例:cron 拉起时若已在跑就退出
exec 9>"$LOCK"
flock -n 9 || { log "已有实例在跑,退出"; exit 0; }

cd "$REPO" || exit 1

# ── 队列:"模型:benchmark:臂" ────────────────────────────────────────────────
# 只放已验证可跑的组合。llama-33 x gaia2 被排除:实测每个 parser x chat-template
# 组合都是 0 分(模型在多工具下输出格式不稳定),不是配置问题,别再试。
QUEUE=(
  "llama-33:tau2:A,B,C"
  "gpt-oss:tau2:A,B,C"
)

# ── 判断一项是否已完成(幂等的关键)────────────────────────────────────────
is_done(){  # $1=model $2=bench
  $PY - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
m, b = sys.argv[1], sys.argv[2]
p = Path(f"experiments_results/latest_evolving/{m}/{b}/trace.jsonl")
if not p.exists(): sys.exit(1)
rows = [json.loads(l) for l in open(p) if l.strip()]
if not rows: sys.exit(1)
kf = max(int(r.get("iter_total", 1) or 1) for r in rows) - 1
for g in ("no_mem", "raw_patch", "curated_patch"):
    fin = {r.get("task_id") for r in rows
           if r.get("group") == g and int(r.get("iteration", 0) or 0) == kf}
    if len(fin) < 50:          # 三臂都要有足量完成任务才算这项做完
        sys.exit(1)
sys.exit(0)
PY
}

# ── 确保某模型的 vLLM 在线 ────────────────────────────────────────────────
ensure_vllm(){  # $1=model
  local m=$1 port=8000
  curl -sf --max-time 3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q "$m" && return 0
  log "  起 vLLM: $m"
  # 僵尸 vLLM 会占着显存让新进程 OOM(报 "Free memory on device (26/95 GiB) is less than desired")
  pkill -9 -f "vllm.entrypoints" 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 2
  fuser -k /dev/nvidia* 2>/dev/null; sleep 4
  local args=(--model "$WEIGHTS" --served-model-name "$m"
    --tensor-parallel-size "${TP:-1}" --max-model-len 32768
    --gpu-memory-utilization 0.90 --enforce-eager --trust-remote-code
    --port $port --dtype auto)
  # llama 的 compressed-tensors 需要显式声明;parser 实测只有 llama3_json 对
  case "$m" in
    llama-33) args+=(--quantization compressed-tensors
                     --enable-auto-tool-choice --tool-call-parser llama3_json) ;;
    gpt-oss)  args+=(--enable-auto-tool-choice --tool-call-parser pythonic) ;;
  esac
  ( cd /tmp && setsid nohup $PY -m vllm.entrypoints.openai.api_server "${args[@]}" \
      > /tmp/vllm_$m.log 2>&1 < /dev/null & )
  for _ in $(seq 1 60); do   # 共享 ceph 加载约 1-3 分钟
    curl -sf --max-time 3 "http://localhost:$port/v1/models" >/dev/null 2>&1 && { log "  vLLM READY"; return 0; }
    pgrep -f "vllm.entrypoints" >/dev/null || break
    sleep 10
  done
  log "  ✗ vLLM 起不来,根因:"
  grep -oE "(GLIBC_[0-9.]+|ImportError:.*|RuntimeError:.*|NotImplementedError:.*|torch.OutOfMemoryError.*|ValueError: Free memory.*)" \
    /tmp/vllm_$m.log 2>/dev/null | sort -u | head -3
  return 1
}

# ── 数据健康检查:行数涨不代表数据有效 ────────────────────────────────────
health(){  # $1=model $2=bench  -> 0=健康
  $PY - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
m, b = sys.argv[1], sys.argv[2]
p = Path(f"experiments_results/latest_evolving/{m}/{b}/trace.jsonl")
if not p.exists(): print("    健康检查: 无 trace"); sys.exit(1)
rows = [json.loads(l) for l in open(p) if l.strip()]
ans = [r for r in rows if str(r.get("response") or "").strip()]
nz = sum(1 for r in rows if float(r.get("score") or 0))
leak = sum(1 for r in rows if "<|python_tag|>" in str(r.get("response") or ""))
print(f"    健康检查: {len(rows)} 行, 答题 {len(ans)}, 非零分 {nz}, python_tag 泄漏 {leak}")
if ans and nz == 0:
    print("    ✗ 答题行全 0 分 —— harness 死了,不是 0 基线"); sys.exit(1)
if leak:
    print("    ✗ tool-call parser 没生效(原始调用漏进 response)"); sys.exit(1)
sys.exit(0)
PY
}

# ── 主循环 ────────────────────────────────────────────────────────────────
log "════ autopilot 启动 (py=$PY) ════"
for item in "${QUEUE[@]}"; do
  IFS=: read -r MODEL BENCH ARMS <<<"$item"
  log "▶ $MODEL / $BENCH (arms=$ARMS)"

  if is_done "$MODEL" "$BENCH"; then log "  已完成,跳过"; continue; fi
  ensure_vllm "$MODEL" || { log "  跳过(vLLM 起不来)"; continue; }

  log "  跑 sweep..."
  if [ "$BENCH" = tau2 ]; then
    for arm in ${ARMS//,/ }; do
      OPENAI_API_BASE=http://localhost:8000/v1 OPENAI_API_KEY=dummy PYTHONPATH="$REPO" \
        $PY scripts/latest/tau2_bridge.py --arm "$arm" --iters 3 \
        --model "openai/$MODEL" --domain airline,retail --n-tasks 0 \
        >> "$REPO/run_${MODEL}_tau2_${arm}.log" 2>&1
      log "    arm $arm exit=$?"
    done
  else
    # GAIA2_SCENARIO_DIR 必须显式给:loader 的默认路径 /tmp/harbor-datasets/... 不存在,
    # 会静默加载 0 个任务然后打印 "ALL BENCHMARKS COMPLETE"(几分钟就"跑完"= 什么都没跑)
    OPENAI_API_BASE=http://localhost:8000/v1 OPENAI_API_KEY=dummy CODEBUDDY_MODEL="$MODEL" \
    GAIA2_SCENARIO_DIR="$REPO/.datasets/gaia2-cli-loaded" \
    RESULTS_BASE=latest_evolving BENCHMARKS="$BENCH" ITER_CHAIN=3 ITER_MUTATE=1 \
    ITER_FEEDBACK=self RESUME=1 TASK_CONCURRENCY=10 ARMS="$ARMS" TASK_LIMIT=100 \
      $PY -u scripts/latest/latest_runner.py >> "$REPO/run_${MODEL}_${BENCH}.log" 2>&1
    log "    sweep exit=$?"
  fi

  health "$MODEL" "$BENCH" || { log "  ✗ 数据不健康,不 gate 不 push,留待人工"; continue; }

  log "  gate..."
  REQUIRE_C=1 $PY scripts/latest/v2_gate.py "$MODEL" latest_evolving 2>&1 | tail -12
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    git add experiments_results/ 2>/dev/null
    git -c user.email=yizhouyang@tencent.com -c user.name=yizhouyang \
      commit -q -m "experiment($MODEL): $BENCH $ARMS via autopilot" 2>&1 | tail -1
    git push origin main 2>&1 | tail -1
    log "  ✅ gate 通过,已 push"
  else
    log "  ✗ gate FAIL — 不 push"
  fi
done
log "════ 队列跑完 ════"
