#!/usr/bin/env bash
# ============================================================================
# autopilot — 无人值守跑实验队列,不依赖任何 SSH 会话或 AI 助手存活。
#
# 按队列逐项跑 sweep -> 健康检查 -> gate -> 只在 gate 通过时 push。已完成的项
# 直接跳过(幂等),所以可以被 cron 反复拉起来自愈。
#
# 用法(一个 worker = 一个模型 + 一组显卡 + 一个端口):
#   WORKER=g1a AUTOPILOT_MODELS=gpt-oss AUTOPILOT_CUDA=0,1 AUTOPILOT_PORT=8001 \
#     setsid nohup bash scripts/latest/autopilot.sh > /dev/null 2>&1 &
#
# ── 2026-07-15 多写者事故后的硬约束(改之前先读)──────────────────────────
# 1. gpu1 和 gpu2 挂的是同一个 ceph 仓库(同 inode),trace.jsonl 是同一个文件。
#    两台机器各跑一份 autopilot = 同一个 trace 被两个进程追加,行会交错、
#    code_rev 会混、同一臂里混进不同 parser 的结果。/tmp 里的锁是机器本地的,
#    拦不住跨机器。所以:每个 (model,bench) 在 ceph 上有独占锁,谁抢到谁跑。
# 2. 每个模型的 parser 必须钉死且开跑前校验。ensure_vllm 以前只用 /v1/models
#    确认"名字对得上"就复用,于是一台手工起的 pythonic server 会被静默复用。
#    llama-33 配 pythonic 时工具调用根本不执行,DB 没被动过 => db_match=true =>
#    reward 1.0:**什么都不做反而满分**。所以 parser 不对必须重起。
# 3. tau2 的参数(n-tasks/num-trials)必须三臂一致,钉在 TAU2_ARGS 里。
# 4. 一个 worker 只伺候一个模型,显卡/端口由 env 指定,锁也带 worker 名 ——
#    否则同机第二个 worker 会被单实例锁挡掉,显卡白白空着。
# ============================================================================
set -u

REPO=/apdcephfs/private_yizhouyang/MemorySkillGenerator
PY=${AUTOPILOT_PY:-/apdcephfs_hzlf/share_1227201/yizhouyang/conda_envs/llm/bin/python}
LOG=$REPO/autopilot.log
WORKER=${WORKER:-w0}
LOCK=/tmp/autopilot_${WORKER}.lock

# 绝不写别人的共享 env
case "$PY" in *samzxge*) echo "FATAL: PY 指向 samzxge,拒绝" >&2; exit 1;; esac

exec >>"$LOG" 2>&1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$WORKER] $*"; }

# 单实例:同名 worker 只能有一个(不同 worker 互不影响)
exec 9>"$LOCK"
flock -n 9 || { log "已有同名 worker 在跑,退出"; exit 0; }

cd "$REPO" || exit 1

# ── 方法配置(整晚必须钉死;混了就是两种方法,G6 会硬失败)────────────────
# C_POLICY=meta:arm C 只用 store 里的客观 metadata 排序 —— 版本谱系(哪一版
# 取代了哪一版)+ 执行记录(patch_history 里 tool_sequence 的增删)。不问任何
# 模型,不需要 oracle/gold,部署时也拿得到。w_c 只在 provenance 是 gold/env 时
# 才参与,ITER_FEEDBACK=self 下它是"自评穿了个数字的外衣",直接跳过。
# 论文预注册的确认性对比是 guarded(判断出内容,metadata 把关背书;背书需要
# grounded 分数或外部 critic 这第二把钥匙,自评永不背书)。没有外部 critic 时
# guarded 依旧成立:QA 上没有条目会被背书(中性渲染),tau2 上 env 分数就是
# 第二把钥匙 —— 那是它的主场。CRITIC_MODEL/CRITIC_BASE_URL/CRITIC_API_KEY
# 若在环境里会被 llm_client 自动接走。
export C_POLICY=${C_POLICY:-guarded}
export ITER_FEEDBACK=${ITER_FEEDBACK:-self}
# 确认性运行的方差控制:三臂统一贪心解码(GEN_TEMPERATURE=0)。配对检验的方差
# 里一大块是采样运气;归零后 delta 只剩"注入内容不同"的差。这也是为什么新政策
# 的 A/B 必须在同一 base 重跑 —— 拿旧的热采样 B 配对,温度效应会混进估计。
export GEN_TEMPERATURE=${GEN_TEMPERATURE:-0}
# 政策隔离:非 judgment 写自己的 RESULTS_BASE(G6 禁止同 trace 混政策)。
# 变体文件指回主 sweep 的,新旧政策在同一批变异任务上可比、臂间严格配对。
if [ "$C_POLICY" = judgment ]; then RUN_BASE=latest_evolving
else RUN_BASE="${C_POLICY}_v1"
     export MUTATIONS_PATH=$REPO/experiments_results/latest_evolving/mutations.json
fi

# ── 每个模型的部署参数(parser 钉死,别改)────────────────────────────────
model_parser(){ case "$1" in llama-33) echo llama3_json;; gpt-oss) echo openai;; *) echo llama3_json;; esac; }
model_weights(){
  case "$1" in
    llama-33) echo /apdcephfs_hzlf/share_1227201/models/Llama-3.3-70B-Instruct-FP8;;
    gpt-oss)  echo /apdcephfs_hzlf/share_1227201/models/GPT-OSS-120B;;   # 共享ceph(6.5GB/s),不是个人ceph(56MB/s)
  esac
}
# worker 自己的显卡/端口/TP
CUDA=${AUTOPILOT_CUDA:-0}
PORT=${AUTOPILOT_PORT:-8000}
TP=${AUTOPILOT_TP:-$(echo "$CUDA" | tr ',' '\n' | grep -c .)}

# tau2 三臂必须完全一致的参数
# 三域全开(airline 上限 50 取 40,retail/telecom 各 40):stratum 60->120 题,
# 配对 SE 缩 ~sqrt(2)。tau2 是唯一 env-grounded 的基准,guarded 的主场。
TAU2_ARGS=(--iters 3 --domain airline,retail,telecom --n-tasks 40 --num-trials 2 --n-concurrent 8)

# ── 队列:"模型:benchmark:臂" ────────────────────────────────────────────────
# gaia/locomo 的 A/B 已有干净数据且新代码没动 A/B 的执行路径(diff 只加字段),
# 所以只补 C(meta);tau2/gaia2 从零跑三臂。
# llama-33 x gaia2 留在队列末尾:历史结论是"每个 parser x chat-template 都 0 分",
# 但那些实验是在 pythonic(工具根本不执行)时代做的,值得用 llama3_json 复核一次;
# 真跑出全 0,health() 会拦下来,不会污染 gate。
# 全部整套 A/B/C:确认性协议是 temperature=0,旧 base 的 A/B 是热采样,拿来
# 配对会把温度效应混进 delta,所以新 base 三臂全重跑、自成一体。
# (旧队列"只补 C"还有第二个死因:is_done 不认政策,看到旧 judgment C 的
# 100 个完成就跳过 —— 最关键的四格根本不会跑。)
QUEUE=(
  "llama-33:gaia:A,B,C"
  "llama-33:locomo:A,B,C"
  "llama-33:tau2:A,B,C"
  "gpt-oss:gaia:A,B,C"
  "gpt-oss:locomo:A,B,C"
  "gpt-oss:tau2:A,B,C"
  "gpt-oss:gaia2:A,B,C"
  "llama-33:gaia2:A,B,C"
)
MODELS=${AUTOPILOT_MODELS:-}

# ── 判断一项是否已完成(幂等的关键)────────────────────────────────────────
is_done(){  # $1=model $2=bench  (在 RUN_BASE 里查;C 臂还要求政策匹配)
  $PY - "$1" "$2" "$RUN_BASE" "$C_POLICY" <<'PY'
import json, sys
from pathlib import Path
m, b, base, pol = sys.argv[1:5]
p = Path(f"experiments_results/{base}/{m}/{b}/trace.jsonl")
if not p.exists(): sys.exit(1)
rows = [json.loads(l) for l in open(p) if l.strip()]
if not rows: sys.exit(1)
kf = max(int(r.get("iter_total", 1) or 1) for r in rows) - 1
def row_pol(r):
    return r.get("c_policy") or ("meta" if r.get("c_meta") else "judgment")
for g in ("no_mem", "raw_patch", "curated_patch"):
    fin = {r.get("task_id") for r in rows
           if r.get("group") == g and int(r.get("iteration", 0) or 0) == kf
           and (g != "curated_patch" or row_pol(r) == pol)}
    if len(fin) < 50:
        sys.exit(1)
sys.exit(0)
PY
}

# ── 收回显存 ──────────────────────────────────────────────────────────────
# `pkill -f -- "--served-model-name X"` 只杀得掉 APIServer:worker 叫
# VLLM::Worker_TPn,cmdline 里没有 model 名,活着攥着 ~90G 显存,而且
# nvidia-smi 的进程列表里根本看不到。新 server 只会看到
# "Free memory on device (8.43/95.0 GiB) is less than desired" 然后死。
# fuser -k /dev/nvidia* 是核武器,会杀光本机所有 GPU 进程 —— 同机有别的 worker
# 时绝不能用,所以只在本机没有别的模型 server 时才敢。
reclaim_vram(){  # $1=model
  local m=$1 i free other
  pkill -9 -f -- "--served-model-name $m" 2>/dev/null; sleep 3
  for i in $(seq 1 8); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA" 2>/dev/null | sort -n | head -1)
    [ "${free:-0}" -ge 80000 ] && return 0
    sleep 3
  done
  other=$(ps -eo cmd | grep "[v]llm.entrypoints" | grep -v -- "--served-model-name $m" | wc -l)
  if [ "$other" -gt 0 ]; then
    log "  ! 显存不足但本机还有别的模型 server 活着,不敢 fuser -k;跳过 $m"
    return 1
  fi
  log "  僵尸显存(nvidia-smi 看不见持有者)→ fuser -k /dev/nvidia*"
  fuser -k /dev/nvidia* 2>/dev/null; sleep 5
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA" 2>/dev/null | sort -n | head -1)
  [ "${free:-0}" -ge 80000 ]
}

# ── 确保本 worker 的 vLLM 在线,且 parser 是对的 ──────────────────────────
ensure_vllm(){  # $1=model
  local m=$1 parser weights live
  parser=$(model_parser "$m"); weights=$(model_weights "$m")
  if curl -sf --max-time 3 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$m"; then
    live=$(ps -eo cmd | grep "[v]llm.entrypoints" | grep -- "--port $PORT" | head -1)
    if echo "$live" | grep -q -- "--tool-call-parser $parser"; then
      log "  vLLM $m 已在线且 parser=$parser ✓ (port=$PORT)"; return 0
    fi
    log "  ✗ port=$PORT 上的 $m parser 不对(要 $parser),重起"
  fi
  reclaim_vram "$m" || return 1
  log "  起 vLLM: $m parser=$parser cuda=$CUDA tp=$TP port=$PORT"
  # 个人 ceph 的权重先并行预热:瓶颈是 vLLM 单流 mmap(56MB/s),不是 ceph 带宽
  case "$weights" in
    /apdcephfs/private_yizhouyang/*) ls "$weights"/*.safetensors 2>/dev/null | \
        xargs -P 16 -I{} dd if={} of=/dev/null bs=16M status=none 2>/dev/null;;
  esac
  local args=(--model "$weights" --served-model-name "$m"
    --tensor-parallel-size "$TP" --max-model-len 32768
    --gpu-memory-utilization 0.90 --trust-remote-code
    --enable-auto-tool-choice --tool-call-parser "$parser"
    --port "$PORT" --dtype auto)
  case "$m" in llama-33) args+=(--quantization compressed-tensors --enforce-eager) ;; esac
  ( cd /tmp && CUDA_VISIBLE_DEVICES=$CUDA setsid nohup $PY -m vllm.entrypoints.openai.api_server \
      "${args[@]}" > /tmp/vllm_${WORKER}_$m.log 2>&1 < /dev/null & )
  for _ in $(seq 1 90); do
    curl -sf --max-time 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { log "  vLLM READY"; return 0; }
    sleep 10
  done
  log "  ✗ vLLM 起不来,根因:"
  grep -oE "(GLIBC_[0-9.]+|ImportError:.*|RuntimeError:.*|NotImplementedError:.*|torch.OutOfMemoryError.*|ValueError: Free memory.*)" \
    /tmp/vllm_${WORKER}_$m.log 2>/dev/null | sort -u | head -3
  return 1
}

# ── 数据存活:checkpoint 到备份分支,不受 gate 约束 ────────────────────────
# 2026-07-15 的教训:autopilot 只在 gate 通过时才 push,而 gate 要求 A/B/C 全部
# 跑完(~9h)。机器在跑完前被回收 ⇒ 一整夜的 meta-policy 数据从未离开过机器。
# **数据的持久化绝不能被实验的完整性 gate 住** —— 那是两件事:
#   gate 决定"这个数字能不能进论文",checkpoint 决定"这个数字还在不在"。
# 所以:每跑完一个臂就把 trace 推到备份分支(明确标 UNGATED,不进 main,
# 不会被误当成已验证结果);main 依旧只接受 gate 通过的数据(用户的铁律)。
# 机器随时可能被回收,ceph 上的东西不等于安全。
CKPT_BRANCH="data/checkpoint-$(hostname -s | cut -c1-12)"
checkpoint(){  # $1=model $2=bench $3=arm
  local msg="checkpoint(UNGATED): $1/$2 arm=$3 @$(date '+%m-%d %H:%M') worker=$WORKER"
  # 关键:绝不 commit 到本地 main —— 否则这些 UNGATED commit 会留在 HEAD 上,
  # 等 gate 通过时最后那句 `git push origin main` 会把它们一起推上去,直接违反
  # "main 只收 gated 数据"。所以用 commit-tree 造一个游离 commit,HEAD 不动:
  #   1. 把数据加进 index,write-tree 得到 tree 对象
  #   2. commit-tree 以备份分支现有 tip 为父,造 commit —— 不碰任何分支引用
  #   3. 直接把这个 commit 推到备份分支
  #   4. reset 把 index 还原,好让之后真正的 gated commit 从干净状态开始
  # 只圈本 worker 的地盘:两个 worker 共写一个 ceph 仓库,-A 全目录会把别的
  # worker 跑到一半的行也扫进来。
  git add -A "experiments_results/$RUN_BASE/$1" 2>/dev/null
  git add "experiments_results/latest_evolving/mutations.json" 2>/dev/null || true
  git diff --cached --quiet 2>/dev/null && { log "    checkpoint: 无新数据"; return 0; }
  local tree parent commit
  tree=$(git write-tree 2>/dev/null) || { log "    ! checkpoint: write-tree 失败"; git reset -q; return 0; }
  parent=$(git rev-parse -q --verify "refs/remotes/origin/$CKPT_BRANCH" 2>/dev/null)
  commit=$(printf '%s\n' "$msg" | git -c user.email=yizhouyang@tencent.com -c user.name=yizhouyang \
             commit-tree "$tree" ${parent:+-p "$parent"} -p HEAD 2>/dev/null)
  git reset -q   # index 还原,工作区文件不动;本地 main 全程没被碰过
  [ -n "$commit" ] || { log "    ! checkpoint: commit-tree 失败"; return 0; }
  # push 成功与否直接测它自己的退出码;套管道测到的是管道尾的退出码,恒为 0,
  # 会把失败报成成功(正是"看起来健康"的那类假象)。
  if git push -q origin "$commit:refs/heads/$CKPT_BRANCH" 2>/dev/null; then
    git update-ref "refs/remotes/origin/$CKPT_BRANCH" "$commit"   # 记住新 tip 做下次的父
    log "    ✓ checkpoint 已推到 $CKPT_BRANCH ($1/$2 arm=$3)"
  else
    log "    ! checkpoint push 失败(网络/回收;下一个臂会重推,数据在 ceph 未丢)"
  fi
}

# ── 数据健康检查:行数涨不代表数据有效 ────────────────────────────────────
health(){  # $1=model $2=bench
  RUN_BASE=$RUN_BASE $PY - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path
m, b = sys.argv[1], sys.argv[2]
p = Path(f"experiments_results/{os.environ.get('RUN_BASE','latest_evolving')}/{m}/{b}/trace.jsonl")
if not p.exists(): print("    健康检查: 无 trace"); sys.exit(1)
rows = [json.loads(l) for l in open(p) if l.strip()]
ans = [r for r in rows if str(r.get("response") or "").strip()]
nz = sum(1 for r in rows if float(r.get("score") or 0))
leak = sum(1 for r in rows if "<|python_tag|>" in str(r.get("response") or ""))
revs = sorted({str(r.get("code_rev")) for r in rows})
pol  = sorted({str(r.get("c_policy")) for r in rows if r.get("group") == "curated_patch"})
print(f"    健康检查: {len(rows)} 行, 答题 {len(ans)}, 非零分 {nz}, "
      f"泄漏 {leak}, revs={revs}, C策略={pol}")
if ans and nz == 0:
    print("    ✗ 答题行全 0 分 —— harness 死了,不是 0 基线"); sys.exit(1)
if leak:
    print("    ✗ tool-call parser 没生效(原始调用漏进 response)"); sys.exit(1)
sys.exit(0)
PY
}

# ── 主循环 ────────────────────────────────────────────────────────────────
log "════ 启动 (models=${MODELS:-ALL} cuda=$CUDA port=$PORT tp=$TP C_POLICY=$C_POLICY) ════"
for item in "${QUEUE[@]}"; do
  IFS=: read -r MODEL BENCH ARMS <<<"$item"
  [ -n "$MODELS" ] && ! echo ",$MODELS," | grep -q ",$MODEL," && continue

  # 跨机器/跨 worker 锁:同一个 (model,bench) 全局只能有一个写者
  mkdir -p "$REPO/.locks"
  exec 8>"$REPO/.locks/${MODEL}_${BENCH}.lock"
  if ! flock -n 8; then log "▶ $MODEL/$BENCH 已被别的 worker 占着,跳过"; continue; fi

  log "▶ $MODEL / $BENCH (arms=$ARMS)"
  if is_done "$MODEL" "$BENCH"; then log "  已完成,跳过"; flock -u 8; continue; fi
  ensure_vllm "$MODEL" || { log "  跳过(vLLM 起不来)"; flock -u 8; continue; }

  log "  跑 sweep..."
  if [ "$BENCH" = tau2 ]; then
    # tau2 CLI 带 --auto-resume:残留的 tau2_*_iter*.json 会被"续跑",上一轮
    # (可能是别的 env/parser/参数)的 sims 会被当成本次结果解析进 trace,而
    # code_rev 写的是现在的 —— 行会撒谎,任何 infra 门都发现不了。
    if [ ! -f "experiments_results/$RUN_BASE/$MODEL/tau2/trace.jsonl" ]; then
      n=$(ls -d experiments_results/$RUN_BASE/$MODEL/tau2_*_iter*.json 2>/dev/null | wc -l)
      if [ "$n" -gt 0 ]; then
        rm -rf experiments_results/$RUN_BASE/$MODEL/tau2_*_iter*.json
        rm -f experiments_results/$RUN_BASE/$MODEL/tau2_mem_*.pkl
        log "  清掉 $n 个残留 tau2 artifact(全新开跑)"
      fi
    fi
    for arm in ${ARMS//,/ }; do
      OPENAI_API_BASE=http://localhost:$PORT/v1 OPENAI_API_KEY=dummy PYTHONPATH="$REPO" \
      RESULTS_BASE=$RUN_BASE CODEBUDDY_MODEL="$MODEL" \
        $PY -u scripts/latest/tau2_bridge.py --arm "$arm" "${TAU2_ARGS[@]}" \
        --model "openai/$MODEL" >> "$REPO/run_${MODEL}_tau2_${arm}.log" 2>&1
      log "    arm $arm exit=$?"
      checkpoint "$MODEL" "$BENCH" "$arm"   # 臂级存活点:机器被回收也不丢
    done
  else
    # GAIA2_SCENARIO_DIR 必须显式给:loader 默认路径 /tmp/harbor-datasets/... 不存在,
    # 会静默加载 0 个任务然后打印 "ALL BENCHMARKS COMPLETE"(几分钟"跑完"= 什么都没跑)
    OPENAI_API_BASE=http://localhost:$PORT/v1 OPENAI_API_KEY=dummy CODEBUDDY_MODEL="$MODEL" \
    GAIA2_SCENARIO_DIR="$REPO/.datasets/gaia2-cli-loaded" \
    RESULTS_BASE=$RUN_BASE BENCHMARKS="$BENCH" ITER_CHAIN=3 ITER_MUTATE=1 \
    RESUME=1 TASK_CONCURRENCY=10 ARMS="$ARMS" TASK_LIMIT=100 \
      $PY -u scripts/latest/latest_runner.py >> "$REPO/run_${MODEL}_${BENCH}.log" 2>&1
    log "    sweep exit=$?"
    checkpoint "$MODEL" "$BENCH" "$ARMS"
  fi

  health "$MODEL" "$BENCH" || { log "  ✗ 数据不健康,不 gate 不 push,留待人工"; flock -u 8; continue; }

  log "  gate..."
  REQUIRE_C=1 $PY scripts/latest/v2_gate.py "$MODEL" "$RUN_BASE" 2>&1 | tail -14
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    # 仓库级 push 锁:两个 worker 并发 commit/rebase 会撞 index.lock;add 只圈
    # 本项数据,别的 worker 未 gate 的行绝不搭车进 main。
    exec 7>"$REPO/.locks/gitpush.lock"; flock 7
    git add "experiments_results/$RUN_BASE/$MODEL/$BENCH" 2>/dev/null
    git add "experiments_results/latest_evolving/mutations.json" 2>/dev/null || true
    git -c user.email=yizhouyang@tencent.com -c user.name=yizhouyang \
      commit -q -m "experiment($MODEL): $BENCH $ARMS C_POLICY=$C_POLICY temp=${GEN_TEMPERATURE} via autopilot" 2>&1 | tail -1
    git pull --rebase -q origin main 2>&1 | tail -1
    git push origin main 2>&1 | tail -1
    flock -u 7
    log "  ✅ gate 通过,已 push"
  else
    log "  ✗ gate FAIL — 不 push"
  fi
  flock -u 8
done
log "════ 队列跑完 ════"
