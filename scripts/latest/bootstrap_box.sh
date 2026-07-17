#!/usr/bin/env bash
# ============================================================================
# bootstrap_box.sh — 新机器一条命令进入无人值守状态。
#
#   ssh gpuN 'cd /apdcephfs/private_yizhouyang/MemorySkillGenerator \
#     && git pull --ff-only origin main && bash scripts/latest/bootstrap_box.sh'
#
# 为什么需要它:2026-07-15 两台 GPU 被回收,机器上的 crontab / 手工起的进程 /
# 未提交的脚本改动全部随之消失。**容器是易失的,ceph 也不等于安全**(数据在
# ceph 上但没有机器就取不到)。所以恢复步骤必须在 git 里,而不是在某次会话的
# 记忆里 —— 任何人(或任何 agent)拿到新机器,跑这一条就够了。
#
# 它做的事(全部幂等,重复跑安全):
#   1. 起 crond(容器重启后 crond 不会自动起,这是反复踩到的坑)
#   2. 按显卡数决定本机跑哪个模型 + 装 per-worker 自愈 cron
#   3. 立刻拉起 worker,不用等 cron 的第一个 10 分钟
# ============================================================================
set -u
R=/apdcephfs/private_yizhouyang/MemorySkillGenerator
cd "$R" || { echo "FATAL: repo 不在 $R —— 新机器挂载了个人 ceph 吗?"; exit 1; }

PY=/apdcephfs_hzlf/share_1227201/yizhouyang/conda_envs/llm/bin/python
[ -x "$PY" ] || echo "WARN: 用户自己的 env 不在($PY) —— 绝不要退回 samzxge 的 env"

NGPU=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")
echo "[bootstrap] host=$(hostname -s) gpus=$NGPU"

# crond:容器重启后不会自动起,不起则所有自愈都是空谈
pgrep -x crond >/dev/null || { crond 2>/dev/null && echo "[bootstrap] crond 已拉起"; }

# 分片:gpt-oss 需要 TP=2,单卡机跑不了 ⇒ 多卡机跑 gpt-oss,单卡机跑 llama-33。
# 两台机器挂的是同一个 ceph 仓库(同 inode),同一个 (model,bench) 只能有一个
# 写者,否则 trace.jsonl 会被两个进程交错追加(2026-07-15 事故)。
# 除了这里的分片,autopilot 里还有 ceph 上的 per-(model,bench) flock 兜底。
install_worker(){  # $1=worker名 $2=模型 $3=显卡 $4=端口
  local w=$1 m=$2 c=$3 p=$4
  ( crontab -l 2>/dev/null | grep -v "autopilot_$w.lock"
    # 存活判定用 flock 试锁,不用 pgrep:env 变量不在 /proc/pid/cmdline 里,
    # `pgrep -f WORKER=xxx` 永远匹配不到,会每 10 分钟 fork 一个重复 worker。
    echo "*/10 * * * * flock -n /tmp/autopilot_$w.lock true 2>/dev/null && (cd $R && WORKER=$w AUTOPILOT_MODELS=$m AUTOPILOT_CUDA=$c AUTOPILOT_PORT=$p setsid nohup bash scripts/latest/autopilot.sh >/dev/null 2>&1 &)"
  ) | crontab -
  WORKER=$w AUTOPILOT_MODELS=$m AUTOPILOT_CUDA=$c AUTOPILOT_PORT=$p \
    setsid nohup bash scripts/latest/autopilot.sh >/dev/null 2>&1 < /dev/null &
  echo "[bootstrap] worker $w: model=$m cuda=$c port=$p (cron + 已启动)"
}

if [ "$NGPU" -ge 4 ]; then
  install_worker g1a gpt-oss  0,1 8001
  install_worker g1b llama-33 2,3 8002
elif [ "$NGPU" -ge 2 ]; then
  install_worker g1a gpt-oss  0,1 8001
else
  install_worker g2  llama-33 0   8000
fi

sleep 3
echo "[bootstrap] workers 存活: $(pgrep -fc 'bash scripts/latest/autopilot.sh' 2>/dev/null || echo 0)"
echo "[bootstrap] cron:"; crontab -l 2>/dev/null | grep autopilot | sed 's/^/    /'
echo "[bootstrap] 完成。看进度: tail -f $R/autopilot.log"
