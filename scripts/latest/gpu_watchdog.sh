#!/usr/bin/env bash
# ============================================================================
# gpu_watchdog — 巡检并自动修复 GPU / vLLM / sweep,不依赖任何 SSH 会话。
#
#   bash scripts/latest/gpu_watchdog.sh            # 巡检一次,只报告
#   FIX=1 bash scripts/latest/gpu_watchdog.sh      # 巡检并自动修复
#
# 建议进 crontab(和 autopilot 一起):
#   */5 * * * * cd <repo> && FIX=1 bash scripts/latest/gpu_watchdog.sh >> gpu_watchdog.log 2>&1
#
# 修的是这个环境实际会坏的东西,不是通用模板 —— 每条都对应今天踩过的坑。
# ============================================================================
set -u
REPO=/apdcephfs/private_yizhouyang/MemorySkillGenerator
PY=${WATCHDOG_PY:-/apdcephfs_hzlf/share_1227201/yizhouyang/conda_envs/llm/bin/python}
FIX=${FIX:-0}
cd "$REPO" 2>/dev/null || exit 1
case "$PY" in *samzxge*) echo "FATAL: PY 指向 samzxge(别人的共享 env),拒绝"; exit 1;; esac

ts(){ date '+%m-%d %H:%M:%S'; }
say(){ echo "[$(ts)] $*"; }
ISSUES=0
flag(){ ISSUES=$((ISSUES+1)); say "  ⚠ $*"; }

say "════ watchdog (FIX=$FIX) ════"

# ── 1. 僵尸显存 ────────────────────────────────────────────────────────────
# 死掉的 vLLM 会攥着 ~70G 不放且 nvidia-smi 的进程列表里看不到,
# 下一个进程就报 "Free memory on device (26.34/95.0 GiB) is less than desired"。
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc)
ALIVE=$(pgrep -c -f "vllm.entrypoints" 2>/dev/null || echo 0)
if [ "${USED:-0}" -gt 1000 ] && [ "$ALIVE" -eq 0 ]; then
  flag "僵尸显存: 占用 ${USED}MiB 但没有 vLLM 进程"
  if [ "$FIX" = 1 ]; then
    pkill -9 -f "vllm.entrypoints" 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 2
    fuser -k /dev/nvidia* 2>/dev/null; sleep 4
    say "  → 已强制释放,现占用 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ')"
  fi
fi

# ── 2. GPU 掉卡 / ECC ──────────────────────────────────────────────────────
nvidia-smi >/dev/null 2>&1 || { flag "nvidia-smi 无响应 —— 驱动或卡异常,需人工介入"; exit 1; }
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null | grep -vc '^0$\|N/A' || true)
[ "${ECC:-0}" -gt 0 ] && flag "有卡报告 uncorrected ECC 错误 —— 结果可能已被静默污染"

# ── 3. vLLM 存活 ───────────────────────────────────────────────────────────
if ! curl -sf --max-time 5 http://localhost:8000/v1/models >/dev/null 2>&1; then
  if [ "$ALIVE" -gt 0 ]; then
    say "  vLLM 进程在,但端口没响应(可能还在加载权重)"
  else
    flag "vLLM 未运行"
    # 提根因:vLLM 的 traceback 会被 multiproc_executor.py:511 的框架行淹没
    for L in /tmp/vllm_*.log; do
      [ -f "$L" ] || continue
      R=$(grep -oE "(GLIBC_[0-9.]+|ImportError:.*|RuntimeError:.*|NotImplementedError:.*|torch.OutOfMemoryError.*|ValueError: Free memory.*)" "$L" 2>/dev/null | sort -u | head -2)
      [ -n "$R" ] && say "    $(basename $L) 根因: $R"
    done
  fi
fi

# ── 4. autopilot 存活(它才是真正跑实验的东西)──────────────────────────
if ! pgrep -f autopilot.sh >/dev/null 2>&1; then
  flag "autopilot 未运行"
  if [ "$FIX" = 1 ]; then
    ( cd "$REPO" && setsid nohup bash scripts/latest/autopilot.sh >/dev/null 2>&1 & )
    sleep 3
    pgrep -f autopilot.sh >/dev/null && say "  → 已拉起" || say "  → 拉起失败"
  fi
fi

# ── 5. 环境完整性:ceph 写坏过 .so,而且是静默的 ──────────────────────────
# 2026-07-15 实证:一次 pip install 把 pydantic_core.so 写坏,vLLM 段错误且无 traceback。
# 用 pip 自己的 RECORD 哈希裁决,这是唯一可靠的判据。
$PY - <<'PY' 2>/dev/null
import base64, hashlib, glob, os, sys
SP = os.path.dirname(os.path.dirname(os.__file__)) + "/site-packages"
bad = []
for rec in glob.glob(SP + "/pydantic_core-*.dist-info/RECORD") + glob.glob(SP + "/vllm-*.dist-info/RECORD"):
    for line in open(rec, errors="ignore"):
        p = line.strip().split(',')
        if len(p) >= 2 and p[0].endswith('.so') and p[1].startswith('sha256='):
            f = os.path.join(SP, p[0])
            if not os.path.exists(f): continue
            a = base64.urlsafe_b64encode(hashlib.sha256(open(f, 'rb').read()).digest()).rstrip(b'=').decode()
            if a != p[1][7:]: bad.append(p[0])
if bad:
    print(f"  ⚠ 关键 .so 与 pip RECORD 不符(文件被写坏): {bad[:3]}")
    print(f"    修法: {sys.executable} -m pip install --force-reinstall --no-deps --no-cache-dir pydantic-core==2.46.4")
PY

# ── 6. 数据健康:行数在涨 ≠ 数据有效 ──────────────────────────────────────
$PY - <<'PY' 2>/dev/null
import json
from pathlib import Path
base = Path("experiments_results/latest_evolving")
if base.exists():
    for m in sorted(p.name for p in base.iterdir() if p.is_dir()):
        for b in ("gaia", "gaia2", "locomo", "tau2"):
            p = base / m / b / "trace.jsonl"
            if not p.exists(): continue
            try: rows = [json.loads(l) for l in open(p) if l.strip()]
            except Exception: continue
            if not rows: continue
            ans = [r for r in rows if str(r.get("response") or "").strip()]
            nz = sum(1 for r in rows if float(r.get("score") or 0))
            leak = sum(1 for r in rows if "<|python_tag|>" in str(r.get("response") or ""))
            if ans and nz == 0:
                print(f"  ⚠ {m}/{b}: {len(ans)} 行答了题但全 0 分 —— harness 死了,不是 0 基线")
            if leak:
                print(f"  ⚠ {m}/{b}: {leak} 行有 <|python_tag|> —— tool parser 没生效")
PY

say "════ 完成,发现 $ISSUES 个进程级问题(数据问题见上) ════"
