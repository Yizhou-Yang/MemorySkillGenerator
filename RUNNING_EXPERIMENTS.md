# 在跑的实验 — 中断/改码后的重跑决策表

> 用途:你随时可能改代码。改完看这份,判断**哪些已产出的数据作废要重跑、哪些不受影响可续**。
> 维护者:autopilot(session 无关 worker)+ 我。最后更新见文件末尾时间戳。
> 权威状态在 **ceph 上的 trace**,不是我的记忆。判断"跑到哪了"永远以 `git grep`/trace 行数为准。

## 当前配置(所有在跑的实验共用)

- **机器**:gpu3(2×H20),gpu1/2 已删。无 crontab,纯 autopilot + claude routine。
- **后端模型**:这两天**只跑 gpt-oss-120b**(TP=2,parser=`openai`,mxfp4,shared ceph 权重)。
- **判官**:deepseek-v4-pro,走 CodeBuddy SDK(`JUDGE_MODEL=deepseek-v4-pro JUDGE_VIA_SDK=1`)。只在 EM<1.0 打平时触发。
- **政策**:`C_POLICY=guarded`(预注册确认臂;✓ 需两把钥匙 = grounded 分 OR 外部 critic,自评永不背书)。
- **温度**:`GEN_TEMPERATURE=0`(确认协议锁定)。
- **protocol_hash**:每个 arm 内必须恒定(G9 gate 拦)。改任一 knob → hash 变 → 旧行与新行不能混池 → **那个 base 必须清后重跑**。

## 队列(autopilot 按序跑,flock 串行,UNGATED checkpoint 防回收丢数)

| # | 阶段 | base 目录 | benchmark | arms | 状态(见时间戳) |
|---|---|---|---|---|---|
| 1 | 主 sweep | `guarded_v1/gpt-oss/` | gaia,locomo,tau2,gaia2 | A,B,C | gaia≈A满B/C中,locomo在跑,tau2/gaia2待 |
| 2 | ablation | `ablation/<arm>/` | locomo,gaia2 | 8 消融臂 | 主队列跑完后启动 |
| 3 | pass@k | `passk_gpt-oss/` | locomo,gaia2 | A(PASSK=3,ITER_CHAIN=1) | ablation 后 |
| 4 | external | `external/` (**手动**) | — | vs A-Mem/Mem0/MemoryOS | **不自动**,风险高需我手起 |

**8 个消融臂**(ablation_runner.py 的 ARMS):C_refine, C_refine_critic(+critic), +enrich, C_no_wc, C_no_partition, C_small_inject(dose L=500), C_no_budget(dose L=∞), C_no_fallback, C_weak_compact。

## 不受本轮影响、已定稿的历史数据(别动)

- `latest_evolving/deepseek-v4-pro/{gaia,gaia2,locomo}` —— **论文唯一可用主数据**(Table 2 九格已逐格核对,非占位)。judgment 时代,two-sided。**改码不重跑这个**,除非你要重做 deepseek 主表。
- `origin/data/meta-exploratory-g2`(tag `local-meta-ff4d2d8e`)—— meta 政策首个正向证据(gpt-oss gaia C−B −3.0→+4.0),探索性,别覆盖。
- `_archive/hy3*`, `harbor_tb2/*` —— 归档,只读。

## 改码 → 重跑决策(核心:改的 knob 是否进 protocol_hash)

protocol_hash 覆盖 20 个 knob(见 `latest_runner._protocol_dict`)。分三类:

### A) 改这些 → 当前 guarded_v1/gpt-oss **全部作废,清后重跑**
`c_policy`, `critic_model`, `judge_model`, `score_provenance`, `temperature`, `iter_chain`, `iter_mutate`, `iter_feedback`, `c_inject_budget_ch`, `c_critic_gate`, `c_raw_fallback`, `c_use_critic`, `c_use_enrich`, `c_no_partition`, `w_c_disabled`, `reprompt_control`, `passk`, `external_mems`, `code_rev`。
→ hash 变。G9 会拦混池。**A/B/C 都要清**(不只 C),因为 hash 是全 arm 一致性检查。
清法:`rm -rf experiments_results/guarded_v1/gpt-oss/<bench>` 后重跑。

### B) 改**只影响 C 臂逻辑**的代码(evomem_bridge 里 curation/rendering,不改上面任何 knob 默认值)
→ 若 code_rev 变了,hash 也变 → 严格说仍作废。但**若只想续 C、复用 A/B**:A/B 与 curation 无关(no_mem/raw_patch 不碰 store),可保留 A/B trace 行、只清 `curated_patch` 组重跑,再手动放宽该 bench 的 G9(或接受 A/B/C code_rev 不齐的 warn)。**这是特例,默认还是整清。**
> 依据上次结论:gpt-oss gaia 的 A/B 判官影响近零(0 和 2 行 score≠em),理论上 A/B 不必清;但为 protocol_hash 一致默认整清。

### C) 改这些 → **不影响任何已产出数据,直接续**
- `autopilot.sh` 的编排(队列顺序、ablation/passk 追加段、锁、checkpoint)——纯调度,不进 trace。改完**重启 autopilot 即可**,已完成的 base 会被 RESUME 跳过。
- `v2_gate.py`, `pooled_stats.py`, `breakdown.py` —— 只读分析,不产 trace。改完直接重跑分析。
- `paper/main.tex` —— gitignored,本地,与实验数据解耦。

## 重启 autopilot 的安全动作(改了编排/代码后)

```bash
ssh gpu3 'cd /apdcephfs/private_yizhouyang/MemorySkillGenerator
  git pull --ff-only origin main
  find scripts/latest/__pycache__ src/__pycache__ -delete 2>/dev/null
  pkill -9 -f autopilot.sh; sleep 2; rm -f /tmp/autopilot_g3oss.lock
  # vLLM 不动(权重没变,复用在线的)
  WORKER=g3oss AUTOPILOT_MODELS=gpt-oss AUTOPILOT_CUDA=0,1 AUTOPILOT_PORT=8000 \
    JUDGE_MODEL=deepseek-v4-pro JUDGE_VIA_SDK=1 RUN_ABLATION=1 \
    setsid nohup bash scripts/latest/autopilot.sh >/dev/null 2>&1 &'
```
注意:pkill autopilot **不杀在跑的 runner 子进程**(它会跑完当前 bench 并 checkpoint),也不杀 vLLM。所以重启不丢主 sweep 进度。

## 给我看进展时,我会报

每 bench 的 A/B/C 完成数 + 健康(非零分/无泄漏)、**deepseek-judged gpt-oss C−B**(这是"符合预期吗"的真答案)、ablation 各臂进度、任何 gate FAIL(附证据)。

---
_更新:2026-07-20 11:xx,HEAD b4ccbfb6。当前:老 runner(pid 20608)收尾 gaia,新 worker(162757)跑 locomo。判官=deepseek-v4-pro 已在 gaia/locomo 行确认。_
