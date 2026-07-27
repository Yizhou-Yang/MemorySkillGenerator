# 在跑的实验 — 中断/改码后的重跑决策表

> 用途:你随时可能改代码。改完看这份,判断**哪些已产出的数据作废要重跑、哪些不受影响可续**。
> 维护者:autopilot(session 无关 worker)+ 我。最后更新见文件末尾时间戳。
> 权威状态在 **ceph 上的 trace**,不是我的记忆。判断"跑到哪了"永远以 `git grep`/trace 行数为准。

## 当前配置(所有在跑的实验共用)

- **机器**:gpu3(2×H20),gpu1/2 已删。无 crontab,纯 autopilot + claude routine。
- **后端模型**:这两天**只跑 gpt-oss-120b**(TP=2,parser=`openai`,mxfp4,shared ceph 权重)。
- **判官 judge**:deepseek-v4-pro,走 CodeBuddy SDK。只在 EM<1.0 打平时触发(~1K 次/全程,~500 token/次,可忽略)。启动必须 `set -a; source .env; set +a` 拿 `CODEBUDDY_API_KEY`,否则 SDK 报 Authentication required、分数全 0。`.env` 的 `LLM_PROVIDER=vllm` 保 backbone 走 vLLM 不被劫持。
- **评审 critic**:⏸ **暂停,`DEFER_CRITIC=1`**。将改用**免费 HY3**(走特殊 API,用户本周内提供)。当前**完全不跑 critic**。
  - **为什么停**:critic 每次 curation 调 **2 次** deepseek(cross_agent_evaluate + critic_refine),C 臂每题每轮都触发。主 C ~2400 次 + ablation 8 臂 ~9600 次 = **~1.2 万次 / 12–24M token** → 经费撑不住。judge(~1K 次)是零头。
  - **换 critic 的依据(自然实验)**:judge 恒定弱(gpt-oss 自判)时,只去掉 critic 的自我背书,gpt-oss/gaia C−B 从 **−4.8 翻到 +3.2** → **弱的是 critic 不是 judge**。所以要外部强 critic,只是先用免费的 HY3。
  - 短暂跑过的 critic=deepseek 数据已清;更早的 critic=gpt-oss(自评)归档在 gpu3 `_ablation_selfcritic/gpt-oss-selfcritic-20260720/`(自评 vs 外部 critic 消融点)。

## 当前实际在跑(DEFER_CRITIC=1,base=latest_evolving,最大化复用)

启动:`C_POLICY=judgment DEFER_CRITIC=1 JUDGE_MODEL=deepseek-v4-pro`(base=latest_evolving,好让 is_done 认出并**跳过**已完成的 gaia/locomo)。

**复用,不重跑**(数据审查结论:A/B 与 critic 无关,已有的有效):
- **gaia A/B**:复用 latest_evolving 现有(900 行,判官对 gaia 几乎零影响:0–4/300 行)。
- **locomo A/B**:复用现有。⚠ 注意 locomo 判官影响 ~15%(44/300),现有是 gpt-oss 自判 —— 如需干净,后续对存好的答案用强判官重判(便宜,不重跑 backbone),暂按现状。

**补缺口**(gpt-oss,零/极低花费):
- **gaia2 A/B**:新跑(之前没有;软召回不用判官)。
- **tau2 A/B**:补完(之前 57/58 部分;env 反馈不用判官)。
- **critic-free ablation**:`C_refine`(纯精炼,`C_USE_CRITIC=0`)+ `ctrl_reprompt`(A 臂对照),benches locomo/gaia2。这两臂不调 critic,**是最终数据**。

**等 HY3 再补**:主 sweep 的 **C 臂** + **其余 7 个 ablation 臂**(都带 critic)。
- HY3 接入:给 critic 配 HY3 特殊 API(在 `llm_client` 加一条 critic 路由,类似 `CRITIC_VIA_SDK`);autopilot 去 `DEFER_CRITIC`、设 `CRITIC_MODEL=<hy3>`。
- 届时会跑一个干净的 guarded 基底做 C;A/B 可复用现有或届时随 C 一起重跑对齐 protocol。
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

## 2026-07-21 机制升级:metadata catalog 三层化(critic 持笔)

- **变了什么**:store metadata 按 DB-catalog 纪律分三层(system 测量 / critic 撰写 / backbone 自述)。核心:叙述性 metadata(causal_lesson 等)的**笔从 backbone 换到固定外部 critic(HY3)**,`METADATA_AUTHOR=critic` 为新默认(`backbone` 保留作对照臂);新增 `sys_stats`(inject_count + 带 provenance 的 reuse_deltas,`_grounded_key` 第三键只吃 env/gold)。
- **协议影响**:`metadata_author` 已进 protocol_hash → **所有旧 C 行与新 C 不混池**(旧 C 全是 backbone-authored,breakdown 自动归为 backbone)。**A/B 完全不受影响**(不走 curation),已产出的 A/B 数据全部有效,继续复用。
- **依赖**:METADATA_AUTHOR=critic 下 reviewer 走 llm_critic_fn,**HY3 端点未配则 fail-loud**——所以 C 臂继续暂停,等 HY3(用户明日提供 HY3_BASE_URL/HY3_API_KEY)。到位后 C 按新机制跑,不返工。
- **可检验预言**:`breakdown.py` 新增 "C−B by metadata author" 表——critic-authored 下弱 backbone(gpt-oss)的 C−B 翻转应消失;backbone-authored 下应复现。跑对照臂用 `METADATA_AUTHOR=backbone`。
- **backbone 阵容(最终)**:gpt-oss-120B(最弱)/ HY3(主, 兼统一 critic)/ DeepSeek-v4-pro / gpt-5.5(论文 GPT-5 (low))。Claude/llama 已从论文移除。

## 骨干模型怎么接(2026-07-27)

- **hy3 走 taiji 直连,不经 CodeBuddy SDK**:`HY3_BASE_URL=http://api.taiji.woa.com/openapi/v2`,
  模型名 `hy3`,Bearer 认证。key 放机器上的 `.env` 或 `/tmp/hy3_env.sh`,**永不进 git**。
  `reasoning_effort` 平台缺省已于 2026-07-16 从 `no_think` 改为 `high` —— 显式传值,
  别依赖缺省,否则平台改默认会在 sweep 中途悄悄换掉骨干行为。
- **CodeBuddy SDK 只用于 deepseek-v4-pro**(判官)。其余骨干一律走 OpenAI 兼容端点。
- **⚠ 只导 `CODEBUDDY_MODEL` 换不了骨干。** `LLM_PROVIDER=vllm` 下解析顺序是
  `OPENAI_MODEL > CODEBUDDY_MODEL`,所以导 `CODEBUDDY_MODEL=hy3` 只改了**标签**
  (结果目录 `hy3/`、trace 按 HY3 读),实际请求发的还是 `.env` 里的 `OPENAI_MODEL`。
  **trace 不记录实际模型**,跑完无从查证。已在 `latest_runner.py` 加启动闸:
  标签与 `llm_client.served_model()` 不符直接退出,并打印 `Served model:`。
  换骨干必须同时设 `OPENAI_MODEL`+`OPENAI_API_BASE`+`OPENAI_API_KEY`。

## LoCoMo native protocol(记忆层对比,独立于 A/B/C)

机器 **any4 容器**(`ssh -p 36000 root@yizhouyang-any4.devcloud.woa.com`,repo 在
`/data/workspace/MemorySkillGenerator`),与 gpu3 的 A/B/C sweep 无关,可并行。

- **是什么**:在 mem0/A-Mem 的主场用**他们的口径**比 ours vs mem0 vs amem——逐 session 摄入 →
  top-10 检索 → 仅凭检索作答 → LLM judge(J,0/1)+ token-F1。跑 `scripts/latest/locomo_native.py`。
- **为什么单独跑**:A/B/C 那套是"重试同一任务",在单轮对话 QA 上让"记住过去答案"失去意义,
  会把检索型 baseline 压到 no-memory 水平。**结论进独立表,不是 `tab:framework`**(指标和骨干都不同)。
- **模型**:answerer `gpt-5.5` / 抽取 `gpt-5.4-mini` / 判官 `gpt-5.6-sol`,全走 `api.mxzzz.xyz`。
  grok 自建号池会 504(120s 无响应),**别用 grok 做抽取/判官**。
- **口径锚点**:mem0 的 F1 应落在 **~39**(其论文报 38.7)。偏离说明 harness 坏了,先查再信任何数字。
- **数据落点**:`experiments_results/locomo_native/{staged,ours_sdk}/`,汇总 `FINAL_COMPARISON.txt`。

### 改码后什么作废(踩过的坑)

| 改动 | 作废范围 |
|---|---|
| **`ANSWER_SYS`**(四个检索臂共用) | **全部五臂**。只重跑 ours 而复用旧 baseline 行 = 把提示词红利算成我们方法的功劳。 |
| `_EXTRACT_SYS` / `OURS_RECALL_K` / `_OursQA` | 仅 ours,baseline 行可合法复用 |
| `NOMEM_SYS` | 仅 nomem |
| `memlayer/vgr.py` 检索路径 | 仅 ours(A/B/C 走词面默认,不受影响) |

- `MEM0_DIR` 必须每次跑设成独立目录,否则和并发的 mem0 进程抢固定路径的内部 qdrant,**一条都存不进去**。
- A-Mem 需要 `AMEM_PATH=/data/workspace/AgenticMemory`,否则 `_build_sys` 抛 `SystemExit` 把整个进程带崩。
- 容器 load 高时 ssh 会空返回/断连;**别用 `git pull`**(工作树被在跑的实验写脏,ff 不动),直接 `scp` 单文件。

## 论文命名映射(2026-07-21)

- **实验里的 `gpt-5.5`(mxzzz API)= 论文里的 `GPT-5 (low)`**。填 tab:main 时,读 `latest_evolving/gpt-5.5/<bench>/` 的数据,填进 **GPT-5 (low)** 那一行(已在 GAIA/GAIA2/LoCoMo 三块的 HY3 之后加好,现为 dash)。
- 依据:官方 Vanilla GAIA2 榜单 GPT-5 (low) pass@1 = 34.6;论文 app:anchors 已引用此锚点。我们的 gpt-5.5 经 loop 对齐后 gaia2 ~25%(核心 splits,比全榜难),gaia A ~35%,落点与 34.6 档一致。
- GPT-5 (low) 是**新增的探索性 backbone**,不属于预注册冻结的三 backbone 确认集({deepseek, llama, gpt-oss});别改 §pre-reg 的"three backbones"表述。
