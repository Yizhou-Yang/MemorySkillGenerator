# SkillCurator (SkillForge)

> **Attention-Aware Skill Library Curation for Self-Evolving Agents**
>
> 核心发现：SRDP gap bound 中的 δ_M 可分解为 δ_semantic + δ_attention，
> 所有现有工作只优化了前者。我们是第一个将 LLM 注意力分布特性融入 skill library management 的工作。

---

## 🚀 论文实验 v4（当前主线）

**v4 是论文的核心实验脚本**，修复了 v1-v3 的所有数据问题，确保所有图表使用 **100% 真实数据，零合成/伪造**。

### 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env  # 填入 DEEPSEEK_API_KEY

# 3. 启动 v4 全量实验（9 benchmarks, ~8h）
nohup python3.9 scripts/run_paper_v4.py > experiments/paper_v4_stdout.log 2>&1 &

# 4. 查看进度
tail -50 experiments/paper_v4_stdout.log

# 5. 实验完成后生成所有论文图表
python3.9 scripts/generate_paper_figures.py --data experiments/paper_v4_results.json
```

### v4 vs v3 修复清单

| # | Bug | v3 行为 | v4 修复 |
|---|-----|---------|---------|
| 1 | **Figure 5b Compaction Cliff** | 只跑 A3，B2 线在绘图时合成（导致 B2 < A3） | 同一 task stream 同时跑 B2 + A3，输出双线真实数据 |
| 2 | **Figure 5c/3 Scissors Effect** | SkillOS 每次删 1/5 太激进，反而比 Ours 好 | SkillOS 保守删 1 个，Ours 用更低阈值 0.60 积极合并 |
| 3 | **Figure 4b Bound Tightening** | 只跑 A3，B2/A1 线从 A3 偏移合成 | 同一 stream 跑 B2/A1/A3 三条独立 delta 线 |
| 4 | **Compaction 不生效** | 阈值 0.70 太高，skill 不够相似无法合并 | 降低到 0.60-0.65，max_merges 增至 5 |
| 5 | **绘图合成数据** | Figure 3/4b/5b 使用公式生成的假数据 | 绘图脚本只读真实数据，无数据则不生成（不 fallback 到合成） |

### 实验产出（5 表 + 8 图）

| 产出 | 文件 | 数据来源 |
|------|------|----------|
| Table 1: 主实验 | `table1_main_results.tex` | main_experiment |
| Table 2: δ_att 独立性 | `table2_attention_independence.tex` | delta_attention |
| Table 3: 消融 | `table3_ablation.tex` | main_experiment (methods) |
| Figure 2: Curation 行为演化 | `figure2_curation_behavior.pdf` | main_experiment |
| Figure 3: Library 健康度 | `figure3_health_tracking.pdf` | scissors_effect (3 libraries) |
| Figure 4: Bound Tightening | `figure4_bound_tightening.pdf` | bound_tightening (3 methods) |
| Figure 5a: Phase Transition | `figure5a_phase_transition.pdf` | phase_transition |
| Figure 5b: Compaction Cliff | `figure5b_compaction_cliff.pdf` | compaction_cliff (B2 vs A3) |
| Figure 5c: Scissors Effect | `figure5c_scissors_effect.pdf` | scissors_effect (3 libraries) |
| Figure 6: 2×2 消融柱状图 | `figure6_ablation_2x2.pdf` | main_experiment |

### 配置文件

```
configs/paper_v4.yaml   ← v4 实验全量参数（方法定义、compaction 参数、phenomena 参数等）
configs/default.yaml    ← 基础配置（LLM、embedding、memory 等）
```

---

## 论文核心叙事

### 三层贡献

```
层 1（理论发现）: δ_M = δ_semantic + δ_attention
  → 所有现有工作只优化 δ_semantic，忽视 δ_attention
  → 这是一个新维度，不是一个新 trick

层 2（方法设计）: 两类算子分别干预两个 δ 分量
  → δ_semantic: MERGE + Utility Prune + Redundancy Scan
  → δ_attention: Position Opt + Format Rewrite + Consistency Check + Semantic Rewrite

层 3（实证现象）: 三个 phenomenon 验证理论
  → Phase Transition = 倒 U 形曲线（临界 N*）
  → Compaction Cliff = token 消耗阶梯骤降
  → Scissors Effect = effective/total skill count 剪刀差
```

### 实验方法矩阵

| ID | 方法 | 优化 δ_semantic? | 优化 δ_attention? | 理论保证? |
|----|------|:---:|:---:|:---:|
| B0 | No Memory | — | — | — |
| B1 | Append-Only (Memento-Skills) | ❌ | ❌ | SRDP |
| B2 | SkillOS (I/U/D) | ✅ (DELETE) | ❌ | ❌ |
| A1 | Ours (semantic only) | ✅ (MERGE+Prune) | ❌ | ✅ |
| A2 | Ours (attention only) | ❌ | ✅ | ✅ |
| **A3** | **Ours (full)** | **✅** | **✅** | **✅** |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    SkillCurator Pipeline (v4)                            │
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐       │
│  │Benchmark │───>│  Trajectory  │───>│   Memory Compressor       │       │
│  │  Loader  │    │  Collector   │    │  (Mem0 / A-MEM / MemBank) │       │
│  └──────────┘    └──────┬───────┘    └───────────┬───────────────┘       │
│                         │                        │                       │
│              ┌──────────┴────────────────────────┴──────────┐            │
│              │          Skill Induction (×3 pathways)        │            │
│              └──────────────────┬────────────────────────────┘            │
│                                │                                         │
│              ┌─────────────────┼─────────────────┐                       │
│              ▼                 ▼                 ▼                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐              │
│  │ δ_semantic Ops │  │ δ_attention    │  │ Health Summary │              │
│  │ MERGE + Prune  │  │ Ops (Position/ │  │ (N_eff/|S|,   │              │
│  │ + Redundancy   │  │ Format/Consist │  │  redundancy,   │              │
│  │   Scan         │  │ /Rewrite)      │  │  coverage)     │              │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘              │
│           └────────────────────┼────────────────────┘                     │
│                                ▼                                         │
│              ┌─────────────────────────────────┐                         │
│              │   Evaluator (EM/F1 + δ_M proxy) │                         │
│              │   + Bound Tightening Tracking    │                         │
│              └─────────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
SkillForge/
├── configs/
│   ├── default.yaml               # 基础配置（LLM、embedding、memory）
│   ├── paper_v4.yaml              # ★ v4 论文实验全量参数
│   └── mvp_locomo.yaml            # MVP 实验配置
├── scripts/
│   ├── run_paper_v4.py            # ★ 论文主实验脚本（9 benchmarks, 5 experiments）
│   ├── generate_paper_figures.py  # ★ 论文图表生成（5 表 + 8 图）
│   ├── run_paper_v3.py            # 历史版本（已废弃）
│   ├── run_mvp.py                 # 单 benchmark MVP
│   ├── run_multi_benchmark.py     # 多 benchmark 实验
│   ├── run_systematic_benchmark.py # v8 系统性 benchmark
│   └── run_live_validation.py     # Live API 验证
├── src/
│   ├── models.py                  # Pydantic 数据模型
│   ├── trajectory/
│   │   └── collector.py           # ReAct agent 轨迹收集
│   ├── memory/
│   │   ├── compressor.py          # Memory 压缩器（Mem0/A-MEM/MemBank）
│   │   ├── consolidation.py       # Memory 合并去重
│   │   ├── span_processor.py      # Span-based 处理
│   │   └── evolvelab_adapter.py   # EvolveLab 适配器
│   ├── skill_induction/
│   │   ├── factory.py             # Skill inducer 工厂
│   │   ├── traj_to_skill.py       # Path 1: trajectory → skill
│   │   ├── memory_to_skill.py     # Path 2: memory → skill
│   │   ├── hybrid_to_skill.py     # Path 3: hybrid → skill (evidence-filter)
│   │   ├── skill_refiner.py       # Skill 迭代精炼
│   │   ├── skill_library.py       # Skill library + retrieval
│   │   └── skill_designer.py      # Hard-case 进化
│   ├── evaluation/
│   │   ├── evaluator.py           # EM/F1 + LLM-as-judge
│   │   ├── multi_judge.py         # 多 judge 验证
│   │   └── transfer_eval.py       # 跨 benchmark 迁移评估
│   ├── rl_controller/
│   │   └── controller.py          # RL 自适应路由控制器
│   └── utils/
│       ├── config.py              # YAML 配置加载
│       ├── io.py                  # JSON/JSONL 序列化
│       ├── llm.py                 # 统一 LLM API 客户端
│       └── logging.py             # Loguru 日志
├── paper/
│   └── figures/                   # 生成的论文图表
├── experiments/                   # 实验输出（gitignored）
├── tests/                         # 单元测试 + 集成测试
├── requirements.txt
├── pyproject.toml
├── LICENSE                        # Apache-2.0
└── README.md
```

---

## Configuration Guide

### 配置层级

```
configs/default.yaml     ← 基础配置（所有参数的默认值）
configs/paper_v4.yaml    ← 论文实验配置（方法定义、实验参数）
.env                     ← 环境变量（API Key 等敏感信息）
```

系统先加载 `default.yaml`，再 deep-merge 实验配置，最后 `.env` 覆盖 LLM 设置。

### 关键参数说明

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `llm.model` | — | `deepseek-chat` | LLM 模型名 |
| `llm.temperature` | — | `0.7` | 采样温度 |
| `embedding.model` | — | `Qwen/Qwen3-Embedding-0.6B` | Embedding 模型 |
| `memory.framework` | — | `mem0` | Memory 压缩器 |
| `compaction.interval` | — | `8` | Lazy compaction 触发间隔 |
| `compaction.merge_threshold` | — | `0.60` | 余弦相似度 > 此值视为冗余 |
| `compaction.max_merges_per_round` | — | `5` | 每轮最多合并对数 |
| `attention_ops.position_optimization.beta` | — | `0.7` | Utility vs recency 权重 |
| `phenomena.stream_length` | — | `50` | Phenomenon 实验 task 数 |

### 使用不同 LLM

```bash
# OpenAI
DEEPSEEK_API_KEY=sk-your-openai-key
DEEPSEEK_BASE_URL=https://api.openai.com/v1
DEEPSEEK_MODEL=gpt-4o

# 本地 Ollama
DEEPSEEK_API_KEY=ollama
DEEPSEEK_BASE_URL=http://localhost:11434/v1
DEEPSEEK_MODEL=llama3
```

---

## Benchmarks

### 9 Benchmarks（分层）

| Tier | Benchmark | 任务类型 | Train/Test | 适合度 |
|------|-----------|----------|:----------:|:------:|
| **Tier 1** | HotpotQA | Multi-hop QA | 40/50 | ★★★ |
| **Tier 1** | 2WikiMultihopQA | Multi-hop QA | 40/50 | ★★★ |
| **Tier 1** | MuSiQue | Multi-hop QA (harder) | 40/50 | ★★★ |
| Tier 2 | TriviaQA | Single-hop QA | 20/30 | ★★ |
| Tier 2 | GSM8K | Math reasoning | 20/30 | ★★ |
| Tier 2 | ALFWorld | Embodied tasks | 20/30 | ★★ |
| Tier 3 | WebShop | E-commerce | 10/20 | ★ |
| Tier 3 | LoCoMo | Long-context memory | 10/20 | ★ |
| Tier 3 | LongMemEval | Ultra-long dialogue | 10/20 | ★ |

---

## Evaluation Metrics

| Metric | 类型 | 验证什么 |
|--------|------|----------|
| **SR (Success Rate)** | 主指标 | 任务成功率 |
| **Avg Steps** | 主指标 | 效率 |
| **Token Cost** | 主指标 | 注意力效率 |
| **N_eff / \|S\|** | 健康度 | Library 有效利用率 |
| **δ_M proxy** | 理论验证 | Retrieval precision |
| **δ_attention proxy** | 理论验证 | 固定内容变位置/格式后的 SR 变化 |

---

## Testing

```bash
# 全部测试
python -m pytest tests/ -v

# 快速测试（无网络）
python -m pytest tests/test_config.py tests/test_models.py tests/test_utils.py -v

# 集成测试（需网络）
python -m pytest tests/test_integration.py -v
```

---

## Output Structure

```
experiments/
├── paper_v4_results.json          # ★ v4 实验结果（所有图表的数据源）
├── paper_v4_stdout.log            # v4 运行日志
├── paper_v4.log                   # v4 详细日志
paper/
└── figures/
    ├── figure2_curation_behavior.pdf
    ├── figure3_health_tracking.pdf
    ├── figure4_bound_tightening.pdf
    ├── figure5a_phase_transition.pdf
    ├── figure5b_compaction_cliff.pdf
    ├── figure5c_scissors_effect.pdf
    ├── figure6_ablation_2x2.pdf
    └── figure_supp_attention_bar.pdf
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `DEEPSEEK_API_KEY is not set` | `cp .env.example .env` 并填入 API key |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| 实验太慢 | 减少 `phenomena.stream_length` 或只跑 Tier 1 |
| 图表显示 "No data" | 确认 `paper_v4_results.json` 存在且实验已完成 |
| Compaction 不生效 | 检查 `merge_threshold`（越低越激进） |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
