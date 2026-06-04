# SkillForge V6

> **Experience-Augmented Agent Framework — Learn from Execution History to Improve Future Performance**
>
> SkillForge 从 agent 的执行轨迹中提取经验（Experience），通过语义检索、AI 精炼和成本感知注入，
> 在动态 benchmark 上实现显著提升：**Gaia2 +3.5pp, SWE-bench Patch Rate +14pp**。

---

## Architecture

```
                        ┌──────────────────────────┐
                        │     Agent Execution       │
                        │   (CodeBuddy SDK loop)    │
                        └────────────┬─────────────┘
                                     │ trajectory + score
                                     ▼
              ┌──────────────────────────────────────────┐
              │           analysis.py                     │
              │  Fuzzy match vs oracle → Experience       │
              │  Classify failure: tool/model/over/       │
              │  task_mismatch                            │
              └──────────────────┬───────────────────────┘
                                 │ Experience
                                 ▼
              ┌──────────────────────────────────────────┐
              │           refine.py                       │
              │  Version-Conditioned AI Refinement        │
              │  LLM generalizes steps, extracts causal   │
              │  lesson, analyzes version history diff    │
              │  ⚡ ADDS information, never removes       │
              └──────────────────┬───────────────────────┘
                                 │ refined experience
                                 ▼
              ┌──────────────────────────────────────────┐
              │         experience.py                     │
              │  ExperienceLibrary                        │
              │  ├─ Semantic retrieval (sentence-         │
              │  │   transformers + TF-IDF fallback)      │
              │  ├─ Per-experience effectiveness tracking │
              │  └─ Weighted retrieval ranking            │
              └──────────────────┬───────────────────────┘
                                 │ top-k relevant experiences
                                 ▼
              ┌──────────────────────────────────────────┐
              │          injection.py                     │
              │  ├─ gate.py: classify_task_type           │
              │  │   (agentic / qa / embodied)            │
              │  ├─ Route: qa→light hints,                │
              │  │   agentic→full experience injection    │
              │  ├─ tiktoken token budget management      │
              │  └─ No content truncation                 │
              └──────────────────┬───────────────────────┘
                                 │ augmented prompt
                                 ▼
                        ┌──────────────┐
                        │  Next Task   │  (feedback loop)
                        └──────────────┘
```

### 核心模块

| 模块 | 职责 | 关键依赖 |
|------|------|---------|
| `analysis.py` | 执行轨迹 vs oracle fuzzy matching → Experience + 4 类失败分类 | rapidfuzz |
| `refine.py` | Version-Conditioned AI Refinement：LLM 泛化 + 因果 lesson + patch_history 演进洞察 | json_repair |
| `experience.py` | Experience 数据结构 + 语义嵌入检索 + per-experience effectiveness 加权 | sentence-transformers, sklearn |
| `gate.py` | 任务类型分类（agentic/qa/embodied）— 控制注入格式 | — |
| `injection.py` | 按 task_type 路由，格式化经验为 prompt，tiktoken token budget，无内容截断 | tiktoken |

### 关键设计决策

1. **Version-Conditioned Refine**：LLM 看到完整的 patch_history diff chain（跨版本演进），而非单次 reflexion
2. **语义检索**：sentence-transformers embedding（synonym 相似度 0.47 vs Jaccard 0.0）+ TF-IDF fallback
3. **Per-experience effectiveness**：每次注入后追踪 score_delta，检索时加权（负效果经验降权至 0.3x）
4. **无信息丢失**：refine.py 传全部数据给 LLM；injection.py 按字段优先级 drop 而非截断
5. **零手写算法**：全部使用 sentence-transformers / sklearn / rapidfuzz / tiktoken / json_repair

---

## Results

### 动态 Benchmark（核心）

| Benchmark | Metric | A (Baseline) | C (AI-Refined) | Δ |
|-----------|--------|:------------:|:--------------:|:--:|
| **Gaia2** | Soft Recall | 41.6% | **45.1%** | **+3.5pp** |
| **SWE-bench** | Patch Rate | 40.0% | **54.0%** | **+14pp** |

*Gaia2 n=25 test, SWE-bench n=50 test. A=no injection, C=version-conditioned AI-refined injection.*

### QA Benchmark（验证无负优化）

| Benchmark | A | C | Δ |
|-----------|:---:|:---:|:--:|
| LoCoMo | 7.4% | 6.8% | -0.6pp (中性) |
| GAIA HF | 18.0% | 4.0% | -14pp ⚠ |

*Task-type-aware routing 成功隔离了 QA 任务（LoCoMo 无变化），GAIA HF 仍有干扰待修复。*

### Ablation

| Benchmark | A (Baseline) | B (Raw) | C (Refined) | 结论 |
|-----------|:---:|:---:|:---:|------|
| Gaia2 | 41.6% | 38.6% | **45.1%** | Raw 引入噪音，AI-refine 是必要的 |
| SWE-bench | 40.0% | 50.0% | **54.0%** | Raw 已有增益，Refine 进一步提升 |

---

## Project Structure

```
SkillForge/
├── src/
│   └── v6/
│       ├── __init__.py         # SkillForgeV6 orchestrator
│       ├── experience.py       # Experience dataclass + ExperienceLibrary
│       ├── analysis.py         # Execution analysis + failure classification
│       ├── refine.py           # Version-Conditioned AI Refinement
│       ├── injection.py        # Cost-aware prompt injection
│       └── gate.py             # Task type classification
├── benchmarks/
│   ├── __init__.py
│   └── loader.py               # Benchmark dataset loader (Gaia2/SWE-bench/ALFWorld/etc.)
├── configs/
├── scripts/
├── tests/
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
