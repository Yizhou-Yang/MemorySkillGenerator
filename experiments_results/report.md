# SkillForge 实验报告 — hy3-preview-ioa

**实验日期**: 2026-07-02 20:42 启动，22:37 中断
**模型**: hy3-preview-ioa
**设计**: EvoArena EvoMem + Failure-Aware Attention Routing + Cross-Agent Critic Gating + Exact Match Metrics
**并发**: 24 slots (2 docker), 1 embed thread
**Benchmark**: GAIA / GAIA2 / LOCOMO / Terminal-Bench-2

---

## 总览

| Benchmark | 状态 | C_gpr (SkillForge) | vs Baseline | vs EvoMem |
|-----------|------|---------------------|-------------|-----------|
| GAIA (100) | ✅ 完成 | **54% EM** (avg=0.600) | +3% EM / +6% avg | +3% EM / +4.2% avg |
| GAIA2 (200) | ✅ 完成 | 6.5% EM (avg=0.330) | +1.5% EM / +1.5% avg | -2.5% EM / -2.5% avg |
| LOCOMO (100) | ✅ 完成 | **0% EM** | 0 | 0 |
| Terminal-Bench-2 (89) | ❌ 中断 | 无数据 | — | — |

---

## GAIA (100 题)

| 方案 | avg_score | EM |
|------|-----------|-----|
| A (baseline) | 0.540 | 51% |
| B (evomem) | 0.558 | 51% |
| **C (SkillForge)** | **0.600** | **54%** |

**按难度**:

| Level | 题数 | A | B | C |
|-------|------|---|---|---|
| 1 (easy) | 31 | 64.5% | 51.6% | 54.8% |
| 2 (medium) | 53 | 47.2% | 45.3% | **54.7%** |
| 3 (hard) | 16 | **62.5%** | **62.5%** | 31.3% |

**分析**: SkillForge 在 Level 2（中等难度）表现最优 (+7.5% vs baseline)，但 Level 3（高难度）退化严重 (-31.2%)。Level 1 不及 baseline。

**平均耗时**: 1454s/task (纯 LLM IO)

---

## GAIA2 (200 题)

| 方案 | avg_score | EM |
|------|-----------|-----|
| A (baseline) | 0.314 | 5.0% |
| **B (evomem)** | **0.357** | **9.0%** |
| C (SkillForge) | 0.330 | 6.5% |

**按配置**:

| 配置 | A | B | C |
|------|---|---|---|
| adaptability | 0.356 | 0.309 | **0.417** |
| ambiguity | 0.288 | **0.316** | 0.190 |
| execution | 0.364 | 0.373 | **0.426** |
| search | 0.332 | **0.349** | 0.331 |
| time | 0.300 | **0.337** | 0.315 |

**分析**: SkillForge 在 adaptability 和 execution 场景表现最佳，但 ambiguity 场景退化严重 (-9.8% vs baseline)。EvoMem 整体最优。

**平均耗时**: 1863s/task (纯 LLM IO)

---

## LOCOMO (100 题)

全部三组 (A/B/C) **EM = 0%**，avg_score = 0。

**分析**: hy3-preview-ioa 模型在 LOCOMO benchmark 上完全不适用，可能原因包括：模型对 LOCOMO 的指令格式不兼容、工具调用能力不足、或 benchmark 本身需要特定能力该模型不具备。

**平均耗时**: 728s/task

---

## Terminal-Bench-2 (89 题)

**未完成** — 实验在运行 GAIA/GAIA2 全面任务 + terminal-bench compile-compcert 时中断。日志最后记录正在拉取 compile-compcert Docker 镜像并启动容器，随后无输出。

日志最后时间：2026-07-02 22:37:54。

---

## 问题与建议

1. **实验异常中断**: runner 进程不在了，但 17 个 codebuddy-headless 僵尸进程仍在运行，建议清理：
   ```
   sudo pkill -f "codebuddy-headless.*deepseek-v4-pro"
   ```

2. **LOCOMO 全零**: 需要确认是 benchmark 数据问题还是模型能力问题，建议单独对 LOCOMO 做小规模验证。

3. **GAIA Level 3 退化**: SkillForge 在高难度任务上表现差于 baseline 和 EvoMem（62.5% → 31.3%），需要排查 GPR 策略在复杂任务上的副作用。

4. **GAIA2 ambiguity 退化**: SkillForge 在模糊性场景下显著弱于 baseline（0.288 → 0.190），需要优化歧义处理逻辑。

5. **Terminal-Bench-2 未跑**: 如需补跑，可用单独命令重跑 terminal-bench-2 部分。
