# EvoArena 论文精讲：用 Patch 历史追踪环境演化的 Agent Memory

> *EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments* (2026, 投 NeurIPS 2026 Main Track)

---

## 0. 资源链接

| 资源 | 状态 |
|---|---|
| 代码 | 未公开（NeurIPS 2026 匿名 review 中） |
| Base Agents | Terminus2, OpenHands, A-Mem, Memento-Skill |
| Models | GPT-5.5, GPT-5.4-mini, Gemini-3.1-Pro, Qwen3.6-27B, Kimi-K2.6, Deepseek-V4-Pro, GLM-5.1, Gemma4-31B |
| Benchmarks | 自建 EvoArena (3 subsets) + GAIA 1 (Mialon et al. 2024, 可能为 Level 1 子集) + LoCoMo |
| 注意 | **不是 GAIA 2**；SWE 部分是自建 SWE-Chain-Evo（不是 SWE-bench Verified） |

---

## 1. 一句话定位

**"环境会变，但现有 agent 的 memory 只保留最新状态——旧知识被覆盖了就再也找不回来。"** EvoArena 提出了评估这个问题的 benchmark，EvoMem 提出了解决方案（给 memory 加 git-like patch history）。

按贡献类型分类：

| 维度 | 判定 |
|---|---|
| **贡献类型** | **Capability + Benchmark**：做到前人做不到的事（跟踪环境演化）+ 提供新评测 |
| **核心卖点** | benchmark（EvoArena）占 60%，方法（EvoMem）占 40% |
| **理论深度** | 无形式化理论，纯工程+实证 |

---

## 2. 核心问题：State Collapse

**观察**：现有 memory agent（A-MEM、Mem0、Memento-Skill、LangGraph）都把 memory 维护成**单一最新状态**。

**问题**：当环境变了（新版本 API、用户偏好翻转、代码库重构），agent 更新 memory 时会**直接覆盖旧信息**。覆盖后：
1. 旧行为丢了
2. "为什么之前那样做"的上下文丢了
3. "旧版本什么时候还有效"的条件丢了

**论文称此为 State Collapse**：memory 坍缩为单一最新态，版本信息全部丢失。

**关键 insight**：在动态环境中，知识往往是 **version-dependent** 而非 simply outdated。比如一个权限规则更新了，但旧规则在老版本系统/其他组织/回滚场景下仍然有效。需要的是 **version-aware state tracking**。

---

## 3. EvoArena：Benchmark 设计

### 3.1 设计原则

把静态 benchmark 改造为**版本演化链（evolution chain）**——同一个高层目标不变，但环境跨版本递进变化。测两件事：
- **Forward adaptation**：能适应新版本
- **Version compatibility**：不破坏仍有效的旧行为

与现有"动态"benchmark 的区别：

| Benchmark | 类型 | 和 EvoArena 的区别 |
|---|---|---|
| SWE-bench-Live | 刷新任务保鲜 | 不是同一环境跨版本演化 |
| GAIA2 | 加异步事件 | 单点扰动，不是持续演化 |
| HorizonBench | 用户偏好变一次 | 只一次变化，不是多步版本历史 |
| **EvoArena** | **同一环境跨版本持续演化** | — |

### 3.2 三个子集

#### Terminal-Bench-Evo（可执行工作流演化）

**来源**：从 Terminal-Bench 静态任务出发

**演化维度**（5 类）：
1. I/O 和协议合约变更
2. CLI/API 和配置接口变更
3. 依赖和工具链更新
4. 工作空间和路径布局重构
5. 验证逻辑/策略变更（如更严的 edge case）

**构造方式**：
- 每个静态任务 → 一条版本序列（chain）
- 后续版本**继承**前序版本的环境变更（不是独立变体！）
- 每个版本配：instruction + 可执行环境 + 参考解 + version-specific validation tests

**规模**：89 chains, 441 versioned tasks, avg 4.96 versions/chain

**测什么**：agent 能否识别当前工作流状态，避免重用只在早期版本有效的 procedure。

---

#### SWE-Chain-Evo（软件演化）

**来源**：26 个活跃 GitHub 仓库

**任务单元**：**Milestone**（一组相关 commit 实现一个局部仓库更新）
- 比 release-note 级更可测试
- 比单 commit 更干净
- 覆盖：web 框架、云基础设施、可观测性、安全、开发工具、数据处理、科学计算、测试框架

**构造方式**：
1. 提取连续更新窗口
2. 按 commit message 语义 + 代码变更分组为 milestones
3. 检查 milestone 连贯性，过滤无关变更
4. 合成 SWE-bench-style task description
5. 构建 Docker 环境 + Fail-to-Pass tests + Pass-to-Pass regression tests

**关键设计**：每个任务要在**之前所有 milestone 已应用**的代码状态上完成（accumulated codebase history）——不是孤立快照。

**规模**：48 chains, 135 task-step slots, 120 unique milestones, avg 2.81 steps/chain, avg 1.77 commits/milestone, avg 2.40 files + 111.92 lines modified

**测什么**：agent 能否在演进的代码状态上解决新需求，同时不引入回归。

---

#### PersonaMem-Evo（用户偏好演化）

**来源**：PersonaMem-v2 + PersonaHub

**构造方式**：
1. 从 PersonaHub 采样 seed persona
2. 生成混合话题对话历史（日常/写作/工作沟通/推荐/翻译/知识问答）
3. 偏好通过**行为隐式表达**（不是直接陈述）

**核心设计——结构化偏好演化轨迹**：

不是随机翻转，而是有**因果触发**的演化：
- 新体验、约束变化、习惯形成、时间条件

**5 种变化类型（change families）**：
1. **Same-object attitude revision**（对同一事物态度翻转）
2. **Object replacement**（换了喜欢的对象）
3. **Conditional preference shift**（条件性偏好变化）
4. **Attribute shift**（属性偏好变化）
5. **Temporal-validity shift**（时间有效性变化）

每个 eligible 偏好接受 1-5 次连续更新。

**OOD 问题设计**（4 种推理类型）：
1. Single-pattern transfer（单模式迁移）
2. Multi-pattern synthesis（多模式综合）
3. Conflict resolution（冲突解析）
4. Temporal trajectory prediction（时间轨迹预测）

**过滤**：blind filtering——如果问题仅从 persona profile 就能答、或不需要对话历史就能答，则剔除。

**规模**：50 personas, 2474 questions

**测什么**：agent 能否从长程对话中推断、追踪、泛化演化的用户偏好。

---

### 3.3 评估指标

| 子集 | Task-level metric | Chain-level metric |
|---|---|---|
| Terminal-Bench-Evo | Task accuracy (单步成功率) | Chain success rate (整条链全对) |
| SWE-Chain-Evo | Task accuracy | Chain success rate |
| PersonaMem-Evo | Exact Match | — |
| GAIA | LLM-judge accuracy | — |
| LoCoMo | Exact Match | — |

**Chain success rate 是更严格的指标**：整条演化链每一步都对才算对。

---

## 4. EvoMem：方法设计

### 4.1 Overview

现有 memory = **单一可变文件**（不断被 overwrite）

EvoMem = **最新文件 + append-only patch log**（像 git）

两个组件：
1. **Patch Recording**（写入时）：监控 memory 更新，记录有意义的非 additive 变更
2. **Patch-Augmented Retrieval**（读取时）：按需检索历史 patch 作为 version-aware 证据

### 4.2 Patch Recording

base memory updater 完全不变：$M_t = U(M_{t-1}, x_t)$

EvoMem 在旁边计算 diff：$\Delta_t = \text{Diff}(M_{t-1}, M_t)$

**只对非 additive 更新**（修改/覆盖/重解释现有 memory，不是纯新增）生成 patch：

$$p_t = (\tau_t,\ C_t^-,\ C_t^+,\ r_t,\ z_t,\ e_t)$$

| 字段 | 含义 | 例子 |
|---|---|---|
| $\tau_t$ | 时间元数据 | turn 42, session 3 |
| $C_t^-$ | 更新**前**的 memory 内容 | "用户喜欢日式料理" |
| $C_t^+$ | 更新**后**的 memory 内容 | "用户最近转向意大利菜" |
| $r_t$ | 更新理由 (rationale) | "用户在第 38 轮明确说厌倦了日料" |
| $z_t$ | 变更语义摘要 | "cuisine preference: Japanese → Italian" |
| $e_t$ | 触发上下文/证据 | 第 38 轮对话原文 |

Patch 追加到 append-only 历史：$\mathcal{P}_{1:t} = \{p_1, ..., p_t\}$

**核心区别**：$M_t$ 是最新 consolidated state；$\mathcal{P}_{1:t}$ 保留了 memory 到达当前状态的**全部中间转换**。

### 4.3 Patch-Augmented Retrieval

给定 query $q$：

```
Step 1: c_mem = R_mem(q, M_T)          ← 标准检索，从最新 memory 取
Step 2: P_q = R_patch(q, P_{1:T})      ← 额外检索，从 patch 历史取 top-k
Step 3: c(q) = Concat(c_mem, P_q)      ← 拼接给 agent
```

**什么时候 patch 有用**：
- query 依赖被覆盖的信息（"用户之前喜欢什么？"）
- query 需要理解变化原因（"为什么策略改了？"）
- query 涉及版本相关行为（"在 v2.0 下该怎么做？"）

**什么时候不需要**：普通 query 只看最新 memory 就够了。

### 4.4 四个 Agent 实例化

| Agent | 领域 | $M_T$ 是什么 | Patch 记录什么 | 检索信号 |
|---|---|---|---|---|
| **Terminus2** | Terminal | 从历史轨迹蒸馏的策略知识 | 环境演化导致的策略变更 | 语义相似度 |
| **OpenHands** | SWE | 从轨迹蒸馏的代码上下文(files/symbols/constraints) | 测试失败导致的实现策略修改 | 语义 + 文件路径结构 |
| **A-Mem** | 对话 memory | 语义网络 (notes + links) | note/relation 级别的修改 | 语义相似度 |
| **Memento-Skill** | Skill library | 全局 TIP.md 文件 | TIP.md 每次修订的 diff + 触发失败 + 理由 | 语义相似度 |

**设计亮点**：EvoMem 是 **agent-agnostic** 的——它不替换任何 memory updater，只在旁边加一层 patch 监控。因此能即插即用到架构完全不同的 agent 上。

---

## 5. 实验结果

### 5.1 主表

| Suite | Benchmark | Agent | Avg Base | Avg +EvoMem | Δ |
|---|---|---|---|---|---|
| EvoArena | Terminal-Bench-Evo | Terminus2 | 48.8% | 51.4% | **+2.6** |
| EvoArena | SWE-Chain-Evo | OpenHands | 42.0% | 45.2% | **+3.2** |
| EvoArena | PersonaMem-Evo | A-Mem | 41.8% | 43.6% | **+1.8** |
| Typical | GAIA | Memento-S | 62.5% | 66.5% | **+4.0** |
| Typical | LoCoMo | A-Mem | 26.3% | 29.3% | **+3.0** |
| **All** | **Overall** | | **44.3%** | **47.2%** | **+2.9** |

### 5.2 核心发现

**发现 1：EvoArena 很难**
- 现有最强 agent 在三个演化子集上平均只有 44.2%（对比它们在静态 benchmark 上 60%+）
- 环境演化直接导致 agent 退化

**发现 2：EvoMem 全面正向**
- 所有 benchmark × 所有 model 组合中，没有负向
- 最大增益：Deepseek-V4-Pro on GAIA +10.0pp, Gemini-3.1-Pro on LoCoMo +7.5pp

**发现 3：Chain-level accuracy 提升更大**
- Terminal + SWE 上 chain-level accuracy +6.01%
- 说明 patch 历史帮助 agent 在**整个演化序列**上保持一致性（而不只是单步）

**发现 4：标准 benchmark 也有收益**
- GAIA +4.0pp, LoCoMo +3.0pp
- 说明 EvoMem 不只在"显式演化"场景有用——即使环境没有 explicitly evolve，memory 更新本身就是一种演化，patch 历史帮助 agent 做更好的时序推理

**发现 5：Mechanistic analysis（PersonaMem-Evo）**
- EvoMem 在 **temporal trajectory** 和 **multi-pattern synthesis** 问题上增益最大
- 这些问题需要追踪分散的、演化的偏好证据——正是 patch 历史的 sweet spot
- Row-level evidence capture：72.5% → 74.9%——patch 更好地保留了完整偏好状态

---

## 6. 方法论评估

### 6.1 优点

1. **问题定义清晰**：State Collapse 是一个真实、普遍、被忽视的问题
2. **Benchmark 设计扎实**：三个子集覆盖三种完全不同的演化形态，且构造过程严谨（可执行环境、version-specific tests、blind filtering）
3. **方法极简优雅**：不改任何现有系统，只加一层 patch monitoring——即插即用
4. **跨 agent 验证**：4 种完全不同架构的 agent 都有正向效果，不是 one-trick
5. **实验覆盖广**：5 个 benchmark × 5-8 个 model，没有 cherry-picking

### 6.2 弱点

1. **无理论基础**：为什么 patch 有用没有形式化分析（对比 SRDP 的 gap bound）
2. **增益幅度中等**：avg +2.9pp，不是 dramatic improvement
3. **Patch 质量依赖 LLM**：rationale、summary 都是 LLM 生成的，质量不可控
4. **检索 overhead**：多了一次 patch retrieval，增加延迟和成本（论文没分析）
5. **何时不需要 patch 没有自动判断**：当前是所有 query 都检索 patch，浪费
6. **PersonaMem-Evo 的 OOD 问题**：由 LLM 生成 + LLM judge 评估，存在自闭合风险

### 6.3 红旗检查（hzwer 清单）

| 检查项 | 状态 |
|---|---|
| 算力碾压？ | ❌ 没有——EvoMem 是 inference-time 增强，不需要额外训练 |
| 超参操纵？ | 看不出——没有需要 tuning 的超参（top-k retrieval 是唯一的） |
| 评估操纵？ | ⚠ PersonaMem-Evo 的 LLM-generated questions + LLM-judge 有 self-play 风险 |
| 凑工作量？ | ❌ benchmark 构建本身就是实质贡献 |
| 代码可用？ | ❌ 未开源，但 benchmark 声称会公开 |

---

## 7. 与我们工作的关系

### 7.1 定位对比

| 维度 | EvoArena/EvoMem | 我们 (Attention-Aware Skill Curation) |
|---|---|---|
| **Memory 层级** | L1 episodic / 通用 memory | L2 skill library |
| **核心问题** | 环境演化 → 旧 memory 被覆盖 → state collapse | Skill 库膨胀 → 检索退化 + 注意力退化 |
| **问题本质** | 时间维度：什么时候有效 | 空间维度：什么东西有用 |
| **方法范式** | 监控+记录（不改 updater） | 主动整理（Merge/Prune/Reformat） |
| **操作类型** | Append-only patch log | Compaction (减少 skill) |
| **理论基础** | 无 | SRDP gap bound 分解 + 收敛保证 |
| **改变 memory 内容？** | 不改（只在旁边加 log） | 改（合并、删除、重排） |
| **评估环境** | 动态环境（环境跨版本变） | 静态环境（环境不变，库自身退化） |

### 7.2 互补性分析

两个工作解决的是**正交问题**：

```
环境维度：
  静态环境 → 我们的问题（库膨胀/退化）
  动态环境 → EvoArena 的问题（state collapse）

Memory 操作维度：
  主动压缩（Merge/Delete/Reformat）→ 我们
  被动记录（Patch log）→ EvoMem
```

**理论上可以同时用**：
- 用我们的方法压缩和整理 skill 库（降 δ_sem + δ_att）
- 同时用 EvoMem 的 patch 机制记录每次压缩的 diff（保留版本历史）
- 这样既解决了"库太大太乱"，又解决了"压缩后旧知识找不回来"

### 7.3 对我们的启示

1. **EvoMem 的 patch 机制可以作为我们 Merge 的安全网**：每次 Merge 前记录 patch $(C^-, C^+, r, e)$，如果后续发现 Merge 有问题可以回滚
2. **PersonaMem-Evo 的偏好演化设计**可以参考——我们的 skill 也会"演化"（从泛化到特化、从独立到有依赖）
3. **Chain-level evaluation** 是个好 idea——我们也可以报告"整个 compaction 序列是否保持了性能"而不只是最终快照
4. **他们没有理论**——这是我们的差异化优势（SRDP bound + 收敛保证）

### 7.4 如果审稿人问"和 EvoMem 什么关系"

> "EvoMem solves the **temporal** problem (when was knowledge valid?); we solve the **spatial** problem (what knowledge is worth keeping?). EvoMem records memory changes as patches without modifying the memory itself; we actively compress and restructure the memory to reduce retrieval and attention errors. The two approaches are complementary: EvoMem preserves version history while we improve the quality of the latest state."

---

## 8. 核心 Takeaway

1. **State Collapse 是真问题**：所有 memory agent 都有，论文给了一个简洁的名字和清晰的解法
2. **Patch = Git for Memory**：概念极简但 surprisingly effective——不需要改任何现有系统
3. **Benchmark 是主要贡献**：EvoArena 的三个子集（Terminal/SWE/PersonaMem）定义了一个新的评估维度（persistent environment evolution），比 EvoMem 方法本身更有长期价值
4. **增益不大但 consistent**：+2.9pp avg，没有 dramatic win，但也没有 failure——属于"稳定有用但不 exciting"类型的工作
5. **无理论是主要弱点**：对比 Memento 2（SRDP 收敛保证）和我们（gap bound 分解），EvoMem 纯实证——这在 NeurIPS 可能会被 theory reviewer 质疑

---

## 9. 论文写作技巧参考

### 9.1 好的做法

- **State Collapse 命名**：给问题起一个 catchy 的名字，让审稿人记住
- **Git 类比**：用大家熟悉的 dev 概念（patch, append-only log, diff）解释 memory design
- **四个不同 agent 实例化**：展示 generality 而不只在一个 agent 上做
- **Chain-level metric**：比 task-level 更严格，更能说明"演化"维度的价值
- **Mechanistic analysis**：不只报数字，还分析"为什么有效"（evidence capture）

### 9.2 可以借鉴的

- 我们的 δ_att independence verification 类似他们的 mechanistic analysis——都在回答"为什么有效"
- 他们的 chain-level metric 我们也可以有——"整个 compaction 序列的 cumulative performance"
- 他们明确列出 agent instantiation details（每个 agent 的 $M_T$ 是什么、patch 是什么）——我们也应该明确说清楚每个 operator 的 input/output
