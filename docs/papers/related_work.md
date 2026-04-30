## 15. 相关工作（Related Work）

> **检索时间**: 2026-04-30 | **来源**: arXiv | **覆盖范围**: 2024.06 – 2026.04

### 15.1 论文分类索引

#### 🔥 A. 直接竞品 — Skill Induction / Skill Distillation from Trajectories

这些论文与我们的工作**直接竞争或互补**，必须精读并在论文中对比。

| # | 论文 | 作者 | 时间 | arXiv | 核心贡献 | 与我们的关系 |
|---|------|------|------|-------|---------|------------|
| A1 | **SkillForge: Forging Domain-Specific, Self-Evolving Agent Skills in Cloud Technical Support** | Xingyan Liu et al. | 2026-04 | [2604.08618](https://arxiv.org/abs/2604.08618v2) | 企业场景 skill 生成 + 执行失败追溯 + 定向 refinement | ⚠️ **同名论文！** 必须精读，确认差异化 |
| A2 | **ClawTrace: Cost-Aware Tracing for LLM Agent Skill Distillation** | Boqin Yuan et al. | 2026-04 | [2604.23853](https://arxiv.org/abs/2604.23853v1) | 引入 per-step cost 信号到 skill distillation pipeline | 直接相关：我们的 compression ROI 分析是类似思路 |
| A3 | **SkillX: Automatically Constructing Skill Knowledge Bases for Agents** | Chenxi Wang et al. | 2026-04 | [2604.04804](https://arxiv.org/abs/2604.04804v2) | 自动化 skill KB 构建，解决 agent 孤立学习问题 | 直接相关：skill 从经验中提取 |
| A4 | **EffiSkill: Agent Skill Based Automated Code Efficiency Optimization** | Zimu Wang et al. | 2026-03 | [2603.27850](https://arxiv.org/abs/2603.27850v1) | 从代码优化经验中 distill reusable optimization knowledge | 方法论相似：trajectory → reusable knowledge |
| A5 | **CoEvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification** | Hanrong Zhang et al. | 2026-04 | [2604.01687](https://arxiv.org/abs/2604.01687v2) | Skill 生成 + 验证的 co-evolution | 直接相关：skill quality 验证机制 |
| A6 | **MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents** | Haozhen Zhang et al. | 2026-02 | [2602.02474](https://arxiv.org/abs/2602.02474v1) | 将 memory operations 重构为 learnable skills | **核心 baseline**：我们项目的直接灵感来源 |
| A7 | **Memento-Skills: Let Agents Design Agents** | Huichi Zhou et al. | 2026-03 | [2603.18743](https://arxiv.org/abs/2603.18743v1) | Memory-based RL + stateful prompts，skill 作为 structured markdown | 方法论相似：memory → skill 的路径 |
| A8 | **TCOD: Temporal Curriculum in On-Policy Distillation for Multi-turn Agents** | Jiaqi Wang et al. | 2026-04 | [2604.24005](https://arxiv.org/abs/2604.24005v3) | 解决 trajectory-level KL instability | 相关：trajectory distillation 的稳定性问题 |

#### 🧠 B. Agent Memory Systems — 我们的 Memory Compressor 的理论基础

| # | 论文 | 作者 | 时间 | arXiv | 核心贡献 | 与我们的关系 |
|---|------|------|------|-------|---------|------------|
| B1 | **A-MEM: Agentic Memory for LLM Agents** | Wujiang Xu et al. | 2025-02 | [2502.12110](https://arxiv.org/abs/2502.12110v11) | 两阶段 agentic memory：提取→反思+链接 | **我们直接使用的 compressor 之一** |
| B2 | **Lightweight LLM Agent Memory with Small Language Models** | Jiaquan Zhang et al. | 2026-04 | [2604.07798](https://arxiv.org/abs/2604.07798v3) | 用小模型替代大模型做 memory 管理，降低成本 | 相关：memory compression 的成本优化 |
| B3 | **Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory** | Boqin Yuan et al. | 2026-03 | [2603.02473](https://arxiv.org/abs/2603.02473v2) | 3×3 study：write strategy × retrieval method × utilization | **高度相关**：我们的 compressor 对比是类似设计 |
| B4 | **E-mem: Multi-agent based Episodic Context Reconstruction for LLM Agent Memory** | Kaixiang Wang et al. | 2026-01 | [2601.21714](https://arxiv.org/abs/2601.21714v1) | 批判 destructive de-contextualization，提出 episodic reconstruction | 相关：memory compression 可能丢失 context 的问题 |
| B5 | **OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory** | Jinze Li et al. | 2026-04 | [2604.26622](https://arxiv.org/abs/2604.26622v1) | 解决 raw trajectory 存储的 token 开销问题 | 相关：trajectory 压缩的动机 |
| B6 | **ShardMemo: Masked MoE Routing for Sharded Agentic LLM Memory** | Yang Zhao et al. | 2026-01 | [2601.21545](https://arxiv.org/abs/2601.21545v1) | 分层 memory 服务（Tier A/B/C） | 相关：memory tiering 策略 |
| B7 | **Mem-T: Densifying Rewards for Long-Horizon Memory Agents** | Yanwei Yue et al. | 2026-01 | [2601.23014](https://arxiv.org/abs/2601.23014v2) | 解决 memory agent 的 sparse reward 问题 | 相关：memory 操作的训练信号 |
| B8 | **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** | Preston Rasmussen et al. | 2025-01 | [2501.13956](https://arxiv.org/abs/2501.13956) | 时序知识图谱做 agent memory | 相关：structured memory 的另一种形式 |

#### 🏗️ C. Skill Library Architecture & Retrieval — Skill 的组织和使用

| # | 论文 | 作者 | 时间 | arXiv | 核心贡献 | 与我们的关系 |
|---|------|------|------|-------|---------|------------|
| C1 | **Skill Retrieval Augmentation for Agentic AI** | Weihang Su et al. | 2026-04 | [2604.24594](https://arxiv.org/abs/2604.24594v1) | Skill RAG：从大规模 skill corpus 中检索 | 下游应用：我们生成的 skill 如何被检索使用 |
| C2 | **GraSP: Graph-Structured Skill Compositions for LLM Agents** | Tianle Xia et al. | 2026-04 | [2604.17870](https://arxiv.org/abs/2604.17870v1) | Skill 组合的图结构，解决"更多 skill ≠ 更好"问题 | 相关：skill 信息密度的重要性 |
| C3 | **Graph of Skills: Dependency-Aware Structural Retrieval** | Dawei Liu et al. | 2026-04 | [2604.05333](https://arxiv.org/abs/2604.05333v2) | 大规模 skill library 的依赖感知检索 | 相关：skill library 的 scalability |
| C4 | **From Skill Text to Skill Structure: The SSL Representation** | Qiliang Liang et al. | 2026-04 | [2604.24026](https://arxiv.org/abs/2604.24026v2) | Skill 的结构化表示（超越 SKILL.md） | 相关：skill 的表示形式 |
| C5 | **SKILLFOUNDRY: Building Self-Evolving Agent Skill Libraries from Heterogeneous Scientific Resources** | Shuaike Shen et al. | 2026-04 | [2604.03964](https://arxiv.org/abs/2604.03964v1) | 从异构科学资源构建 skill library | 相关：skill 的来源多样性 |
| C6 | **Agent Skills for LLMs: Architecture, Acquisition, Security, and the Path Forward** | Renjun Xu, Yang Yan | 2026-02 | [2602.12430](https://arxiv.org/abs/2602.12430v3) | **综述论文**：skill 的架构、获取、安全 | 必读综述，定位我们的工作 |
| C7 | **Reinforcement Learning for Self-Improving Agent with Skill Library** | Jiongxiao Wang et al. | 2025-12 | [2512.17102](https://arxiv.org/abs/2512.17102v2) | RL 驱动的 skill library 自我改进 | 相关：我们的 Adaptive Routing 方向 |

#### 🔄 D. Self-Evolving Agents — 更广泛的 Agent 自我进化框架

| # | 论文 | 作者 | 时间 | arXiv | 核心贡献 | 与我们的关系 |
|---|------|------|------|-------|---------|------------|
| D1 | **Mem²Evolve: Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation** | Zihao Cheng et al. | 2026-04 | [2604.10923](https://arxiv.org/abs/2604.10923v1) | Experience distillation + capability expansion 的 co-evolution | **高度相关**：experience → skill 的另一种路径 |
| D2 | **SEA-Eval: A Benchmark for Evaluating Self-Evolving Agents** | Sihang Jiang et al. | 2026-04 | [2604.08988](https://arxiv.org/abs/2604.08988v2) | 首个 Self-Evolving Agent 的形式化定义和 benchmark | 相关：评估方法论 |
| D3 | **Autogenesis: A Self-Evolving Agent Protocol** | Wentao Zhang et al. | 2026-04 | [2604.15034](https://arxiv.org/abs/2604.15034v2) | Agent 自进化协议（lifecycle + version tracking） | 相关：skill 的版本管理 |
| D4 | **SEARL: Joint Optimization of Policy and Tool Graph Memory** | Xinshun Feng et al. | 2026-04 | [2604.07791](https://arxiv.org/abs/2604.07791v3) | RL + tool graph memory 联合优化 | 相关：从 trajectory 中学习 tool/skill |
| D5 | **ARISE: Agent Reasoning with Intrinsic Skill Evolution in HRL** | Yu Li et al. | 2026-03 | [2603.16060](https://arxiv.org/abs/2603.16060v2) | 层次 RL 中的 intrinsic skill evolution | 相关：skill 的自动进化 |
| D6 | **Building Self-Evolving Agents via Experience-Driven Lifelong Learning** | Yuxuan Cai et al. | 2025-08 | [2508.19005](https://arxiv.org/abs/2508.19005v6) | ELL 框架：4 原则的 lifelong learning | 相关：experience → reusable skill 的框架 |
| D7 | **TARSE: Test-Time Adaptation via Retrieval of Skills and Experience** | Junda Wang et al. | 2026-03 | [2603.01241](https://arxiv.org/abs/2603.01241v1) | Test-time 检索 skill + experience | 相关：skill 的使用方式 |
| D8 | **Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills** | Pengcheng Jiang et al. | 2025-12 | [2512.16301](https://arxiv.org/abs/2512.16301v3) | **综述论文**：post-training + memory + skills | 必读综述，全景定位 |

#### 📊 E. Trajectory Analysis & Agent Learning

| # | 论文 | 作者 | 时间 | arXiv | 核心贡献 | 与我们的关系 |
|---|------|------|------|-------|---------|------------|
| E1 | **What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces?** | Weizheng Gu et al. | 2026-02 | [2602.01611](https://arxiv.org/abs/2602.01611v1) | 区分 semantic tool-use vs interface memorization | **关键洞察**：skill 应该捕获 semantics 而非 interface |
| E2 | **SkillFlow: Benchmarking Lifelong Skill Discovery and Evolution** | Ziao Zhang et al. | 2026-04 | [2604.17308](https://arxiv.org/abs/2604.17308v1) | Skill 的 discover → repair → maintain lifecycle benchmark | 相关：skill lifecycle 评估 |
| E3 | **ClawGym: A Scalable Framework for Building Effective Claw Agents** | Fei Bai et al. | 2026-04 | [2604.26904](https://arxiv.org/abs/2604.26904v1) | 可验证训练数据合成 + agent training | 相关：trajectory 数据的质量 |
| E4 | **Agent-World: Scaling Real-World Environment Synthesis for Evolving Agent Intelligence** | Guanting Dong et al. | 2026-04 | [2604.18292](https://arxiv.org/abs/2604.18292v1) | MCP + skill 的统一接口 + lifelong learning | 相关：skill 的标准化 |

#### 📚 F. 经典基础工作（必须引用）

| # | 论文 | 作者 | 时间 | arXiv | 核心贡献 |
|---|------|------|------|-------|---------|
| F1 | **Voyager: An Open-Ended Embodied Agent with Large Language Models** | Guanzhi Wang et al. | 2023-05 | [2305.16291](https://arxiv.org/abs/2305.16291) | 首个 LLM skill library（Minecraft），开创性工作 |
| F2 | **ExpeL: LLM Agents Are Experiential Learners** | Andrew Zhao et al. | 2023-08 | [2308.10144](https://arxiv.org/abs/2308.10144) | 从 trajectory 中提取 experience → insight |
| F3 | **Reflexion: Language Agents with Verbal Reinforcement Learning** | Noah Shinn et al. | 2023-03 | [2303.11366](https://arxiv.org/abs/2303.11366) | Verbal self-reflection 作为 learning signal |
| F4 | **MemGPT: Towards LLMs as Operating Systems** | Charles Packer et al. | 2023-10 | [2310.08560](https://arxiv.org/abs/2310.08560) | 虚拟 memory 管理，分层 context |
| F5 | **JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models** | Zihao Wang et al. | 2023-11 | [2311.05997](https://arxiv.org/abs/2311.05997) | Memory-augmented skill planning |

---

### 15.2 避坑指南（Pitfall Guide）

基于对上述论文的分析，总结以下避坑建议：

#### ⚠️ 坑 1：与同名论文 "SkillForge" (A1) 的差异化

**问题**：Liu et al. (2604.08618) 已经发表了一篇名为 "SkillForge" 的论文，聚焦企业云技术支持场景。

**避坑策略**：
- 我们的工作聚焦 **memory compression 对 skill quality 的影响**（信息论视角）
- 他们聚焦 **execution failure → skill refinement 的闭环**（工程视角）
- 我们需要在论文中明确区分，或考虑改名
- **必须精读此论文**，确认方法论层面没有重叠

#### ⚠️ 坑 2：Skill 的评估标准不统一

**问题**：不同论文使用完全不同的评估方式：
- SkillFlow (E2) 用 lifecycle benchmark
- TARSE (D7) 用 downstream task accuracy
- CoEvoSkills (A5) 用 co-evolutionary verification
- 我们之前用 LLM-judge（已被证明不够客观）

**避坑策略**：
- 坚持 **EM/F1 作为主指标**（与 ExpeL、HotpotQA leaderboard 对齐）
- 参考 SEA-Eval (D2) 的评估框架设计
- 在论文中明确说明为什么选择这些指标

#### ⚠️ 坑 3：Trajectory → Skill 的信息损失问题

**问题**：E-mem (B4) 明确批判了 "destructive de-contextualization"——压缩过程中丢失 sequential dependencies。

**避坑策略**：
- 我们的实验数据已经证实了这个 tradeoff（memory 的 Self 低于 traj）
- 论文中需要正面讨论这个 limitation
- 可以引用 E-mem 作为理论支撑

#### ⚠️ 坑 4：Skill Library 的 Scalability 问题

**问题**：GraSP (C2) 发现 "more skills ≠ better performance"，过多 skill 反而有害。

**避坑策略**：
- 我们的 compression ratio 分析正好回应了这个问题
- 高信息密度的 skill 在 library 场景下更有价值
- 论文中可以引用 GraSP 来支撑我们的 "信息密度" 论点

#### ⚠️ 坑 5：Cost-Awareness 的缺失

**问题**：ClawTrace (A2) 指出现有 skill distillation pipeline 缺少 per-step cost 信号。

**避坑策略**：
- 我们的 §10 (Memory Compressor 开销 vs Skill 质量) 已经在做类似分析
- 可以引用 ClawTrace 来强化我们的 cost-quality tradeoff 论述
- 考虑在实验中记录每个 variant 的实际 token 消耗

#### ⚠️ 坑 6：Semantic vs Interface Learning

**问题**：Gu et al. (E1) 发现 trajectory-SFT 学到的可能是 interface patterns 而非 semantic understanding。

**避坑策略**：
- 这解释了为什么 traj→skill 的 Cross/Transfer 差——它可能过拟合了 interface
- memory→skill 天然避免了这个问题（compression 去掉了 interface 细节）
- 论文中可以引用此工作来解释我们的核心发现

#### ⚠️ 坑 7：与 RL-based 方法的对比

**问题**：ARISE (D5) 和 SEARL (D4) 使用 RL 来优化 skill evolution，而我们是纯 prompting。

**避坑策略**：
- 明确定位：我们研究的是 **skill induction 的信息源选择**（traj vs memory vs hybrid）
- RL 方法解决的是 **skill 的迭代优化**，是正交的问题
- 可以在 Future Work 中提到 RL 作为 skill refinement 的方向

---

### 15.3 最新进展总结（2026 年 4 月 Landscape）

```
                    Agent Skill 研究全景图 (2026-04)
                    
┌─────────────────────────────────────────────────────────────────┐
│                     Skill Lifecycle                              │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Discovery│───▶│ Induction│───▶│ Retrieval│───▶│ Evolution│ │
│  │          │    │          │    │          │    │          │ │
│  │SkillFlow │    │SkillForge│    │ Skill RA │    │CoEvoSkill│ │
│  │EvoSkill  │    │SkillX    │    │ GraSP    │    │ARISE     │ │
│  │          │    │ClawTrace │    │ GoS      │    │SEARL     │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                       ▲                                         │
│                       │                                         │
│              ┌────────┴────────┐                               │
│              │  Memory Systems │                               │
│              │                 │                               │
│              │ A-MEM, Mem-T    │                               │
│              │ E-mem, ShardMemo│                               │
│              │ Zep, OCR-Memory │                               │
│              └─────────────────┘                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Surveys & Frameworks                        │   │
│  │  "Agent Skills: Architecture, Acquisition, Security"    │   │
│  │  "Adaptation of Agentic AI: Post-Training, Memory, Skills"│ │
│  │  SEA-Eval, SkillFlow Benchmark                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

我们的定位：Skill Induction 阶段，聚焦 Memory Compression 对 Skill Quality 的影响
```

### 15.4 关键趋势观察

1. **Skill 已成为 2026 年 Agent 研究的核心概念**：仅 2026 年 4 月就有 10+ 篇直接相关论文
2. **从 "能不能生成 skill" 转向 "如何生成高质量 skill"**：ClawTrace 引入 cost-awareness，GraSP 发现 more ≠ better
3. **Memory → Skill 的路径被多篇论文验证**：MemSkill、Memento-Skills、Mem²Evolve 都在探索
4. **评估标准仍未统一**：这是我们可以贡献的方向（EM/F1 + compression ratio）
5. **RL 与 Prompting 两条路线并行**：ARISE/SEARL 用 RL，SkillX/EffiSkill 用 prompting

### 15.5 推荐阅读优先级

**第一优先级（必须精读，直接影响论文定位）**：
1. A1 — SkillForge (Liu et al.) — 同名论文，必须差异化
2. A6 — MemSkill — 我们的直接灵感来源
3. B3 — Diagnosing Retrieval vs. Utilization — 方法论最相似
4. C6 — Agent Skills Survey — 全景定位
5. D8 — Adaptation of Agentic AI Survey — 全景定位

**第二优先级（方法论参考）**：
6. A2 — ClawTrace — cost-aware distillation
7. A7 — Memento-Skills — memory → skill 路径
8. D1 — Mem²Evolve — experience distillation
9. E1 — What Do Agents Learn from Trajectory-SFT — semantic vs interface
10. B1 — A-MEM — 我们使用的 compressor

**第三优先级（评估和 benchmark 参考）**：
11. D2 — SEA-Eval — self-evolving agent 评估
12. E2 — SkillFlow — skill lifecycle benchmark
13. A5 — CoEvoSkills — co-evolutionary verification

**经典必引**：
14. F1 — Voyager
15. F2 — ExpeL
16. F3 — Reflexion
17. F4 — MemGPT

---

### 15.6 ?????CCF Paper Researcher ???2026-04-30?

?????? CCF Paper Researcher ?????????? arXiv ?????

#### ? G. ?????????

| # | ?? | arXiv | ???? | ?????? |
|---|------|-------|---------|------------|
| G1 | **Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents** | [2604.15877](https://arxiv.org/abs/2604.15877) | ?????memory/skill/rule ??? spectrum ???????? | ?? **?????** ???? compression ? skill quality ??? |
| G2 | **Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols** | [2604.08224](https://arxiv.org/abs/2604.08224) | ???memory/skill/protocol ????? | ???????????? |
| G3 | **Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Self-** | [2512.18950](https://arxiv.org/abs/2512.18950) | ??? procedural memory ?? | ???hierarchical memory ? skill |
| G4 | **Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks** | [2604.20987](https://arxiv.org/abs/2604.20987) | Decision agent + Skill bank ? co-evolution | ???skill bank ????? |
| G5 | **Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents** | [2604.10674](https://arxiv.org/abs/2604.10674) | Skill-conditioned self-distillation | ?????????? distill skill |
| G6 | **Agentic Skill Discovery** | [2405.15019](https://arxiv.org/abs/2405.15019) | Agentic ???? skill | ???skill discovery ???? |
| G7 | **Distilling Feedback into Memory-as-a-Tool** | [2601.05960](https://arxiv.org/abs/2601.05960) | ? feedback distill ? memory tool | ???feedback ? memory ? skill ??? |
| G8 | **A Plan Reuse Mechanism for LLM-Driven Agent** | [2512.21309](https://arxiv.org/abs/2512.21309) | Plan reuse??? skill reuse? | ???plan ? skill ??? |
| G9 | **MemoryCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong** | [2603.25973](https://arxiv.org/abs/2603.25973) | Memory benchmark for lifelong agents | ???memory ???? |

### 15.7 ?????????

**? ????????????????????**?
1. **G1 ? Experience Compression Spectrum** ? ??????????? compression ? skill quality ?????????????
2. **G2 ? Externalization in LLM Agents** ? ??????????????

**?????????**?
3. A1 ? SkillForge (Liu et al.) ? ????
4. A6 ? MemSkill ? ??????
5. B3 ? Diagnosing Retrieval vs. Utilization ? ??????
6. C6 ? Agent Skills Survey ? ????
7. D8 ? Adaptation of Agentic AI Survey ? ????

**?????????**?
8. G3 ? Hierarchical Procedural Memory ? ??? memory ??
9. G4 ? Co-Evolving Decision and Skill Bank ? skill bank co-evolution
10. G5 ? Skill-SD ? skill-conditioned self-distillation

---

### 15.8 ?? PDF ????

???? PDF ??? `docs/papers/pdfs/` ????? gitignore??

???????`{arxiv_id}_{short_name}.pdf`


---

### 15.9 ??????2026-04-30??????

???? SkillRL ???????????????"skill extraction + agent"?"self-evolving agent + skill"?"memory to skill + agent"?"trajectory distillation + agent"?"skill library + LLM"?????? **13 ?????**?

#### ? H. ????????

| # | ?? | arXiv | ???? | ?????? |
|---|------|-------|---------|------------|
| H1 | **SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning** | [2602.08234](https://arxiv.org/abs/2602.08234) | ? RL ? trajectory ????? reusable skill patterns | ?? **?????** ???? trajectory ? skill ? RL ??????? traj_to_skill baseline ?? |
| H2 | **MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation** | [2603.23234](https://arxiv.org/abs/2603.23234) | ? agent ? memory ????? contrastive trajectory distillation | ???trajectory distillation ??? |
| H3 | **MemEvolve: Meta-Evolution of Agent Memory Systems** | [2512.18746](https://arxiv.org/abs/2512.18746) | Agent memory ??? meta-evolution | ???memory ??????? |
| H4 | **MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory** | [2601.03192](https://arxiv.org/abs/2601.03192) | ? episodic memory ?? runtime RL ?? self-evolving | ?? **?????** memory + RL ? skill ??? |
| H5 | **CASCADE: Cumulative Agentic Skill Creation through Autonomous Development and Evolution** | [2512.23880](https://arxiv.org/abs/2512.23880) | ??? skill ????? | ???skill ? incremental creation |
| H6 | **FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery** | [2602.14670](https://arxiv.org/abs/2602.14670) | ????? skill + experience memory ??? agent | ???domain-specific skill + memory |
| H7 | **SAGER: Self-Evolving User Policy Skills for Recommendation Agent** | [2604.14972](https://arxiv.org/abs/2604.14972) | ?????? self-evolving user policy skills | ???skill ???????? |
| H8 | **WebXSkill: Skill Learning for Autonomous Web Agents** | [2604.13318](https://arxiv.org/abs/2604.13318) | Web agent ? skill learning | ???web ??? skill ?? |
| H9 | **MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild** | [2603.17187](https://arxiv.org/abs/2603.17187) | Meta-learning + skill library ? in-the-wild ?? | ???meta-learning ??? skill ?? |
| H10 | **SCALAR: Learning and Composing Skills through LLM Guided Symbolic Planning and Deep RL** | [2603.09036](https://arxiv.org/abs/2603.09036) | LLM ??? symbolic planning + RL ?????? skill | ???skill composition ?? |
| H11 | **Ask Only When Needed: Proactive Retrieval from Memory and Skills for Experience-Driven Lifelong Agents** | [2604.20572](https://arxiv.org/abs/2604.20572) | ??? memory ? skill ? proactive retrieval | ???skill retrieval ?? |
| H12 | **GEMS: Agent-Native Multimodal Generation with Memory and Skills** | [2603.28088](https://arxiv.org/abs/2603.28088) | ??? agent ?? memory + skill ?? | ???memory + skill ????? |
| H13 | **AEL: Agent Evolving Learning for Open-Ended Environments** | [2604.21725](https://arxiv.org/abs/2604.21725) | ?????? agent ???? | ???open-ended skill evolution |

### 15.10 ???????????

**? ???????????????????**?
1. **G1 ? Experience Compression Spectrum** (2604.15877) ? compression ? skill quality
2. **H1 ? SkillRL** (2602.08234) ? RL-based trajectory ? skill extraction
3. **A1 ? SkillForge (Liu et al.)** (2604.08618) ? ????
4. **G2 ? Externalization in LLM Agents** (2604.08224) ? ???
5. **H4 ? MemRL** (2601.03192) ? memory + RL ? self-evolving

**?????????????**?
6. A6 ? MemSkill (2602.02474) ? ??????
7. B3 ? Diagnosing Retrieval vs. Utilization (2603.02473)
8. H2 ? MemCollab (2603.23234) ? contrastive trajectory distillation
9. H5 ? CASCADE (2512.23880) ? cumulative skill creation
10. G4 ? Co-Evolving Decision and Skill Bank (2604.20987)

**?????????? + ?????**?
11. C6 ? Agent Skills Survey (2602.12430)
12. D8 ? Adaptation of Agentic AI Survey (2512.16301)
13. H3 ? MemEvolve (2512.18746)
14. G3 ? Hierarchical Procedural Memory (2512.18950)
15. H9 ? MetaClaw (2603.17187)

---

### 15.11 ????

| ?? | ?? |
|------|------|
| ?????? | **57 ?** |
| ??? PDF | **40 ?** |
| ??????? | 12 ? |
| ???? | arXiv Paper Search + CCF Paper Researcher |
| ???? | 2026-04-30 |

