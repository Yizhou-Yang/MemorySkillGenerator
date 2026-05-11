"""
Skill Designer — 基于 Hard-Case 分析的技能演化机制。

参考 MemSkill 论文 §3.8:
- Hard-Case Buffer: 记录答错的 query + 失败次数 + reward
- Difficulty Score: d(q) = (1 - r(q)) · c(q)
- Cluster + Filter: KMeans 聚类 hard cases，每个簇选代表
- Two-Stage Evolution: 分析失败 → 提出 skill 修改/新增
- Early Stop + Rollback: 连续 N 个 cycle 没改善就回滚

Reference: docs/internal/memskill_analysis.md §3.8
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from src.models import Skill
from src.utils.llm import LLMClient


# ============================================================
# 数据结构
# ============================================================

@dataclass
class HardCase:
    """一个 hard case 记录"""
    query: str
    retrieved_memories: list[str] = field(default_factory=list)
    model_prediction: str = ""
    ground_truth: str = ""
    reward: float = 0.0
    fail_count: int = 1
    step: int = 0  # 记录时的训练步数
    embedding: np.ndarray | None = None  # 用于聚类

    @property
    def difficulty_score(self) -> float:
        """
        Difficulty Score (MemSkill 公式 5):
        d(q) = (1 - r(q)) · c(q)

        低 reward × 反复失败 = 最该被关注的难 case
        """
        return (1.0 - self.reward) * self.fail_count


@dataclass
class EvolutionProposal:
    """Designer 提出的 skill 演化提案"""
    action: str  # "add" | "modify" | "remove"
    skill_name: str
    description: str = ""
    content: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class EvolutionCycleResult:
    """一个演化 cycle 的结果"""
    cycle_id: int
    proposals: list[EvolutionProposal]
    pre_reward: float
    post_reward: float
    accepted: bool
    analysis: str = ""


# ============================================================
# Hard-Case Buffer
# ============================================================

class HardCaseBuffer:
    """
    Hard-Case Buffer (MemSkill §3.8.1)

    滑动窗口 buffer，记录答错的 query。
    支持按 difficulty score 排序和聚类采样。
    """

    def __init__(
        self,
        max_size: int = 200,
        max_step_gap: int = 500,
    ) -> None:
        self.max_size = max_size
        self.max_step_gap = max_step_gap
        self._cases: list[HardCase] = []
        self._query_index: dict[str, int] = {}  # query -> index in _cases

    @property
    def size(self) -> int:
        return len(self._cases)

    def add(self, case: HardCase) -> None:
        """添加或更新一个 hard case"""
        query_key = case.query[:200]  # 截断作为 key

        if query_key in self._query_index:
            # 更新已有 case 的失败次数
            idx = self._query_index[query_key]
            if idx < len(self._cases):
                self._cases[idx].fail_count += 1
                self._cases[idx].reward = min(
                    self._cases[idx].reward, case.reward
                )
                self._cases[idx].step = case.step
                return

        self._cases.append(case)
        self._query_index[query_key] = len(self._cases) - 1

        # 容量管理
        if len(self._cases) > self.max_size:
            self._evict_oldest()

    def get_top_cases(
        self,
        n: int = 20,
        current_step: int = 0,
    ) -> list[HardCase]:
        """
        获取 difficulty score 最高的 N 个 case。

        先清理过期 case，再按 difficulty_score 排序。
        """
        # 清理过期 case
        if current_step > 0:
            self._cases = [
                c for c in self._cases
                if (current_step - c.step) <= self.max_step_gap
            ]
            self._rebuild_index()

        # 按 difficulty score 降序排序
        sorted_cases = sorted(
            self._cases, key=lambda c: c.difficulty_score, reverse=True
        )
        return sorted_cases[:n]

    def get_clustered_representatives(
        self,
        n_clusters: int = 5,
        representatives_per_cluster: int = 3,
        current_step: int = 0,
    ) -> list[HardCase]:
        """
        聚类采样 (MemSkill §3.8.3)

        对 hard cases 做 KMeans 聚类，每个簇选 difficulty 最高的代表。
        保证 designer 看到的 case 类型多样。
        """
        top_cases = self.get_top_cases(
            n=min(100, self.size), current_step=current_step
        )

        if len(top_cases) <= n_clusters * representatives_per_cluster:
            return top_cases

        # 简单的基于文本特征的聚类
        # 用 query 的 token 集合做 Jaccard 距离
        clusters = self._simple_cluster(top_cases, n_clusters)

        representatives: list[HardCase] = []
        for cluster in clusters:
            # 每个簇按 difficulty_score 排序，取 top
            sorted_cluster = sorted(
                cluster, key=lambda c: c.difficulty_score, reverse=True
            )
            representatives.extend(
                sorted_cluster[:representatives_per_cluster]
            )

        return representatives

    def clear(self) -> None:
        """清空 buffer"""
        self._cases.clear()
        self._query_index.clear()

    def _evict_oldest(self) -> None:
        """淘汰最老的 case"""
        if self._cases:
            self._cases.sort(key=lambda c: c.step)
            self._cases = self._cases[-(self.max_size):]
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """重建 query 索引"""
        self._query_index = {
            c.query[:200]: i for i, c in enumerate(self._cases)
        }

    @staticmethod
    def _simple_cluster(
        cases: list[HardCase], n_clusters: int
    ) -> list[list[HardCase]]:
        """
        简单的基于文本特征的聚类。

        用 query token 集合的 Jaccard 距离做贪心聚类。
        """
        if not cases:
            return []

        # 为每个 case 计算 token 集合
        token_sets = [set(c.query.lower().split()) for c in cases]

        # 贪心聚类
        clusters: list[list[HardCase]] = [[] for _ in range(n_clusters)]
        assigned = [False] * len(cases)

        # 选择 n_clusters 个种子（均匀间隔）
        step = max(1, len(cases) // n_clusters)
        seeds = [i * step for i in range(n_clusters)]
        seeds = [min(s, len(cases) - 1) for s in seeds]

        for ci, seed_idx in enumerate(seeds):
            clusters[ci].append(cases[seed_idx])
            assigned[seed_idx] = True

        # 分配剩余 case 到最近的簇
        for i, case in enumerate(cases):
            if assigned[i]:
                continue

            best_cluster = 0
            best_sim = -1.0

            for ci, cluster in enumerate(clusters):
                if not cluster:
                    continue
                # 与簇中第一个 case 的 Jaccard 相似度
                seed_tokens = set(cluster[0].query.lower().split())
                intersection = token_sets[i] & seed_tokens
                union = token_sets[i] | seed_tokens
                sim = len(intersection) / len(union) if union else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = ci

            clusters[best_cluster].append(case)

        # 过滤空簇
        return [c for c in clusters if c]


# ============================================================
# Skill Designer
# ============================================================

class SkillDesigner:
    """
    Skill Designer — 基于 Hard-Case 分析的技能演化 (MemSkill §3.8)

    核心机制:
    1. 收集 hard cases 到 buffer
    2. 聚类采样保证多样性
    3. Two-Stage Evolution: 分析失败 → 提出修改
    4. Early Stop + Rollback: 连续 N 个 cycle 没改善就回滚
    """

    DEFAULT_TRIGGER_INTERVAL = 100  # 每 100 步触发一次
    DEFAULT_MAX_EDITS_PER_CYCLE = 3  # 每轮最多 3 个 edit
    DEFAULT_PATIENCE = 3  # 连续 N 个 cycle 没改善就 early stop

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or {}

        self.trigger_interval: int = self.config.get(
            "trigger_interval", self.DEFAULT_TRIGGER_INTERVAL
        )
        self.max_edits: int = self.config.get(
            "max_edits_per_cycle", self.DEFAULT_MAX_EDITS_PER_CYCLE
        )
        self.patience: int = self.config.get(
            "patience", self.DEFAULT_PATIENCE
        )

        # Hard-Case Buffer
        self.hard_case_buffer = HardCaseBuffer(
            max_size=self.config.get("buffer_max_size", 200),
            max_step_gap=self.config.get("buffer_max_step_gap", 500),
        )

        # 演化历史
        self._cycle_history: list[EvolutionCycleResult] = []
        self._best_reward: float = -float("inf")
        self._patience_counter: int = 0
        self._current_cycle: int = 0

    @property
    def should_stop(self) -> bool:
        """是否应该 early stop"""
        return self._patience_counter >= self.patience

    def record_failure(
        self,
        query: str,
        prediction: str,
        ground_truth: str,
        reward: float,
        step: int,
        retrieved_memories: list[str] | None = None,
    ) -> None:
        """记录一个失败 case"""
        case = HardCase(
            query=query,
            model_prediction=prediction,
            ground_truth=ground_truth,
            reward=reward,
            step=step,
            retrieved_memories=retrieved_memories or [],
        )
        self.hard_case_buffer.add(case)

    def should_trigger(self, current_step: int) -> bool:
        """是否应该触发演化"""
        return (
            current_step > 0
            and current_step % self.trigger_interval == 0
            and not self.should_stop
            and self.hard_case_buffer.size > 0
        )

    def evolve(
        self,
        current_skills: list[Skill],
        current_step: int,
    ) -> list[EvolutionProposal]:
        """
        执行一轮 skill 演化 (MemSkill §3.8.4)

        Two-Stage:
        1. Analyze Failures: 分析 hard cases + 当前 skill bank
        2. Propose Changes: 提出具体的 skill 修改/新增

        Args:
            current_skills: 当前 skill bank
            current_step: 当前训练步数

        Returns:
            演化提案列表
        """
        if self.llm_client is None:
            logger.warning("[Designer] No LLM client, cannot evolve")
            return []

        # 获取聚类采样的 hard cases
        representatives = self.hard_case_buffer.get_clustered_representatives(
            n_clusters=5,
            representatives_per_cluster=3,
            current_step=current_step,
        )

        if not representatives:
            logger.info("[Designer] No hard cases to analyze")
            return []

        logger.info(
            f"[Designer] Cycle {self._current_cycle}: "
            f"analyzing {len(representatives)} representative hard cases"
        )

        # Stage 1: Analyze Failures
        analysis = self._analyze_failures(representatives, current_skills)

        # Stage 2: Propose Changes
        proposals = self._propose_changes(analysis, current_skills)

        self._current_cycle += 1
        return proposals

    def update_reward(self, tail_reward: float) -> bool:
        """
        更新 cycle 的 stabilized reward (MemSkill §3.8.6 公式 9)

        只看最后 1/4 的平均 reward。

        Args:
            tail_reward: 当前 cycle 最后 1/4 的平均 reward

        Returns:
            True 如果有改善
        """
        improved = tail_reward > self._best_reward

        if improved:
            self._best_reward = tail_reward
            self._patience_counter = 0
            logger.info(
                f"[Designer] Reward improved: {tail_reward:.4f} "
                f"(new best)"
            )
        else:
            self._patience_counter += 1
            logger.info(
                f"[Designer] No improvement: {tail_reward:.4f} "
                f"(best={self._best_reward:.4f}, "
                f"patience={self._patience_counter}/{self.patience})"
            )

        return improved

    def _analyze_failures(
        self,
        hard_cases: list[HardCase],
        current_skills: list[Skill],
    ) -> str:
        """
        Stage 1: 分析失败模式 (MemSkill §3.8.4)

        输入: 代表 hard cases + 当前 skill bank
        输出: 自然语言分析
        """
        # 格式化 hard cases
        cases_text = []
        for i, case in enumerate(hard_cases[:10]):
            cases_text.append(
                f"Case {i+1} (difficulty={case.difficulty_score:.2f}, "
                f"fails={case.fail_count}):\n"
                f"  Query: {case.query[:200]}\n"
                f"  Expected: {case.ground_truth[:100]}\n"
                f"  Got: {case.model_prediction[:100]}"
            )
        cases_str = "\n".join(cases_text)

        # 格式化当前 skill bank
        skills_text = []
        for skill in current_skills:
            skills_text.append(
                f"- {skill.name}: {skill.description[:100]}"
            )
        skills_str = "\n".join(skills_text) if skills_text else "(empty skill bank)"

        prompt = f"""You are a skill evolution analyst. Analyze why the agent fails on these hard cases
and identify what's missing in the current skill bank.

Current Skill Bank:
{skills_str}

Hard Cases (sorted by difficulty):
{cases_str}

Analyze:
1. What PATTERNS do you see in the failures? (e.g., temporal reasoning, spatial tracking, etc.)
2. Which existing skills are INSUFFICIENT for these cases?
3. What NEW capabilities are needed?
4. Are there skills that should be MODIFIED to handle these cases better?

Provide a detailed analysis in natural language."""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a skill evolution analyst. "
                    "Identify failure patterns and suggest improvements."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            return self.llm_client.chat(messages)
        except Exception as exc:
            logger.error(f"[Designer] Analysis failed: {exc}")
            return f"Analysis failed: {exc}"

    def _propose_changes(
        self,
        analysis: str,
        current_skills: list[Skill],
    ) -> list[EvolutionProposal]:
        """
        Stage 2: 提出具体的 skill 修改/新增 (MemSkill §3.8.4)

        每轮最多 max_edits 个 edit。
        """
        skills_text = []
        for skill in current_skills:
            skills_text.append(
                f"- {skill.name} (id={skill.skill_id[:8]}): "
                f"{skill.description[:100]}"
            )
        skills_str = "\n".join(skills_text) if skills_text else "(empty)"

        prompt = f"""Based on the failure analysis below, propose specific skill changes.

Analysis:
{analysis[:2000]}

Current Skills:
{skills_str}

Propose up to {self.max_edits} changes. Each change should be one of:
- "add": Create a new skill to address a missing capability
- "modify": Improve an existing skill to handle failure cases better
- "remove": Remove a skill that is harmful or redundant

Return JSON:
{{
  "proposals": [
    {{
      "action": "add|modify|remove",
      "skill_name": "name of the skill",
      "description": "brief description for skill selection",
      "content": {{
        "purpose": "what this skill does",
        "when_to_use": "when to apply this skill",
        "how_to_apply": "step-by-step instructions",
        "constraints": "what to avoid"
      }},
      "reasoning": "why this change is needed"
    }}
  ]
}}"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a skill designer. Propose concrete skill changes "
                    "based on failure analysis. Return only JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm_client.chat_json(messages)
            data = json.loads(response)
            raw_proposals = data.get("proposals", [])

            proposals: list[EvolutionProposal] = []
            for p in raw_proposals[: self.max_edits]:
                proposals.append(
                    EvolutionProposal(
                        action=p.get("action", "add"),
                        skill_name=p.get("skill_name", "Unnamed"),
                        description=p.get("description", ""),
                        content=p.get("content", {}),
                        reasoning=p.get("reasoning", ""),
                    )
                )

            logger.info(
                f"[Designer] Proposed {len(proposals)} changes: "
                + ", ".join(f"{p.action}:{p.skill_name}" for p in proposals)
            )
            return proposals

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error(f"[Designer] Proposal generation failed: {exc}")
            return []

    def apply_proposal(
        self,
        proposal: EvolutionProposal,
        skill_bank: list[Skill],
    ) -> tuple[list[Skill], Skill | None]:
        """
        应用一个演化提案到 skill bank。

        Args:
            proposal: 演化提案
            skill_bank: 当前 skill bank

        Returns:
            (更新后的 skill bank, 新增/修改的 skill 或 None)
        """
        if proposal.action == "add":
            new_skill = Skill(
                name=proposal.skill_name,
                description=proposal.description,
                procedure=[
                    proposal.content.get("purpose", ""),
                    proposal.content.get("how_to_apply", ""),
                ],
                constraints=[proposal.content.get("constraints", "")],
                preconditions=[proposal.content.get("when_to_use", "")],
                metadata={
                    "evolved": True,
                    "cycle": self._current_cycle,
                    "reasoning": proposal.reasoning,
                },
            )
            skill_bank.append(new_skill)
            logger.info(f"[Designer] Added skill: {proposal.skill_name}")
            return skill_bank, new_skill

        elif proposal.action == "modify":
            for i, skill in enumerate(skill_bank):
                if skill.name == proposal.skill_name:
                    # 更新 skill 内容
                    skill.description = proposal.description or skill.description
                    if proposal.content.get("how_to_apply"):
                        skill.procedure = [proposal.content["how_to_apply"]]
                    if proposal.content.get("constraints"):
                        skill.constraints = [proposal.content["constraints"]]
                    skill.version += 1
                    skill.metadata["last_evolved_cycle"] = self._current_cycle
                    logger.info(
                        f"[Designer] Modified skill: {proposal.skill_name} "
                        f"-> v{skill.version}"
                    )
                    return skill_bank, skill
            logger.warning(
                f"[Designer] Skill '{proposal.skill_name}' not found for modify"
            )
            return skill_bank, None

        elif proposal.action == "remove":
            before = len(skill_bank)
            skill_bank = [
                s for s in skill_bank if s.name != proposal.skill_name
            ]
            if len(skill_bank) < before:
                logger.info(f"[Designer] Removed skill: {proposal.skill_name}")
            return skill_bank, None

        return skill_bank, None

    def get_cycle_history(self) -> list[EvolutionCycleResult]:
        """获取演化历史"""
        return self._cycle_history
