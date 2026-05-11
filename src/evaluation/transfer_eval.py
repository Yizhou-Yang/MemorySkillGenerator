"""
跨模型迁移评测 — 衡量 skill 的跨模型可迁移性。

参考 MemSkill 论文 §5.1:
- 用 Model A 训练 skill → 用 Model B 执行 → 对比性能
- 这是衡量"真演化" vs "prompt 包装"的核心指标
- MemSkill 的消融显示 designer 在跨模型时贡献放大

Reference: docs/internal/memskill_analysis.md §5.1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.models import Skill
from src.utils.llm import LLMClient


@dataclass
class TransferResult:
    """单个 skill 的跨模型迁移结果"""
    skill_id: str
    skill_name: str
    source_model: str
    target_model: str
    source_em: float = 0.0
    source_f1: float = 0.0
    target_em: float = 0.0
    target_f1: float = 0.0
    transfer_gap: float = 0.0  # target_f1 - source_f1
    num_tasks: int = 0

    @property
    def transfer_ratio(self) -> float:
        """迁移比率: target_f1 / source_f1"""
        if self.source_f1 <= 0:
            return 0.0
        return self.target_f1 / self.source_f1


@dataclass
class TransferReport:
    """跨模型迁移评测报告"""
    source_model: str
    target_model: str
    results: list[TransferResult] = field(default_factory=list)
    avg_source_f1: float = 0.0
    avg_target_f1: float = 0.0
    avg_transfer_gap: float = 0.0
    avg_transfer_ratio: float = 0.0

    def compute_aggregates(self) -> None:
        """计算聚合指标"""
        if not self.results:
            return
        n = len(self.results)
        self.avg_source_f1 = sum(r.source_f1 for r in self.results) / n
        self.avg_target_f1 = sum(r.target_f1 for r in self.results) / n
        self.avg_transfer_gap = self.avg_target_f1 - self.avg_source_f1
        ratios = [r.transfer_ratio for r in self.results if r.source_f1 > 0]
        self.avg_transfer_ratio = sum(ratios) / len(ratios) if ratios else 0.0


class CrossModelTransferEvaluator:
    """
    跨模型迁移评测器。

    核心思路:
    1. 用 source_model 训练/生成 skill
    2. 用 target_model 执行相同 skill
    3. 对比两者的 EM/F1 性能
    4. 如果 target 性能接近或超过 source，说明 skill 有真正的跨模型语义价值

    MemSkill 的关键发现:
    - Qwen 上 MemSkill 比 LLaMA 上还强 (52.07 vs 50.96)
    - 去掉 designer 在 Qwen 上跌 17.36 (vs LLaMA 跌 6.85)
    - 说明演化出的 skill 有跨模型的"语义价值"
    """

    def __init__(
        self,
        source_client: LLMClient | None = None,
        target_client: LLMClient | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.source_client = source_client
        self.target_client = target_client
        self.config = config or {}

    def evaluate_transfer(
        self,
        skills: list[Skill],
        tasks: list[dict[str, str]],
        source_model_name: str = "source",
        target_model_name: str = "target",
    ) -> TransferReport:
        """
        评测 skill 的跨模型迁移性能。

        Args:
            skills: 要评测的 skill 列表
            tasks: 评测任务列表
            source_model_name: 源模型名称
            target_model_name: 目标模型名称

        Returns:
            TransferReport
        """
        report = TransferReport(
            source_model=source_model_name,
            target_model=target_model_name,
        )

        for skill in skills:
            result = self._evaluate_single_skill(
                skill, tasks, source_model_name, target_model_name
            )
            report.results.append(result)

        report.compute_aggregates()

        logger.info(
            f"[Transfer] {source_model_name} -> {target_model_name}: "
            f"avg_source_f1={report.avg_source_f1:.4f}, "
            f"avg_target_f1={report.avg_target_f1:.4f}, "
            f"transfer_ratio={report.avg_transfer_ratio:.4f}"
        )
        return report

    def _evaluate_single_skill(
        self,
        skill: Skill,
        tasks: list[dict[str, str]],
        source_model_name: str,
        target_model_name: str,
    ) -> TransferResult:
        """评测单个 skill 的跨模型迁移"""
        result = TransferResult(
            skill_id=skill.skill_id,
            skill_name=skill.name,
            source_model=source_model_name,
            target_model=target_model_name,
            num_tasks=len(tasks),
        )

        skill_prompt = self._format_skill_prompt(skill)

        # 用 source model 评测
        if self.source_client:
            source_scores = self._run_tasks(
                self.source_client, skill_prompt, tasks
            )
            result.source_em = source_scores["avg_em"]
            result.source_f1 = source_scores["avg_f1"]

        # 用 target model 评测
        if self.target_client:
            target_scores = self._run_tasks(
                self.target_client, skill_prompt, tasks
            )
            result.target_em = target_scores["avg_em"]
            result.target_f1 = target_scores["avg_f1"]

        result.transfer_gap = result.target_f1 - result.source_f1
        return result

    def _run_tasks(
        self,
        client: LLMClient,
        skill_prompt: str,
        tasks: list[dict[str, str]],
    ) -> dict[str, float]:
        """用指定 LLM 客户端运行任务"""
        total_em = 0.0
        total_f1 = 0.0
        count = 0

        for task in tasks:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            f"You are a task-execution agent. "
                            f"Use the following skill:\n\n{skill_prompt}\n\n"
                            f"Apply it to complete the task. Give your final answer directly."
                        ),
                    },
                    {"role": "user", "content": task.get("description", "")},
                ]
                response = client.chat(messages)
                expected = task.get("expected", "")

                # 简单的 EM/F1 计算
                em = 1.0 if expected.lower().strip() in response.lower() else 0.0
                f1 = self._compute_token_f1(response, expected)

                total_em += em
                total_f1 += f1
                count += 1
            except Exception as exc:
                logger.error(f"[Transfer] Task failed: {exc}")
                count += 1

        n = max(count, 1)
        return {"avg_em": total_em / n, "avg_f1": total_f1 / n}

    @staticmethod
    def _compute_token_f1(response: str, expected: str) -> float:
        """计算 token-level F1"""
        if not expected:
            return 1.0

        resp_tokens = response.lower().split()
        exp_tokens = expected.lower().split()

        if not resp_tokens or not exp_tokens:
            return 0.0

        from collections import Counter
        common = Counter(exp_tokens) & Counter(resp_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(resp_tokens)
        recall = num_common / len(exp_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _format_skill_prompt(skill: Skill) -> str:
        """格式化 skill 为 prompt"""
        parts = [f"## Skill: {skill.name}", skill.description, ""]
        if skill.procedure:
            parts.append("**Procedure:**")
            for i, step in enumerate(skill.procedure, 1):
                parts.append(f"{i}. {step}")
            parts.append("")
        if skill.constraints:
            parts.append("**Constraints:**")
            for c in skill.constraints:
                parts.append(f"- {c}")
        return "\n".join(parts)

    def generate_comparison_table(
        self, report: TransferReport
    ) -> str:
        """生成对比表格（Markdown 格式）"""
        lines = [
            f"## Cross-Model Transfer: {report.source_model} → {report.target_model}",
            "",
            "| Skill | Source F1 | Target F1 | Gap | Ratio |",
            "|-------|----------|----------|-----|-------|",
        ]
        for r in report.results:
            lines.append(
                f"| {r.skill_name} | {r.source_f1:.4f} | "
                f"{r.target_f1:.4f} | {r.transfer_gap:+.4f} | "
                f"{r.transfer_ratio:.2f} |"
            )
        lines.extend([
            "",
            f"**Average**: Source F1={report.avg_source_f1:.4f}, "
            f"Target F1={report.avg_target_f1:.4f}, "
            f"Transfer Ratio={report.avg_transfer_ratio:.2f}",
        ])
        return "\n".join(lines)
