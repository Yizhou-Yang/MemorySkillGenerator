"""
Span-level Processor — 将长文本按 span 切分并批量处理。

参考 MemSkill 论文 §3.5:
- 将长对话/文档按固定 token 数的 span 切分
- 每个 span 一次 LLM 调用，生成 memory updates
- 将 LLM 调用次数从 O(turns) 降到 O(spans)

Reference: docs/internal/memskill_analysis.md §3.5
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TextSpan:
    """一个文本 span"""
    span_id: int
    text: str
    start_char: int  # 在原文中的起始字符位置
    end_char: int  # 在原文中的结束字符位置
    approx_tokens: int  # 近似 token 数

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class SpanProcessingResult:
    """单个 span 的处理结果"""
    span_id: int
    memory_updates: list[dict[str, Any]] = field(default_factory=list)
    selected_skill_ids: list[str] = field(default_factory=list)
    raw_response: str = ""


class SpanProcessor:
    """
    Span-level 文本处理器。

    将长文本按固定 token 数切分为 span，支持：
    1. 按 token 数切分（近似，用 word count / 0.75 估算）
    2. 按句子边界对齐（避免切断句子）
    3. 可选的 overlap（相邻 span 有重叠，保持上下文连续性）
    """

    DEFAULT_SPAN_SIZE = 512  # 默认 span 大小（token 数）
    DEFAULT_OVERLAP = 64  # 默认 overlap（token 数）
    CHARS_PER_TOKEN = 4.0  # 近似: 1 token ≈ 4 字符（英文）

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.span_size: int = self.config.get("span_size", self.DEFAULT_SPAN_SIZE)
        self.overlap: int = self.config.get("overlap", self.DEFAULT_OVERLAP)
        self.align_sentences: bool = self.config.get("align_sentences", True)

    def split_into_spans(self, text: str) -> list[TextSpan]:
        """
        将文本切分为 span 列表。

        Args:
            text: 输入文本

        Returns:
            TextSpan 列表
        """
        if not text.strip():
            return []

        # 估算目标字符数
        target_chars = int(self.span_size * self.CHARS_PER_TOKEN)
        overlap_chars = int(self.overlap * self.CHARS_PER_TOKEN)

        spans: list[TextSpan] = []
        pos = 0
        span_id = 0

        while pos < len(text):
            end = min(pos + target_chars, len(text))

            # 对齐到句子边界
            if self.align_sentences and end < len(text):
                end = self._find_sentence_boundary(text, pos, end)

            span_text = text[pos:end].strip()
            if span_text:
                approx_tokens = max(1, int(len(span_text) / self.CHARS_PER_TOKEN))
                spans.append(TextSpan(
                    span_id=span_id,
                    text=span_text,
                    start_char=pos,
                    end_char=end,
                    approx_tokens=approx_tokens,
                ))
                span_id += 1

            # 下一个 span 的起始位置（考虑 overlap）
            pos = end - overlap_chars
            if pos <= spans[-1].start_char if spans else 0:
                pos = end  # 防止无限循环

        logger.info(
            f"[SpanProcessor] Split text ({len(text)} chars) into "
            f"{len(spans)} spans (target={self.span_size} tokens)"
        )
        return spans

    def split_dialogue_into_spans(
        self,
        turns: list[str],
        sessions: list[str] | None = None,
    ) -> list[TextSpan]:
        """
        将对话 turns 切分为 span。

        优先按 session 边界切分，然后在 session 内按 token 数切分。

        Args:
            turns: 对话 turn 列表
            sessions: 可选的 session 列表（每个 session 包含多个 turn）

        Returns:
            TextSpan 列表
        """
        if sessions:
            # 按 session 切分，每个 session 内再按 span_size 切分
            all_spans: list[TextSpan] = []
            offset = 0
            for session_text in sessions:
                session_spans = self.split_into_spans(session_text)
                # 调整 span_id 和 offset
                for span in session_spans:
                    span.span_id = len(all_spans)
                    span.start_char += offset
                    span.end_char += offset
                    all_spans.append(span)
                offset += len(session_text)
            return all_spans
        else:
            # 将 turns 拼接后切分
            full_text = "\n".join(turns)
            return self.split_into_spans(full_text)

    @staticmethod
    def _find_sentence_boundary(
        text: str, start: int, target_end: int
    ) -> int:
        """
        在 target_end 附近找最近的句子边界。

        向后搜索最近的句号/问号/感叹号/换行符。
        """
        # 在 target_end 前后 200 字符范围内搜索
        search_start = max(start + 100, target_end - 200)
        search_end = min(len(text), target_end + 200)
        search_region = text[search_start:search_end]

        # 找所有句子结束位置
        boundaries = []
        for match in re.finditer(r'[.!?。！？]\s|\n\n|\n\s*\n', search_region):
            abs_pos = search_start + match.end()
            boundaries.append(abs_pos)

        if not boundaries:
            return target_end

        # 选择最接近 target_end 的边界
        best = min(boundaries, key=lambda b: abs(b - target_end))
        return best

    def estimate_token_count(self, text: str) -> int:
        """估算文本的 token 数"""
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))

    def get_processing_stats(self, spans: list[TextSpan]) -> dict[str, Any]:
        """获取 span 处理统计信息"""
        if not spans:
            return {"num_spans": 0, "total_tokens": 0, "avg_tokens": 0}

        total_tokens = sum(s.approx_tokens for s in spans)
        return {
            "num_spans": len(spans),
            "total_tokens": total_tokens,
            "avg_tokens": total_tokens / len(spans),
            "min_tokens": min(s.approx_tokens for s in spans),
            "max_tokens": max(s.approx_tokens for s in spans),
        }
