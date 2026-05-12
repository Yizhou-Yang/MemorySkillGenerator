"""
Span-level Processor — splits long text into spans for batch processing.

Reference: MemSkill paper §3.5:
- Split long dialogues/documents into fixed-token-size spans
- One LLM call per span to generate memory updates
- Reduces LLM calls from O(turns) to O(spans)

Reference: docs/internal/memskill_analysis.md §3.5
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TextSpan:
    """A single text span"""
    span_id: int
    text: str
    start_char: int  # Start character position in original text
    end_char: int  # End character position in original text
    approx_tokens: int  # Approximate token count

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class SpanProcessingResult:
    """Processing result for a single span"""
    span_id: int
    memory_updates: list[dict[str, Any]] = field(default_factory=list)
    selected_skill_ids: list[str] = field(default_factory=list)
    raw_response: str = ""


class SpanProcessor:
    """
    Span-level text processor.

    Splits long text into fixed-token-size spans, supporting:
    1. Split by token count (approximate, estimated via word count / 0.75)
    2. Align to sentence boundaries (avoid splitting mid-sentence)
    3. Optional overlap (adjacent spans overlap to maintain context continuity)
    """

    DEFAULT_SPAN_SIZE = 512  # Default span size (in tokens)
    DEFAULT_OVERLAP = 64  # Default overlap (in tokens)
    CHARS_PER_TOKEN = 4.0  # Approximate: 1 token ≈ 4 characters (English)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.span_size: int = self.config.get("span_size", self.DEFAULT_SPAN_SIZE)
        self.overlap: int = self.config.get("overlap", self.DEFAULT_OVERLAP)
        self.align_sentences: bool = self.config.get("align_sentences", True)

    def split_into_spans(self, text: str) -> list[TextSpan]:
        """
        Split text into a list of spans.

        Args:
            text: Input text

        Returns:
            List of TextSpan objects
        """
        if not text.strip():
            return []

        # Estimate target character count
        target_chars = int(self.span_size * self.CHARS_PER_TOKEN)
        overlap_chars = int(self.overlap * self.CHARS_PER_TOKEN)

        spans: list[TextSpan] = []
        pos = 0
        span_id = 0

        while pos < len(text):
            end = min(pos + target_chars, len(text))

            # Align to sentence boundary
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

            # Next span start position (considering overlap)
            pos = end - overlap_chars
            if pos <= spans[-1].start_char if spans else 0:
                pos = end  # Prevent infinite loop

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
        Split dialogue turns into spans.

        Prioritize splitting at session boundaries, then by token count within sessions.

        Args:
            turns: List of dialogue turns
            sessions: Optional list of sessions (each containing multiple turns)

        Returns:
            List of TextSpan objects
        """
        if sessions:
            # Split by session, then by span_size within each session
            all_spans: list[TextSpan] = []
            offset = 0
            for session_text in sessions:
                session_spans = self.split_into_spans(session_text)
                # Adjust span_id and offset
                for span in session_spans:
                    span.span_id = len(all_spans)
                    span.start_char += offset
                    span.end_char += offset
                    all_spans.append(span)
                offset += len(session_text)
            return all_spans
        else:
            # Concatenate turns then split
            full_text = "\n".join(turns)
            return self.split_into_spans(full_text)

    @staticmethod
    def _find_sentence_boundary(
        text: str, start: int, target_end: int
    ) -> int:
        """
        Find nearest sentence boundary near target_end.

        Search backward for nearest period/question mark/exclamation/newline.
        """
        # Search within 200 chars before/after target_end
        search_start = max(start + 100, target_end - 200)
        search_end = min(len(text), target_end + 200)
        search_region = text[search_start:search_end]

        # Find all sentence end positions
        boundaries = []
        for match in re.finditer(r'[.!?]\s|\n\n|\n\s*\n', search_region):
            abs_pos = search_start + match.end()
            boundaries.append(abs_pos)

        if not boundaries:
            return target_end

        # Select boundary closest to target_end
        best = min(boundaries, key=lambda b: abs(b - target_end))
        return best

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text"""
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))

    def get_processing_stats(self, spans: list[TextSpan]) -> dict[str, Any]:
        """Get span processing statistics"""
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
