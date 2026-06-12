"""
A-Mem Agent — memory-augmented agent for LoCoMo conversation tasks.

Wraps the AgenticMemorySystem (A-Mem, NeurIPS 2025) from
WujiangXu/AgenticMemory to provide semantic memory retrieval
over long conversation histories.

The core idea: instead of dumping the entire conversation into
the prompt, A-Mem indexes conversation turns as retrievable
memory notes, then retrieves only the most relevant parts for
each question. This improves accuracy by focusing attention on
evidence-bearing segments and reduces noise.

Architecture:
    Conversation → add_note(turn) per turn → BM25+semantic index
    Question → keyword extraction → retrieve top-k notes → LLM answer
"""
from __future__ import annotations
import json
import time
import re
from typing import Any

from .base import BaseAgent


class AmemAgent(BaseAgent):
    """Memory-augmented agent using A-Mem's note-based retrieval.

    Supports LoCoMo (conversation memory QA) out of the box.
    Can be extended to any task where background context is a
    long document/conversation and questions target specific facts.
    """

    BENCHMARKS = {"locomo", "personamem_v2"}

    def __init__(self, model: str = "deepseek-v4-pro",
                 backend: str = "codebuddy",
                 retrieve_k: int = 10,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = model
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.embedding_model = embedding_model
        self._memory_system: AgenticMemorySystem | None = None
        self._loaded_session_idx: int | None = None

    def supports_benchmark(self, benchmark: str) -> bool:
        return benchmark in self.BENCHMARKS

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    def _ensure_memory_system(self):
        if self._memory_system is None:
            from .amem_core import AgenticMemorySystem
            self._memory_system = AgenticMemorySystem(
                model_name=self.embedding_model,
                llm_backend=self.backend,
                llm_model=self.model,
            )

    def load_conversation(self, sessions: list[str], date_time: str = ""):
        """Index a conversation into the memory system.

        Args:
            sessions: List of pre-formatted session strings.
                Each session typically contains multiple speaker turns.
            date_time: Optional datetime for temporal anchoring.
        """
        self._ensure_memory_system()
        self._memory_system.memories.clear()

        for session_text in sessions:
            if not session_text.strip():
                continue
            # Parse speaker turns from session text
            # Format varies by dataset; common pattern: "Speaker X: text"
            turns = self._parse_turns(session_text)
            for turn in turns:
                self._memory_system.add_note(turn, time=date_time)

    def _parse_turns(self, session_text: str) -> list[str]:
        """Parse individual speaker turns from a session string.

        Handles common formats:
        - "Speaker 1: message\\nSpeaker 2: message"
        - "SPEAKER_1: message\\nSPEAKER_2: message"
        - Raw text without speaker labels
        """
        # Try speaker-labeled format first
        speaker_pattern = re.compile(
            r'(?:Speaker\s*\d+|SPEAKER_\d+|[A-Z][a-z]+)\s*:\s*(.+?)(?=\n(?:Speaker|SPEAKER|[A-Z][a-z]+\s*:)|$)',
            re.DOTALL
        )
        matches = speaker_pattern.findall(session_text)
        if matches:
            return [m.strip() for m in matches if m.strip()]

        # Fallback: split by newlines, filter empty
        lines = [l.strip() for l in session_text.split('\n') if l.strip()]
        if len(lines) <= 1:
            return [session_text.strip()] if session_text.strip() else []
        return lines

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def retrieve_context(self, question: str) -> str:
        """Retrieve relevant conversation segments for a question."""
        self._ensure_memory_system()
        if not self._memory_system.memories:
            return "[No conversation loaded]"

        memories = self._memory_system.find_related_memories_raw(
            question, k=self.retrieve_k
        )
        return memories if memories else "[No relevant memories found]"

    def _extract_temporal_keywords(self, question: str) -> list[str]:
        """Extract time-related keywords from a question for temporal reasoning.

        LoCoMo category 3 (temporal) questions require timeline awareness.
        This method identifies time references to help the retriever
        prioritize temporally-anchored conversation notes.
        """
        temporal_patterns = [
            r'(?:after|before|during|until|since|from|to)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)',
            r'(?:at|on|in)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)',
            r'(day\s*\d+|week\s*\d+|month\s*\d+|year\s*\d{4})',
            r'(morning|afternoon|evening|night|noon|midnight)',
            r'(today|yesterday|tomorrow|weekend|weekday)',
            r'(last|next|this|previous)\s+(week|month|year|day|night)',
            r'(\d{1,2}:\d{2})',
        ]
        keywords = set()
        for pattern in temporal_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    keywords.update(k.lower() for k in m if k)
                else:
                    keywords.add(m.lower())
        return sorted(keywords)

    def _extract_entity_keywords(self, question: str) -> list[str]:
        """Extract named entities and key phrases from a question.

        LoCoMo questions often reference specific people, places, objects,
        or activities mentioned in the conversation. Extracting these
        improves retrieval precision.
        """
        # Remove question words and stop words to get key content
        cleaned = re.sub(r'(?i)\b(what|who|where|when|why|how|is|are|was|were|'
                         r'do|does|did|can|could|will|would|the|a|an|in|on|at|'
                         r'to|for|of|with|by|from|about)\b', '', question)
        cleaned = re.sub(r'[?.,!]', ' ', cleaned)
        words = [w.strip().lower() for w in cleaned.split() if len(w.strip()) > 1]
        return words[:10]  # Top 10 content words

    def build_prompt(self, task_desc: str, question: str,
                     context: str, experience_section: str = "") -> str:
        """Build the LoCoMo-optimized prompt with A-Mem context.

        LoCoMo-specific optimizations (vs. generic QA):
        - Temporal reasoning: highlights time references in the question and
          instructs the LLM to pay attention to timeline consistency.
        - Multi-hop awareness: notes that some questions require combining
          information from multiple conversation turns.
        - Evidence citation: asks the LLM to identify which conversation
          segment supports the answer, improving verifiability.
        - Short-answer enforcement: matches LoCoMo eval protocol (1-5 words).
        """
        # Detect temporal / multi-hop question patterns
        temporal_kw = self._extract_temporal_keywords(question)
        entity_kw = self._extract_entity_keywords(question)
        is_temporal = bool(temporal_kw)
        is_multi_hop = any(
            kw in question.lower()
            for kw in ['and', 'both', 'also', 'after', 'before', 'then', 'finally']
        )

        system = (
            "You are a memory-augmented assistant specialized in answering "
            "questions about long conversations. You have access to relevant "
            "conversation segments retrieved via semantic search.\n\n"
            "CRITICAL RULES:\n"
            "1. Answer in 1-5 words MAXIMUM. No bullet points, no explanations.\n"
            "2. The answer MUST be directly stated in the conversation context.\n"
            "3. Do NOT make up information — if not found, say 'not found'.\n"
        )
        if is_temporal:
            system += (
                "\n4. TEMPORAL REASONING: This question involves time references "
                f"({', '.join(temporal_kw[:5])}). "
                "Pay close attention to the TIMELINE of events. "
                "Cross-check that your answer is temporally consistent with "
                "the conversation order."
            )
        if is_multi_hop:
            system += (
                "\n5. MULTI-HOP: This question may require combining information "
                "from MULTIPLE conversation segments. Do not stop at the first "
                "relevant segment — search across ALL retrieved context."
            )
        if entity_kw:
            system += (
                f"\n6. Key entities in question: {', '.join(entity_kw[:6])}. "
                "Focus retrieval on segments mentioning these entities."
            )
        system += "\n\nProvide ONLY the answer, no explanation."
        if experience_section:
            system += f"\n\n{experience_section}"

        prompt = (
            f"[System]\n{system}\n\n"
            f"Conversation Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer (1-5 words only):"
        )
        return prompt

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def run_task(self, task: dict, experience_section: str = "",
                       group: str = "A") -> dict:
        """Execute a LoCoMo QA task with A-Mem memory retrieval.

        Flow:
        1. Parse conversation sessions from task metadata
        2. Index turns into A-Mem memory (or reuse cached index)
        3. Retrieve relevant context for the question
        4. Build augmented prompt with retrieved context + SkillForge experience
        5. Call LLM for answer
        """
        from scripts.latest.llm_client import _llm_call, _check_api_error
        from scripts.latest.trace import APIUnavailableError

        task_id = task["task_id"]
        description = task.get("description", "")
        expected = task.get("expected", "")
        metadata = task.get("metadata", {})
        sample_idx = metadata.get("sample_idx", -1)
        context_str = task.get("context", "")

        # Extract question from description
        question = description
        if "Question:" in description:
            question = description.split("Question:")[-1].strip()

        # Load/index conversation if new session
        if self._loaded_session_idx != sample_idx:
            sessions = self._parse_sessions_from_context(context_str)
            self.load_conversation(sessions)
            self._loaded_session_idx = sample_idx

        # Retrieve relevant memories
        amem_context = self.retrieve_context(question)

        # Build prompt
        prompt = self.build_prompt(description, question, amem_context, experience_section)

        result = {"task_id": task_id, "expected": expected, "response": "",
                  "error": None, "time_cost": 0, "augmented": bool(experience_section),
                  "group": group}
        t0 = time.time()

        r = await _llm_call(prompt, max_turns=1, timeout=180)
        if _check_api_error(r):
            raise APIUnavailableError("API unavailable")

        result["response"] = r.get("text", "")
        result["error"] = r.get("error")
        result["time_cost"] = time.time() - t0
        return result

    def _parse_sessions_from_context(self, context_str: str) -> list[str]:
        """Parse conversation context string into session list.

        The context is typically session blocks joined by '\n\n---\n\n'.
        """
        if not context_str:
            return []
        # Split on session separators
        sessions = re.split(r'\n*---+\n*', context_str)
        return [s.strip() for s in sessions if s.strip()]
