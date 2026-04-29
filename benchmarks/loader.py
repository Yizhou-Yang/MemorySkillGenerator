"""
Benchmark dataset loader.

Loads real datasets from HuggingFace Hub:
- HotpotQA: multi-hop reasoning QA (classic, used in MemSkill paper)
- HotpotQA-hard: hard subset of HotpotQA for transfer evaluation
- TriviaQA: single-hop factoid QA (classic baseline, 65K+ downloads)
- GSM8K: grade-school math reasoning with chain-of-thought (831K+ downloads, MIT)
- MuSiQue: multi-hop QA with explicit question decomposition (11K+ downloads)
- SWE-bench Lite: code bug-fixing (300 tasks, patch-diff expected answers)
"""

from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset
from loguru import logger


class BenchmarkLoader:
    """Benchmark dataset loader backed by HuggingFace datasets."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.benchmark_name: str = self.config.get("name", "hotpotqa")
        self.num_samples: int = self.config.get("num_samples", 20)

    def load(self) -> list[dict[str, Any]]:
        """
        Load benchmark tasks.

        Returns:
            A list of task dicts, each containing ``task_id``,
            ``description``, ``expected``, ``context``, etc.
        """
        loader_map = {
            "hotpotqa": self._load_hotpotqa,
            "swebench": self._load_swebench,
            "hotpotqa_hard": self._load_hotpotqa_hard,
            "triviaqa": self._load_triviaqa,
            "gsm8k": self._load_gsm8k,
            "musique": self._load_musique,
        }

        loader_fn = loader_map.get(self.benchmark_name)
        if loader_fn is None:
            raise ValueError(
                f"Unsupported benchmark: {self.benchmark_name}. "
                f"Available: {list(loader_map.keys())}"
            )

        tasks = loader_fn()
        logger.info(
            f"Loaded benchmark '{self.benchmark_name}': {len(tasks)} tasks"
        )
        return tasks

    # ------------------------------------------------------------------
    # HotpotQA (multi-hop reasoning, classic benchmark used in MemSkill)
    # ------------------------------------------------------------------

    def _load_hotpotqa(self) -> list[dict[str, Any]]:
        """
        Load the HotpotQA dataset (distractor setting, validation split).

        Source: https://huggingface.co/datasets/hotpotqa/hotpot_qa
        """
        logger.info("Loading HotpotQA (distractor, validation) from HuggingFace...")
        raw_dataset = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split="validation"
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break
            tasks.append(self._format_hotpotqa_row(row, idx))

        return tasks

    def _load_hotpotqa_hard(self) -> list[dict[str, Any]]:
        """
        Load only the 'hard' subset of HotpotQA for transfer evaluation.
        """
        logger.info("Loading HotpotQA hard subset from HuggingFace...")
        raw_dataset = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split="validation"
        )

        tasks: list[dict[str, Any]] = []
        for row in raw_dataset:
            if row.get("level") != "hard":
                continue
            tasks.append(self._format_hotpotqa_row(row, len(tasks)))
            if len(tasks) >= self.num_samples:
                break

        return tasks

    def _format_hotpotqa_row(
        self, row: dict[str, Any], idx: int
    ) -> dict[str, Any]:
        """Convert a single HotpotQA row into the unified task format."""
        # Build context from the provided paragraphs
        context_parts: list[str] = []
        titles = row.get("context", {}).get("title", [])
        sentences_list = row.get("context", {}).get("sentences", [])
        for title, sentences in zip(titles, sentences_list):
            paragraph = "".join(sentences)
            context_parts.append(f"[{title}]\n{paragraph}")
        context_text = "\n\n".join(context_parts)

        question = row.get("question", "")
        description = (
            f"Answer the following question using the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}"
        )

        return {
            "task_id": f"hotpotqa_{row.get('id', idx)}",
            "description": description,
            "expected": row.get("answer", ""),
            "context": context_text,
            "metadata": {
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "supporting_facts": row.get("supporting_facts", {}),
            },
        }

    # ------------------------------------------------------------------
    # TriviaQA (single-hop factoid QA, classic baseline)
    # License: unknown (academic use widely accepted)
    # Source: https://huggingface.co/datasets/mandarjoshi/trivia_qa
    # ------------------------------------------------------------------

    def _load_triviaqa(self) -> list[dict[str, Any]]:
        """
        Load the TriviaQA dataset (rc config, validation split).

        TriviaQA is a classic single-hop factoid QA benchmark.  It serves
        as a useful baseline: skills induced from single-hop tasks should
        be simpler and less transferable than multi-hop skills, providing
        a clear contrast with HotpotQA / MuSiQue.
        """
        logger.info("Loading TriviaQA (rc, validation) from HuggingFace...")
        raw_dataset = load_dataset(
            "mandarjoshi/trivia_qa", "rc", split="validation"
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            question = row.get("question", "")
            answer_obj = row.get("answer", {})
            # TriviaQA answer is a dict with 'value' and 'aliases'
            expected = answer_obj.get("value", "") if isinstance(answer_obj, dict) else str(answer_obj)
            aliases = answer_obj.get("aliases", []) if isinstance(answer_obj, dict) else []

            description = (
                f"Answer the following trivia question.\n\n"
                f"Question: {question}"
            )

            tasks.append({
                "task_id": f"triviaqa_{row.get('question_id', idx)}",
                "description": description,
                "expected": expected,
                "context": "",
                "metadata": {
                    "aliases": aliases,
                    "question_source": row.get("question_source", ""),
                },
            })

        return tasks

    # ------------------------------------------------------------------
    # GSM8K (grade-school math, chain-of-thought reasoning)
    # License: MIT
    # Source: https://huggingface.co/datasets/openai/gsm8k
    # ------------------------------------------------------------------

    def _load_gsm8k(self) -> list[dict[str, Any]]:
        """
        Load the GSM8K dataset (main config, test split).

        GSM8K is a math reasoning benchmark that naturally produces
        chain-of-thought trajectories.  The final numeric answer is
        easy to evaluate precisely, and the multi-step reasoning
        process generates rich trajectories for skill induction.
        """
        logger.info("Loading GSM8K (main, test) from HuggingFace...")
        raw_dataset = load_dataset("openai/gsm8k", "main", split="test")

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            question = row.get("question", "")
            raw_answer = row.get("answer", "")

            # Extract the final numeric answer after '####'
            final_answer = self._extract_gsm8k_answer(raw_answer)

            description = (
                f"Solve the following math problem step by step.\n\n"
                f"Problem: {question}"
            )

            tasks.append({
                "task_id": f"gsm8k_{idx}",
                "description": description,
                "expected": final_answer,
                "context": "",
                "metadata": {
                    "full_solution": raw_answer,
                },
            })

        return tasks

    @staticmethod
    def _extract_gsm8k_answer(raw_answer: str) -> str:
        """
        Extract the final numeric answer from a GSM8K solution.

        GSM8K answers follow the pattern:
            <chain-of-thought>\\n#### <number>
        """
        match = re.search(r"####\s*(.+)", raw_answer)
        if match:
            return match.group(1).strip()
        # Fallback: return the last line
        lines = raw_answer.strip().split("\n")
        return lines[-1].strip() if lines else raw_answer

    # ------------------------------------------------------------------
    # MuSiQue (multi-hop QA with question decomposition)
    # License: CC-BY-4.0
    # Source: https://huggingface.co/datasets/dgslibisey/MuSiQue
    # ------------------------------------------------------------------

    def _load_musique(self) -> list[dict[str, Any]]:
        """
        Load the MuSiQue dataset (validation split).

        MuSiQue is a multi-hop QA dataset with explicit question
        decomposition annotations.  It is harder and more controlled
        than HotpotQA, making it ideal for transfer evaluation:
        skills induced from HotpotQA should transfer to MuSiQue.
        """
        logger.info("Loading MuSiQue (validation) from HuggingFace...")
        raw_dataset = load_dataset(
            "dgslibisey/MuSiQue", split="validation"
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            question = row.get("question", "")
            answer = row.get("answer", "")
            answerable = row.get("answerable", True)

            # Build context from paragraphs
            paragraphs = row.get("paragraphs", [])
            context_parts: list[str] = []
            for para in paragraphs:
                if isinstance(para, dict):
                    title = para.get("title", "")
                    text = para.get("paragraph_text", "")
                    if title and text:
                        context_parts.append(f"[{title}]\n{text}")
            context_text = "\n\n".join(context_parts)

            description = (
                f"Answer the following multi-hop question using the provided context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}"
            )

            # Extract decomposition steps if available
            decomposition = row.get("question_decomposition", [])
            decomp_steps: list[str] = []
            for step in decomposition:
                if isinstance(step, dict):
                    decomp_steps.append(step.get("question", ""))

            tasks.append({
                "task_id": f"musique_{row.get('id', idx)}",
                "description": description,
                "expected": answer,
                "context": context_text,
                "metadata": {
                    "answerable": answerable,
                    "answer_aliases": row.get("answer_aliases", []),
                    "decomposition": decomp_steps,
                },
            })

        return tasks

    # ------------------------------------------------------------------
    # SWE-bench Lite (code bug-fixing, 300 tasks)
    # NOTE: expected answer is a patch diff — not suitable for simple
    #       substring evaluation.  Prefer GSM8K or TriviaQA for MVP.
    # ------------------------------------------------------------------

    def _load_swebench(self) -> list[dict[str, Any]]:
        """
        Load the SWE-bench Lite dataset.

        Source: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite

        WARNING: The expected answer is a patch diff, which makes
        evaluation via simple substring matching unreliable.  Use
        this benchmark only with a specialised evaluator.
        """
        logger.info("Loading SWE-bench Lite from HuggingFace...")
        raw_dataset = load_dataset(
            "princeton-nlp/SWE-bench_Lite", split="test"
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            problem = row.get("problem_statement", "")
            repo = row.get("repo", "")
            instance_id = row.get("instance_id", "")
            hints = row.get("hints_text", "") or ""

            description = (
                f"Fix the following issue in the {repo} repository.\n\n"
                f"Issue:\n{problem}"
            )
            if hints:
                description += f"\n\nHints:\n{hints}"

            # The expected answer is the patch diff
            expected_patch = row.get("patch", "")

            tasks.append({
                "task_id": f"swebench_{instance_id}",
                "description": description,
                "expected": expected_patch,
                "context": "",
                "metadata": {
                    "repo": repo,
                    "instance_id": instance_id,
                    "base_commit": row.get("base_commit", ""),
                    "version": row.get("version", ""),
                    "fail_to_pass": row.get("FAIL_TO_PASS", ""),
                },
            })

        return tasks
