"""
Benchmark dataset loader.

Loads real datasets from HuggingFace Hub:
- HotpotQA: multi-hop reasoning QA (classic, used in MemSkill paper)
- SWE-bench Lite: code bug-fixing (popular, 300 tasks)
- HotpotQA-hard: hard subset of HotpotQA for transfer evaluation
"""

from __future__ import annotations

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
    # SWE-bench Lite (code bug-fixing, 300 tasks)
    # ------------------------------------------------------------------

    def _load_swebench(self) -> list[dict[str, Any]]:
        """
        Load the SWE-bench Lite dataset.

        Source: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
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
