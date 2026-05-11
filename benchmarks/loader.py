"""
Benchmark dataset loader.

Loads real datasets from HuggingFace Hub:

Primary benchmarks (active):
- GAIA (L1/L2/L3): general assistant tasks (466 questions, EM + human eval)
- ALFWorld: embodied text game (~134 tasks, task completion rate)
- HotpotQA: multi-hop QA (dev set subsample, EM/F1)
- 2WikiMultihopQA: multi-hop QA (dev set subsample, EM/F1)
- AIME 24/25: math competition (30 questions each, answer matching)
- TravelPlanner: long-horizon planning (~180 tasks, multi-constraint satisfaction)
- WebShop: web shopping simulation (test subset, task completion rate)

Legacy benchmarks (disabled, still loadable):
- HotpotQA-hard: hard subset of HotpotQA for transfer evaluation
- TriviaQA: single-hop factoid QA (classic baseline)
- GSM8K: grade-school math reasoning with chain-of-thought
- MuSiQue: multi-hop QA with explicit question decomposition
- SWE-bench Lite: code bug-fixing (300 tasks, patch-diff expected answers)
"""

from __future__ import annotations

import json
import re
from typing import Any

from datasets import load_dataset
from loguru import logger


# Primary benchmarks (new evaluation suite)
PRIMARY_BENCHMARKS = [
    "gaia",
    "alfworld",
    "hotpotqa",
    "2wikimultihopqa",
    "aime",
    "travelplanner",
    "webshop",
]

# Legacy benchmarks (disabled but not deleted, still loadable by name)
LEGACY_BENCHMARKS = [
    "hotpotqa_hard",
    "triviaqa",
    "gsm8k",
    "musique",
    "swebench",
]


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
            # Primary benchmarks
            "gaia": self._load_gaia,
            "alfworld": self._load_alfworld,
            "hotpotqa": self._load_hotpotqa,
            "2wikimultihopqa": self._load_2wikimultihopqa,
            "aime": self._load_aime,
            "travelplanner": self._load_travelplanner,
            "webshop": self._load_webshop,
            # Legacy benchmarks (disabled but still loadable)
            "hotpotqa_hard": self._load_hotpotqa_hard,
            "triviaqa": self._load_triviaqa,
            "gsm8k": self._load_gsm8k,
            "musique": self._load_musique,
            "swebench": self._load_swebench,
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

    # ==================================================================
    # PRIMARY BENCHMARKS
    # ==================================================================

    # ------------------------------------------------------------------
    # GAIA (L1/L2/L3) — General Assistant (EM + human evaluation)
    # Source: Intelligent-Internet/ii-agent_gaia-benchmark_validation
    # 165 tasks with Level 1/2/3, Question + Final answer
    # ------------------------------------------------------------------

    def _load_gaia(self) -> list[dict[str, Any]]:
        """
        Load the GAIA benchmark (validation split).

        GAIA evaluates general-purpose assistant capabilities across
        three difficulty levels. Evaluation uses Exact Match on the
        final answer string.

        Source: https://huggingface.co/datasets/Intelligent-Internet/ii-agent_gaia-benchmark_validation
        """
        logger.info("Loading GAIA (validation) from HuggingFace...")
        raw_dataset = load_dataset(
            "Intelligent-Internet/ii-agent_gaia-benchmark_validation",
            split="train",
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            question = row.get("Question", "")
            level = row.get("Level", "")
            final_answer = row.get("Final answer", "")
            task_id_raw = row.get("task_id", str(idx))

            description = (
                f"Answer the following question accurately.\n\n"
                f"Question: {question}"
            )

            tasks.append({
                "task_id": f"gaia_{task_id_raw}",
                "description": description,
                "expected": final_answer,
                "context": "",
                "metadata": {
                    "level": str(level),
                    "file_name": row.get("file_name", ""),
                    "annotator_metadata": row.get("Annotator Metadata", ""),
                },
            })

        return tasks

    # ------------------------------------------------------------------
    # ALFWorld — Embodied Text Game (task completion rate)
    # Source: awawa-agi/alfworld-raw (eval_out_of_distribution split)
    # 134 tasks with task_type + walkthrough (gold solution)
    # ------------------------------------------------------------------

    def _load_alfworld(self) -> list[dict[str, Any]]:
        """
        Load the ALFWorld benchmark (eval out-of-distribution split).

        ALFWorld is an embodied text-game benchmark where agents must
        complete household tasks. Evaluation is based on task completion
        rate (whether the agent reaches the goal state).

        Source: https://huggingface.co/datasets/awawa-agi/alfworld-raw
        """
        logger.info("Loading ALFWorld (eval_out_of_distribution) from HuggingFace...")
        raw_dataset = load_dataset(
            "awawa-agi/alfworld-raw",
            split="eval_out_of_distribution",
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            task_id_raw = row.get("id", str(idx))
            task_type = row.get("task_type", "")
            game_file_path = row.get("game_file_path", "")

            # Parse game_content JSON for walkthrough (gold solution)
            game_content_str = row.get("game_content", "{}")
            try:
                game_content = json.loads(game_content_str)
            except (json.JSONDecodeError, TypeError):
                game_content = {}

            walkthrough = game_content.get("walkthrough", [])
            # Build a human-readable task description from the file path
            # Format: task_type-Object-Receptacle/trial_xxx
            task_desc = self._alfworld_task_description(task_type, game_file_path)

            description = (
                f"Complete the following household task in a text-based environment.\n\n"
                f"Task: {task_desc}\n"
                f"Task type: {task_type}"
            )

            # Expected is the walkthrough steps (gold solution)
            expected = " -> ".join(walkthrough) if walkthrough else ""

            tasks.append({
                "task_id": f"alfworld_{task_id_raw}",
                "description": description,
                "expected": expected,
                "context": "",
                "metadata": {
                    "task_type": task_type,
                    "game_file_path": game_file_path,
                    "walkthrough_steps": walkthrough,
                    "num_steps": len(walkthrough),
                },
            })

        return tasks

    @staticmethod
    def _alfworld_task_description(task_type: str, game_file_path: str) -> str:
        """Generate a human-readable task description from ALFWorld metadata."""
        # Parse game_file_path like:
        # look_at_obj_in_light-CD-None-DeskLamp-308/trial_xxx
        parts = game_file_path.split("/")[0] if "/" in game_file_path else game_file_path
        segments = parts.split("-")

        task_type_readable = task_type.replace("_", " ")
        if len(segments) >= 3:
            obj = segments[1] if segments[1] != "None" else ""
            receptacle = segments[3] if len(segments) > 3 and segments[3] != "None" else ""
            if obj and receptacle:
                return f"{task_type_readable}: {obj} with {receptacle}"
            elif obj:
                return f"{task_type_readable}: {obj}"

        return task_type_readable

    # ------------------------------------------------------------------
    # HotpotQA (multi-hop reasoning, classic benchmark)
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

    # ------------------------------------------------------------------
    # 2WikiMultihopQA (multi-hop QA, EM/F1)
    # Source: scholarly-shadows-syndicate/2WikiMultiHopQA
    # 12576 validation tasks with question + answer + context
    # ------------------------------------------------------------------

    def _load_2wikimultihopqa(self) -> list[dict[str, Any]]:
        """
        Load the 2WikiMultihopQA dataset (validation split).

        2WikiMultihopQA is a multi-hop QA dataset constructed from
        two Wikipedia articles, requiring compositional reasoning.
        Evaluation uses EM and token-level F1.

        Source: https://huggingface.co/datasets/scholarly-shadows-syndicate/2WikiMultiHopQA
        """
        logger.info("Loading 2WikiMultihopQA (validation) from HuggingFace...")
        raw_dataset = load_dataset(
            "scholarly-shadows-syndicate/2WikiMultiHopQA",
            split="validation",
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            question = row.get("question", "")
            answer = row.get("answer", "")
            row_id = row.get("_id", str(idx))
            row_type = row.get("type", "")

            # Parse context (stored as JSON string)
            context_raw = row.get("context", "")
            context_text = self._parse_2wiki_context(context_raw)

            description = (
                f"Answer the following multi-hop question using the provided context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}"
            )

            # Parse supporting facts
            supporting_facts_raw = row.get("supporting_facts", "")
            try:
                supporting_facts = json.loads(supporting_facts_raw) if isinstance(
                    supporting_facts_raw, str
                ) else supporting_facts_raw
            except (json.JSONDecodeError, TypeError):
                supporting_facts = []

            tasks.append({
                "task_id": f"2wikimqa_{row_id}",
                "description": description,
                "expected": answer,
                "context": context_text,
                "metadata": {
                    "type": row_type,
                    "supporting_facts": supporting_facts,
                },
            })

        return tasks

    @staticmethod
    def _parse_2wiki_context(context_raw: str) -> str:
        """Parse 2WikiMultihopQA context from JSON string to readable text."""
        try:
            context_data = json.loads(context_raw) if isinstance(
                context_raw, str
            ) else context_raw
        except (json.JSONDecodeError, TypeError):
            return str(context_raw)[:2000]

        if not isinstance(context_data, list):
            return str(context_data)[:2000]

        parts: list[str] = []
        for item in context_data:
            if isinstance(item, list) and len(item) >= 2:
                title = item[0]
                sentences = item[1]
                if isinstance(sentences, list):
                    paragraph = " ".join(sentences)
                else:
                    paragraph = str(sentences)
                parts.append(f"[{title}]\n{paragraph}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # AIME 24/25 — Math Competition (answer matching)
    # Source: Maxwell-Jia/AIME_2024 (30 tasks)
    # Each task has Problem, Solution, Answer (integer)
    # ------------------------------------------------------------------

    def _load_aime(self) -> list[dict[str, Any]]:
        """
        Load the AIME 2024 dataset (30 competition math problems).

        AIME (American Invitational Mathematics Examination) problems
        require advanced mathematical reasoning. Each answer is an
        integer from 000 to 999. Evaluation uses exact answer matching.

        Source: https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
        """
        logger.info("Loading AIME 2024 from HuggingFace...")
        raw_dataset = load_dataset(
            "Maxwell-Jia/AIME_2024",
            split="train",
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            problem = row.get("Problem", "")
            solution = row.get("Solution", "")
            answer = str(row.get("Answer", ""))
            problem_id = row.get("ID", str(idx))

            description = (
                f"Solve the following AIME competition math problem. "
                f"The answer is an integer between 000 and 999.\n\n"
                f"Problem: {problem}"
            )

            tasks.append({
                "task_id": f"aime_{problem_id}",
                "description": description,
                "expected": answer,
                "context": "",
                "metadata": {
                    "problem_id": problem_id,
                    "solution": solution,
                },
            })

        return tasks

    # ------------------------------------------------------------------
    # TravelPlanner — Long-horizon Planning (multi-constraint satisfaction)
    # Source: osunlp/TravelPlanner (validation config, 180 tasks)
    # ------------------------------------------------------------------

    def _load_travelplanner(self) -> list[dict[str, Any]]:
        """
        Load the TravelPlanner benchmark (validation split, 180 tasks).

        TravelPlanner evaluates long-horizon planning ability with
        multiple constraints (budget, dates, preferences). Evaluation
        is based on multi-constraint satisfaction rate.

        Source: https://huggingface.co/datasets/osunlp/TravelPlanner
        """
        logger.info("Loading TravelPlanner (validation) from HuggingFace...")
        raw_dataset = load_dataset(
            "osunlp/TravelPlanner",
            "validation",
            split="validation",
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            query = row.get("query", "")
            org = row.get("org", "")
            dest = row.get("dest", "")
            days = row.get("days", "")
            people_number = row.get("people_number", "")
            budget = row.get("budget", "")
            local_constraint = row.get("local_constraint", "")
            level = row.get("level", "")
            reference_info = row.get("reference_information", "")

            description = (
                f"Create a detailed travel plan based on the following requirements.\n\n"
                f"Query: {query}\n\n"
                f"Constraints:\n"
                f"- Origin: {org}\n"
                f"- Destination: {dest}\n"
                f"- Duration: {days} days\n"
                f"- Number of people: {people_number}\n"
                f"- Budget: ${budget}\n"
                f"- Local constraints: {local_constraint}"
            )

            tasks.append({
                "task_id": f"travelplanner_{idx}",
                "description": description,
                "expected": reference_info,
                "context": "",
                "metadata": {
                    "org": org,
                    "dest": dest,
                    "days": days,
                    "people_number": people_number,
                    "budget": budget,
                    "local_constraint": local_constraint,
                    "level": level,
                },
            })

        return tasks

    # ------------------------------------------------------------------
    # WebShop — Web Shopping Simulation (task completion rate)
    # Source: Skyler215/webshop-agent-cot (test split, 2225 tasks)
    # ------------------------------------------------------------------

    def _load_webshop(self) -> list[dict[str, Any]]:
        """
        Load the WebShop benchmark (test split).

        WebShop simulates web shopping tasks where agents must find
        and purchase products matching specific criteria. Evaluation
        is based on task completion rate (correct product purchased).

        Source: https://huggingface.co/datasets/Skyler215/webshop-agent-cot
        """
        logger.info("Loading WebShop (test) from HuggingFace...")
        raw_dataset = load_dataset(
            "Skyler215/webshop-agent-cot",
            split="test",
        )

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            prompt = row.get("prompt", "")
            response = row.get("response", "")
            row_id = row.get("id", str(idx))

            # Extract the instruction from the prompt
            instruction = self._extract_webshop_instruction(prompt)

            description = (
                f"Complete the following web shopping task.\n\n"
                f"Instruction: {instruction}\n\n"
                f"Context:\n{prompt}"
            )

            tasks.append({
                "task_id": f"webshop_{row_id}",
                "description": description,
                "expected": response,
                "context": prompt,
                "metadata": {
                    "original_id": row_id,
                },
            })

        return tasks

    @staticmethod
    def _extract_webshop_instruction(prompt: str) -> str:
        """Extract the shopping instruction from a WebShop prompt."""
        # Look for "Instruction:" line in the prompt
        match = re.search(r"Instruction:\s*(.+?)(?:\n|$)", prompt)
        if match:
            return match.group(1).strip()
        # Fallback: return first 200 chars
        return prompt[:200]

    # ==================================================================
    # LEGACY BENCHMARKS (disabled but not deleted)
    # ==================================================================

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

    def _load_triviaqa(self) -> list[dict[str, Any]]:
        """
        Load the TriviaQA dataset (rc config, validation split).

        TriviaQA is a classic single-hop factoid QA benchmark.
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

    def _load_gsm8k(self) -> list[dict[str, Any]]:
        """
        Load the GSM8K dataset (main config, test split).

        GSM8K is a math reasoning benchmark that naturally produces
        chain-of-thought trajectories.
        """
        logger.info("Loading GSM8K (main, test) from HuggingFace...")
        raw_dataset = load_dataset("openai/gsm8k", "main", split="test")

        tasks: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_dataset):
            if idx >= self.num_samples:
                break

            question = row.get("question", "")
            raw_answer = row.get("answer", "")
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
        lines = raw_answer.strip().split("\n")
        return lines[-1].strip() if lines else raw_answer

    def _load_musique(self) -> list[dict[str, Any]]:
        """
        Load the MuSiQue dataset (validation split).

        MuSiQue is a multi-hop QA dataset with explicit question
        decomposition annotations.
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

    def _load_swebench(self) -> list[dict[str, Any]]:
        """
        Load the SWE-bench Lite dataset.

        Source: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite

        WARNING: The expected answer is a patch diff, which makes
        evaluation via simple substring matching unreliable.
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
