"""
Benchmark dataset loader.

Supports loading LoCoMo, GAIA, and other benchmark datasets.
Provides sample data in the MVP phase; real datasets will be integrated later.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class BenchmarkLoader:
    """Benchmark dataset loader."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.benchmark_name: str = self.config.get("name", "locomo")
        self.num_samples: int = self.config.get("num_samples", 20)

    def load(self) -> list[dict[str, Any]]:
        """
        Load benchmark tasks.

        Returns:
            A list of task dicts, each containing ``task_id``,
            ``description``, ``expected``, etc.
        """
        loader_map = {
            "locomo": self._load_locomo,
            "gaia": self._load_gaia,
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

    def _load_locomo(self) -> list[dict[str, Any]]:
        """
        Load the LoCoMo dataset.

        TODO: Integrate HuggingFace datasets for real data.
        """
        logger.warning("LoCoMo dataset not yet integrated — using sample data")
        # Sample data simulating LoCoMo long-dialogue memory tasks
        return [
            {
                "task_id": "locomo_001",
                "description": (
                    "Below is a long conversation. Answer the question.\n\n"
                    "Conversation:\n"
                    "A: I'm going to Tokyo on a business trip next week, "
                    "staying about 5 days.\n"
                    "B: Nice! I went to Tokyo last year and stayed at a "
                    "hotel in Shinjuku.\n"
                    "A: Any restaurant recommendations?\n"
                    "B: There's a great ramen shop called Ichiran near "
                    "Shinjuku Station.\n"
                    "A: Got it. By the way, I also need to visit a client "
                    "in Osaka.\n"
                    "B: For Osaka, there's lots of great food around "
                    "Dotonbori.\n\n"
                    "Question: Which cities is A travelling to for business?"
                ),
                "expected": "Tokyo and Osaka",
                "context": "",
            },
        ][: self.num_samples]

    def _load_gaia(self) -> list[dict[str, Any]]:
        """
        Load the GAIA dataset.

        TODO: Integrate the real GAIA dataset.
        """
        logger.warning("GAIA dataset not yet integrated — using sample data")
        return [
            {
                "task_id": "gaia_001",
                "description": "How many days are there in February 2024?",
                "expected": "29",
                "context": "",
            },
        ][: self.num_samples]

    def _load_swebench(self) -> list[dict[str, Any]]:
        """
        Load the SWE-Bench dataset.

        TODO: Integrate the real SWE-Bench dataset.
        """
        logger.warning("SWE-Bench dataset not yet integrated — using sample data")
        return [
            {
                "task_id": "swebench_001",
                "description": (
                    "Fix the bug in the following Python function:\n\n"
                    "```python\n"
                    "def fibonacci(n):\n"
                    "    if n <= 0:\n"
                    "        return 0\n"
                    "    elif n == 1:\n"
                    "        return 1\n"
                    "    else:\n"
                    "        return fibonacci(n-1) + fibonacci(n-3)  # bug\n"
                    "```\n\n"
                    "Where is the bug? How should it be fixed?"
                ),
                "expected": "fibonacci(n-2)",
                "context": "",
            },
        ][: self.num_samples]
