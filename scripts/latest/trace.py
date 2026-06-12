#!/usr/bin/env python3
"""Trace logger for SkillForge experiments.

Thread-safe JSONL append-only logger for full prompt/response/score traces.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path


class TraceLogger:
    """Append-only JSONL logger for full prompt/response/score traces.

    Each line in the trace file is a JSON object with:
      - timestamp, benchmark, group, phase (train/test)
      - task_id, task_desc
      - augmented_prompt (the injected experience section)
      - response (agent's final answer)
      - expected (ground truth)
      - score (EM or pass@1)
    """
    def __init__(self, results_dir: str):
        self._lock = threading.Lock()
        self._files: dict[str, object] = {}  # benchmark -> file handle
        self._results_dir = results_dir

    def _get_file(self, benchmark: str):
        if benchmark not in self._files:
            trace_dir = Path(self._results_dir) / benchmark
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / "trace.jsonl"
            self._files[benchmark] = open(str(trace_path), "a", encoding="utf-8")
        return self._files[benchmark]

    def log(self, benchmark: str, group: str, phase: str,
            task_id: str, task_desc: str, augmented_prompt: str,
            response: str, expected: str, score: float,
            extra: dict = None):
        """Write one trace record."""
        import datetime
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "benchmark": benchmark,
            "group": group,
            "phase": phase,
            "task_id": task_id,
            "task_desc": task_desc[:500],
            "augmented_prompt": augmented_prompt,
            "response": response[:20000],
            "expected": expected,
            "score": score,
        }
        if extra:
            record.update(extra)
        with self._lock:
            f = self._get_file(benchmark)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()

    def clear_benchmark(self, benchmark: str):
        """Thread-safe: close and remove a single benchmark's trace file handle."""
        with self._lock:
            if benchmark in self._files:
                try:
                    self._files[benchmark].close()
                except Exception:
                    pass
                del self._files[benchmark]


class APIUnavailableError(Exception):
    """Raised when DeepSeek V4 Pro API is confirmed unavailable."""
    pass