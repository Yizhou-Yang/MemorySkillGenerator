"""Unit tests for configuration loading."""

from pathlib import Path

import pytest

from src.utils.config import PROJECT_ROOT, _deep_merge, load_config


class TestDeepMerge:
    """Deep-merge tests."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"llm": {"provider": "deepseek", "temperature": 0.7}}
        override = {"llm": {"temperature": 0.5}}
        result = _deep_merge(base, override)
        assert result["llm"]["provider"] == "deepseek"
        assert result["llm"]["temperature"] == 0.5


class TestLoadConfig:
    """Configuration loading tests."""

    def test_load_default_config(self):
        config = load_config("default")
        assert "llm" in config
        assert "memory" in config
        assert "benchmark" in config

    def test_load_mvp_config(self):
        config = load_config("mvp_locomo")
        assert config["benchmark"]["name"] == "locomo"
        assert config["memory"]["framework"] == "mem0"

    def test_project_root(self):
        assert (PROJECT_ROOT / "configs" / "default.yaml").exists()
