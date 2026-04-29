"""Unit tests for utility modules (IO, LLM client, logging)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models import MemoryEntry, MemoryStore, Skill, TransformVariant
from src.utils.io import load_json, load_jsonl, save_json, save_jsonl


# ============================================================
# IO utilities
# ============================================================


class TestSaveJson:
    """Tests for save_json."""

    def test_save_pydantic_model(self, tmp_path):
        skill = Skill(
            name="Test Skill",
            description="A test skill",
            procedure=["Step 1"],
            source_variant=TransformVariant.TRAJ_TO_SKILL,
        )
        path = tmp_path / "skill.json"
        save_json(skill, path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "Test Skill"
        assert data["source_variant"] == "traj_to_skill"

    def test_save_dict(self, tmp_path):
        data = {"key": "value", "nested": {"a": 1}}
        path = tmp_path / "data.json"
        save_json(data, path)

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "dir" / "file.json"
        save_json({"test": True}, path)
        assert path.exists()

    def test_save_unicode(self, tmp_path):
        data = {"content": "Unicode test: Chinese characters and emojis"}
        path = tmp_path / "unicode.json"
        save_json(data, path)

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["content"] == data["content"]


class TestLoadJson:
    """Tests for load_json."""

    def test_load_dict(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"key": "value"}))

        result = load_json(path)
        assert result == {"key": "value"}

    def test_load_pydantic_model(self, tmp_path):
        skill = Skill(
            name="Test Skill",
            description="A test skill",
        )
        path = tmp_path / "skill.json"
        path.write_text(skill.model_dump_json(indent=2))

        loaded = load_json(path, model_class=Skill)
        assert isinstance(loaded, Skill)
        assert loaded.name == "Test Skill"

    def test_load_file_not_found(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_json(path)


class TestSaveJsonl:
    """Tests for save_jsonl."""

    def test_save_models(self, tmp_path):
        entries = [
            MemoryEntry(content="entry 1", category="fact"),
            MemoryEntry(content="entry 2", category="rule"),
        ]
        path = tmp_path / "entries.jsonl"
        save_jsonl(entries, path)

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["content"] == "entry 1"
        assert json.loads(lines[1])["content"] == "entry 2"

    def test_save_empty_list(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        save_jsonl([], path)
        assert path.exists()
        assert path.read_text() == ""

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "dir" / "file.jsonl"
        save_jsonl([MemoryEntry(content="test")], path)
        assert path.exists()


class TestLoadJsonl:
    """Tests for load_jsonl."""

    def test_load_models(self, tmp_path):
        entries = [
            MemoryEntry(content="entry 1", category="fact"),
            MemoryEntry(content="entry 2", category="rule"),
        ]
        path = tmp_path / "entries.jsonl"
        save_jsonl(entries, path)

        loaded = load_jsonl(path, MemoryEntry)
        assert len(loaded) == 2
        assert loaded[0].content == "entry 1"
        assert loaded[1].category == "rule"

    def test_load_file_not_found(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        result = load_jsonl(path, MemoryEntry)
        assert result == []

    def test_load_with_blank_lines(self, tmp_path):
        entry = MemoryEntry(content="test")
        path = tmp_path / "with_blanks.jsonl"
        path.write_text(entry.model_dump_json() + "\n\n\n")

        loaded = load_jsonl(path, MemoryEntry)
        assert len(loaded) == 1


# ============================================================
# LLM Client
# ============================================================


class TestLLMClient:
    """Tests for LLMClient initialisation and configuration."""

    def test_init_default_config(self):
        from src.utils.llm import LLMClient

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            client = LLMClient()
            assert client.temperature == 0.7
            assert client.max_tokens == 4096
            assert client.timeout == 120
            assert client.model == "deepseek-chat"

    def test_init_custom_config(self):
        from src.utils.llm import LLMClient

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            client = LLMClient({
                "temperature": 0.5,
                "max_tokens": 2048,
                "timeout": 60,
                "model": "deepseek-reasoner",
            })
            assert client.temperature == 0.5
            assert client.max_tokens == 2048
            assert client.timeout == 60
            assert client.model == "deepseek-reasoner"

    def test_init_env_override(self):
        from src.utils.llm import LLMClient

        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "env-key",
            "DEEPSEEK_MODEL": "env-model",
        }, clear=False):
            # Config model takes precedence over env
            client = LLMClient({"model": "config-model"})
            assert client.model == "config-model"

    def test_stats_initial(self):
        from src.utils.llm import LLMClient

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            client = LLMClient()
            assert client.stats == {"total_calls": 0, "total_tokens": 0}

    def test_chat_json_calls_chat_with_format(self):
        from src.utils.llm import LLMClient

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            client = LLMClient()
            # Mock the chat method
            client.chat = MagicMock(return_value='{"result": "ok"}')

            result = client.chat_json([{"role": "user", "content": "test"}])

            assert result == '{"result": "ok"}'
            client.chat.assert_called_once()
            call_kwargs = client.chat.call_args
            assert call_kwargs[1]["response_format"] == {"type": "json_object"}
