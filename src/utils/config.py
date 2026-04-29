"""
Configuration loader.

Supports loading configs from YAML files and merging with .env environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_env() -> None:
    """Load environment variables from the .env file."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from: {env_path}")
    else:
        logger.warning(
            f".env file not found: {env_path}. "
            f"Please copy .env.example to .env and fill in actual values."
        )


def load_config(config_name: str = "default") -> dict[str, Any]:
    """
    Load an experiment configuration file.

    Args:
        config_name: Config file name (without .yaml extension).

    Returns:
        Merged configuration dictionary.
    """
    # Load default config
    default_path = CONFIGS_DIR / "default.yaml"
    config: dict[str, Any] = {}
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

    # If a non-default config is specified, merge overrides
    if config_name != "default":
        override_path = CONFIGS_DIR / f"{config_name}.yaml"
        if override_path.exists():
            with open(override_path, "r", encoding="utf-8") as fh:
                override = yaml.safe_load(fh) or {}
            config = _deep_merge(config, override)
            logger.info(f"Loaded config override: {override_path}")
        else:
            logger.warning(f"Config file not found: {override_path}, using defaults")

    # Override LLM-related config from environment variables
    _override_from_env(config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts; *override* takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _override_from_env(config: dict) -> None:
    """Override LLM config entries from environment variables."""
    provider = os.getenv("LLM_PROVIDER")
    if provider:
        config.setdefault("llm", {})["provider"] = provider

    log_level = os.getenv("LOG_LEVEL")
    if log_level:
        config.setdefault("output", {})["log_level"] = log_level
