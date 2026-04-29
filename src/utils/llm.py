"""
Unified LLM API wrapper.

Provides a single interface for calling DeepSeek and other
OpenAI-compatible APIs, with built-in retry, token counting, and logging.
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient:
    """Unified LLM API client."""

    # Default API settings for DeepSeek.
    # Official API: https://platform.deepseek.com/api_keys
    # Models: deepseek-chat (V3), deepseek-reasoner (R1)
    API_KEY_ENV = "DEEPSEEK_API_KEY"
    BASE_URL_ENV = "DEEPSEEK_BASE_URL"
    BASE_URL_DEFAULT = "https://api.deepseek.com/v1"
    MODEL_ENV = "DEEPSEEK_MODEL"
    MODEL_DEFAULT = "deepseek-chat"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialise the LLM client.

        Args:
            config: LLM config dict from the ``llm`` section of configs/default.yaml.
        """
        config = config or {}
        self.temperature: float = config.get("temperature", 0.7)
        self.max_tokens: int = config.get("max_tokens", 4096)
        self.timeout: int = config.get("timeout", 120)
        self.max_retries: int = config.get("max_retries", 3)

        # Resolve API settings from env vars, falling back to defaults
        api_key = os.getenv(self.API_KEY_ENV, "")
        base_url = os.getenv(self.BASE_URL_ENV, self.BASE_URL_DEFAULT)
        self.model: str = config.get(
            "model", os.getenv(self.MODEL_ENV, self.MODEL_DEFAULT)
        )

        if not api_key:
            logger.warning(
                f"{self.API_KEY_ENV} is not set — LLM calls will fail. "
                f"Please configure it in the .env file."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
        )

        # Cumulative statistics
        self._total_calls: int = 0
        self._total_tokens: int = 0

        logger.info(f"LLM client initialised: model={self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: Message list, e.g.
                ``[{"role": "system", "content": "..."}]``.
            temperature: Sampling temperature (overrides default).
            max_tokens: Max tokens (overrides default).
            response_format: Response format, e.g. ``{"type": "json_object"}``.

        Returns:
            The LLM response text.
        """
        start_time = time.time()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        # Update statistics
        elapsed = time.time() - start_time
        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0
        self._total_calls += 1
        self._total_tokens += tokens_used

        logger.debug(
            f"LLM call completed: {tokens_used} tokens, {elapsed:.1f}s "
            f"(cumulative: {self._total_calls} calls, {self._total_tokens} tokens)"
        )

        return response.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat request and require a JSON-formatted response."""
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    @property
    def stats(self) -> dict[str, int]:
        """Return cumulative call statistics."""
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
        }
