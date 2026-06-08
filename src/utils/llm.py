"""Unified LLM API wrapper — supports DeepSeek V4 Pro and CodeBuddy SDK (hy3-preview-ioa)."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

MODEL_INFO = {
    "deepseek-v4-pro": {
        "concurrency": 500,
        "context": 1_000_000,
        "max_output": 384_000,
        "tool_calling": True,
        "thinking": True,
    },
    "deepseek-v4-flash": {
        "concurrency": 2500,
        "context": 1_000_000,
        "max_output": 384_000,
        "tool_calling": True,
        "thinking": True,
    },
    "hy3-preview-ioa": {
        "concurrency": 20,
        "context": 256_000,
        "max_output": 32_000,
        "tool_calling": True,
        "thinking": True,
    },
}

class LLMClient:
    """Unified LLM client — DeepSeek (OpenAI-compat) or CodeBuddy SDK."""

    API_KEY_ENV = "DEEPSEEK_API_KEY"
    BASE_URL_ENV = "DEEPSEEK_BASE_URL"
    BASE_URL_DEFAULT = "https://api.deepseek.com/v1"
    MODEL_ENV = "DEEPSEEK_MODEL"
    MODEL_DEFAULT = "deepseek-v4-pro"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.temperature: float = config.get("temperature", 0.7)
        self.max_tokens: int = config.get("max_tokens", 4096)
        self.timeout: int = config.get("timeout", 120)
        self.max_retries: int = config.get("max_retries", 3)

        # Provider selection
        self._provider = os.getenv("LLM_PROVIDER", config.get("provider", "deepseek"))

        if self._provider == "codebuddy":
            self.model = os.getenv("CODEBUDDY_MODEL", config.get("model", "hy3-preview-ioa"))
            self._model_info = MODEL_INFO.get(self.model, MODEL_INFO["hy3-preview-ioa"])
            self.client = None  # Will use SDK query()
            logger.info(f"LLM client: provider=codebuddy, model={self.model}")
        else:
            api_key = os.getenv(self.API_KEY_ENV, "")
            base_url = os.getenv(self.BASE_URL_ENV, self.BASE_URL_DEFAULT)
            self.model = config.get("model", os.getenv(self.MODEL_ENV, self.MODEL_DEFAULT))
            self._model_info = MODEL_INFO.get(self.model, MODEL_INFO["deepseek-v4-pro"])

            if not api_key:
                logger.warning("No API key found (set DEEPSEEK_API_KEY).")

            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.timeout,
            )
            logger.info(
                f"LLM client: provider=deepseek, model={self.model}, "
                f"concurrency={self._model_info.get('concurrency', '?')}"
            )

        self._total_calls: int = 0
        self._total_tokens: int = 0

    def _chat_codebuddy(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Chat via CodeBuddy SDK or headless CLI (compatible with any Python version)."""
        # Convert messages to a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[System]\n{content}")
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous response]\n{content}")
        prompt = "\n\n".join(prompt_parts)

        # Try SDK first (fast, in-process), fall back to CLI subprocess
        try:
            return self._chat_codebuddy_sdk(prompt)
        except (ImportError, RuntimeError):
            return self._chat_codebuddy_cli(prompt)

    def _chat_codebuddy_sdk(self, prompt: str) -> str:
        """In-process SDK call (requires Python 3.12+)."""
        from codebuddy_agent_sdk import query, CodeBuddyAgentOptions

        options = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions",
            model=self.model,
            max_turns=2,
            cwd="/tmp",
        )

        async def _run():
            result = ""
            async for msg in query(prompt=prompt, options=options):
                cls = type(msg).__name__
                if "Assistant" in cls:
                    for block in msg.content:
                        block_type = type(block).__name__
                        if block_type == "TextBlock" and hasattr(block, 'text') and block.text:
                            if '429' in block.text and '额度' in block.text:
                                raise RuntimeError(f"429: {block.text[:100]}")
                            result += block.text
                    if result:
                        break
            return result

        return asyncio.run(_run())

    def _chat_codebuddy_cli(self, prompt: str) -> str:
        """Subprocess CLI call (works with any Python version)."""
        import subprocess
        import json as _json

        headless = "/root/.conda/envs/skillforge/lib/python3.11/site-packages/codebuddy_agent_sdk/bin/codebuddy-headless"
        cmd = [
            headless, "--print",
            "--model", self.model,
            "--permission-mode", "bypassPermissions",
            "--output-format", "text",
            prompt,
        ]
        env = os.environ.copy()
        env["CODEBUDDY_API_KEY"] = os.getenv("CODEBUDDY_API_KEY", "")
        env["CODEBUDDY_INTERNET_ENVIRONMENT"] = os.getenv("CODEBUDDY_INTERNET_ENVIRONMENT", "ioa")

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            else:
                logger.warning(f"CLI failed: {r.stderr[:200]}")
                return ""
        except subprocess.TimeoutExpired:
            return ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=30),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> str:
        """Send a chat completion request."""
        start_time = time.time()

        if self._provider == "codebuddy":
            content = self._chat_codebuddy(messages, temperature=temperature)
            elapsed = time.time() - start_time
            tokens_est = len(content) // 4
            self._total_calls += 1
            self._total_tokens += tokens_est
            logger.debug(f"LLM call (codebuddy): ~{tokens_est} tok, {elapsed:.1f}s")
            return content

        # DeepSeek path (OpenAI-compatible)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        response = self.client.chat.completions.create(**kwargs)

        # Update statistics
        elapsed = time.time() - start_time
        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0
        self._total_calls += 1
        self._total_tokens += tokens_used

        logger.debug(
            f"LLM call: {tokens_used} tok, {elapsed:.1f}s "
            f"(total: {self._total_calls} calls, {self._total_tokens} tok)"
        )

        # Handle different response types
        if not response.choices:
            logger.warning("Empty response.choices — possible rate limit or empty response")
            return ""

        choice = response.choices[0]

        # Tool call response
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            import json
            tool_calls = []
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                })
            return json.dumps(tool_calls)

        # Text response
        content = choice.message.content
        if content is None:
            content = getattr(choice.message, "reasoning_content", "") or ""

        return content

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

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat request with function calling tools."""
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
        )

    @property
    def supports_tool_calling(self) -> bool:
        """Check if current model supports function calling."""
        return self._model_info.get("tool_calling", True)

    @property
    def concurrency(self) -> int:
        """Max concurrent requests for current model."""
        return self._model_info.get("concurrency", 500)

    @property
    def stats(self) -> dict[str, Any]:
        """Return cumulative call statistics."""
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "model": self.model,
            "provider": self._provider,
        }
