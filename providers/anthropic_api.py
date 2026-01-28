"""Anthropic API Provider - API課金フォールバック

Pro版 claude-cli が制限に達した場合に使用。
console.anthropic.com のAPIキーで課金。
"""

import logging
import os
from typing import AsyncIterator

logger = logging.getLogger("shogun.provider.anthropic_api")

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class AnthropicAPIProvider:
    """Anthropic Messages API client (fallback for Pro rate limits)."""

    # Model mapping: role -> API model ID
    MODEL_MAP = {
        "opus": "claude-opus-4-5-20250514",
        "sonnet": "claude-sonnet-4-5-20250514",
    }

    def __init__(self, api_key: str | None = None):
        if not HAS_ANTHROPIC:
            raise RuntimeError("pip install anthropic required for API fallback")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        model: str = "sonnet",
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Single-turn generation."""
        model_id = self.MODEL_MAP.get(model, model)
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        logger.info("[API] model=%s, prompt_len=%d", model_id, len(prompt))
        response = await self._client.messages.create(**kwargs)
        text = response.content[0].text
        logger.info("[API] response_len=%d", len(text))
        return text

    async def chat(
        self,
        messages: list[dict],
        model: str = "sonnet",
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Multi-turn chat."""
        model_id = self.MODEL_MAP.get(model, model)
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def generate_stream(
        self,
        prompt: str,
        model: str = "sonnet",
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Streaming generation."""
        model_id = self.MODEL_MAP.get(model, model)
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def close(self) -> None:
        if hasattr(self._client, "_client"):
            await self._client._client.aclose()
