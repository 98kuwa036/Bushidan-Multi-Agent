"""Anthropic API Client - クラウドLLM (将軍/家老)"""

import logging
import os
from typing import AsyncIterator

logger = logging.getLogger("shogun.anthropic")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logger.warning("anthropic package not installed. Cloud agents unavailable.")


class AnthropicClient:
    """Wrapper for Anthropic Messages API."""

    def __init__(self, api_key: str | None = None):
        if not HAS_ANTHROPIC:
            raise RuntimeError(
                "anthropic package is required. Install: pip install anthropic"
            )
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Export it or pass api_key to AnthropicClient."
            )
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Single-turn generation."""
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def chat(
        self,
        model: str,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Multi-turn chat completion."""
        kwargs: dict = {
            "model": model,
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
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Streaming generation."""
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": model,
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
        """Close underlying HTTP client."""
        if hasattr(self._client, "_client"):
            await self._client._client.aclose()
