"""OpenVINO Client - 侍大将R1推論サーバー接続

CT 101 (192.168.1.11:11434) で稼働する OpenVINO R1 サーバーへの
HTTP クライアント。Ollama互換APIフォーマットを使用。
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger("shogun.provider.openvino")


class OpenVINOClient:
    """HTTP client for the OpenVINO R1 inference server on CT 101."""

    def __init__(self, base_url: str = "http://192.168.1.11:11434"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        )

    async def health(self) -> bool:
        """Check if the OpenVINO server is running."""
        try:
            resp = await self._client.get("/")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.warning("OpenVINO server unreachable: %s", self.base_url)
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        system: str = "",
    ) -> str:
        """Generate text from the R1 model.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            system: System prompt (prepended to prompt).

        Returns:
            Generated text response.
        """
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt

        payload = {
            "model": "taisho-openvino",
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        logger.info("[R1] prompt_len=%d, max_tokens=%d", len(full_prompt), max_tokens)

        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("response", "")
            logger.info("[R1] response_len=%d", len(result))
            return result
        except httpx.TimeoutException:
            logger.error("[R1] Timeout (300s)")
            raise
        except httpx.HTTPStatusError as e:
            logger.error("[R1] HTTP error: %s", e)
            raise

    async def generate_with_think(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.6,
        system: str = "",
    ) -> dict[str, str]:
        """Generate with <think> tag parsing.

        Returns:
            {"thinking": "...", "answer": "..."}
        """
        raw = await self.generate(prompt, max_tokens, temperature, system)
        return self._parse_think(raw)

    @staticmethod
    def _parse_think(text: str) -> dict[str, str]:
        """Parse <think>...</think> from R1 output."""
        if "<think>" in text and "</think>" in text:
            think_start = text.index("<think>") + len("<think>")
            think_end = text.index("</think>")
            thinking = text[think_start:think_end].strip()
            answer = text[think_end + len("</think>"):].strip()
            return {"thinking": thinking, "answer": answer}
        return {"thinking": "", "answer": text.strip()}

    async def close(self) -> None:
        await self._client.aclose()
