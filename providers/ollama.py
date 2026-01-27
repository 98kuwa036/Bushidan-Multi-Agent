"""Ollama API Client - ローカルLLMの制御"""

import asyncio
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger("shogun.ollama")


class OllamaClient:
    """Ollama REST API wrapper for model management and inference."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Model Management
    # ------------------------------------------------------------------

    async def list_models(self) -> list[dict]:
        """List all locally available models."""
        client = await self._get_client()
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])

    async def is_model_loaded(self, model: str) -> bool:
        """Check if a model is currently loaded in memory."""
        client = await self._get_client()
        resp = await client.get("/api/ps")
        resp.raise_for_status()
        running = resp.json().get("models", [])
        return any(m.get("name", "").startswith(model) for m in running)

    async def load_model(self, model: str, keep_alive: str = "30m") -> bool:
        """Load a model into memory by sending a dummy request."""
        logger.info("Loading model: %s (keep_alive=%s)", model, keep_alive)
        client = await self._get_client()
        resp = await client.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": "",
                "keep_alive": keep_alive,
            },
        )
        resp.raise_for_status()
        logger.info("Model loaded: %s", model)
        return True

    async def unload_model(self, model: str) -> bool:
        """Unload a model from memory (keep_alive=0)."""
        logger.info("Unloading model: %s", model)
        client = await self._get_client()
        try:
            resp = await client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "keep_alive": 0,
                },
            )
            resp.raise_for_status()
            logger.info("Model unloaded: %s", model)
            return True
        except httpx.HTTPError as e:
            logger.warning("Failed to unload %s: %s", model, e)
            return False

    async def unload_all(self) -> None:
        """Unload all currently loaded models."""
        client = await self._get_client()
        resp = await client.get("/api/ps")
        resp.raise_for_status()
        running = resp.json().get("models", [])
        for m in running:
            name = m.get("name", "")
            if name:
                await self.unload_model(name)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        num_ctx: int = 4096,
        stream: bool = False,
    ) -> str:
        """Generate a completion (non-streaming)."""
        client = await self._get_client()
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        if system:
            payload["system"] = system

        resp = await client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        num_ctx: int = 4096,
    ) -> AsyncIterator[str]:
        """Generate a completion with streaming."""
        client = await self._get_client()
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        if system:
            payload["system"] = system

        async with client.stream("POST", "/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    import json as _json
                    data = _json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        return

    async def chat(
        self,
        model: str,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.7,
        num_ctx: int = 4096,
    ) -> str:
        """Chat completion (multi-turn)."""
        client = await self._get_client()
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}] + messages

        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            client = await self._get_client()
            resp = await client.get("/")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def wait_ready(self, timeout: float = 60.0, interval: float = 2.0) -> bool:
        """Wait until Ollama server is ready."""
        elapsed = 0.0
        while elapsed < timeout:
            if await self.health():
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        return False
