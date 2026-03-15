"""
utils/client_registry.py — クライアントレジストリ v14

role_key → BaseLLMClient のキャッシュ付きシングルトン。
各ロールが ClientRegistry.get().get_client(role_key) で取得する。

ヘルスチェック結果は5分間キャッシュされる。
"""

import asyncio
import logging
import os
import time
from typing import Dict, Optional

from utils.base_client import BaseLLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

# ヘルスチェックキャッシュ TTL (秒)
_HEALTH_TTL = 300


class _CohereAdapter(BaseLLMClient):
    """Cohere CohereClient → BaseLLMClient"""

    def __init__(self, model: str):
        from utils.cohere_client import CohereClient
        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key or len(api_key) < 5:
            raise RuntimeError(f"COHERE_API_KEY が未設定です (モデル: {model})")
        self._client = CohereClient(api_key=api_key, model=model)

    async def generate(self, messages, system="", max_tokens=2048, **kw):
        return await self._client.generate(
            messages=messages, max_tokens=max_tokens,
            system_prompt=system or None, **kw,
        )


class _GroqAdapter(BaseLLMClient):
    """GroqClient → BaseLLMClient"""

    def __init__(self):
        from utils.groq_client import GroqClient
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key or len(api_key) < 5:
            raise RuntimeError("GROQ_API_KEY が未設定です (斥候用)")
        self._client = GroqClient(api_key=api_key)

    async def generate(self, messages, system="", max_tokens=1000, **kw):
        if system:
            full = [{"role": "system", "content": system}] + [
                m for m in messages if m.get("role") != "system"
            ]
        else:
            full = messages
        return await self._client.generate(messages=full, max_tokens=max_tokens, **kw)

    async def health_check(self) -> bool:
        return await self._client.health_check()


class _O3MiniAdapter(BaseLLMClient):
    """O3MiniClient → BaseLLMClient"""

    def __init__(self):
        from utils.o3mini_client import O3MiniClient
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or len(api_key) < 5:
            raise RuntimeError("OPENAI_API_KEY が未設定です (軍師用)")
        self._client = O3MiniClient(api_key=api_key)

    async def generate(self, messages, system="", max_tokens=4000, **kw):
        reasoning = kw.pop("reasoning_effort", "high")
        return await self._client.generate(
            messages=messages, max_tokens=max_tokens,
            reasoning_effort=reasoning, **kw,
        )

    async def health_check(self) -> bool:
        return await self._client.health_check()


class _MistralAdapter(BaseLLMClient):
    """MistralClient → BaseLLMClient"""

    def __init__(self):
        from utils.mistral_client import MistralClient
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key or len(api_key) < 5:
            raise RuntimeError("MISTRAL_API_KEY が未設定です (参謀用)")
        self._client = MistralClient(api_key=api_key)

    async def generate(self, messages, system="", max_tokens=4000, **kw):
        return await self._client.generate(
            messages=messages, system=system, max_tokens=max_tokens, **kw,
        )


class _ClaudeAdapter(BaseLLMClient):
    """Claude CLI + API → BaseLLMClient"""

    def __init__(self, model: str):
        self._model = model
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    async def generate(self, messages, system="", max_tokens=4000, **kw):
        if len(messages) > 1:
            from utils.claude_cli_client import call_claude_with_history
            return await call_claude_with_history(
                messages=messages, model=self._model,
                api_key=self._api_key, system=system or None,
                max_tokens=max_tokens,
            )
        else:
            from utils.claude_cli_client import call_claude_with_fallback
            prompt = messages[0]["content"] if messages else ""
            return await call_claude_with_fallback(
                prompt=prompt, model=self._model,
                api_key=self._api_key, system=system or None,
                max_tokens=max_tokens,
            )


class _Gemini3Adapter(BaseLLMClient):
    """Gemini3Client → BaseLLMClient"""

    def __init__(self):
        from utils.gemini3_client import Gemini3Client
        api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        if not api_key or len(api_key) < 5:
            raise RuntimeError("GEMINI_API_KEY が未設定です (検校用)")
        self._client = Gemini3Client(api_key=api_key)

    async def generate(self, messages, system="", max_tokens=2048, **kw):
        return await self._client.generate(
            messages=messages, system=system, max_tokens=max_tokens, **kw,
        )


class _ElyzaAdapter(BaseLLMClient):
    """ELYZA Local → BaseLLMClient (Nemotron フォールバック付き)"""

    async def generate(self, messages, system="", max_tokens=2048, **kw):
        from utils.local_model_manager import get_local_model_manager
        manager = get_local_model_manager()
        elyza = await manager.get_japanese_client()
        if elyza:
            prompt = messages[-1]["content"] if messages else ""
            return await elyza.generate_japanese(prompt, context=system)
        # Nemotron fallback
        from utils.nemotron_llamacpp_client import NemotronLlamaCppClient
        client = NemotronLlamaCppClient()
        full = ([{"role": "system", "content": system}] if system else []) + messages
        return await client.generate(full)


class _NemotronAdapter(BaseLLMClient):
    """Nemotron Local → BaseLLMClient"""

    async def generate(self, messages, system="", max_tokens=4096, **kw):
        from utils.nemotron_llamacpp_client import NemotronLlamaCppClient
        client = NemotronLlamaCppClient()
        task_type = kw.pop("task_type", "confidential")
        prompt = messages[-1]["content"] if messages else ""
        return await client.process_confidential(prompt, task_type=task_type)


# ─── ロールキー → アダプタ ファクトリ ─────────────────────────────────────────
_FACTORIES = {
    "uketuke":   lambda: _CohereAdapter("command-r-7b"),      # v14.1: command-r → command-r-7b (軽量・安価)
    "gaiji":     lambda: _CohereAdapter("command-a-03-2025"), # v14.1: command-r-plus → command-a-03-2025 (最新・RAG最適)
    "seppou":    lambda: _GroqAdapter(),
    "gunshi":    lambda: _MistralAdapter(),  # v14.1: o3-mini → Mistral Large 3 に統一
    "sanbo":     lambda: _MistralAdapter(),
    "shogun":    lambda: _ClaudeAdapter("claude-sonnet-4-6"),
    "daigensui": lambda: _ClaudeAdapter("claude-opus-4-6"),
    "kengyo":    lambda: _Gemini3Adapter(),
    "yuhitsu":   lambda: _ElyzaAdapter(),
    "onmitsu":   lambda: _NemotronAdapter(),
}


class ClientRegistry:
    """
    クライアントレジストリ — シングルトン

    Usage:
        registry = ClientRegistry.get()
        client = registry.get_client("shogun")
        response = await client.generate(messages, system="...")
    """

    _instance: Optional["ClientRegistry"] = None

    def __init__(self):
        self._clients: Dict[str, BaseLLMClient] = {}
        self._health_cache: Dict[str, tuple] = {}  # key → (bool, timestamp)

    @classmethod
    def get(cls) -> "ClientRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_client(self, role_key: str) -> Optional[BaseLLMClient]:
        """role_key からクライアントを取得 (キャッシュ付き)"""
        if role_key not in self._clients:
            factory = _FACTORIES.get(role_key)
            if not factory:
                logger.warning("未知のロールキー: %s", role_key)
                return None
            try:
                self._clients[role_key] = factory()
                logger.debug("✅ %s クライアント初期化", role_key)
            except Exception as e:
                logger.error("❌ %s クライアント初期化失敗: %s", role_key, e)
                return None
        return self._clients[role_key]

    async def health_check(self, role_key: str) -> bool:
        """ロールの死活チェック (5分キャッシュ)"""
        now = time.time()
        cached = self._health_cache.get(role_key)
        if cached and (now - cached[1]) < _HEALTH_TTL:
            return cached[0]

        client = self.get_client(role_key)
        if not client:
            self._health_cache[role_key] = (False, now)
            return False

        try:
            result = await client.health_check()
        except Exception:
            result = False

        self._health_cache[role_key] = (result, now)
        return result

    def is_healthy_cached(self, role_key: str) -> bool:
        """同期キャッシュ読み取り — LangGraph _route_decision() 用。
        キャッシュが無い場合は True (楽観) を返す。"""
        cached = self._health_cache.get(role_key)
        if cached is None:
            return True  # 未チェック = 楽観的に healthy
        ts = cached[1]
        if (time.time() - ts) > _HEALTH_TTL:
            return True  # 期限切れ = 楽観
        return cached[0]

    async def health_check_all(self) -> Dict[str, bool]:
        """全ロールの死活チェック"""
        results = {}
        for key in _FACTORIES:
            results[key] = await self.health_check(key)
        return results

    @property
    def available_roles(self):
        return list(_FACTORIES.keys())
