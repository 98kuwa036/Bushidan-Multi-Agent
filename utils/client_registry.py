"""
utils/client_registry.py — クライアントレジストリ v18

role_key → BaseLLMClient のキャッシュ付きシングルトン。
各ロールが ClientRegistry.get().get_client(role_key) で取得する。

ヘルスチェック結果は5分間キャッシュされる。
"""

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


class _CerebrasAdapter(BaseLLMClient):
    """CerebrasClient → BaseLLMClient (ウェーハスケール超高速推論)"""

    def __init__(self, model: str = "llama3.1-8b"):
        from utils.cerebras_client import CerebrasClient
        api_key = os.environ.get("CEREBRAS_API_KEY", "")
        if not api_key or len(api_key) < 5:
            raise RuntimeError("CEREBRAS_API_KEY が未設定です")
        self._client = CerebrasClient(api_key=api_key, model=model)

    async def generate(self, messages, system="", max_tokens=1000, **kw):
        return await self._client.generate(
            messages=messages, system=system, max_tokens=max_tokens,
            temperature=kw.get("temperature", 0.3),
        )

    async def health_check(self) -> bool:
        return await self._client.health_check()


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
        # GroqClient は (messages, max_tokens, temperature, stream) のみ受け付ける
        groq_kw = {}
        if "temperature" in kw:
            groq_kw["temperature"] = kw["temperature"]
        return await self._client.generate(messages=full, max_tokens=max_tokens, **groq_kw)

    async def health_check(self) -> bool:
        return await self._client.health_check()



class _MistralAdapter(BaseLLMClient):
    """MistralClient → BaseLLMClient（モデル指定可能）"""

    def __init__(self, model: str = "mistral-large-latest"):
        from utils.mistral_client import MistralClient
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key or len(api_key) < 5:
            raise RuntimeError("MISTRAL_API_KEY が未設定です")
        self._client = MistralClient(api_key=api_key)
        self._model = model

    async def generate(self, messages, system="", max_tokens=4000, **kw):
        return await self._client.generate(
            messages=messages, system=system, max_tokens=max_tokens,
            model=self._model, **kw,
        )


class _ClaudeAdapter(BaseLLMClient):
    """Claude CLI + API → BaseLLMClient"""

    def __init__(self, model: str):
        self._model = model
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.last_backend: str = "unknown"

    async def generate(self, messages, system="", max_tokens=4000, **kw):
        from utils.claude_cli_client import get_last_backend
        if len(messages) > 1:
            from utils.claude_cli_client import call_claude_with_history
            result = await call_claude_with_history(
                messages=messages, model=self._model,
                api_key=self._api_key, system=system or None,
                max_tokens=max_tokens,
            )
        else:
            from utils.claude_cli_client import call_claude_with_fallback
            prompt = messages[0]["content"] if messages else ""
            result = await call_claude_with_fallback(
                prompt=prompt, model=self._model,
                api_key=self._api_key, system=system or None,
                max_tokens=max_tokens,
            )
        self.last_backend = get_last_backend()
        return result


class _Gemini3Adapter(BaseLLMClient):
    """Gemini3Client → BaseLLMClient（モデル指定可能）"""

    def __init__(self, model: str = "gemini-3.5-flash"):
        from utils.gemini3_client import Gemini3Client
        api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        if not api_key or len(api_key) < 5:
            raise RuntimeError("GEMINI_API_KEY が未設定です")
        self._client = Gemini3Client(api_key=api_key, model=model)

    async def generate(self, messages, system="", max_tokens=2048, **kw):
        # system をメッセージ先頭に挿入して Gemini API に渡す
        all_msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        # attachments (画像等) を転送 — kengyo (検校) の画像解析に必要
        attachments = kw.get("attachments")
        return await self._client.generate(
            messages=all_msgs,
            max_output_tokens=max_tokens,
            attachments=attachments,
        )

    async def health_check(self) -> bool:
        try:
            return await self._client.health_check()
        except Exception:
            return False


class _GemmaLocalAdapter(BaseLLMClient):
    """Gemma Local (3/4) Japanese → BaseLLMClient（192.168.11.239 経由）"""

    async def generate(self, messages, system="", max_tokens=2048, **kw):
        from utils.local_model_manager import LocalModelManager
        manager = LocalModelManager.get()

        prompt_parts = []
        if system:
            prompt_parts.append(system)
        for m in messages:
            role = "ユーザー" if m.get("role") == "user" else "アシスタント"
            prompt_parts.append(f"{role}: {m.get('content', '')}")
        prompt_parts.append("アシスタント:")
        prompt = "\n\n".join(prompt_parts)

        return await manager.generate_gemma(prompt, max_tokens=max_tokens)

    async def health_check(self) -> bool:
        from utils.local_model_manager import LocalModelManager
        return await LocalModelManager.get().health_check()


class _NemotronAdapter(BaseLLMClient):
    """Nemotron Local → BaseLLMClient（192.168.11.239 経由・排他制御は onmitsu.py が担う）"""

    async def generate(self, messages, system="", max_tokens=4096, **kw):
        from utils.local_model_manager import LocalModelManager
        manager = LocalModelManager.get()
        prompt = messages[-1]["content"] if messages else ""
        return await manager.generate_nemotron(prompt, max_tokens=max_tokens)

    async def health_check(self) -> bool:
        from utils.local_model_manager import LocalModelManager
        return await LocalModelManager.get().health_check()


# claude_fallback と crosscheck_heavy は同一モデル (gemini-3.1-pro-preview) を使うため
# インスタンスを共有してコネクションを節約する
_gemini_pro_shared: Optional["_Gemini3Adapter"] = None


def _gemini_pro_factory() -> "_Gemini3Adapter":
    global _gemini_pro_shared
    if _gemini_pro_shared is None:
        _gemini_pro_shared = _Gemini3Adapter("gemini-3.1-pro")
    return _gemini_pro_shared


# ─── ロールキー → アダプタ ファクトリ ─────────────────────────────────────────
_FACTORIES = {
    "uketuke":         lambda: _Gemini3Adapter("gemini-2.5-flash-lite"),               # 受付フォールバック: Gemini 2.5 Flash-Lite
    "gaiji":           lambda: _CohereAdapter("command-a-plus-05-2026"),               # RAG特化: Command A+ (MoE・Vision・Reasoning統合)
    "sanbo":           lambda: _Gemini3Adapter("gemini-3.5-flash"),                    # 汎用処理+ツール実行: Gemini 3.5 Flash
    "shogun":          lambda: _ClaudeAdapter("claude-sonnet-4-6"),                    # 計画立案: Sonnet
    "daigensui":       lambda: _ClaudeAdapter("claude-opus-4-6"),                      # 最終判断: Opus (変更なし)
    "kengyo":          lambda: _Gemini3Adapter("gemini-3.5-flash"),                    # 画像解析: Gemini 3.5 Flash (Vision統合)
    "onmitsu":         lambda: _NemotronAdapter(),                                     # 機密+日本語処理: Gemma4 Local
    "claude_fallback": _gemini_pro_factory,                                            # Claude障害時: Gemini 3.1 Pro (共有)
    "crosscheck_light":lambda: _CerebrasAdapter("llama-3.3-70b"),                      # 軽量クロスチェック: Cerebras Llama 3.3 70B (2100 tok/s)
    "crosscheck_heavy":_gemini_pro_factory,                                            # 重量クロスチェック: Gemini 3.1 Pro (共有)
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
        """
        ロールの死活チェック。
        バックグラウンドタスクが実行した最新の結果を返す。
        """
        cached = self._health_cache.get(role_key)
        if cached:
            return cached[0]

        # キャッシュがない場合のみ初期化（ただし外部通信は最小限に）
        client = self.get_client(role_key)
        if not client:
            return False

        # ここで外部通信を伴う client.health_check() は呼ばず、
        # 次のバックグラウンドタスクが回るまで True (楽観) を返して待つ
        return True

    def is_healthy_cached(self, role_key: str) -> bool:
        """同期キャッシュ読み取り — 外部通信ゼロで即答する"""
        cached = self._health_cache.get(role_key)
        if cached is None:
            return True  # 未チェック = 楽観的に healthy
        return cached[0]

    def force_update_health(self, role_key: str, status: bool) -> None:
        """バックグラウンドタスク等から健康状態を強制上書きする"""
        self._health_cache[role_key] = (status, time.time())

    async def health_check_all(self) -> Dict[str, bool]:
        """全ロールの死活チェックを【並列】で実行。バックグラウンドタスク用。"""
        roles = list(_FACTORIES.keys())
        
        async def _check_one(rk):
            client = self.get_client(rk)
            if not client:
                return rk, False
            try:
                # 実際の外部通信を伴うチェック
                res = await client.health_check()
                return rk, res
            except Exception:
                return rk, False

        # すべての役職を一斉に診断
        results = await asyncio.gather(*[_check_one(rk) for rk in roles], return_exceptions=True)
        
        final = {}
        for res in results:
            if isinstance(res, tuple):
                rk, status = res
                self.force_update_health(rk, status)
                final[rk] = status
        return final

    @property
    def available_roles(self):
        return list(_FACTORIES.keys())
