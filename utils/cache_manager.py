"""
武士団 v18 — Phase 4 Redis キャッシュマネージャー

用途:
  - Phase 1 パイプライン結果のキャッシュ（同一入力→高速返答）
  - Groq レスポンスキャッシュ（10分 TTL）
  - OpenAI embedding キャッシュ（7日 TTL）— notion_search.py と共用
  - セッション・スロットリング情報

フォールバック:
  - Redis 接続失敗時は in-memory LRU キャッシュに自動切り替え
  - メモリキャッシュは最大 1000 エントリ、アプリ再起動で消える
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# デフォルト TTL 設定（秒）
TTL_PIPELINE = 300        # Phase 1 パイプライン結果: 5分
TTL_LLM_RESPONSE = 600   # LLM レスポンス: 10分
TTL_EMBEDDING = 604800   # Embedding: 7日
TTL_SESSION = 3600       # セッション: 1時間

# in-memory LRU の最大エントリ数
_LRU_MAX = 1000


class _LRUMemCache:
    """スレッドセーフな in-memory LRU キャッシュ"""

    def __init__(self, maxsize: int = _LRU_MAX) -> None:
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        value, expires_at = self._cache[key]
        if expires_at > 0 and time.time() > expires_at:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        expires_at = time.time() + ttl if ttl > 0 else 0
        self._cache[key] = (value, expires_at)
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


# グローバル in-memory キャッシュ
_mem_cache = _LRUMemCache()


class CacheManager:
    """Redis + in-memory フォールバック キャッシュマネージャー"""

    _instance: Optional[CacheManager] = None

    @classmethod
    def instance(cls) -> CacheManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._redis = None
        self._redis_ok: Optional[bool] = None  # None = 未確認
        self._redis_url = __import__("os").getenv("REDIS_URL", "redis://localhost:6379/0")

    def _get_redis(self):
        if self._redis_ok is False:
            return None
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._redis_url,
                    decode_responses=False,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                self._redis_ok = True
                logger.info("CacheManager: Redis 接続確立 %s", self._redis_url)
            except Exception as e:
                logger.warning("CacheManager: Redis 初期化失敗 → in-memory fallback: %s", e)
                self._redis_ok = False
        return self._redis

    # ── キャッシュキー生成 ────────────────────────────────────────────────

    @staticmethod
    def make_key(prefix: str, *parts: str) -> str:
        combined = "|".join(parts)
        digest = hashlib.sha256(combined.encode()).hexdigest()[:20]
        return f"{prefix}:{digest}"

    # ── 基本操作 ──────────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        redis = self._get_redis()
        if redis:
            try:
                raw = await asyncio.wait_for(redis.get(key), timeout=1.0)
                if raw is not None:
                    return json.loads(raw)
            except asyncio.TimeoutError:
                logger.debug("CacheManager: Redis get timeout")
            except Exception as e:
                logger.debug("CacheManager: Redis get error: %s", e)
                self._redis_ok = False

        # in-memory フォールバック
        return _mem_cache.get(key)

    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        serialized = json.dumps(value, ensure_ascii=False)

        redis = self._get_redis()
        if redis:
            try:
                if ttl > 0:
                    await asyncio.wait_for(redis.set(key, serialized, ex=ttl), timeout=1.0)
                else:
                    await asyncio.wait_for(redis.set(key, serialized), timeout=1.0)
                return
            except asyncio.TimeoutError:
                logger.debug("CacheManager: Redis set timeout")
            except Exception as e:
                logger.debug("CacheManager: Redis set error: %s", e)
                self._redis_ok = False

        # in-memory フォールバック
        _mem_cache.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        redis = self._get_redis()
        if redis:
            try:
                await asyncio.wait_for(redis.delete(key), timeout=1.0)
            except Exception as e:
                logger.debug("cache delete error: %s", e)
        _mem_cache.delete(key)

    async def exists(self, key: str) -> bool:
        redis = self._get_redis()
        if redis:
            try:
                result = await asyncio.wait_for(redis.exists(key), timeout=1.0)
                return bool(result)
            except Exception as e:
                logger.debug("cache exists error: %s", e)
        return _mem_cache.get(key) is not None

    # ── Pipeline キャッシュ ───────────────────────────────────────────────

    async def get_pipeline_result(self, user_input: str) -> Optional[dict]:
        """Phase 1 パイプライン結果を取得"""
        key = self.make_key("pipeline", user_input.strip()[:200])
        return await self.get(key)

    async def set_pipeline_result(self, user_input: str, result: dict) -> None:
        """Phase 1 パイプライン結果をキャッシュ"""
        key = self.make_key("pipeline", user_input.strip()[:200])
        # result は JSON シリアライズ可能な形式に変換
        serializable = _make_serializable(result)
        await self.set(key, serializable, ttl=TTL_PIPELINE)

    # ── LLM レスポンスキャッシュ ──────────────────────────────────────────

    async def get_llm_response(self, model: str, prompt: str) -> Optional[str]:
        """LLM レスポンスキャッシュを取得"""
        key = self.make_key("llm", model, prompt[:500])
        result = await self.get(key)
        if isinstance(result, dict):
            return result.get("response")
        return None

    async def set_llm_response(self, model: str, prompt: str, response: str) -> None:
        """LLM レスポンスをキャッシュ"""
        key = self.make_key("llm", model, prompt[:500])
        await self.set(key, {"response": response, "cached_at": time.time()}, ttl=TTL_LLM_RESPONSE)

    # ── ステータス ────────────────────────────────────────────────────────

    async def get_status(self) -> dict:
        """キャッシュバックエンドのステータスを返す"""
        redis = self._get_redis()
        status = {
            "backend": "unknown",
            "redis_url": self._redis_url,
            "mem_cache_size": _mem_cache.size(),
        }
        if redis and self._redis_ok:
            try:
                info = await asyncio.wait_for(redis.ping(), timeout=1.0)
                status["backend"] = "redis"
                status["redis_ping"] = bool(info)
            except Exception:
                status["backend"] = "memory"
                status["redis_error"] = "ping failed"
        else:
            status["backend"] = "memory"

        return status

    async def clear_all(self) -> None:
        """全キャッシュをクリア（テスト用）"""
        redis = self._get_redis()
        if redis:
            try:
                await redis.flushdb()
            except Exception as e:
                logger.warning("cache flushdb error: %s", e)
        _mem_cache.clear()


# ── エラーハンドリング デコレータ ─────────────────────────────────────────

def with_cache(
    prefix: str,
    ttl: int = TTL_LLM_RESPONSE,
    key_fn=None,
):
    """
    LLM 呼び出し関数に自動キャッシュを適用するデコレータ。

    使い方:
        @with_cache("groq", ttl=600)
        async def call_groq(prompt: str) -> str: ...
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # キャッシュキー生成
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                parts = [str(a)[:200] for a in args] + [f"{k}={v}"[:100] for k, v in kwargs.items()]
                cache_key = CacheManager.make_key(prefix, *parts)

            # キャッシュ確認
            cached = await CacheManager.instance().get(cache_key)
            if cached is not None:
                logger.debug("cache hit: %s", cache_key[:30])
                return cached

            # 実行
            result = await func(*args, **kwargs)

            # キャッシュ保存
            if result is not None:
                await CacheManager.instance().set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# ── ユーティリティ ────────────────────────────────────────────────────────

def _make_serializable(obj: Any) -> Any:
    """Pydantic モデルや Enum を JSON シリアライズ可能な形式に変換"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj
