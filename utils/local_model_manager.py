"""
utils/local_model_manager.py — ローカルLLMサーバークライアント v15

pct100 から 192.168.11.239:8082 の local_llm_server.py を HTTP 経由で呼び出す。

v15 排他制御設計:
  - Gemma3 27B (右筆) と Nemotron (隠密) は 32GB RAM で排他動作
  - asyncio.Lock でアプリ側からも同時呼び出しを防止
  - generate_gemma() / generate_nemotron() がロックを内部管理
  - ロック取得タイムアウト: 180秒 (切り替え + 推論 + バッファ)
  - 切り替え失敗時: 最大2回リトライ (3秒間隔)

使用側 (onmitsu.py / yuhitsu.py) は generate_* を呼ぶだけでよい。
switch_to_*/switch_from_* の直接呼び出し不要。
"""

import asyncio
import logging
import os
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

LOCAL_LLM_BASE_URL  = os.environ.get("LOCAL_LLM_URL",     "http://192.168.11.239:8082")  # .env: LOCAL_LLM_URL
LOCAL_LLM_TIMEOUT   = int(os.environ.get("LOCAL_LLM_TIMEOUT",  "180"))  # 推論タイムアウト
_SWITCH_TIMEOUT     = 60    # モデル切り替えタイムアウト (秒)
_LOCK_WAIT_TIMEOUT  = 180   # ロック取得タイムアウト (秒)
_SWITCH_RETRIES     = 2     # 切り替え失敗時リトライ回数
_SWITCH_RETRY_WAIT  = 3.0   # リトライ間隔 (秒)


class LocalModelManager:
    """
    ローカルLLMサーバー HTTP クライアント（シングルトン）

    排他制御:
      - _model_lock: asyncio.Lock — 同時に1タスクのみ generate_* を実行可能
      - _active_model: "gemma" | "nemotron" — pct100 側の認識状態
      - generate_gemma() → ロック取得 → (必要なら gemma に切り替え) → 推論 → 解放
      - generate_nemotron() → ロック取得 → nemotron に切り替え → 推論
                              → gemma に復帰 → 解放
    """

    _instance: Optional["LocalModelManager"] = None

    def __init__(self, base_url: str = LOCAL_LLM_BASE_URL):
        self.base_url       = base_url.rstrip("/")
        self._model_lock    = asyncio.Lock()
        self._active_model  = "gemma"   # 起動時はサーバーの初期状態に合わせる
        self._lock_holder   = ""        # デバッグ用
        self._lock_since    = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("📦 LocalModelManager → %s", self.base_url)

    @classmethod
    def get(cls) -> "LocalModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # 公開 API — generate_gemma / generate_nemotron
    # =========================================================================

    async def generate_gemma(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        Gemma3 27B で推論。排他ロックを内部管理する。

        隠密 (Nemotron) 実行中は最大 _LOCK_WAIT_TIMEOUT 秒待機する。
        """
        await self._acquire_lock("yuhitsu(gemma)")
        try:
            # 現在 Nemotron が active なら Gemma3 に復帰してから推論
            if self._active_model != "gemma":
                await self._switch_with_retry("gemma")
            return await self._call_generate("/generate/gemma", prompt, max_tokens)
        finally:
            self._release_lock()

    async def generate_nemotron(self, prompt: str, max_tokens: int = 4096) -> str:
        """
        Nemotron 3 で推論。ロック取得 → Nemotron 切り替え → 推論 → Gemma3 復帰。

        右筆 (Gemma3) 実行中は最大 _LOCK_WAIT_TIMEOUT 秒待機する。
        """
        await self._acquire_lock("onmitsu(nemotron)")
        try:
            await self._switch_with_retry("nemotron")
            return await self._call_generate(
                "/generate/nemotron", prompt, max_tokens, temperature=0.5
            )
        finally:
            # 必ず Gemma3 に復帰してからロック解放
            try:
                await self._switch_with_retry("gemma")
            except Exception as e:
                logger.error("⚠️ Gemma3 復帰失敗 (要手動対応): %s", e)
                # "unknown" にすると毎回 switch が走るため nemotron のまま記録する
                self._active_model = "nemotron"
            self._release_lock()

    async def close(self) -> None:
        """永続セッションを閉じる (アプリ終了時に呼び出す)"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """永続 aiohttp セッションを返す (遅延初期化)"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # =========================================================================
    # ステータス / ヘルスチェック
    # =========================================================================

    async def health_check(self) -> bool:
        try:
            s = await self._get_session()
            async with s.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                return r.status == 200
        except Exception as e:
            logger.debug("LocalModelManager health_check 失敗: %s", e)
            return False

    async def get_status(self) -> dict:
        try:
            s = await self._get_session()
            async with s.get(
                f"{self.base_url}/status",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                remote = await r.json()
        except Exception as e:
            remote = {"error": str(e)}

        return {
            **remote,
            "lock_held":    self._model_lock.locked(),
            "lock_holder":  self._lock_holder,
            "lock_seconds": round(time.time() - self._lock_since, 1) if self._model_lock.locked() else 0,
            "local_active": self._active_model,
        }

    # =========================================================================
    # 内部ヘルパー
    # =========================================================================

    async def _acquire_lock(self, holder: str) -> None:
        """排他ロックを取得。タイムアウト超過で RuntimeError を送出。"""
        if self._model_lock.locked():
            logger.info(
                "⏳ [%s] モデルロック待機中 (保持者: %s, %.0f秒経過)",
                holder, self._lock_holder, time.time() - self._lock_since,
            )
        try:
            await asyncio.wait_for(self._model_lock.acquire(), timeout=_LOCK_WAIT_TIMEOUT)
            self._lock_holder = holder
            self._lock_since  = time.time()
            logger.debug("🔒 ロック取得: %s", holder)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"ローカルLLMロック取得タイムアウト ({_LOCK_WAIT_TIMEOUT}s) — "
                f"保持者: {self._lock_holder}"
            )

    def _release_lock(self) -> None:
        elapsed = time.time() - self._lock_since
        logger.debug("🔓 ロック解放: %s (%.1fs)", self._lock_holder, elapsed)
        self._lock_holder = ""
        try:
            self._model_lock.release()
        except RuntimeError:
            pass  # 既に解放済み

    async def _switch_with_retry(self, target: str) -> None:
        """
        モデル切り替え with リトライ。

        Args:
            target: "gemma" or "nemotron"
        """
        if self._active_model == target:
            return  # 既に目的のモデルがアクティブ

        endpoint = f"/switch/{target}"
        for attempt in range(1, _SWITCH_RETRIES + 2):
            try:
                s = await self._get_session()
                async with s.post(
                    f"{self.base_url}{endpoint}",
                    timeout=aiohttp.ClientTimeout(total=_SWITCH_TIMEOUT),
                ) as r:
                    data = await r.json()
                    if data.get("success"):
                        self._active_model = target
                        logger.info(
                            "✅ モデル切り替え完了: %s → %s (試行%d)",
                            "?" if self._active_model == target else "other",
                            target, attempt,
                        )
                        return
                    msg = data.get("message", "unknown error")
                    logger.warning("⚠️ 切り替え失敗 (試行%d): %s", attempt, msg)
            except Exception as e:
                logger.warning("⚠️ 切り替えエラー (試行%d): %s", attempt, e)

            if attempt <= _SWITCH_RETRIES:
                await asyncio.sleep(_SWITCH_RETRY_WAIT)

        raise RuntimeError(
            f"モデル切り替え失敗: {self._active_model} → {target} "
            f"({_SWITCH_RETRIES + 1}回試行)"
        )

    async def _call_generate(
        self,
        endpoint: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
    ) -> str:
        """推論 HTTP リクエスト共通ヘルパー (永続セッション使用)"""
        payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        s = await self._get_session()
        async with s.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=LOCAL_LLM_TIMEOUT),
        ) as r:
            if r.status != 200:
                body = await r.text()
                raise RuntimeError(f"LLM error {r.status} [{endpoint}]: {body[:200]}")
            data = await r.json()
            return data["content"]

    # =========================================================================
    # 後方互換 — onmitsu.py の switch_to_* 呼び出しをサポート
    # (非推奨: generate_nemotron() に切り替えてください)
    # =========================================================================

    async def switch_to_nemotron(self) -> bool:
        """非推奨: generate_nemotron() が内部で switch を管理します"""
        logger.warning("switch_to_nemotron() は非推奨。generate_nemotron() を使用してください")
        try:
            await self._switch_with_retry("nemotron")
            return True
        except Exception:
            return False

    async def switch_to_gemma(self) -> bool:
        """非推奨: generate_nemotron() が内部で switch を管理します"""
        logger.warning("switch_to_gemma() は非推奨。generate_nemotron() を使用してください")
        try:
            await self._switch_with_retry("gemma")
            return True
        except Exception:
            return False


# 後方互換性
def get_local_model_manager() -> LocalModelManager:
    return LocalModelManager.get()
