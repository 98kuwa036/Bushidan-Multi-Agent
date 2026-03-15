"""
Bushidan Multi-Agent System v12 - Cohere Client

受付 (Uketuke): Command R  - ルーター・受付・汎用中量タスク
外事 (Gaiji):   Command R+ - 外部ツール・RAG・外部情報収集

Cohere Chat API v2 (https://api.cohere.com/v2/chat)

モデル:
  command-r       : 受付 (汎用・コスト効率重視)
  command-r-plus  : 外事 (高性能・RAG特化)

認証: COHERE_API_KEY 環境変数

Usage:
    from utils.cohere_client import create_uketuke_client, create_gaiji_client

    client = create_uketuke_client()
    result = await client.generate(
        messages=[{"role": "user", "content": "タスクを教えて"}]
    )
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

COHERE_BASE_URL = "https://api.cohere.com/v2"

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class CohereClient:
    """
    Cohere API クライアント — 受付 / 外事 共通基盤

    Model: command-r (受付) または command-r-plus (外事)
    """

    def __init__(self, api_key: str, model: str = "command-r"):
        if not api_key or len(api_key) < 5:
            raise ValueError("COHERE_API_KEY が未設定または不正です (env変数を確認してください)")
        self.api_key    = api_key
        self.model      = model
        self._requests  = 0
        self._failures  = 0

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens:   int   = 2048,
        temperature:  float = 0.3,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        テキスト生成 (Cohere Chat API v2)。

        Args:
            messages:      [{"role": "user"|"assistant", "content": "..."}]
            max_tokens:    最大出力トークン数
            temperature:   温度 (0.0–1.0)
            system_prompt: システムプロンプト (任意)

        Returns:
            生成テキスト
        """
        if not HAS_HTTPX:
            raise RuntimeError(
                "httpx が必要です: pip install httpx\n"
                "  .venv/bin/pip install httpx"
            )

        body: Dict[str, Any] = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        # Cohere Chat API v2: system_prompt は "preamble" キーで渡す
        if system_prompt:
            body["preamble"] = system_prompt

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        self._requests += 1
        start = time.time()

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{COHERE_BASE_URL}/chat",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            elapsed = time.time() - start

            # Cohere v2 chat レスポンス形式:
            # {"message": {"role": "assistant",
            #              "content": [{"type": "text", "text": "..."}]}}
            content_blocks = data.get("message", {}).get("content", [])
            text = "".join(
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if not text:
                text = data.get("text", str(data))

            logger.info(
                "✅ Cohere %s 完了: %.2fs, %d文字",
                self.model, elapsed, len(text),
            )
            return text

        except Exception as e:
            self._failures += 1
            logger.error("❌ Cohere %s 失敗: %s", self.model, e)
            raise

    async def chat(self, content: str, **kwargs) -> str:
        """シンプルなチャット呼び出し (1ターン)."""
        return await self.generate(
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )

    @property
    def stats(self) -> Dict[str, int]:
        return {"requests": self._requests, "failures": self._failures}


# ─── ファクトリ関数 ───────────────────────────────────────────────────────────

def create_uketuke_client() -> Optional[CohereClient]:
    """
    受付 (Command R) クライアントを生成。

    受付は武士団の玄関口: 軽量ルーティング・汎用中量タスクを担当。
    COHERE_API_KEY が未設定の場合は None を返す。
    """
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        logger.warning("⚠️ COHERE_API_KEY 未設定 — 受付 (Command R) 無効")
        return None
    logger.info("✅ 受付クライアント (Command R) 初期化")
    return CohereClient(api_key=api_key, model="command-r")


def create_gaiji_client() -> Optional[CohereClient]:
    """
    外事 (Command R+) クライアントを生成。

    外事は外部情報・RAG・ツール連携特化の高性能エンジン。
    COHERE_API_KEY が未設定の場合は None を返す。
    """
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        logger.warning("⚠️ COHERE_API_KEY 未設定 — 外事 (Command R+) 無効")
        return None
    logger.info("✅ 外事クライアント (Command R+) 初期化")
    return CohereClient(api_key=api_key, model="command-r-plus")
