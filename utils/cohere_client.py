"""
Bushidan Multi-Agent System v18 - Cohere Client

外事 (Gaiji): Command A 03-2025 - 外部ツール・RAG・外部情報収集

Cohere Chat API v2 (https://api.cohere.com/v2/chat)

モデル:
  command-a-03-2025 : 外事 (RAG特化)

認証: COHERE_API_KEY 環境変数

Usage:
    from utils.cohere_client import create_uketuke_client, create_gaiji_client

    client = create_uketuke_client()
    result = await client.generate(
        messages=[{"role": "user", "content": "タスクを教えて"}]
    )
"""

import json
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
    Cohere API クライアント — 外事 (Command A)
    """

    def __init__(self, api_key: str, model: str = "command-a-03-2025"):
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

        # Cohere Chat API v2: system_prompt を messages の最初に追加
        final_messages = list(messages)
        if system_prompt:
            # システムプロンプトを messages の最初に挿入
            final_messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })

        body: Dict[str, Any] = {
            "model":       self.model,
            "messages":    final_messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        self._requests += 1
        start = time.time()

        try:
            # デバッグ: リクエストボディをログ出力
            logger.info(f"📤 Cohere Request Body: {json.dumps(body, ensure_ascii=False)[:300]}")

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{COHERE_BASE_URL}/chat",
                    json=body,
                    headers=headers,
                )
                # デバッグ: レスポンスコードとボディをログ出力
                logger.info(f"📥 Cohere Response {resp.status_code}: {resp.text[:500]}")

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

def create_gaiji_client() -> Optional[CohereClient]:
    """
    外事 (Command A) クライアントを生成。

    外事は外部情報・RAG・ツール連携特化の高性能エンジン。
    COHERE_API_KEY が未設定の場合は None を返す。
    """
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        logger.warning("⚠️ COHERE_API_KEY 未設定 — 外事 (Command A) 無効")
        return None
    logger.info("✅ 外事クライアント (Command A) 初期化")
    return CohereClient(api_key=api_key, model="command-a-03-2025")
