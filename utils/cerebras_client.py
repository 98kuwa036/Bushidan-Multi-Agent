"""
utils/cerebras_client.py — Cerebras Cloud API クライアント

ウェーハスケールチップによる超高速推論 (~2000 tok/s)。
クロスチェック・軽量検証用途に特化。

利用可能モデル:
  llama3.1-8b       — 超高速・軽量 (クロスチェック推奨)
  gpt-oss-120b      — 高品質・中速
  qwen-3-235b-a22b-instruct-2507 — 大規模MoE
"""

import os
from typing import List, Dict, Optional, Any

from utils.logger import get_logger

logger = get_logger(__name__)


class CerebrasClient:
    """Cerebras Cloud API クライアント (AsyncCerebras ラッパー)"""

    DEFAULT_MODEL = "llama3.1-8b"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY", "")
        self.model = model
        self._client: Optional[Any] = None

    def _get_client(self):
        if self._client is None:
            from cerebras.cloud.sdk import AsyncCerebras
            self._client = AsyncCerebras(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        **_kw,
    ) -> str:
        """テキスト生成。system プロンプトを messages 先頭に挿入して送信。"""
        if not self.api_key:
            raise RuntimeError("CEREBRAS_API_KEY が未設定です")

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            logger.debug("⚡ Cerebras (%s): %d chars", self.model, len(text))
            return text
        except Exception as e:
            logger.warning("Cerebras generate 失敗: %s", e)
            raise

    async def health_check(self) -> bool:
        try:
            result = await self.generate(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return bool(result)
        except Exception:
            return False
