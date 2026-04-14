"""
utils/mistral_client.py — Mistral API クライアント (参謀用)

Mistral Large 3 を非同期で呼び出す。
環境変数: MISTRAL_API_KEY
"""

import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger("utils.mistral_client")


class MistralClient:
    """Mistral API 非同期クライアント"""

    BASE_URL = "https://api.mistral.ai/v1/chat/completions"
    DEFAULT_MODEL = "mistral-large-latest"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def generate(
        self,
        messages: List[dict],
        model: str = DEFAULT_MODEL,
        system: str = "",
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
        """
        Mistral チャット API を呼び出してレスポンステキストを返す。

        Args:
            messages:    conversation messages (role/content)
            model:       使用モデル
            system:      システムプロンプト (先頭に挿入)
            max_tokens:  最大トークン数
            temperature: 温度

        Returns:
            生成されたテキスト
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("pip install httpx が必要です")

        # システムプロンプトを先頭に挿入
        if system:
            full_messages = [{"role": "system", "content": system}] + [
                m for m in messages if m.get("role") != "system"
            ]
        else:
            full_messages = messages

        payload = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(self.BASE_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"Mistral API 応答パースエラー: {e} — data={str(data)[:200]}")
