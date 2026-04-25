"""
utils/base_client.py — LLMクライアント統一インターフェース v18

全クライアントが実装する共通インターフェース。
ClientRegistry がこのインターフェースを通じてクライアントを管理する。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

# レート制限・一時障害とみなすHTTPステータスコード
_RETRYABLE_STATUS = frozenset([429, 500, 502, 503, 504])


class BaseLLMClient(ABC):
    """LLMクライアント統一インターフェース"""

    @abstractmethod
    async def generate(
        self,
        messages: List[dict],
        system: str = "",
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """
        メッセージリストからテキストを生成する。

        Args:
            messages:   [{"role": "user"|"assistant", "content": "..."}]
            system:     システムプロンプト
            max_tokens: 最大出力トークン数
            **kwargs:   モデル固有パラメータ

        Returns:
            生成テキスト
        """

    async def generate_with_retry(
        self,
        messages: List[dict],
        system: str = "",
        max_tokens: int = 4096,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        **kwargs,
    ) -> str:
        """
        指数バックオフ付きリトライで generate を呼び出す。

        レート制限 (429) や一時的なサーバーエラー (500/502/503/504) に対して
        最大 max_attempts 回リトライする。最終的に失敗した場合は例外を再送出。
        """
        last_exc: Exception = RuntimeError("generate_with_retry: no attempts made")
        for attempt in range(max_attempts):
            try:
                return await self.generate(
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as e:
                last_exc = e
                status = getattr(e, "status_code", None) or getattr(e, "status", None)
                # レート制限・一時障害のみリトライ
                if status and int(status) not in _RETRYABLE_STATUS:
                    raise
                if attempt + 1 < max_attempts:
                    delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s ...
                    logger.warning(
                        "⚠️ %s generate 失敗 (attempt %d/%d, status=%s): %s — %.1fs後リトライ",
                        self.__class__.__name__, attempt + 1, max_attempts, status, e, delay,
                    )
                    await asyncio.sleep(delay)
        raise last_exc

    async def health_check(self) -> bool:
        """クライアントの死活チェック。デフォルトは True。"""
        return True
