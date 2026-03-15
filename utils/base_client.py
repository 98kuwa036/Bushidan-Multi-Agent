"""
utils/base_client.py — LLMクライアント統一インターフェース v14

全クライアントが実装する共通インターフェース。
ClientRegistry がこのインターフェースを通じてクライアントを管理する。
"""

from abc import ABC, abstractmethod
from typing import List, Optional


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

    async def health_check(self) -> bool:
        """クライアントの死活チェック。デフォルトは True。"""
        return True
