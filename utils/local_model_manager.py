"""
Bushidan Multi-Agent System v11.5 - ローカルモデルマネージャー

EliteDesk (32GB RAM) でのオンデマンドモデル管理。

構成:
  port 8080 - Nemotron-3-Nano-30B (常駐, ~22GB, 機密/オフライン)
  port 8081 - ELYZA モデル (オンデマンド, ~5GB, 日本語特化)

LangGraph からの呼び出しフロー:
  1. _detect_japanese_priority(task) → True
  2. local_manager.get_japanese_client() → ElyzaClient or NemotronClient (フォールバック)
  3. 呼び出し元がレスポンス取得
"""

import asyncio
import logging
from typing import Optional

from utils.elyza_client import ElyzaClient, ElyzaConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class LocalModelManager:
    """
    ローカルLLMのルーティングを管理するマネージャー。

    実際のプロセス起動/停止は PM2 や systemd が担う。
    このクラスはヘルスチェックとクライアント提供のみを行う。

    メモリ配分 (32GB):
      Nemotron  : ~22GB (常駐)
      ELYZA-8B  : ~5GB  (オンデマンド, Nemotronと共存可能)
      OS/その他 : ~5GB
    """

    def __init__(self):
        self._elyza_client: Optional[ElyzaClient] = None
        self._elyza_available: Optional[bool] = None
        self._check_lock = asyncio.Lock()

    async def get_japanese_client(self) -> Optional[ElyzaClient]:
        """
        日本語タスク用クライアントを返す。

        ELYZA が起動中なら ElyzaClient を返す。
        未起動の場合は None を返し、呼び出し元が Nemotron にフォールバックする。
        """
        async with self._check_lock:
            # キャッシュが古くなっていたら再チェック (簡易TTL: 呼び出しごと)
            elyza = self._get_or_create_elyza_client()
            available = await elyza.is_available()
            self._elyza_available = available

            if available:
                model = await elyza.detect_loaded_model()
                if model:
                    logger.info("🎌 ELYZA (%s) を使用", model)
                return elyza

            logger.info("🎌 ELYZA 未起動 → Nemotron にフォールバック")
            return None

    def _get_or_create_elyza_client(self) -> ElyzaClient:
        if self._elyza_client is None:
            self._elyza_client = ElyzaClient(ElyzaConfig())
        return self._elyza_client

    @property
    def elyza_available(self) -> Optional[bool]:
        """最後に確認した ELYZA の可用性 (None = 未確認)"""
        return self._elyza_available

    def get_status(self) -> dict:
        return {
            "nemotron": {"port": 8080, "status": "always_on"},
            "elyza": {
                "port": 8081,
                "status": "available" if self._elyza_available else (
                    "unavailable" if self._elyza_available is False else "unknown"
                ),
            },
        }


# シングルトン
_manager: Optional[LocalModelManager] = None


def get_local_model_manager() -> LocalModelManager:
    global _manager
    if _manager is None:
        _manager = LocalModelManager()
    return _manager
