"""
Bushidan Multi-Agent System v11.5 - ELYZA オンデマンドクライアント

隠密サポート（日本語特化）- ELYZA GGUF モデル via llama.cpp

対応モデル:
  - Llama-3-ELYZA-JP-8B  (推奨: ~5GB Q4, Nemotronと共存可能)
  - ELYZA-japanese-Llama-2-7b-instruct (~4.5GB Q4)

運用方針:
  - Nemotron (port 8080) が常駐
  - ELYZA (port 8081) がオンデマンド起動
  - 日本語優先タスク検出時に LangGraph が呼び出す
  - Nemotron と同ホストの別ポートで動作
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ELYZA モデル識別子
ELYZA_LLAMA3 = "elyza-llama3-8b"
ELYZA_LLAMA2 = "elyza-llama2-7b"


@dataclass
class ElyzaConfig:
    """ELYZA llama.cpp サーバー設定"""
    host: str = field(default_factory=lambda: os.getenv(
        "ELYZA_HOST",
        os.getenv("LOCAL_LLM_HOST", "192.168.11.239")
    ))
    port: int = 8081
    model_name: str = ELYZA_LLAMA3    # 現在ロードされているモデルの識別名
    context_size: int = 4096
    timeout_seconds: float = 180.0    # 日本語生成は若干時間かかる

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


class ElyzaClient:
    """
    ELYZA 日本語特化モデル クライアント

    llama.cpp サーバー (OpenAI 互換 API) 経由で ELYZA モデルを呼び出す。
    Nemotron と同じ API 形式なので同じ呼び出し方が使える。

    日本語優先タスク向け:
      - 自然な日本語テキスト生成
      - 日本語翻訳・添削・校正
      - 日本語創作（小説・詩・俳句）
      - 日本語文法チェック
    """

    VERSION = "11.5"

    def __init__(self, config: Optional[ElyzaConfig] = None):
        self.config = config or ElyzaConfig()
        self._server_available: Optional[bool] = None
        self._detected_model: Optional[str] = None

        logger.info(
            "🎌 ELYZA Client 初期化 (v%s)\n"
            "   Endpoint: %s\n"
            "   Model: %s",
            self.VERSION, self.config.endpoint, self.config.model_name
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        ELYZA モデルでテキスト生成。

        日本語生成に最適化された温度設定 (0.7) をデフォルトにしている。
        (Nemotron の 0.3 より高め → 自然な日本語表現のため)
        """
        try:
            import httpx

            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            timeout = httpx.Timeout(
                connect=10.0,
                read=self.config.timeout_seconds,
                write=10.0,
                pool=10.0,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.config.endpoint}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                if response.status_code != 200:
                    raise Exception(
                        f"ELYZA API error {response.status_code}: {response.text}"
                    )
                result = response.json()
                text = result["choices"][0]["message"].get("content", "")
                tokens = result.get("usage", {}).get("completion_tokens", 0)
                logger.info("🎌 ELYZA 生成完了: %d tokens", tokens)
                return text

        except Exception as e:
            logger.error("🎌 ELYZA 生成失敗: %s", e)
            raise

    async def generate_japanese(self, task: str, context: str = "") -> str:
        """日本語タスク専用エントリーポイント。システムプロンプトを日本語に最適化。"""
        system = (
            "あなたは高品質な日本語テキスト生成に特化したAIアシスタントです。"
            "自然で流暢な日本語を使い、文法・表現・文体に細心の注意を払ってください。"
        )
        if context:
            system += f"\n\n【参考情報】\n{context[:300]}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        return await self.generate(messages)

    async def is_available(self) -> bool:
        """llama.cpp サーバーの可用性を確認。"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.config.endpoint}/health")
                available = response.status_code == 200
                self._server_available = available
                if available:
                    logger.info("🎌 ELYZA サーバー: 利用可能 ✅ (%s)", self.config.endpoint)
                else:
                    logger.debug("🎌 ELYZA サーバー: 応答異常 (%d)", response.status_code)
                return available
        except Exception:
            self._server_available = False
            logger.debug("🎌 ELYZA サーバー: 未起動 / 接続不可 (%s)", self.config.endpoint)
            return False

    async def detect_loaded_model(self) -> Optional[str]:
        """
        起動中の llama.cpp サーバーからロード済みモデル名を取得。
        モデル情報が取れない場合は None。
        """
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.config.endpoint}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    if models:
                        self._detected_model = models[0].get("id", "unknown")
                        return self._detected_model
        except Exception:
            pass
        return None
