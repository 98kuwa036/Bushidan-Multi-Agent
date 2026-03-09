"""
Bushidan Multi-Agent System v11.4 - Nemotron-3-Nano llama.cpp Client

隠密 (Onmitsu) - 機密処理・超長文・ローカル推論エンジン
NVIDIA Nemotron-3-Nano-30B-A3B をローカル llama.cpp で動かすクライアント。

Model: Nemotron-3-Nano-30B-A3B (Q4_K_M ~21GB)
- NVIDIA製・脱中国・信頼性の高いローカルLLM
- 30B パラメータ・MoE アーキテクチャ（有効 3B 相当）
- HP ProDesk 600 (i5-8500, 32GB DDR4) で CPU 推論
- llama.cpp サーバー経由（OpenAI 互換 API）

Role: 隠密 (Onmitsu) - 機密情報処理・オフライン保証
- 秘匿コード・機密情報を API に送信せずローカル処理
- ネットワーク障害時のオフライン保証
- ¥0 運用（電気代のみ）
- 15-25 tok/s (CPU, i5-8500)

Usage Context (運用黄金律):
- confidential_data: API送信不可の機密情報
- sensitive_code: 秘匿すべきコード処理
- offline_required: ネットワーク障害時
- long_context_local: ローカルで長文処理
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class LlamaCppConfig:
    """llama.cpp サーバー設定"""
    model_path: str = "models/nemotron/Nemotron-3-Nano-Q4_K_M.gguf"
    host: str = "192.168.11.239"     # ローカルLLMサーバー
    port: int = 8080
    threads: int = 6                  # i5-8500: 6C/6T
    context_size: int = 8192
    batch_size: int = 512
    mlock: bool = True                # メモリロック（スワップ防止）
    mmap: bool = True
    gpu_layers: int = 0               # CPU のみ


@dataclass
class NemotronUsageStats:
    """Nemotron-3-Nano 使用統計"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    total_inference_time_seconds: float = 0.0
    average_tokens_per_second: float = 0.0


class NemotronLlamaCppClient:
    """
    NVIDIA Nemotron-3-Nano llama.cpp Client
    隠密 (Onmitsu) - 機密・超長文・ローカル処理エンジン

    llama.cpp サーバー (http://192.168.11.239:8080) と通信し、
    OpenAI 互換 API 経由で Nemotron-3-Nano を呼び出す。

    特徴:
    - ローカル完結: API 送信なし、機密情報安全
    - NVIDIA 製: 脱中国・信頼性の高いモデル
    - CPU 最適化: AVX2, FMA 命令セット活用
    - ¥0 運用: 電気代のみ（~¥3/日）
    """

    VERSION = "11.4"

    def __init__(
        self,
        config: Optional[LlamaCppConfig] = None,
        endpoint: Optional[str] = None
    ):
        """
        Nemotron-3-Nano llama.cpp クライアントを初期化

        Args:
            config: llama.cpp 設定 (LlamaCppConfig)
            endpoint: 直接エンドポイント指定 (e.g. "http://192.168.11.239:8080")
        """
        self.config = config or LlamaCppConfig()

        # エンドポイント設定
        if endpoint:
            self.base_url = endpoint.rstrip("/")
        else:
            self.base_url = f"http://{self.config.host}:{self.config.port}"

        # Statistics / 統計
        self.stats = NemotronUsageStats()

        # 設定
        self.default_max_tokens = 2048
        self.default_temperature = 0.3
        self._server_available = None  # None = 未確認

        logger.info(
            f"隠密 Nemotron-3-Nano client initialized (v{self.VERSION})\n"
            f"   Endpoint: {self.base_url}\n"
            f"   Model: {self.config.model_path}\n"
            f"   Threads: {self.config.threads}\n"
            f"   Context: {self.config.context_size}\n"
            f"   Expected speed: 15-25 tok/s (CPU)"
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Nemotron-3-Nano でテキスト生成（ローカル完結）

        Args:
            messages: メッセージリスト [{"role": "user", "content": "..."}]
            max_tokens: 最大生成トークン数
            temperature: サンプリング温度

        Returns:
            生成されたテキスト応答

        Raises:
            Exception: llama.cpp サーバーが利用不可の場合
        """
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if temperature is None:
            temperature = self.default_temperature

        start_time = asyncio.get_event_loop().time()

        try:
            response_data = await self._make_request(messages, max_tokens, temperature)
            response_text = response_data["content"]

            # 統計更新
            elapsed_time = asyncio.get_event_loop().time() - start_time
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.total_inference_time_seconds += elapsed_time

            prompt_tokens = response_data.get("prompt_tokens", 0)
            completion_tokens = response_data.get("completion_tokens", 0)

            self.stats.prompt_tokens_total += prompt_tokens
            self.stats.completion_tokens_total += completion_tokens
            self.stats.total_tokens += prompt_tokens + completion_tokens

            if elapsed_time > 0 and completion_tokens > 0:
                tok_per_sec = completion_tokens / elapsed_time
                n = self.stats.successful_requests
                self.stats.average_tokens_per_second = (
                    (self.stats.average_tokens_per_second * (n - 1) + tok_per_sec) / n
                )
            else:
                tok_per_sec = 0

            logger.info(
                f"🥷 隠密 Nemotron生成完了: {completion_tokens} tokens / {elapsed_time:.2f}s "
                f"({tok_per_sec:.1f} tok/s) [ローカル完結]"
            )

            return response_text

        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            logger.error(f"隠密 Nemotron生成失敗: {e}")
            raise

    async def process_confidential(
        self,
        content: str,
        task_type: str = "general"
    ) -> str:
        """
        機密情報処理専用メソッド

        ローカル完結を保証し、API への送信を行わない。

        Args:
            content: 処理する機密コンテンツ
            task_type: タスクタイプ (general, code, analysis)

        Returns:
            処理結果（ローカル生成）
        """
        logger.info(f"🥷 隠密: 機密処理開始 [タイプ: {task_type}] - API送信なし")

        system_prompt = (
            "あなたはローカル環境でのみ動作する機密情報処理エンジンです。"
            "この会話はネットワークを経由せず、ローカルマシンのみで処理されます。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        result = await self.generate(messages)
        logger.info("🥷 隠密: 機密処理完了 [ローカル完結]")
        return result

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        llama.cpp サーバーへの HTTP リクエスト

        llama.cpp は OpenAI 互換 API を提供するため、
        /v1/chat/completions エンドポイントを使用。

        Args:
            messages: 会話メッセージ
            max_tokens: 最大生成トークン数
            temperature: サンプリング温度

        Returns:
            レスポンス辞書 (content, token counts)
        """
        try:
            import httpx

            headers = {"Content-Type": "application/json"}

            payload = {
                "model": "nemotron",  # llama.cpp サーバーのモデル識別子
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            # llama.cpp は低速なため、タイムアウトを長めに設定
            timeout = httpx.Timeout(
                connect=10.0,
                read=300.0,   # 5分：長文生成に対応
                write=10.0,
                pool=10.0
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code != 200:
                    error_detail = response.text
                    raise Exception(
                        f"llama.cpp API error {response.status_code}: {error_detail}"
                    )

                result = response.json()
                choice = result["choices"][0]["message"]
                usage = result.get("usage", {})

                return {
                    "content": choice.get("content", ""),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                }

        except Exception as e:
            logger.error(f"llama.cpp API リクエスト失敗: {e}")
            raise

    async def is_available(self) -> bool:
        """
        llama.cpp サーバーの可用性確認

        Returns:
            True: サーバー起動中, False: 利用不可
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                available = response.status_code == 200
                self._server_available = available

                if available:
                    logger.info("🥷 隠密 llama.cpp サーバー: 利用可能 ✅")
                else:
                    logger.warning("🥷 隠密 llama.cpp サーバー: 応答異常 ⚠️")

                return available

        except Exception as e:
            self._server_available = False
            logger.warning(f"🥷 隠密 llama.cpp サーバー: 接続不可 ({e})")
            return False

    async def health_check(self) -> bool:
        """健全性確認（is_available のエイリアス）"""
        return await self.is_available()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Nemotron-3-Nano 使用統計を取得

        Returns:
            使用メトリクスの辞書
        """
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests

        return {
            "client": "nemotron-3-nano",
            "role": "隠密 (Onmitsu)",
            "version": self.VERSION,
            "endpoint": self.base_url,
            "model_path": self.config.model_path,
            "server_available": self._server_available,
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": round(success_rate, 3),
            "total_tokens": self.stats.total_tokens,
            "prompt_tokens_total": self.stats.prompt_tokens_total,
            "completion_tokens_total": self.stats.completion_tokens_total,
            "average_tokens_per_second": round(self.stats.average_tokens_per_second, 1),
            "total_inference_time_seconds": round(self.stats.total_inference_time_seconds, 1),
            "cost_yen": 0,  # ローカル運用は無料（電気代除く）
            "hardware": "HP ProDesk 600 G4 (i5-8500, 32GB DDR4, CPU only)"
        }

    def reset_statistics(self) -> None:
        """使用統計をリセット"""
        self.stats = NemotronUsageStats()
        logger.info("隠密 Nemotron 統計リセット完了")
