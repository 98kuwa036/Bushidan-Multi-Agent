"""
Bushidan Multi-Agent System v11.4 - OpenAI GPT-5 Client

参謀-A (Sanbo-A) - 高度汎用推論エンジン
OpenAI GPT-5モデルのクライアント。
広範な知識と高度な推論能力を兼ね備えた汎用参謀。

Model: OpenAI GPT-5
- 最新世代の汎用大規模言語モデル
- 広範な知識と高度な推論能力
- OpenAI API経由でのクラウド推論
- Cost: API usage based (per-token pricing)

Role: 参謀-A (Sanbo-A) - Chief Staff Officer A
- 戦略立案の補佐・実行計画の策定
- 多角的な分析と提案を担当
- 軍師 (o3-mini) と連携した意思決定支援

Usage Context (運用黄金律):
- 複雑なタスクの分析・計画策定
- コードレビュー・設計判断
- 軍師の推論結果を実行可能な計画に落とし込む
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class GPT5UsageStats:
    """GPT-5 API usage statistics / GPT-5 API使用統計"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_count: int = 0
    total_tokens: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    total_inference_time_seconds: float = 0.0
    average_tokens_per_second: float = 0.0
    estimated_cost_yen: float = 0.0


class GPT5Client:
    """
    OpenAI GPT-5 Client - 参謀-A (Sanbo-A) 高度汎用推論エンジン

    Model: GPT-5
    - 最新世代のOpenAI旗艦モデル
    - 広範な知識ベースと高精度推論
    - 大規模コンテキストウィンドウ対応

    Rate Limits:
    - Tier依存 (組織のAPIプランによる)
    - 自動レートリミット管理を実装
    """

    VERSION = "11.4"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        base_url: str = "https://api.openai.com/v1"
    ):
        """
        GPT-5クライアントを初期化

        Args:
            api_key: OpenAI APIキー (未指定時は環境変数 OPENAI_API_KEY から取得)
            model: モデル名 (default: gpt-5)
            base_url: API base URL
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url

        # Statistics / 統計
        self.stats = GPT5UsageStats()

        # Rate limiting / レートリミット管理
        self.request_timestamps: List[datetime] = []
        self.rate_limit_window = timedelta(minutes=1)
        self.max_requests_per_minute = 60

        # Configuration / 設定
        self.default_max_tokens = 4096
        self.default_temperature = 0.7

        # Cost estimation (approximate yen per 1K tokens)
        self._cost_per_1k_input_tokens_yen = 7.5    # ~$0.05
        self._cost_per_1k_output_tokens_yen = 22.5   # ~$0.15

        if not self.api_key:
            logger.warning("OPENAI_API_KEY が設定されていません")

        logger.info(
            f"参謀-A GPT-5 client initialized (v{self.VERSION})\n"
            f"   Model: {self.model}\n"
            f"   Base URL: {self.base_url}"
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        GPT-5 APIを使用してテキスト生成を実行

        Args:
            messages: メッセージリスト [{"role": "user", "content": "..."}]
            max_tokens: 生成する最大トークン数
            temperature: サンプリング温度 (0.0-2.0)
            stream: ストリーミング応答 (未実装)

        Returns:
            生成されたテキスト応答

        Raises:
            Exception: API呼び出しが失敗した場合
        """
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if temperature is None:
            temperature = self.default_temperature

        # Rate limit check / レートリミット確認
        await self._rate_limit_check()

        start_time = asyncio.get_event_loop().time()

        try:
            response_data = await self._make_request(
                messages, max_tokens, temperature
            )

            response_text = response_data["content"]

            # Update statistics / 統計更新
            elapsed_time = asyncio.get_event_loop().time() - start_time
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.total_inference_time_seconds += elapsed_time

            # Token tracking
            prompt_tokens = response_data.get("prompt_tokens", 0)
            completion_tokens = response_data.get("completion_tokens", 0)

            self.stats.prompt_tokens_total += prompt_tokens
            self.stats.completion_tokens_total += completion_tokens
            self.stats.total_tokens += prompt_tokens + completion_tokens

            # Cost estimation / コスト見積
            input_cost = (prompt_tokens / 1000) * self._cost_per_1k_input_tokens_yen
            output_cost = (completion_tokens / 1000) * self._cost_per_1k_output_tokens_yen
            self.stats.estimated_cost_yen += input_cost + output_cost

            # Tokens per second / トークン毎秒
            if elapsed_time > 0:
                tok_per_sec = completion_tokens / elapsed_time if completion_tokens > 0 else len(response_text.split()) / elapsed_time
                n = self.stats.successful_requests
                self.stats.average_tokens_per_second = (
                    (self.stats.average_tokens_per_second * (n - 1) + tok_per_sec) / n
                )
            else:
                tok_per_sec = 0

            logger.info(
                f"参謀-A GPT-5生成完了: {completion_tokens} tokens / {elapsed_time:.2f}s "
                f"({tok_per_sec:.1f} tok/s)"
            )

            return response_text

        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1

            if "rate" in str(e).lower() or "429" in str(e):
                self.stats.rate_limited_count += 1
                logger.warning(f"GPT-5 レートリミット: {e}")
            else:
                logger.error(f"GPT-5 生成失敗: {e}")

            raise

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        OpenAI APIへのHTTPリクエストを実行

        Args:
            messages: 会話メッセージ
            max_tokens: 最大生成トークン数
            temperature: サンプリング温度

        Returns:
            レスポンス辞書 (content, token counts)
        """
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code != 200:
                    error_detail = response.text
                    raise Exception(
                        f"GPT-5 API error {response.status_code}: {error_detail}"
                    )

                result = response.json()
                choice = result["choices"][0]["message"]
                usage = result.get("usage", {})

                return {
                    "content": choice["content"],
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0)
                }

        except Exception as e:
            logger.error(f"GPT-5 APIリクエスト失敗: {e}")
            raise

    async def _rate_limit_check(self) -> None:
        """
        レートリミットの確認と制御

        スライディングウィンドウ方式でリクエスト頻度を管理
        """
        now = datetime.now()

        # Remove timestamps outside the window / ウィンドウ外のタイムスタンプを除去
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if now - ts < self.rate_limit_window
        ]

        # Check if at limit / リミット到達確認
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            oldest = self.request_timestamps[0]
            wait_time = (oldest + self.rate_limit_window - now).total_seconds()

            if wait_time > 0:
                logger.warning(f"GPT-5 レートリミット到達、{wait_time:.1f}秒待機中")
                await asyncio.sleep(wait_time + 0.1)

        # Record this request / リクエスト記録
        self.request_timestamps.append(now)

    async def health_check(self) -> bool:
        """
        GPT-5 APIの健全性を確認

        Returns:
            True: 正常, False: 異常
        """
        try:
            test_messages = [
                {"role": "user", "content": "Hi"}
            ]
            await self.generate(test_messages, max_tokens=10)
            logger.info("GPT-5 健全性確認: OK")
            return True

        except Exception as e:
            logger.warning(f"GPT-5 健全性確認失敗: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        GPT-5使用統計を取得

        Returns:
            使用メトリクスの辞書
        """
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests

        return {
            "client": "gpt-5",
            "role": "参謀-A (Sanbo-A)",
            "version": self.VERSION,
            "model": self.model,
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": round(success_rate, 3),
            "rate_limited_count": self.stats.rate_limited_count,
            "total_tokens": self.stats.total_tokens,
            "prompt_tokens_total": self.stats.prompt_tokens_total,
            "completion_tokens_total": self.stats.completion_tokens_total,
            "average_tokens_per_second": round(self.stats.average_tokens_per_second, 1),
            "total_inference_time_seconds": round(self.stats.total_inference_time_seconds, 1),
            "estimated_cost_yen": round(self.stats.estimated_cost_yen, 2)
        }

    def reset_statistics(self) -> None:
        """使用統計をリセット"""
        self.stats = GPT5UsageStats()
        logger.info("GPT-5 統計リセット完了")
