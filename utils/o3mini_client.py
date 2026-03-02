"""
Bushidan Multi-Agent System v11.4 - OpenAI o3-mini Client

軍師 (Gunshi) - 戦略的推論エンジン
高度な推論能力を持つOpenAI o3-miniモデルのクライアント。
reasoning_effort="high"により、深い思考を要するタスクに最適化。

Model: OpenAI o3-mini
- 高精度推論 (reasoning_effort=high)
- 戦略立案・分析・判断に特化
- OpenAI API経由でのクラウド推論
- Cost: API usage based (per-token pricing)

Role: 軍師 (Gunshi) - Strategic Advisor
- 複雑な判断が必要な場面で活躍
- 戦略立案と深い分析を担当
- reasoning_effort="high"で最大限の推論品質を保証

Usage Context (運用黄金律):
- 戦略的判断・分析タスクに使用
- 他のモデルが判断に迷う場面でのエスカレーション先
- 高精度が求められる推論タスク
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
class O3MiniUsageStats:
    """o3-mini API usage statistics / o3-mini API使用統計"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_count: int = 0
    total_tokens: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    reasoning_tokens_total: int = 0
    total_inference_time_seconds: float = 0.0
    average_tokens_per_second: float = 0.0
    estimated_cost_yen: float = 0.0


class O3MiniClient:
    """
    OpenAI o3-mini Client - 軍師 (Gunshi) 戦略的推論エンジン

    Model: o3-mini
    - reasoning_effort="high" による深い推論
    - 複雑な戦略立案・分析に最適化
    - OpenAI Chat Completions API 準拠

    Rate Limits:
    - Tier依存 (組織のAPIプランによる)
    - 自動レートリミット管理を実装
    """

    VERSION = "11.4"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "o3-mini",
        reasoning_effort: str = "high",
        base_url: str = "https://api.openai.com/v1"
    ):
        """
        o3-miniクライアントを初期化

        Args:
            api_key: OpenAI APIキー (未指定時は環境変数 OPENAI_API_KEY から取得)
            model: モデル名 (default: o3-mini)
            reasoning_effort: 推論努力レベル ("low", "medium", "high")
            base_url: API base URL
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url

        # Statistics / 統計
        self.stats = O3MiniUsageStats()

        # Rate limiting / レートリミット管理
        self.request_timestamps: List[datetime] = []
        self.rate_limit_window = timedelta(minutes=1)
        self.max_requests_per_minute = 60

        # Configuration / 設定
        self.default_max_tokens = 4096
        self.default_temperature = 1.0  # o3-mini uses temperature=1 by default

        # Cost estimation (approximate yen per 1K tokens)
        self._cost_per_1k_input_tokens_yen = 1.65   # ~$0.011
        self._cost_per_1k_output_tokens_yen = 6.6    # ~$0.044

        if not self.api_key:
            logger.warning("OPENAI_API_KEY が設定されていません")

        logger.info(
            f"軍師 o3-mini client initialized (v{self.VERSION})\n"
            f"   Model: {self.model}\n"
            f"   Reasoning Effort: {self.reasoning_effort}\n"
            f"   Base URL: {self.base_url}"
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        o3-mini APIを使用してテキスト生成を実行

        Args:
            messages: メッセージリスト [{"role": "user", "content": "..."}]
            max_tokens: 生成する最大トークン数
            temperature: サンプリング温度 (o3-miniでは通常1.0)
            reasoning_effort: 推論努力レベル (未指定時はインスタンスデフォルト)
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
        if reasoning_effort is None:
            reasoning_effort = self.reasoning_effort

        # Rate limit check / レートリミット確認
        await self._rate_limit_check()

        start_time = asyncio.get_event_loop().time()

        try:
            response_data = await self._make_request(
                messages, max_tokens, temperature, reasoning_effort
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
            reasoning_tokens = response_data.get("reasoning_tokens", 0)

            self.stats.prompt_tokens_total += prompt_tokens
            self.stats.completion_tokens_total += completion_tokens
            self.stats.reasoning_tokens_total += reasoning_tokens
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
                f"軍師 o3-mini生成完了: {completion_tokens} tokens / {elapsed_time:.2f}s "
                f"({tok_per_sec:.1f} tok/s) [reasoning_effort={reasoning_effort}]"
            )

            return response_text

        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1

            if "rate" in str(e).lower() or "429" in str(e):
                self.stats.rate_limited_count += 1
                logger.warning(f"o3-mini レートリミット: {e}")
            else:
                logger.error(f"o3-mini 生成失敗: {e}")

            raise

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        reasoning_effort: str
    ) -> Dict[str, Any]:
        """
        OpenAI APIへのHTTPリクエストを実行

        Args:
            messages: 会話メッセージ
            max_tokens: 最大生成トークン数
            temperature: サンプリング温度
            reasoning_effort: 推論努力レベル

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
                "max_completion_tokens": max_tokens,
                "reasoning_effort": reasoning_effort
            }

            # o3-mini may not support arbitrary temperature; only include if non-default
            if temperature != 1.0:
                payload["temperature"] = temperature

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code != 200:
                    error_detail = response.text
                    raise Exception(
                        f"o3-mini API error {response.status_code}: {error_detail}"
                    )

                result = response.json()
                choice = result["choices"][0]["message"]
                usage = result.get("usage", {})

                return {
                    "content": choice["content"],
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                }

        except Exception as e:
            logger.error(f"o3-mini APIリクエスト失敗: {e}")
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
                logger.warning(f"o3-mini レートリミット到達、{wait_time:.1f}秒待機中")
                await asyncio.sleep(wait_time + 0.1)

        # Record this request / リクエスト記録
        self.request_timestamps.append(now)

    async def health_check(self) -> bool:
        """
        o3-mini APIの健全性を確認

        Returns:
            True: 正常, False: 異常
        """
        try:
            test_messages = [
                {"role": "user", "content": "Hi"}
            ]
            await self.generate(test_messages, max_tokens=10)
            logger.info("o3-mini 健全性確認: OK")
            return True

        except Exception as e:
            logger.warning(f"o3-mini 健全性確認失敗: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        o3-mini使用統計を取得

        Returns:
            使用メトリクスの辞書
        """
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests

        return {
            "client": "o3-mini",
            "role": "軍師 (Gunshi)",
            "version": self.VERSION,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": round(success_rate, 3),
            "rate_limited_count": self.stats.rate_limited_count,
            "total_tokens": self.stats.total_tokens,
            "prompt_tokens_total": self.stats.prompt_tokens_total,
            "completion_tokens_total": self.stats.completion_tokens_total,
            "reasoning_tokens_total": self.stats.reasoning_tokens_total,
            "average_tokens_per_second": round(self.stats.average_tokens_per_second, 1),
            "total_inference_time_seconds": round(self.stats.total_inference_time_seconds, 1),
            "estimated_cost_yen": round(self.stats.estimated_cost_yen, 2)
        }

    def reset_statistics(self) -> None:
        """使用統計をリセット"""
        self.stats = O3MiniUsageStats()
        logger.info("o3-mini 統計リセット完了")
