"""
Bushidan Multi-Agent System v11.4 - xAI Grok Client

参謀-B (Sanbo-B) - 実装特化・高速エンジン
xAI Grok-code-fast-1 モデルのクライアント。
コーディング特化・240tok/s 超高速実装に特化した参謀。

Model: xAI Grok-code-fast-1
- xAI (Elon Musk) 製コーディング特化モデル
- 約240 tok/s の超高速推論
- OpenAI 互換 API (api.x.ai/v1)

Role: 参謀-B (Sanbo-B) - Chief Staff Officer B
- 実装・バグ修正・高速コーディング
- 軍師 PDCA の Do フェーズ主力実行者
- 独立サブタスクの並列実装

Usage Context (運用黄金律):
- Medium タスクの第1実装者
- PDCA Do フェーズ：独立サブタスクの並列実行
- バグ修正・パッチ適用の高速処理
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
class GrokUsageStats:
    """Grok API usage statistics / Grok API使用統計"""
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


class GrokClient:
    """
    xAI Grok-code-fast-1 Client - 参謀-B (Sanbo-B) 実装特化エンジン

    Model: grok-code-fast-1
    - xAI (Elon Musk) 製コーディング特化最速モデル
    - 約240 tok/s の超高速推論
    - OpenAI 互換 API

    Rate Limits:
    - xAI API の制限に依存
    - 自動レートリミット管理を実装
    """

    VERSION = "11.4"
    DEFAULT_MODEL = "grok-code-fast-1"
    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL
    ):
        """
        Grok クライアントを初期化

        Args:
            api_key: xAI APIキー (未指定時は環境変数 XAI_API_KEY から取得)
            model: モデル名 (default: grok-code-fast-1)
            base_url: API base URL (default: https://api.x.ai/v1)
        """
        self.api_key = api_key or os.environ.get("XAI_API_KEY", "")
        self.model = model
        self.base_url = base_url

        # Statistics / 統計
        self.stats = GrokUsageStats()

        # Rate limiting / レートリミット管理
        self.request_timestamps: List[datetime] = []
        self.rate_limit_window = timedelta(minutes=1)
        self.max_requests_per_minute = 60

        # Configuration / 設定
        self.default_max_tokens = 8192
        self.default_temperature = 0.2  # 実装特化：低温度で確実なコード生成

        # Cost estimation (approximate yen per 1K tokens)
        # grok-code-fast-1 は grok-2 より安価な見込み
        self._cost_per_1k_input_tokens_yen = 2.2    # ~$0.015
        self._cost_per_1k_output_tokens_yen = 4.5   # ~$0.030

        if not self.api_key:
            logger.warning("XAI_API_KEY が設定されていません")

        logger.info(
            f"参謀-B Grok client initialized (v{self.VERSION})\n"
            f"   Model: {self.model}\n"
            f"   Base URL: {self.base_url}\n"
            f"   Expected speed: ~240 tok/s"
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Grok API を使用してテキスト生成を実行

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
                f"参謀-B Grok生成完了: {completion_tokens} tokens / {elapsed_time:.2f}s "
                f"({tok_per_sec:.1f} tok/s)"
            )

            return response_text

        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1

            if "rate" in str(e).lower() or "429" in str(e):
                self.stats.rate_limited_count += 1
                logger.warning(f"Grok レートリミット: {e}")
            else:
                logger.error(f"Grok 生成失敗: {e}")

            raise

    async def implement_code(
        self,
        task_description: str,
        context: Optional[str] = None,
        language: str = "python"
    ) -> str:
        """
        コード実装に特化した高速生成

        参謀-B の主要メソッド。実装タスクに最適化されたプロンプトで呼び出す。

        Args:
            task_description: 実装するタスクの説明
            context: 既存コードや関連コンテキスト
            language: プログラミング言語

        Returns:
            実装されたコード
        """
        system_prompt = (
            f"あなたは高速・高精度な{language}実装エンジンです。"
            "要求された機能を正確に、バグなく実装してください。"
            "コードのみを出力し、説明は最小限にしてください。"
        )

        messages = [{"role": "system", "content": system_prompt}]

        user_content = task_description
        if context:
            user_content = f"コンテキスト:\n```\n{context}\n```\n\n実装要求:\n{task_description}"

        messages.append({"role": "user", "content": user_content})

        return await self.generate(messages, temperature=0.1)

    async def fix_bug(
        self,
        code: str,
        error_description: str,
        error_output: Optional[str] = None
    ) -> str:
        """
        バグ修正に特化した高速処理

        Args:
            code: バグのあるコード
            error_description: エラーの説明
            error_output: エラー出力（スタックトレース等）

        Returns:
            修正されたコード
        """
        content = f"以下のコードにバグがあります。修正してください。\n\nコード:\n```\n{code}\n```\n\nエラー: {error_description}"
        if error_output:
            content += f"\n\nエラー出力:\n```\n{error_output}\n```"

        messages = [
            {
                "role": "system",
                "content": "バグ修正の専門家です。コードのバグを特定し、修正したコードを出力してください。"
            },
            {"role": "user", "content": content}
        ]

        return await self.generate(messages, temperature=0.1)

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        xAI API へのHTTPリクエストを実行

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
                        f"Grok API error {response.status_code}: {error_detail}"
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
            logger.error(f"Grok API リクエスト失敗: {e}")
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
                logger.warning(f"Grok レートリミット到達、{wait_time:.1f}秒待機中")
                await asyncio.sleep(wait_time + 0.1)

        # Record this request / リクエスト記録
        self.request_timestamps.append(now)

    async def health_check(self) -> bool:
        """
        Grok API の健全性を確認

        Returns:
            True: 正常, False: 異常
        """
        try:
            test_messages = [
                {"role": "user", "content": "print('hello')"}
            ]
            await self.generate(test_messages, max_tokens=20)
            logger.info("Grok 健全性確認: OK")
            return True

        except Exception as e:
            logger.warning(f"Grok 健全性確認失敗: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Grok 使用統計を取得

        Returns:
            使用メトリクスの辞書
        """
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests

        return {
            "client": "grok-code-fast-1",
            "role": "参謀-B (Sanbo-B)",
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
        self.stats = GrokUsageStats()
        logger.info("Grok 統計リセット完了")
