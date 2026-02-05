"""
Bushidan Multi-Agent System v10.1 - Kimi K2.5 Client (傭兵: Mercenary)

Moonshot AI Kimi K2.5 - 128K context, multimodal, agent-capable.
OpenAI-compatible API at api.moonshot.cn/v1 or via OpenRouter.

Role in v10.1:
- PDCA Do フェーズの第1実行者 (128K context, 本当の並列実行)
- マルチモーダル: 検校(Kengyo) と連携した Check フェーズ視覚検証
- 傭兵 (Yohei): クラウドの物量で並列サブタスク実行

Fallback chain position:
  Kimi K2.5 (128K, cloud) → Local Qwen3 (4K, local) → Kagemusha (32K, cloud) → Gemini 3 Flash

Cost: Input ~¥0.002/1K tokens, Output ~¥0.008/1K tokens (Moonshot pricing)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("bushidan.kimi_k2")


@dataclass
class KimiConfig:
    """Kimi K2.5 client configuration"""
    api_key: str
    provider: str = "moonshot"          # moonshot, openrouter
    model: str = "kimi-k2-0711"         # Moonshot model ID
    max_context: int = 131072           # 128K context window
    default_max_tokens: int = 8192      # Default output limit
    timeout: float = 120.0              # Request timeout (seconds)
    max_retries: int = 2                # Retry count on failure
    retry_delay: float = 2.0            # Base retry delay (seconds)

    @property
    def base_url(self) -> str:
        if self.provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        return "https://api.moonshot.cn/v1"

    @property
    def model_id(self) -> str:
        if self.provider == "openrouter":
            return "moonshot/kimi-k2.5"
        return self.model


class KimiK2Client:
    """
    Kimi K2.5 API クライアント - 傭兵 (Mercenary)

    特性:
    - 128K context: 軍師(256K) と 大将(4K) のギャップを埋める
    - クラウド推論: asyncio.gather で本当の並列サブタスク実行が可能
    - マルチモーダル: 画像入力に対応 (検校 Kengyo ビジュアル検証)
    - OpenAI互換API: 既存の統合パターンと親和性が高い
    """

    def __init__(self, config: KimiConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_time_seconds": 0.0,
        }

    async def initialize(self) -> None:
        """HTTPクライアント初期化"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/bushidan"
            headers["X-Title"] = "Bushidan Multi-Agent System"

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.config.timeout),
        )
        logger.info(
            "✅ Kimi K2.5 client initialized "
            f"(provider={self.config.provider}, model={self.config.model_id})"
        )

    async def close(self) -> None:
        """クライアントクローズ"""
        if self._client:
            await self._client.aclose()
            self._client = None

    # ==================== Core API ====================

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        チャット補完リクエスト

        Args:
            messages: チャットメッセージ配列
            max_tokens: 最大出力トークン数
            temperature: 温度パラメータ (Plan=0.3, Check=0.1, Do=0.7)
            system_prompt: システムプロンプト (先頭に追加)
            tools: ツール定義 (function calling)

        Returns:
            生成されたテキスト
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        # Build messages with optional system prompt
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": self.config.model_id,
            "messages": full_messages,
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        # Retry loop
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.monotonic()
                response = await self._client.post(
                    "/chat/completions", json=payload
                )
                elapsed = time.monotonic() - start_time

                if response.status_code == 429:
                    # Rate limited - backoff and retry
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"⚠️ Kimi rate limited, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                data = response.json()

                # Extract response text
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""

                # Update stats
                usage = data.get("usage", {})
                self._stats["total_requests"] += 1
                self._stats["successful"] += 1
                self._stats["total_input_tokens"] += usage.get(
                    "prompt_tokens", 0
                )
                self._stats["total_output_tokens"] += usage.get(
                    "completion_tokens", 0
                )
                self._stats["total_time_seconds"] += elapsed

                logger.debug(
                    f"Kimi response: {len(text)} chars, "
                    f"{usage.get('total_tokens', 0)} tokens, "
                    f"{elapsed:.1f}s"
                )
                return text

            except httpx.HTTPStatusError as e:
                last_error = e
                self._stats["failed"] += 1
                if e.response.status_code >= 500:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"⚠️ Kimi server error {e.response.status_code}, "
                        f"retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                self._stats["failed"] += 1
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"⚠️ Kimi connection error, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        raise RuntimeError(
            f"Kimi K2.5: all {self.config.max_retries + 1} attempts failed: "
            f"{last_error}"
        )

    # ==================== Specialized Methods ====================

    async def implement_subtask(
        self,
        description: str,
        context: str = "",
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> str:
        """
        サブタスク実装 (PDCA Do フェーズ向け)

        128K context を活用し、大将の 4K 制限では不可能な
        大規模コンテキストを含む実装タスクを処理する。

        Args:
            description: サブタスク説明
            context: 関連コンテキスト (ファイル内容等)
            max_tokens: 最大出力トークン数
            temperature: 温度 (実装向けは 0.7 推奨)

        Returns:
            実装結果テキスト
        """
        prompt = description
        if context:
            prompt = f"## コンテキスト\n{context}\n\n## タスク\n{description}"

        messages = [{"role": "user", "content": prompt}]
        return await self.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=(
                "あなたは武士団マルチエージェントシステムの傭兵（Kimi K2.5）です。"
                "与えられたサブタスクを正確に実装してください。"
                "コードを生成する場合は、完全で動作するコードを出力してください。"
            ),
        )

    async def implement_subtasks_parallel(
        self,
        subtasks: List[Dict[str, str]],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        max_concurrency: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        複数サブタスクの並列実行 (Kimi の真価)

        llama.cpp はシングルスレッドで asyncio.gather が実質直列になるが、
        Kimi API は複数リクエストを真に並列処理できる。

        Args:
            subtasks: [{"id": str, "description": str, "context": str}, ...]
            max_tokens: 各サブタスクの最大出力
            temperature: 温度
            max_concurrency: 最大並列数 (API レート制限考慮)

        Returns:
            [{"id": str, "result": str, "status": str, "time": float}, ...]
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_one(task: Dict[str, str]) -> Dict[str, Any]:
            async with semaphore:
                start = time.monotonic()
                try:
                    result = await self.implement_subtask(
                        description=task["description"],
                        context=task.get("context", ""),
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return {
                        "id": task["id"],
                        "result": result,
                        "status": "completed",
                        "time": time.monotonic() - start,
                    }
                except Exception as e:
                    logger.error(f"❌ Kimi subtask {task['id']} failed: {e}")
                    return {
                        "id": task["id"],
                        "result": str(e),
                        "status": "failed",
                        "time": time.monotonic() - start,
                    }

        results = await asyncio.gather(
            *[_run_one(t) for t in subtasks], return_exceptions=False
        )
        return list(results)

    async def visual_check(
        self,
        image_url: str,
        check_prompt: str,
    ) -> str:
        """
        ビジュアル検証 (検校 Kengyo から呼び出される)

        Playwright MCP でキャプチャしたスクリーンショットを
        Kimi K2.5 のマルチモーダル能力で視覚的に検証する。

        Args:
            image_url: 画像URL or base64 data URI
            check_prompt: 検証指示

        Returns:
            検証結果テキスト
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": check_prompt},
            ],
        }]
        return await self.generate(
            messages,
            temperature=0.1,
            system_prompt=(
                "あなたはUI/UX検証の専門家です。"
                "スクリーンショットを分析し、問題点を具体的に指摘してください。"
            ),
        )

    # ==================== Statistics ====================

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        stats = dict(self._stats)
        if stats["successful"] > 0:
            stats["avg_time_seconds"] = (
                stats["total_time_seconds"] / stats["successful"]
            )
        else:
            stats["avg_time_seconds"] = 0.0

        # Cost estimate (Moonshot pricing)
        input_cost = stats["total_input_tokens"] * 0.002 / 1000
        output_cost = stats["total_output_tokens"] * 0.008 / 1000
        stats["estimated_cost_jpy"] = round(input_cost + output_cost, 4)

        return stats
