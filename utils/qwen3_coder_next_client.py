"""
Bushidan Multi-Agent System v10 - Qwen3-Coder-Next API Client (軍師クライアント)

Qwen3-Coder-Next 80B-A3B: 軍師専用クライアント
- 80B総パラメータ / 3B活性パラメータ (ultra-sparse MoE)
- 256K コンテキスト (コードベース全体を俯瞰)
- SWE-Bench Verified 70.6%
- Non-thinking mode (即断即決)

API Providers:
- Alibaba Cloud DashScope (ALIBABA_API_KEY)
- OpenRouter (OPENROUTER_API_KEY)
- OpenAI互換 API

Role: 軍師 (Gunshi) - 作戦立案・コード監査・複雑タスク分解
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class CoderNextConfig:
    """Qwen3-Coder-Next API configuration"""

    # API Provider
    provider: str = "dashscope"  # dashscope, openrouter, openai_compatible
    api_key: str = ""
    api_base: str = ""  # Auto-detected from provider

    # Model
    model_name: str = "qwen3-coder-next"

    # Generation defaults (軍師向け: 正確性重視)
    default_temperature: float = 0.3  # 低め = 正確な戦略立案
    default_top_p: float = 0.9
    default_max_tokens: int = 8192  # 長めの分析を許容
    context_window: int = 262144  # 256K

    # 軍師の特性
    role_description: str = "軍師 - 作戦立案・コードアーキテクチャ監査・複雑タスク分解"

    def get_api_base(self) -> str:
        """Get API base URL for the configured provider"""
        if self.api_base:
            return self.api_base

        providers = {
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        return providers.get(self.provider, self.api_base)

    def get_model_name(self) -> str:
        """Get provider-specific model name"""
        if self.model_name != "qwen3-coder-next":
            return self.model_name

        names = {
            "dashscope": "qwen3-coder-next",
            "openrouter": "qwen/qwen3-coder-next",
        }
        return names.get(self.provider, self.model_name)


@dataclass
class CoderNextStats:
    """Qwen3-Coder-Next usage statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_inference_time_seconds: float = 0.0
    estimated_cost_yen: float = 0.0

    # 軍師固有統計
    strategy_plans_generated: int = 0
    code_reviews_completed: int = 0
    task_decompositions: int = 0


class Qwen3CoderNextClient:
    """
    Qwen3-Coder-Next 80B-A3B API Client

    軍師 (Gunshi) 専用クライアント
    - 256K コンテキストでコードベース全体を俯瞰
    - SWE-Bench 70.6% の実力で複雑タスクを分解
    - Non-thinking mode = 即断即決

    API経由でのアクセス (3B活性のためコスト効率が高い)
    """

    VERSION = "10.0"

    # 軍師システムプロンプト
    GUNSHI_SYSTEM_PROMPT = """あなたは武士団マルチエージェントシステムの「軍師」です。

【役割】
- 複雑なタスクを分析し、具体的な作戦計画を立案する
- コードアーキテクチャを俯瞰し、設計の問題点を発見する
- 大将（実装層）への指示を明確に構造化する
- 実装結果を監査し、品質を保証する

【行動規範】
- 即断即決: 冗長な思考過程は不要。結論と根拠を簡潔に示す
- 全体俯瞰: 個別ファイルではなくシステム全体の整合性を重視
- 実践重視: 理論より実装可能な具体的指示を出す
- 品質保証: 侍大将の実装を常に監査する視点を持つ

【出力形式】
作戦計画は以下の形式で出力:
1. 目標: 何を達成するか
2. 分析: 現状の問題点
3. 作戦: 具体的な実装ステップ (優先度付き)
4. リスク: 注意すべき点
5. 検証: 完了条件"""

    def __init__(
        self,
        config: Optional[CoderNextConfig] = None,
        api_key: Optional[str] = None,
        provider: str = "dashscope"
    ):
        self.config = config or CoderNextConfig(
            provider=provider,
            api_key=api_key or ""
        )
        if api_key:
            self.config.api_key = api_key

        self.stats = CoderNextStats()

        logger.info(
            f"🧠 軍師クライアント初期化 (Qwen3-Coder-Next 80B)\n"
            f"   Provider: {self.config.provider}\n"
            f"   Model: {self.config.get_model_name()}\n"
            f"   Context: {self.config.context_window:,} tokens"
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate completion via API

        Args:
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Override system prompt

        Returns:
            Generated text response
        """
        if max_tokens is None:
            max_tokens = self.config.default_max_tokens
        if temperature is None:
            temperature = self.config.default_temperature

        # 軍師システムプロンプト注入
        full_messages = self._prepare_messages(messages, system_prompt)

        start_time = asyncio.get_event_loop().time()

        try:
            response_data = await self._make_request(full_messages, max_tokens, temperature)

            elapsed = asyncio.get_event_loop().time() - start_time
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.total_inference_time_seconds += elapsed

            # Token tracking
            prompt_tokens = response_data.get("prompt_tokens", 0)
            completion_tokens = response_data.get("completion_tokens", 0)
            self.stats.total_prompt_tokens += prompt_tokens
            self.stats.total_completion_tokens += completion_tokens

            # Cost estimation (3B active params = very cheap)
            cost = self._estimate_cost(prompt_tokens, completion_tokens)
            self.stats.estimated_cost_yen += cost

            content = response_data.get("content", "")

            logger.info(
                f"🧠 軍師応答完了: {completion_tokens} tokens / {elapsed:.2f}s "
                f"(cost: ¥{cost:.2f})"
            )

            return content

        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            logger.error(f"❌ 軍師API失敗: {e}")
            raise

    async def plan_strategy(
        self,
        task_content: str,
        codebase_context: Optional[str] = None
    ) -> str:
        """
        作戦立案: 複雑タスクの実行計画を生成

        Args:
            task_content: タスク内容
            codebase_context: コードベースのコンテキスト

        Returns:
            構造化された作戦計画
        """
        messages = []

        if codebase_context:
            messages.append({
                "role": "user",
                "content": f"【コードベース情報】\n{codebase_context}"
            })
            messages.append({
                "role": "assistant",
                "content": "コードベースを把握しました。タスクの指示をお願いします。"
            })

        messages.append({
            "role": "user",
            "content": f"以下のタスクの作戦計画を立案してください。\n\n【タスク】\n{task_content}"
        })

        result = await self.generate(messages)
        self.stats.strategy_plans_generated += 1
        return result

    async def review_code(
        self,
        code: str,
        review_focus: str = "品質・セキュリティ・パフォーマンス"
    ) -> str:
        """
        コードレビュー: 実装結果を監査

        Args:
            code: レビュー対象コード
            review_focus: レビューの焦点

        Returns:
            レビュー結果
        """
        messages = [{
            "role": "user",
            "content": (
                f"以下のコードをレビューしてください。\n"
                f"観点: {review_focus}\n\n"
                f"```\n{code}\n```\n\n"
                f"問題点と改善案を具体的に示してください。"
            )
        }]

        result = await self.generate(messages)
        self.stats.code_reviews_completed += 1
        return result

    async def decompose_task(
        self,
        complex_task: str,
        constraints: Optional[str] = None
    ) -> str:
        """
        タスク分解: 複雑タスクを実装可能な単位に分割

        Args:
            complex_task: 複雑なタスク
            constraints: 制約条件

        Returns:
            分解されたサブタスクリスト
        """
        content = f"以下の複雑タスクを実装可能な単位に分解してください。\n\n【タスク】\n{complex_task}"

        if constraints:
            content += f"\n\n【制約】\n{constraints}"

        content += (
            "\n\n各サブタスクには以下を含めてください:\n"
            "- 実行順序 (依存関係考慮)\n"
            "- 担当推奨 (ローカルQwen3 / API / 手動)\n"
            "- 推定難易度\n"
            "- 完了条件"
        )

        messages = [{"role": "user", "content": content}]

        result = await self.generate(messages)
        self.stats.task_decompositions += 1
        return result

    def _prepare_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages with system prompt"""
        prompt = system_prompt or self.GUNSHI_SYSTEM_PROMPT

        full = [{"role": "system", "content": prompt}]

        for msg in messages:
            if msg.get("role") != "system":
                full.append(msg)

        return full

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Make OpenAI-compatible API request"""

        try:
            import httpx

            api_base = self.config.get_api_base()
            model_name = self.config.get_model_name()
            url = f"{api_base}/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }

            # OpenRouter requires extra headers
            if self.config.provider == "openrouter":
                headers["HTTP-Referer"] = "https://github.com/98kuwa036/Bushidan-Multi-Agent"

            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": self.config.default_top_p,
                "stream": False
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code != 200:
                    error = response.text
                    raise Exception(f"API error {response.status_code}: {error}")

                result = response.json()

                if "choices" in result and result["choices"]:
                    choice = result["choices"][0]
                    usage = result.get("usage", {})

                    return {
                        "content": choice["message"]["content"],
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0)
                    }

                raise Exception("Empty response from API")

        except Exception as e:
            logger.error(f"❌ Qwen3-Coder-Next API request failed: {e}")
            raise

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in yen (3B active = very cheap)"""
        # DashScope pricing for Qwen3 (approximate)
        # Input: ~¥0.002/1K tokens, Output: ~¥0.006/1K tokens
        input_cost = (prompt_tokens / 1000) * 0.002
        output_cost = (completion_tokens / 1000) * 0.006
        return input_cost + output_cost

    async def health_check(self) -> bool:
        """Check API availability"""
        try:
            messages = [{"role": "user", "content": "OK"}]
            await self.generate(messages, max_tokens=5, temperature=0.1)
            return True
        except Exception as e:
            logger.warning(f"⚠️ 軍師ヘルスチェック失敗: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests

        return {
            "model": "Qwen3-Coder-Next (80B/3B active)",
            "provider": self.config.provider,
            "role": "軍師 (Gunshi)",
            "total_requests": self.stats.total_requests,
            "success_rate": round(success_rate, 3),
            "total_prompt_tokens": self.stats.total_prompt_tokens,
            "total_completion_tokens": self.stats.total_completion_tokens,
            "total_inference_time_seconds": round(self.stats.total_inference_time_seconds, 1),
            "estimated_cost_yen": round(self.stats.estimated_cost_yen, 2),
            "gunshi_stats": {
                "strategy_plans": self.stats.strategy_plans_generated,
                "code_reviews": self.stats.code_reviews_completed,
                "task_decompositions": self.stats.task_decompositions
            },
            "context_window": self.config.context_window,
            "capabilities": [
                "256K context (codebase overview)",
                "SWE-Bench 70.6%",
                "Non-thinking mode (fast decisions)",
                "358 coding languages"
            ]
        }
