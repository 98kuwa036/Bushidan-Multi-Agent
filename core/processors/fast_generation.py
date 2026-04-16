"""
武士団 v18 — Phase 2 高速筆耕ライン
Cerebras → Groq → Haiku 3 段階生成

Stage 1: Cerebras (gemma2-9b-it) で日本語荒削りドラフト生成 (~80%完成)
Stage 2: Groq (llama-3.2-3b) で超軽量・爆速整形
Stage 3: Haiku で最終清書・品質仕上げ
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# モデル設定
_CEREBRAS_DRAFT_MODEL = "gemma2-9b-it"          # Stage 1: 日本語対応 9B 荒削り
_GROQ_REFINE_MODEL    = "llama-3.2-3b-preview"  # Stage 2: 超軽量 3B 爆速整形
_HAIKU_POLISH_MODEL   = "claude-haiku-4-5-20251001"  # Stage 3: 最終清書

# タイムアウト
_CEREBRAS_TIMEOUT = 15.0
_GROQ_TIMEOUT     = 10.0
_HAIKU_TIMEOUT    = 20.0

# トークン制限
_CEREBRAS_DRAFT_MAX_TOKENS = 1024
_GROQ_REFINE_MAX_TOKENS    = 1024
_HAIKU_POLISH_MAX_TOKENS   = 2048

# Groq 整形プロンプト
_REFINE_SYSTEM = """\
あなたは武士団マルチエージェントシステムの整形担当です。
以下のドラフト回答を簡潔に整形してください。

整形の観点:
- 論理的な流れ・構造の改善
- 冗長性の排除・簡潔化
- 形式や見出しの改善（見やすさ重視）

ドラフトの核心的な情報は保持しつつ、より洗練された形で返してください。
"""

# Haiku ポリッシュ プロンプト
_POLISH_SYSTEM = """\
あなたは武士団マルチエージェントシステムの品質担当です。
以下の整形済み回答を、ユーザーの質問に対してさらに洗練・整形してください。

ガイドライン:
- 内容の正確性を最優先
- マークダウンで読みやすく構造化
- コードブロックは適切にフォーマット
- 日本語で回答（コードは英語OK）
- 整形済みドラフトの品質を損なわない
"""


class FastGenerationPipeline:
    """Cerebras → Groq → Haiku 3 段階生成パイプライン"""

    def __init__(self) -> None:
        self._groq_key = os.getenv("GROQ_API_KEY", "")
        self._cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._groq = None
        self._cerebras = None
        self._anthropic = None

    def _get_cerebras(self):
        if self._cerebras is None and self._cerebras_key:
            try:
                from cerebras.cloud.sdk import Cerebras
                self._cerebras = Cerebras(api_key=self._cerebras_key)
            except ImportError:
                logger.warning("cerebras SDK not installed")
        return self._cerebras

    def _get_groq(self):
        if self._groq is None and self._groq_key:
            try:
                from groq import AsyncGroq
                self._groq = AsyncGroq(api_key=self._groq_key)
            except ImportError:
                logger.warning("groq package not installed")
        return self._groq

    def _get_anthropic(self):
        if self._anthropic is None and self._anthropic_key:
            try:
                import anthropic
                self._anthropic = anthropic.AsyncAnthropic(api_key=self._anthropic_key)
            except ImportError:
                logger.warning("anthropic package not installed")
        return self._anthropic

    async def generate_draft(
        self,
        user_input: str,
        system_prompt: str,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Stage 1: Cerebras (gemma2-9b-it) で日本語荒削りドラフト生成"""
        cerebras = self._get_cerebras()
        if cerebras is None:
            return None

        try:
            response = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: cerebras.chat.completions.create(
                        model=_CEREBRAS_DRAFT_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input},
                        ],
                        max_tokens=_CEREBRAS_DRAFT_MAX_TOKENS,
                        temperature=temperature,
                    ),
                ),
                timeout=_CEREBRAS_TIMEOUT,
            )
            return response.choices[0].message.content or "" if response.choices else ""
        except asyncio.TimeoutError:
            logger.warning("FastGen: Cerebras draft timeout")
            return None
        except Exception as e:
            logger.warning("FastGen: Cerebras draft error: %s", e)
            return None

    async def refine_draft(
        self,
        draft: str,
        user_input: str,
    ) -> Optional[str]:
        """Stage 2: Groq (llama-3.2-3b) で超軽量・爆速整形"""
        groq = self._get_groq()
        if groq is None:
            return None

        refine_user = f"ユーザーの質問:\n{user_input}\n\nドラフト回答:\n{draft}"

        try:
            response = await asyncio.wait_for(
                groq.chat.completions.create(
                    model=_GROQ_REFINE_MODEL,
                    messages=[
                        {"role": "system", "content": _REFINE_SYSTEM},
                        {"role": "user", "content": refine_user},
                    ],
                    max_tokens=_GROQ_REFINE_MAX_TOKENS,
                    temperature=0.4,
                ),
                timeout=_GROQ_TIMEOUT,
            )
            return response.choices[0].message.content if response.choices else None
        except asyncio.TimeoutError:
            logger.warning("FastGen: Groq refine timeout")
            return None
        except Exception as e:
            logger.warning("FastGen: Groq refine error: %s", e)
            return None

    async def polish_draft(
        self,
        draft: str,
        user_input: str,
        context: str = "",
    ) -> str:
        """Stage 3: Haiku でポリッシュ"""
        anthropic = self._get_anthropic()
        if anthropic is None:
            # Anthropic 未設定ならドラフトをそのまま返す
            return draft

        polish_user = f"ユーザーの質問:\n{user_input}\n\n最適化済みドラフト:\n{draft}"
        if context:
            polish_user = f"コンテキスト:\n{context}\n\n{polish_user}"

        try:
            response = await asyncio.wait_for(
                anthropic.messages.create(
                    model=_HAIKU_POLISH_MODEL,
                    system=_POLISH_SYSTEM,
                    messages=[{"role": "user", "content": polish_user}],
                    max_tokens=_HAIKU_POLISH_MAX_TOKENS,
                    temperature=0.3,
                ),
                timeout=_HAIKU_TIMEOUT,
            )
            return response.content[0].text if response.content else draft
        except asyncio.TimeoutError:
            logger.warning("FastGen: Haiku polish timeout, using draft")
            return draft
        except Exception as e:
            logger.warning("FastGen: Haiku polish error: %s, using draft", e)
            return draft

    async def generate(
        self,
        user_input: str,
        system_prompt: str,
        context: str = "",
        skip_refine_polish: bool = False,
    ) -> dict:
        """
        3 段階生成のメインエントリポイント
        Stage 1: Cerebras (gemma2-9b-it) 日本語荒削りドラフト
        Stage 2: Groq (llama-3.2-3b) 超軽量爆速整形
        Stage 3: Haiku 最終清書

        Returns:
            dict with keys: response, draft, refine, stage, draft_ms, refine_ms, polish_ms, total_ms
        """
        t0 = time.time()
        stages: list[str] = []

        # Stage 1: Cerebras 荒削りドラフト
        t1 = time.time()
        draft = await self.generate_draft(user_input, system_prompt)
        draft_ms = (time.time() - t1) * 1000

        if draft is None:
            # Cerebras 失敗 → Haiku で直接生成
            logger.warning("FastGen: Cerebras failed, falling back to Haiku directly")
            anthropic = self._get_anthropic()
            if anthropic is None:
                return {
                    "response": "⚠️ 生成サービスが利用不可です",
                    "draft": "",
                    "refine": "",
                    "stage": "error",
                    "draft_ms": draft_ms,
                    "refine_ms": 0.0,
                    "polish_ms": 0.0,
                    "total_ms": (time.time() - t0) * 1000,
                }
            try:
                t3 = time.time()
                response = await asyncio.wait_for(
                    anthropic.messages.create(
                        model=_HAIKU_POLISH_MODEL,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_input}],
                        max_tokens=_HAIKU_POLISH_MAX_TOKENS,
                        temperature=0.7,
                    ),
                    timeout=_HAIKU_TIMEOUT,
                )
                final = response.content[0].text if response.content else ""
                polish_ms = (time.time() - t3) * 1000
                return {
                    "response": final,
                    "draft": "",
                    "refine": "",
                    "stage": "haiku_direct",
                    "draft_ms": 0.0,
                    "refine_ms": 0.0,
                    "polish_ms": polish_ms,
                    "total_ms": (time.time() - t0) * 1000,
                }
            except Exception as e:
                return {
                    "response": f"⚠️ 生成エラー: {e}",
                    "draft": "",
                    "refine": "",
                    "stage": "error",
                    "draft_ms": 0.0,
                    "refine_ms": 0.0,
                    "polish_ms": 0.0,
                    "total_ms": (time.time() - t0) * 1000,
                }

        stages.append("cerebras_draft")

        if skip_refine_polish:
            return {
                "response": draft,
                "draft": draft,
                "refine": "",
                "stage": "cerebras_only",
                "draft_ms": draft_ms,
                "refine_ms": 0.0,
                "polish_ms": 0.0,
                "total_ms": (time.time() - t0) * 1000,
            }

        # Stage 2: Groq 超軽量整形
        t2 = time.time()
        refined = await self.refine_draft(draft, user_input)
        refine_ms = (time.time() - t2) * 1000

        if refined is None:
            refined = draft  # Groq 失敗時はドラフトをそのまま使用
            logger.debug("FastGen: Groq refine skipped, using draft")
        else:
            stages.append("groq_refine")

        # Stage 3: Haiku 最終清書
        t3 = time.time()
        final = await self.polish_draft(refined, user_input, context)
        polish_ms = (time.time() - t3) * 1000
        stages.append("haiku_polish")

        total_ms = (time.time() - t0) * 1000
        logger.debug(
            "FastGen: draft=%.0fms refine=%.0fms polish=%.0fms total=%.0fms",
            draft_ms, refine_ms, polish_ms, total_ms,
        )

        return {
            "response": final,
            "draft": draft,
            "refine": refined,
            "stage": "+".join(stages),
            "draft_ms": draft_ms,
            "refine_ms": refine_ms,
            "polish_ms": polish_ms,
            "total_ms": total_ms,
        }

    async def stream_generate(
        self,
        user_input: str,
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Groq ストリーミング生成（Stage 2 軽量整形、ポリッシュなし）"""
        groq = self._get_groq()
        if groq is None:
            yield "⚠️ Groq サービスが利用不可です"
            return

        try:
            stream = await asyncio.wait_for(
                groq.chat.completions.create(
                    model=_GROQ_REFINE_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=0.7,
                    max_tokens=_GROQ_REFINE_MAX_TOKENS,
                    stream=True,
                ),
                timeout=_GROQ_TIMEOUT,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except asyncio.TimeoutError:
            yield "\n\n⚠️ Groq タイムアウト"
        except Exception as e:
            yield f"\n\n⚠️ Groq エラー: {e}"
