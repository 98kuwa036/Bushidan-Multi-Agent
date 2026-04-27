"""
武士団 v18 — 高速筆耕ライン
Gemini 3 Flash → Cerebras gpt-oss-120b → Haiku 3 段階生成

Stage 1: Gemini 3 Flash (gemini-3-flash-preview)
         日本語対応・高品質な荒削りドラフト生成

Stage 2: Cerebras gpt-oss-120b (3,000 tok/s)
         OpenAI 120B 品質で爆速整形・構造化

Stage 3: Claude Haiku (claude-haiku-4-5-20251001)
         最終清書・マークダウン仕上げ
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ── モデル設定 ────────────────────────────────────────────────────
_GEMINI_DRAFT_MODEL  = "gemini-3-flash-preview"      # Stage 1: 日本語荒削り
_CEREBRAS_REFINE_MODEL = "gpt-oss-120b"               # Stage 2: 3,000 tok/s 爆速整形
_HAIKU_POLISH_MODEL  = "claude-haiku-4-5-20251001"   # Stage 3: 最終清書

# ── タイムアウト ──────────────────────────────────────────────────
_GEMINI_TIMEOUT   = 20.0
_CEREBRAS_TIMEOUT = 15.0
_HAIKU_TIMEOUT    = 20.0

# ── トークン制限 ──────────────────────────────────────────────────
_GEMINI_DRAFT_MAX_TOKENS   = 1024
_CEREBRAS_REFINE_MAX_TOKENS = 1024
_HAIKU_POLISH_MAX_TOKENS   = 2048

# ── Cerebras 整形プロンプト ────────────────────────────────────────
_REFINE_SYSTEM = """\
あなたは武士団マルチエージェントシステムの整形担当です。
以下のドラフト回答を簡潔に整形してください。

整形の観点:
- 論理的な流れ・構造の改善
- 冗長性の排除・簡潔化
- 形式や見出しの改善（見やすさ重視）

ドラフトの核心的な情報は保持しつつ、より洗練された形で返してください。
"""

# ── Haiku 清書プロンプト ──────────────────────────────────────────
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

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class FastGenerationPipeline:
    """Gemini 3 Flash → Cerebras gpt-oss-120b → Haiku 3 段階生成パイプライン"""

    def __init__(self) -> None:
        self._gemini_key    = os.getenv("GEMINI_API_KEY", "")
        self._cerebras_key  = os.getenv("CEREBRAS_API_KEY", "")
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        # AsyncCerebras: 接続プールを維持してAPI前後のオーバーヘッドを削減
        self._cerebras:  Optional[object] = None
        self._anthropic: Optional[object] = None
        # httpx: Gemini/Cerebras REST 用の永続接続プール
        self._http: Optional[object] = None

    def _get_http(self):
        """永続 httpx.AsyncClient（接続プール維持）"""
        if self._http is None:
            try:
                import httpx
                # limits で接続プールサイズを制御
                self._http = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                )
            except ImportError:
                logger.warning("httpx not installed")
        return self._http

    def _get_cerebras(self):
        if self._cerebras is None and self._cerebras_key:
            try:
                # AsyncCerebras: ネイティブ async + 内部接続プール = run_in_executor 不要
                from cerebras.cloud.sdk import AsyncCerebras
                self._cerebras = AsyncCerebras(api_key=self._cerebras_key)
            except ImportError:
                logger.warning("cerebras SDK not installed")
        return self._cerebras

    def _get_anthropic(self):
        if self._anthropic is None and self._anthropic_key:
            try:
                import anthropic
                self._anthropic = anthropic.AsyncAnthropic(api_key=self._anthropic_key)
            except ImportError:
                logger.warning("anthropic package not installed")
        return self._anthropic

    async def close(self) -> None:
        """接続プールを明示的に閉じる（アプリ終了時）"""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        if self._cerebras is not None:
            try:
                await self._cerebras.close()
            except Exception:
                pass
            self._cerebras = None

    async def generate_draft(
        self,
        user_input: str,
        system_prompt: str,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Stage 1: Gemini 3 Flash で日本語荒削りドラフト生成"""
        if not self._gemini_key:
            return None

        try:
            client = self._get_http()
            if client is None:
                return None

            prompt = f"[System]: {system_prompt}\n\n[User]: {user_input}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": _GEMINI_DRAFT_MAX_TOKENS,
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",        "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",  "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",  "threshold": "BLOCK_ONLY_HIGH"},
                ],
            }
            url = f"{_GEMINI_BASE_URL}/models/{_GEMINI_DRAFT_MODEL}:generateContent"

            # 永続クライアントで接続再利用（毎回 async with で開閉しない）
            resp = await asyncio.wait_for(
                client.post(url, params={"key": self._gemini_key}, json=payload),
                timeout=_GEMINI_TIMEOUT,
            )

            if resp.status_code != 200:
                logger.warning("FastGen: Gemini draft HTTP %s", resp.status_code)
                return None

            result = resp.json()
            candidates = result.get("candidates", [])
            if not candidates:
                logger.warning("FastGen: Gemini draft no candidates")
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts or "text" not in parts[0]:
                logger.warning("FastGen: Gemini draft blocked/empty")
                return None

            return parts[0]["text"]

        except asyncio.TimeoutError:
            logger.warning("FastGen: Gemini draft timeout")
            return None
        except Exception as e:
            logger.warning("FastGen: Gemini draft error: %s", e)
            return None

    async def refine_draft(
        self,
        draft: str,
        user_input: str,
    ) -> Optional[str]:
        """Stage 2: Cerebras gpt-oss-120b (3,000 tok/s) で爆速整形
        AsyncCerebras + 永続接続で API 前後オーバーヘッドを最小化
        """
        cerebras = self._get_cerebras()
        if cerebras is None:
            return None

        refine_user = f"ユーザーの質問:\n{user_input}\n\nドラフト回答:\n{draft}"

        try:
            # AsyncCerebras はネイティブ async → run_in_executor 不要
            response = await asyncio.wait_for(
                cerebras.chat.completions.create(
                    model=_CEREBRAS_REFINE_MODEL,
                    messages=[
                        {"role": "system", "content": _REFINE_SYSTEM},
                        {"role": "user",   "content": refine_user},
                    ],
                    max_tokens=_CEREBRAS_REFINE_MAX_TOKENS,
                    temperature=0.4,
                ),
                timeout=_CEREBRAS_TIMEOUT,
            )
            return response.choices[0].message.content if response.choices else None
        except asyncio.TimeoutError:
            logger.warning("FastGen: Cerebras refine timeout")
            return None
        except Exception as e:
            logger.warning("FastGen: Cerebras refine error: %s", e)
            return None

    async def polish_draft(
        self,
        draft: str,
        user_input: str,
        context: str = "",
    ) -> str:
        """Stage 3: Haiku で最終清書"""
        anthropic = self._get_anthropic()
        if anthropic is None:
            return draft

        polish_user = f"ユーザーの質問:\n{user_input}\n\n整形済みドラフト:\n{draft}"
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

        Stage 1: Gemini 3 Flash  — 日本語荒削りドラフト
        Stage 2: Cerebras 120B   — 3,000 tok/s 爆速整形
        Stage 3: Haiku           — 最終清書

        Returns:
            dict: response, draft, refine, stage, draft_ms, refine_ms, polish_ms, total_ms
        """
        t0 = time.time()
        stages: list[str] = []

        # ── Stage 1: Gemini 荒削りドラフト ───────────────────────────
        t1 = time.time()
        draft = await self.generate_draft(user_input, system_prompt)
        draft_ms = (time.time() - t1) * 1000

        if draft is None:
            # Gemini 失敗 → Haiku 直接生成にフォールバック
            logger.warning("FastGen: Gemini failed, falling back to Haiku directly")
            anthropic = self._get_anthropic()
            if anthropic is None:
                return {
                    "response": "⚠️ 生成サービスが利用不可です",
                    "draft": "", "refine": "", "stage": "error",
                    "draft_ms": draft_ms, "refine_ms": 0.0, "polish_ms": 0.0,
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
                return {
                    "response": final,
                    "draft": "", "refine": "", "stage": "haiku_direct",
                    "draft_ms": 0.0, "refine_ms": 0.0,
                    "polish_ms": (time.time() - t3) * 1000,
                    "total_ms": (time.time() - t0) * 1000,
                }
            except Exception as e:
                return {
                    "response": f"⚠️ 生成エラー: {e}",
                    "draft": "", "refine": "", "stage": "error",
                    "draft_ms": 0.0, "refine_ms": 0.0, "polish_ms": 0.0,
                    "total_ms": (time.time() - t0) * 1000,
                }

        stages.append("gemini_draft")

        if skip_refine_polish:
            return {
                "response": draft, "draft": draft, "refine": "",
                "stage": "gemini_only",
                "draft_ms": draft_ms, "refine_ms": 0.0, "polish_ms": 0.0,
                "total_ms": (time.time() - t0) * 1000,
            }

        # ── Stage 2: Cerebras 爆速整形 ───────────────────────────────
        t2 = time.time()
        refined = await self.refine_draft(draft, user_input)
        refine_ms = (time.time() - t2) * 1000

        if refined is None:
            refined = draft
            logger.debug("FastGen: Cerebras refine skipped, using draft")
        else:
            stages.append("cerebras_refine")

        # ── Stage 3: Haiku 最終清書 ───────────────────────────────────
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
        """
        Gemini ストリーミング生成（Stage 1 のみ、整形・清書なし）
        リアルタイム表示が必要な場合に使用
        """
        if not self._gemini_key:
            yield "⚠️ Gemini サービスが利用不可です"
            return

        try:
            client = self._get_http()
            if client is None:
                yield "⚠️ HTTP クライアント初期化失敗"
                return

            prompt = f"[System]: {system_prompt}\n\n[User]: {user_input}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": _GEMINI_DRAFT_MAX_TOKENS,
                },
            }
            url = f"{_GEMINI_BASE_URL}/models/{_GEMINI_DRAFT_MODEL}:streamGenerateContent"

            # 永続クライアントでストリーミング
            async with client.stream(
                "POST", url,
                params={"key": self._gemini_key, "alt": "sse"},
                json=payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        try:
                            chunk = json.loads(line[6:])
                            parts = (
                                chunk.get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [])
                            )
                            if parts and "text" in parts[0]:
                                yield parts[0]["text"]
                        except Exception:
                            pass

        except asyncio.TimeoutError:
            yield "\n\n⚠️ Gemini タイムアウト"
        except Exception as e:
            yield f"\n\n⚠️ Gemini エラー: {e}"
