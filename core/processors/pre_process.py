"""
武士団 v18 — Karasu (Pre-Process) プロセッサ
Tavily 検索 + キーワード抽出で外部情報を取得
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import List, Optional

from utils.logger import get_logger
from core.models.karasu import KarasuOutput, SearchResult

logger = get_logger(__name__)

# ─── 簡易 NER: 正規表現ベース（spacy 未インストール環境向け）───────────────────
_STOP_WORDS = {
    "について", "を", "は", "が", "に", "で", "の", "て", "も", "か",
    "して", "する", "です", "ます", "ください", "おく", "いる", "ある",
    "this", "that", "the", "a", "an", "is", "are", "was", "were",
    "how", "what", "why", "when", "where", "which",
}


def _extract_keywords(text: str) -> List[str]:
    """簡易キーワード抽出（NER 代替）"""
    # カタカナ語（技術用語）
    katakana = re.findall(r'[ァ-ヶー]{3,}', text)
    # 英数字の識別子・ライブラリ名
    identifiers = re.findall(r'\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{1,})?\b', text)
    # Python/プログラミング関連キーワード
    code_kws = re.findall(r'\b(?:asyncio|async|await|Django|FastAPI|React|PostgreSQL|Redis|Docker|Kubernetes|Python|JavaScript|TypeScript|Go|Rust|Java|C\+\+|SQL|API|REST|GraphQL|OAuth|JWT|WebSocket|CI/CD|Git|Linux|Nginx|pgvector|LangGraph|LLM|RAG|GPT|Claude|Gemini|Groq|Notion)\b', text)
    # 日本語の固有名詞（カタカナ + 漢字 3 文字以上）
    jp_nouns = re.findall(r'[一-龯]{2,}(?:処理|設計|最適化|実装|開発|管理|分析|構築|統合|改善|自動化|検索|学習|推論|生成)', text)

    keywords = katakana + identifiers + code_kws + jp_nouns
    # 重複除去・短すぎるものを除外
    seen = set()
    unique = []
    for kw in keywords:
        kw = kw.strip()
        if kw and len(kw) >= 2 and kw.lower() not in _STOP_WORDS and kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:10]


class KarasuProcessor:
    """Pre-Process（烏）プロセッサ"""

    def __init__(self) -> None:
        self._api_key: str = os.getenv("TAVILY_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None and self._api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self._api_key)
            except ImportError:
                logger.warning("tavily-python not installed")
        return self._client

    async def process(self, user_input: str) -> KarasuOutput:
        """ユーザー入力を受け取り KarasuOutput を返す"""
        start = time.time()

        keywords = _extract_keywords(user_input)
        search_query = " ".join(keywords[:3]) if keywords else user_input[:60]

        search_results: List[SearchResult] = []
        fallback_used = False
        search_reasoning = f"キーワード抽出: {keywords[:5]}。検索クエリ: 「{search_query}」"

        # 入力が短すぎる・コード片の場合は検索をスキップ
        skip_search = (
            len(user_input.strip()) < 10
            or "```" in user_input
            or user_input.strip().startswith("def ")
            or user_input.strip().startswith("class ")
        )

        if not skip_search:
            client = self._get_client()
            if client:
                try:
                    raw = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda: client.search(
                                query=search_query,
                                max_results=5,
                                search_depth="basic",
                            )
                        ),
                        timeout=5.0,
                    )
                    for item in raw.get("results", [])[:5]:
                        search_results.append(SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=(item.get("content", "") or item.get("snippet", ""))[:300],
                            published_date=item.get("published_date"),
                        ))
                    search_reasoning += f"。Tavily 検索成功: {len(search_results)} 件取得"
                except asyncio.TimeoutError:
                    logger.warning("Karasu: Tavily timeout")
                    fallback_used = True
                    search_reasoning += "。Tavily タイムアウト → fallback"
                except Exception as e:
                    logger.warning("Karasu: Tavily error: %s", e)
                    fallback_used = True
                    search_reasoning += f"。Tavily エラー ({type(e).__name__}) → fallback"
            else:
                fallback_used = True
                search_reasoning += "。Tavily クライアント未設定 → fallback"
        else:
            search_reasoning += "。短入力/コード片のため検索スキップ"

        elapsed_ms = (time.time() - start) * 1000
        confidence = 0.85 if search_results else (0.4 if fallback_used else 0.6)

        return KarasuOutput(
            original_input=user_input,
            keywords=keywords,
            entities={},
            search_results=search_results,
            search_query=search_query,
            processing_time_ms=elapsed_ms,
            confidence=confidence,
            fallback_used=fallback_used,
            search_reasoning=search_reasoning,
        )

    def build_system_prompt_injection(self, karasu: KarasuOutput) -> str:
        """System prompt 先頭への注入テキストを生成"""
        if not karasu.search_results:
            return ""

        lines = ["【本日の時事情報】", f"検索キーワード: {', '.join(karasu.keywords[:5])}", ""]
        for sr in karasu.search_results[:3]:
            lines.append(f"▍ {sr.title}")
            lines.append(sr.snippet)
            lines.append(f"URL: {sr.url}")
            lines.append("")

        lines.append("このニュース・情報を踏まえて、ユーザーの質問に回答してください。")
        return "\n".join(lines)
