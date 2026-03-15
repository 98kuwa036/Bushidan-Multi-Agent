"""
integrations/notion/retrieval.py — Notion RAG 検索

LangGraph の notion_retrieve ノードから呼び出す。

検索戦略:
  1. Notion search API でキーワード全文検索 (最速)
  2. ナレッジベース DB のタイトル一致クエリ
  3. 各ページのブロック内容を取得してチャンクに変換
  4. スコアリングして上位 N 件を返す

返り値: List[NotionChunk]
  {title, content, page_id, url, score, entry_type}
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from integrations.notion.client import get_notion_client

logger = logging.getLogger("integrations.notion.retrieval")


@dataclass
class NotionChunk:
    """RAG 検索結果チャンク"""
    title: str
    content: str
    page_id: str
    url: str = ""
    score: float = 0.0
    entry_type: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "page_id": self.page_id,
            "url": self.url,
            "score": self.score,
            "entry_type": self.entry_type,
        }


async def retrieve(
    query: str,
    top_k: int = 4,
    max_content_chars: int = 600,
) -> List[dict]:
    """
    クエリに関連する Notion ナレッジチャンクを返す。

    Args:
        query:             検索クエリ (自然言語)
        top_k:             返す最大件数
        max_content_chars: 各チャンクの最大文字数

    Returns:
        List[dict] — NotionChunk.to_dict() のリスト
    """
    client = get_notion_client()
    if not client.is_available() or not query.strip():
        return []

    try:
        # 検索を並列実行: search API + DB クエリ
        search_task = _search_all(client, query, top_k)
        db_task = _query_kb_db(client, query, top_k)
        search_pages, db_pages = await asyncio.gather(search_task, db_task, return_exceptions=True)

        pages_seen: set = set()
        candidates: List[dict] = []

        for pages in (search_pages, db_pages):
            if isinstance(pages, Exception):
                logger.debug("検索エラー (無視): %s", pages)
                continue
            for page in pages:
                page_id = page.get("id", "")
                if page_id in pages_seen:
                    continue
                pages_seen.add(page_id)
                candidates.append(page)

        if not candidates:
            return []

        # 各ページのコンテンツを並列取得 (上位 top_k*2 件)
        content_tasks = [
            _fetch_chunk(client, page, query, max_content_chars)
            for page in candidates[:top_k * 2]
        ]
        chunks: List[Optional[NotionChunk]] = await asyncio.gather(
            *content_tasks, return_exceptions=True
        )

        results: List[NotionChunk] = []
        for c in chunks:
            if isinstance(c, NotionChunk) and c.content:
                results.append(c)

        # スコアで降順ソート
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info("[Notion RAG] クエリ='%s' → %d件取得", query[:40], len(results[:top_k]))
        return [c.to_dict() for c in results[:top_k]]

    except Exception as e:
        logger.error("[Notion RAG] retrieve エラー: %s", e)
        return []


async def _search_all(client, query: str, limit: int) -> list:
    """Notion search API で全文検索"""
    return await client.search_all_text(query, limit=limit)


async def _query_kb_db(client, query: str, limit: int) -> list:
    """ナレッジベース DB をタイトルフィルタでクエリ"""
    if not client.kb_database_id:
        return []
    filter_obj = {
        "property": "Title",
        "title": {"contains": query[:50]},
    }
    return await client.search_database(
        client.kb_database_id,
        filter_obj=filter_obj,
        sorts=[{"property": "Date", "direction": "descending"}],
        limit=limit,
    )


async def _fetch_chunk(
    client, page: dict, query: str, max_chars: int
) -> Optional[NotionChunk]:
    """ページのコンテンツを取得して NotionChunk に変換"""
    page_id = page.get("id", "")
    url = page.get("url", "")
    title = await client.get_page_title(page)
    content = await client.get_page_content(page_id, max_chars=max_chars)
    if not content:
        return None

    # 簡易スコアリング: クエリの単語がタイトル/コンテンツに含まれる回数
    query_words = set(query.lower().split())
    hits = sum(
        1 for w in query_words
        if w in title.lower() or w in content.lower()
    )
    score = hits / max(len(query_words), 1)

    # エントリタイプを取得 (DB の "Type" プロパティ)
    entry_type = ""
    props = page.get("properties", {})
    type_prop = props.get("Type", {})
    if type_prop.get("select"):
        entry_type = type_prop["select"].get("name", "")

    return NotionChunk(
        title=title,
        content=content,
        page_id=page_id,
        url=url,
        score=score,
        entry_type=entry_type,
    )
