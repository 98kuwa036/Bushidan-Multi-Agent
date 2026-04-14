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
import re
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


def _tokenize(text: str) -> List[str]:
    """日本語・英語混合テキストをトークン分割する"""
    # 英数字はスペース区切り、日本語は2文字N-gram
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    # 日本語文字 (ひらがな・カタカナ・漢字) のbigram
    ja_chars = re.findall(r'[\u3040-\u9FFF]', text)
    bigrams = [ja_chars[i] + ja_chars[i+1] for i in range(len(ja_chars) - 1)]
    return tokens + bigrams


def _score_chunk(query: str, title: str, content: str) -> float:
    """
    クエリとチャンクの関連度スコアを計算する。

    - タイトル完全一致: +2.0
    - タイトルにクエリ単語: +1.5 per word
    - コンテンツにクエリ単語: +1.0 per word (出現頻度で重み付け)
    - 最大スコアで正規化
    """
    if not query.strip():
        return 0.0

    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0.0

    t_lower = title.lower()
    c_lower = content.lower()

    score = 0.0

    # タイトル完全一致ボーナス
    if query.lower() in t_lower:
        score += 2.0

    # タイトルトークン一致 (重み高)
    t_tokens = set(_tokenize(title))
    title_hits = len(q_tokens & t_tokens)
    score += title_hits * 1.5

    # コンテンツトークン一致 (出現頻度加味)
    for tok in q_tokens:
        count = c_lower.count(tok)
        if count > 0:
            score += min(count * 0.3, 1.5)  # 上限1.5

    # 正規化: クエリトークン数で割る
    return score / max(len(q_tokens), 1)


async def _search_all(client, query: str, limit: int) -> list:
    """Notion search API で全文検索"""
    return await client.search_all_text(query, limit=limit)


async def _query_kb_db(client, query: str, limit: int) -> list:
    """ナレッジベース DB をタイトル・コンテンツフィルタでクエリ"""
    if not client.kb_database_id:
        return []

    # 主クエリ (先頭50文字) でタイトル検索
    filter_obj = {
        "property": "Title",
        "title": {"contains": query[:50]},
    }

    results = await client.search_database(
        client.kb_database_id,
        filter_obj=filter_obj,
        sorts=[{"property": "Date", "direction": "descending"}],
        limit=limit,
    )

    # ヒット数が少ない場合、最初の有効単語でも検索して補完
    if len(results) < 2:
        import re as _re
        words = _re.findall(r'[\w\u3040-\u9FFF]{2,}', query)
        for word in words[:3]:
            if word == query[:len(word)]:
                continue  # 同じなのでスキップ
            try:
                extra = await client.search_database(
                    client.kb_database_id,
                    filter_obj={"property": "Title", "title": {"contains": word}},
                    sorts=[{"property": "Date", "direction": "descending"}],
                    limit=limit,
                )
                results = results + extra
                if len(results) >= limit:
                    break
            except Exception:
                pass

    return results


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

    # 改良スコアリング: タイトル一致を重視 + 出現頻度を加味
    score = _score_chunk(query, title, content)

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
