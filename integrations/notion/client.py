"""
integrations/notion/client.py — Notion API 非同期ラッパー

notion-client (同期) を asyncio.to_thread でラップ。
retrieval.py / storage.py から共通利用する。

環境変数:
  NOTION_API_KEY        - Notion Integration トークン
  NOTION_DATABASE_ID    - タスク記録用データベース ID
  NOTION_KB_DATABASE_ID - ナレッジベース用データベース ID (省略時は DATABASE_ID と同じ)
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("integrations.notion.client")

try:
    from notion_client import Client as NotionSDK
    HAS_NOTION = True
except ImportError:
    HAS_NOTION = False
    NotionSDK = None


class NotionClient:
    """
    Notion API 非同期ラッパー。

    使い方:
        client = NotionClient.from_env()
        if client.is_available():
            pages = await client.search_database(db_id, filter_obj, limit=5)
            content = await client.get_page_content(page_id)
            page_id = await client.create_page(db_id, properties, children)
    """

    # notion-client v3 は API 2025-09-03 を使用するが、
    # databases.query が廃止されているため 2022-06-28 に固定する
    _NOTION_VERSION = "2022-06-28"

    def __init__(
        self,
        api_key: str,
        database_id: str = "",
        kb_database_id: str = "",
    ) -> None:
        self._api_key = api_key
        self.database_id = database_id
        self.kb_database_id = kb_database_id or database_id
        self._sdk: Optional[NotionSDK] = None
        if HAS_NOTION and api_key:
            self._sdk = NotionSDK(auth=api_key, notion_version=self._NOTION_VERSION)
            logger.info("[Notion] クライアント初期化完了 db=%s kb=%s (api-version=%s)",
                        database_id[:8] if database_id else "未設定",
                        kb_database_id[:8] if kb_database_id else "同上",
                        self._NOTION_VERSION)
        else:
            if not HAS_NOTION:
                logger.warning("[Notion] notion-client 未インストール: pip install notion-client")
            elif not api_key:
                logger.warning("[Notion] NOTION_API_KEY 未設定")

    @classmethod
    def from_env(cls) -> "NotionClient":
        """環境変数からクライアントを生成する。"""
        return cls(
            api_key=os.environ.get("NOTION_API_KEY", ""),
            database_id=os.environ.get("NOTION_DATABASE_ID", ""),
            kb_database_id=os.environ.get("NOTION_KB_DATABASE_ID", ""),
        )

    def is_available(self) -> bool:
        return self._sdk is not None

    # ── 検索 ─────────────────────────────────────────────────────────

    async def search_database(
        self,
        database_id: str,
        filter_obj: Optional[Dict] = None,
        sorts: Optional[List[Dict]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """データベースをクエリして結果リストを返す。
        notion-client v3 では databases.query が削除されたため
        request() メソッドで直接 REST エンドポイントを呼び出す。
        """
        if not self._sdk:
            return []
        def _query():
            body: Dict[str, Any] = {"page_size": limit}
            if filter_obj:
                body["filter"] = filter_obj
            if sorts:
                body["sorts"] = sorts
            return self._sdk.request(
                path=f"databases/{database_id}/query",
                method="POST",
                body=body,
            )
        try:
            result = await asyncio.to_thread(_query)
            return result.get("results", [])
        except Exception as e:
            logger.error("[Notion] search_database エラー: %s", e)
            return []

    async def search_all_text(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Notion 全体からテキスト検索 (search API)。
        タイトル一致 + ページ内容を取得する。
        """
        if not self._sdk or not query.strip():
            return []
        def _search():
            return self._sdk.search(
                query=query,
                filter={"value": "page", "property": "object"},
                page_size=limit,
            )
        try:
            result = await asyncio.to_thread(_search)
            return result.get("results", [])
        except Exception as e:
            logger.error("[Notion] search_all_text エラー: %s", e)
            return []

    # ── ページ内容取得 ────────────────────────────────────────────────

    async def get_page_content(self, page_id: str, max_chars: int = 2000) -> str:
        """ページのブロック内容を結合してテキストで返す。"""
        if not self._sdk:
            return ""
        def _get_blocks():
            return self._sdk.blocks.children.list(block_id=page_id)
        try:
            blocks_resp = await asyncio.to_thread(_get_blocks)
            texts: List[str] = []
            for block in blocks_resp.get("results", []):
                block_type = block.get("type", "")
                block_data = block.get(block_type, {})
                for rt in block_data.get("rich_text", []):
                    text = rt.get("text", {}).get("content", "")
                    if text:
                        texts.append(text)
            return "\n".join(texts)[:max_chars]
        except Exception as e:
            logger.debug("[Notion] get_page_content エラー (id=%s): %s", page_id, e)
            return ""

    async def get_page_title(self, page: Dict) -> str:
        """ページオブジェクトからタイトルを抽出する。"""
        props = page.get("properties", {})
        # タイトル型プロパティを名前に関係なく検索
        for key in ("タイトル", "名前", "Title", "Name", "スキル名"):
            prop = props.get(key, {})
            prop_type = prop.get("type", "")
            if prop_type == "title":
                items = prop.get("title", [])
                if items:
                    return items[0].get("plain_text", items[0].get("text", {}).get("content", ""))
        # どのプロパティもタイトル型でなければ、title型を持つプロパティを探す
        for key, prop in props.items():
            if prop.get("type") == "title":
                items = prop.get("title", [])
                if items:
                    return items[0].get("plain_text", items[0].get("text", {}).get("content", ""))
        return page.get("url", "")[:60]

    # ── ページ作成 / 更新 ─────────────────────────────────────────────

    async def create_page(
        self,
        database_id: str,
        properties: Dict[str, Any],
        children: Optional[List[Dict]] = None,
    ) -> Optional[str]:
        """ページを作成してページIDを返す。"""
        if not self._sdk:
            return None
        def _create():
            kwargs: Dict[str, Any] = {
                "parent": {"database_id": database_id},
                "properties": properties,
            }
            if children:
                kwargs["children"] = children
            return self._sdk.pages.create(**kwargs)
        try:
            page = await asyncio.to_thread(_create)
            return page.get("id")
        except Exception as e:
            logger.error("[Notion] create_page エラー: %s", e)
            return None

    async def update_page(
        self,
        page_id: str,
        properties: Optional[Dict] = None,
        append_content: str = "",
    ) -> bool:
        """ページを更新する (プロパティ更新 + コンテンツ追記)。"""
        if not self._sdk:
            return False
        try:
            if properties:
                await asyncio.to_thread(
                    self._sdk.pages.update, page_id=page_id, properties=properties
                )
            if append_content:
                blocks = [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": append_content[:1900]}}]
                        },
                    }
                ]
                await asyncio.to_thread(
                    self._sdk.blocks.children.append, block_id=page_id, children=blocks
                )
            return True
        except Exception as e:
            logger.warning("[Notion] update_page エラー (id=%s): %s", page_id, e)
            return False


# モジュールレベルのシングルトン (遅延初期化)
_client: Optional[NotionClient] = None


def get_notion_client() -> NotionClient:
    """シングルトン NotionClient を返す (環境変数から自動初期化)。"""
    global _client
    if _client is None:
        _client = NotionClient.from_env()
    return _client
