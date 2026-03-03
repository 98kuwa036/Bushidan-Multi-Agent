"""Notion Integration v11.5 - 積極活用モード

v11.5 強化: Notion を「受動的記録」から「能動的知識連携」へ昇格。

新機能:
  - get_routing_context(): 家訓 + 直近タスク要約をルーター判断用テキストで返す
  - auto_save_task_result(): 全タスク完了後に自動保存 (エージェント役職・ルート情報付き)
  - get_page_content(): ページのブロックコンテンツを全文取得
  - update_entry(): 既存ページの内容・ステータスを更新

LangGraph Router v11.5 連携:
  - fetch_context ノードが get_routing_context() を呼び出してルーター判断に活用
  - persist_notion ノードが auto_save_task_result() でタスク履歴を自動永続化

既存機能:
  - save_summary(): 60日要約保存
  - save_family_precepts(): 家訓保存
  - save_knowledge_entry(): 汎用知識エントリ保存
  - search_knowledge(): 知識ベース検索
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

try:
    from notion_client import Client
except ImportError:
    Client = None

logger = logging.getLogger("shogun.notion")


class NotionIntegration:
    """Notion integration for knowledge management."""

    def __init__(self, token: str, database_id: str):
        self.token = token
        self.database_id = database_id
        self.client = None
        
        # Statistics
        self.stats = {
            "summaries_saved": 0,
            "precepts_saved": 0,
            "knowledge_entries": 0,
            "search_queries": 0,
        }
        
        if Client is None:
            logger.warning("[Notion] notion-clientライブラリ未インストール - pip install notion-client")
            return
            
        if not token or not database_id:
            logger.warning("[Notion] トークンまたはDB ID未設定")
            return
            
        self.client = Client(auth=token)
        logger.info("[Notion] ナレッジ統合初期化完了")
    
    async def save_summary(self, summary: str, metadata: Optional[Dict] = None) -> bool:
        """Save 60-day summary to Notion."""
        if not self.client:
            return False
            
        try:
            properties = {
                "Title": {
                    "title": [{
                        "type": "text",
                        "text": {
                            "content": f"60日要約 - {datetime.now().strftime('%Y-%m-%d')}"
                        }
                    }]
                },
                "Type": {
                    "select": {
                        "name": "60日要約"
                    }
                },
                "Date": {
                    "date": {
                        "start": datetime.now().isoformat()
                    }
                },
                "Status": {
                    "select": {
                        "name": "完了"
                    }
                }
            }
            
            # Add metadata if provided
            if metadata:
                if metadata.get("cost_total"):
                    properties["Cost (¥)"] = {
                        "number": metadata["cost_total"]
                    }
                if metadata.get("session_count"):
                    properties["Sessions"] = {
                        "number": metadata["session_count"]
                    }
            
            # Create page
            page = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties,
                children=[
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": summary}
                            }]
                        }
                    }
                ]
            )
            
            self.stats["summaries_saved"] += 1
            self.stats["knowledge_entries"] += 1
            
            logger.info("[Notion] 60日要約保存完了: %s", page["id"])
            return True
            
        except Exception as e:
            logger.error("[Notion] 要約保存失敗: %s", e)
            return False
    
    async def save_family_precepts(self, precepts: List[str], context: str = "") -> bool:
        """Save family precepts (家訓) to Notion."""
        if not self.client or not precepts:
            return False
            
        try:
            # Create one page for all precepts
            properties = {
                "Title": {
                    "title": [{
                        "type": "text",
                        "text": {
                            "content": f"家訓集 - {datetime.now().strftime('%Y-%m-%d')}"
                        }
                    }]
                },
                "Type": {
                    "select": {
                        "name": "家訓"
                    }
                },
                "Date": {
                    "date": {
                        "start": datetime.now().isoformat()
                    }
                },
                "Count": {
                    "number": len(precepts)
                }
            }
            
            # Build content blocks
            children = []
            
            if context:
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": "コンテキスト"}
                        }]
                    }
                })
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": context[:1000]}
                        }]
                    }
                })
            
            children.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": "家訓一覧"}
                    }]
                }
            })
            
            # Add each precept as bullet point
            for precept in precepts:
                children.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": precept}
                        }]
                    }
                })
            
            # Create page
            page = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties,
                children=children
            )
            
            self.stats["precepts_saved"] += len(precepts)
            self.stats["knowledge_entries"] += 1
            
            logger.info("[Notion] 家訓保存完了: %d件, ID: %s", len(precepts), page["id"])
            return True
            
        except Exception as e:
            logger.error("[Notion] 家訓保存失敗: %s", e)
            return False
    
    async def save_knowledge_entry(
        self, 
        title: str, 
        content: str, 
        entry_type: str = "知識",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save general knowledge entry to Notion."""
        if not self.client:
            return False
            
        try:
            properties = {
                "Title": {
                    "title": [{
                        "type": "text",
                        "text": {"content": title}
                    }]
                },
                "Type": {
                    "select": {
                        "name": entry_type
                    }
                },
                "Date": {
                    "date": {
                        "start": datetime.now().isoformat()
                    }
                }
            }
            
            # Add tags if provided
            if tags:
                properties["Tags"] = {
                    "multi_select": [
                        {"name": tag} for tag in tags[:5]  # Limit to 5 tags
                    ]
                }
            
            # Add metadata
            if metadata:
                if metadata.get("cost"):
                    properties["Cost (¥)"] = {"number": metadata["cost"]}
                if metadata.get("agent"):
                    properties["Agent"] = {
                        "select": {"name": metadata["agent"]}
                    }
            
            # Create content blocks
            children = []
            
            # Split content into chunks (Notion has block size limits)
            content_chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
            
            for chunk in content_chunks:
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": chunk}
                        }]
                    }
                })
            
            # Create page
            page = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties,
                children=children
            )
            
            self.stats["knowledge_entries"] += 1
            
            logger.info("[Notion] 知識エントリ保存完了: %s", page["id"])
            return True
            
        except Exception as e:
            logger.error("[Notion] 知識エントリ保存失敗: %s", e)
            return False
    
    async def search_knowledge(
        self, 
        query: str, 
        entry_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search knowledge base in Notion."""
        if not self.client:
            return []
            
        try:
            # Build filter
            filter_conditions = {
                "and": []
            }
            
            # Add text search
            if query.strip():
                filter_conditions["and"].append({
                    "property": "Title",
                    "title": {
                        "contains": query
                    }
                })
            
            # Add type filter
            if entry_type:
                filter_conditions["and"].append({
                    "property": "Type",
                    "select": {
                        "equals": entry_type
                    }
                })
            
            # Search database
            results = self.client.databases.query(
                database_id=self.database_id,
                filter=filter_conditions if filter_conditions["and"] else None,
                sorts=[
                    {
                        "property": "Date",
                        "direction": "descending"
                    }
                ],
                page_size=limit
            )
            
            self.stats["search_queries"] += 1
            
            # Format results
            formatted_results = []
            for page in results.get("results", []):
                properties = page.get("properties", {})
                
                title = ""
                if "Title" in properties and properties["Title"]["title"]:
                    title = properties["Title"]["title"][0]["text"]["content"]
                
                entry_type_val = ""
                if "Type" in properties and properties["Type"]["select"]:
                    entry_type_val = properties["Type"]["select"]["name"]
                
                date_val = ""
                if "Date" in properties and properties["Date"]["date"]:
                    date_val = properties["Date"]["date"]["start"]
                
                formatted_results.append({
                    "id": page["id"],
                    "title": title,
                    "type": entry_type_val,
                    "date": date_val,
                    "url": page["url"],
                })
            
            logger.info("[Notion] 検索結果: %d件 (クエリ: '%s')", len(formatted_results), query)
            return formatted_results
            
        except Exception as e:
            logger.error("[Notion] 検索エラー: %s", e)
            return []
    
    async def get_recent_entries(self, limit: int = 20) -> List[Dict]:
        """Get recent knowledge entries."""
        return await self.search_knowledge("", limit=limit)
    
    async def get_family_precepts(self, limit: int = 50) -> List[str]:
        """Get all family precepts."""
        entries = await self.search_knowledge("", entry_type="家訓", limit=limit)
        
        precepts = []
        for entry in entries:
            # In a real implementation, we'd fetch the page content
            # For now, just return the titles
            precepts.append(entry["title"])
        
        return precepts
    
    # =========================================================================
    # v11.5 新機能: 積極活用メソッド
    # =========================================================================

    async def get_routing_context(self, max_chars: int = 1200) -> str:
        """
        LangGraph Router v11.5 用: 家訓 + 直近タスク要約をテキストで返す。

        fetch_context ノードから呼び出され、route_decision のヒントとして
        システムプロンプトに注入される。

        Args:
            max_chars: 返すテキストの最大文字数

        Returns:
            家訓と直近タスク要約を結合したテキスト (空文字列 = Notion 未接続)
        """
        if not self.client:
            return ""

        parts: list[str] = []

        # 家訓を取得
        try:
            precept_entries = await self.search_knowledge("", entry_type="家訓", limit=3)
            if precept_entries:
                precept_texts: list[str] = []
                for entry in precept_entries[:2]:
                    content = await self.get_page_content(entry["id"])
                    if content:
                        precept_texts.append(content[:300])
                if precept_texts:
                    parts.append("【家訓】\n" + "\n---\n".join(precept_texts))
        except Exception as e:
            logger.debug("家訓取得スキップ: %s", e)

        # 直近の完了タスクを取得
        try:
            recent = await self.search_knowledge("", entry_type="タスク完了", limit=3)
            if recent:
                recent_texts = [
                    f"- {e['title']} ({e['date'][:10] if e['date'] else ''})"
                    for e in recent
                ]
                parts.append("【直近タスク】\n" + "\n".join(recent_texts))
        except Exception as e:
            logger.debug("直近タスク取得スキップ: %s", e)

        result = "\n\n".join(parts)
        return result[:max_chars] if result else ""

    async def auto_save_task_result(
        self,
        task: str,
        result: str,
        metadata: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        タスク完了後に自動保存。LangGraph Router の persist_notion ノードから呼び出す。

        Args:
            task:     タスク説明 (タイトル用)
            result:   タスク実行結果テキスト
            metadata: ルート・エージェント役職・実行時間など

        Returns:
            作成された Notion ページ ID (失敗時は None)
        """
        if not self.client:
            return None

        meta = metadata or {}
        title = f"[{meta.get('agent_role', '?')}] {task[:60]}"
        agent_name = meta.get("agent_role", meta.get("handled_by", "unknown"))
        route = meta.get("route", "")
        exec_time = meta.get("execution_time", 0)
        tools_used = meta.get("mcp_tools_used", [])

        try:
            properties: Dict[str, Any] = {
                "Title": {
                    "title": [{"type": "text", "text": {"content": title}}]
                },
                "Type": {
                    "select": {"name": "タスク完了"}
                },
                "Date": {
                    "date": {"start": datetime.now().isoformat()}
                },
                "Status": {
                    "select": {"name": "完了"}
                },
            }
            if agent_name:
                properties["Agent"] = {"select": {"name": agent_name[:100]}}
            if meta.get("cost"):
                properties["Cost (¥)"] = {"number": meta["cost"]}
            if tools_used:
                properties["Tags"] = {
                    "multi_select": [{"name": t[:100]} for t in tools_used[:5]]
                }

            # コンテンツブロック構築
            children: list[Dict] = [
                {
                    "object": "block", "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": "タスク"}}]
                    },
                },
                {
                    "object": "block", "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": task[:2000]}}]
                    },
                },
                {
                    "object": "block", "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": "実行結果"}}]
                    },
                },
            ]
            # 結果を 1900 文字ずつブロックに分割 (Notion 上限対応)
            for chunk in [result[i:i+1900] for i in range(0, min(len(result), 9500), 1900)]:
                children.append({
                    "object": "block", "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": chunk}}]
                    },
                })

            # メタデータブロック
            meta_text = (
                f"Route: {route} | Agent: {agent_name} | "
                f"Time: {exec_time:.1f}s | Tools: {', '.join(tools_used) or 'none'}"
            )
            children.append({
                "object": "block", "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": meta_text}}],
                    "icon": {"emoji": "🤖"},
                },
            })

            page = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties,
                children=children,
            )
            self.stats["knowledge_entries"] += 1
            logger.debug("[Notion] タスク自動保存: %s", page["id"])
            return page["id"]

        except Exception as e:
            logger.warning("[Notion] タスク自動保存失敗: %s", e)
            return None

    async def get_page_content(self, page_id: str, max_chars: int = 3000) -> str:
        """
        Notion ページのブロックコンテンツを全文テキストで取得。

        既存の search_knowledge() はタイトルしか返さないが、このメソッドは
        ページ内のすべてのテキストブロックを結合して返す。

        Args:
            page_id:   ページ ID
            max_chars: 取得する最大文字数

        Returns:
            ページ内テキストの結合文字列
        """
        if not self.client:
            return ""

        try:
            blocks = self.client.blocks.children.list(block_id=page_id)
            texts: list[str] = []
            for block in blocks.get("results", []):
                block_type = block.get("type", "")
                block_data = block.get(block_type, {})
                rich_text = block_data.get("rich_text", [])
                for rt in rich_text:
                    text_content = rt.get("text", {}).get("content", "")
                    if text_content:
                        texts.append(text_content)
            combined = "\n".join(texts)
            return combined[:max_chars]
        except Exception as e:
            logger.debug("[Notion] ページコンテンツ取得失敗 (id=%s): %s", page_id, e)
            return ""

    async def update_entry(
        self,
        page_id: str,
        content: str = "",
        status: str = "",
        title: str = "",
    ) -> bool:
        """
        既存 Notion ページを更新。

        Args:
            page_id: 更新対象ページ ID
            content: 新しい本文 (空文字列の場合は更新しない)
            status:  新しいステータス (空文字列の場合は更新しない)
            title:   新しいタイトル (空文字列の場合は更新しない)

        Returns:
            更新成否
        """
        if not self.client:
            return False

        try:
            properties: Dict[str, Any] = {}
            if title:
                properties["Title"] = {
                    "title": [{"type": "text", "text": {"content": title}}]
                }
            if status:
                properties["Status"] = {"select": {"name": status}}

            if properties:
                self.client.pages.update(page_id=page_id, properties=properties)

            if content:
                # 既存ブロックの後ろに追記
                self.client.blocks.children.append(
                    block_id=page_id,
                    children=[{
                        "object": "block", "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": content[:1900]}}]
                        },
                    }],
                )

            logger.debug("[Notion] ページ更新完了: %s", page_id)
            return True
        except Exception as e:
            logger.warning("[Notion] ページ更新失敗 (id=%s): %s", page_id, e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "initialized": self.client is not None,
            "database_id": self.database_id[:10] + "..." if self.database_id else None,
            "stats": dict(self.stats),
        }
    
    def show_stats(self) -> str:
        """Format stats for display."""
        s = self.stats
        
        lines = [
            "=" * 50,
            "📁 Notion統合 統計",
            "=" * 50,
            f"60日要約保存: {s['summaries_saved']}件",
            f"家訓保存: {s['precepts_saved']}件",
            f"総知識エントリ: {s['knowledge_entries']}件",
            f"検索クエリ: {s['search_queries']}回",
            "",
            f"接続状態: {'OK' if self.client else 'NG'}",
            "=" * 50,
        ]
        return "\n".join(lines)


# Utility functions for easy integration
async def create_default_database(
    client: Client, 
    title: str = "将軍システム 知識ベース"
) -> Optional[str]:
    """Create default knowledge database in Notion."""
    try:
        # Create database with standard properties
        database = client.databases.create(
            parent={
                "type": "page_id",
                "page_id": "your-parent-page-id"  # This needs to be provided
            },
            title=[
                {
                    "type": "text",
                    "text": {"content": title}
                }
            ],
            properties={
                "Title": {
                    "title": {}
                },
                "Type": {
                    "select": {
                        "options": [
                            {"name": "60日要約", "color": "blue"},
                            {"name": "家訓", "color": "green"},
                            {"name": "知識", "color": "yellow"},
                            {"name": "エラー対応", "color": "red"},
                        ]
                    }
                },
                "Date": {
                    "date": {}
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "完了", "color": "green"},
                            {"name": "進行中", "color": "yellow"},
                            {"name": "保留", "color": "red"},
                        ]
                    }
                },
                "Cost (¥)": {
                    "number": {
                        "format": "yen"
                    }
                },
                "Agent": {
                    "select": {
                        "options": [
                            {"name": "将軍", "color": "purple"},
                            {"name": "家老", "color": "blue"},
                            {"name": "侍大将", "color": "green"},
                            {"name": "足軽", "color": "gray"},
                        ]
                    }
                },
                "Tags": {
                    "multi_select": {
                        "options": [
                            {"name": "ESP32", "color": "blue"},
                            {"name": "Home Assistant", "color": "green"},
                            {"name": "AI", "color": "purple"},
                            {"name": "Hardware", "color": "orange"},
                            {"name": "Software", "color": "pink"},
                        ]
                    }
                },
            }
        )
        
        return database["id"]
        
    except Exception as e:
        logger.error("データベース作成エラー: %s", e)
        return None
