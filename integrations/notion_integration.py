"""Notion Integration for Shogun System v7.0

Automatic knowledge management and 60-day summary storage.

Features:
  - Automatic family precepts (家訓) storage
  - 60-day summary archival
  - Knowledge base construction
  - Search and retrieval capabilities

Integration with 9th Ashigaru (Groq Recorder) for seamless knowledge flow.
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
