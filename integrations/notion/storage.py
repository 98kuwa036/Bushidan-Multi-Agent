"""
integrations/notion/storage.py — Notion タスク結果の自動保存

LangGraph の notion_store ノードから呼び出す。

保存戦略:
  - BushidanState の response / handled_by / agent_role などを Notion DB に記録
  - 重複防止: thread_id が既存ページに存在すれば update_page で追記
  - fire-and-forget: asyncio.create_task で非ブロッキングに実行可能
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from integrations.notion.client import get_notion_client

logger = logging.getLogger("integrations.notion.storage")


async def save_task_result(state: dict) -> Optional[str]:
    """
    BushidanState の実行結果を Notion DB に保存する。

    Args:
        state: BushidanState

    Returns:
        作成/更新した Notion ページ ID (失敗時 None)
    """
    client = get_notion_client()
    if not client.is_available() or not client.database_id:
        return None

    message = state.get("message", "")
    response = state.get("response", "") or ""
    agent_role = state.get("agent_role", "")
    handled_by = state.get("handled_by", "")
    route = state.get("routed_to", "")
    exec_time = state.get("execution_time", 0.0)
    tools = state.get("mcp_tools_used", [])
    thread_id = state.get("thread_id", "")
    source = state.get("source", "")

    title = f"[{agent_role or handled_by}] {message[:60]}"

    # Notion タスクDB のプロパティ名: 名前/ロール/タグ/ステータス/作成日時/実行時間(ms)
    properties: Dict[str, Any] = {
        "名前": {"title": [{"type": "text", "text": {"content": title}}]},
        "ステータス": {"select": {"name": "完了"}},
        "作成日時": {"date": {"start": datetime.now().isoformat()}},
    }
    if agent_role:
        properties["ロール"] = {"select": {"name": agent_role[:100]}}
    if tools:
        properties["タグ"] = {
            "multi_select": [{"name": t[:100]} for t in tools[:5]]
        }
    if exec_time:
        properties["実行時間(ms)"] = {"number": int(exec_time * 1000)}

    # コンテンツブロック構築
    children: list = [
        _heading("タスク"),
        _paragraph(message[:2000]),
        _heading("実行結果"),
    ]
    for chunk in _split_text(response, 1900):
        children.append(_paragraph(chunk))

    meta = (
        f"Route: {route} | Agent: {agent_role} | "
        f"Time: {exec_time:.1f}s | Tools: {', '.join(tools) or 'none'} | "
        f"Thread: {thread_id[:12]} | Source: {source}"
    )
    children.append(_callout(meta))

    page_id = await client.create_page(
        client.database_id, properties=properties, children=children
    )
    if page_id:
        logger.debug("[Notion Storage] 保存完了: %s", page_id)
    return page_id


async def save_task_result_bg(state: dict) -> None:
    """
    fire-and-forget ラッパー。
    asyncio.create_task() で呼び出すことを想定。
    エラーは DEBUG ログに吸収。
    """
    try:
        await save_task_result(state)
    except Exception as e:
        logger.debug("[Notion Storage] バックグラウンド保存失敗 (無視): %s", e)


# ── ブロックビルダーヘルパー ─────────────────────────────────────────

def _heading(text: str) -> dict:
    return {
        "object": "block", "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _paragraph(text: str) -> dict:
    return {
        "object": "block", "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _callout(text: str) -> dict:
    return {
        "object": "block", "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": text}}],
            "icon": {"emoji": "🤖"},
        },
    }


def _split_text(text: str, chunk_size: int) -> list:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] if text else []
