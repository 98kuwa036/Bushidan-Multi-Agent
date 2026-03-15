"""
武士団 BushidanState v14 — LangGraph MemorySaver + HITL 対応

v14 追加フィールド:
  - context_summary: リレー間コンテキスト要約
  - dialog_status: 対話状態 ("active" | "waiting_for_human" | "completed")
  - awaiting_human_input: HITL 中断フラグ
  - human_question: エージェントから人間への質問
  - human_response: 人間からの応答
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class BushidanState(TypedDict):
    """LangGraph v14 ステート — HITL + コンテキスト要約"""

    # ── スレッド識別 ──────────────────────────────────────────────────
    thread_id: str
    channel_id: str
    user_id: str
    source: str

    # ── 現在のメッセージ ─────────────────────────────────────────────
    message: str
    attachments: list

    # ── 会話履歴 (MemorySaver でターン間持続) ────────────────────────
    conversation_history: Annotated[list, operator.add]

    # ── Notion RAG ────────────────────────────────────────────────
    notion_chunks: list

    # ── タスク分析 ────────────────────────────────────────────────────
    complexity: str
    is_multi_step: bool
    is_action_task: bool
    is_simple_qa: bool
    is_japanese_priority: bool
    is_confidential: bool

    # ── ルーティング ─────────────────────────────────────────────────
    forced_role: Optional[str]
    routed_to: Optional[str]

    # ── MCP ──────────────────────────────────────────────────────
    available_tools: list
    tool_schemas: dict

    # ── 実行結果 ─────────────────────────────────────────────────────
    response: Optional[str]
    handled_by: Optional[str]
    agent_role: Optional[str]
    execution_time: float
    error: Optional[str]
    mcp_tools_used: list

    # ── 自律ループ制御 ────────────────────────────────────────────────
    requires_followup: bool
    iteration: int
    max_iterations: int

    # ── Notion 永続化 ─────────────────────────────────────────────────
    should_save: bool
    notion_page_id: Optional[str]

    # ── v14: コンテキスト要約 ──────────────────────────────────────────
    context_summary: str

    # ── v14: Human-in-the-loop ────────────────────────────────────────
    dialog_status: str              # "active" | "waiting_for_human" | "completed"
    awaiting_human_input: bool
    human_question: str             # エージェント → 人間への質問
    human_response: str             # 人間 → エージェントへの応答
