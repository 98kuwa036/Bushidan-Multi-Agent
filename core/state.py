"""
武士団 BushidanState v15
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class BushidanState(TypedDict):
    """LangGraph v15 ステート — HITL + コンテキスト要約"""

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
    available_tools: list       # ツール名リスト
    tool_schemas: dict
    mcp_tools: list             # LangChain BaseTool オブジェクト (ロールが直接実行可)

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

    # ── v15
    context_summary: str

    # ── v15
    dialog_status: str              # "active" | "waiting_for_human" | "completed"
    awaiting_human_input: bool
    human_question: str             # エージェント → 人間への質問
    human_response: str             # 人間 → エージェントへの応答

    # ── v15: コード検証 (sandbox_verify ノード) ───────────────────────
    code_verified: bool             # 検証済みフラグ
    code_verify_result: str         # 実行結果サマリ ("ok" | "error: ..." | "skipped")

    # ── v15: 並列 Groq (parallel_groq ノード) ────────────────────────
    sub_queries: list               # 分割されたサブクエリリスト
    sub_responses: list             # 各サブクエリの応答リスト

    # ── v16: 受付 構造化インテント ────────────────────────────────────
    intent_structured: dict         # {intent_type, domain, required_capabilities, user_goal}

    # ── v16: 将軍ロードマップ実行 ─────────────────────────────────────
    roadmap: dict                   # {goal, steps:[{id,task,capability,assigned_role,status,result}], needs_audit}
    roadmap_step: int               # 現在実行中のステップインデックス
    roadmap_results: Annotated[list, operator.add]  # 各ステップ結果の累積リスト
    needs_audit: bool               # 大元帥監査フラグ

    # ── v18: ステップ実行コンテキスト ─────────────────────────────────
    _step_task: str                  # execute_step → 各ロールへの個別タスク指示
