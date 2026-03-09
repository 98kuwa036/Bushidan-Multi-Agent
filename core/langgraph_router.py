"""
武士団 Multi-Agent System v12 - LangGraph Router (MCP + Notion 密結合)

v12 メンバー再構成: 10役職・Cohere Command R/R+ 追加

StateGraph v12:
  [START]
    ↓
  [analyze]        タスク複雑度・マルチステップ・アクション検出
    ↓
  [fetch_context]  並列: MCP ツール一覧 + Notion 家訓/直近タスク取得
    ↓
  [route_decision] ツール認識・パーミッション・Notion 知識参照ルーティング
    ↓
  ┌──────────────────────────────────────────────────────┐
  │  groq_qa          │  高速フィルタ (斥候 Llama 3.3)          │
  │  gunshi_pdca      │  深層推論 (軍師 o3-mini)                │
  │  gaiji_rag        │  外部/RAG (外事 Command R+)             │
  │  taisho_mcp       │  Tool chain (参謀 + MCP)               │
  │  yuhitsu_jp       │  日本語清書 (右筆 ELYZA)                │
  │  onmitsu_local    │  完全ローカル (隠密 Nemotron)           │
  │  karo_default     │  受付フォールバック (受付 Command R)    │
  └──────────────────────────────────────────────────────┘
    ↓
  [persist_notion]  非同期: Notion に全タスク結果を自動保存
    ↓
  [END]

v12 変更点:
  - gaiji_rag ノード追加: 外事 (Command R+) による外部ツール/RAG
  - yuhitsu_jp ノード追加: 右筆 (ELYZA) による日本語清書
  - onmitsu_local: 機密専用 (Nemotron のみ)、日本語は yuhitsu_jp に移管
  - karo_default: 受付 (Command R) へ更新
  - 受付→外事→検校→将軍→軍師→参謀→右筆→斥候→隠密→大元帥 (10役職)
"""

import asyncio
import time
from typing import Any, Literal, Optional

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# State Definition v11.5
# =============================================================================

class TaskState(TypedDict):
    """LangGraph state v11.5 — MCP ツール認識 + Notion コンテキスト付き"""

    # ── 入力 ────────────────────────────────────────────────
    content: str
    context: dict
    priority: int
    source: str

    # ── 分析結果 (analyze ノード) ─────────────────────────
    is_multi_step: bool
    is_action_task: bool
    is_simple_qa: bool
    is_japanese_priority: bool  # 日本語特化タスク → ELYZA ルート
    is_confidential: bool       # 機密タスク → Nemotron ルート
    complexity: str           # "simple" | "medium" | "complex" | "strategic"
    confidence: float

    # ── コンテキスト取得 (fetch_context ノード) ─────────────
    available_tools: list     # 実行中 MCP サーバーのツール名一覧
    tool_schemas: dict        # ツール名 → スキーマの辞書
    notion_context: str       # 家訓 + 直近タスク要約（LLM プロンプト用）

    # ── ルーティング決定 ─────────────────────────────────
    route: str                # 選択された実行ルート

    # ── 実行結果 ─────────────────────────────────────────
    result: Optional[dict]
    status: str               # "pending" | "executing" | "completed" | "failed"
    handled_by: str
    agent_role: str           # 実際に処理したエージェント役職
    execution_time: float
    error: Optional[str]
    mcp_tools_used: list      # このタスクで呼び出した MCP ツール名

    # ── Notion 永続化 ─────────────────────────────────────
    notion_page_id: Optional[str]


# =============================================================================
# LangGraph Router v11.5
# =============================================================================

class LangGraphRouter:
    """
    LangGraph StateGraph ベースのタスクルーター v11.5

    変更点 (v10.x → v11.5):
      - fetch_context ノード追加: MCP ツール一覧 + Notion 家訓を事前取得
      - route_decision がツール認識・Notion 参照ベースに進化
      - gunshi_pdca ルート追加: COMPLEX/STRATEGIC タスクを軍師 o3-mini に委譲
      - persist_notion ノード追加: 全タスク結果を Notion へ自動保存
      - MCPPermissionManager との統合
    """

    def __init__(self, orchestrator: "SystemOrchestrator"):
        self.orchestrator = orchestrator
        self._graph = None
        self._compiled = None
        self._auto_save_notion = True  # Notion 自動保存フラグ

    async def initialize(self) -> None:
        """LangGraph ルーターを初期化."""
        logger.info("🔗 LangGraph Router v11.5 初期化中...")
        try:
            self._graph = self._build_graph()
            self._compiled = self._graph.compile()
            logger.info("✅ LangGraph Router v11.5 初期化完了")
        except Exception as e:
            import traceback
            logger.error("❌ LangGraph Router 初期化エラー: %s\n%s", e, traceback.format_exc())
            raise

    def _build_graph(self) -> StateGraph:
        """v11.5 StateGraph を構築."""
        graph = StateGraph(TaskState)

        # ── ノード登録 ────────────────────────────────────
        graph.add_node("analyze",          self._analyze_task)
        graph.add_node("fetch_context",    self._fetch_context)
        graph.add_node("groq_qa",          self._execute_groq_qa)           # 斥候 Llama 3.3
        graph.add_node("gunshi_pdca",      self._execute_gunshi_pdca)       # 軍師 o3-mini PDCA
        graph.add_node("gaiji_rag",        self._execute_gaiji_rag)         # v12: 外事 Command R+
        graph.add_node("taisho_mcp",       self._execute_taisho_mcp)        # MCP ツール連携
        graph.add_node("yuhitsu_jp",       self._execute_yuhitsu_jp)        # v12: 右筆 ELYZA
        graph.add_node("karo_default",     self._execute_karo_default)      # 受付 Command R フォールバック
        graph.add_node("onmitsu_local",    self._execute_onmitsu_local)     # 隠密 Nemotron (機密専用)
        graph.add_node("persist_notion",   self._persist_notion)

        # ── エントリーポイント ───────────────────────────
        graph.set_entry_point("analyze")

        # analyze → fetch_context (直通)
        graph.add_edge("analyze", "fetch_context")

        # fetch_context → 各実行ルート (条件分岐) v12
        graph.add_conditional_edges(
            "fetch_context",
            self._route_decision,
            {
                "groq_qa":       "groq_qa",
                "gunshi_pdca":   "gunshi_pdca",
                "gaiji_rag":     "gaiji_rag",
                "taisho_mcp":    "taisho_mcp",
                "yuhitsu_jp":    "yuhitsu_jp",
                "karo_default":  "karo_default",
                "onmitsu_local": "onmitsu_local",
            },
        )

        # 全実行ルート → persist_notion → END
        for node in ("groq_qa", "gunshi_pdca", "gaiji_rag", "taisho_mcp",
                     "yuhitsu_jp", "karo_default", "onmitsu_local"):
            graph.add_edge(node, "persist_notion")
        graph.add_edge("persist_notion", END)

        return graph

    # =========================================================================
    # Node: analyze — タスク特性の分析
    # =========================================================================

    def _analyze_task(self, state: TaskState) -> dict:
        """タスクの複雑度・種別を分析してルーティング用メタデータを付与."""
        content = state["content"]
        logger.info("📊 [analyze] タスク分析中... (%s...)", content[:60])

        from core.multi_step_task_detector import MultiStepTaskDetector
        detector = MultiStepTaskDetector()
        analysis = detector.analyze(content)
        is_multi_step = analysis.is_multi_step and analysis.confidence >= 0.6

        is_action_task        = self._detect_action_task(content)
        is_simple_qa          = self._detect_simple_qa(content) and not is_action_task
        is_japanese_priority  = self._detect_japanese_priority(content)
        is_confidential       = self._detect_confidential(content)
        complexity            = self._assess_complexity(content)

        logger.info(
            "📊 分析結果: multi_step=%s action=%s simple_qa=%s "
            "japanese=%s confidential=%s complexity=%s",
            is_multi_step, is_action_task, is_simple_qa,
            is_japanese_priority, is_confidential, complexity,
        )
        return {
            "is_multi_step": is_multi_step,
            "is_action_task": is_action_task,
            "is_simple_qa": is_simple_qa,
            "is_japanese_priority": is_japanese_priority,
            "is_confidential": is_confidential,
            "complexity": complexity,
            "confidence": analysis.confidence,
            "status": "analyzed",
        }

    def _detect_action_task(self, content: str) -> bool:
        """単一アクションタスク (git push など) を検出."""
        content_lower = content.lower()
        action_patterns = [
            "clone ", "クローン", "git clone",
            "push ", "プッシュ", "git push",
            "pull ", "プル", "git pull",
            "install ", "インストール", "npm install", "pip install",
            "run ", "実行して", "execute",
            "delete ", "削除して",
            "create ", "作成して",
            "search ", "検索して", "調べて",
        ]
        for pattern in action_patterns:
            if pattern in content_lower:
                action_count = sum(1 for p in action_patterns if p in content_lower)
                if action_count <= 2:  # 単純アクション (2つ以内)
                    return True
        return False

    def _detect_simple_qa(self, content: str) -> bool:
        """シンプルな Q&A タスクを検出."""
        content_lower = content.lower()
        qa_patterns = [
            "とは", "って何", "ですか", "でしょうか", "教えて", "説明", "意味",
            "what is", "what's", "how does", "explain", "tell me",
        ]
        for pattern in qa_patterns:
            if pattern in content_lower:
                return True
        # アクションキーワードなしの短文
        if len(content) < 80:
            action_kws = ["clone", "push", "pull", "create", "delete", "edit", "install", "deploy"]
            if not any(kw in content_lower for kw in action_kws):
                return True
        return False

    def _detect_japanese_priority(self, content: str) -> bool:
        """
        日本語特化タスクを検出 → ELYZA ルート候補。

        「全体が日本語」ではなく「日本語品質が重要なタスク」を対象にする。
        単なる問い合わせ (ですか / とは) は groq_qa で十分。
        """
        japanese_quality_patterns = [
            # 日本語生成・ライティング
            "日本語で書いて", "日本語で回答", "日本語で作成",
            "和文", "日本語テキスト", "日本語文章",
            # 翻訳
            "日本語に翻訳", "和訳", "translate.*japanese", "japanese.*translat",
            # 校正・添削
            "添削", "校正", "文法チェック", "文章を直して",
            # 創作
            "小説", "俳句", "短歌", "詩を書", "物語を書",
            # メール・ビジネス文書
            "ビジネスメール", "敬語で", "丁寧な日本語",
        ]
        content_lower = content.lower()
        for pattern in japanese_quality_patterns:
            if pattern in content_lower or pattern in content:
                return True
        return False

    def _detect_confidential(self, content: str) -> bool:
        """機密・ローカル処理タスクを検出 → Nemotron ルート。"""
        confidential_patterns = [
            "機密", "秘密", "confidential", "private", "internal",
            "オフライン", "offline", "local only", "api送信不可",
            "社外秘", "外部送信禁止",
        ]
        content_lower = content.lower()
        return any(p in content_lower or p in content for p in confidential_patterns)

    def _assess_complexity(self, content: str) -> str:
        """タスク複雑度をヒューリスティックで判定."""
        strategic_kws = [
            "設計", "アーキテクチャ", "システム全体", "戦略", "提案",
            "design", "architecture", "strategy", "roadmap", "full system",
        ]
        complex_kws = [
            "実装して", "作って", "修正して", "リファクタ", "テスト",
            "implement", "build", "refactor", "fix", "debug",
        ]
        if any(kw in content.lower() for kw in strategic_kws) or len(content) > 800:
            return "strategic"
        if any(kw in content.lower() for kw in complex_kws) or len(content) > 300:
            return "complex"
        if len(content) < 60:
            return "simple"
        return "medium"

    # =========================================================================
    # Node: fetch_context — MCP ツール + Notion コンテキスト取得 (v11.5 新規)
    # =========================================================================

    async def _fetch_context(self, state: TaskState) -> dict:
        """
        MCP ツール一覧と Notion コンテキストを並列取得。

        ルーター判断に必要な外部情報をここで一括収集することで、
        route_decision ノードが全情報を持った状態で動作できる。
        """
        logger.info("🔍 [fetch_context] MCP ツール + Notion コンテキスト取得中...")

        mcp_result, notion_result = await asyncio.gather(
            self._get_mcp_tool_info(),
            self._get_notion_context(),
            return_exceptions=True,
        )

        available_tools: list = []
        tool_schemas: dict = {}
        if isinstance(mcp_result, dict):
            tool_schemas = mcp_result
            for tools in mcp_result.values():
                available_tools.extend(tools)

        notion_context: str = ""
        if isinstance(notion_result, str):
            notion_context = notion_result

        logger.info(
            "🔍 取得完了: tools=%d種類, notion=%d文字",
            len(available_tools), len(notion_context),
        )
        return {
            "available_tools": available_tools,
            "tool_schemas":    tool_schemas,
            "notion_context":  notion_context,
        }

    async def _get_mcp_tool_info(self) -> dict:
        """
        MCP Bridge (langchain-mcp-adapters) からツールスキーマ一覧を取得。
        Bridge 未使用時は旧 MCPManager にフォールバック。
        """
        try:
            bridge = getattr(self.orchestrator, "_mcp_bridge", None)
            if bridge and bridge.is_available():
                return bridge.get_tool_schemas()

            # フォールバック: 旧 MCPManager
            mcp_manager = getattr(self.orchestrator, "mcp_manager", None)
            if mcp_manager and hasattr(mcp_manager, "list_tools"):
                return mcp_manager.list_tools()
            return {}
        except Exception as e:
            logger.debug("MCP ツール取得スキップ: %s", e)
            return {}

    async def _get_notion_context(self) -> str:
        """Notion から家訓と直近タスク要約を取得してルーター用テキストを生成."""
        try:
            notion: Optional["NotionIntegration"] = getattr(
                self.orchestrator, "notion", None
            )
            if notion and hasattr(notion, "get_routing_context"):
                return await notion.get_routing_context()
            return ""
        except Exception as e:
            logger.debug("Notion コンテキスト取得スキップ: %s", e)
            return ""

    # =========================================================================
    # Routing Decision — ツール認識 + Notion 参照 (v11.5 強化)
    # =========================================================================

    def _route_decision(self, state: TaskState) -> str:
        """
        v12 ルーティング判断 (10役職対応)。

        優先度:
          0. 強制ルーティング (@メンション時)
          1. 機密タスク → onmitsu_local (隠密 Nemotron)
          2. 日本語清書タスク → yuhitsu_jp (右筆 ELYZA)
          3. 戦略/複雑 → gunshi_pdca (軍師 o3-mini PDCA)
          4. ツール連携アクション → taisho_mcp (参謀 + MCP tools)
          5. 外部/RAG/マルチステップ → gaiji_rag (外事 Command R+)
          6. シンプル Q&A / 高速 → groq_qa (斥候 Llama 3.3)
          7. デフォルト → karo_default (受付 Command R)
        """
        content         = state["content"]
        tools           = state.get("available_tools", [])
        complexity      = state.get("complexity", "medium")
        is_multi        = state.get("is_multi_step", False)
        is_action       = state.get("is_action_task", False)
        is_simple_qa    = state.get("is_simple_qa", False)
        is_japanese     = state.get("is_japanese_priority", False)
        is_confidential = state.get("is_confidential", False)

        # ── 強制ルーティング (特定エージェントへの直接メンション時) ─────────
        _valid_routes = {
            "groq_qa", "gunshi_pdca", "gaiji_rag", "taisho_mcp",
            "yuhitsu_jp", "karo_default", "onmitsu_local",
            # 後方互換 (旧ノード名)
            "gemini_autonomous",
        }
        forced_route = state.get("context", {}).get("forced_route")
        if forced_route in _valid_routes:
            # 旧名 gemini_autonomous → gaiji_rag に透過的にリダイレクト
            if forced_route == "gemini_autonomous":
                forced_route = "gaiji_rag"
            logger.info("🎯 Route: %s (強制ルーティング)", forced_route)
            return forced_route

        # ── 1. 機密タスク → 隠密 (Nemotron ローカル処理) ────────────────────
        if is_confidential:
            logger.info("🥷 Route: onmitsu_local (機密 → 隠密 Nemotron)")
            return "onmitsu_local"

        # ── 2. 日本語清書タスク → 右筆 (ELYZA) ──────────────────────────────
        if is_japanese:
            logger.info("🖊️ Route: yuhitsu_jp (日本語清書 → 右筆 ELYZA)")
            return "yuhitsu_jp"

        # ── 3. 戦略的 / 複雑タスク → 軍師 PDCA ─────────────────────────────
        if complexity in ("strategic", "complex") or (is_multi and complexity != "simple"):
            logger.info(
                "🧠 Route: gunshi_pdca (complexity=%s, multi=%s)", complexity, is_multi
            )
            return "gunshi_pdca"

        # ── 4. ツール特化アクション → taisho_mcp ─────────────────────────────
        if is_action and tools:
            content_lower = content.lower()
            tool_routes = {
                "github":     ["git", "github", "commit", "push", "pull", "pr", "issue"],
                "filesystem": ["ファイル", "file", "read", "write", "directory"],
                "tavily":     ["search", "検索", "web", "調べ"],
                "exa":        ["semantic", "意味検索"],
                "playwright": ["browser", "ブラウザ", "screenshot", "スクリーン"],
                "mattermost": ["mattermost", "channel", "チャンネル", "投稿"],
            }
            for tool_name, keywords in tool_routes.items():
                if tool_name in tools and any(kw in content_lower for kw in keywords):
                    logger.info("🔧 Route: taisho_mcp (tool=%s)", tool_name)
                    return "taisho_mcp"

        # ── 5. マルチステップ / 外部情報収集 → 外事 (Command R+) ────────────
        if is_multi or is_action:
            logger.info("🌐 Route: gaiji_rag (外部/RAG → 外事 Command R+)")
            return "gaiji_rag"

        # ── 6. シンプル Q&A → 斥候 Groq (即応・無料) ────────────────────────
        if is_simple_qa and complexity == "simple":
            logger.info("⚡ Route: groq_qa (simple Q&A → 斥候 Llama 3.3)")
            return "groq_qa"

        # ── 7. デフォルト → 受付 (Command R) ────────────────────────────────
        logger.info("🏯 Route: karo_default (受付 Command R)")
        return "karo_default"

    # =========================================================================
    # Execution Nodes
    # =========================================================================

    async def _execute_groq_qa(self, state: TaskState) -> dict:
        """Simple Q&A を家老-B Groq (Llama 3.3 70B) で即時処理."""
        start = time.time()
        try:
            groq_client = self.orchestrator.get_client("groq")
            if not groq_client:
                return await self._execute_karo_default(state)

            # Notion コンテキストをシステムプロンプトに注入
            system_prompt = "あなたは簡潔で正確な回答を提供するアシスタントです。"
            if state.get("notion_context"):
                system_prompt += f"\n\n【参照知識】\n{state['notion_context'][:500]}"

            response = await groq_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": state["content"]},
                ],
                max_tokens=512,
            )
            return {
                "result": {"response": response, "status": "completed"},
                "status": "completed",
                "handled_by": "groq_qa",
                "agent_role": "家老-B",
                "execution_time": time.time() - start,
                "route": "groq_qa",
                "mcp_tools_used": [],
            }
        except Exception as e:
            logger.error("❌ Groq 実行失敗: %s", e)
            return {
                "status": "failed", "error": str(e),
                "handled_by": "groq_qa", "agent_role": "家老-B",
                "execution_time": time.time() - start, "mcp_tools_used": [],
            }

    async def _execute_gunshi_pdca(self, state: TaskState) -> dict:
        """Complex/Strategic タスクを軍師 (o3-mini) PDCA → 参謀A/B に委譲 (v11.5 新規)."""
        start = time.time()
        try:
            gunshi = getattr(self.orchestrator, "gunshi", None)
            if not gunshi:
                logger.warning("⚠️ 軍師未初期化 → karo_default にフォールバック")
                return await self._execute_karo_default(state)

            # Notion コンテキストをタスクに追加
            task_content = state["content"]
            if state.get("notion_context"):
                task_content = (
                    f"{task_content}\n\n"
                    f"【Notion 知識ベース参照】\n{state['notion_context'][:800]}"
                )

            context = {
                **state.get("context", {}),
                "available_tools": state.get("available_tools", []),
                "langgraph_route": "gunshi_pdca",
            }

            result = await gunshi.process_task(task_content, context)

            return {
                "result": result,
                "status": result.get("status", "completed"),
                "handled_by": "gunshi_pdca",
                "agent_role": "軍師",
                "execution_time": time.time() - start,
                "route": "gunshi_pdca",
                "mcp_tools_used": result.get("mcp_tools_used", []),
            }
        except Exception as e:
            logger.error("❌ 軍師 PDCA 実行失敗: %s", e)
            return {
                "status": "failed", "error": str(e),
                "handled_by": "gunshi_pdca", "agent_role": "軍師",
                "execution_time": time.time() - start, "mcp_tools_used": [],
            }

    async def _execute_gemini_autonomous(self, state: TaskState) -> dict:
        """マルチステップタスクを Gemini Flash 自律実行器で処理."""
        start = time.time()
        try:
            karo = getattr(self.orchestrator, "karo", None)
            if not karo or not getattr(karo, "gemini_autonomous_executor", None):
                return await self._execute_karo_default(state)

            executor = karo.gemini_autonomous_executor
            exec_result = await executor.execute_autonomous_task(
                task_content=state["content"],
                max_iterations=5,
            )

            if exec_result.status == "failed":
                return await self._execute_taisho_mcp(state)

            return {
                "result": {
                    "response": exec_result.final_result,
                    "status": "completed",
                    "steps_executed": exec_result.steps_executed,
                    "tool_calls": len(exec_result.outputs),
                },
                "status": "completed",
                "handled_by": "gemini_autonomous",
                "agent_role": "家老-A",
                "execution_time": time.time() - start,
                "route": "gemini_autonomous",
                "mcp_tools_used": exec_result.tool_calls_made or [],
            }
        except Exception as e:
            logger.error("❌ Gemini 自律実行失敗: %s", e)
            return {
                "status": "failed", "error": str(e),
                "handled_by": "gemini_autonomous", "agent_role": "家老-A",
                "execution_time": time.time() - start, "mcp_tools_used": [],
            }

    # =========================================================================
    # Node: gaiji_rag — 外事 Command R+ (外部ツール・RAG) v12 新規
    # =========================================================================

    async def _execute_gaiji_rag(self, state: TaskState) -> dict:
        """
        外事 (Command R+) による外部情報収集・RAG・マルチステップ処理。

        Command R+ はグラウンディング・RAG 特化モデルなので、
        Web 検索 / 外部ドキュメント参照タスクに最適。
        未設定時は groq_qa にフォールバック。
        """
        start = time.time()
        try:
            from utils.cohere_client import create_gaiji_client

            client = create_gaiji_client()
            if not client:
                logger.warning("⚠️ 外事クライアント未設定 → groq_qa フォールバック")
                return await self._execute_groq_qa(state)

            notion_ctx = state.get("notion_context", "")
            system_prompt = (
                "あなたは外事担当 (Command R+)。外部情報収集・RAG・マルチステップ処理の専門家です。"
                "正確で包括的な回答を日本語で提供してください。"
            )
            if notion_ctx:
                system_prompt += f"\n\n【Notion 知識】\n{notion_ctx[:600]}"

            response = await client.generate(
                messages=[{"role": "user", "content": state["content"]}],
                max_tokens=2048,
                system_prompt=system_prompt,
            )
            return {
                "result": {"response": response, "status": "completed"},
                "status": "completed",
                "handled_by": "gaiji_rag",
                "agent_role": "外事",
                "execution_time": time.time() - start,
                "route": "gaiji_rag",
                "mcp_tools_used": [],
            }
        except Exception as e:
            logger.error("❌ 外事 (Command R+) 実行失敗: %s", e)
            return {
                "status": "failed", "error": str(e),
                "handled_by": "gaiji_rag", "agent_role": "外事",
                "execution_time": time.time() - start, "mcp_tools_used": [],
            }

    async def _execute_taisho_mcp(self, state: TaskState) -> dict:
        """MCP ツールを活用したアクションタスクを Taisho + ツールで実行 (v11.5 新規)."""
        start = time.time()
        used_tools: list = []
        try:
            karo = getattr(self.orchestrator, "karo", None)
            if not karo:
                return await self._execute_karo_default(state)

            from core.shogun import Task, TaskComplexity
            task = Task(
                content=state["content"],
                complexity=TaskComplexity.MEDIUM,
                context={
                    **state.get("context", {}),
                    "available_tools": state.get("available_tools", []),
                    "tool_schemas":    state.get("tool_schemas", {}),
                },
                priority=state.get("priority", 1),
            )

            result = await karo._execute_with_taisho(task, None)
            used_tools = result.get("mcp_tools_used", [])

            return {
                "result": result,
                "status": result.get("status", "completed"),
                "handled_by": "taisho_mcp",
                "agent_role": "大将",
                "execution_time": time.time() - start,
                "route": "taisho_mcp",
                "mcp_tools_used": used_tools,
            }
        except Exception as e:
            logger.error("❌ Taisho MCP 実行失敗: %s", e)
            return {
                "status": "failed", "error": str(e),
                "handled_by": "taisho_mcp", "agent_role": "大将",
                "execution_time": time.time() - start, "mcp_tools_used": used_tools,
            }

    async def _execute_karo_default(self, state: TaskState) -> dict:
        """デフォルト家老ルーティング (既存ロジックへのフォールバック)."""
        start = time.time()
        try:
            karo = getattr(self.orchestrator, "karo", None)
            if not karo:
                return {"status": "failed", "error": "家老未初期化",
                        "handled_by": "karo_default", "execution_time": time.time() - start,
                        "mcp_tools_used": []}

            from core.shogun import Task, TaskComplexity
            task = Task(
                content=state["content"],
                complexity=TaskComplexity.MEDIUM,
                context=state.get("context", {}),
                priority=state.get("priority", 1),
            )
            result = await karo.execute_task_with_routing(task, None)
            return {
                "result": result,
                "status": result.get("status", "completed"),
                "handled_by": "karo_default",
                "agent_role": "家老",
                "execution_time": time.time() - start,
                "route": "karo_default",
                "mcp_tools_used": [],
            }
        except Exception as e:
            logger.error("❌ 家老デフォルト実行失敗: %s", e)
            return {
                "status": "failed", "error": str(e),
                "handled_by": "karo_default", "agent_role": "家老",
                "execution_time": time.time() - start, "mcp_tools_used": [],
            }

    # =========================================================================
    # Node: yuhitsu_jp — 右筆 ELYZA (日本語清書) v12 新規
    # =========================================================================

    async def _execute_yuhitsu_jp(self, state: TaskState) -> dict:
        """
        右筆 (ELYZA) による日本語清書・翻訳・添削処理。

        日本語品質が重要なタスク専用ノード。
        ELYZA 未起動時は Nemotron フォールバック → groq_qa フォールバック。
        """
        start = time.time()
        notion_ctx = state.get("notion_context", "")
        try:
            from utils.local_model_manager import get_local_model_manager

            manager = get_local_model_manager()
            elyza = await manager.get_japanese_client()

            if elyza:
                logger.info("🖊️ 右筆 ELYZA: 日本語清書タスク処理")
                result_text = await elyza.generate_japanese(
                    state["content"], context=notion_ctx
                )
                agent_role = "右筆-ELYZA"
                handled_by = "yuhitsu_elyza"
            else:
                # ELYZA 未起動 → Nemotron フォールバック
                logger.info("🥷 ELYZA未起動 → 右筆 Nemotron フォールバック")
                from utils.nemotron_llamacpp_client import NemotronLlamaCppClient
                client = NemotronLlamaCppClient()
                system = (
                    "あなたは日本語テキスト清書アシスタント (右筆) です。"
                    "自然で流暢な日本語で回答してください。"
                )
                result_text = await client.generate([
                    {"role": "system", "content": system},
                    {"role": "user",   "content": state["content"]},
                ])
                agent_role = "右筆-Nemotron"
                handled_by = "yuhitsu_nemotron"

            return {
                "result": {"response": result_text, "status": "completed"},
                "status": "completed",
                "handled_by": handled_by,
                "agent_role": agent_role,
                "execution_time": time.time() - start,
                "route": "yuhitsu_jp",
                "mcp_tools_used": [],
            }
        except Exception as e:
            logger.error("❌ 右筆 (ELYZA) 実行失敗: %s → groq_qa フォールバック", e)
            return await self._execute_groq_qa(state)

    # =========================================================================
    # Node: onmitsu_local — 隠密 Nemotron (機密専用) v12: 日本語は yuhitsu_jp へ
    # =========================================================================

    async def _execute_onmitsu_local(self, state: TaskState) -> dict:
        """
        隠密 (Nemotron) — 機密・完全ローカル処理専用 v12

        v12 変更: 日本語特化は yuhitsu_jp ノードに移管。
        このノードは機密データ処理のみ担当 (API 送信不可)。
        """
        start = time.time()
        notion_ctx = state.get("notion_context", "")

        try:
            from utils.nemotron_llamacpp_client import NemotronLlamaCppClient

            logger.info("🥷 隠密 Nemotron: 機密タスク処理 (完全ローカル)")
            client = NemotronLlamaCppClient()
            result_text = await client.process_confidential(
                state["content"], task_type="confidential"
            )
            agent_role = "隠密-Nemotron"
            handled_by = "onmitsu_nemotron"

            return {
                "result": {"response": result_text, "status": "completed"},
                "status": "completed",
                "handled_by": handled_by,
                "agent_role": agent_role,
                "execution_time": time.time() - start,
                "route": "onmitsu_local",
                "mcp_tools_used": [],
            }

        except Exception as e:
            logger.error("❌ onmitsu_local 実行失敗: %s", e)
            return {
                "status": "failed", "error": str(e),
                "handled_by": "onmitsu_local", "agent_role": "隠密",
                "execution_time": time.time() - start, "mcp_tools_used": [],
            }

    # =========================================================================
    # Node: persist_notion — 自動 Notion 永続化 (v11.5 新規)
    # =========================================================================

    async def _persist_notion(self, state: TaskState) -> dict:
        """
        タスク完了後に Notion へ結果を自動保存 (fire-and-forget)。

        応答遅延を避けるため、保存処理は非同期バックグラウンドタスクとして
        起動してすぐに制御を返す。Notion 保存の成否は応答に影響しない。
        """
        if not self._auto_save_notion:
            return {"notion_page_id": None}

        notion = getattr(self.orchestrator, "notion", None)
        if not notion or not hasattr(notion, "auto_save_task_result"):
            return {"notion_page_id": None}

        # fire-and-forget: Notion 保存を非同期で開始して即座に返す
        asyncio.create_task(
            self._save_to_notion_bg(state, notion),
            name=f"notion_save_{state.get('content', '')[:30]}",
        )
        return {"notion_page_id": "pending"}

    async def _save_to_notion_bg(
        self,
        state: TaskState,
        notion: Any,
    ) -> None:
        """バックグラウンドで Notion に保存."""
        try:
            result_dict = state.get("result") or {}
            content_str = ""
            if isinstance(result_dict, dict):
                content_str = result_dict.get("response", result_dict.get("result", ""))
            else:
                content_str = str(result_dict)

            page_id = await notion.auto_save_task_result(
                task=state["content"],
                result=content_str,
                metadata={
                    "route":          state.get("route", ""),
                    "agent_role":     state.get("agent_role", ""),
                    "handled_by":     state.get("handled_by", ""),
                    "complexity":     state.get("complexity", ""),
                    "execution_time": state.get("execution_time", 0),
                    "mcp_tools_used": state.get("mcp_tools_used", []),
                    "source":         state.get("source", ""),
                },
            )
            logger.debug("✅ Notion 保存完了: page_id=%s", page_id)
        except Exception as e:
            logger.debug("Notion バックグラウンド保存失敗 (無視): %s", e)

    # =========================================================================
    # Public API
    # =========================================================================

    async def process_task(
        self,
        content: str,
        context: dict = None,
        priority: int = 1,
        source: str = "api",
    ) -> dict:
        """
        タスクを v11.5 LangGraph ルーターで処理。

        Args:
            content:  タスク説明
            context:  コンテキスト辞書 (source, mode など)
            priority: タスク優先度 1-5
            source:   呼び出し元 (discord, mattermost, api など)

        Returns:
            status, result, handled_by, route, execution_time などを含む辞書
        """
        if not self._compiled:
            await self.initialize()

        initial_state: TaskState = {
            "content":           content,
            "context":           context or {},
            "priority":          priority,
            "source":            source,
            # 分析結果 (analyze ノードで設定)
            "is_multi_step":     False,
            "is_action_task":    False,
            "is_simple_qa":      False,
            "complexity":        "medium",
            "confidence":        0.0,
            # コンテキスト (fetch_context ノードで設定)
            "available_tools":   [],
            "tool_schemas":      {},
            "notion_context":    "",
            # ルーティング・実行
            "route":             "",
            "result":            None,
            "status":            "pending",
            "handled_by":        "",
            "agent_role":        "",
            "execution_time":    0.0,
            "error":             None,
            "mcp_tools_used":    [],
            # Notion
            "notion_page_id":    None,
        }

        logger.info("🔗 LangGraph v11.5: タスク処理開始 (%s...)", content[:60])
        start = time.time()

        try:
            final_state = await self._compiled.ainvoke(initial_state)
            total_time = time.time() - start

            logger.info(
                "✅ LangGraph v11.5: 完了 route=%s agent=%s time=%.2fs tools=%s",
                final_state.get("route"),
                final_state.get("agent_role"),
                total_time,
                final_state.get("mcp_tools_used", []),
            )

            result = final_state.get("result") or {}
            if isinstance(result, dict):
                response = result.get("response", result.get("result", ""))
            else:
                response = str(result)

            return {
                "status":         final_state.get("status", "completed"),
                "result":         response,
                "response":       response,
                "handled_by":     final_state.get("handled_by", "unknown"),
                "agent_role":     final_state.get("agent_role", ""),
                "route":          final_state.get("route", "unknown"),
                "complexity":     final_state.get("complexity", ""),
                "available_tools": final_state.get("available_tools", []),
                "mcp_tools_used": final_state.get("mcp_tools_used", []),
                "notion_page_id": final_state.get("notion_page_id"),
                "execution_time": total_time,
                "langgraph":      True,
                "version":        "11.5",
            }

        except Exception as e:
            logger.exception("❌ LangGraph v11.5 処理失敗: %s", e)
            return {
                "status":         "failed",
                "error":          str(e),
                "handled_by":     "langgraph_router",
                "execution_time": time.time() - start,
            }
