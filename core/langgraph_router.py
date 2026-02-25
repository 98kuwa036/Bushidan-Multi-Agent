"""
Bushidan Multi-Agent System v10.2 - LangGraph Router
ハイブリッドアプローチ: LangGraph で状態管理・ルーティング、MCP 実行は既存コード流用

StateGraph:
  [START] → [analyze] → [route]
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
  [groq_qa]    [gemini_autonomous]   [taisho_action]
        ↓                 ↓                 ↓
        └─────────────────┴─────────────────┘
                          ↓
                       [END]
"""

from typing import TypedDict, Literal, Any, Optional
from dataclasses import dataclass
import time

from langgraph.graph import StateGraph, END

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# State Definition
# =============================================================================

class TaskState(TypedDict):
    """LangGraph state for task routing"""
    # Input
    content: str
    context: dict
    priority: int
    source: str

    # Analysis results
    is_multi_step: bool
    is_action_task: bool
    is_simple_qa: bool
    complexity: str  # "simple", "medium", "strategic"
    confidence: float

    # Routing decision
    route: str  # "groq_qa", "gemini_autonomous", "taisho_action", "karo_default"

    # Execution result
    result: Optional[dict]
    status: str  # "pending", "executing", "completed", "failed"
    handled_by: str
    execution_time: float
    error: Optional[str]


# =============================================================================
# LangGraph Router
# =============================================================================

class LangGraphRouter:
    """
    LangGraph StateGraph-based task router.

    Replaces: Shogun._assess_complexity + Karo._determine_delegation
    Keeps: MCP tool execution (Taisho, GeminiFlashAutonomousExecutor)
    """

    def __init__(self, orchestrator: "SystemOrchestrator"):
        self.orchestrator = orchestrator
        self._graph = None
        self._compiled = None

    async def initialize(self):
        """Initialize the LangGraph router"""
        logger.info("🔗 LangGraph Router 初期化中...")

        # Build the graph
        self._graph = self._build_graph()
        self._compiled = self._graph.compile()

        logger.info("✅ LangGraph Router 初期化完了")

    def _build_graph(self) -> StateGraph:
        """Build the StateGraph for task routing"""

        # Create graph with state schema
        graph = StateGraph(TaskState)

        # Add nodes
        graph.add_node("analyze", self._analyze_task)
        graph.add_node("groq_qa", self._execute_groq_qa)
        graph.add_node("gemini_autonomous", self._execute_gemini_autonomous)
        graph.add_node("taisho_action", self._execute_taisho_action)
        graph.add_node("karo_default", self._execute_karo_default)

        # Set entry point
        graph.set_entry_point("analyze")

        # Add conditional edges from analyze
        graph.add_conditional_edges(
            "analyze",
            self._route_decision,
            {
                "groq_qa": "groq_qa",
                "gemini_autonomous": "gemini_autonomous",
                "taisho_action": "taisho_action",
                "karo_default": "karo_default",
            }
        )

        # All execution nodes go to END
        graph.add_edge("groq_qa", END)
        graph.add_edge("gemini_autonomous", END)
        graph.add_edge("taisho_action", END)
        graph.add_edge("karo_default", END)

        return graph

    # =========================================================================
    # Node: Analyze Task
    # =========================================================================

    def _analyze_task(self, state: TaskState) -> dict:
        """Analyze task to determine routing"""
        content = state["content"]

        logger.info(f"📊 LangGraph: タスク分析中... ({content[:50]}...)")

        # Multi-step detection
        from core.multi_step_task_detector import MultiStepTaskDetector
        detector = MultiStepTaskDetector()
        analysis = detector.analyze(content)
        is_multi_step = analysis.is_multi_step and analysis.confidence >= 0.6

        # Action task detection (single action like "clone X")
        is_action_task = self._detect_action_task(content)

        # Simple Q&A detection
        is_simple_qa = self._detect_simple_qa(content) and not is_action_task

        # Complexity heuristics
        complexity = self._assess_complexity(content)

        logger.info(
            f"📊 分析結果: multi_step={is_multi_step}, action={is_action_task}, "
            f"simple_qa={is_simple_qa}, complexity={complexity}"
        )

        return {
            "is_multi_step": is_multi_step,
            "is_action_task": is_action_task,
            "is_simple_qa": is_simple_qa,
            "complexity": complexity,
            "confidence": analysis.confidence,
            "status": "analyzed",
        }

    def _detect_action_task(self, content: str) -> bool:
        """Detect if task is a single action (clone, push, etc.)"""
        content_lower = content.lower()
        action_patterns = [
            "clone ", "クローン", "git clone",
            "push ", "プッシュ", "git push",
            "pull ", "プル", "git pull",
            "install ", "インストール",
            "run ", "実行して",
            "delete ", "削除して",
            "create ", "作成して",
        ]

        # Check if content starts with or contains action pattern
        for pattern in action_patterns:
            if pattern in content_lower:
                # But not if combined with other actions (multi-step)
                action_count = sum(1 for p in action_patterns if p in content_lower)
                if action_count == 1:
                    return True
        return False

    def _detect_simple_qa(self, content: str) -> bool:
        """Detect if task is a simple Q&A (no action required)"""
        content_lower = content.lower()

        # Question indicators
        qa_patterns = [
            "とは", "って何", "ですか", "でしょうか",
            "what is", "what's", "how does", "explain",
            "教えて", "説明", "意味",
        ]

        # Check for question patterns
        for pattern in qa_patterns:
            if pattern in content_lower:
                return True

        # Short content without action keywords is likely Q&A
        if len(content) < 100:
            action_keywords = ["clone", "push", "pull", "create", "delete", "edit", "install"]
            has_action = any(kw in content_lower for kw in action_keywords)
            if not has_action:
                return True

        return False

    def _assess_complexity(self, content: str) -> str:
        """Assess task complexity using heuristics"""
        # Length-based heuristic
        if len(content) < 50:
            return "simple"
        elif len(content) > 500:
            return "strategic"

        # Keyword-based heuristic
        strategic_keywords = ["設計", "アーキテクチャ", "システム全体", "戦略", "design", "architecture"]
        for kw in strategic_keywords:
            if kw in content.lower():
                return "strategic"

        return "medium"

    # =========================================================================
    # Routing Decision
    # =========================================================================

    def _route_decision(self, state: TaskState) -> str:
        """Determine which execution node to route to"""

        # Priority 1: Multi-step tasks → Gemini Flash autonomous
        if state["is_multi_step"]:
            logger.info("🔀 Route: gemini_autonomous (multi-step detected)")
            return "gemini_autonomous"

        # Priority 2: Single action tasks → Taisho subprocess
        if state["is_action_task"]:
            logger.info("🔧 Route: taisho_action (single action)")
            return "taisho_action"

        # Priority 3: Simple Q&A → Groq (fast, free)
        if state["is_simple_qa"] and state["complexity"] == "simple":
            logger.info("⚡ Route: groq_qa (simple Q&A)")
            return "groq_qa"

        # Default: Karo routing (existing logic)
        logger.info("🚩 Route: karo_default")
        return "karo_default"

    # =========================================================================
    # Execution Nodes (delegate to existing implementations)
    # =========================================================================

    async def _execute_groq_qa(self, state: TaskState) -> dict:
        """Execute simple Q&A via Groq"""
        start_time = time.time()

        try:
            groq_client = self.orchestrator.get_client("groq")
            if not groq_client:
                logger.warning("⚠️ Groq client not available, falling back to karo_default")
                return await self._execute_karo_default(state)

            response = await groq_client.generate(
                system="あなたは簡潔で正確な回答を提供するアシスタントです。",
                user=state["content"],
                max_tokens=512,
            )

            return {
                "result": {"response": response, "status": "completed"},
                "status": "completed",
                "handled_by": "groq_qa",
                "execution_time": time.time() - start_time,
                "route": "groq_qa",
            }

        except Exception as e:
            logger.error(f"❌ Groq execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "handled_by": "groq_qa",
                "execution_time": time.time() - start_time,
            }

    async def _execute_gemini_autonomous(self, state: TaskState) -> dict:
        """Execute multi-step task via Gemini Flash autonomous executor"""
        start_time = time.time()

        try:
            # Get the autonomous executor from Karo
            karo = self.orchestrator.karo
            if not karo or not karo.gemini_autonomous_executor:
                logger.warning("⚠️ Gemini autonomous executor not available, falling back")
                return await self._execute_karo_default(state)

            executor = karo.gemini_autonomous_executor
            exec_result = await executor.execute_autonomous_task(
                task_content=state["content"],
                max_iterations=5
            )

            if exec_result.status == "failed":
                # Fallback to Taisho
                logger.warning("⚠️ Gemini autonomous failed, falling back to Taisho")
                return await self._execute_taisho_action(state)

            return {
                "result": {
                    "response": exec_result.final_result,
                    "status": "completed",
                    "steps_executed": exec_result.steps_executed,
                    "tool_calls": exec_result.tool_calls_made,
                },
                "status": "completed",
                "handled_by": "gemini_autonomous",
                "execution_time": time.time() - start_time,
                "route": "gemini_autonomous",
            }

        except Exception as e:
            logger.error(f"❌ Gemini autonomous execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "handled_by": "gemini_autonomous",
                "execution_time": time.time() - start_time,
            }

    async def _execute_taisho_action(self, state: TaskState) -> dict:
        """Execute single action via Taisho subprocess"""
        start_time = time.time()

        try:
            karo = self.orchestrator.karo
            if not karo:
                logger.error("❌ Karo not available")
                return {"status": "failed", "error": "Karo not initialized"}

            # Use Karo's Taisho execution
            from core.shogun import Task, TaskComplexity
            task = Task(
                content=state["content"],
                complexity=TaskComplexity.MEDIUM,
                context=state.get("context", {}),
                priority=state.get("priority", 1),
            )

            result = await karo._execute_with_taisho(task, None)

            return {
                "result": result,
                "status": result.get("status", "completed"),
                "handled_by": "taisho_action",
                "execution_time": time.time() - start_time,
                "route": "taisho_action",
            }

        except Exception as e:
            logger.error(f"❌ Taisho execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "handled_by": "taisho_action",
                "execution_time": time.time() - start_time,
            }

    async def _execute_karo_default(self, state: TaskState) -> dict:
        """Execute via default Karo routing (existing logic)"""
        start_time = time.time()

        try:
            karo = self.orchestrator.karo
            if not karo:
                logger.error("❌ Karo not available")
                return {"status": "failed", "error": "Karo not initialized"}

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
                "execution_time": time.time() - start_time,
                "route": "karo_default",
            }

        except Exception as e:
            logger.error(f"❌ Karo default execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "handled_by": "karo_default",
                "execution_time": time.time() - start_time,
            }

    # =========================================================================
    # Public API
    # =========================================================================

    async def process_task(self, content: str, context: dict = None, priority: int = 1, source: str = "api") -> dict:
        """
        Process a task through the LangGraph router.

        This is the main entry point, replacing Shogun.process_task() for routing.

        Args:
            content: Task description
            context: Optional context dict
            priority: Task priority (1-5)
            source: Task source (discord, api, etc.)

        Returns:
            Result dict with status, response, handled_by, etc.
        """
        if not self._compiled:
            await self.initialize()

        # Create initial state
        initial_state: TaskState = {
            "content": content,
            "context": context or {},
            "priority": priority,
            "source": source,
            "is_multi_step": False,
            "is_action_task": False,
            "is_simple_qa": False,
            "complexity": "medium",
            "confidence": 0.0,
            "route": "",
            "result": None,
            "status": "pending",
            "handled_by": "",
            "execution_time": 0.0,
            "error": None,
        }

        logger.info(f"🔗 LangGraph: タスク処理開始 ({content[:50]}...)")
        start_time = time.time()

        try:
            # Run the graph
            final_state = await self._compiled.ainvoke(initial_state)

            total_time = time.time() - start_time
            logger.info(
                f"✅ LangGraph: タスク完了 - route={final_state['route']}, "
                f"handled_by={final_state['handled_by']}, time={total_time:.2f}s"
            )

            # Extract result for compatibility with existing code
            result = final_state.get("result", {})
            if isinstance(result, dict):
                response = result.get("response", result.get("result", ""))
            else:
                response = str(result)

            return {
                "status": final_state.get("status", "completed"),
                "result": response,
                "response": response,
                "handled_by": final_state.get("handled_by", "unknown"),
                "route": final_state.get("route", "unknown"),
                "execution_time": total_time,
                "langgraph": True,
            }

        except Exception as e:
            logger.exception(f"❌ LangGraph processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "handled_by": "langgraph_router",
                "execution_time": time.time() - start_time,
            }
