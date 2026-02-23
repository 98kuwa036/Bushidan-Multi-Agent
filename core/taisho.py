"""
Bushidan Multi-Agent System v10.1 - Taisho (大将: 実装層)

大将は4層フォールバックチェーンを持つ主要実装層として機能。
実際のコード生成、ファイル操作、重量計算タスクを処理する。

v10.1 機能強化:
- 4層フォールバックチェーン: Kimi K2.5(傭兵) → ローカルQwen3 → クラウドQwen3（影武者）→ Gemini 3 Flash
- Kimi K2.5 (128K context): 並列サブタスク実行、大規模コンテキスト処理
- ローカルQwen3: 秘匿情報処理・オフライン保証・Kimi成果物の統合
- llama.cpp CPU最適化（HP ProDesk 600対応、Ollama不要）
- Qwen3-Coder-30B-A3B-instruct-q4_k_m.gguf（4kコンテキスト、CPU推論）
- クラウドQwen3-plus（影武者）コンテキストオーバーフロー対応（32k容量）
- Gemini 3.0 Flash最終防衛線
- 自己修復実行（Layer 2）
- DSPy検証（Layer 3）
- BDIフレームワーク統合（形式的実装推論）
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from utils.logger import get_logger
from core.bdi_framework import (
    BeliefBase, DesireSet, IntentionStack,
    Belief, Desire, Intention, BeliefType, DesireType
)

if TYPE_CHECKING:
    from core.system_orchestrator import SystemOrchestrator

logger = get_logger(__name__)


class ImplementationMode(Enum):
    """Implementation modes for different task types"""
    LIGHTWEIGHT = "lightweight"    # Single file, simple tasks
    STANDARD = "standard"          # Multi-file, standard complexity
    HEAVY = "heavy"               # Complex architecture, multiple components
    PARALLEL = "parallel"         # Multiple parallel implementations


@dataclass
class ImplementationTask:
    """Task representation for implementation"""
    content: str
    mode: ImplementationMode
    context: Optional[Dict[str, Any]] = None
    files_needed: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None


class FallbackStatus(Enum):
    """Status of fallback chain execution"""
    KIMI_SUCCESS = "kimi_k2_success"        # Tier 1: 傭兵 Kimi K2.5
    LOCAL_SUCCESS = "local_qwen3_success"    # Tier 2: ローカル Qwen3
    CLOUD_SUCCESS = "cloud_qwen3_success"    # Tier 3: 影武者 Cloud Qwen3+
    GEMINI_SUCCESS = "gemini3_success"       # Tier 4: 最終防衛 Gemini 3 Flash
    ALL_FAILED = "all_failed"


class Taisho:
    """
    大将 (Taisho) - 実装層 v10.1

    武士団システムの実装担当として、以下の責務を担う:

    主要責務:
    1. 4層フォールバックチェーン付き重量実装タスク
    2. Kimi K2.5（傭兵、128Kコンテキスト）で並列実行
    3. ローカルQwen3（4kコンテキスト、¥0）で秘匿処理・統合・オフライン保証
    4. クラウドQwen3-plus（影武者、32kコンテキスト）容量拡張
    5. Gemini 3.0 Flash最終防衛
    6. MCP駆動ファイル操作
    7. 自己修復コード実行（Layer 2）
    8. DSPy検証（Layer 3）

    BDI統合:
    - 信念基盤: クライアント可用性、コンテキストサイズ
    - 願望集合: 正確実装、効率実行、品質検証、自己修復
    - 意図スタック: 実装計画の実行

    v10.1 フォールバックチェーン (4層鉄壁):
    Tier 1: Kimi K2.5 傭兵 (128K ctx, 並列実行, マルチモーダル)
    Tier 2: ローカルQwen3-Coder-30B (4K ctx, ¥0, 秘匿・統合・オフライン)
    Tier 3: クラウドQwen3-plus (32K ctx, ¥3, 影武者/容量拡張)
    Tier 4: Gemini 3.0 Flash (防衛, ¥0.04)
    """

    VERSION = "10.1"
    LOCAL_CONTEXT_LIMIT = 4096  # ローカルQwen3最適化コンテキスト

    def __init__(self, orchestrator: "SystemOrchestrator"):
        self.orchestrator = orchestrator

        # AI clients (initialized from orchestrator)
        self.kimi_client = None           # Tier 1: 傭兵 Kimi K2.5
        self.qwen3_client = None          # Tier 2: ローカル Qwen3
        self.alibaba_qwen_client = None   # Tier 3: 影武者 Cloud Qwen3+
        self.gemini3_client = None        # Tier 4: 最終防衛 Gemini 3 Flash

        # MCP connections
        self.mcp_manager = None
        self.memory_mcp = None
        self.filesystem_mcp = None
        self.git_mcp = None

        # Error handling layers
        self.self_healing = None
        self.validator = None

        # BDI Framework components
        self.belief_base = BeliefBase()
        self.desire_set = DesireSet()
        self.intention_stack = IntentionStack()
        self.bdi_enabled = True

        # Statistics
        self.execution_stats = {
            "total_tasks": 0,
            "kimi_success": 0,
            "local_success": 0,
            "cloud_fallback": 0,
            "gemini_fallback": 0,
            "total_failures": 0,
            "total_time_seconds": 0.0,
            "context_overflows": 0
        }
        self.bdi_stats = {
            "bdi_cycles": 0,
            "self_healing_applied": 0,
            "validations_passed": 0
        }

    async def initialize(self) -> None:
        """大将と4層フォールバックチェーンの初期化"""
        logger.info(f"⚔️ 大将 v{self.VERSION} 初期化開始...")

        # AIクライアント取得 (4層フォールバックチェーン順)
        self.kimi_client = self.orchestrator.get_client("kimi_k2")
        self.qwen3_client = self.orchestrator.get_client("qwen3")
        self.alibaba_qwen_client = self.orchestrator.get_client("alibaba_qwen")
        self.gemini3_client = self.orchestrator.get_client("gemini3")

        # 利用可能クライアントログ
        if self.kimi_client:
            logger.info("✅ Kimi K2.5（傭兵, Tier 1）有効 - 128K context, 並列実行")
        else:
            logger.warning("⚠️ Kimi K2.5利用不可 → ローカルQwen3がプライマリ")

        if self.qwen3_client:
            logger.info("✅ ローカルQwen3（Tier 2）有効 - 秘匿処理・統合・オフライン")
        else:
            logger.warning("⚠️ ローカルQwen3利用不可")

        if self.alibaba_qwen_client:
            logger.info("✅ クラウドQwen3-plus（影武者, Tier 3）有効")
        else:
            logger.warning("⚠️ 影武者利用不可")

        if self.gemini3_client:
            logger.info("✅ Gemini 3.0 Flash（最終防衛, Tier 4）有効")
        else:
            # 標準Geminiにフォールバック
            self.gemini3_client = self.orchestrator.get_client("gemini")
            if self.gemini3_client:
                logger.info("✅ Geminiクライアント（フォールバック）有効")
            else:
                logger.warning("⚠️ Geminiクライアント利用不可")

        # MCP接続取得
        self.mcp_manager = self.orchestrator.mcp_manager if hasattr(self.orchestrator, 'mcp_manager') else None
        self.memory_mcp = self.orchestrator.get_mcp("memory")
        self.filesystem_mcp = self.orchestrator.get_mcp("filesystem")
        self.git_mcp = self.orchestrator.get_mcp("git")

        # エラーハンドリングレイヤー初期化
        await self._initialize_error_handling()

        # BDIフレームワーク初期化
        self._initialize_bdi()

        logger.info(f"✅ 大将 v{self.VERSION} 初期化完了（BDI有効）")
        self._log_fallback_chain_status()

    async def _initialize_error_handling(self) -> None:
        """Initialize Layer 2 (Self-healing) and Layer 3 (Validation)"""

        # Layer 2: Self-healing executor
        try:
            from utils.self_healing import SelfHealingExecutor

            # Use best available LLM client
            llm_client = self.qwen3_client or self.gemini3_client
            if llm_client:
                self.self_healing = SelfHealingExecutor(
                    llm_client=llm_client,
                    max_correction_attempts=3
                )
                logger.info("✅ Layer 2 (Self-healing) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Self-healing not available: {e}")

        # Layer 3: DSPy validator
        try:
            from utils.dspy_validators import DSPyValidator
            self.validator = DSPyValidator(max_backtracks=2)
            logger.info("✅ Layer 3 (DSPy Validation) initialized")
        except Exception as e:
            logger.warning(f"⚠️ DSPy validator not available: {e}")

    def _log_fallback_chain_status(self) -> None:
        """4層フォールバックチェーンの状態をログ出力"""

        chain = []
        if self.kimi_client:
            chain.append("Kimi K2.5 傭兵 (128K ctx)")
        if self.qwen3_client:
            chain.append("ローカルQwen3 (4K ctx, ¥0)")
        if self.alibaba_qwen_client:
            chain.append("クラウドQwen3+ (32K ctx, ¥3)")
        if self.gemini3_client:
            chain.append("Gemini 3 Flash (防衛)")

        if chain:
            logger.info(f"🔗 4層フォールバックチェーン: {' → '.join(chain)}")
        else:
            logger.error("❌ AIクライアント利用不可！")

    async def execute_implementation(self, task: ImplementationTask) -> Dict[str, Any]:
        """
        4層フォールバックチェーン付きメイン実装実行

        優先順位:
        1. Kimi K2.5（128Kコンテキスト, クラウド並列実行）
        2. ローカルQwen3（4Kコンテキスト, 秘匿・統合・オフライン）
        3. クラウドQwen3-plus（影武者）コンテキストオーバーフロー対応
        4. Gemini 3.0 Flash最終防衛
        """
        start_time = time.time()
        logger.info(f"⚔️ 大将、実装開始: {task.content[:50]}...")

        try:
            # Gather context
            context = await self._gather_context(task)

            # Estimate context size
            context_size = self._estimate_context_size(task, context)
            logger.info(f"📏 Estimated context size: {context_size} tokens")

            # Plan implementation
            plan = await self._plan_implementation(task, context)

            # Execute with 4-tier fallback chain
            result, fallback_status = await self._execute_with_fallback(
                task, plan, context, context_size
            )

            # Diagnostic tasks: skip file validation and git commit
            if self._is_diagnostic_task(task):
                validation = {"valid": True, "skipped": "diagnostic_task"}
            else:
                # Validate results
                validation = await self._validate_implementation(result)

                # Git operations if successful
                if validation.get("valid", False):
                    await self._commit_changes(task, result)

            # Update statistics
            elapsed_time = time.time() - start_time
            self._update_stats(fallback_status, elapsed_time)

            logger.info(f"✅ Taisho implementation complete ({fallback_status.value}) in {elapsed_time:.1f}s")

            return {
                "status": "completed",
                "result": result,
                "validation": validation,
                "mode": task.mode.value,
                "handled_by": "taisho",
                "fallback_status": fallback_status.value,
                "execution_time": elapsed_time
            }

        except Exception as e:
            import traceback
            logger.exception(f"❌ 大将の実装処理中に致命的なエラーが発生しました: {e}")
            self.execution_stats["total_failures"] += 1
            return {"error": str(e), "traceback": traceback.format_exc(), "status": "failed"}

    def _estimate_context_size(self, task: ImplementationTask, context: Dict[str, Any]) -> int:
        """Estimate context size in tokens (rough approximation)"""

        # Rough estimate: 1 token ≈ 4 characters
        total_chars = len(task.content)

        for entry in context.get("memory_entries", []):
            total_chars += len(str(entry))

        for file_info in context.get("existing_files", []):
            if file_info.get("content"):
                total_chars += len(file_info["content"])

        return total_chars // 4

    async def _execute_with_fallback(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any],
        context_size: int
    ) -> tuple[Dict[str, Any], FallbackStatus]:
        """
        Execute with 4-tier fallback chain (鉄壁チェーン)

        1. Kimi K2.5 傭兵 (128K context, 並列実行可能)
        2. ローカル Qwen3 (4K context, 秘匿・統合・オフライン)
        3. Cloud Qwen3-plus 影武者 (32K context, 容量拡張)
        4. Gemini 3.0 Flash (最終防衛)

        秘匿タスク (from_gunshi + confidential) はローカル Qwen3 から開始。
        """

        # 秘匿タスク判定: ローカル直行 (API に送信しない)
        is_confidential = (
            task.context
            and task.context.get("confidential", False)
        )

        # Tier 1: Kimi K2.5 傭兵 (非秘匿 + クライアント有効時)
        if self.kimi_client and not is_confidential:
            try:
                result = await self._execute_with_kimi(task, plan, context)
                if result.get("status") != "failed":
                    self.execution_stats["kimi_success"] += 1
                    return result, FallbackStatus.KIMI_SUCCESS
                logger.warning("⚠️ Kimi K2.5 failed, falling back to local Qwen3")
            except Exception as e:
                logger.warning(f"⚠️ Kimi K2.5 error: {e}, falling back to local Qwen3")

        # Tier 2: ローカル Qwen3 (context fits + 秘匿処理 + Kimi 成果物統合)
        if self.qwen3_client and context_size <= self.LOCAL_CONTEXT_LIMIT:
            try:
                result = await self._execute_with_qwen3(task, plan, context)
                if result.get("status") != "failed":
                    self.execution_stats["local_success"] += 1
                    return result, FallbackStatus.LOCAL_SUCCESS
                logger.warning("⚠️ Local Qwen3 failed, activating Kagemusha")
            except Exception as e:
                logger.warning(f"⚠️ Local Qwen3 error: {e}, activating Kagemusha")

        # Context overflow logging
        if context_size > self.LOCAL_CONTEXT_LIMIT:
            self.execution_stats["context_overflows"] += 1
            logger.info(
                f"🏯 Context overflow ({context_size} > {self.LOCAL_CONTEXT_LIMIT}), "
                "activating Kagemusha"
            )

        # Tier 3: Cloud Qwen3-plus (影武者 / 容量拡張)
        if self.alibaba_qwen_client:
            try:
                result = await self._execute_with_kagemusha(task, plan, context)
                if result.get("status") != "failed":
                    self.execution_stats["cloud_fallback"] += 1
                    return result, FallbackStatus.CLOUD_SUCCESS
                logger.warning("⚠️ Kagemusha failed, activating Gemini final defense")
            except Exception as e:
                logger.warning(f"⚠️ Kagemusha error: {e}, activating Gemini final defense")

        # Tier 4: Gemini 3.0 Flash (最終防衛)
        if self.gemini3_client:
            try:
                result = await self._execute_with_gemini(task, plan, context)
                if result.get("status") != "failed":
                    self.execution_stats["gemini_fallback"] += 1
                    return result, FallbackStatus.GEMINI_SUCCESS
            except Exception as e:
                logger.error(f"❌ Gemini final defense failed: {e}")

        # All 4 tiers failed
        return {"status": "failed", "error": "All 4 fallback tiers exhausted"}, FallbackStatus.ALL_FAILED

    async def _execute_with_kimi(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute implementation with Kimi K2.5 (傭兵, Tier 1)

        128K context で大規模タスクも処理可能。
        クラウド推論のため asyncio.gather で真の並列実行が可能。
        """
        logger.info("⚔️ Executing with Kimi K2.5 傭兵 (Tier 1, 128K context)")

        implementation_prompt = self._create_implementation_prompt(task, plan, context)

        response = await self.kimi_client.generate(
            messages=[{"role": "user", "content": implementation_prompt}],
            max_tokens=8192,
            temperature=0.7,
        )

        files_created = await self._parse_and_save_files(response)

        return {
            "status": "completed",
            "files_created": files_created,
            "implementation": response,
            "client": "kimi_k2"
        }

    async def _execute_with_qwen3(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute implementation with Local Qwen3 (Tier 2: 秘匿・統合・オフライン)"""

        logger.info("🏯 Executing with Local Qwen3 (primary)")

        implementation_prompt = self._create_implementation_prompt(task, plan, context)

        response = await self.qwen3_client.generate(
            messages=[{"role": "user", "content": implementation_prompt}],
            max_tokens=3000  # Leave room for context in 4k limit
        )

        files_created = await self._parse_and_save_files(response)

        return {
            "status": "completed",
            "files_created": files_created,
            "implementation": response,
            "client": "local_qwen3"
        }

    async def _execute_with_kagemusha(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute implementation with Cloud Qwen3-plus (Kagemusha)"""

        logger.info("🏯 Executing with Kagemusha (Cloud Qwen3-plus, 32k context)")

        implementation_prompt = self._create_implementation_prompt(task, plan, context)

        response = await self.alibaba_qwen_client.generate(
            messages=[{"role": "user", "content": implementation_prompt}],
            max_tokens=4096,
            as_kagemusha=True,
            context_overflow=True
        )

        files_created = await self._parse_and_save_files(response)

        return {
            "status": "completed",
            "files_created": files_created,
            "implementation": response,
            "client": "kagemusha"
        }

    async def _execute_with_gemini(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute implementation with Gemini 3.0 Flash (Final Defense)"""

        logger.info("🛡️ Executing with Gemini 3.0 Flash (Final Defense)")

        implementation_prompt = self._create_implementation_prompt(task, plan, context)

        response = await self.gemini3_client.generate(
            prompt=implementation_prompt,
            max_output_tokens=4000,
            as_final_defense=True
        )

        files_created = await self._parse_and_save_files(response)

        return {
            "status": "completed",
            "files_created": files_created,
            "implementation": response,
            "client": "gemini3"
        }

    _DIAGNOSTIC_KEYWORDS = (
        "status", "check", "確認", "状態", "diagnose", "diagnos",
        "inspect", "verify", "health", "report", "show", "list",
        "表示", "一覧", "調べ", "確かめ",
    )

    def _is_diagnostic_task(self, task: ImplementationTask) -> bool:
        """Return True if the task is a read-only diagnostic/inspection request."""
        content_lower = task.content.lower()
        return any(kw in content_lower for kw in self._DIAGNOSTIC_KEYWORDS)

    def _create_implementation_prompt(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Create appropriate prompt based on task type."""

        if self._is_diagnostic_task(task):
            return self._create_diagnostic_prompt(task)

        return f"""
As Taisho (大将) in Bushidan v{self.VERSION}, implement this task following the plan:

Task: {task.content}
Mode: {task.mode.value}
Plan: {plan.get('plan_text', '')}

Generate complete, working code. Include:
- All necessary imports and dependencies
- Proper error handling
- Clean, readable code structure
- Basic documentation

Output each file separately with clear markers:
=== FILENAME: path/to/file.py ===
[file content]
=== END FILE ===
"""

    def _create_diagnostic_prompt(self, task: ImplementationTask) -> str:
        """Create a diagnostic prompt that returns a status report, not code."""

        return f"""
As Taisho (大将) in Bushidan v{self.VERSION}, answer this diagnostic request directly.

Request: {task.content}

Do NOT write or generate any new files or code.
Instead, report the current status based on what you know about the Bushidan system:

- MCP servers initialized: Memory, Filesystem, Git, Web Search, Smithery (sequential_thinking, playwright, tavily, filesystem, prisma, github, git)
- AI clients available: Claude (API), Groq, Gemini 3.0 Flash, Qwen3 (llama.cpp), Kimi K2.5, Qwen3-Coder-Next, Alibaba Qwen, Opus
- Known issues: LiteLLM unavailable, Ashigaru (足軽) init failure

Provide a concise plain-text status summary. No code, no file output.
"""

    async def _gather_context(self, task: ImplementationTask) -> Dict[str, Any]:
        """Gather relevant context from Memory MCP and filesystem"""

        context = {
            "memory_entries": [],
            "existing_files": [],
            "project_structure": {}
        }

        try:
            if self.memory_mcp:
                memory_query = f"project context for: {task.content}"
                memory_entries = await self.memory_mcp.search(memory_query)
                if isinstance(memory_entries, list):
                    context["memory_entries"] = memory_entries[:5]
                elif memory_entries:
                    context["memory_entries"] = [memory_entries]

            if self.filesystem_mcp and task.files_needed:
                for file_path in task.files_needed:
                    try:
                        content = await self.filesystem_mcp.read_file(file_path)
                        context["existing_files"].append({
                            "path": file_path,
                            "content": content[:2000]
                        })
                    except Exception:
                        context["existing_files"].append({
                            "path": file_path,
                            "content": None
                        })

            logger.info(f"📋 Context gathered: {len(context['memory_entries'])} memory, {len(context['existing_files'])} files")
            return context

        except Exception as e:
            logger.warning(f"⚠️ Context gathering failed: {e}")
            return context

    async def _plan_implementation(self, task: ImplementationTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan implementation approach"""

        # Use best available client for planning
        client = self.qwen3_client or self.gemini3_client
        if not client:
            return {"plan_text": "Direct implementation without explicit plan", "context": context}

        planning_prompt = f"""
As Taisho (大将), plan the implementation for this task:

Task: {task.content}
Mode: {task.mode.value}

Context:
{self._format_memory_context(context.get('memory_entries', []))}

Create a brief implementation plan with:
1. Files to create/modify
2. Key dependencies
3. Implementation approach
"""

        try:
            if hasattr(client, 'generate'):
                if self.qwen3_client:
                    response = await client.generate(
                        messages=[{"role": "user", "content": planning_prompt}],
                        max_tokens=500
                    )
                else:
                    response = await client.generate(
                        prompt=planning_prompt,
                        max_output_tokens=500
                    )
            else:
                response = "Direct implementation"

            return {"plan_text": response, "context": context}

        except Exception as e:
            logger.warning(f"⚠️ Planning failed: {e}")
            return {"plan_text": "Direct implementation", "context": context}

    async def _parse_and_save_files(self, implementation_text: str) -> List[str]:
        """Parse generated code and save files"""

        files_created = []

        if not self.filesystem_mcp:
            logger.warning("⚠️ Filesystem MCP not available")
            return files_created

        sections = implementation_text.split("=== FILENAME:")

        for section in sections[1:]:
            try:
                lines = section.split("\n")
                filename = lines[0].strip().split("===")[0].strip()

                content_lines = []
                for line in lines[1:]:
                    if "=== END FILE ===" in line:
                        break
                    content_lines.append(line)

                content = "\n".join(content_lines).strip()

                if content:
                    await self.filesystem_mcp.write_file(filename, content)
                    files_created.append(filename)
                    logger.info(f"📄 Created file: {filename}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to parse/save file: {e}")

        return files_created

    async def _validate_implementation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate implementation with Layer 3 DSPy validation"""

        validation = {
            "valid": True,
            "files_count": len(result.get("files_created", [])),
            "issues": [],
            "validation_details": []
        }

        if validation["files_count"] == 0 and result.get("status") != "failed":
            # Check if implementation text contains code
            impl_text = result.get("implementation", "")
            if "def " in impl_text or "class " in impl_text or "import " in impl_text:
                validation["valid"] = True
                validation["issues"].append("Code generated but not saved to files")
            else:
                validation["valid"] = False
                validation["issues"].append("No code generated")
            return validation

        # Layer 3: DSPy validation
        if self.validator:
            try:
                implementation_text = result.get("implementation", "")

                validation_rules = [
                    "code_block_present",
                    "no_japanese_comments",
                    "proper_formatting",
                    "complete_implementation"
                ]

                # Use best available client for validation
                llm_client = self.qwen3_client or self.gemini3_client

                if llm_client:
                    validation_passed, validated_output, attempts = await self.validator.validate_with_retry(
                        llm_client=llm_client,
                        output=implementation_text,
                        validation_rules=validation_rules,
                        original_prompt="Implementation task",
                        task_context={"task_type": "code_implementation"}
                    )

                    validation["valid"] = validation_passed
                    validation["validation_attempts"] = attempts

                    if not validation_passed:
                        validation["issues"].append(f"Validation failed after {attempts} attempts")

            except Exception as e:
                logger.warning(f"⚠️ Validation error: {e}")

        return validation

    async def execute_code_with_healing(
        self,
        code: str,
        task_description: str = "",
        allow_installation: bool = False
    ) -> Dict[str, Any]:
        """Execute code with Layer 2 self-healing"""

        if not self.self_healing:
            return {"status": "error", "error": "Self-healing not available"}

        logger.info("🔧 Executing code with self-healing enabled")

        try:
            execution_result = await self.self_healing.run_and_fix(
                code=code,
                task_description=task_description,
                language="python",
                allow_installation=allow_installation
            )

            if execution_result.success:
                logger.info(f"✅ Code executed (attempt {execution_result.attempt_number})")
                return {
                    "status": "success",
                    "stdout": execution_result.stdout,
                    "execution_time": execution_result.execution_time,
                    "attempts": execution_result.attempt_number
                }
            else:
                return {
                    "status": "failed",
                    "error": execution_result.stderr,
                    "attempts": execution_result.attempt_number
                }

        except Exception as e:
            return {"status": "system_error", "error": str(e)}

    async def _commit_changes(self, task: ImplementationTask, result: Dict[str, Any]) -> None:
        """Commit changes using Git MCP"""

        if not self.git_mcp:
            return

        try:
            files_created = result.get("files_created", [])
            if files_created:
                await self.git_mcp.add(files_created)

            commit_message = f"Implement: {task.content[:50]}\n\nGenerated by Taisho v{self.VERSION}"
            await self.git_mcp.commit(commit_message)

            logger.info(f"✅ Changes committed: {len(files_created)} files")

        except Exception as e:
            logger.warning(f"⚠️ Git commit failed: {e}")

    def _update_stats(self, fallback_status: FallbackStatus, elapsed_time: float) -> None:
        """Update execution statistics"""

        self.execution_stats["total_tasks"] += 1
        self.execution_stats["total_time_seconds"] += elapsed_time

        if fallback_status == FallbackStatus.LOCAL_SUCCESS:
            self.execution_stats["local_success"] += 1

    def _format_memory_context(self, memory_entries: List[Dict]) -> str:
        """Format memory entries for context"""
        if not memory_entries:
            return "No relevant memory entries."

        formatted = []
        for entry in memory_entries[:3]:
            formatted.append(f"- {entry.get('content', str(entry))[:200]}")

        return "\n".join(formatted)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive Taisho statistics"""

        stats = {
            "version": self.VERSION,
            "execution_stats": self.execution_stats,
            "fallback_chain": {
                "local_qwen3": self.qwen3_client is not None,
                "kagemusha": self.alibaba_qwen_client is not None,
                "gemini3": self.gemini3_client is not None
            },
            "error_handling": {
                "self_healing": self.self_healing is not None,
                "validator": self.validator is not None
            }
        }

        # Calculate success rate
        total = self.execution_stats["total_tasks"]
        if total > 0:
            local = self.execution_stats["local_success"]
            cloud = self.execution_stats["cloud_fallback"]
            gemini = self.execution_stats["gemini_fallback"]
            failures = self.execution_stats["total_failures"]

            stats["success_rate"] = round((local + cloud + gemini) / total * 100, 1)
            stats["local_rate"] = round(local / total * 100, 1)
            stats["fallback_rate"] = round((cloud + gemini) / total * 100, 1)

        # Add client statistics
        if self.qwen3_client and hasattr(self.qwen3_client, 'get_statistics'):
            stats["qwen3_statistics"] = self.qwen3_client.get_statistics()

        if self.alibaba_qwen_client and hasattr(self.alibaba_qwen_client, 'get_statistics'):
            stats["kagemusha_statistics"] = self.alibaba_qwen_client.get_statistics()

        # Add BDI stats
        stats["bdi_stats"] = self.bdi_stats
        stats["bdi_state"] = self.get_bdi_state()

        return stats

    # ==================== BDI Framework Integration ====================

    def _initialize_bdi(self) -> None:
        """大将BDIフレームワーク初期化"""
        logger.info("🧠 大将BDIフレームワーク初期化...")

        # Initialize operational beliefs about implementation capabilities
        self.belief_base.add_belief(Belief(
            id="has_qwen3",
            type=BeliefType.OPERATIONAL,
            content={"capability": "local_inference", "available": self.qwen3_client is not None, "cost": 0},
            confidence=1.0,
            source="system_init"
        ))

        self.belief_base.add_belief(Belief(
            id="has_kagemusha",
            type=BeliefType.OPERATIONAL,
            content={"capability": "cloud_inference", "available": self.alibaba_qwen_client is not None, "context_limit": 32000},
            confidence=1.0,
            source="system_init"
        ))

        self.belief_base.add_belief(Belief(
            id="has_gemini3",
            type=BeliefType.OPERATIONAL,
            content={"capability": "final_defense", "available": self.gemini3_client is not None},
            confidence=1.0,
            source="system_init"
        ))

        self.belief_base.add_belief(Belief(
            id="has_self_healing",
            type=BeliefType.OPERATIONAL,
            content={"capability": "error_correction", "available": self.self_healing is not None, "max_attempts": 3},
            confidence=1.0,
            source="system_init"
        ))

        self.belief_base.add_belief(Belief(
            id="has_validator",
            type=BeliefType.OPERATIONAL,
            content={"capability": "code_validation", "available": self.validator is not None},
            confidence=1.0,
            source="system_init"
        ))

        mcp_tools = []
        if self.filesystem_mcp:
            mcp_tools.append("filesystem")
        if self.git_mcp:
            mcp_tools.append("git")
        if self.memory_mcp:
            mcp_tools.append("memory")

        self.belief_base.add_belief(Belief(
            id="available_mcp_tools",
            type=BeliefType.OPERATIONAL,
            content={"tools": mcp_tools, "count": len(mcp_tools)},
            confidence=1.0,
            source="system_init"
        ))

        # Initialize implementation desires
        self.desire_set.add_desire(Desire(
            id="correct_implementation",
            type=DesireType.ACHIEVEMENT,
            description="Generate syntactically and semantically correct code",
            priority=1.0,
            feasibility=0.95,
            conditions=["has_qwen3"]
        ))

        self.desire_set.add_desire(Desire(
            id="efficient_execution",
            type=DesireType.OPTIMIZATION,
            description="Minimize resource usage and execution time",
            priority=0.7,
            feasibility=0.9
        ))

        self.desire_set.add_desire(Desire(
            id="quality_validation",
            type=DesireType.ACHIEVEMENT,
            description="Validate code quality before submission",
            priority=0.85,
            feasibility=1.0,
            conditions=["has_validator"]
        ))

        self.desire_set.add_desire(Desire(
            id="self_healing_recovery",
            type=DesireType.MAINTENANCE,
            description="Automatically fix errors through self-healing",
            priority=0.8,
            feasibility=0.9,
            conditions=["has_self_healing"]
        ))

        logger.info(f"🧠 Taisho BDI initialized: {len(self.belief_base.beliefs)} beliefs, {len(self.desire_set.desires)} desires")

    async def execute_implementation_with_bdi(self, task: ImplementationTask) -> Dict[str, Any]:
        """Execute implementation using BDI reasoning cycle"""
        if not self.bdi_enabled:
            return await self.execute_implementation(task)

        logger.info(f"🧠 Taisho BDI cycle starting...")
        self.bdi_stats["bdi_cycles"] += 1

        try:
            # Perceive: Analyze task and gather context
            await self._bdi_perceive(task)

            # Deliberate: Select implementation desire
            selected_desire = await self._bdi_deliberate(task)
            if not selected_desire:
                return await self.execute_implementation(task)

            # Plan: Create implementation intention
            intention = await self._bdi_plan(task, selected_desire)
            if not intention:
                return await self.execute_implementation(task)

            self.intention_stack.adopt_intention(intention)
            self.intention_stack.update_status(intention.id, "executing")

            # Execute: Carry out the implementation plan
            result = await self._bdi_execute(task, intention)

            # Reconsider: Update beliefs based on results
            await self._bdi_reconsider(intention, result)

            return result

        except Exception as e:
            logger.error(f"❌ Taisho BDI cycle failed: {e}")
            return await self.execute_implementation(task)

    async def _bdi_perceive(self, task: ImplementationTask) -> None:
        """Perceive task requirements and context"""

        # Gather context
        context = await self._gather_context(task)

        # Estimate context size
        context_size = self._estimate_context_size(task, context)

        # Add task belief
        self.belief_base.add_belief(Belief(
            id=f"impl_task_{id(task)}",
            type=BeliefType.FACTUAL,
            content={
                "content": task.content[:200],
                "mode": task.mode.value,
                "context_size": context_size,
                "requires_overflow": context_size > self.LOCAL_CONTEXT_LIMIT,
                "files_needed": task.files_needed or []
            },
            confidence=1.0,
            source="task_analysis",
            timestamp=datetime.now()
        ))

        # Add context belief
        if context.get("memory_entries"):
            self.belief_base.add_belief(Belief(
                id=f"impl_context_{id(task)}",
                type=BeliefType.CONTEXTUAL,
                content={"entries_count": len(context["memory_entries"])},
                confidence=0.8,
                source="memory_mcp"
            ))

        logger.debug(f"👁️ Taisho perceived: context_size={context_size}, overflow={context_size > self.LOCAL_CONTEXT_LIMIT}")

    async def _bdi_deliberate(self, task: ImplementationTask) -> Optional[Desire]:
        """Select implementation desire based on task requirements"""

        feasible = self.desire_set.filter_feasible(self.belief_base)
        if not feasible:
            return None

        # Get task info
        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if task_beliefs:
            task_info = task_beliefs[-1].content
            mode = task_info.get("mode", "standard")
            requires_overflow = task_info.get("requires_overflow", False)

            # Adjust priorities based on task
            for desire in feasible:
                if mode == "heavy" and desire.id == "correct_implementation":
                    desire.priority = 1.0

                if mode == "lightweight" and desire.id == "efficient_execution":
                    desire.priority = 0.9

                if requires_overflow and desire.id == "self_healing_recovery":
                    desire.priority = 0.95  # More likely to need healing with complex context

        selected = sorted(feasible, key=lambda d: d.priority * d.feasibility, reverse=True)[0]
        logger.debug(f"🎯 Taisho selected desire: {selected.id}")
        return selected

    async def _bdi_plan(self, task: ImplementationTask, desire: Desire) -> Optional[Intention]:
        """Create implementation plan"""

        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if not task_beliefs:
            return None

        task_info = task_beliefs[-1].content
        requires_overflow = task_info.get("requires_overflow", False)

        plan = []

        if desire.id == "correct_implementation":
            plan = [
                {"action": "gather_context", "agent": "taisho"},
                {"action": "plan_implementation", "agent": "taisho"},
                {"action": "execute_with_fallback", "agent": "llm"},
                {"action": "validate_result", "agent": "validator"},
                {"action": "save_files", "agent": "filesystem_mcp"}
            ]

        elif desire.id == "efficient_execution":
            if requires_overflow:
                plan = [
                    {"action": "execute_with_kagemusha", "agent": "kagemusha"},
                    {"action": "save_files", "agent": "filesystem_mcp"}
                ]
            else:
                plan = [
                    {"action": "execute_with_qwen3", "agent": "qwen3"},
                    {"action": "save_files", "agent": "filesystem_mcp"}
                ]

        elif desire.id == "quality_validation":
            plan = [
                {"action": "gather_context", "agent": "taisho"},
                {"action": "execute_with_fallback", "agent": "llm"},
                {"action": "comprehensive_validation", "agent": "validator"},
                {"action": "self_healing_if_needed", "agent": "self_healing"},
                {"action": "save_files", "agent": "filesystem_mcp"},
                {"action": "git_commit", "agent": "git_mcp"}
            ]

        elif desire.id == "self_healing_recovery":
            plan = [
                {"action": "execute_with_fallback", "agent": "llm"},
                {"action": "self_healing_loop", "agent": "self_healing"},
                {"action": "validate_result", "agent": "validator"},
                {"action": "save_files", "agent": "filesystem_mcp"}
            ]

        intention = Intention(
            id=f"taisho_intention_{datetime.now().timestamp()}",
            desire_id=desire.id,
            plan=plan,
            metadata={"requires_overflow": requires_overflow}
        )

        logger.debug(f"📋 Taisho planned: {len(plan)} steps")
        return intention

    async def _bdi_execute(self, task: ImplementationTask, intention: Intention) -> Dict[str, Any]:
        """Execute the implementation plan"""

        result = {"status": "executing", "steps_completed": [], "bdi_intention": intention.id}

        try:
            context = {}
            plan_result = {}
            impl_result = {}

            for step in intention.plan:
                action = step["action"]

                if action == "gather_context":
                    context = await self._gather_context(task)
                    result["context_gathered"] = True

                elif action == "plan_implementation":
                    plan_result = await self._plan_implementation(task, context)
                    result["plan"] = plan_result.get("plan_text", "")

                elif action == "execute_with_fallback":
                    context_size = self._estimate_context_size(task, context)
                    impl_result, fallback_status = await self._execute_with_fallback(
                        task, plan_result, context, context_size
                    )
                    result["implementation"] = impl_result
                    result["fallback_status"] = fallback_status.value

                elif action == "execute_with_qwen3":
                    impl_result = await self._execute_with_qwen3(task, plan_result, context)
                    result["implementation"] = impl_result

                elif action == "execute_with_kagemusha":
                    impl_result = await self._execute_with_kagemusha(task, plan_result, context)
                    result["implementation"] = impl_result

                elif action in ["validate_result", "comprehensive_validation"]:
                    validation = await self._validate_implementation(impl_result)
                    result["validation"] = validation
                    if validation.get("valid"):
                        self.bdi_stats["validations_passed"] += 1

                elif action == "self_healing_if_needed":
                    if not result.get("validation", {}).get("valid", True):
                        code = impl_result.get("implementation", "")
                        if code:
                            healing_result = await self.execute_code_with_healing(code, task.content)
                            if healing_result.get("status") == "success":
                                self.bdi_stats["self_healing_applied"] += 1

                elif action == "self_healing_loop":
                    code = impl_result.get("implementation", "")
                    if code:
                        healing_result = await self.execute_code_with_healing(code, task.content)
                        result["self_healing"] = healing_result
                        if healing_result.get("status") == "success":
                            self.bdi_stats["self_healing_applied"] += 1

                elif action == "save_files":
                    if impl_result.get("implementation"):
                        files = await self._parse_and_save_files(impl_result["implementation"])
                        result["files_created"] = files

                elif action == "git_commit":
                    if result.get("validation", {}).get("valid", False):
                        await self._commit_changes(task, impl_result)
                        result["git_committed"] = True

                result["steps_completed"].append({"action": action, "status": "completed"})

            result["status"] = "completed"
            result["handled_by"] = "taisho"
            logger.info(f"✅ Taisho BDI execution complete")

        except Exception as e:
            logger.error(f"❌ Taisho BDI execution failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    async def _bdi_reconsider(self, intention: Intention, result: Dict[str, Any]) -> None:
        """Update beliefs based on execution results"""

        if result.get("status") == "completed":
            self.intention_stack.update_status(intention.id, "completed")

            self.belief_base.add_belief(Belief(
                id=f"impl_success_{intention.id}",
                type=BeliefType.HISTORICAL,
                content={
                    "desire_id": intention.desire_id,
                    "fallback_used": result.get("fallback_status", "local_qwen3_success"),
                    "files_created": len(result.get("files_created", [])),
                    "validation_passed": result.get("validation", {}).get("valid", False)
                },
                confidence=1.0,
                source="execution_result",
                timestamp=datetime.now()
            ))
        else:
            self.intention_stack.update_status(intention.id, "failed")
            self.execution_stats["total_failures"] += 1

    def get_bdi_state(self) -> Dict[str, Any]:
        """Get current BDI state"""
        return {
            "beliefs": self.belief_base.get_statistics(),
            "desires": self.desire_set.get_statistics(),
            "intentions": self.intention_stack.get_statistics()
        }
