"""
Bushidan Multi-Agent System v10.1 - Gemini Flash Autonomous Executor

複合タスク（複数ステップ）の自律実行エンジン

Gemini 3.0 Flash が MCP tools を使って自律的に複数ステップを実行。
例：「クローンして内容を確認してまとめて」
  1. git clone
  2. ls / cat で内容確認
  3. 分析結果をまとめる

フォールバック：失敗時は Haiku でリトライ
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger

if False:  # TYPE_CHECKING
    from core.system_orchestrator import SystemOrchestrator
    from core.taisho import ImplementationTask

logger = get_logger(__name__)


@dataclass
class AutonomousExecutionResult:
    """複合タスク自律実行の結果"""
    status: str  # "success", "failed", "partial"
    steps_executed: int
    outputs: List[Dict[str, Any]]  # 各ステップの出力
    final_result: str  # 最終的なまとめ・結果
    total_tokens_used: int
    execution_time_seconds: float
    fallback_used: bool  # Haiku へのフォールバック有無
    error_message: Optional[str] = None


class GeminiFlashAutonomousExecutor:
    """
    Gemini Flash で複合タスクを自律実行するエンジン

    機能：
    1. MCP tools（filesystem, git）を tool_use 形式で定義
    2. Loop実行：応答 → tool call 抽出 → 実行 → 結果を入力に含める
    3. Haiku フォールバック：Flash が失敗時にリトライ
    4. 複数ステップの自動実行と結果統合

    コスト：
    - Gemini Flash: ¥0.075/100k tokens（最安）
    - Haiku: ¥0.5/task（フォールバック時のみ）
    """

    def __init__(self, orchestrator: "SystemOrchestrator"):
        self.orchestrator = orchestrator
        self.gemini3_client = orchestrator.get_client("gemini3")
        self.haiku_client = None  # Haiku は後で統合予定
        self.filesystem_mcp = orchestrator.get_mcp("filesystem")
        self.git_mcp = orchestrator.get_mcp("git")

        logger.info("🤖 Gemini Flash Autonomous Executor initialized")

    def _get_mcp_tools_definition(self) -> List[Dict[str, Any]]:
        """
        MCP tools を tool_use 形式で定義

        Gemini API の tool_use (function calling) に対応
        """
        return [
            {
                "name": "filesystem_read",
                "description": "ファイルの内容を読み取る",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "読み取るファイルのパス"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "filesystem_list",
                "description": "ディレクトリの内容を一覧表示",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "一覧表示するディレクトリのパス"
                        }
                    },
                    "required": ["directory_path"]
                }
            },
            {
                "name": "filesystem_write",
                "description": "ファイルに内容を書き込む",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "書き込むファイルのパス"
                        },
                        "content": {
                            "type": "string",
                            "description": "ファイルの内容"
                        }
                    },
                    "required": ["file_path", "content"]
                }
            },
            {
                "name": "git_clone",
                "description": "Git リポジトリをクローン",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "repository_url": {
                            "type": "string",
                            "description": "クローンするリポジトリの URL"
                        }
                    },
                    "required": ["repository_url"]
                }
            },
            {
                "name": "git_commit",
                "description": "Git に変更をコミット",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "コミットメッセージ"
                        }
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "git_push",
                "description": "Git にコミットをプッシュ",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "remote": {
                            "type": "string",
                            "description": "プッシュ先のリモート（デフォルト: origin）"
                        },
                        "branch": {
                            "type": "string",
                            "description": "プッシュするブランチ（デフォルト: main）"
                        }
                    },
                    "required": []
                }
            }
        ]

    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, str]) -> str:
        """
        MCP tool を実行

        実際にファイルシステムやGitを操作
        """
        try:
            if tool_name == "filesystem_read":
                if self.filesystem_mcp:
                    content = await self.filesystem_mcp.read_file(tool_input["file_path"])
                    return content[:5000]  # トークン制限のため先頭5000文字
                else:
                    return "❌ Filesystem MCP not available"

            elif tool_name == "filesystem_list":
                if self.filesystem_mcp:
                    items = await self.filesystem_mcp.list_directory(tool_input["directory_path"])
                    return "\n".join(items)
                else:
                    return "❌ Filesystem MCP not available"

            elif tool_name == "filesystem_write":
                if self.filesystem_mcp:
                    await self.filesystem_mcp.write_file(
                        tool_input["file_path"],
                        tool_input["content"]
                    )
                    return f"✅ Written to {tool_input['file_path']}"
                else:
                    return "❌ Filesystem MCP not available"

            elif tool_name == "git_clone":
                if self.git_mcp:
                    result = await self.git_mcp.clone(tool_input["repository_url"])
                    return f"✅ Cloned: {result}"
                else:
                    return "❌ Git MCP not available"

            elif tool_name == "git_commit":
                if self.git_mcp:
                    result = await self.git_mcp.commit(tool_input["message"])
                    return f"✅ Committed: {result}"
                else:
                    return "❌ Git MCP not available"

            elif tool_name == "git_push":
                if self.git_mcp:
                    remote = tool_input.get("remote", "origin")
                    branch = tool_input.get("branch", "main")
                    result = await self.git_mcp.push(remote, branch)
                    return f"✅ Pushed: {result}"
                else:
                    return "❌ Git MCP not available"

            else:
                return f"❌ Unknown tool: {tool_name}"

        except Exception as e:
            logger.error(f"❌ Tool execution failed: {tool_name} - {e}")
            return f"❌ Error: {str(e)}"

    def _to_gemini_tools_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        MCP ツール定義を Gemini functionDeclarations 形式に変換

        input_schema → parameters キーに変換
        """
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"]
            }
            for t in tools
        ]

    async def execute_autonomous_task(
        self,
        task_content: str,
        max_iterations: int = 5
    ) -> AutonomousExecutionResult:
        """
        複合タスクを自律実行（本格版 Gemini Function Calling）

        フロー：
        1. Gemini Flash に MCP tools（functionDeclarations）を提供
        2. ループ（最大 max_iterations）：
           a. generate_with_tools() で Gemini Flash に問い合わせ
           b. tool_calls があれば順次実行
           c. 実行結果を function response として会話に追加
           d. tool_calls がなければ実行完了（テキスト応答）
        3. 最終結果をまとめる
        4. 失敗時は Haiku でリトライ
        """
        start_time = datetime.now()
        outputs: List[Dict[str, Any]] = []
        messages: List[Dict[str, str]] = []
        iteration = 0

        if not self.gemini3_client:
            return AutonomousExecutionResult(
                status="failed",
                steps_executed=0,
                outputs=[],
                final_result="❌ Gemini Flash not available",
                total_tokens_used=0,
                execution_time_seconds=0,
                fallback_used=False,
                error_message="Gemini 3.0 Flash client not found"
            )

        try:
            # MCP ツールを Gemini 形式に変換
            gemini_tools = self._to_gemini_tools_format(self._get_mcp_tools_definition())

            # 初期メッセージ
            messages.append({
                "role": "user",
                "content": (
                    f"You are an autonomous task executor in Bushidan Multi-Agent System.\n"
                    f"Execute this task step by step using the available tools.\n"
                    f"When all steps are done, provide a final summary.\n\n"
                    f"Task: {task_content}"
                )
            })

            completed = False
            final_result = ""

            while iteration < max_iterations and not completed:
                iteration += 1
                logger.info(f"🔄 Autonomous iteration {iteration}/{max_iterations}")

                # Gemini Flash に function calling で問い合わせ
                response = await self.gemini3_client.generate_with_tools(
                    messages=messages,
                    tools=gemini_tools,
                    tool_choice="auto",
                    max_output_tokens=1024,
                    temperature=0.3
                )

                tool_calls = response.get("tool_calls", [])
                text = response.get("text")

                if tool_calls:
                    # ツールを実行し、結果を会話に追加
                    for tool_call in tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        logger.info(f"🔧 Tool call: {tool_name}({tool_args})")

                        tool_result = await self._execute_tool(tool_name, tool_args)

                        logger.info(f"✅ Tool result: {tool_result[:200]}")

                        outputs.append({
                            "step": iteration,
                            "tool": tool_name,
                            "args": tool_args,
                            "result": tool_result
                        })

                        # assistant の tool call を会話に追加
                        messages.append({
                            "role": "assistant",
                            "content": f"[Calling tool: {tool_name}]"
                        })

                        # tool result を会話に追加
                        messages.append({
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": tool_result
                        })

                else:
                    # tool_calls がない = 実行完了（最終テキスト応答）
                    completed = True
                    final_result = text or ""
                    outputs.append({
                        "step": iteration,
                        "status": "completed",
                        "output": final_result
                    })

                    logger.info(f"✅ Autonomous execution completed after {iteration} iterations")

            execution_time = (datetime.now() - start_time).total_seconds()

            # max_iterations に達した場合は partial
            if not completed and messages:
                final_result = f"(Partial: reached max {max_iterations} iterations)"
                logger.warning(f"⚠️ Autonomous execution partial: max iterations reached")

            return AutonomousExecutionResult(
                status="success" if completed else "partial",
                steps_executed=iteration,
                outputs=outputs,
                final_result=final_result,
                total_tokens_used=0,
                execution_time_seconds=execution_time,
                fallback_used=False
            )

        except Exception as e:
            logger.error(f"❌ Autonomous execution failed: {e}")

            execution_time = (datetime.now() - start_time).total_seconds()

            return AutonomousExecutionResult(
                status="failed",
                steps_executed=iteration,
                outputs=outputs,
                final_result="",
                total_tokens_used=0,
                execution_time_seconds=execution_time,
                fallback_used=False,
                error_message=str(e)
            )

    async def _fallback_to_haiku(
        self,
        task_content: str,
        previous_attempts: List[Dict[str, Any]]
    ) -> AutonomousExecutionResult:
        """
        Haiku へのフォールバック実行（未実装）

        TODO: Haiku クライアントを統合後に実装
        """
        logger.info("🔄 Falling back to Haiku...")
        return AutonomousExecutionResult(
            status="failed",
            steps_executed=0,
            outputs=previous_attempts,
            final_result="",
            total_tokens_used=0,
            execution_time_seconds=0,
            fallback_used=True,
            error_message="Haiku fallback not yet implemented"
        )

    def get_executor_status(self) -> Dict[str, Any]:
        """実行エンジンの状態を返却"""
        return {
            "gemini3_available": self.gemini3_client is not None,
            "haiku_available": self.haiku_client is not None,
            "filesystem_mcp_available": self.filesystem_mcp is not None,
            "git_mcp_available": self.git_mcp is not None,
            "supported_tools": [
                "filesystem_read", "filesystem_list", "filesystem_write",
                "git_clone", "git_commit", "git_push"
            ]
        }
