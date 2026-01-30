"""
Bushidan Multi-Agent System v9.1 - Ashigaru (Execution Layer)

The Ashigaru serve as the execution layer using Qwen2.5-Coder-32B (Japanese imatrix).
Handles direct implementation tasks through MCP integration.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from utils.logger import get_logger
from utils.qwen_client import QwenClient
from core.system_orchestrator import SystemOrchestrator


logger = get_logger(__name__)


class Ashigaru:
    """
    足軽 (Ashigaru) - Execution Layer
    
    Primary responsibilities:
    1. Direct task implementation using Qwen2.5-Coder
    2. MCP operations (filesystem, git, web search)
    3. Code generation and file manipulation
    4. Command execution and result reporting
    """
    
    def __init__(self, orchestrator: SystemOrchestrator):
        self.orchestrator = orchestrator
        self.qwen_client = QwenClient(
            endpoint=orchestrator.config.litellm_endpoint
        )
        self.mcps = {}
        
    async def initialize(self) -> None:
        """Initialize Ashigaru and MCP connections"""
        logger.info("🏃 Initializing Ashigaru (Execution Layer)...")
        
        # Get all available MCPs
        self.mcps = {
            "filesystem": self.orchestrator.get_mcp("filesystem"),
            "git": self.orchestrator.get_mcp("git"), 
            "memory": self.orchestrator.get_mcp("memory"),
            "web_search": self.orchestrator.get_mcp("web_search")
        }
        
        # Remove None values
        self.mcps = {k: v for k, v in self.mcps.items() if v is not None}
        
        logger.info(f"✅ Ashigaru initialized with {len(self.mcps)} MCP connections")
    
    async def execute_direct(self, task) -> Dict[str, Any]:
        """Execute task directly without subtask decomposition"""
        
        logger.info(f"🏃 Ashigaru executing direct task: {task.content[:50]}...")
        
        # Build execution prompt for Qwen2.5-Coder
        execution_prompt = f"""
        あなたは武士の足軽（Ashigaru）として、以下のタスクを実行してください：
        
        タスク: {task.content}
        複雑度: {task.complexity.value}
        
        利用可能なMCP機能:
        {', '.join(self.mcps.keys())}
        
        適切なMCP機能を使用して、タスクを完了してください。
        日本語での応答も可能です。実装が必要な場合は、適切なコードを生成してください。
        """
        
        try:
            response = await self.qwen_client.generate(
                messages=[{"role": "user", "content": execution_prompt}],
                max_tokens=2000
            )
            
            return {
                "status": "completed",
                "result": response,
                "handled_by": "ashigaru_direct"
            }
            
        except Exception as e:
            logger.error(f"❌ Direct execution failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def execute_subtask(self, subtask) -> Dict[str, Any]:
        """Execute a specific subtask using designated MCP"""
        
        logger.info(f"🏃 Ashigaru executing subtask {subtask.id}: {subtask.content[:30]}...")
        
        # Get designated MCP
        mcp = self.mcps.get(subtask.ashigaru_type)
        if not mcp:
            return {
                "error": f"MCP '{subtask.ashigaru_type}' not available",
                "status": "failed"
            }
        
        try:
            # Execute through appropriate MCP
            if subtask.ashigaru_type == "filesystem":
                result = await self._execute_filesystem_task(subtask, mcp)
            elif subtask.ashigaru_type == "git":
                result = await self._execute_git_task(subtask, mcp)
            elif subtask.ashigaru_type == "web_search":
                result = await self._execute_web_search_task(subtask, mcp)
            elif subtask.ashigaru_type == "memory":
                result = await self._execute_memory_task(subtask, mcp)
            else:
                result = await self._execute_generic_task(subtask)
            
            logger.info(f"✅ Subtask {subtask.id} completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Subtask {subtask.id} failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _execute_filesystem_task(self, subtask, mcp) -> Dict[str, Any]:
        """Execute filesystem operations"""
        
        # Use Qwen to determine what filesystem operation is needed
        analysis_prompt = f"""
        ファイルシステムタスク: {subtask.content}
        
        必要な操作を判断し、以下から選択してください：
        - read_file: ファイル読み込み
        - write_file: ファイル書き込み
        - list_directory: ディレクトリ一覧
        - search_files: ファイル検索
        
        操作とパラメータをJSON形式で返してください：
        {{"operation": "read_file", "path": "/path/to/file"}}
        """
        
        operation_response = await self.qwen_client.generate(
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=200
        )
        
        try:
            import json
            operation = json.loads(operation_response)
            
            # Execute the operation
            if operation["operation"] == "read_file":
                content = await mcp.read_file(operation["path"])
                return {"status": "completed", "result": content}
            elif operation["operation"] == "write_file":
                await mcp.write_file(operation["path"], operation["content"])
                return {"status": "completed", "result": f"File written: {operation['path']}"}
            # Add other filesystem operations as needed
            
        except Exception as e:
            return {"error": f"Filesystem operation failed: {e}", "status": "failed"}
    
    async def _execute_git_task(self, subtask, mcp) -> Dict[str, Any]:
        """Execute git operations"""
        
        # Simplified git operations
        try:
            if "status" in subtask.content.lower():
                result = await mcp.status()
            elif "commit" in subtask.content.lower():
                result = await mcp.commit(subtask.content)
            elif "diff" in subtask.content.lower():
                result = await mcp.diff()
            else:
                result = "Git operation not recognized"
            
            return {"status": "completed", "result": result}
            
        except Exception as e:
            return {"error": f"Git operation failed: {e}", "status": "failed"}
    
    async def _execute_web_search_task(self, subtask, mcp) -> Dict[str, Any]:
        """Execute web search operations"""
        
        try:
            # Extract search query from subtask content
            search_result = await mcp.search(subtask.content)
            
            return {"status": "completed", "result": search_result}
            
        except Exception as e:
            return {"error": f"Web search failed: {e}", "status": "failed"}
    
    async def _execute_memory_task(self, subtask, mcp) -> Dict[str, Any]:
        """Execute memory operations"""
        
        try:
            if "store" in subtask.content.lower() or "save" in subtask.content.lower():
                # Store information
                await mcp.store({"content": subtask.content, "type": "task_result"})
                return {"status": "completed", "result": "Information stored in memory"}
            else:
                # Search memory
                results = await mcp.search(subtask.content)
                return {"status": "completed", "result": results}
                
        except Exception as e:
            return {"error": f"Memory operation failed: {e}", "status": "failed"}
    
    async def _execute_generic_task(self, subtask) -> Dict[str, Any]:
        """Execute generic task using Qwen without specific MCP"""
        
        generic_prompt = f"""
        以下のタスクを実行してください：
        
        タスク: {subtask.content}
        タイプ: {subtask.ashigaru_type}
        
        適切な実装やアドバイスを提供してください。
        """
        
        response = await self.qwen_client.generate(
            messages=[{"role": "user", "content": generic_prompt}],
            max_tokens=1500
        )
        
        return {"status": "completed", "result": response}