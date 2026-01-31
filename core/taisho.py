"""
Bushidan Multi-Agent System v9.3 - Taisho (Implementation Layer)

The Taisho serves as the primary implementation layer using Qwen3-Coder-30B-A3B.
Handles actual code generation, file operations, and heavy computational tasks.

v9.3 Enhancements:
- Qwen3-Coder-30B-A3B integration (MoE, 32B-class intelligence in 24GB RAM)
- DSPy translation layer for Japanese→structured instructions
- LiteLLM middleware for 4k context compression
- Cost ¥0 local inference with unlimited context capacity
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.qwen_client import QwenClient
from core.mcp_manager import MCPManager


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
    files_needed: List[str] = None
    dependencies: List[str] = None


class Taisho:
    """
    大将 (Taisho) - Implementation Layer
    
    Primary responsibilities:
    1. Heavy implementation tasks using local Qwen3-Coder-30B-A3B
    2. MCP-driven file operations and system interactions
    3. Multi-file code generation and refactoring
    4. Cost-effective processing with unlimited local compute
    5. Parallel task execution for complex projects
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.qwen_client = QwenClient(
            api_base=orchestrator.config.get("qwen_api_base", "http://localhost:11434"),
            model_name="qwen3-coder-30b-a3b"
        )
        self.mcp_manager = orchestrator.mcp_manager
        self.memory_mcp = None
        self.filesystem_mcp = None
        self.git_mcp = None
        
    async def initialize(self) -> None:
        """Initialize Taisho and MCP connections"""
        logger.info("🏯 Initializing Taisho (Implementation Layer)...")
        
        # Get MCP connections
        self.memory_mcp = self.mcp_manager.get_mcp("memory")
        self.filesystem_mcp = self.mcp_manager.get_mcp("filesystem")
        self.git_mcp = self.mcp_manager.get_mcp("git")
        
        # Verify Qwen3-Coder connection
        try:
            await self.qwen_client.health_check()
            logger.info("✅ Qwen3-Coder-30B-A3B connection verified")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qwen3-Coder: {e}")
            raise
            
        logger.info("✅ Taisho initialization complete")
    
    async def execute_implementation(self, task: ImplementationTask) -> Dict[str, Any]:
        """
        Main implementation execution pipeline
        
        1. Context gathering and analysis
        2. Implementation planning
        3. Code generation with MCP operations
        4. Quality validation and testing
        5. Git operations and cleanup
        """
        
        logger.info(f"🏯 Taisho executing implementation: {task.content[:50]}...")
        
        try:
            # Step 1: Gather context from Memory MCP and filesystem
            context = await self._gather_context(task)
            
            # Step 2: Plan implementation approach
            plan = await self._plan_implementation(task, context)
            
            # Step 3: Execute implementation based on mode
            if task.mode == ImplementationMode.PARALLEL:
                result = await self._execute_parallel_implementation(task, plan, context)
            else:
                result = await self._execute_sequential_implementation(task, plan, context)
            
            # Step 4: Validate and test results
            validation = await self._validate_implementation(result)
            
            # Step 5: Git operations if successful
            if validation.get("valid", False):
                await self._commit_changes(task, result)
            
            logger.info("✅ Taisho implementation complete")
            return {
                "status": "completed",
                "result": result,
                "validation": validation,
                "mode": task.mode.value,
                "handled_by": "taisho"
            }
            
        except Exception as e:
            logger.error(f"❌ Taisho implementation failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _gather_context(self, task: ImplementationTask) -> Dict[str, Any]:
        """Gather relevant context from Memory MCP and filesystem"""
        
        context = {
            "memory_entries": [],
            "existing_files": [],
            "project_structure": {}
        }
        
        try:
            # Search Memory MCP for relevant decisions and patterns
            if self.memory_mcp:
                memory_query = f"project context for: {task.content}"
                memory_entries = await self.memory_mcp.search(memory_query)
                context["memory_entries"] = memory_entries[:5]  # Top 5 relevant entries
            
            # Analyze existing project structure
            if self.filesystem_mcp and task.files_needed:
                for file_path in task.files_needed:
                    try:
                        content = await self.filesystem_mcp.read_file(file_path)
                        context["existing_files"].append({
                            "path": file_path,
                            "content": content[:2000]  # First 2k chars for context
                        })
                    except FileNotFoundError:
                        context["existing_files"].append({
                            "path": file_path,
                            "content": None  # File doesn't exist
                        })
            
            logger.info(f"📋 Context gathered: {len(context['memory_entries'])} memory entries, {len(context['existing_files'])} files")
            return context
            
        except Exception as e:
            logger.warning(f"⚠️ Context gathering failed: {e}")
            return context
    
    async def _plan_implementation(self, task: ImplementationTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan implementation approach using Qwen3-Coder"""
        
        planning_prompt = f"""
        As Taisho (大将), plan the implementation for this task:
        
        Task: {task.content}
        Mode: {task.mode.value}
        
        Context from Memory MCP:
        {self._format_memory_context(context.get('memory_entries', []))}
        
        Existing files:
        {self._format_file_context(context.get('existing_files', []))}
        
        Create a detailed implementation plan including:
        1. Files to create/modify
        2. Dependencies and imports needed
        3. Step-by-step implementation approach
        4. Testing and validation steps
        
        Focus on practical, working code that follows project conventions.
        """
        
        response = await self.qwen_client.generate(
            messages=[{"role": "user", "content": planning_prompt}],
            max_tokens=1000
        )
        
        return {"plan_text": response, "context": context}
    
    async def _execute_sequential_implementation(self, task: ImplementationTask, plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute implementation sequentially"""
        
        implementation_prompt = f"""
        As Taisho (大将), implement this task following the plan:
        
        Task: {task.content}
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
        
        response = await self.qwen_client.generate(
            messages=[{"role": "user", "content": implementation_prompt}],
            max_tokens=4000
        )
        
        # Parse and save files
        files_created = await self._parse_and_save_files(response)
        
        return {
            "files_created": files_created,
            "implementation": response
        }
    
    async def _execute_parallel_implementation(self, task: ImplementationTask, plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute implementation in parallel for complex tasks"""
        
        # For parallel execution, break down into subtasks and process simultaneously
        subtasks = await self._break_down_task(task, plan)
        
        # Execute subtasks in parallel
        parallel_results = await asyncio.gather(
            *[self._execute_subtask(subtask, context) for subtask in subtasks],
            return_exceptions=True
        )
        
        # Combine results
        combined_result = await self._combine_parallel_results(parallel_results)
        
        return combined_result
    
    async def _break_down_task(self, task: ImplementationTask, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down complex task into parallel subtasks"""
        
        breakdown_prompt = f"""
        Break down this complex implementation into 3-5 independent subtasks:
        
        Task: {task.content}
        Plan: {plan.get('plan_text', '')}
        
        Each subtask should be:
        - Independent and parallelizable
        - Focused on specific components
        - Clearly defined with expected outputs
        
        Format as JSON array of subtasks.
        """
        
        response = await self.qwen_client.generate(
            messages=[{"role": "user", "content": breakdown_prompt}],
            max_tokens=500
        )
        
        # Parse JSON response (simplified)
        try:
            import json
            subtasks = json.loads(response)
            return subtasks
        except:
            # Fallback: create simple breakdown
            return [
                {"name": "core_implementation", "description": task.content},
                {"name": "tests", "description": f"Tests for {task.content}"},
                {"name": "documentation", "description": f"Documentation for {task.content}"}
            ]
    
    async def _execute_subtask(self, subtask: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual subtask"""
        
        subtask_prompt = f"""
        Implement this specific subtask:
        
        Subtask: {subtask.get('description', '')}
        Name: {subtask.get('name', '')}
        
        Generate focused, working code for this specific component.
        """
        
        response = await self.qwen_client.generate(
            messages=[{"role": "user", "content": subtask_prompt}],
            max_tokens=2000
        )
        
        return {
            "subtask_name": subtask.get('name', ''),
            "implementation": response
        }
    
    async def _combine_parallel_results(self, parallel_results: List[Any]) -> Dict[str, Any]:
        """Combine results from parallel execution"""
        
        combined = {
            "files_created": [],
            "implementations": [],
            "errors": []
        }
        
        for result in parallel_results:
            if isinstance(result, Exception):
                combined["errors"].append(str(result))
            elif isinstance(result, dict):
                combined["implementations"].append(result)
        
        return combined
    
    async def _parse_and_save_files(self, implementation_text: str) -> List[str]:
        """Parse generated code and save files using Filesystem MCP"""
        
        files_created = []
        
        if not self.filesystem_mcp:
            logger.warning("⚠️ Filesystem MCP not available")
            return files_created
        
        # Simple file parsing (look for === FILENAME: markers)
        sections = implementation_text.split("=== FILENAME:")
        
        for section in sections[1:]:  # Skip first empty section
            try:
                lines = section.split("\n")
                filename = lines[0].strip().split("===")[0].strip()
                
                # Find end of file marker
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
                logger.warning(f"⚠️ Failed to parse/save file section: {e}")
        
        return files_created
    
    async def _validate_implementation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Basic validation of implementation results"""
        
        validation = {
            "valid": True,
            "files_count": len(result.get("files_created", [])),
            "issues": []
        }
        
        # Basic checks
        if validation["files_count"] == 0:
            validation["valid"] = False
            validation["issues"].append("No files were created")
        
        # TODO: Add more sophisticated validation
        # - Syntax checking
        # - Import validation
        # - Basic testing
        
        return validation
    
    async def _commit_changes(self, task: ImplementationTask, result: Dict[str, Any]) -> None:
        """Commit changes using Git MCP"""
        
        if not self.git_mcp:
            logger.warning("⚠️ Git MCP not available")
            return
        
        try:
            # Add files
            files_created = result.get("files_created", [])
            for file_path in files_created:
                await self.git_mcp.add_file(file_path)
            
            # Commit with descriptive message
            commit_message = f"Implement: {task.content[:50]}\n\nGenerated by Taisho (大将)\nFiles: {', '.join(files_created[:5])}"
            await self.git_mcp.commit(commit_message)
            
            logger.info(f"✅ Changes committed: {len(files_created)} files")
            
        except Exception as e:
            logger.warning(f"⚠️ Git commit failed: {e}")
    
    def _format_memory_context(self, memory_entries: List[Dict]) -> str:
        """Format memory entries for context"""
        if not memory_entries:
            return "No relevant memory entries found."
        
        formatted = []
        for entry in memory_entries[:3]:  # Top 3 most relevant
            formatted.append(f"- {entry.get('content', entry)}")
        
        return "\n".join(formatted)
    
    def _format_file_context(self, existing_files: List[Dict]) -> str:
        """Format existing file context"""
        if not existing_files:
            return "No existing files to reference."
        
        formatted = []
        for file_info in existing_files[:3]:  # Top 3 most relevant
            path = file_info.get("path", "unknown")
            content = file_info.get("content")
            if content:
                formatted.append(f"- {path}: {len(content)} chars")
            else:
                formatted.append(f"- {path}: (new file)")
        
        return "\n".join(formatted)