"""
Bushidan Multi-Agent System v9.1 - Karo (Tactical Layer)

The Karo serves as the tactical coordination layer using Gemini 2.0 Flash.
Handles task decomposition, Ashigaru coordination, and result integration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from utils.logger import get_logger
from utils.gemini_client import GeminiClient
from core.ashigaru import Ashigaru
from core.system_orchestrator import SystemOrchestrator


logger = get_logger(__name__)


@dataclass
class Subtask:
    """Individual subtask for Ashigaru execution"""
    id: str
    content: str
    dependencies: List[str]
    ashigaru_type: str  # filesystem, git, web_search, etc.
    priority: int = 1


class Karo:
    """
    家老 (Karo) - Tactical Coordination Layer
    
    Primary responsibilities:
    1. Task decomposition and analysis
    2. Subtask generation for Ashigaru
    3. Parallel execution coordination
    4. Result integration and synthesis
    5. Memory MCP consultation for context
    """
    
    def __init__(self, orchestrator: SystemOrchestrator):
        self.orchestrator = orchestrator
        self.gemini_client = GeminiClient(
            api_key=orchestrator.config.gemini_api_key
        )
        self.ashigaru: Optional[Ashigaru] = None
        self.memory_mcp = None
        
    async def initialize(self) -> None:
        """Initialize Karo and Ashigaru systems"""
        logger.info("🏛️ Initializing Karo (Tactical Layer)...")
        
        # Initialize Ashigaru (Execution Layer)
        self.ashigaru = Ashigaru(self.orchestrator)
        await self.ashigaru.initialize()
        
        # Get Memory MCP for context consultation
        self.memory_mcp = self.orchestrator.get_mcp("memory")
        
        logger.info("✅ Karo initialization complete")
    
    async def execute_task(self, task) -> Dict[str, Any]:
        """
        Execute task through tactical coordination
        
        1. Consult Memory MCP for relevant context
        2. Decompose task into subtasks
        3. Coordinate parallel Ashigaru execution
        4. Integrate and synthesize results
        """
        
        logger.info(f"🏛️ Karo executing task: {task.content[:50]}...")
        
        try:
            # Step 1: Consult Memory MCP for context
            context = await self._consult_memory(task)
            
            # Step 2: Decompose task into subtasks
            subtasks = await self._decompose_task(task, context)
            
            # Step 3: Execute subtasks in parallel
            if subtasks:
                results = await self._execute_subtasks(subtasks)
            else:
                # Direct execution if no decomposition needed
                results = [await self.ashigaru.execute_direct(task)]
            
            # Step 4: Integrate results
            final_result = await self._integrate_results(task, results)
            
            logger.info("✅ Karo task execution complete")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Karo task execution failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _consult_memory(self, task) -> Optional[Dict[str, Any]]:
        """Consult Memory MCP for relevant past decisions"""
        
        if not self.memory_mcp:
            return None
        
        try:
            # Search for relevant past decisions
            query = task.content[:100]  # First 100 chars as search query
            context = await self.memory_mcp.search(query)
            
            if context:
                logger.info(f"📚 Found {len(context)} relevant memory entries")
                return context
            
        except Exception as e:
            logger.warning(f"⚠️ Memory consultation failed: {e}")
        
        return None
    
    async def _decompose_task(self, task, context: Optional[Dict[str, Any]]) -> List[Subtask]:
        """Decompose task into subtasks using Gemini 2.0 Flash"""
        
        # Build decomposition prompt
        decomposition_prompt = f"""
        As the Karo (tactical coordinator) in the Bushidan Multi-Agent System v9.1, decompose this task:
        
        Task: {task.content}
        Complexity: {task.complexity.value}
        
        Available Ashigaru (execution units):
        - filesystem: File operations (read, write, search)
        - git: Version control operations  
        - memory: Knowledge storage and retrieval
        - web_search: Information gathering from web
        
        Past Context: {context or "None available"}
        
        If the task can be done directly without decomposition, respond with "DIRECT".
        Otherwise, break it down into subtasks and specify which Ashigaru should handle each.
        
        Format as JSON:
        {{
            "subtasks": [
                {{
                    "id": "1",
                    "content": "Specific task description",
                    "ashigaru_type": "filesystem",
                    "dependencies": [],
                    "priority": 1
                }}
            ]
        }}
        """
        
        try:
            response = await self.gemini_client.generate(
                prompt=decomposition_prompt,
                max_output_tokens=1000
            )
            
            if "DIRECT" in response.upper():
                logger.info("📋 Task will be executed directly")
                return []
            
            # Parse JSON response
            import json
            decomposition = json.loads(response)
            
            subtasks = []
            for st in decomposition.get("subtasks", []):
                subtask = Subtask(
                    id=st["id"],
                    content=st["content"],
                    ashigaru_type=st["ashigaru_type"],
                    dependencies=st.get("dependencies", []),
                    priority=st.get("priority", 1)
                )
                subtasks.append(subtask)
            
            logger.info(f"📋 Task decomposed into {len(subtasks)} subtasks")
            return subtasks
            
        except Exception as e:
            logger.warning(f"⚠️ Task decomposition failed, using direct execution: {e}")
            return []
    
    async def _execute_subtasks(self, subtasks: List[Subtask]) -> List[Dict[str, Any]]:
        """Execute subtasks in parallel with dependency management"""
        
        results = {}
        remaining_tasks = subtasks.copy()
        
        while remaining_tasks:
            # Find tasks with no unfulfilled dependencies
            ready_tasks = [
                task for task in remaining_tasks 
                if all(dep in results for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                logger.error("❌ Circular dependency detected in subtasks")
                break
            
            # Execute ready tasks in parallel
            async def execute_subtask(subtask):
                return await self.ashigaru.execute_subtask(subtask)
            
            # Run parallel execution
            parallel_results = await asyncio.gather(
                *[execute_subtask(task) for task in ready_tasks],
                return_exceptions=True
            )
            
            # Process results
            for task, result in zip(ready_tasks, parallel_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Subtask {task.id} failed: {result}")
                    results[task.id] = {"error": str(result), "status": "failed"}
                else:
                    results[task.id] = result
                
                remaining_tasks.remove(task)
            
            logger.info(f"✅ Completed {len(ready_tasks)} subtasks")
        
        return list(results.values())
    
    async def _integrate_results(self, task, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate subtask results into final response"""
        
        if len(results) == 1:
            return results[0]
        
        # Use Gemini to synthesize multiple results
        integration_prompt = f"""
        As the Karo, integrate these subtask results for the original task:
        
        Original Task: {task.content}
        
        Subtask Results:
        {chr(10).join([f"Result {i+1}: {str(result)}" for i, result in enumerate(results)])}
        
        Provide a synthesized, coherent final result that addresses the original task.
        """
        
        try:
            integrated_response = await self.gemini_client.generate(
                prompt=integration_prompt,
                max_output_tokens=1500
            )
            
            return {
                "status": "completed",
                "result": integrated_response,
                "subtask_count": len(results),
                "handled_by": "karo"
            }
            
        except Exception as e:
            logger.error(f"❌ Result integration failed: {e}")
            return {
                "status": "integration_failed",
                "raw_results": results,
                "error": str(e)
            }