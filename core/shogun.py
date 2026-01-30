"""
Bushidan Multi-Agent System v9.1 - Shogun (Strategic Layer)

The Shogun serves as the highest decision-making authority using Claude Sonnet 4.5.
Handles complexity assessment, delegation to Karo, and final quality assurance.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from utils.logger import get_logger
from utils.claude_client import ClaudeClient
from core.karo import Karo
from core.system_orchestrator import SystemOrchestrator


logger = get_logger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for delegation decisions"""
    SIMPLE = "simple"      # 10 seconds - Direct answers, simple queries
    MEDIUM = "medium"      # 25 seconds - Standard implementation tasks
    COMPLEX = "complex"    # 40 seconds - Multi-component architecture
    STRATEGIC = "strategic" # 60 seconds - High-level design decisions


@dataclass
class Task:
    """Task representation for the system"""
    content: str
    complexity: TaskComplexity
    context: Optional[Dict[str, Any]] = None
    priority: int = 1
    source: str = "direct"  # direct, slack, ha_os


class Shogun:
    """
    将軍 (Shogun) - Strategic Decision Layer
    
    Primary responsibilities:
    1. Task intake and complexity assessment
    2. Strategic decision making
    3. Delegation to Karo (Tactical Layer)
    4. Final quality assurance and approval
    5. Ethical and security oversight
    """
    
    def __init__(self, orchestrator: SystemOrchestrator):
        self.orchestrator = orchestrator
        self.claude_client = ClaudeClient(
            api_key=orchestrator.config.claude_api_key
        )
        self.karo: Optional[Karo] = None
        self.memory_mcp = None
        
    async def initialize(self) -> None:
        """Initialize Shogun and subordinate systems"""
        logger.info("🎌 Initializing Shogun (Strategic Layer)...")
        
        # Initialize Karo (Tactical Layer)
        self.karo = Karo(self.orchestrator)
        await self.karo.initialize()
        
        # Get Memory MCP for decision logging
        self.memory_mcp = self.orchestrator.get_mcp("memory")
        
        logger.info("✅ Shogun initialization complete")
    
    async def start_service(self) -> None:
        """Start the main service loop"""
        logger.info("🏯 Shogun service started - Ready for commands")
        
        # Main event loop - simplified for v9.1
        # In production, this would connect to various interfaces:
        # - CLI input
        # - Slack webhooks  
        # - HA OS integration
        # - REST API endpoints
        
        while True:
            try:
                await asyncio.sleep(1)  # Placeholder event loop
            except KeyboardInterrupt:
                logger.info("📴 Shogun service stopping...")
                break
            except Exception as e:
                logger.error(f"❌ Error in service loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Main task processing pipeline
        
        1. Assess complexity and validate task
        2. Make strategic decisions if needed
        3. Delegate to Karo for execution
        4. Review and approve results
        5. Log important decisions to Memory MCP
        """
        
        logger.info(f"🎌 Shogun processing task: {task.content[:50]}...")
        
        try:
            # Step 1: Complexity assessment and validation
            assessed_complexity = await self._assess_complexity(task)
            task.complexity = assessed_complexity
            
            # Step 2: Strategic decision (if Strategic level)
            if task.complexity == TaskComplexity.STRATEGIC:
                result = await self._handle_strategic_task(task)
            else:
                # Step 3: Delegate to Karo for tactical execution
                result = await self.karo.execute_task(task)
                
                # Step 4: Review and approve results
                result = await self._review_results(task, result)
            
            # Step 5: Log important decisions
            await self._log_decision(task, result)
            
            logger.info("✅ Shogun task processing complete")
            return result
            
        except Exception as e:
            logger.error(f"❌ Shogun task processing failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _assess_complexity(self, task: Task) -> TaskComplexity:
        """Assess task complexity using Claude Sonnet 4.5"""
        
        assessment_prompt = f"""
        As the Shogun (strategic decision maker) in a Japanese-inspired AI system, assess the complexity of this task:
        
        Task: {task.content}
        
        Classify as:
        - SIMPLE: Direct answers, information queries, simple modifications
        - MEDIUM: Standard implementation, typical development tasks
        - COMPLEX: Multi-component systems, architectural changes
        - STRATEGIC: High-level decisions, technology choices, major design
        
        Respond with just the classification: SIMPLE, MEDIUM, COMPLEX, or STRATEGIC
        """
        
        try:
            response = await self.claude_client.generate(
                messages=[{"role": "user", "content": assessment_prompt}],
                max_tokens=10
            )
            
            complexity_str = response.strip().upper()
            complexity = TaskComplexity(complexity_str.lower())
            
            logger.info(f"📊 Task complexity assessed: {complexity.value}")
            return complexity
            
        except Exception as e:
            logger.warning(f"⚠️ Complexity assessment failed, defaulting to MEDIUM: {e}")
            return TaskComplexity.MEDIUM
    
    async def _handle_strategic_task(self, task: Task) -> Dict[str, Any]:
        """Handle strategic-level tasks directly with Claude"""
        
        strategic_prompt = f"""
        As the Shogun (highest authority) in the Bushidan Multi-Agent System v9.1, handle this strategic task:
        
        Task: {task.content}
        Context: {task.context or "None provided"}
        
        This is a STRATEGIC level decision requiring your highest-level analysis. Consider:
        - Technical implications
        - Resource requirements
        - Long-term consequences
        - Security and ethical considerations
        
        Provide a comprehensive strategic response.
        """
        
        response = await self.claude_client.generate(
            messages=[{"role": "user", "content": strategic_prompt}],
            max_tokens=2000
        )
        
        return {
            "status": "completed",
            "result": response,
            "complexity": "strategic",
            "handled_by": "shogun"
        }
    
    async def _review_results(self, task: Task, result: Dict[str, Any]) -> Dict[str, Any]:
        """Review and approve results from Karo"""
        
        if result.get("status") != "completed":
            return result  # Pass through failed tasks
        
        review_prompt = f"""
        As the Shogun, review this completed task:
        
        Original Task: {task.content}
        Result: {result.get("result", "No result provided")}
        
        Check for:
        - Quality and correctness
        - Security considerations
        - Adherence to best practices
        - Completeness
        
        If approved, respond with "APPROVED". If issues found, provide feedback.
        """
        
        try:
            review = await self.claude_client.generate(
                messages=[{"role": "user", "content": review_prompt}],
                max_tokens=500
            )
            
            if "APPROVED" in review.upper():
                result["shogun_approval"] = "approved"
                logger.info("✅ Shogun approved task results")
            else:
                result["shogun_feedback"] = review
                result["shogun_approval"] = "feedback_provided"
                logger.info("📝 Shogun provided feedback on task results")
                
        except Exception as e:
            logger.warning(f"⚠️ Review process failed: {e}")
            result["shogun_approval"] = "review_failed"
        
        return result
    
    async def _log_decision(self, task: Task, result: Dict[str, Any]) -> None:
        """Log important decisions to Memory MCP"""
        
        if not self.memory_mcp or task.complexity not in [TaskComplexity.COMPLEX, TaskComplexity.STRATEGIC]:
            return
        
        decision_log = {
            "timestamp": asyncio.get_event_loop().time(),
            "category": "decision",
            "task": task.content,
            "complexity": task.complexity.value,
            "status": result.get("status"),
            "approval": result.get("shogun_approval")
        }
        
        try:
            await self.memory_mcp.store(decision_log)
            logger.info("📝 Decision logged to Memory MCP")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log decision: {e}")