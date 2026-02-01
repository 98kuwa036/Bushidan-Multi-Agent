"""
Bushidan Multi-Agent System - BDI-Enhanced Karo

Karo agent with BDI (Belief-Desire-Intention) reasoning capabilities.
Adds formal multi-agent system theory to the tactical coordination layer.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.bdi_framework import (
    BDIAgent, Belief, Desire, Intention,
    BeliefType, DesireType
)
from core.karo import Karo
from utils.logger import get_logger


logger = get_logger(__name__)


class BDIKaro(BDIAgent):
    """
    BDI-enhanced Karo agent
    
    Adds formal BDI reasoning to the Karo (Tactical Layer):
    - Beliefs: Task decomposition insights, resource availability, coordination state
    - Desires: Efficient decomposition, parallel execution, quality integration
    - Intentions: Plans for task breakdown and Ashigaru coordination
    """
    
    def __init__(self, orchestrator):
        super().__init__(agent_name="Karo")
        
        # Initialize original Karo
        self.karo = Karo(orchestrator)
        self.orchestrator = orchestrator
        
        # BDI mode flag
        self.bdi_mode_enabled = True
        
        logger.info("🧠 BDI-enhanced Karo initialized")
    
    async def initialize(self) -> None:
        """Initialize Karo and BDI components"""
        await self.karo.initialize()
        
        # Initialize core beliefs
        self._initialize_core_beliefs()
        
        # Initialize tactical desires
        self._initialize_tactical_desires()
        
        logger.info("✅ BDI-Karo initialization complete")
    
    def _initialize_core_beliefs(self) -> None:
        """Initialize core beliefs about coordination capabilities"""
        
        # Belief: System has parallel execution capability
        self.belief_base.add_belief(Belief(
            id="has_parallel_execution",
            type=BeliefType.OPERATIONAL,
            content={"capability": "parallel_ashigaru_coordination", "available": True},
            confidence=1.0,
            source="system_init"
        ))
        
        # Belief: System has task decomposition capability
        self.belief_base.add_belief(Belief(
            id="has_decomposition",
            type=BeliefType.OPERATIONAL,
            content={"capability": "task_decomposition", "available": True},
            confidence=1.0,
            source="system_init"
        ))
        
        # Belief: Ashigaru types available
        ashigaru_types = ["filesystem", "git", "memory", "web_search"]
        self.belief_base.add_belief(Belief(
            id="available_ashigaru",
            type=BeliefType.OPERATIONAL,
            content={"ashigaru_types": ashigaru_types, "count": len(ashigaru_types)},
            confidence=1.0,
            source="system_init"
        ))
        
        logger.debug("💭 Initialized core tactical beliefs")
    
    def _initialize_tactical_desires(self) -> None:
        """Initialize tactical-level desires"""
        
        # Desire: Efficient task decomposition
        self.desire_set.add_desire(Desire(
            id="efficient_decomposition",
            type=DesireType.OPTIMIZATION,
            description="Decompose tasks efficiently for parallel execution",
            priority=0.9,
            feasibility=1.0,
            conditions=["has_decomposition"]
        ))
        
        # Desire: Maximize parallelization
        self.desire_set.add_desire(Desire(
            id="maximize_parallelization",
            type=DesireType.OPTIMIZATION,
            description="Maximize parallel execution of subtasks",
            priority=0.8,
            feasibility=0.9,
            conditions=["has_parallel_execution", "available_ashigaru"]
        ))
        
        # Desire: Maintain coordination quality
        self.desire_set.add_desire(Desire(
            id="maintain_coordination_quality",
            type=DesireType.MAINTENANCE,
            description="Maintain high quality in result integration",
            priority=0.85,
            feasibility=1.0
        ))
        
        # Desire: Learn optimal decomposition patterns
        self.desire_set.add_desire(Desire(
            id="learn_decomposition_patterns",
            type=DesireType.EXPLORATION,
            description="Learn which decomposition strategies work best",
            priority=0.6,
            feasibility=0.7,
            conditions=["has_memory_mcp"]
        ))
        
        logger.debug("🎯 Initialized tactical desires")
    
    async def execute_task(self, task) -> Dict[str, Any]:
        """
        Execute task with optional BDI reasoning
        
        Uses BDI cycle if enabled and requested, otherwise delegates to original Karo
        """
        
        # Check if BDI mode requested
        use_bdi = self.bdi_mode_enabled and hasattr(task, 'context') and task.context and task.context.get("use_bdi", False)
        
        if use_bdi:
            logger.info("🧠 Executing task with BDI reasoning")
            return await self.bdi_cycle(task)
        else:
            # Standard processing via original Karo
            return await self.karo.execute_task(task)
    
    # BDI Agent abstract methods implementation
    
    async def perceive(self, observations: Dict[str, Any]) -> None:
        """
        Update beliefs based on task observations
        
        Perceives:
        - Task decomposability
        - Available Ashigaru resources
        - Historical decomposition patterns
        """
        
        task = observations.get("task")
        if not task:
            return
        
        # Perceive task decomposability
        context = await self.karo._consult_memory(task)
        
        # Add belief about historical context
        if context:
            self.belief_base.add_belief(Belief(
                id=f"task_context_{id(task)}",
                type=BeliefType.CONTEXTUAL,
                content={"context": context, "available": True},
                confidence=0.8,
                source="memory_mcp",
                timestamp=observations.get("timestamp", datetime.now())
            ))
        
        # Perceive task characteristics
        task_info = {
            "content": task.content,
            "complexity": task.complexity.value if hasattr(task, 'complexity') else "unknown",
            "estimated_subtasks": self._estimate_subtask_count(task)
        }
        
        self.belief_base.add_belief(Belief(
            id=f"task_info_{id(task)}",
            type=BeliefType.FACTUAL,
            content=task_info,
            confidence=0.9,
            source="task_analysis"
        ))
        
        logger.debug(f"👁️ Perceived task: ~{task_info['estimated_subtasks']} potential subtasks")
    
    def _estimate_subtask_count(self, task) -> int:
        """Estimate number of subtasks needed"""
        # Simple heuristic based on task content length and keywords
        content = task.content
        
        # Count action words that suggest multiple operations
        action_keywords = ["implement", "create", "update", "test", "document", "refactor"]
        action_count = sum(1 for keyword in action_keywords if keyword in content.lower())
        
        # Estimate based on length and actions
        estimated = max(1, min(action_count, 5))
        
        return estimated
    
    async def deliberate(self) -> Optional[Desire]:
        """
        Select tactical desire to pursue
        
        Considers:
        - Task decomposability
        - Available resources
        - Parallelization potential
        """
        
        # Get feasible desires
        feasible_desires = self.desire_set.filter_feasible(self.belief_base)
        
        if not feasible_desires:
            return None
        
        # Get task info belief
        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if task_beliefs:
            task_info = task_beliefs[-1].content
            estimated_subtasks = task_info.get("estimated_subtasks", 1)
            
            # Adjust feasibility based on task characteristics
            for desire in feasible_desires:
                if desire.id == "maximize_parallelization" and estimated_subtasks < 2:
                    desire.feasibility = 0.3  # Low parallelization potential
                elif desire.id == "efficient_decomposition" and estimated_subtasks >= 3:
                    desire.feasibility = 1.0  # High decomposition value
        
        # Select highest priority feasible desire
        selected = sorted(
            feasible_desires,
            key=lambda d: d.priority * d.feasibility,
            reverse=True
        )[0]
        
        logger.debug(f"🎯 Selected desire: {selected.description}")
        return selected
    
    async def plan(self, desire: Desire) -> Optional[Intention]:
        """
        Create coordination plan (intention) for the selected desire
        
        Plans include:
        - Task decomposition strategy
        - Ashigaru allocation
        - Integration approach
        """
        
        # Get task info
        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if not task_beliefs:
            return None
        
        task_info = task_beliefs[-1].content
        estimated_subtasks = task_info.get("estimated_subtasks", 1)
        
        plan = []
        
        if desire.id == "efficient_decomposition":
            # Decomposition-focused plan
            if estimated_subtasks > 1:
                plan = [
                    {"action": "consult_memory", "agent": "memory"},
                    {"action": "decompose_task", "agent": "self"},
                    {"action": "execute_subtasks", "agent": "ashigaru"},
                    {"action": "integrate_results", "agent": "self"}
                ]
            else:
                plan = [
                    {"action": "direct_execution", "agent": "ashigaru"}
                ]
        
        elif desire.id == "maximize_parallelization":
            # Parallelization-focused plan
            plan = [
                {"action": "decompose_task", "agent": "self"},
                {"action": "parallel_execution", "agent": "ashigaru", "parallel": True},
                {"action": "integrate_results", "agent": "self"}
            ]
        
        elif desire.id == "maintain_coordination_quality":
            # Quality-focused plan
            plan = [
                {"action": "decompose_task", "agent": "self"},
                {"action": "execute_subtasks", "agent": "ashigaru"},
                {"action": "validate_consistency", "agent": "self"},
                {"action": "integrate_results", "agent": "self"}
            ]
        
        elif desire.id == "learn_decomposition_patterns":
            # Learning-focused plan
            plan = [
                {"action": "consult_memory", "agent": "memory"},
                {"action": "decompose_task", "agent": "self"},
                {"action": "execute_subtasks", "agent": "ashigaru"},
                {"action": "integrate_results", "agent": "self"},
                {"action": "log_pattern", "agent": "memory"}
            ]
        
        intention = Intention(
            id=f"intention_{desire.id}_{datetime.now().timestamp()}",
            desire_id=desire.id,
            plan=plan,
            metadata={"estimated_subtasks": estimated_subtasks}
        )
        
        logger.debug(f"📋 Created plan with {len(plan)} steps")
        return intention
    
    async def execute(self, intention: Intention) -> Dict[str, Any]:
        """
        Execute the coordination plan
        
        Delegates to original Karo methods while tracking execution
        """
        
        result = {"status": "executing", "steps_completed": []}
        subtasks = []
        subtask_results = []
        
        try:
            # Get task from beliefs
            task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
            if not task_beliefs:
                return {"error": "No task information", "status": "failed"}
            
            task_info = task_beliefs[-1].content
            
            # Reconstruct task (simplified)
            class SimpleTask:
                def __init__(self, content, complexity):
                    self.content = content
                    self.complexity = complexity
            
            from core.shogun import TaskComplexity
            task = SimpleTask(
                content=task_info.get("content", ""),
                complexity=TaskComplexity(task_info.get("complexity", "medium"))
            )
            
            for step in intention.plan:
                action = step["action"]
                
                if action == "consult_memory":
                    context = await self.karo._consult_memory(task)
                    result["context"] = context
                
                elif action == "decompose_task":
                    context = result.get("context")
                    subtasks = await self.karo._decompose_task(task, context)
                    result["subtask_count"] = len(subtasks)
                
                elif action in ["execute_subtasks", "parallel_execution"]:
                    if subtasks:
                        subtask_results = await self.karo._execute_subtasks(subtasks)
                    else:
                        # Direct execution
                        direct_result = await self.karo.ashigaru.execute_direct(task)
                        subtask_results = [direct_result]
                
                elif action == "validate_consistency":
                    # Check consistency of subtask results
                    consistency = self._validate_result_consistency(subtask_results)
                    result["consistency_check"] = consistency
                
                elif action == "integrate_results":
                    if subtask_results:
                        integrated = await self.karo._integrate_results(task, subtask_results)
                        result.update(integrated)
                
                elif action == "direct_execution":
                    direct_result = await self.karo.ashigaru.execute_direct(task)
                    result.update(direct_result)
                
                elif action == "log_pattern":
                    # Log successful decomposition pattern
                    if self.karo.memory_mcp and subtasks:
                        pattern = {
                            "task_type": task_info.get("complexity"),
                            "subtask_count": len(subtasks),
                            "success": result.get("status") == "completed"
                        }
                        try:
                            await self.karo.memory_mcp.store({
                                "category": "decomposition_pattern",
                                "pattern": pattern,
                                "timestamp": datetime.now().isoformat()
                            })
                        except:
                            pass
                
                result["steps_completed"].append({"action": action, "status": "completed"})
            
            result["status"] = "completed"
            logger.info(f"✅ Coordination plan executed: {len(intention.plan)} steps")
            
        except Exception as e:
            logger.error(f"❌ Coordination plan execution failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        
        return result
    
    def _validate_result_consistency(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate consistency of subtask results"""
        
        if not results:
            return {"consistent": True, "issues": []}
        
        issues = []
        
        # Check for failed subtasks
        failed_count = sum(1 for r in results if r.get("status") == "failed")
        if failed_count > 0:
            issues.append(f"{failed_count} subtasks failed")
        
        # Check for conflicting information (simplified)
        # In a full implementation, this would do semantic analysis
        
        return {
            "consistent": len(issues) == 0,
            "issues": issues,
            "checked_results": len(results)
        }
