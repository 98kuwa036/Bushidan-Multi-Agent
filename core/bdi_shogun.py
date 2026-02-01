"""
Bushidan Multi-Agent System - BDI-Enhanced Shogun

Shogun agent with BDI (Belief-Desire-Intention) reasoning capabilities.
Adds formal multi-agent system theory to the strategic decision layer.

This is a wrapper around the original Shogun that adds BDI semantics
without breaking existing functionality.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from core.bdi_framework import (
    BDIAgent, Belief, Desire, Intention,
    BeliefType, DesireType
)
from core.shogun import Shogun, Task, TaskComplexity
from utils.logger import get_logger


logger = get_logger(__name__)


class BDIShogun(BDIAgent):
    """
    BDI-enhanced Shogun agent
    
    Adds formal BDI reasoning to the Shogun (Strategic Layer):
    - Beliefs: Task complexity assessments, system state, past decisions
    - Desires: Strategic goals (quality, efficiency, learning)
    - Intentions: Plans for task delegation and quality assurance
    
    Maintains backward compatibility - can be used as drop-in replacement
    """
    
    def __init__(self, orchestrator):
        super().__init__(agent_name="Shogun")
        
        # Initialize original Shogun
        self.shogun = Shogun(orchestrator)
        self.orchestrator = orchestrator
        
        # BDI mode flag
        self.bdi_mode_enabled = True
        
        logger.info("🧠 BDI-enhanced Shogun initialized")
    
    async def initialize(self) -> None:
        """Initialize Shogun and BDI components"""
        await self.shogun.initialize()
        
        # Initialize core beliefs about system capabilities
        self._initialize_core_beliefs()
        
        # Initialize strategic desires
        self._initialize_strategic_desires()
        
        logger.info("✅ BDI-Shogun initialization complete")
    
    def _initialize_core_beliefs(self) -> None:
        """Initialize core beliefs about system capabilities"""
        
        # Belief: System has tactical coordination (Karo)
        self.belief_base.add_belief(Belief(
            id="has_karo",
            type=BeliefType.OPERATIONAL,
            content={"capability": "tactical_coordination", "available": True},
            confidence=1.0,
            source="system_init"
        ))
        
        # Belief: System has implementation capability (Taisho)
        self.belief_base.add_belief(Belief(
            id="has_taisho",
            type=BeliefType.OPERATIONAL,
            content={"capability": "implementation", "available": True},
            confidence=1.0,
            source="system_init"
        ))
        
        # Belief: System has quality assurance (Opus)
        self.belief_base.add_belief(Belief(
            id="has_opus",
            type=BeliefType.OPERATIONAL,
            content={"capability": "premium_quality_assurance", "available": True},
            confidence=1.0,
            source="system_init"
        ))
        
        logger.debug("💭 Initialized core operational beliefs")
    
    def _initialize_strategic_desires(self) -> None:
        """Initialize strategic-level desires"""
        
        # Desire: Maintain high quality standards
        self.desire_set.add_desire(Desire(
            id="maintain_quality",
            type=DesireType.MAINTENANCE,
            description="Maintain high quality standards (95+ points)",
            priority=0.9,
            feasibility=1.0
        ))
        
        # Desire: Optimize cost-efficiency
        self.desire_set.add_desire(Desire(
            id="optimize_cost",
            type=DesireType.OPTIMIZATION,
            description="Optimize cost while maintaining quality",
            priority=0.7,
            feasibility=1.0
        ))
        
        # Desire: Learn from past decisions
        self.desire_set.add_desire(Desire(
            id="learn_and_improve",
            type=DesireType.EXPLORATION,
            description="Learn from past decisions to improve future performance",
            priority=0.6,
            feasibility=0.8,
            conditions=["has_memory_mcp"]
        ))
        
        logger.debug("🎯 Initialized strategic desires")
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Process task with optional BDI reasoning
        
        If BDI mode is enabled and task context requests it, uses BDI cycle.
        Otherwise, delegates to original Shogun for backward compatibility.
        """
        
        # Check if BDI mode requested
        use_bdi = self.bdi_mode_enabled and task.context and task.context.get("use_bdi", False)
        
        if use_bdi:
            logger.info("🧠 Processing task with BDI reasoning")
            return await self.bdi_cycle(task)
        else:
            # Standard processing via original Shogun
            return await self.shogun.process_task(task)
    
    # BDI Agent abstract methods implementation
    
    async def perceive(self, observations: Dict[str, Any]) -> None:
        """
        Update beliefs based on task observations
        
        Perceives:
        - Task complexity assessment
        - Available resources
        - Historical context from Memory MCP
        """
        
        task = observations.get("task")
        if not task:
            return
        
        # Perceive task complexity
        complexity = await self.shogun._assess_complexity(task)
        
        self.belief_base.add_belief(Belief(
            id=f"task_complexity_{id(task)}",
            type=BeliefType.FACTUAL,
            content={"task": task.content, "complexity": complexity.value},
            confidence=0.9,
            source="complexity_assessment",
            timestamp=observations.get("timestamp", datetime.now())
        ))
        
        # Perceive historical context from Memory MCP
        if self.shogun.memory_mcp:
            try:
                memory_entries = await self.shogun.memory_mcp.search(task.content[:100])
                if memory_entries:
                    self.belief_base.add_belief(Belief(
                        id=f"historical_context_{id(task)}",
                        type=BeliefType.HISTORICAL,
                        content={"entries": memory_entries},
                        confidence=0.8,
                        source="memory_mcp"
                    ))
            except Exception as e:
                logger.warning(f"⚠️ Failed to query Memory MCP: {e}")
        
        logger.debug(f"👁️ Perceived task with complexity: {complexity.value}")
    
    async def deliberate(self) -> Optional[Desire]:
        """
        Select strategic desire to pursue
        
        Considers:
        - Current beliefs about task complexity
        - System capabilities
        - Strategic priorities (quality vs cost vs learning)
        """
        
        # Get feasible desires based on current beliefs
        feasible_desires = self.desire_set.filter_feasible(self.belief_base)
        
        if not feasible_desires:
            return None
        
        # Select highest priority feasible desire
        top_desires = sorted(
            feasible_desires,
            key=lambda d: d.priority * d.feasibility,
            reverse=True
        )
        
        selected = top_desires[0]
        logger.debug(f"🎯 Selected desire: {selected.description}")
        
        return selected
    
    async def plan(self, desire: Desire) -> Optional[Intention]:
        """
        Create execution plan (intention) for the selected desire
        
        Plans include:
        - Delegation strategy (to Karo or direct handling)
        - Quality assurance approach (Basic/Detailed/Premium)
        - Resource allocation
        """
        
        # Get task complexity belief
        complexity_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if not complexity_beliefs:
            return None
        
        latest_complexity = complexity_beliefs[-1]
        complexity_value = latest_complexity.content.get("complexity", "medium")
        
        # Create plan based on desire and complexity
        plan = []
        
        if desire.id == "maintain_quality":
            # Quality-focused plan
            if complexity_value == "strategic":
                plan = [
                    {"action": "handle_strategic_directly", "agent": "self"},
                    {"action": "opus_premium_review", "agent": "opus"}
                ]
            else:
                plan = [
                    {"action": "delegate_to_karo", "agent": "karo"},
                    {"action": "adaptive_quality_review", "agent": "self"}
                ]
        
        elif desire.id == "optimize_cost":
            # Cost-optimized plan
            plan = [
                {"action": "delegate_to_karo", "agent": "karo"},
                {"action": "basic_review", "agent": "self"}
            ]
        
        elif desire.id == "learn_and_improve":
            # Learning-focused plan
            plan = [
                {"action": "consult_memory_mcp", "agent": "memory"},
                {"action": "delegate_to_karo", "agent": "karo"},
                {"action": "log_decision", "agent": "memory"}
            ]
        
        intention = Intention(
            id=f"intention_{desire.id}_{datetime.now().timestamp()}",
            desire_id=desire.id,
            plan=plan,
            metadata={"complexity": complexity_value}
        )
        
        logger.debug(f"📋 Created plan with {len(plan)} steps")
        return intention
    
    async def execute(self, intention: Intention) -> Dict[str, Any]:
        """
        Execute the intention plan
        
        Delegates to original Shogun methods while tracking execution
        """
        
        result = {"status": "executing", "steps_completed": []}
        
        try:
            for step in intention.plan:
                action = step["action"]
                
                if action == "handle_strategic_directly":
                    # Get task from beliefs
                    task_belief = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)[-1]
                    task_content = task_belief.content.get("task", "")
                    
                    # Create Task object
                    task = Task(
                        content=task_content,
                        complexity=TaskComplexity.STRATEGIC
                    )
                    
                    step_result = await self.shogun._handle_strategic_task(task)
                
                elif action == "delegate_to_karo":
                    # Get task from beliefs
                    task_belief = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)[-1]
                    task_content = task_belief.content.get("task", "")
                    complexity = task_belief.content.get("complexity", "medium")
                    
                    task = Task(
                        content=task_content,
                        complexity=TaskComplexity(complexity)
                    )
                    
                    step_result = await self.shogun.karo.execute_task(task)
                    result.update(step_result)
                
                elif action in ["adaptive_quality_review", "basic_review", "opus_premium_review"]:
                    # Quality review step
                    task_belief = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)[-1]
                    task_content = task_belief.content.get("task", "")
                    complexity = task_belief.content.get("complexity", "medium")
                    
                    task = Task(
                        content=task_content,
                        complexity=TaskComplexity(complexity)
                    )
                    
                    if action == "opus_premium_review" and self.shogun.opus_client:
                        result = await self.shogun._adaptive_review(task, result)
                    else:
                        result = await self.shogun._adaptive_review(task, result)
                
                elif action == "consult_memory_mcp" or action == "log_decision":
                    # Memory operations - already handled by perceive/reconsider
                    step_result = {"action": action, "status": "completed"}
                
                result["steps_completed"].append({"action": action, "status": "completed"})
            
            result["status"] = "completed"
            logger.info(f"✅ Intention executed: {len(intention.plan)} steps")
            
        except Exception as e:
            logger.error(f"❌ Intention execution failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        
        return result
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get review statistics including BDI state"""
        stats = self.shogun.get_review_statistics()
        stats["bdi_state"] = self.get_agent_state()
        return stats
