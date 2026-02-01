"""
Bushidan Multi-Agent System - BDI-Enhanced Taisho

Taisho agent with BDI (Belief-Desire-Intention) reasoning capabilities.
Adds formal multi-agent system theory to the implementation layer.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from core.bdi_framework import (
    BDIAgent, Belief, Desire, Intention,
    BeliefType, DesireType
)
from core.taisho import Taisho, ImplementationTask, ImplementationMode
from utils.logger import get_logger


logger = get_logger(__name__)


class BDITaisho(BDIAgent):
    """
    BDI-enhanced Taisho agent
    
    Adds formal BDI reasoning to the Taisho (Implementation Layer):
    - Beliefs: Code quality metrics, MCP tool availability, implementation context
    - Desires: Correct implementation, efficient execution, quality validation
    - Intentions: Plans for code generation and MCP operations
    """
    
    def __init__(self, orchestrator):
        super().__init__(agent_name="Taisho")
        
        # Initialize original Taisho
        self.taisho = Taisho(orchestrator)
        self.orchestrator = orchestrator
        
        # BDI mode flag
        self.bdi_mode_enabled = True
        
        logger.info("🧠 BDI-enhanced Taisho initialized")
    
    async def initialize(self) -> None:
        """Initialize Taisho and BDI components"""
        await self.taisho.initialize()
        
        # Initialize core beliefs
        self._initialize_core_beliefs()
        
        # Initialize implementation desires
        self._initialize_implementation_desires()
        
        logger.info("✅ BDI-Taisho initialization complete")
    
    def _initialize_core_beliefs(self) -> None:
        """Initialize core beliefs about implementation capabilities"""
        
        # Belief: MCP tools available
        mcp_tools = ["filesystem", "git", "memory"]
        self.belief_base.add_belief(Belief(
            id="available_mcp_tools",
            type=BeliefType.OPERATIONAL,
            content={"tools": mcp_tools, "count": len(mcp_tools)},
            confidence=1.0,
            source="system_init"
        ))
        
        # Belief: Local Qwen3-Coder available
        self.belief_base.add_belief(Belief(
            id="has_qwen3_coder",
            type=BeliefType.OPERATIONAL,
            content={"model": "qwen3-coder-30b-a3b", "available": True, "cost": 0.0},
            confidence=1.0,
            source="system_init"
        ))
        
        # Belief: Self-healing capability
        self.belief_base.add_belief(Belief(
            id="has_self_healing",
            type=BeliefType.OPERATIONAL,
            content={"capability": "error_correction", "max_attempts": 3},
            confidence=1.0,
            source="system_init"
        ))
        
        logger.debug("💭 Initialized core implementation beliefs")
    
    def _initialize_implementation_desires(self) -> None:
        """Initialize implementation-level desires"""
        
        # Desire: Generate correct, working code
        self.desire_set.add_desire(Desire(
            id="correct_implementation",
            type=DesireType.ACHIEVEMENT,
            description="Generate syntactically and semantically correct code",
            priority=1.0,
            feasibility=0.95,
            conditions=["has_qwen3_coder"]
        ))
        
        # Desire: Efficient resource usage
        self.desire_set.add_desire(Desire(
            id="efficient_execution",
            type=DesireType.OPTIMIZATION,
            description="Minimize memory usage and execution time",
            priority=0.7,
            feasibility=0.9
        ))
        
        # Desire: High quality validation
        self.desire_set.add_desire(Desire(
            id="quality_validation",
            type=DesireType.ACHIEVEMENT,
            description="Validate code quality before submission",
            priority=0.85,
            feasibility=1.0,
            conditions=["has_self_healing"]
        ))
        
        # Desire: Learn from corrections
        self.desire_set.add_desire(Desire(
            id="learn_from_corrections",
            type=DesireType.EXPLORATION,
            description="Learn from error corrections to improve future implementations",
            priority=0.6,
            feasibility=0.8,
            conditions=["has_self_healing", "has_memory_mcp"]
        ))
        
        logger.debug("🎯 Initialized implementation desires")
    
    async def execute_implementation(self, task: ImplementationTask) -> Dict[str, Any]:
        """
        Execute implementation with optional BDI reasoning
        
        Uses BDI cycle if enabled and requested
        """
        
        # Check if BDI mode requested
        use_bdi = self.bdi_mode_enabled and task.context and task.context.get("use_bdi", False)
        
        if use_bdi:
            logger.info("🧠 Executing implementation with BDI reasoning")
            return await self.bdi_cycle(task)
        else:
            # Standard processing via original Taisho
            return await self.taisho.execute_implementation(task)
    
    # BDI Agent abstract methods implementation
    
    async def perceive(self, observations: Dict[str, Any]) -> None:
        """
        Update beliefs based on implementation observations
        
        Perceives:
        - Task requirements and complexity
        - Available context from Memory MCP
        - File system state
        """
        
        task = observations.get("task")
        if not task:
            return
        
        # Perceive task requirements
        task_info = {
            "content": task.content,
            "mode": task.mode.value,
            "files_needed": task.files_needed or [],
            "dependencies": task.dependencies or []
        }
        
        self.belief_base.add_belief(Belief(
            id=f"task_requirements_{id(task)}",
            type=BeliefType.FACTUAL,
            content=task_info,
            confidence=1.0,
            source="task_observation",
            timestamp=observations.get("timestamp", datetime.now())
        ))
        
        # Perceive available context
        context = await self.taisho._gather_context(task)
        
        if context.get("memory_entries"):
            self.belief_base.add_belief(Belief(
                id=f"historical_context_{id(task)}",
                type=BeliefType.HISTORICAL,
                content={"memory_entries": context["memory_entries"]},
                confidence=0.8,
                source="memory_mcp"
            ))
        
        if context.get("existing_files"):
            self.belief_base.add_belief(Belief(
                id=f"existing_files_{id(task)}",
                type=BeliefType.CONTEXTUAL,
                content={"files": context["existing_files"]},
                confidence=0.9,
                source="filesystem_mcp"
            ))
        
        logger.debug(f"👁️ Perceived implementation task: {task.mode.value} mode")
    
    async def deliberate(self) -> Optional[Desire]:
        """
        Select implementation desire to pursue
        
        Considers:
        - Task requirements
        - Quality vs efficiency trade-off
        - Available tools and context
        """
        
        # Get feasible desires
        feasible_desires = self.desire_set.filter_feasible(self.belief_base)
        
        if not feasible_desires:
            return None
        
        # Get task requirements
        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if task_beliefs:
            task_info = task_beliefs[-1].content
            mode = task_info.get("mode", "standard")
            
            # Adjust priorities based on task mode
            for desire in feasible_desires:
                if mode == "heavy" and desire.id == "correct_implementation":
                    desire.priority = 1.0  # Correctness critical for heavy tasks
                elif mode == "lightweight" and desire.id == "efficient_execution":
                    desire.priority = 0.9  # Efficiency matters for lightweight
        
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
        Create implementation plan (intention) for the selected desire
        
        Plans include:
        - Context gathering
        - Implementation strategy
        - Validation approach
        - Git operations
        """
        
        # Get task requirements
        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if not task_beliefs:
            return None
        
        task_info = task_beliefs[-1].content
        mode = task_info.get("mode", "standard")
        
        plan = []
        
        if desire.id == "correct_implementation":
            # Correctness-focused plan with thorough validation
            plan = [
                {"action": "gather_context", "agent": "self"},
                {"action": "plan_implementation", "agent": "self"},
                {"action": "generate_code", "agent": "qwen"},
                {"action": "validate_syntax", "agent": "self"},
                {"action": "self_healing_check", "agent": "self"},
                {"action": "save_files", "agent": "filesystem_mcp"},
                {"action": "git_commit", "agent": "git_mcp"}
            ]
        
        elif desire.id == "efficient_execution":
            # Efficiency-focused plan
            if mode == "parallel":
                plan = [
                    {"action": "gather_context", "agent": "self"},
                    {"action": "parallel_implementation", "agent": "qwen", "parallel": True},
                    {"action": "validate_basic", "agent": "self"},
                    {"action": "save_files", "agent": "filesystem_mcp"}
                ]
            else:
                plan = [
                    {"action": "gather_context", "agent": "self"},
                    {"action": "generate_code", "agent": "qwen"},
                    {"action": "save_files", "agent": "filesystem_mcp"}
                ]
        
        elif desire.id == "quality_validation":
            # Quality-focused plan
            plan = [
                {"action": "gather_context", "agent": "self"},
                {"action": "plan_implementation", "agent": "self"},
                {"action": "generate_code", "agent": "qwen"},
                {"action": "comprehensive_validation", "agent": "self"},
                {"action": "self_healing_loop", "agent": "self"},
                {"action": "save_files", "agent": "filesystem_mcp"},
                {"action": "git_commit", "agent": "git_mcp"}
            ]
        
        elif desire.id == "learn_from_corrections":
            # Learning-focused plan
            plan = [
                {"action": "consult_memory", "agent": "memory"},
                {"action": "gather_context", "agent": "self"},
                {"action": "generate_code", "agent": "qwen"},
                {"action": "self_healing_loop", "agent": "self", "track_corrections": True},
                {"action": "log_corrections", "agent": "memory"},
                {"action": "save_files", "agent": "filesystem_mcp"}
            ]
        
        intention = Intention(
            id=f"intention_{desire.id}_{datetime.now().timestamp()}",
            desire_id=desire.id,
            plan=plan,
            metadata={"mode": mode}
        )
        
        logger.debug(f"📋 Created implementation plan with {len(plan)} steps")
        return intention
    
    async def execute(self, intention: Intention) -> Dict[str, Any]:
        """
        Execute the implementation plan
        
        Delegates to original Taisho methods while tracking execution
        """
        
        result = {"status": "executing", "steps_completed": [], "corrections": []}
        implementation_context = {}
        
        try:
            # Get task from beliefs
            task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
            if not task_beliefs:
                return {"error": "No task information", "status": "failed"}
            
            task_info = task_beliefs[-1].content
            
            # Reconstruct task
            task = ImplementationTask(
                content=task_info.get("content", ""),
                mode=ImplementationMode(task_info.get("mode", "standard")),
                files_needed=task_info.get("files_needed"),
                dependencies=task_info.get("dependencies")
            )
            
            for step in intention.plan:
                action = step["action"]
                
                if action == "gather_context" or action == "consult_memory":
                    implementation_context = await self.taisho._gather_context(task)
                    result["context_gathered"] = True
                
                elif action == "plan_implementation":
                    plan = await self.taisho._plan_implementation(task, implementation_context)
                    result["plan"] = plan
                
                elif action == "generate_code":
                    if task.mode == ImplementationMode.PARALLEL:
                        code_result = await self.taisho._execute_parallel_implementation(
                            task, result.get("plan", {}), implementation_context
                        )
                    else:
                        code_result = await self.taisho._execute_sequential_implementation(
                            task, result.get("plan", {}), implementation_context
                        )
                    result["implementation"] = code_result
                
                elif action in ["validate_syntax", "validate_basic", "comprehensive_validation"]:
                    validation = await self.taisho._validate_implementation(result.get("implementation", {}))
                    result["validation"] = validation
                
                elif action in ["self_healing_check", "self_healing_loop"]:
                    # Use self-healing if validation failed
                    if not result.get("validation", {}).get("valid", True):
                        track = step.get("track_corrections", False)
                        healing_result = await self._apply_self_healing(
                            result.get("implementation", {}),
                            track_corrections=track
                        )
                        result["implementation"] = healing_result["implementation"]
                        if track:
                            result["corrections"].extend(healing_result.get("corrections", []))
                
                elif action == "save_files":
                    files = result.get("implementation", {}).get("files_created", [])
                    result["files_saved"] = len(files)
                
                elif action == "git_commit":
                    if result.get("validation", {}).get("valid", False):
                        await self.taisho._commit_changes(task, result.get("implementation", {}))
                        result["git_committed"] = True
                
                elif action == "log_corrections":
                    if self.taisho.memory_mcp and result.get("corrections"):
                        try:
                            await self.taisho.memory_mcp.store({
                                "category": "implementation_corrections",
                                "corrections": result["corrections"],
                                "timestamp": datetime.now().isoformat()
                            })
                        except:
                            pass
                
                result["steps_completed"].append({"action": action, "status": "completed"})
            
            result["status"] = "completed"
            logger.info(f"✅ Implementation plan executed: {len(intention.plan)} steps")
            
        except Exception as e:
            logger.error(f"❌ Implementation plan execution failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        
        return result
    
    async def _apply_self_healing(self, implementation: Dict[str, Any], track_corrections: bool = False) -> Dict[str, Any]:
        """Apply self-healing to fix implementation issues"""
        
        corrections = []
        
        # Use Taisho's self-healing executor
        code = implementation.get("implementation", "")
        
        if code and self.taisho.self_healing:
            try:
                healing_result = await self.taisho.self_healing.run_with_healing(
                    code=code,
                    task_description="Fix code errors",
                    allow_installation=False
                )
                
                if healing_result.get("status") == "success":
                    implementation["implementation"] = healing_result.get("final_code", code)
                    
                    if track_corrections and healing_result.get("attempts", 0) > 1:
                        corrections.append({
                            "attempts": healing_result["attempts"],
                            "success": True
                        })
                        
            except Exception as e:
                logger.warning(f"⚠️ Self-healing failed: {e}")
        
        return {
            "implementation": implementation,
            "corrections": corrections
        }
