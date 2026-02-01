"""
Bushidan Multi-Agent System v9.3.2 - Karo (Enhanced Tactical Layer)

The Karo serves as the enhanced tactical coordination layer supporting 4-tier architecture.
Coordinates between Taisho (implementation) and Ashigaru (support) with intelligent routing.

v9.3.2 Enhancements:
- Gemini 3.0 Flash integration (final defense line)
- Groq integration for simple task speed
- Routing decision execution from Shogun
- Dynamic client selection based on task complexity
- Taisho coordination with fallback chain
- BDI Framework integration for formal tactical reasoning
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from utils.logger import get_logger
from core.system_orchestrator import SystemOrchestrator
from core.bdi_framework import (
    BeliefBase, DesireSet, IntentionStack,
    Belief, Desire, Intention, BeliefType, DesireType
)


logger = get_logger(__name__)


class TaskDelegation(Enum):
    """Task delegation strategies in 4-tier architecture"""
    TAISHO_PRIMARY = "taisho_primary"      # Heavy implementation to Taisho
    ASHIGARU_PARALLEL = "ashigaru_parallel"  # Light tasks to Ashigaru
    HYBRID_COORDINATION = "hybrid_coordination"  # Taisho + Ashigaru coordination
    GROQ_INSTANT = "groq_instant"          # v9.3.2: Simple tasks to Groq
    GEMINI_DEFENSE = "gemini_defense"      # v9.3.2: Final defense line


@dataclass
class EnhancedSubtask:
    """Enhanced subtask for 4-tier coordination"""
    id: str
    content: str
    dependencies: List[str]
    delegation_target: str  # "taisho", "ashigaru", "groq", "gemini"
    complexity: str  # "simple", "medium", "complex"
    priority: int = 1
    estimated_time: int = 30
    groq_eligible: bool = False


@dataclass
class Subtask:
    """Basic subtask for backward compatibility"""
    id: str
    content: str
    ashigaru_type: str
    dependencies: List[str]
    priority: int = 1


class Karo:
    """
    家老 (Karo) - Enhanced Tactical Coordination Layer v9.3.2

    Enhanced responsibilities:
    1. Execute routing decisions from Shogun
    2. Dynamic client selection (Groq for speed, Gemini3 for defense)
    3. Taisho coordination for heavy implementation
    4. 3-tier fallback chain management
    5. Ashigaru coordination for parallel support
    6. DSPy-optimized task decomposition

    v9.3.2 Features:
    - Intelligent routing execution
    - Gemini 3.0 Flash as final defense
    - Groq for instant simple task response
    """

    VERSION = "9.3.2"

    def __init__(self, orchestrator: SystemOrchestrator):
        self.orchestrator = orchestrator

        # AI clients (initialized from orchestrator)
        self.gemini3_client = None
        self.gemini_client = None
        self.groq_client = None

        # Components
        self.dspy_client = None
        self.taisho = None
        self.ashigaru = None
        self.memory_mcp = None
        self.web_search_mcp = None

        # BDI Framework components
        self.belief_base = BeliefBase()
        self.desire_set = DesireSet()
        self.intention_stack = IntentionStack()
        self.bdi_enabled = True

        # Statistics
        self.execution_stats = {
            "total_tasks": 0,
            "by_delegation": {d.value: 0 for d in TaskDelegation},
            "fallback_count": 0,
            "total_time_seconds": 0.0
        }
        self.bdi_stats = {
            "bdi_cycles": 0,
            "decompositions": 0,
            "parallel_executions": 0
        }

    async def initialize(self) -> None:
        """Initialize Karo and subordinate systems"""
        logger.info(f"🏛️ Initializing Karo v{self.VERSION} (Tactical Layer)...")

        # Get AI clients from orchestrator
        self.gemini3_client = self.orchestrator.get_client("gemini3")
        if not self.gemini3_client:
            self.gemini_client = self.orchestrator.get_client("gemini")
            if self.gemini_client:
                logger.info("📝 Using standard Gemini client (Gemini3 not available)")

        self.groq_client = self.orchestrator.get_client("groq")
        if self.groq_client:
            logger.info("⚡ Groq client available for instant responses")

        # Initialize DSPy client
        try:
            from utils.dspy_client import DSPyClient
            self.dspy_client = DSPyClient()
            logger.info("✅ DSPy client initialized")
        except Exception as e:
            logger.warning(f"⚠️ DSPy client not available: {e}")

        # Initialize Taisho (Implementation Layer)
        try:
            from core.taisho import Taisho
            self.taisho = Taisho(self.orchestrator)
            await self.taisho.initialize()
            logger.info("✅ Taisho (Implementation Layer) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Taisho initialization failed: {e}")

        # Initialize Ashigaru (Execution Layer)
        try:
            from core.ashigaru import Ashigaru
            self.ashigaru = Ashigaru(self.orchestrator)
            await self.ashigaru.initialize()
            logger.info("✅ Ashigaru (Execution Layer) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Ashigaru initialization failed: {e}")

        # Get MCP connections
        self.memory_mcp = self.orchestrator.get_mcp("memory")
        self.web_search_mcp = self.orchestrator.get_mcp("web_search")

        # Initialize BDI Framework
        self._initialize_bdi()

        logger.info(f"✅ Karo v{self.VERSION} initialization complete (BDI enabled)")

    async def execute_task(self, task) -> Dict[str, Any]:
        """
        Execute task through tactical coordination (legacy method)

        For backward compatibility - delegates to execute_task_with_routing
        """
        return await self.execute_task_with_routing(task, None)

    async def execute_task_with_routing(self, task, routing_decision) -> Dict[str, Any]:
        """
        Execute task using routing decision from Shogun

        v9.3.2 routing logic:
        - Simple → Groq (instant, power-saving)
        - Medium → Taisho (Local Qwen3)
        - Complex → Taisho with fallback chain
        - Strategic → Handled by Shogun (shouldn't reach here)
        """

        start_time = time.time()
        logger.info(f"🏛️ Karo executing task: {task.content[:50]}...")

        try:
            # Determine delegation strategy
            delegation = self._determine_delegation(task, routing_decision)
            logger.info(f"📋 Delegation strategy: {delegation.value}")

            # Execute based on delegation
            if delegation == TaskDelegation.GROQ_INSTANT:
                result = await self._execute_with_groq(task)

            elif delegation == TaskDelegation.TAISHO_PRIMARY:
                result = await self._execute_with_taisho(task, routing_decision)

            elif delegation == TaskDelegation.GEMINI_DEFENSE:
                result = await self._execute_with_gemini(task, as_final_defense=True)

            elif delegation == TaskDelegation.ASHIGARU_PARALLEL:
                result = await self._execute_with_ashigaru(task)

            else:
                # Hybrid coordination
                result = await self._execute_hybrid(task, routing_decision)

            # Update statistics
            elapsed_time = time.time() - start_time
            self._update_stats(delegation, elapsed_time)

            logger.info(f"✅ Karo task complete in {elapsed_time:.1f}s")
            return result

        except Exception as e:
            logger.error(f"❌ Karo task execution failed: {e}")
            return {"error": str(e), "status": "failed"}

    def _determine_delegation(self, task, routing_decision) -> TaskDelegation:
        """Determine delegation strategy based on task and routing decision"""

        # Use routing decision target if available
        if routing_decision:
            try:
                from core.intelligent_router import RouteTarget

                target_map = {
                    RouteTarget.GROQ: TaskDelegation.GROQ_INSTANT,
                    RouteTarget.LOCAL_QWEN3: TaskDelegation.TAISHO_PRIMARY,
                    RouteTarget.CLOUD_QWEN3: TaskDelegation.TAISHO_PRIMARY,
                    RouteTarget.GEMINI3: TaskDelegation.GEMINI_DEFENSE,
                    RouteTarget.SHOGUN: TaskDelegation.TAISHO_PRIMARY  # Shouldn't happen
                }

                if hasattr(routing_decision, 'target'):
                    return target_map.get(routing_decision.target, TaskDelegation.TAISHO_PRIMARY)

            except Exception as e:
                logger.warning(f"⚠️ Routing decision parsing failed: {e}")

        # Fallback: Use task complexity
        complexity = getattr(task, 'complexity', None)
        if complexity:
            complexity_value = complexity.value if hasattr(complexity, 'value') else str(complexity)

            if complexity_value == "simple":
                if self.groq_client:
                    return TaskDelegation.GROQ_INSTANT
                return TaskDelegation.ASHIGARU_PARALLEL

            elif complexity_value in ["medium", "complex"]:
                return TaskDelegation.TAISHO_PRIMARY

        # Default: Taisho
        return TaskDelegation.TAISHO_PRIMARY

    async def _execute_with_groq(self, task) -> Dict[str, Any]:
        """Execute simple task with Groq for instant response"""

        if not self.groq_client:
            logger.warning("⚠️ Groq not available, falling back to Gemini")
            return await self._execute_with_gemini(task)

        logger.info("⚡ Executing with Groq (instant response)")

        try:
            response = await self.groq_client.generate(
                messages=[{"role": "user", "content": task.content}],
                max_tokens=1000
            )

            return {
                "status": "completed",
                "result": response,
                "handled_by": "groq",
                "power_saving": True
            }

        except Exception as e:
            logger.warning(f"⚠️ Groq failed: {e}, falling back to Gemini")
            self.execution_stats["fallback_count"] += 1
            return await self._execute_with_gemini(task, as_final_defense=True)

    async def _execute_with_taisho(self, task, routing_decision) -> Dict[str, Any]:
        """Execute implementation task with Taisho and fallback chain"""

        if not self.taisho:
            logger.warning("⚠️ Taisho not available, using Gemini defense")
            return await self._execute_with_gemini(task, as_final_defense=True)

        logger.info("🏯 Executing with Taisho (Implementation Layer)")

        try:
            # Convert task to Taisho format
            from core.taisho import ImplementationTask, ImplementationMode

            complexity = getattr(task, 'complexity', None)
            if complexity:
                complexity_value = complexity.value if hasattr(complexity, 'value') else str(complexity)
                if complexity_value == "complex":
                    mode = ImplementationMode.HEAVY
                elif complexity_value == "medium":
                    mode = ImplementationMode.STANDARD
                else:
                    mode = ImplementationMode.LIGHTWEIGHT
            else:
                mode = ImplementationMode.STANDARD

            impl_task = ImplementationTask(
                content=task.content,
                mode=mode,
                context=getattr(task, 'context', None)
            )

            result = await self.taisho.execute_implementation(impl_task)

            result["handled_by"] = "taisho"
            return result

        except Exception as e:
            logger.warning(f"⚠️ Taisho execution failed: {e}")
            self.execution_stats["fallback_count"] += 1

            # Fallback to Gemini as final defense
            logger.info("🔄 Activating Gemini final defense")
            return await self._execute_with_gemini(task, as_final_defense=True)

    async def _execute_with_gemini(self, task, as_final_defense: bool = False) -> Dict[str, Any]:
        """Execute task with Gemini (3.0 Flash preferred)"""

        # Prefer Gemini 3.0 Flash
        client = self.gemini3_client or self.gemini_client

        if not client:
            logger.error("❌ No Gemini client available")
            return {"error": "No Gemini client", "status": "failed"}

        defense_label = "Final Defense" if as_final_defense else "Gemini"
        logger.info(f"🛡️ Executing with Gemini ({defense_label})")

        try:
            # Use different method signature based on client type
            if self.gemini3_client:
                response = await client.generate(
                    prompt=task.content,
                    max_output_tokens=2000,
                    as_final_defense=as_final_defense
                )
            else:
                response = await client.generate(
                    prompt=task.content,
                    max_output_tokens=2000
                )

            return {
                "status": "completed",
                "result": response,
                "handled_by": "gemini3" if self.gemini3_client else "gemini",
                "final_defense": as_final_defense
            }

        except Exception as e:
            logger.error(f"❌ Gemini execution failed: {e}")
            return {"error": str(e), "status": "failed"}

    async def _execute_with_ashigaru(self, task) -> Dict[str, Any]:
        """Execute light task with Ashigaru parallel support"""

        if not self.ashigaru:
            logger.warning("⚠️ Ashigaru not available")
            return await self._execute_with_gemini(task)

        logger.info("👣 Executing with Ashigaru (parallel support)")

        try:
            result = await self.ashigaru.execute_direct(task)
            result["handled_by"] = "ashigaru"
            return result

        except Exception as e:
            logger.warning(f"⚠️ Ashigaru failed: {e}")
            return await self._execute_with_gemini(task)

    async def _execute_hybrid(self, task, routing_decision) -> Dict[str, Any]:
        """Execute task with hybrid Taisho + Ashigaru coordination"""

        logger.info("🔄 Executing with hybrid coordination")

        # Consult memory for context
        context = await self._consult_memory(task)

        # Decompose task
        subtasks = await self._decompose_task(task, context)

        if subtasks:
            # Execute subtasks in parallel
            results = await self._execute_subtasks(subtasks)
            # Integrate results
            return await self._integrate_results(task, results)
        else:
            # Direct execution
            return await self._execute_with_taisho(task, routing_decision)

    async def _consult_memory(self, task) -> Optional[Dict[str, Any]]:
        """Consult Memory MCP for relevant context"""

        if not self.memory_mcp:
            return None

        try:
            query = task.content[:100]
            context = await self.memory_mcp.search(query)

            if context:
                logger.info(f"📚 Found {len(context)} relevant memory entries")
                return context

        except Exception as e:
            logger.warning(f"⚠️ Memory consultation failed: {e}")

        return None

    async def _decompose_task(self, task, context: Optional[Dict[str, Any]]) -> List[Subtask]:
        """Decompose task using Gemini"""

        client = self.gemini3_client or self.gemini_client
        if not client:
            return []

        decomposition_prompt = f"""
As the Karo (tactical coordinator) in Bushidan v{self.VERSION}, analyze this task:

Task: {task.content}
Complexity: {getattr(task, 'complexity', 'unknown')}

Available Ashigaru (execution units):
- filesystem: File operations
- git: Version control
- memory: Knowledge storage
- web_search: Information gathering

Past Context: {context or "None"}

If the task can be done directly, respond with "DIRECT".
Otherwise, decompose into subtasks as JSON:
{{
    "subtasks": [
        {{"id": "1", "content": "description", "ashigaru_type": "filesystem", "dependencies": [], "priority": 1}}
    ]
}}
"""

        try:
            if self.gemini3_client:
                response = await client.generate(prompt=decomposition_prompt, max_output_tokens=1000)
            else:
                response = await client.generate(prompt=decomposition_prompt, max_output_tokens=1000)

            if "DIRECT" in response.upper():
                logger.info("📋 Task will be executed directly")
                return []

            decomposition = json.loads(response)
            subtasks = []

            for st in decomposition.get("subtasks", []):
                subtask = Subtask(
                    id=st["id"],
                    content=st["content"],
                    ashigaru_type=st.get("ashigaru_type", "filesystem"),
                    dependencies=st.get("dependencies", []),
                    priority=st.get("priority", 1)
                )
                subtasks.append(subtask)

            logger.info(f"📋 Task decomposed into {len(subtasks)} subtasks")
            return subtasks

        except Exception as e:
            logger.warning(f"⚠️ Task decomposition failed: {e}")
            return []

    async def _execute_subtasks(self, subtasks: List[Subtask]) -> List[Dict[str, Any]]:
        """Execute subtasks in parallel"""

        if not self.ashigaru:
            return []

        results = {}
        remaining = subtasks.copy()

        while remaining:
            ready = [t for t in remaining if all(d in results for d in t.dependencies)]

            if not ready:
                logger.error("❌ Circular dependency detected")
                break

            parallel_results = await asyncio.gather(
                *[self.ashigaru.execute_subtask(t) for t in ready],
                return_exceptions=True
            )

            for task, result in zip(ready, parallel_results):
                if isinstance(result, Exception):
                    results[task.id] = {"error": str(result), "status": "failed"}
                else:
                    results[task.id] = result
                remaining.remove(task)

            logger.info(f"✅ Completed {len(ready)} subtasks")

        return list(results.values())

    async def _integrate_results(self, task, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate subtask results"""

        if len(results) == 1:
            return results[0]

        client = self.gemini3_client or self.gemini_client
        if not client:
            return {
                "status": "completed",
                "result": results,
                "handled_by": "karo"
            }

        integration_prompt = f"""
As the Karo, integrate these subtask results:

Original Task: {task.content}

Results:
{chr(10).join([f"Result {i+1}: {str(r)}" for i, r in enumerate(results)])}

Provide a synthesized, coherent final result.
"""

        try:
            if self.gemini3_client:
                response = await client.generate(prompt=integration_prompt, max_output_tokens=1500)
            else:
                response = await client.generate(prompt=integration_prompt, max_output_tokens=1500)

            return {
                "status": "completed",
                "result": response,
                "subtask_count": len(results),
                "handled_by": "karo"
            }

        except Exception as e:
            logger.error(f"❌ Integration failed: {e}")
            return {
                "status": "integration_failed",
                "raw_results": results,
                "error": str(e)
            }

    def _update_stats(self, delegation: TaskDelegation, elapsed_time: float) -> None:
        """Update execution statistics"""
        self.execution_stats["total_tasks"] += 1
        self.execution_stats["by_delegation"][delegation.value] += 1
        self.execution_stats["total_time_seconds"] += elapsed_time

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive Karo statistics"""

        stats = {
            "version": self.VERSION,
            "execution_stats": self.execution_stats,
            "components": {
                "gemini3": self.gemini3_client is not None,
                "gemini": self.gemini_client is not None,
                "groq": self.groq_client is not None,
                "taisho": self.taisho is not None,
                "ashigaru": self.ashigaru is not None
            }
        }

        # Add Gemini3 stats if available
        if self.gemini3_client and hasattr(self.gemini3_client, 'get_statistics'):
            stats["gemini3_statistics"] = self.gemini3_client.get_statistics()

        # Add Groq stats if available
        if self.groq_client and hasattr(self.groq_client, 'get_statistics'):
            stats["groq_statistics"] = self.groq_client.get_statistics()

        # Add BDI stats
        stats["bdi_stats"] = self.bdi_stats
        stats["bdi_state"] = self.get_bdi_state()

        return stats

    # ==================== BDI Framework Integration ====================

    def _initialize_bdi(self) -> None:
        """Initialize BDI Framework for tactical coordination"""
        logger.info("🧠 Initializing BDI Framework for Karo...")

        # Initialize operational beliefs about available resources
        self.belief_base.add_belief(Belief(
            id="has_taisho",
            type=BeliefType.OPERATIONAL,
            content={"capability": "implementation", "available": self.taisho is not None},
            confidence=1.0,
            source="system_init"
        ))

        self.belief_base.add_belief(Belief(
            id="has_ashigaru",
            type=BeliefType.OPERATIONAL,
            content={"capability": "parallel_execution", "available": self.ashigaru is not None},
            confidence=1.0,
            source="system_init"
        ))

        self.belief_base.add_belief(Belief(
            id="has_groq",
            type=BeliefType.OPERATIONAL,
            content={"capability": "instant_response", "available": self.groq_client is not None},
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

        ashigaru_types = ["filesystem", "git", "memory", "web_search"]
        self.belief_base.add_belief(Belief(
            id="available_ashigaru_types",
            type=BeliefType.OPERATIONAL,
            content={"types": ashigaru_types, "count": len(ashigaru_types)},
            confidence=1.0,
            source="system_init"
        ))

        # Initialize tactical desires
        self.desire_set.add_desire(Desire(
            id="efficient_decomposition",
            type=DesireType.OPTIMIZATION,
            description="Decompose tasks efficiently for parallel execution",
            priority=0.9,
            feasibility=1.0,
            conditions=["has_ashigaru"]
        ))

        self.desire_set.add_desire(Desire(
            id="maximize_parallelization",
            type=DesireType.OPTIMIZATION,
            description="Maximize parallel execution of subtasks",
            priority=0.8,
            feasibility=0.9,
            conditions=["has_ashigaru"]
        ))

        self.desire_set.add_desire(Desire(
            id="maintain_coordination_quality",
            type=DesireType.MAINTENANCE,
            description="Maintain high quality in result integration",
            priority=0.85,
            feasibility=1.0
        ))

        self.desire_set.add_desire(Desire(
            id="minimize_fallbacks",
            type=DesireType.OPTIMIZATION,
            description="Minimize fallback chain activations",
            priority=0.7,
            feasibility=0.8
        ))

        logger.info(f"🧠 Karo BDI initialized: {len(self.belief_base.beliefs)} beliefs, {len(self.desire_set.desires)} desires")

    async def execute_task_with_bdi(self, task, routing_decision=None) -> Dict[str, Any]:
        """Execute task using BDI reasoning cycle"""
        if not self.bdi_enabled:
            return await self.execute_task_with_routing(task, routing_decision)

        logger.info(f"🧠 Karo BDI cycle starting...")
        self.bdi_stats["bdi_cycles"] += 1

        try:
            # Perceive: Update beliefs about the task
            await self._bdi_perceive(task)

            # Deliberate: Select tactical desire
            selected_desire = await self._bdi_deliberate(task)
            if not selected_desire:
                return await self.execute_task_with_routing(task, routing_decision)

            # Plan: Create coordination intention
            intention = await self._bdi_plan(task, selected_desire)
            if not intention:
                return await self.execute_task_with_routing(task, routing_decision)

            self.intention_stack.adopt_intention(intention)
            self.intention_stack.update_status(intention.id, "executing")

            # Execute: Carry out the coordination plan
            result = await self._bdi_execute(task, intention, routing_decision)

            # Reconsider: Update beliefs based on results
            await self._bdi_reconsider(intention, result)

            return result

        except Exception as e:
            logger.error(f"❌ Karo BDI cycle failed: {e}")
            return await self.execute_task_with_routing(task, routing_decision)

    async def _bdi_perceive(self, task) -> None:
        """Perceive task characteristics and environment"""

        # Estimate subtask count
        subtask_estimate = self._estimate_subtask_count(task)

        # Add task belief
        self.belief_base.add_belief(Belief(
            id=f"task_info_{id(task)}",
            type=BeliefType.FACTUAL,
            content={
                "content": task.content[:200],
                "complexity": getattr(task, 'complexity', 'unknown'),
                "estimated_subtasks": subtask_estimate,
                "parallelizable": subtask_estimate > 1
            },
            confidence=0.85,
            source="task_analysis",
            timestamp=datetime.now()
        ))

        # Check memory for context
        if self.memory_mcp:
            try:
                context = await self.memory_mcp.search(task.content[:100])
                if context:
                    self.belief_base.add_belief(Belief(
                        id=f"task_context_{id(task)}",
                        type=BeliefType.CONTEXTUAL,
                        content={"entries": context[:3]},
                        confidence=0.8,
                        source="memory_mcp"
                    ))
            except:
                pass

        logger.debug(f"👁️ Karo perceived: ~{subtask_estimate} subtasks")

    def _estimate_subtask_count(self, task) -> int:
        """Estimate number of subtasks for decomposition"""
        content = task.content.lower()

        action_keywords = ["implement", "create", "update", "test", "document",
                         "refactor", "add", "modify", "fix", "analyze"]
        action_count = sum(1 for kw in action_keywords if kw in content)

        return max(1, min(action_count, 5))

    async def _bdi_deliberate(self, task) -> Optional[Desire]:
        """Select tactical desire based on task characteristics"""

        feasible = self.desire_set.filter_feasible(self.belief_base)
        if not feasible:
            return None

        # Get task info
        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if task_beliefs:
            task_info = task_beliefs[-1].content
            parallelizable = task_info.get("parallelizable", False)
            subtask_count = task_info.get("estimated_subtasks", 1)

            # Adjust priorities based on task
            for desire in feasible:
                if desire.id == "maximize_parallelization":
                    desire.feasibility = 0.9 if parallelizable else 0.3

                if desire.id == "efficient_decomposition" and subtask_count >= 3:
                    desire.priority = 0.95

        selected = sorted(feasible, key=lambda d: d.priority * d.feasibility, reverse=True)[0]
        logger.debug(f"🎯 Karo selected desire: {selected.id}")
        return selected

    async def _bdi_plan(self, task, desire: Desire) -> Optional[Intention]:
        """Create tactical coordination plan"""

        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if not task_beliefs:
            return None

        task_info = task_beliefs[-1].content
        subtask_count = task_info.get("estimated_subtasks", 1)

        plan = []

        if desire.id == "efficient_decomposition" and subtask_count > 1:
            plan = [
                {"action": "consult_memory", "agent": "memory"},
                {"action": "decompose_task", "agent": "karo"},
                {"action": "execute_subtasks", "agent": "ashigaru"},
                {"action": "integrate_results", "agent": "karo"}
            ]
            self.bdi_stats["decompositions"] += 1

        elif desire.id == "maximize_parallelization":
            plan = [
                {"action": "decompose_task", "agent": "karo"},
                {"action": "parallel_execution", "agent": "ashigaru"},
                {"action": "integrate_results", "agent": "karo"}
            ]
            self.bdi_stats["parallel_executions"] += 1

        elif desire.id == "maintain_coordination_quality":
            plan = [
                {"action": "delegate_to_taisho", "agent": "taisho"},
                {"action": "validate_result", "agent": "karo"}
            ]

        else:
            plan = [
                {"action": "direct_execution", "agent": "taisho"}
            ]

        intention = Intention(
            id=f"karo_intention_{datetime.now().timestamp()}",
            desire_id=desire.id,
            plan=plan,
            metadata={"subtask_count": subtask_count}
        )

        logger.debug(f"📋 Karo planned: {len(plan)} steps")
        return intention

    async def _bdi_execute(self, task, intention: Intention, routing_decision) -> Dict[str, Any]:
        """Execute the tactical coordination plan"""

        result = {"status": "executing", "steps_completed": [], "bdi_intention": intention.id}
        subtasks = []
        subtask_results = []

        try:
            for step in intention.plan:
                action = step["action"]

                if action == "consult_memory":
                    context = await self._consult_memory(task)
                    result["memory_context"] = context

                elif action == "decompose_task":
                    context = result.get("memory_context")
                    subtasks = await self._decompose_task(task, context)
                    result["subtask_count"] = len(subtasks)

                elif action in ["execute_subtasks", "parallel_execution"]:
                    if subtasks:
                        subtask_results = await self._execute_subtasks(subtasks)
                    else:
                        taisho_result = await self._execute_with_taisho(task, routing_decision)
                        subtask_results = [taisho_result]

                elif action == "integrate_results":
                    if subtask_results:
                        integrated = await self._integrate_results(task, subtask_results)
                        result.update(integrated)

                elif action == "delegate_to_taisho":
                    taisho_result = await self._execute_with_taisho(task, routing_decision)
                    result.update(taisho_result)

                elif action == "direct_execution":
                    direct_result = await self._execute_with_taisho(task, routing_decision)
                    result.update(direct_result)

                elif action == "validate_result":
                    result["validated"] = True

                result["steps_completed"].append({"action": action, "status": "completed"})

            result["status"] = "completed"
            logger.info(f"✅ Karo BDI execution complete")

        except Exception as e:
            logger.error(f"❌ Karo BDI execution failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    async def _bdi_reconsider(self, intention: Intention, result: Dict[str, Any]) -> None:
        """Update beliefs based on execution results"""

        if result.get("status") == "completed":
            self.intention_stack.update_status(intention.id, "completed")

            self.belief_base.add_belief(Belief(
                id=f"coordination_success_{intention.id}",
                type=BeliefType.HISTORICAL,
                content={
                    "desire_id": intention.desire_id,
                    "subtask_count": result.get("subtask_count", 0),
                    "success": True
                },
                confidence=1.0,
                source="execution_result",
                timestamp=datetime.now()
            ))
        else:
            self.intention_stack.update_status(intention.id, "failed")
            self.execution_stats["fallback_count"] += 1

    def get_bdi_state(self) -> Dict[str, Any]:
        """Get current BDI state"""
        return {
            "beliefs": self.belief_base.get_statistics(),
            "desires": self.desire_set.get_statistics(),
            "intentions": self.intention_stack.get_statistics()
        }
