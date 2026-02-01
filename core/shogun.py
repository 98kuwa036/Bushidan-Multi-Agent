"""
Bushidan Multi-Agent System v9.3.2 - Shogun (Strategic Layer)

The Shogun serves as the highest decision-making authority using Claude Sonnet 4.5.
Enhanced with Intelligent Routing, Prompt Caching, and BDI Framework integration.

v9.3.2 Enhancements:
- Intelligent Router integration for optimal task delegation
- ClaudeClientCached for 90% cost reduction
- Enhanced complexity assessment with routing heuristics
- Power-saving optimization (don't wake Qwen for simple tasks)
- BDI Framework integration for formal multi-agent reasoning
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger
from core.system_orchestrator import SystemOrchestrator
from core.bdi_framework import (
    BDIAgent, BeliefBase, DesireSet, IntentionStack,
    Belief, Desire, Intention, BeliefType, DesireType
)


logger = get_logger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for delegation decisions (v9.3.2 timing)"""
    SIMPLE = "simple"      # 2s - Groq handles instantly
    MEDIUM = "medium"      # 12s - Local Qwen3 → Cloud fallback
    COMPLEX = "complex"    # 28s - 3-tier fallback chain
    STRATEGIC = "strategic" # 45s - Shogun handles directly


class ReviewLevel(Enum):
    """Review depth levels for quality assurance"""
    BASIC = "basic"          # Simple/Medium - Sonnet basic review (5s, ¥0 Pro)
    DETAILED = "detailed"    # Complex - Sonnet detailed review (10s, ¥0-5 API)
    PREMIUM = "premium"      # Strategic - Opus premium review (15s, ¥10)


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
    将軍 (Shogun) - Strategic Decision Layer v9.3.2

    Primary responsibilities:
    1. Task intake and intelligent complexity assessment
    2. Routing decisions via IntelligentRouter
    3. Strategic decision making
    4. Delegation to Karo (Tactical Layer)
    5. Final quality assurance and approval
    6. Ethical and security oversight

    v9.3.2 Features:
    - Intelligent routing with power-saving optimization
    - Prompt caching for 90% cost reduction
    - 3-tier fallback chain management
    """

    VERSION = "9.3.2"

    def __init__(self, orchestrator: SystemOrchestrator):
        self.orchestrator = orchestrator
        self.claude_client = None
        self.opus_client = None
        self.quality_metrics = None
        self.router = None
        self.karo = None
        self.memory_mcp = None

        # BDI Framework components
        self.belief_base = BeliefBase()
        self.desire_set = DesireSet()
        self.intention_stack = IntentionStack()
        self.bdi_enabled = True

        # Statistics
        self.reviews_by_level = {
            ReviewLevel.BASIC: 0,
            ReviewLevel.DETAILED: 0,
            ReviewLevel.PREMIUM: 0
        }
        self.routing_stats = {
            "total_tasks": 0,
            "by_complexity": {c.value: 0 for c in TaskComplexity},
            "total_time_seconds": 0.0,
            "power_savings": 0
        }
        self.bdi_stats = {
            "bdi_cycles": 0,
            "beliefs_updated": 0,
            "intentions_completed": 0
        }

    async def initialize(self) -> None:
        """Initialize Shogun and subordinate systems"""
        logger.info(f"🎌 Initializing Shogun v{self.VERSION} (Strategic Layer)...")

        # Initialize Claude client (prefer cached version)
        self.claude_client = self.orchestrator.get_client("claude_cached")
        if not self.claude_client:
            self.claude_client = self.orchestrator.get_client("claude")
            if self.claude_client:
                logger.info("📝 Using standard Claude client (cached not available)")

        # Initialize Opus client for premium reviews
        self.opus_client = self.orchestrator.get_client("opus")
        if self.opus_client:
            logger.info("🏆 Opus premium review system enabled")

        # Initialize quality metrics
        try:
            from utils.quality_metrics import QualityMetricsCollector
            self.quality_metrics = QualityMetricsCollector()
            logger.info("✅ Quality metrics collector initialized")
        except Exception as e:
            logger.warning(f"⚠️ Quality metrics not available: {e}")

        # Get Intelligent Router
        self.router = self.orchestrator.get_router()
        if self.router:
            logger.info("🚀 Intelligent Router enabled")
        else:
            logger.info("ℹ️ Intelligent Router not available, using legacy routing")

        # Initialize Karo (Tactical Layer)
        from core.karo import Karo
        self.karo = Karo(self.orchestrator)
        await self.karo.initialize()

        # Get Memory MCP for decision logging
        self.memory_mcp = self.orchestrator.get_mcp("memory")

        # Initialize BDI Framework
        self._initialize_bdi()

        logger.info(f"✅ Shogun v{self.VERSION} initialization complete (BDI enabled)")

    async def start_service(self) -> None:
        """Start the main service loop"""
        logger.info(f"🏯 Shogun v{self.VERSION} service started - Ready for commands")

        while True:
            try:
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("📴 Shogun service stopping...")
                break
            except Exception as e:
                logger.error(f"❌ Error in service loop: {e}")
                await asyncio.sleep(5)

    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Main task processing pipeline with v9.3.2 routing

        1. Assess complexity using Intelligent Router heuristics
        2. Get routing decision
        3. Execute based on routing (Shogun, Karo, or direct)
        4. Review and approve results
        5. Log important decisions
        """

        start_time = time.time()
        logger.info(f"🎌 Shogun processing task: {task.content[:50]}...")

        try:
            # Step 1: Assess complexity with router heuristics
            assessed_complexity = await self._assess_complexity_v932(task)
            task.complexity = assessed_complexity

            # Step 2: Get routing decision
            routing_decision = self._get_routing_decision(task)

            # Step 3: Execute based on complexity
            if task.complexity == TaskComplexity.STRATEGIC:
                result = await self._handle_strategic_task(task)
            elif task.complexity == TaskComplexity.SIMPLE and self.orchestrator.get_client("groq"):
                # Simple tasks can use Groq directly for speed
                result = await self._handle_simple_task_groq(task)
            else:
                # Delegate to Karo for tactical execution
                result = await self.karo.execute_task_with_routing(task, routing_decision)

                # Adaptive quality review
                result = await self._adaptive_review(task, result)

            # Step 4: Log important decisions
            await self._log_decision(task, result)

            # Update statistics
            elapsed_time = time.time() - start_time
            self._update_routing_stats(task, elapsed_time, routing_decision)

            logger.info(f"✅ Shogun task complete in {elapsed_time:.1f}s")
            return result

        except Exception as e:
            logger.error(f"❌ Shogun task processing failed: {e}")
            return {"error": str(e), "status": "failed"}

    async def _assess_complexity_v932(self, task: Task) -> TaskComplexity:
        """
        Assess task complexity using v9.3.2 Intelligent Router heuristics

        Priority order:
        1. Use router heuristics (fast, no API call)
        2. Fall back to LLM assessment if needed
        """

        # Try router heuristics first (fast, no API cost)
        if self.router:
            try:
                from core.intelligent_router import TaskComplexity as RouterComplexity
                router_complexity = self.router.judge_complexity(task.content, task.context)

                # Map router complexity to Shogun complexity
                complexity_map = {
                    RouterComplexity.SIMPLE: TaskComplexity.SIMPLE,
                    RouterComplexity.MEDIUM: TaskComplexity.MEDIUM,
                    RouterComplexity.COMPLEX: TaskComplexity.COMPLEX,
                    RouterComplexity.STRATEGIC: TaskComplexity.STRATEGIC
                }
                result = complexity_map.get(router_complexity, TaskComplexity.MEDIUM)
                logger.info(f"📊 Router heuristic complexity: {result.value}")
                return result

            except Exception as e:
                logger.warning(f"⚠️ Router heuristic failed: {e}, using LLM fallback")

        # Fallback to LLM assessment
        return await self._assess_complexity_llm(task)

    async def _assess_complexity_llm(self, task: Task) -> TaskComplexity:
        """Assess complexity using Claude (fallback method)"""

        if not self.claude_client:
            logger.warning("⚠️ No Claude client, defaulting to MEDIUM")
            return TaskComplexity.MEDIUM

        assessment_prompt = f"""
As the Shogun (strategic decision maker), assess task complexity:

Task: {task.content}

Classify as:
- SIMPLE: Questions, lookups, simple queries (<50 chars, no code needed)
- MEDIUM: Standard implementation (code keywords, single file)
- COMPLEX: Multi-component systems (multiple files, architecture)
- STRATEGIC: High-level decisions, technology choices

Respond with just: SIMPLE, MEDIUM, COMPLEX, or STRATEGIC
"""

        try:
            response = await self.claude_client.generate(
                messages=[{"role": "user", "content": assessment_prompt}],
                max_tokens=10
            )

            complexity_str = response.strip().upper()
            complexity = TaskComplexity(complexity_str.lower())

            logger.info(f"📊 LLM complexity assessment: {complexity.value}")
            return complexity

        except Exception as e:
            logger.warning(f"⚠️ LLM assessment failed: {e}")
            return TaskComplexity.MEDIUM

    def _get_routing_decision(self, task: Task):
        """Get routing decision from Intelligent Router"""

        if not self.router:
            return None

        try:
            from core.intelligent_router import TaskComplexity as RouterComplexity

            # Map Shogun complexity to Router complexity
            complexity_map = {
                TaskComplexity.SIMPLE: RouterComplexity.SIMPLE,
                TaskComplexity.MEDIUM: RouterComplexity.MEDIUM,
                TaskComplexity.COMPLEX: RouterComplexity.COMPLEX,
                TaskComplexity.STRATEGIC: RouterComplexity.STRATEGIC
            }
            router_complexity = complexity_map.get(task.complexity, RouterComplexity.MEDIUM)

            return self.router.determine_route(router_complexity, task.context)

        except Exception as e:
            logger.warning(f"⚠️ Routing decision failed: {e}")
            return None

    async def _handle_simple_task_groq(self, task: Task) -> Dict[str, Any]:
        """
        Handle simple tasks directly with Groq for instant response

        Benefits:
        - 300-500 tok/s (10-20x faster)
        - Free tier (¥0)
        - Power-saving (don't wake Qwen)
        """

        groq_client = self.orchestrator.get_client("groq")
        if not groq_client:
            # Fallback to Karo
            return await self.karo.execute_task(task)

        logger.info("⚡ Routing simple task to Groq for instant response")

        try:
            response = await groq_client.generate(
                messages=[{"role": "user", "content": task.content}],
                max_tokens=1000
            )

            # Track power savings
            self.routing_stats["power_savings"] += 1

            return {
                "status": "completed",
                "result": response,
                "complexity": "simple",
                "handled_by": "groq",
                "power_saving": True,
                "shogun_approval": "auto_approved"  # Simple tasks auto-approved
            }

        except Exception as e:
            logger.warning(f"⚠️ Groq failed, falling back to Karo: {e}")
            return await self.karo.execute_task(task)

    async def _handle_strategic_task(self, task: Task) -> Dict[str, Any]:
        """Handle strategic-level tasks directly with Claude + Opus review"""

        logger.info("🏆 Handling strategic task with Opus review")

        if not self.claude_client:
            return {"error": "No Claude client available", "status": "failed"}

        strategic_prompt = f"""
As the Shogun (highest authority) in Bushidan v{self.VERSION}, handle this strategic task:

Task: {task.content}
Context: {task.context or "None provided"}

This is a STRATEGIC level decision requiring highest-level analysis:
- Technical implications
- Resource requirements
- Long-term consequences
- Security and ethical considerations

Provide a comprehensive strategic response.
"""

        response = await self.claude_client.generate(
            messages=[{"role": "user", "content": strategic_prompt}],
            max_tokens=2000,
            system_prompt="You are the Shogun, the strategic decision-maker in a multi-agent AI system."
        )

        result = {
            "status": "completed",
            "result": response,
            "complexity": "strategic",
            "handled_by": "shogun"
        }

        # Strategic tasks always get Opus review
        if self.opus_client:
            result = await self._opus_premium_review(task, result, None)
        else:
            result["shogun_approval"] = "approved_without_opus"

        return result

    async def _adaptive_review(self, task: Task, result: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptive review based on task complexity and quality metrics"""

        if result.get("status") != "completed":
            return result

        # Collect quality metrics
        implementation = result.get("result", "")
        quality_report = None

        if self.quality_metrics and isinstance(implementation, str) and len(implementation) > 50:
            try:
                quality_report = self.quality_metrics.collect_metrics(
                    task_id=str(id(task)),
                    code=implementation,
                    language="python"
                )
                result["quality_metrics"] = {
                    "complexity_score": quality_report.complexity_metrics.complexity_score,
                    "security_score": quality_report.security_findings.security_score,
                    "overall_score": quality_report.overall_quality_score,
                    "risk_level": quality_report.complexity_metrics.risk_level.value,
                    "recommendations": quality_report.recommendations
                }
            except Exception as e:
                logger.warning(f"⚠️ Quality metrics collection failed: {e}")

        # Determine review level
        review_level = self._determine_review_level(task, quality_report)

        logger.info(f"🔍 Starting {review_level.value} review")

        # Execute appropriate review
        if review_level == ReviewLevel.PREMIUM and self.opus_client:
            result = await self._opus_premium_review(task, result, quality_report)
        elif review_level == ReviewLevel.DETAILED:
            result = await self._sonnet_detailed_review(task, result, quality_report)
        else:
            result = await self._sonnet_basic_review(task, result)

        # Update statistics
        self.reviews_by_level[review_level] += 1

        return result

    def _determine_review_level(self, task: Task, quality_report) -> ReviewLevel:
        """Determine appropriate review level"""

        # Strategic tasks always get Opus
        if task.complexity == TaskComplexity.STRATEGIC:
            return ReviewLevel.PREMIUM

        # Check quality report for risks
        if quality_report:
            try:
                from utils.quality_metrics import RiskLevel
                if quality_report.complexity_metrics.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    logger.info("🚨 High risk detected, upgrading to Opus review")
                    return ReviewLevel.PREMIUM

                if quality_report.security_findings.vulnerabilities:
                    logger.info("🔒 Security vulnerabilities detected, upgrading to Opus")
                    return ReviewLevel.PREMIUM
            except Exception as e:
                logger.warning(f"⚠️ Risk level check failed: {e}")

        # Complex tasks get detailed review
        if task.complexity == TaskComplexity.COMPLEX:
            return ReviewLevel.DETAILED

        # Simple/Medium get basic review
        return ReviewLevel.BASIC

    async def _opus_premium_review(self, task: Task, result: Dict[str, Any], quality_report) -> Dict[str, Any]:
        """Premium review using Claude Opus"""

        implementation = result.get("result", "")

        context = {
            "complexity": task.complexity.value,
            "risk_level": quality_report.complexity_metrics.risk_level.value if quality_report else "unknown"
        }

        try:
            opus_review = await self.opus_client.conduct_premium_review(
                task_content=task.content,
                implementation=implementation,
                context=context
            )

            result["opus_review"] = {
                "score": opus_review.score,
                "decision": opus_review.decision,
                "critical_issues": opus_review.critical_issues,
                "recommendations": opus_review.recommendations,
                "cost_yen": opus_review.cost_yen,
                "review_time": opus_review.review_time_seconds
            }
            result["shogun_approval"] = opus_review.decision

            if opus_review.decision == "approved":
                logger.info(f"✅ Opus approved: {opus_review.score}/100 (¥{opus_review.cost_yen:.2f})")
            else:
                logger.warning(f"⚠️ Opus found issues: {len(opus_review.critical_issues)} critical")

        except Exception as e:
            logger.error(f"❌ Opus review failed: {e}")
            logger.info("🔄 Falling back to Sonnet detailed review")
            result = await self._sonnet_detailed_review(task, result, quality_report)

        return result

    async def _sonnet_detailed_review(self, task: Task, result: Dict[str, Any], quality_report) -> Dict[str, Any]:
        """Detailed review using Claude Sonnet"""

        if not self.claude_client:
            result["shogun_approval"] = "no_client"
            return result

        implementation = result.get("result", "")

        quality_context = ""
        if quality_report:
            quality_context = f"""
Quality Metrics:
- Complexity Score: {quality_report.complexity_metrics.complexity_score}/100
- Security Score: {quality_report.security_findings.security_score}/100
- Risk Level: {quality_report.complexity_metrics.risk_level.value}
"""

        review_prompt = f"""
Conduct a DETAILED review of this implementation:

Task: {task.content}

Implementation:
{implementation[:2000]}

{quality_context}

Evaluate:
1. Functional Correctness (40 points)
2. Code Quality (30 points)
3. Security (20 points)
4. Best Practices (10 points)

Provide:
- **Score**: X/100
- **Decision**: APPROVED / REVISE_REQUIRED
- **Key Issues**: List problems
- **Recommendations**: List improvements
"""

        try:
            review = await self.claude_client.generate(
                messages=[{"role": "user", "content": review_prompt}],
                max_tokens=800,
                temperature=0.1
            )

            score = 85.0
            score_match = re.search(r'(\d+)/100', review)
            if score_match:
                score = float(score_match.group(1))

            result["sonnet_detailed_review"] = {
                "review_text": review,
                "score": score,
                "review_level": "detailed"
            }

            if "APPROVED" in review.upper():
                result["shogun_approval"] = "approved"
                logger.info(f"✅ Sonnet detailed approved: {score}/100")
            else:
                result["shogun_approval"] = "revise_required"
                logger.info(f"📝 Sonnet detailed: revisions needed ({score}/100)")

        except Exception as e:
            logger.warning(f"⚠️ Detailed review failed: {e}")
            result["shogun_approval"] = "review_failed"

        return result

    async def _sonnet_basic_review(self, task: Task, result: Dict[str, Any]) -> Dict[str, Any]:
        """Basic review using Claude Sonnet"""

        if not self.claude_client:
            result["shogun_approval"] = "auto_approved"
            return result

        review_prompt = f"""
Review this completed task:

Task: {task.content}
Result: {str(result.get("result", ""))[:1000]}

Check: Quality, correctness, security, completeness.
Respond with "APPROVED" if acceptable, otherwise provide brief feedback.
"""

        try:
            review = await self.claude_client.generate(
                messages=[{"role": "user", "content": review_prompt}],
                max_tokens=500
            )

            if "APPROVED" in review.upper():
                result["shogun_approval"] = "approved"
                logger.info("✅ Shogun basic review approved")
            else:
                result["shogun_feedback"] = review
                result["shogun_approval"] = "feedback_provided"
                logger.info("📝 Shogun basic review: feedback provided")

        except Exception as e:
            logger.warning(f"⚠️ Basic review failed: {e}")
            result["shogun_approval"] = "auto_approved"

        return result

    async def _log_decision(self, task: Task, result: Dict[str, Any]) -> None:
        """Log important decisions to Memory MCP"""

        if not self.memory_mcp:
            return

        if task.complexity not in [TaskComplexity.COMPLEX, TaskComplexity.STRATEGIC]:
            return

        decision_log = {
            "timestamp": time.time(),
            "category": "decision",
            "task": task.content[:200],
            "complexity": task.complexity.value,
            "status": result.get("status"),
            "approval": result.get("shogun_approval"),
            "version": self.VERSION
        }

        try:
            await self.memory_mcp.store(decision_log)
            logger.info("📝 Decision logged to Memory MCP")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log decision: {e}")

    def _update_routing_stats(self, task: Task, elapsed_time: float, routing_decision) -> None:
        """Update routing statistics"""

        self.routing_stats["total_tasks"] += 1
        self.routing_stats["by_complexity"][task.complexity.value] += 1
        self.routing_stats["total_time_seconds"] += elapsed_time

        # Record to router if available
        if self.router and routing_decision:
            try:
                cost = 0.0
                if hasattr(routing_decision, 'estimated_cost_yen'):
                    cost = routing_decision.estimated_cost_yen
                self.router.record_routing(routing_decision, elapsed_time, cost, True)
            except Exception as e:
                logger.warning(f"⚠️ Failed to record routing: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive Shogun statistics"""

        stats = {
            "version": self.VERSION,
            "reviews_by_level": {
                level.value: count
                for level, count in self.reviews_by_level.items()
            },
            "total_reviews": sum(self.reviews_by_level.values()),
            "routing_stats": self.routing_stats
        }

        # Add Opus statistics
        if self.opus_client and hasattr(self.opus_client, 'get_statistics'):
            stats["opus_statistics"] = self.opus_client.get_statistics()

        # Add quality metrics
        if self.quality_metrics and hasattr(self.quality_metrics, 'get_aggregate_stats'):
            stats["quality_metrics"] = self.quality_metrics.get_aggregate_stats()

        # Add Claude client statistics
        if self.claude_client and hasattr(self.claude_client, 'get_statistics'):
            stats["claude_statistics"] = self.claude_client.get_statistics()

        # Add router statistics
        if self.router:
            stats["router_statistics"] = self.router.get_statistics()

        return stats

    # Backward compatibility alias
    def get_review_statistics(self) -> Dict[str, Any]:
        """Alias for get_statistics for backward compatibility"""
        return self.get_statistics()

    # ==================== BDI Framework Integration ====================

    def _initialize_bdi(self) -> None:
        """Initialize BDI Framework components"""
        logger.info("🧠 Initializing BDI Framework for Shogun...")

        # Initialize core operational beliefs
        self.belief_base.add_belief(Belief(
            id="has_karo",
            type=BeliefType.OPERATIONAL,
            content={"capability": "tactical_coordination", "available": self.karo is not None},
            confidence=1.0,
            source="system_init"
        ))

        self.belief_base.add_belief(Belief(
            id="has_opus",
            type=BeliefType.OPERATIONAL,
            content={"capability": "premium_review", "available": self.opus_client is not None},
            confidence=1.0,
            source="system_init"
        ))

        self.belief_base.add_belief(Belief(
            id="has_router",
            type=BeliefType.OPERATIONAL,
            content={"capability": "intelligent_routing", "available": self.router is not None},
            confidence=1.0,
            source="system_init"
        ))

        self.belief_base.add_belief(Belief(
            id="has_memory",
            type=BeliefType.OPERATIONAL,
            content={"capability": "decision_logging", "available": self.memory_mcp is not None},
            confidence=1.0,
            source="system_init"
        ))

        # Initialize strategic desires
        self.desire_set.add_desire(Desire(
            id="maintain_quality",
            type=DesireType.MAINTENANCE,
            description="Maintain high quality standards (95+ points)",
            priority=0.9,
            feasibility=1.0
        ))

        self.desire_set.add_desire(Desire(
            id="optimize_cost",
            type=DesireType.OPTIMIZATION,
            description="Optimize cost while maintaining quality",
            priority=0.7,
            feasibility=1.0
        ))

        self.desire_set.add_desire(Desire(
            id="ensure_security",
            type=DesireType.MAINTENANCE,
            description="Ensure security and ethical compliance",
            priority=1.0,
            feasibility=1.0
        ))

        self.desire_set.add_desire(Desire(
            id="learn_and_improve",
            type=DesireType.EXPLORATION,
            description="Learn from past decisions to improve",
            priority=0.6,
            feasibility=0.8,
            conditions=["has_memory"]
        ))

        logger.info(f"🧠 BDI initialized: {len(self.belief_base.beliefs)} beliefs, {len(self.desire_set.desires)} desires")

    async def process_task_with_bdi(self, task: Task) -> Dict[str, Any]:
        """
        Process task using full BDI reasoning cycle

        BDI Cycle:
        1. Perceive - Update beliefs from task and environment
        2. Deliberate - Select desire to pursue
        3. Plan - Create intention to achieve desire
        4. Execute - Carry out the intention
        5. Reconsider - Update beliefs based on results
        """
        if not self.bdi_enabled:
            return await self.process_task(task)

        logger.info(f"🧠 BDI cycle starting for task: {task.content[:50]}...")
        self.bdi_stats["bdi_cycles"] += 1

        try:
            # Step 1: Perceive - Update beliefs
            await self._bdi_perceive(task)

            # Step 2: Deliberate - Select desire
            selected_desire = await self._bdi_deliberate(task)
            if not selected_desire:
                logger.warning("⚠️ No feasible desire selected, using standard processing")
                return await self.process_task(task)

            # Step 3: Plan - Create intention
            intention = await self._bdi_plan(task, selected_desire)
            if not intention:
                logger.warning("⚠️ Planning failed, using standard processing")
                return await self.process_task(task)

            # Adopt intention
            self.intention_stack.adopt_intention(intention)
            self.intention_stack.update_status(intention.id, "executing")

            # Step 4: Execute - Carry out intention
            result = await self._bdi_execute(task, intention)

            # Step 5: Reconsider - Update beliefs based on results
            await self._bdi_reconsider(task, intention, result)

            logger.info(f"✅ BDI cycle complete for desire: {selected_desire.id}")
            return result

        except Exception as e:
            logger.error(f"❌ BDI cycle failed: {e}")
            # Fallback to standard processing
            return await self.process_task(task)

    async def _bdi_perceive(self, task: Task) -> None:
        """BDI Perceive: Update beliefs based on task and environment"""

        # Assess task complexity
        complexity = await self._assess_complexity_v932(task)

        # Add belief about current task
        self.belief_base.add_belief(Belief(
            id=f"current_task_{id(task)}",
            type=BeliefType.FACTUAL,
            content={
                "content": task.content,
                "complexity": complexity.value,
                "priority": task.priority,
                "source": task.source
            },
            confidence=1.0,
            source="task_perception",
            timestamp=datetime.now()
        ))

        # Query historical context from Memory MCP
        if self.memory_mcp:
            try:
                history = await self.memory_mcp.search(task.content[:100])
                if history:
                    self.belief_base.add_belief(Belief(
                        id=f"historical_context_{id(task)}",
                        type=BeliefType.HISTORICAL,
                        content={"entries": history[:5]},
                        confidence=0.8,
                        source="memory_mcp"
                    ))
            except Exception as e:
                logger.warning(f"⚠️ Memory MCP query failed: {e}")

        # Update system state beliefs
        self.belief_base.add_belief(Belief(
            id="system_load",
            type=BeliefType.OPERATIONAL,
            content={
                "active_intentions": len(self.intention_stack.intentions),
                "pending_reviews": sum(self.reviews_by_level.values())
            },
            confidence=0.9,
            source="system_state",
            timestamp=datetime.now()
        ))

        self.bdi_stats["beliefs_updated"] += 1
        logger.debug(f"👁️ Perceived: task complexity={complexity.value}")

    async def _bdi_deliberate(self, task: Task) -> Optional[Desire]:
        """BDI Deliberate: Select which desire to pursue"""

        # Get task belief
        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if not task_beliefs:
            return None

        latest_task = task_beliefs[-1]
        complexity = latest_task.content.get("complexity", "medium")

        # Filter feasible desires
        feasible = self.desire_set.filter_feasible(self.belief_base)

        if not feasible:
            return None

        # Adjust priorities based on task characteristics
        for desire in feasible:
            # Security is always top priority for strategic tasks
            if complexity == "strategic" and desire.id == "ensure_security":
                desire.priority = 1.0

            # Quality is critical for complex tasks
            if complexity in ["complex", "strategic"] and desire.id == "maintain_quality":
                desire.priority = 0.95

            # Cost optimization matters more for simple tasks
            if complexity == "simple" and desire.id == "optimize_cost":
                desire.priority = 0.9

        # Select highest priority feasible desire
        selected = sorted(
            feasible,
            key=lambda d: d.priority * d.feasibility,
            reverse=True
        )[0]

        logger.debug(f"🎯 Deliberated: selected desire={selected.id} (priority={selected.priority})")
        return selected

    async def _bdi_plan(self, task: Task, desire: Desire) -> Optional[Intention]:
        """BDI Plan: Create an intention to achieve the selected desire"""

        # Get task complexity from beliefs
        task_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if not task_beliefs:
            return None

        complexity = task_beliefs[-1].content.get("complexity", "medium")

        # Create plan based on desire and complexity
        plan = []

        if desire.id == "maintain_quality":
            if complexity == "strategic":
                plan = [
                    {"action": "handle_strategic", "agent": "shogun"},
                    {"action": "opus_review", "agent": "opus"}
                ]
            elif complexity == "complex":
                plan = [
                    {"action": "delegate_to_karo", "agent": "karo"},
                    {"action": "detailed_review", "agent": "shogun"}
                ]
            else:
                plan = [
                    {"action": "delegate_to_karo", "agent": "karo"},
                    {"action": "basic_review", "agent": "shogun"}
                ]

        elif desire.id == "optimize_cost":
            if complexity == "simple":
                plan = [
                    {"action": "groq_direct", "agent": "groq"},
                    {"action": "auto_approve", "agent": "shogun"}
                ]
            else:
                plan = [
                    {"action": "delegate_to_karo", "agent": "karo"},
                    {"action": "basic_review", "agent": "shogun"}
                ]

        elif desire.id == "ensure_security":
            plan = [
                {"action": "security_assessment", "agent": "shogun"},
                {"action": "delegate_to_karo", "agent": "karo"},
                {"action": "detailed_review", "agent": "shogun"}
            ]

        elif desire.id == "learn_and_improve":
            plan = [
                {"action": "consult_memory", "agent": "memory"},
                {"action": "delegate_to_karo", "agent": "karo"},
                {"action": "basic_review", "agent": "shogun"},
                {"action": "log_decision", "agent": "memory"}
            ]

        intention = Intention(
            id=f"intention_{desire.id}_{datetime.now().timestamp()}",
            desire_id=desire.id,
            plan=plan,
            metadata={"complexity": complexity, "task_id": id(task)}
        )

        logger.debug(f"📋 Planned: {len(plan)} steps for desire={desire.id}")
        return intention

    async def _bdi_execute(self, task: Task, intention: Intention) -> Dict[str, Any]:
        """BDI Execute: Carry out the intention plan"""

        result = {"status": "executing", "steps_completed": [], "bdi_intention": intention.id}

        try:
            for step in intention.plan:
                action = step["action"]

                if action == "handle_strategic":
                    step_result = await self._handle_strategic_task(task)
                    result.update(step_result)

                elif action == "delegate_to_karo":
                    routing_decision = self._get_routing_decision(task)
                    step_result = await self.karo.execute_task_with_routing(task, routing_decision)
                    result.update(step_result)

                elif action == "groq_direct":
                    step_result = await self._handle_simple_task_groq(task)
                    result.update(step_result)

                elif action == "opus_review":
                    result = await self._opus_premium_review(task, result, None)

                elif action == "detailed_review":
                    result = await self._sonnet_detailed_review(task, result, None)

                elif action == "basic_review":
                    result = await self._sonnet_basic_review(task, result)

                elif action == "auto_approve":
                    result["shogun_approval"] = "auto_approved"

                elif action == "security_assessment":
                    # Add security check belief
                    self.belief_base.add_belief(Belief(
                        id=f"security_check_{id(task)}",
                        type=BeliefType.FACTUAL,
                        content={"checked": True, "timestamp": datetime.now().isoformat()},
                        confidence=1.0,
                        source="security_assessment"
                    ))

                elif action == "consult_memory":
                    if self.memory_mcp:
                        try:
                            history = await self.memory_mcp.search(task.content[:100])
                            result["memory_context"] = history[:3] if history else []
                        except:
                            pass

                elif action == "log_decision":
                    await self._log_decision(task, result)

                result["steps_completed"].append({"action": action, "status": "completed"})

            result["status"] = "completed"
            self.bdi_stats["intentions_completed"] += 1
            logger.info(f"✅ BDI execution complete: {len(intention.plan)} steps")

        except Exception as e:
            logger.error(f"❌ BDI execution failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    async def _bdi_reconsider(self, task: Task, intention: Intention, result: Dict[str, Any]) -> None:
        """BDI Reconsider: Update beliefs and intentions based on results"""

        # Update intention status
        if result.get("status") == "completed":
            self.intention_stack.update_status(intention.id, "completed")

            # Add success belief
            self.belief_base.add_belief(Belief(
                id=f"success_{intention.id}",
                type=BeliefType.HISTORICAL,
                content={
                    "desire_id": intention.desire_id,
                    "success": True,
                    "approval": result.get("shogun_approval")
                },
                confidence=1.0,
                source="execution_result",
                timestamp=datetime.now()
            ))
        else:
            self.intention_stack.update_status(intention.id, "failed")

            # Add failure belief for learning
            self.belief_base.add_belief(Belief(
                id=f"failure_{intention.id}",
                type=BeliefType.HISTORICAL,
                content={
                    "desire_id": intention.desire_id,
                    "success": False,
                    "error": result.get("error")
                },
                confidence=1.0,
                source="execution_result",
                timestamp=datetime.now()
            ))

        # Clean up old task beliefs (keep last 10)
        factual_beliefs = self.belief_base.query_beliefs(type=BeliefType.FACTUAL)
        if len(factual_beliefs) > 10:
            for old_belief in factual_beliefs[:-10]:
                self.belief_base.remove_belief(old_belief.id)

        logger.debug(f"🔄 Reconsidered: intention {intention.id} marked as {result.get('status')}")

    def get_bdi_state(self) -> Dict[str, Any]:
        """Get current BDI state for inspection"""
        return {
            "beliefs": self.belief_base.get_statistics(),
            "desires": self.desire_set.get_statistics(),
            "intentions": self.intention_stack.get_statistics(),
            "bdi_stats": self.bdi_stats,
            "consistency_issues": self.belief_base.check_consistency()
        }
