"""
Bushidan Multi-Agent System v9.3.2 - Shogun (Strategic Layer)

The Shogun serves as the highest decision-making authority using Claude Sonnet 4.5.
Enhanced with Intelligent Routing and Prompt Caching.

v9.3.2 Enhancements:
- Intelligent Router integration for optimal task delegation
- ClaudeClientCached for 90% cost reduction
- Enhanced complexity assessment with routing heuristics
- Power-saving optimization (don't wake Qwen for simple tasks)
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from utils.logger import get_logger
from core.system_orchestrator import SystemOrchestrator


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

        logger.info(f"✅ Shogun v{self.VERSION} initialization complete")

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
