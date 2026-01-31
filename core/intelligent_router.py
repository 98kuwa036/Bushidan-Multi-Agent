"""
Bushidan Multi-Agent System v9.3.2 - Intelligent Router

Smart routing orchestrator that implements the new 4-tier hybrid architecture:
- Simple tasks → Groq (instant, free, power-saving)
- Medium/Complex → Local Qwen3 → Cloud Qwen3-plus → Gemini 3 Flash
- Strategic → Shogun self-handles

This router optimizes for speed, cost, and reliability through intelligent
task delegation and 3-tier fallback management.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger


logger = get_logger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions"""
    SIMPLE = "simple"          # 2s - Groq handles instantly
    MEDIUM = "medium"          # 12s - Local Qwen3 → Cloud fallback
    COMPLEX = "complex"        # 28s - Local → Cloud → Gemini chain
    STRATEGIC = "strategic"    # 45s - Shogun handles directly


class RouteTarget(Enum):
    """Routing targets in the hybrid architecture"""
    GROQ = "groq"                      # Lightning fast (300-500 tok/s)
    LOCAL_QWEN3 = "local_qwen3"        # Local Qwen3-Coder-30B (4096 context)
    CLOUD_QWEN3 = "cloud_qwen3"        # Alibaba Cloud Qwen3-plus (32k context)
    GEMINI3 = "gemini3"                # Gemini 3 Flash (final defense)
    SHOGUN = "shogun"                  # Claude Sonnet (strategic only)


@dataclass
class RoutingDecision:
    """Represents a routing decision"""
    target: RouteTarget
    complexity: TaskComplexity
    fallback_chain: list[RouteTarget]
    reasoning: str
    estimated_time_seconds: int
    estimated_cost_yen: float
    power_saving: bool  # Whether Qwen should sleep


@dataclass
class RoutingStats:
    """Statistics for routing performance"""
    total_tasks: int = 0
    routes_by_target: Dict[str, int] = None
    fallbacks_triggered: int = 0
    average_time_seconds: float = 0.0
    total_cost_yen: float = 0.0
    power_savings_yen: float = 0.0
    
    def __post_init__(self):
        if self.routes_by_target is None:
            self.routes_by_target = {target.value: 0 for target in RouteTarget}


class IntelligentRouter:
    """
    Intelligent Routing Orchestrator for v9.3.2
    
    Implements the golden rules (運用黄金律):
    1. Simple → Groq (instant, free, power-saving)
    2. Heavy → Local Qwen3 (local volume, ¥0)
    3. Difficult → Cloud/Gemini (quality backup)
    4. Strategic → Shogun (final authority)
    
    Features:
    - Complexity-based routing heuristics
    - 3-tier fallback management
    - Power-saving logic (don't wake Qwen for simple tasks)
    - Cost and performance tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = RoutingStats()
        self.routing_history: list[Tuple[datetime, RoutingDecision]] = []
        
        # Performance targets (seconds)
        self.target_times = {
            TaskComplexity.SIMPLE: 2,
            TaskComplexity.MEDIUM: 12,
            TaskComplexity.COMPLEX: 28,
            TaskComplexity.STRATEGIC: 45
        }
        
        # Cost estimates (yen)
        self.cost_estimates = {
            RouteTarget.GROQ: 0.0,           # Free tier
            RouteTarget.LOCAL_QWEN3: 0.0,    # Local inference
            RouteTarget.CLOUD_QWEN3: 3.0,    # Alibaba Cloud API
            RouteTarget.GEMINI3: 0.04,       # Gemini 3 Flash
            RouteTarget.SHOGUN: 0.0          # Pro CLI (within quota)
        }
    
    def judge_complexity(self, task_content: str, context: Optional[Dict[str, Any]] = None) -> TaskComplexity:
        """
        Judge task complexity using heuristics
        
        Complexity indicators:
        - SIMPLE: Questions, lookups, simple queries (< 50 chars, no code)
        - MEDIUM: Standard implementation, single file (code keywords, reasonable length)
        - COMPLEX: Multi-component, architecture (multiple files, "system", "architecture")
        - STRATEGIC: Design decisions, technology choices (strategic keywords)
        
        Args:
            task_content: The task description
            context: Optional context (past tasks, project info)
        
        Returns:
            TaskComplexity level
        """
        
        # Strategic indicators
        strategic_keywords = [
            "設計", "アーキテクチャ", "技術選定", "architecture", "design decision",
            "technology choice", "strategy", "long-term", "システム全体"
        ]
        
        # Complex indicators
        complex_keywords = [
            "複数", "multiple files", "refactor", "リファクタリング",
            "システム", "統合", "integration", "multi-component"
        ]
        
        # Code/implementation indicators
        code_keywords = [
            "実装", "コード", "implement", "code", "function", "class",
            "プログラム", "create", "build", "develop"
        ]
        
        task_lower = task_content.lower()
        task_length = len(task_content)
        
        # Strategic: Contains strategic keywords
        if any(keyword in task_lower for keyword in strategic_keywords):
            logger.info("📊 Complexity: STRATEGIC (strategic keywords detected)")
            return TaskComplexity.STRATEGIC
        
        # Complex: Contains complex indicators or very long
        if any(keyword in task_lower for keyword in complex_keywords) or task_length > 300:
            logger.info("📊 Complexity: COMPLEX (complex indicators or length > 300)")
            return TaskComplexity.COMPLEX
        
        # Medium: Contains code keywords and reasonable length
        if any(keyword in task_lower for keyword in code_keywords):
            if task_length > 100:
                logger.info("📊 Complexity: MEDIUM (code task, length > 100)")
                return TaskComplexity.MEDIUM
            else:
                logger.info("📊 Complexity: SIMPLE (code task but short)")
                return TaskComplexity.SIMPLE
        
        # Simple: Short queries, questions
        if task_length < 50:
            logger.info("📊 Complexity: SIMPLE (length < 50)")
            return TaskComplexity.SIMPLE
        
        # Default: MEDIUM for ambiguous cases
        logger.info("📊 Complexity: MEDIUM (default for ambiguous)")
        return TaskComplexity.MEDIUM
    
    def determine_route(self, complexity: TaskComplexity, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Determine optimal route based on complexity and context
        
        Routing logic (運用黄金律):
        - SIMPLE → Groq (instant, free, DON'T wake Qwen)
        - MEDIUM → Local Qwen3 → Cloud Qwen3 → Gemini3
        - COMPLEX → Local Qwen3 → Cloud Qwen3 → Gemini3
        - STRATEGIC → Shogun (Claude Sonnet handles directly)
        
        Args:
            complexity: Task complexity level
            context: Optional context for routing decision
        
        Returns:
            RoutingDecision with target, fallback chain, and metadata
        """
        
        if complexity == TaskComplexity.STRATEGIC:
            # Strategic: Shogun handles directly (no delegation)
            return RoutingDecision(
                target=RouteTarget.SHOGUN,
                complexity=complexity,
                fallback_chain=[RouteTarget.SHOGUN],  # No fallback for strategic
                reasoning="Strategic task requires highest authority (Shogun)",
                estimated_time_seconds=self.target_times[complexity],
                estimated_cost_yen=self.cost_estimates[RouteTarget.SHOGUN],
                power_saving=True  # Qwen stays asleep
            )
        
        elif complexity == TaskComplexity.SIMPLE:
            # Simple: Groq handles (lightning fast, free, power-saving)
            return RoutingDecision(
                target=RouteTarget.GROQ,
                complexity=complexity,
                fallback_chain=[RouteTarget.GROQ, RouteTarget.GEMINI3],  # Gemini backup
                reasoning="Simple task → Groq (instant, free, DON'T wake Qwen)",
                estimated_time_seconds=self.target_times[complexity],
                estimated_cost_yen=self.cost_estimates[RouteTarget.GROQ],
                power_saving=True  # Qwen stays asleep
            )
        
        else:
            # Medium/Complex: 3-tier fallback chain
            return RoutingDecision(
                target=RouteTarget.LOCAL_QWEN3,
                complexity=complexity,
                fallback_chain=[
                    RouteTarget.LOCAL_QWEN3,   # Primary: Local Qwen3 (¥0)
                    RouteTarget.CLOUD_QWEN3,   # Shadow: Cloud Qwen3-plus (¥3)
                    RouteTarget.GEMINI3        # Final defense: Gemini 3 Flash (¥0.04)
                ],
                reasoning=f"{complexity.value.title()} task → 3-tier fallback (Local → Cloud → Gemini)",
                estimated_time_seconds=self.target_times[complexity],
                estimated_cost_yen=self.cost_estimates[RouteTarget.LOCAL_QWEN3],  # Optimistic (local)
                power_saving=False  # Wake Qwen for work
            )
    
    def handle_fallback(self, original_target: RouteTarget, error: Exception, fallback_chain: list[RouteTarget]) -> Optional[RouteTarget]:
        """
        Handle fallback when primary target fails
        
        Args:
            original_target: The target that failed
            error: The error that occurred
            fallback_chain: Ordered list of fallback targets
        
        Returns:
            Next fallback target, or None if chain exhausted
        """
        
        try:
            current_index = fallback_chain.index(original_target)
            if current_index + 1 < len(fallback_chain):
                next_target = fallback_chain[current_index + 1]
                logger.warning(f"⚠️ Fallback: {original_target.value} failed ({error}), trying {next_target.value}")
                self.stats.fallbacks_triggered += 1
                return next_target
            else:
                logger.error(f"❌ Fallback chain exhausted for {original_target.value}")
                return None
        except ValueError:
            logger.error(f"❌ {original_target.value} not in fallback chain")
            return None
    
    def record_routing(self, decision: RoutingDecision, actual_time_seconds: float, actual_cost_yen: float, success: bool) -> None:
        """
        Record routing decision and actual performance
        
        Args:
            decision: The routing decision made
            actual_time_seconds: Actual execution time
            actual_cost_yen: Actual cost incurred
            success: Whether the execution succeeded
        """
        
        self.stats.total_tasks += 1
        self.stats.routes_by_target[decision.target.value] += 1
        self.stats.total_cost_yen += actual_cost_yen
        
        # Update average time (running average)
        n = self.stats.total_tasks
        self.stats.average_time_seconds = (
            (self.stats.average_time_seconds * (n - 1) + actual_time_seconds) / n
        )
        
        # Calculate power savings (if Qwen was not woken)
        if decision.power_saving:
            # Estimate: Local Qwen3 would cost ~¥5 in electricity per task
            self.stats.power_savings_yen += 5.0
        
        # Store in history (keep last 1000)
        self.routing_history.append((datetime.now(), decision))
        if len(self.routing_history) > 1000:
            self.routing_history.pop(0)
        
        # Log summary
        logger.info(
            f"📊 Routing recorded: {decision.target.value} | "
            f"Time: {actual_time_seconds:.1f}s | Cost: ¥{actual_cost_yen:.2f} | "
            f"Success: {success}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive routing statistics
        
        Returns:
            Dictionary with routing performance metrics
        """
        
        return {
            "total_tasks": self.stats.total_tasks,
            "routes_by_target": self.stats.routes_by_target,
            "fallbacks_triggered": self.stats.fallbacks_triggered,
            "average_time_seconds": round(self.stats.average_time_seconds, 2),
            "total_cost_yen": round(self.stats.total_cost_yen, 2),
            "power_savings_yen": round(self.stats.power_savings_yen, 2),
            "efficiency_metrics": {
                "groq_usage_ratio": self.stats.routes_by_target[RouteTarget.GROQ.value] / max(1, self.stats.total_tasks),
                "local_first_ratio": self.stats.routes_by_target[RouteTarget.LOCAL_QWEN3.value] / max(1, self.stats.total_tasks),
                "fallback_rate": self.stats.fallbacks_triggered / max(1, self.stats.total_tasks)
            }
        }
    
    def get_recommendations(self) -> list[str]:
        """
        Get optimization recommendations based on routing history
        
        Returns:
            List of actionable recommendations
        """
        
        recommendations = []
        stats = self.get_statistics()
        
        # Check Groq usage
        groq_ratio = stats["efficiency_metrics"]["groq_usage_ratio"]
        if groq_ratio < 0.3:
            recommendations.append(
                "⚡ Consider routing more simple tasks to Groq for speed and cost savings"
            )
        
        # Check fallback rate
        fallback_rate = stats["efficiency_metrics"]["fallback_rate"]
        if fallback_rate > 0.2:
            recommendations.append(
                f"⚠️ High fallback rate ({fallback_rate:.1%}). Consider investigating primary target reliability"
            )
        
        # Check average time vs targets
        if stats["average_time_seconds"] > 20:
            recommendations.append(
                "🐢 Average time exceeds target (20s). Review task complexity judgments"
            )
        
        # Check cost efficiency
        if self.stats.total_tasks > 0:
            avg_cost = stats["total_cost_yen"] / self.stats.total_tasks
            if avg_cost > 5:
                recommendations.append(
                    f"💰 Average cost per task (¥{avg_cost:.2f}) is high. Increase local Qwen3 usage"
                )
        
        if not recommendations:
            recommendations.append("✅ Routing performance is optimal")
        
        return recommendations
