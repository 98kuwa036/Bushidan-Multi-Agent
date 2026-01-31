"""
Bushidan Multi-Agent System v9.3.2 - Intelligent Routing Orchestrator

NEW v9.3.2: Smart routing decision tree

Routing Logic:
将軍 (Shogun) judges task difficulty
  ↓
  Simple → Groq (爆速・無料・Qwen起こさない)
  ↓
  Medium/Complex → Local Qwen3 (4096 context)
    → Fails → Cloud Qwen3-plus (Kagemusha)
    → Fails → Gemini 3 Flash (Final defense)
  ↓
  Strategic → Shogun self-handles
  ↓
  Final Review → Opus (if Critical)

Key Features:
- Power-saving: Groq for Simple tasks (don't wake Qwen)
- Cost-optimization: Local Qwen primary (¥0)
- Reliability: 3-tier fallback (Local → Cloud → Gemini)
- Quality: Opus final inspection for Critical tasks
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger
from utils.groq_client import GroqClient
from utils.qwen3_client import Qwen3Client
from utils.alibaba_qwen_client import AlibabaQwenClient
from utils.gemini3_client import Gemini3Client


logger = get_logger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"        # Groq path
    MEDIUM = "medium"        # Qwen3 path
    COMPLEX = "complex"      # Qwen3 path with high priority
    STRATEGIC = "strategic"  # Shogun self-handles


class RoutingDecision(Enum):
    """Routing destination"""
    GROQ = "groq"                    # Simple tasks
    LOCAL_QWEN = "local_qwen"        # Medium/Complex tasks
    CLOUD_QWEN = "cloud_qwen"        # Local Qwen fallback
    GEMINI3 = "gemini3"              # Final defense
    SHOGUN_SELF = "shogun_self"      # Strategic tasks


@dataclass
class RoutingResult:
    """Result of routing decision"""
    decision: RoutingDecision
    complexity: TaskComplexity
    reasoning: str
    estimated_cost_jpy: float
    estimated_latency_ms: int
    power_saving: bool  # True if avoided Qwen wakeup
    fallback_chain: list


@dataclass
class ExecutionResult:
    """Result of task execution through routing"""
    content: str
    actual_route: RoutingDecision
    fallback_used: bool
    attempts: int
    total_cost_jpy: float
    total_latency_ms: int
    success: bool
    error_message: Optional[str] = None


class IntelligentRouter:
    """
    Intelligent Routing Orchestrator
    
    Implements v9.3.2 routing logic:
    - Judges task complexity
    - Routes to optimal model
    - Manages fallback chain
    - Tracks performance and costs
    - Power-saving optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize intelligent router
        
        Args:
            config: System configuration with API keys
        """
        self.config = config
        
        # Initialize all clients
        self.groq_client = GroqClient(
            api_key=config.get("groq_api_key", "")
        )
        
        self.local_qwen_client = Qwen3Client(
            config=config,
            api_base=config.get("qwen_api_base", "http://192.168.1.11:11434"),
            context_length=4096  # v9.3.2 optimization
        )
        
        self.cloud_qwen_client = AlibabaQwenClient(
            api_key=config.get("alibaba_api_key", "")
        )
        
        self.gemini3_client = Gemini3Client(
            api_key=config.get("gemini_api_key", "")
        )
        
        # Routing statistics
        self.routing_stats = {
            RoutingDecision.GROQ: 0,
            RoutingDecision.LOCAL_QWEN: 0,
            RoutingDecision.CLOUD_QWEN: 0,
            RoutingDecision.GEMINI3: 0,
            RoutingDecision.SHOGUN_SELF: 0
        }
        
        self.fallback_stats = {
            "local_to_cloud": 0,
            "cloud_to_gemini": 0,
            "total_fallbacks": 0
        }
        
        self.power_savings = {
            "qwen_wakeups_avoided": 0,
            "estimated_kwh_saved": 0.0
        }
        
        logger.info("🔀 Intelligent Router initialized with v9.3.2 logic")
    
    def judge_complexity(self, task_description: str, context: Optional[Dict] = None) -> TaskComplexity:
        """
        Judge task complexity using heuristics
        
        Args:
            task_description: Task description
            context: Optional context information
        
        Returns:
            TaskComplexity enum
        """
        
        task_lower = task_description.lower()
        
        # Strategic indicators
        strategic_keywords = [
            "architecture", "design decision", "技術選定", "アーキテクチャ",
            "strategy", "戦略", "方針決定", "技術方針"
        ]
        if any(kw in task_lower for kw in strategic_keywords):
            return TaskComplexity.STRATEGIC
        
        # Complex indicators
        complex_keywords = [
            "refactor", "リファクタリング", "architecture", "multiple files",
            "large codebase", "システム設計", "複雑な実装"
        ]
        if any(kw in task_lower for kw in complex_keywords):
            return TaskComplexity.COMPLEX
        
        # Simple indicators
        simple_keywords = [
            "what is", "explain", "簡単な質問", "教えて", "とは",
            "simple question", "quick answer"
        ]
        if any(kw in task_lower for kw in simple_keywords):
            return TaskComplexity.SIMPLE
        
        # Check length as fallback
        if len(task_description) < 100:
            return TaskComplexity.SIMPLE
        elif len(task_description) > 500:
            return TaskComplexity.COMPLEX
        
        # Default to Medium
        return TaskComplexity.MEDIUM
    
    def make_routing_decision(
        self,
        complexity: TaskComplexity,
        local_qwen_status: Optional[Dict] = None
    ) -> RoutingResult:
        """
        Make routing decision based on complexity and system status
        
        Args:
            complexity: Task complexity
            local_qwen_status: Local Qwen health/performance status
        
        Returns:
            RoutingResult with decision and metadata
        """
        
        # Strategic tasks → Shogun handles directly
        if complexity == TaskComplexity.STRATEGIC:
            return RoutingResult(
                decision=RoutingDecision.SHOGUN_SELF,
                complexity=complexity,
                reasoning="Strategic task requires Shogun's deep insight",
                estimated_cost_jpy=5.0,
                estimated_latency_ms=5000,
                power_saving=False,
                fallback_chain=[]
            )
        
        # Simple tasks → Groq (power-saving)
        if complexity == TaskComplexity.SIMPLE:
            return RoutingResult(
                decision=RoutingDecision.GROQ,
                complexity=complexity,
                reasoning="Simple task → Groq instant response (don't wake Qwen)",
                estimated_cost_jpy=0.0,
                estimated_latency_ms=800,
                power_saving=True,  # Avoid Qwen wakeup
                fallback_chain=[RoutingDecision.LOCAL_QWEN, RoutingDecision.GEMINI3]
            )
        
        # Medium/Complex → Local Qwen3 primary
        # Check if Local Qwen is struggling
        if local_qwen_status:
            consecutive_failures = local_qwen_status.get("consecutive_failures", 0)
            success_rate = local_qwen_status.get("success_rate", 100.0)
            
            # Skip local if struggling badly
            if consecutive_failures >= 2 or success_rate < 50:
                logger.warning(
                    f"⚠️ Local Qwen struggling (failures: {consecutive_failures}, "
                    f"success: {success_rate:.1f}%), routing to Cloud Qwen"
                )
                return RoutingResult(
                    decision=RoutingDecision.CLOUD_QWEN,
                    complexity=complexity,
                    reasoning="Local Qwen struggling → Cloud Qwen Kagemusha",
                    estimated_cost_jpy=3.0,
                    estimated_latency_ms=3000,
                    power_saving=False,
                    fallback_chain=[RoutingDecision.GEMINI3]
                )
        
        # Default: Local Qwen3 with full fallback chain
        return RoutingResult(
            decision=RoutingDecision.LOCAL_QWEN,
            complexity=complexity,
            reasoning="Medium/Complex task → Local Qwen3 (cost ¥0, 4096 context)",
            estimated_cost_jpy=0.0,
            estimated_latency_ms=2500,
            power_saving=False,
            fallback_chain=[RoutingDecision.CLOUD_QWEN, RoutingDecision.GEMINI3]
        )
    
    async def execute_with_routing(
        self,
        task_description: str,
        messages: list,
        max_tokens: int = 2000,
        temperature: float = 0.2
    ) -> ExecutionResult:
        """
        Execute task with intelligent routing and automatic fallback
        
        Args:
            task_description: Task description for complexity judgment
            messages: Chat messages
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
        
        Returns:
            ExecutionResult with content and metadata
        """
        
        # Step 1: Judge complexity
        complexity = self.judge_complexity(task_description)
        logger.info(f"📊 Complexity judged: {complexity.value}")
        
        # Step 2: Get local Qwen status
        local_status = self.local_qwen_client.get_usage_stats().get("performance", {})
        
        # Step 3: Make routing decision
        routing = self.make_routing_decision(complexity, local_status)
        logger.info(
            f"🔀 Routing decision: {routing.decision.value} "
            f"(reasoning: {routing.reasoning})"
        )
        
        # Step 4: Execute with fallback chain
        start_time = datetime.now()
        attempts = 0
        total_cost = 0.0
        actual_route = routing.decision
        fallback_used = False
        
        # Try primary route
        try:
            content = await self._execute_route(
                route=routing.decision,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            attempts = 1
            total_cost = routing.estimated_cost_jpy
            
            # Update stats
            self.routing_stats[routing.decision] += 1
            if routing.power_saving:
                self.power_savings["qwen_wakeups_avoided"] += 1
                self.power_savings["estimated_kwh_saved"] += 0.1
            
            total_latency = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(
                f"✅ Execution success via {routing.decision.value} "
                f"(latency: {total_latency}ms, cost: ¥{total_cost:.2f})"
            )
            
            return ExecutionResult(
                content=content,
                actual_route=actual_route,
                fallback_used=False,
                attempts=attempts,
                total_cost_jpy=total_cost,
                total_latency_ms=total_latency,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Primary route {routing.decision.value} failed: {e}")
            fallback_used = True
            attempts += 1
        
        # Try fallback chain
        for fallback_route in routing.fallback_chain:
            logger.info(f"🔄 Attempting fallback: {fallback_route.value}")
            
            try:
                content = await self._execute_route(
                    route=fallback_route,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                attempts += 1
                actual_route = fallback_route
                
                # Calculate cost
                if fallback_route == RoutingDecision.CLOUD_QWEN:
                    total_cost += 3.0
                    self.fallback_stats["local_to_cloud"] += 1
                elif fallback_route == RoutingDecision.GEMINI3:
                    total_cost += 0.04
                    if routing.decision == RoutingDecision.CLOUD_QWEN:
                        self.fallback_stats["cloud_to_gemini"] += 1
                
                self.fallback_stats["total_fallbacks"] += 1
                self.routing_stats[fallback_route] += 1
                
                total_latency = int((datetime.now() - start_time).total_seconds() * 1000)
                
                logger.info(
                    f"✅ Fallback success via {fallback_route.value} "
                    f"(attempts: {attempts}, cost: ¥{total_cost:.2f})"
                )
                
                return ExecutionResult(
                    content=content,
                    actual_route=actual_route,
                    fallback_used=True,
                    attempts=attempts,
                    total_cost_jpy=total_cost,
                    total_latency_ms=total_latency,
                    success=True
                )
                
            except Exception as e:
                logger.error(f"❌ Fallback {fallback_route.value} failed: {e}")
                attempts += 1
                continue
        
        # All routes failed
        total_latency = int((datetime.now() - start_time).total_seconds() * 1000)
        
        logger.error(f"❌ All routing attempts failed (attempts: {attempts})")
        
        return ExecutionResult(
            content="",
            actual_route=routing.decision,
            fallback_used=True,
            attempts=attempts,
            total_cost_jpy=total_cost,
            total_latency_ms=total_latency,
            success=False,
            error_message="All routing attempts exhausted"
        )
    
    async def _execute_route(
        self,
        route: RoutingDecision,
        messages: list,
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Execute generation via specific route
        
        Args:
            route: Routing decision
            messages: Chat messages
            max_tokens: Maximum tokens
            temperature: Sampling temperature
        
        Returns:
            Generated content
        
        Raises:
            Exception: If generation fails
        """
        
        if route == RoutingDecision.GROQ:
            return await self.groq_client.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                avoid_qwen_wakeup=True
            )
        
        elif route == RoutingDecision.LOCAL_QWEN:
            return await self.local_qwen_client.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                auto_compress=True
            )
        
        elif route == RoutingDecision.CLOUD_QWEN:
            return await self.cloud_qwen_client.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                activation_reason="local_failure"
            )
        
        elif route == RoutingDecision.GEMINI3:
            return await self.gemini3_client.generate(
                messages=messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
                activation_reason="qwen_fallback"
            )
        
        else:
            raise Exception(f"Unsupported route: {route}")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        
        total_routes = sum(self.routing_stats.values())
        
        return {
            "total_routed_tasks": total_routes,
            "routing_distribution": {
                route.value: {
                    "count": count,
                    "percentage": round((count / total_routes * 100) if total_routes > 0 else 0, 1)
                }
                for route, count in self.routing_stats.items()
            },
            "fallback_statistics": self.fallback_stats,
            "fallback_rate_percent": round(
                (self.fallback_stats["total_fallbacks"] / total_routes * 100)
                if total_routes > 0 else 0, 1
            ),
            "power_savings": self.power_savings,
            "estimated_monthly_savings_jpy": round(
                self.power_savings["qwen_wakeups_avoided"] * 0.5, 2
            ),
            "client_statistics": {
                "groq": self.groq_client.get_usage_stats(),
                "local_qwen": self.local_qwen_client.get_usage_stats(),
                "cloud_qwen": self.cloud_qwen_client.get_usage_stats(),
                "gemini3": self.gemini3_client.get_usage_stats()
            }
        }
