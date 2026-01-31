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
from utils.opus_client import OpusClient, OpusReview
from utils.quality_metrics import QualityMetricsCollector, RiskLevel
from core.karo import Karo
from core.system_orchestrator import SystemOrchestrator


logger = get_logger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for delegation decisions"""
    SIMPLE = "simple"      # 10 seconds - Direct answers, simple queries
    MEDIUM = "medium"      # 25 seconds - Standard implementation tasks
    COMPLEX = "complex"    # 40 seconds - Multi-component architecture
    STRATEGIC = "strategic" # 60 seconds - High-level design decisions


class ReviewLevel(Enum):
    """Review depth levels for quality assurance"""
    BASIC = "basic"          # Simple/Medium - Sonnet basic review (5秒, ¥0 Pro内)
    DETAILED = "detailed"    # Complex - Sonnet detailed review (10秒, ¥0-5 API)
    PREMIUM = "premium"      # Strategic - Opus premium review (15秒, ¥10) 🏆


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
        self.opus_client: Optional[OpusClient] = None
        self.quality_metrics = QualityMetricsCollector()
        self.karo: Optional[Karo] = None
        self.memory_mcp = None
        
        # Statistics
        self.reviews_by_level = {
            ReviewLevel.BASIC: 0,
            ReviewLevel.DETAILED: 0,
            ReviewLevel.PREMIUM: 0
        }
        
    async def initialize(self) -> None:
        """Initialize Shogun and subordinate systems"""
        logger.info("🎌 Initializing Shogun (Strategic Layer)...")
        
        # Initialize Opus client for premium reviews
        self.opus_client = OpusClient(
            api_key=self.orchestrator.config.claude_api_key
        )
        logger.info("🏆 Opus premium review system enabled")
        
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
                
                # Step 4: Adaptive quality review
                result = await self._adaptive_review(task, result)
            
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
    
    async def _adaptive_review(self, task: Task, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adaptive review based on task complexity and quality metrics
        
        Review levels:
        - Simple/Medium  → Sonnet Basic (5秒, ¥0 Pro内)
        - Complex        → Sonnet Detailed (10秒, ¥0-5 API)
        - Strategic      → Opus Premium (15秒, ¥10) 🏆
        """
        
        if result.get("status") != "completed":
            return result  # Pass through failed tasks
        
        # Collect quality metrics first
        implementation = result.get("result", "")
        if isinstance(implementation, str) and len(implementation) > 50:
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
        else:
            quality_report = None
        
        # Determine review level
        review_level = self._determine_review_level(task, quality_report)
        
        logger.info(f"🔍 Starting {review_level.value} review for {task.complexity.value} task")
        
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
        """
        Determine appropriate review level based on:
        - Task complexity
        - Quality metrics (risk level, security score)
        - User preferences
        """
        
        # Strategic tasks always get Opus premium review
        if task.complexity == TaskComplexity.STRATEGIC:
            return ReviewLevel.PREMIUM
        
        # High/critical risk code gets Opus review
        if quality_report and quality_report.complexity_metrics.risk_level in [
            RiskLevel.HIGH, RiskLevel.CRITICAL
        ]:
            logger.info(f"🚨 High risk detected, upgrading to Opus review")
            return ReviewLevel.PREMIUM
        
        # Security issues trigger Opus review
        if quality_report and quality_report.security_findings.vulnerabilities:
            logger.info(f"🔒 Security vulnerabilities detected, upgrading to Opus review")
            return ReviewLevel.PREMIUM
        
        # Complex tasks get detailed Sonnet review
        if task.complexity == TaskComplexity.COMPLEX:
            return ReviewLevel.DETAILED
        
        # Simple/Medium get basic review
        return ReviewLevel.BASIC
    
    async def _opus_premium_review(self, task: Task, result: Dict[str, Any], quality_report) -> Dict[str, Any]:
        """
        Premium quality inspection using Claude Opus
        
        Cost: ~¥10/review
        Quality: 98-99点保証
        Use cases: Strategic decisions, critical implementations
        """
        
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
            # Fallback to Sonnet detailed review
            logger.info("🔄 Falling back to Sonnet detailed review")
            result = await self._sonnet_detailed_review(task, result, quality_report)
        
        return result
    
    async def _sonnet_detailed_review(self, task: Task, result: Dict[str, Any], quality_report) -> Dict[str, Any]:
        """
        Detailed review using Claude Sonnet with enhanced prompt
        
        Cost: ¥0-5/review (mostly Pro, occasional API)
        Quality: 95-97点
        """
        
        implementation = result.get("result", "")
        
        quality_context = ""
        if quality_report:
            quality_context = f"""
Quality Metrics:
- Complexity Score: {quality_report.complexity_metrics.complexity_score}/100
- Security Score: {quality_report.security_findings.security_score}/100
- Risk Level: {quality_report.complexity_metrics.risk_level.value}
- Recommendations: {len(quality_report.recommendations)} items
"""
        
        review_prompt = f"""
As the Shogun (strategic decision maker), conduct a DETAILED review of this implementation:

Task: {task.content}

Implementation:
{implementation[:2000]}  # Limit to 2000 chars

{quality_context}

Evaluate comprehensively:

1. **Functional Correctness** (40 points):
   - Logic soundness and edge cases
   - Algorithm efficiency
   - Output validation

2. **Code Quality** (30 points):
   - Architecture and patterns
   - Readability and maintainability
   - Code organization

3. **Security** (20 points):
   - Input validation
   - Authentication/authorization
   - Data handling

4. **Best Practices** (10 points):
   - Error handling
   - Documentation
   - Testing

Provide:
- **Score**: X/100
- **Decision**: APPROVED / REVISE_REQUIRED
- **Key Issues**: List critical problems
- **Recommendations**: List improvements
"""
        
        try:
            review = await self.claude_client.generate(
                messages=[{"role": "user", "content": review_prompt}],
                max_tokens=800,
                temperature=0.1
            )
            
            # Parse score
            score = 85.0  # Default
            if "Score" in review or "SCORE" in review:
                import re
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
        """
        Basic review using Claude Sonnet (original implementation)
        
        Cost: ¥0 (Pro limit)
        Quality: 90-93点
        Speed: 5 seconds
        """
        
        review_prompt = f"""
        As the Shogun, review this completed task:
        
        Original Task: {task.content}
        Result: {result.get("result", "No result provided")[:1000]}
        
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
                logger.info("✅ Shogun basic review approved")
            else:
                result["shogun_feedback"] = review
                result["shogun_approval"] = "feedback_provided"
                logger.info("📝 Shogun basic review: feedback provided")
                
        except Exception as e:
            logger.warning(f"⚠️ Basic review failed: {e}")
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
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive review statistics
        
        Returns statistics about:
        - Reviews by level (Basic/Detailed/Premium)
        - Opus usage and costs
        - Quality metrics aggregates
        """
        
        stats = {
            "reviews_by_level": {
                level.value: count 
                for level, count in self.reviews_by_level.items()
            },
            "total_reviews": sum(self.reviews_by_level.values())
        }
        
        if self.opus_client:
            stats["opus_statistics"] = self.opus_client.get_statistics()
        
        stats["quality_metrics"] = self.quality_metrics.get_aggregate_stats()
        
        return stats