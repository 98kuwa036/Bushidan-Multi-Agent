"""
Bushidan Multi-Agent System v9.3 - Opus Client

Claude Opus client for premium quality inspection.
Cost: ~¥10/review, Quality: 98-99点保証
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class OpusReview:
    """Opus review result"""
    score: float
    decision: str  # "approved", "revise_required", "reject"
    critical_issues: List[str]
    recommendations: List[str]
    review_text: str
    cost_yen: float
    timestamp: str
    review_time_seconds: float


class OpusClient:
    """
    Claude Opus client for premium quality inspection
    
    Use cases:
    - Strategic level tasks (highest importance)
    - Critical implementations (security, finance, healthcare)
    - Final gate before production deployment
    
    Cost: ~¥10 per review
    Quality guarantee: 98-99点
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "claude-opus-4-20250514"  # Latest Opus
        self.reviews_conducted = 0
        self.total_cost_yen = 0.0
        
        # Cost calculation (as of 2025-01-31)
        self.cost_per_1m_input_tokens_usd = 15.0
        self.cost_per_1m_output_tokens_usd = 75.0
        self.usd_to_jpy = 150.0  # Approximate rate
        
    async def conduct_premium_review(
        self, 
        task_content: str,
        implementation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> OpusReview:
        """
        Conduct premium quality inspection using Opus
        
        This is the FINAL gate before production. Maximum scrutiny.
        
        Args:
            task_content: Original task description
            implementation: The code/result to review
            context: Additional context (complexity, risk level, etc.)
            
        Returns:
            OpusReview with detailed analysis
        """
        
        start_time = asyncio.get_event_loop().time()
        logger.info("🏆 Starting Opus premium quality review...")
        
        try:
            # Build comprehensive review prompt
            review_prompt = self._build_review_prompt(
                task_content, 
                implementation, 
                context
            )
            
            # Call Opus API
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            response = await client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": review_prompt}],
                max_tokens=1500,
                temperature=0.0  # Maximum precision for reviews
            )
            
            review_text = response.content[0].text
            
            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_yen = self._calculate_cost(input_tokens, output_tokens)
            
            # Parse review results
            opus_review = self._parse_review(review_text, cost_yen)
            opus_review.review_time_seconds = asyncio.get_event_loop().time() - start_time
            opus_review.timestamp = datetime.now().isoformat()
            
            # Update statistics
            self.reviews_conducted += 1
            self.total_cost_yen += cost_yen
            
            logger.info(
                f"✅ Opus review complete: Score {opus_review.score}/100, "
                f"Decision: {opus_review.decision}, Cost: ¥{cost_yen:.2f}"
            )
            
            return opus_review
            
        except Exception as e:
            logger.error(f"❌ Opus review failed: {e}")
            # Return failed review
            return OpusReview(
                score=0.0,
                decision="review_failed",
                critical_issues=[f"Review system error: {str(e)}"],
                recommendations=["Please retry review or use fallback Sonnet detailed review"],
                review_text=f"Review failed: {str(e)}",
                cost_yen=0.0,
                timestamp=datetime.now().isoformat(),
                review_time_seconds=asyncio.get_event_loop().time() - start_time
            )
    
    def _build_review_prompt(
        self, 
        task: str, 
        implementation: str, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build comprehensive review prompt for Opus"""
        
        context_info = ""
        if context:
            risk_level = context.get("risk_level", "medium")
            complexity = context.get("complexity", "unknown")
            context_info = f"""
Context:
- Risk Level: {risk_level}
- Complexity: {complexity}
"""
        
        prompt = f"""
As the Supreme Quality Inspector (Opus) in the Bushidan Multi-Agent System, conduct the most rigorous code review possible.

This is the FINAL gate before production deployment. Your review determines whether this implementation is safe and correct.

Task:
{task}

Implementation:
{implementation}

{context_info}

Evaluate with MAXIMUM SCRUTINY across these dimensions:

1. **Functional Correctness** (40 points):
   - Logic soundness and completeness
   - Edge case handling (null, empty, boundary values)
   - Algorithm optimality and efficiency
   - Output validation
   - Error propagation

2. **Code Quality** (30 points):
   - Architecture and design patterns
   - Readability and maintainability
   - Code organization and modularity
   - Naming conventions
   - Documentation completeness
   - DRY principle adherence

3. **Security & Safety** (20 points):
   - Input validation and sanitization
   - SQL injection prevention
   - XSS prevention
   - Authentication/authorization
   - Sensitive data handling (no hardcoded secrets)
   - Resource exhaustion prevention
   - OWASP Top 10 compliance

4. **Best Practices** (10 points):
   - Industry standards compliance
   - Testing adequacy (unit, integration)
   - Error handling patterns
   - Logging appropriateness
   - Performance considerations
   - Dependency management

Provide your review in this EXACT format:

SCORE: [X]/100

DECISION: [APPROVED / REVISE_REQUIRED / REJECT]

CRITICAL ISSUES:
- [Issue 1 that MUST be fixed before deployment]
- [Issue 2 that MUST be fixed before deployment]
(or "None" if no critical issues)

RECOMMENDATIONS:
- [Improvement 1 that SHOULD be addressed]
- [Improvement 2 that SHOULD be addressed]
(or "None" if no recommendations)

DETAILED ANALYSIS:
[Your comprehensive analysis here]

Remember: This is the final quality gate. Be thorough and uncompromising.
"""
        
        return prompt
    
    def _parse_review(self, review_text: str, cost_yen: float) -> OpusReview:
        """Parse Opus review response into structured format"""
        
        # Extract score
        score = 0.0
        if "SCORE:" in review_text:
            try:
                score_line = [l for l in review_text.split('\n') if 'SCORE:' in l][0]
                score_str = score_line.split('SCORE:')[1].split('/')[0].strip()
                score = float(score_str)
            except:
                logger.warning("Failed to parse score from Opus review")
        
        # Extract decision
        decision = "review_failed"
        if "DECISION:" in review_text:
            decision_line = [l for l in review_text.split('\n') if 'DECISION:' in l][0]
            decision_str = decision_line.split('DECISION:')[1].strip().upper()
            if "APPROVED" in decision_str:
                decision = "approved"
            elif "REVISE" in decision_str:
                decision = "revise_required"
            elif "REJECT" in decision_str:
                decision = "reject"
        
        # Extract critical issues
        critical_issues = []
        if "CRITICAL ISSUES:" in review_text:
            try:
                issues_section = review_text.split("CRITICAL ISSUES:")[1].split("RECOMMENDATIONS:")[0]
                for line in issues_section.split('\n'):
                    line = line.strip()
                    if line.startswith('-') and "None" not in line:
                        critical_issues.append(line[1:].strip())
            except:
                logger.warning("Failed to parse critical issues")
        
        # Extract recommendations
        recommendations = []
        if "RECOMMENDATIONS:" in review_text:
            try:
                rec_section = review_text.split("RECOMMENDATIONS:")[1].split("DETAILED ANALYSIS:")[0]
                for line in rec_section.split('\n'):
                    line = line.strip()
                    if line.startswith('-') and "None" not in line:
                        recommendations.append(line[1:].strip())
            except:
                logger.warning("Failed to parse recommendations")
        
        return OpusReview(
            score=score,
            decision=decision,
            critical_issues=critical_issues,
            recommendations=recommendations,
            review_text=review_text,
            cost_yen=cost_yen,
            timestamp="",  # Will be set by caller
            review_time_seconds=0.0  # Will be set by caller
        )
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in JPY for this review"""
        
        input_cost_usd = (input_tokens / 1_000_000) * self.cost_per_1m_input_tokens_usd
        output_cost_usd = (output_tokens / 1_000_000) * self.cost_per_1m_output_tokens_usd
        total_usd = input_cost_usd + output_cost_usd
        
        return total_usd * self.usd_to_jpy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get review statistics"""
        
        return {
            "total_reviews": self.reviews_conducted,
            "total_cost_yen": round(self.total_cost_yen, 2),
            "average_cost_per_review_yen": (
                round(self.total_cost_yen / self.reviews_conducted, 2) 
                if self.reviews_conducted > 0 else 0
            ),
            "model": self.model
        }
