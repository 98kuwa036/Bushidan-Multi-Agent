"""
Bushidan Multi-Agent System v9.3.2 - Groq Client

NEW v9.3.2: Groq added as MCP 足軽 member

Role: 爆速足軽 (Instant-response Ashigaru)
Model: Llama 3.3 70B Versatile
Speed: 300-500 tok/s (10-20x faster than typical APIs)
Cost: ¥0 (free tier 14,400 requests/day)

Use cases:
- Simple questions/answers
- Lightweight code generation
- Instant feedback
- Prototype creation

Routing logic:
将軍 judges Simple → Groq (DON'T wake up local Qwen)
"""

import asyncio
import httpx
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class GroqRateLimit:
    """Rate limit tracking"""
    requests_per_minute: int = 30
    tokens_per_minute: int = 14000
    requests_per_day: int = 14400
    
    # Current counts
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    requests_today: int = 0
    
    # Reset times
    minute_reset: datetime = None
    day_reset: datetime = None
    
    def __post_init__(self):
        now = datetime.now()
        self.minute_reset = now + timedelta(minutes=1)
        self.day_reset = now + timedelta(days=1)


@dataclass
class GroqUsage:
    """Usage tracking for Groq calls"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    tokens_per_second: float
    model: str = "llama-3.3-70b-versatile"


class GroqClient:
    """
    Groq API client for instant-response tasks
    
    Role: 爆速足軽 (Bakusoku Ashigaru - Lightning-fast Foot Soldier)
    
    Key features:
    - 300-500 tokens/second generation speed
    - Free tier: 14,400 requests/day
    - Perfect for Simple tasks
    - Power-saving: Don't wake up local Qwen for simple queries
    
    v9.3.2 Routing logic:
    将軍 (Shogun) judges task difficulty:
    - Simple → Groq (instant, free, Qwen stays asleep)
    - Medium/Complex → Local Qwen3 (cost ¥0, may need wakeup)
    - Strategic → Claude Sonnet (deep reasoning)
    
    Rate limits:
    - 30 RPM (requests per minute)
    - 14,000 TPM (tokens per minute)
    - 14,400 requests per day
    """
    
    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.groq.com/openai/v1/chat/completions"
    ):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key
            endpoint: API endpoint URL
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = "llama-3.3-70b-versatile"
        
        # Usage tracking
        self.calls_made = 0
        self.total_tokens = 0
        self.total_latency_ms = 0
        self.average_tokens_per_second = 0.0
        
        # Rate limiting
        self.rate_limit = GroqRateLimit()
        
        # Power saving stats
        self.qwen_wakeups_avoided = 0
        self.estimated_power_saved_kwh = 0.0  # Qwen wakeup ~0.1 kWh
        
        logger.info("⚡ Groq client (爆速足軽) initialized - 300-500 tok/s ready")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.3,
        avoid_qwen_wakeup: bool = True
    ) -> str:
        """
        Generate response using Groq's lightning-fast API
        
        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            avoid_qwen_wakeup: Track if this avoided waking local Qwen
        
        Returns:
            Generated text content
        
        Raises:
            Exception: If rate limit exceeded or API fails
        """
        
        # Check rate limits before calling
        await self._check_rate_limits()
        
        self.calls_made += 1
        if avoid_qwen_wakeup:
            self.qwen_wakeups_avoided += 1
            self.estimated_power_saved_kwh += 0.1  # Approximate Qwen wakeup cost
        
        logger.info(f"⚡ Groq generating (call #{self.calls_made}, avoided Qwen: {avoid_qwen_wakeup})")
        
        # Prepare request
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = datetime.now()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Calculate performance metrics
                latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                
                # Extract response
                if result.get("choices") and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    # Extract usage
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                    
                    # Calculate tokens per second
                    tokens_per_second = (completion_tokens / latency_ms) * 1000 if latency_ms > 0 else 0
                    
                    # Update rate limits
                    self.rate_limit.requests_this_minute += 1
                    self.rate_limit.tokens_this_minute += total_tokens
                    self.rate_limit.requests_today += 1
                    
                    # Update tracking
                    self.total_tokens += total_tokens
                    self.total_latency_ms += latency_ms
                    
                    # Update average tokens/second (running average)
                    if self.calls_made == 1:
                        self.average_tokens_per_second = tokens_per_second
                    else:
                        self.average_tokens_per_second = (
                            (self.average_tokens_per_second * (self.calls_made - 1) + tokens_per_second) 
                            / self.calls_made
                        )
                    
                    logger.info(
                        f"✅ Groq response: {completion_tokens} tokens, "
                        f"{tokens_per_second:.0f} tok/s, {latency_ms}ms, cost: ¥0"
                    )
                    
                    return content
                else:
                    raise Exception(f"Unexpected Groq response format: {result}")
                    
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error(f"❌ Groq rate limit exceeded: {e.response.text}")
                raise Exception("Groq rate limit exceeded - try again in a minute")
            else:
                logger.error(f"❌ Groq API error: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Groq API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"❌ Groq generation failed: {e}")
            raise
    
    async def _check_rate_limits(self) -> None:
        """
        Check and enforce rate limits with automatic waiting
        
        Raises:
            Exception: If daily limit exceeded
        """
        
        now = datetime.now()
        
        # Reset minute counter if needed
        if now >= self.rate_limit.minute_reset:
            self.rate_limit.requests_this_minute = 0
            self.rate_limit.tokens_this_minute = 0
            self.rate_limit.minute_reset = now + timedelta(minutes=1)
        
        # Reset day counter if needed
        if now >= self.rate_limit.day_reset:
            self.rate_limit.requests_today = 0
            self.rate_limit.day_reset = now + timedelta(days=1)
        
        # Check daily limit
        if self.rate_limit.requests_today >= self.rate_limit.requests_per_day:
            raise Exception("Groq daily limit (14,400 requests) exceeded")
        
        # Check minute limits - wait if needed
        if self.rate_limit.requests_this_minute >= self.rate_limit.requests_per_minute:
            wait_seconds = (self.rate_limit.minute_reset - now).total_seconds()
            logger.warning(f"⏳ Groq RPM limit reached, waiting {wait_seconds:.1f}s")
            await asyncio.sleep(wait_seconds + 0.5)
    
    async def health_check(self) -> bool:
        """
        Check if Groq API is accessible
        
        Returns:
            True if service is available
        """
        
        try:
            # Simple test request
            test_messages = [{"role": "user", "content": "test"}]
            await self.generate(
                messages=test_messages,
                max_tokens=5,
                temperature=0.0,
                avoid_qwen_wakeup=False  # Don't count health check
            )
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Groq health check failed: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics
        
        Returns:
            Dictionary with usage metrics
        """
        
        avg_latency = self.total_latency_ms / self.calls_made if self.calls_made > 0 else 0
        
        return {
            "model": self.model,
            "calls_made": self.calls_made,
            "total_tokens": self.total_tokens,
            "average_tokens_per_second": round(self.average_tokens_per_second, 1),
            "average_latency_ms": round(avg_latency, 1),
            "cost_total_jpy": 0.0,  # Free!
            "cost_per_call": 0.0,
            "rate_limits": {
                "requests_today": self.rate_limit.requests_today,
                "daily_limit": self.rate_limit.requests_per_day,
                "remaining_today": self.rate_limit.requests_per_day - self.rate_limit.requests_today,
                "utilization_percent": round(
                    (self.rate_limit.requests_today / self.rate_limit.requests_per_day) * 100, 1
                )
            },
            "power_saving": {
                "qwen_wakeups_avoided": self.qwen_wakeups_avoided,
                "estimated_power_saved_kwh": round(self.estimated_power_saved_kwh, 2),
                "estimated_cost_saved_jpy": round(self.qwen_wakeups_avoided * 0.5, 2),  # ¥0.5 per wakeup
                "environmental_benefit": f"~{round(self.estimated_power_saved_kwh * 0.5, 1)}kg CO2 avoided"
            },
            "performance": {
                "speed_vs_typical_api": "10-20x faster",
                "speed_vs_local_qwen": "Similar output speed, no wakeup latency",
                "speed_vs_claude": "15-30x faster"
            }
        }
    
    def get_routing_recommendation(self, task_type: str) -> Dict[str, Any]:
        """
        Get routing recommendation for different task types
        
        Args:
            task_type: Type of task (simple/medium/complex/strategic)
        
        Returns:
            Routing recommendation
        """
        
        routing_map = {
            "simple": {
                "recommended": "groq",
                "reason": "爆速・無料・Qwen起こさない",
                "alternatives": [],
                "power_saving": True
            },
            "medium": {
                "recommended": "local_qwen3",
                "reason": "コスト¥0・実装品質高い",
                "alternatives": ["groq"],  # For very simple medium tasks
                "power_saving": False
            },
            "complex": {
                "recommended": "local_qwen3",
                "reason": "大容量コンテキスト（4096）",
                "alternatives": ["cloud_qwen3_plus", "gemini3"],
                "power_saving": False
            },
            "strategic": {
                "recommended": "claude_sonnet",
                "reason": "深い洞察力・戦略判断",
                "alternatives": [],
                "power_saving": False
            }
        }
        
        return routing_map.get(task_type, routing_map["simple"])


# Singleton instance management
_groq_instance: Optional[GroqClient] = None


def get_groq_client(api_key: str) -> GroqClient:
    """
    Get or create singleton Groq client
    
    Args:
        api_key: Groq API key
    
    Returns:
        GroqClient instance
    """
    global _groq_instance
    
    if _groq_instance is None:
        _groq_instance = GroqClient(api_key=api_key)
    
    return _groq_instance
