"""
Bushidan Multi-Agent System v9.3.2 - Groq Client

Lightning-fast LLM inference via Groq Cloud API.
Uses Llama 3.3 70B Versatile for instant responses.

Key Features:
- 300-500 tok/s throughput (10-20x faster than standard APIs)
- Free tier: 14,400 requests/day (600/hour)
- Power-saving optimization (don't wake local Qwen for simple tasks)
- Automatic rate limit management
- Cost: ¥0 (free tier)

Usage Context (運用黄金律):
- Simple tasks only (questions, lookups, simple queries)
- Falls back to Gemini 3 if unavailable
- Saves ~¥200/month in electricity (vs waking Qwen)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class GroqUsageStats:
    """Groq API usage statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_count: int = 0
    total_tokens: int = 0
    average_tokens_per_second: float = 0.0
    power_savings_yen: float = 0.0


class GroqClient:
    """
    Groq Cloud API Client for lightning-fast inference
    
    Model: Llama 3.3 70B Versatile
    - 128k context window
    - 300-500 tok/s throughput
    - Excellent reasoning for simple tasks
    - Free tier: 14,400 req/day
    
    Rate Limits (Free tier):
    - 600 requests per hour
    - 30 requests per minute
    - 1 request per second (burst)
    """
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Statistics
        self.stats = GroqUsageStats()
        
        # Rate limiting
        self.request_timestamps: List[datetime] = []
        self.rate_limit_window = timedelta(minutes=1)
        self.max_requests_per_minute = 30  # Conservative (actual: 30)
        
        # Configuration
        self.default_max_tokens = 1000
        self.default_temperature = 0.7
        
        logger.info(f"⚡ Groq client initialized: {self.model}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate completion using Groq API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            stream: Whether to stream response (not implemented)
        
        Returns:
            Generated text response
        
        Raises:
            Exception: If API call fails
        """
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if temperature is None:
            temperature = self.default_temperature
        
        # Rate limit check
        await self._rate_limit_check()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Make API request
            response_text = await self._make_request(messages, max_tokens, temperature)
            
            # Update statistics
            elapsed_time = asyncio.get_event_loop().time() - start_time
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            
            # Estimate tokens (rough)
            estimated_tokens = len(response_text.split())
            self.stats.total_tokens += estimated_tokens
            
            # Calculate tokens/second
            if elapsed_time > 0:
                tok_per_sec = estimated_tokens / elapsed_time
                self.stats.average_tokens_per_second = (
                    (self.stats.average_tokens_per_second * (self.stats.successful_requests - 1) + tok_per_sec)
                    / self.stats.successful_requests
                )
            
            # Power savings (avoiding Qwen wake-up)
            self.stats.power_savings_yen += 5.0  # ~¥5 saved per simple task
            
            logger.info(
                f"⚡ Groq generation: {estimated_tokens} tokens in {elapsed_time:.2f}s "
                f"({tok_per_sec:.0f} tok/s)"
            )
            
            return response_text
            
        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            
            if "rate" in str(e).lower() or "429" in str(e):
                self.stats.rate_limited_count += 1
                logger.warning(f"⚠️ Groq rate limited: {e}")
            else:
                logger.error(f"❌ Groq generation failed: {e}")
            
            raise
    
    async def _make_request(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        """
        Make actual API request to Groq
        
        Args:
            messages: Conversation messages
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Response text
        """
        
        try:
            import httpx
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    raise Exception(f"Groq API error {response.status_code}: {error_detail}")
                
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                return response_text
                
        except Exception as e:
            logger.error(f"❌ Groq API request failed: {e}")
            raise
    
    async def _rate_limit_check(self) -> None:
        """
        Check and enforce rate limits
        
        Free tier limits:
        - 30 requests per minute
        - Implement sliding window
        """
        
        now = datetime.now()
        
        # Remove timestamps outside the window
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if now - ts < self.rate_limit_window
        ]
        
        # Check if at limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            # Calculate wait time
            oldest = self.request_timestamps[0]
            wait_time = (oldest + self.rate_limit_window - now).total_seconds()
            
            if wait_time > 0:
                logger.warning(f"⏱️ Groq rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time + 0.1)  # Small buffer
        
        # Record this request
        self.request_timestamps.append(now)
    
    async def health_check(self) -> bool:
        """
        Check if Groq API is available and responsive
        
        Returns:
            True if healthy, False otherwise
        """
        
        try:
            test_messages = [
                {"role": "user", "content": "Hi"}
            ]
            
            await self.generate(test_messages, max_tokens=10)
            logger.info("✅ Groq health check passed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Groq health check failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive Groq usage statistics
        
        Returns:
            Dictionary with usage metrics
        """
        
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests
        
        return {
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": round(success_rate, 3),
            "rate_limited_count": self.stats.rate_limited_count,
            "total_tokens": self.stats.total_tokens,
            "average_tokens_per_second": round(self.stats.average_tokens_per_second, 1),
            "power_savings_yen": round(self.stats.power_savings_yen, 2),
            "model": self.model
        }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics"""
        self.stats = GroqUsageStats()
        logger.info("📊 Groq statistics reset")
