"""
Bushidan Multi-Agent System v9.3.2 - Claude Client with Prompt Caching

NEW v9.3.2: Prompt Caching enabled for 90% cost reduction

Prompt Caching benefits:
- 90% cost reduction for repeated contexts
- Faster response times
- Efficient system prompt reuse
- TTL: 5 minutes (configurable)

Use cases:
- Repeated system prompts across sessions
- Large codebase context reuse
- Multi-turn conversations with same context
- Iterative refinement tasks

Cost comparison:
- Without caching: $3/MTok input + $15/MTok output
- With caching: $0.30/MTok cached + $3/MTok input + $15/MTok output
- Savings: 90% on repeated context portions
"""

import asyncio
import anthropic
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Prompt caching statistics"""
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_saved: int = 0
    cost_saved_jpy: float = 0.0
    cache_creation_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0


@dataclass
class ClaudeCachedUsage:
    """Usage tracking with cache info"""
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    total_tokens: int
    cost_jpy: float
    cost_saved_jpy: float
    latency_ms: int
    cache_hit: bool


class ClaudeClientCached:
    """
    Enhanced Claude client with Prompt Caching support
    
    v9.3.2 Features:
    - Prompt caching for 90% cost reduction
    - Automatic cache management (5-minute TTL)
    - Cache hit/miss tracking
    - Savings calculation
    
    Caching strategy:
    - System prompts: Always cached
    - Large contexts (>1024 tokens): Cached
    - Frequently used prompts: Cached
    - Short/unique prompts: Not cached
    """
    
    def __init__(
        self,
        api_key: str,
        pro_limit: int = 2000,
        enable_caching: bool = True,
        cache_ttl_minutes: int = 5
    ):
        """
        Initialize Claude client with caching
        
        Args:
            api_key: Anthropic API key
            pro_limit: Monthly Pro CLI limit
            enable_caching: Enable prompt caching
            cache_ttl_minutes: Cache TTL in minutes
        """
        self.api_key = api_key
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Pro tracking
        self.pro_limit = pro_limit
        self.pro_calls_used = 0
        
        # Caching configuration
        self.enable_caching = enable_caching
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # Statistics
        self.cache_stats = CacheStats()
        self.calls_made = 0
        self.total_cost_jpy = 0.0
        
        # Cache management
        self.cached_prompts: Dict[str, datetime] = {}  # prompt_hash -> expiry_time
        
        logger.info(
            f"🎌 Claude client initialized (Prompt Caching: "
            f"{'enabled' if enable_caching else 'disabled'}, TTL: {cache_ttl_minutes}min)"
        )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Generate response with automatic caching
        
        Args:
            messages: Chat messages
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            system_prompt: System prompt (will be cached if enabled)
            use_cache: Whether to use caching for this request
        
        Returns:
            Generated text content
        """
        
        self.calls_made += 1
        
        # Decide: Pro CLI or API
        if self.pro_calls_used < self.pro_limit:
            # Use Pro CLI (no caching available in CLI)
            return await self._generate_pro_cli(messages, max_tokens)
        else:
            # Use API with caching
            return await self._generate_api_cached(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                use_cache=use_cache
            )
    
    async def _generate_pro_cli(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int
    ) -> str:
        """Generate using Pro CLI (no caching)"""
        
        self.pro_calls_used += 1
        logger.info(f"🎌 Using Pro CLI ({self.pro_calls_used}/{self.pro_limit})")
        
        # Import CLI provider
        from providers.claude_cli import call_claude_cli
        
        # Extract user message (simplified)
        user_message = messages[-1].get("content", "") if messages else ""
        
        result = await call_claude_cli(
            prompt=user_message,
            model="sonnet"
        )
        
        if result.success:
            return result.output
        else:
            # Fallback to API
            logger.warning("⚠️ Pro CLI failed, falling back to API")
            return await self._generate_api_cached(messages, max_tokens)
    
    async def _generate_api_cached(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Generate using API with prompt caching
        
        Implements Anthropic's prompt caching:
        https://docs.anthropic.com/claude/docs/prompt-caching
        """
        
        start_time = datetime.now()
        
        # Prepare messages with cache control
        api_messages = messages.copy()
        
        # Add cache control to system prompt if provided and caching enabled
        if system_prompt and self.enable_caching and use_cache:
            # System prompt with cache_control
            system_with_cache = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
            system_param = system_with_cache
        elif system_prompt:
            system_param = system_prompt
        else:
            system_param = None
        
        # Check if we can mark recent messages for caching
        # Cache control can be applied to the last user message if >1024 tokens
        if self.enable_caching and use_cache and len(api_messages) > 0:
            last_msg = api_messages[-1]
            if len(last_msg.get("content", "")) > 4000:  # ~1024 tokens
                # Wrap content with cache control
                api_messages[-1] = {
                    "role": last_msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": last_msg["content"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                }
        
        try:
            # Call Claude API with caching support
            response = await self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_param if system_param else None,
                messages=api_messages
            )
            
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Extract content
            content = response.content[0].text if response.content else ""
            
            # Parse usage with cache info
            usage = response.usage
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)
            cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)
            
            # Calculate costs (JPY, approximate)
            # Standard: $3/MTok input, $15/MTok output
            # Cached: $0.30/MTok cached read, $3.75/MTok cache write
            cost_input = (input_tokens / 1_000_000) * 3.0 * 150  # $3 * ¥150
            cost_output = (output_tokens / 1_000_000) * 15.0 * 150  # $15 * ¥150
            cost_cache_creation = (cache_creation_tokens / 1_000_000) * 3.75 * 150
            cost_cache_read = (cache_read_tokens / 1_000_000) * 0.30 * 150
            
            total_cost = cost_input + cost_output + cost_cache_creation + cost_cache_read
            
            # Calculate savings (vs non-cached)
            if cache_read_tokens > 0:
                # Saved: (cache_read_tokens * $3) - (cache_read_tokens * $0.30)
                cost_saved = (cache_read_tokens / 1_000_000) * (3.0 - 0.30) * 150
                cache_hit = True
                self.cache_stats.cache_hits += 1
                self.cache_stats.tokens_saved += cache_read_tokens
                self.cache_stats.cost_saved_jpy += cost_saved
            else:
                cost_saved = 0.0
                cache_hit = False
                if cache_creation_tokens > 0:
                    self.cache_stats.cache_misses += 1
                    self.cache_stats.cache_creation_count += 1
            
            self.total_cost_jpy += total_cost
            
            logger.info(
                f"✅ Claude API: {output_tokens} tokens, ¥{total_cost:.2f}, "
                f"{latency_ms}ms, cache: {'HIT' if cache_hit else 'MISS'}"
                f"{f', saved ¥{cost_saved:.2f}' if cost_saved > 0 else ''}"
            )
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Claude API call failed: {e}")
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        
        return {
            "calls_made": self.calls_made,
            "pro_calls_used": self.pro_calls_used,
            "pro_limit": self.pro_limit,
            "pro_remaining": self.pro_limit - self.pro_calls_used,
            "total_cost_jpy": round(self.total_cost_jpy, 2),
            "caching_enabled": self.enable_caching,
            "cache_stats": {
                "hit_rate_percent": round(self.cache_stats.hit_rate, 1),
                "cache_hits": self.cache_stats.cache_hits,
                "cache_misses": self.cache_stats.cache_misses,
                "cache_creations": self.cache_stats.cache_creation_count,
                "tokens_saved": self.cache_stats.tokens_saved,
                "cost_saved_jpy": round(self.cache_stats.cost_saved_jpy, 2),
                "average_savings_per_hit": round(
                    self.cache_stats.cost_saved_jpy / self.cache_stats.cache_hits
                    if self.cache_stats.cache_hits > 0 else 0, 2
                )
            },
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get usage recommendations based on stats"""
        
        recommendations = []
        
        if self.cache_stats.hit_rate < 30 and self.cache_stats.cache_hits + self.cache_stats.cache_misses > 10:
            recommendations.append(
                "Low cache hit rate (<30%). Consider using longer system prompts or "
                "more consistent context across requests."
            )
        
        if self.cache_stats.hit_rate > 70:
            recommendations.append(
                f"Excellent cache hit rate ({self.cache_stats.hit_rate:.1f}%)! "
                f"Saved ¥{self.cache_stats.cost_saved_jpy:.2f} so far."
            )
        
        if self.pro_calls_used >= self.pro_limit:
            recommendations.append(
                "Pro limit reached. All requests now use API with caching for cost optimization."
            )
        
        return recommendations


# Singleton instance management
_cached_client_instance: Optional[ClaudeClientCached] = None


def get_claude_client_cached(
    api_key: str,
    enable_caching: bool = True,
    cache_ttl_minutes: int = 5
) -> ClaudeClientCached:
    """
    Get or create singleton Claude client with caching
    
    Args:
        api_key: Anthropic API key
        enable_caching: Enable prompt caching
        cache_ttl_minutes: Cache TTL in minutes
    
    Returns:
        ClaudeClientCached instance
    """
    global _cached_client_instance
    
    if _cached_client_instance is None:
        _cached_client_instance = ClaudeClientCached(
            api_key=api_key,
            enable_caching=enable_caching,
            cache_ttl_minutes=cache_ttl_minutes
        )
    
    return _cached_client_instance
