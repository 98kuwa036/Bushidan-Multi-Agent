"""
Bushidan Multi-Agent System v9.3.2 - Claude Client with Prompt Caching

Enhanced Claude client with Prompt Caching support for 90% cost reduction
on repeated contexts.

Key Features:
- Prompt Caching (90% cost reduction on cache hits)
- 5-minute TTL (Time To Live) management
- Cache hit/miss tracking
- Automatic fallback to Pro CLI
- claude-sonnet-4-5-20250929 model

Prompt Caching Benefits:
- Input tokens (cached): 90% cheaper
- Output tokens: Same price
- Perfect for repeated system prompts, long contexts
- Typical savings: ¥140 → ¥14 per month

Usage:
- System prompts and contexts are automatically cached
- Cache lasts 5 minutes
- Reusing within 5min = 90% savings on input
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Prompt Caching statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_without_cache_yen: float = 0.0
    cost_with_cache_yen: float = 0.0
    savings_yen: float = 0.0


class ClaudeClientCached:
    """
    Claude API client with Prompt Caching for v9.3.2 Shogun
    
    Model: claude-sonnet-4-5-20250929
    - Latest Sonnet 4.5 with caching support
    - Pro CLI: 2,000 requests/month (free)
    - API: Fallback with prompt caching
    
    Caching Strategy:
    - System prompts cached (repeated across tasks)
    - Long contexts cached (project info, past decisions)
    - Cache TTL: 5 minutes
    - 90% cost reduction on cache hits
    """
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        self.api_key = api_key
        self.model = model
        
        # Pro CLI management
        self.pro_calls_used = 0
        self.pro_limit = 2000  # Monthly Pro limit
        
        # Cache management
        self.cache_stats = CacheStats()
        self.cache_ttl = timedelta(minutes=5)
        self.last_cached_context: Optional[str] = None
        self.last_cache_time: Optional[datetime] = None
        
        # Cost tracking (Sonnet 4.5 rates)
        self.cost_per_1k_input_yen = 0.003  # $3/MTok → ¥0.003/1k
        self.cost_per_1k_output_yen = 0.015  # $15/MTok → ¥0.015/1k
        self.cost_per_1k_cached_input_yen = 0.0003  # 90% cheaper
        
        logger.info(f"🎌 Claude client initialized with Prompt Caching: {self.model}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Generate response using Claude with Pro CLI priority and Prompt Caching
        
        Args:
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt (will be cached)
            use_cache: Whether to use prompt caching
        
        Returns:
            Generated text response
        
        Falls back: Pro CLI (free) → API with cache → API without cache
        """
        
        try:
            # Try Pro CLI first if under limit
            if self.pro_calls_used < self.pro_limit:
                return await self._generate_pro_cli(messages, max_tokens, system_prompt)
            else:
                # Use API with caching
                return await self._generate_api_cached(
                    messages, max_tokens, temperature, system_prompt, use_cache
                )
                
        except Exception as e:
            logger.error(f"❌ Claude generation failed: {e}")
            
            # Fallback to API on Pro CLI failure
            if self.pro_calls_used < self.pro_limit:
                logger.info("🔄 Falling back to Claude API with caching")
                return await self._generate_api_cached(
                    messages, max_tokens, temperature, system_prompt, use_cache
                )
            raise
    
    async def _generate_pro_cli(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate using Claude Pro CLI
        
        Note: Pro CLI doesn't support prompt caching, but it's free (2000/month)
        """
        
        logger.info("🎌 Using Claude Pro CLI (free tier)")
        self.pro_calls_used += 1
        self.cache_stats.total_requests += 1
        
        try:
            from providers.claude_cli import call_claude_cli
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages, system_prompt)
            
            # Call Pro CLI
            result = await call_claude_cli(
                prompt=prompt,
                model="sonnet",
                max_tokens=max_tokens
            )
            
            return result.output
            
        except ImportError:
            logger.warning("⚠️ Claude CLI not available, falling back to API")
            return await self._generate_api_cached(
                messages, max_tokens, 0.1, system_prompt, use_cache=True
            )
        except Exception as e:
            logger.error(f"❌ Pro CLI failed: {e}")
            raise
    
    async def _generate_api_cached(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Generate using Claude API with Prompt Caching
        
        Caching strategy:
        - System prompt is cached (marked with cache_control)
        - Cache lasts 5 minutes
        - 90% cost reduction on cached input tokens
        """
        
        logger.info("🌐 Using Claude API with Prompt Caching")
        self.cache_stats.total_requests += 1
        
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            # Prepare messages with cache control
            api_messages = messages.copy()
            
            # Check if we should use cache
            is_cache_hit = False
            if use_cache and system_prompt:
                # Check if context is still fresh (< 5 min)
                if (self.last_cached_context == system_prompt and
                    self.last_cache_time and
                    datetime.now() - self.last_cache_time < self.cache_ttl):
                    is_cache_hit = True
                    self.cache_stats.cache_hits += 1
                    logger.info("✅ Prompt cache HIT (90% savings)")
                else:
                    self.cache_stats.cache_misses += 1
                    logger.info("❌ Prompt cache MISS (creating new cache)")
                    
                    # Update cache tracking
                    self.last_cached_context = system_prompt
                    self.last_cache_time = datetime.now()
                
                # Mark system prompt for caching
                system_messages = [{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # Enable caching
                }]
            else:
                system_messages = None
                if system_prompt:
                    # No caching, just use as regular system prompt
                    system_messages = [{"type": "text", "text": system_prompt}]
            
            # Make API request
            response = await client.messages.create(
                model=self.model,
                system=system_messages,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Track usage and costs
            usage = response.usage
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)
            cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)
            
            self.cache_stats.total_input_tokens += input_tokens
            self.cache_stats.total_output_tokens += output_tokens
            self.cache_stats.cached_input_tokens += cache_read_tokens
            
            # Calculate costs
            cost_without_cache = (
                (input_tokens / 1000) * self.cost_per_1k_input_yen +
                (output_tokens / 1000) * self.cost_per_1k_output_yen
            )
            
            cost_with_cache = (
                ((input_tokens - cache_read_tokens) / 1000) * self.cost_per_1k_input_yen +
                (cache_read_tokens / 1000) * self.cost_per_1k_cached_input_yen +
                (output_tokens / 1000) * self.cost_per_1k_output_yen
            )
            
            self.cache_stats.cost_without_cache_yen += cost_without_cache
            self.cache_stats.cost_with_cache_yen += cost_with_cache
            self.cache_stats.savings_yen = (
                self.cache_stats.cost_without_cache_yen -
                self.cache_stats.cost_with_cache_yen
            )
            
            savings_pct = 0
            if cost_without_cache > 0:
                savings_pct = (1 - cost_with_cache / cost_without_cache) * 100
            
            logger.info(
                f"💰 Cost: ¥{cost_with_cache:.4f} (saved {savings_pct:.0f}% with cache)"
            )
            
            return response.content[0].text
            
        except ImportError:
            logger.error("❌ anthropic library not installed")
            raise
        except Exception as e:
            logger.error(f"❌ Claude API call failed: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Convert messages format to single prompt string"""
        
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        
        return "\n\n".join(parts)
    
    async def health_check(self) -> bool:
        """
        Check if Claude API is available
        
        Returns:
            True if healthy, False otherwise
        """
        
        try:
            test_messages = [
                {"role": "user", "content": "Health check"}
            ]
            
            await self.generate(test_messages, max_tokens=10, use_cache=False)
            logger.info("✅ Claude health check passed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Claude health check failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive usage and caching statistics
        
        Returns:
            Dictionary with usage and cost metrics
        """
        
        cache_hit_rate = 0.0
        if self.cache_stats.total_requests > 0:
            cache_hit_rate = self.cache_stats.cache_hits / self.cache_stats.total_requests
        
        savings_pct = 0.0
        if self.cache_stats.cost_without_cache_yen > 0:
            savings_pct = (
                self.cache_stats.savings_yen / self.cache_stats.cost_without_cache_yen
            ) * 100
        
        return {
            "pro_cli_usage": {
                "calls_used": self.pro_calls_used,
                "calls_remaining": self.pro_limit - self.pro_calls_used,
                "limit": self.pro_limit
            },
            "api_usage": {
                "total_requests": self.cache_stats.total_requests,
                "total_input_tokens": self.cache_stats.total_input_tokens,
                "total_output_tokens": self.cache_stats.total_output_tokens
            },
            "caching": {
                "cache_hits": self.cache_stats.cache_hits,
                "cache_misses": self.cache_stats.cache_misses,
                "cache_hit_rate": round(cache_hit_rate, 3),
                "cached_input_tokens": self.cache_stats.cached_input_tokens,
                "cache_ttl_minutes": self.cache_ttl.total_seconds() / 60
            },
            "costs": {
                "cost_without_cache_yen": round(self.cache_stats.cost_without_cache_yen, 2),
                "cost_with_cache_yen": round(self.cache_stats.cost_with_cache_yen, 2),
                "savings_yen": round(self.cache_stats.savings_yen, 2),
                "savings_percentage": round(savings_pct, 1)
            },
            "model": self.model
        }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics"""
        self.cache_stats = CacheStats()
        logger.info("📊 Claude statistics reset")
