"""Enhanced Anthropic API Provider with Latest Models and Error Handling

Pro版 claude-cli が制限に達した場合に使用。
Latest Claude models (Opus 4.5 / Sonnet 4.5) with comprehensive error handling.
"""

import asyncio
import logging
import os
import time
from typing import AsyncIterator, Dict, Any, Optional

logger = logging.getLogger("shogun.provider.anthropic_api")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class AnthropicAPIProvider:
    """Enhanced Anthropic Messages API client with latest models."""

    # Latest model mapping: role -> actual API model ID (January 2025)
    # Updated to use latest Claude Opus 4.5 and Sonnet 4.5
    MODEL_MAP = {
        "opus": "claude-opus-4-5-20250514",      # Latest Opus 4.5
        "sonnet": "claude-sonnet-4-5-20250514",  # Latest Sonnet 4.5
        "haiku": "claude-3-5-haiku-20241022",    # Latest Haiku 3.5
    }

    # Cost estimates per task (JPY) - v7.0 pricing
    COST_PER_TASK = {
        "claude-opus-4-5-20250514": 24,      # ¥24/task (Strategic only)
        "claude-sonnet-4-5-20250514": 5,     # ¥5/task (Complex tasks)
        "claude-3-5-haiku-20241022": 0.5,    # ¥0.5/task
    }

    # Cost estimates per 1K tokens (JPY) for detailed tracking
    COST_ESTIMATES = {
        "claude-opus-4-5-20250514": {"input": 0.015, "output": 0.075},
        "claude-sonnet-4-5-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    }

    def __init__(self, api_key: str | None = None):
        if not HAS_ANTHROPIC:
            raise RuntimeError("pip install anthropic required for API fallback")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        # Enhanced tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_yen": 0.0,
            "requests_by_model": {},
            "errors": 0,
            "last_request_time": 0,
        }

    async def generate(
        self,
        prompt: str,
        model: str = "sonnet",
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Enhanced single-turn generation with error handling."""
        model_id = self.MODEL_MAP.get(model, model)
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        logger.info("[API] model=%s (%s), prompt_len=%d", model, model_id, len(prompt))
        
        try:
            # Apply rate limiting
            await self._apply_rate_limiting()
            
            # Make API call with timing
            start_time = time.time()
            response = await self._client.messages.create(**kwargs)
            elapsed = time.time() - start_time
            
            # Extract text content
            text = ""
            for content in response.content:
                if hasattr(content, 'text'):
                    text += content.text
                    
            # Update statistics
            await self._update_usage_stats(model_id, response, elapsed)
            
            logger.info("[API] ✅ response_len=%d, time=%.2fs", len(text), elapsed)
            return text
            
        except anthropic.RateLimitError as e:
            self.usage_stats["errors"] += 1
            logger.warning("[API] ❌ Rate limit: %s", e)
            raise
            
        except anthropic.AuthenticationError as e:
            self.usage_stats["errors"] += 1
            logger.error("[API] ❌ Auth error: %s", e)
            raise
            
        except Exception as e:
            self.usage_stats["errors"] += 1
            logger.error("[API] ❌ Unexpected error: %s", e)
            raise

    async def chat(
        self,
        messages: list[dict],
        model: str = "sonnet",
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Multi-turn chat."""
        model_id = self.MODEL_MAP.get(model, model)
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def generate_stream(
        self,
        prompt: str,
        model: str = "sonnet",
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Streaming generation."""
        model_id = self.MODEL_MAP.get(model, model)
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def _apply_rate_limiting(self):
        """Apply basic rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.usage_stats["last_request_time"]
        
        # Minimum 0.5 seconds between requests
        if time_since_last < 0.5:
            wait_time = 0.5 - time_since_last
            await asyncio.sleep(wait_time)
        
        self.usage_stats["last_request_time"] = time.time()
    
    async def _update_usage_stats(self, model_id: str, response: Any, elapsed: float):
        """Update usage statistics with cost estimation."""
        self.usage_stats["total_requests"] += 1
        
        # Track by model
        model_stats = self.usage_stats["requests_by_model"]
        if model_id not in model_stats:
            model_stats[model_id] = {"requests": 0, "tokens": 0, "cost_yen": 0.0}
        
        model_stats[model_id]["requests"] += 1
        
        # Track token usage and cost if available
        if hasattr(response, 'usage'):
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
            total_tokens = input_tokens + output_tokens
            
            self.usage_stats["total_tokens"] += total_tokens
            model_stats[model_id]["tokens"] += total_tokens
            
            # Calculate cost
            if model_id in self.COST_ESTIMATES:
                rates = self.COST_ESTIMATES[model_id]
                cost = (input_tokens / 1000 * rates["input"] + 
                       output_tokens / 1000 * rates["output"])
                self.usage_stats["total_cost_yen"] += cost
                model_stats[model_id]["cost_yen"] += cost
                
                logger.debug(
                    "[API] Tokens: %d+%d=%d, Cost: ¥%.4f", 
                    input_tokens, output_tokens, total_tokens, cost
                )
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        return dict(self.usage_stats)
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_yen": 0.0,
            "requests_by_model": {},
            "errors": 0,
            "last_request_time": 0,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.generate(
                prompt="Health check - respond with 'OK'",
                model="sonnet",
                max_tokens=10,
            )
            return {
                "status": "healthy",
                "response": response,
                "model": self.MODEL_MAP["sonnet"],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self._client, "_client"):
            await self._client._client.aclose()
        logger.info("[API] Connection closed")
