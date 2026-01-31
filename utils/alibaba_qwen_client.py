"""
Bushidan Multi-Agent System v9.3.2 - Alibaba Cloud Qwen3-Coder-Plus Client

Kagemusha (影武者 - Shadow Backup) for Local Qwen3-Coder-30B.
Provides cloud-based fallback with 32k context capacity (8x local).

Key Features:
- Qwen3-Coder-Plus via Alibaba Cloud Model Studio
- 32k context window (vs 4k local)
- Same model family as local (compatible reasoning)
- Cost: ~¥3/task (acceptable for fallback)
- Activates when: Local Qwen3 cannot handle due to complexity/context

Role in v9.3.2:
- Shadow backup (Kagemusha) for local Qwen3
- Handles tasks that exceed local 4k context limit
- Maintains reasoning continuity (same model family)
- Middle fallback: Local → Cloud Qwen3 → Gemini 3
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class AlibabaQwenStats:
    """Alibaba Cloud Qwen3 usage statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    kagemusha_activations: int = 0  # Times activated as shadow
    context_overflow_triggers: int = 0  # Activated due to >4k context
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_yen: float = 0.0
    average_latency_seconds: float = 0.0


class AlibabaQwenClient:
    """
    Alibaba Cloud Qwen3-Coder-Plus Client (Kagemusha)
    
    Model: qwen3-coder-plus
    - 32k context window (8x local Qwen3's 4k)
    - Same model family as local (reasoning continuity)
    - Cloud inference (no local VRAM limit)
    - Cost: ~¥3 per task (acceptable for fallback)
    
    Role: Shadow backup (影武者) for local Qwen3
    - Seamless continuation when local hits limits
    - Maintains reasoning quality and style
    - Middle tier in 3-tier fallback chain
    """
    
    def __init__(self, api_key: str, model: str = "qwen3-coder-plus"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1"
        
        # Statistics
        self.stats = AlibabaQwenStats()
        
        # Configuration
        self.default_max_tokens = 4096
        self.default_temperature = 0.7
        self.default_top_p = 0.8
        
        # Cost tracking (estimated rates for Qwen3-Coder-Plus)
        # Alibaba Cloud pricing: ~¥0.008 per 1k tokens
        self.cost_per_1k_tokens_yen = 0.008
        
        # Context limits
        self.max_context_tokens = 32000  # 32k context
        self.local_context_limit = 4096  # Local Qwen3's limit
        
        logger.info(f"🏯 Alibaba Qwen3-Coder-Plus client initialized (Kagemusha): {self.model}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        as_kagemusha: bool = False,
        context_overflow: bool = False
    ) -> str:
        """
        Generate completion using Alibaba Cloud Qwen3-Coder-Plus
        
        Args:
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            as_kagemusha: Whether this is a Kagemusha (shadow) activation
            context_overflow: Whether activated due to context overflow
        
        Returns:
            Generated text response
        
        Raises:
            Exception: If API call fails
        """
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if temperature is None:
            temperature = self.default_temperature
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Make API request
            response_text, input_tokens, output_tokens = await self._make_request(
                messages, max_tokens, temperature
            )
            
            # Update statistics
            elapsed_time = asyncio.get_event_loop().time() - start_time
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.total_tokens_input += input_tokens
            self.stats.total_tokens_output += output_tokens
            
            # Calculate cost
            total_tokens = input_tokens + output_tokens
            cost_yen = (total_tokens / 1000) * self.cost_per_1k_tokens_yen
            self.stats.total_cost_yen += cost_yen
            
            # Update average latency
            n = self.stats.successful_requests
            self.stats.average_latency_seconds = (
                (self.stats.average_latency_seconds * (n - 1) + elapsed_time) / n
            )
            
            # Track Kagemusha activations
            if as_kagemusha:
                self.stats.kagemusha_activations += 1
                
                if context_overflow:
                    self.stats.context_overflow_triggers += 1
                    logger.info(
                        f"🏯 Kagemusha activated (Context Overflow): "
                        f"{input_tokens} → {output_tokens} tokens in {elapsed_time:.2f}s (¥{cost_yen:.2f})"
                    )
                else:
                    logger.info(
                        f"🏯 Kagemusha activated (Fallback): "
                        f"{input_tokens} → {output_tokens} tokens in {elapsed_time:.2f}s (¥{cost_yen:.2f})"
                    )
            else:
                logger.info(
                    f"☁️ Cloud Qwen3 generation: "
                    f"{output_tokens} tokens in {elapsed_time:.2f}s (¥{cost_yen:.2f})"
                )
            
            return response_text
            
        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            logger.error(f"❌ Alibaba Qwen3 generation failed: {e}")
            raise
    
    async def _make_request(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> tuple[str, int, int]:
        """
        Make actual API request to Alibaba Cloud Model Studio
        
        Args:
            messages: Conversation messages
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        
        try:
            import httpx
            
            url = f"{self.base_url}/services/aigc/text-generation/generation"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-SSE": "disable"  # Disable streaming for simplicity
            }
            
            # Format messages for Alibaba API
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Alibaba uses "system", "user", "assistant"
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
            
            payload = {
                "model": self.model,
                "input": {
                    "messages": formatted_messages
                },
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": self.default_top_p,
                    "result_format": "message"
                }
            }
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    raise Exception(f"Alibaba API error {response.status_code}: {error_detail}")
                
                result = response.json()
                
                # Check for API-level errors
                if "code" in result and result["code"] != "200":
                    raise Exception(f"Alibaba API error: {result.get('message', 'Unknown error')}")
                
                # Extract response
                output = result.get("output", {})
                response_text = output.get("text", "")
                
                if not response_text and "choices" in output:
                    # Alternative format
                    response_text = output["choices"][0]["message"]["content"]
                
                # Extract token counts
                usage = result.get("usage", {})
                input_tokens = usage.get("input_tokens", len(" ".join([m["content"] for m in messages]).split()))
                output_tokens = usage.get("output_tokens", len(response_text.split()))
                
                return response_text, input_tokens, output_tokens
                
        except Exception as e:
            logger.error(f"❌ Alibaba API request failed: {e}")
            raise
    
    def should_activate_kagemusha(self, context_size_estimate: int) -> bool:
        """
        Determine if Kagemusha should be activated based on context size
        
        Args:
            context_size_estimate: Estimated context size in tokens
        
        Returns:
            True if context exceeds local limit and Kagemusha should activate
        """
        
        if context_size_estimate > self.local_context_limit:
            logger.info(
                f"🏯 Kagemusha activation recommended: "
                f"Context {context_size_estimate} > Local limit {self.local_context_limit}"
            )
            return True
        return False
    
    async def health_check(self) -> bool:
        """
        Check if Alibaba Cloud API is available
        
        Returns:
            True if healthy, False otherwise
        """
        
        try:
            test_messages = [
                {"role": "user", "content": "Health check"}
            ]
            
            await self.generate(test_messages, max_tokens=10)
            logger.info("✅ Alibaba Qwen3 health check passed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Alibaba Qwen3 health check failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive Alibaba Qwen3 usage statistics
        
        Returns:
            Dictionary with usage metrics
        """
        
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests
        
        kagemusha_ratio = 0.0
        if self.stats.successful_requests > 0:
            kagemusha_ratio = self.stats.kagemusha_activations / self.stats.successful_requests
        
        context_overflow_ratio = 0.0
        if self.stats.kagemusha_activations > 0:
            context_overflow_ratio = self.stats.context_overflow_triggers / self.stats.kagemusha_activations
        
        return {
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": round(success_rate, 3),
            "kagemusha_activations": self.stats.kagemusha_activations,
            "kagemusha_ratio": round(kagemusha_ratio, 3),
            "context_overflow_triggers": self.stats.context_overflow_triggers,
            "context_overflow_ratio": round(context_overflow_ratio, 3),
            "total_tokens_input": self.stats.total_tokens_input,
            "total_tokens_output": self.stats.total_tokens_output,
            "total_cost_yen": round(self.stats.total_cost_yen, 2),
            "average_latency_seconds": round(self.stats.average_latency_seconds, 2),
            "model": self.model,
            "context_capacity": {
                "max_tokens": self.max_context_tokens,
                "vs_local": f"{self.max_context_tokens // self.local_context_limit}x larger"
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics"""
        self.stats = AlibabaQwenStats()
        logger.info("📊 Alibaba Qwen3 statistics reset")
