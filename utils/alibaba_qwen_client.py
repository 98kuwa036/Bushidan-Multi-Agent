"""
Bushidan Multi-Agent System v9.3.2 - Alibaba Cloud Qwen3-Coder-Plus Client

Kagemusha (影武者) - Shadow backup for Taisho when local inference struggles.
Provides high-capacity context (32k) and superior reasoning for complex tasks.

Provider: Alibaba Cloud Model Studio (DashScope)
Model: qwen3-coder-plus
Context: 32768 tokens (8x local Qwen3)
Cost: ~¥3/task
"""

import asyncio
import httpx
import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class QwenCloudUsage:
    """Usage tracking for cloud Qwen calls"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_cny: float
    cost_jpy: float
    latency_ms: int
    model: str = "qwen3-coder-plus"


class AlibabaQwenClient:
    """
    Alibaba Cloud Qwen3-Coder-Plus Client
    
    Role: 侍大将影武者 (Taisho Kagemusha - Shadow Implementation Specialist)
    
    Activation triggers:
    1. Local Qwen failure (2 consecutive failures)
    2. Context overflow (>4096 tokens needed)
    3. Complex architectural tasks
    4. User explicit request
    
    Advantages over local Qwen3:
    - 32k context (vs 4k local)
    - Latest model improvements
    - Cloud reliability
    - No local resource constraints
    
    Cost: ~¥3/task (acceptable for fallback scenarios)
    """
    
    def __init__(
        self, 
        api_key: str,
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    ):
        """
        Initialize Alibaba Cloud Qwen client
        
        Args:
            api_key: Alibaba Cloud API key (DashScope)
            endpoint: API endpoint URL
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = "qwen3-coder-plus"
        
        # Usage tracking
        self.calls_made = 0
        self.total_cost_jpy = 0.0
        self.total_tokens_used = 0
        self.activation_reasons = {
            "local_failure": 0,
            "context_overflow": 0,
            "complex_task": 0,
            "explicit_request": 0
        }
        
        # Cost configuration (CNY pricing, ~¥20/CNY conversion)
        self.pricing_cny = {
            "input_per_1m": 0.50,   # ¥0.50 CNY per 1M input tokens
            "output_per_1m": 2.00   # ¥2.00 CNY per 1M output tokens
        }
        self.cny_to_jpy = 20.0  # Approximate conversion rate
        
        logger.info("🏯 Alibaba Cloud Qwen3-Coder-Plus (Kagemusha) initialized")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4000,
        temperature: float = 0.2,
        activation_reason: str = "unknown"
    ) -> str:
        """
        Generate response using Alibaba Cloud Qwen3-Coder-Plus
        
        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            activation_reason: Why this was activated (for tracking)
        
        Returns:
            Generated text content
        
        Raises:
            Exception: If API call fails
        """
        
        self.calls_made += 1
        if activation_reason in self.activation_reasons:
            self.activation_reasons[activation_reason] += 1
        
        logger.info(f"☁️ Activating Kagemusha (reason: {activation_reason})")
        
        # Prepare request payload (DashScope format)
        payload = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "result_format": "message"
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = datetime.now()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Calculate latency
                latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                
                # Parse response
                if result.get("output") and result["output"].get("choices"):
                    content = result["output"]["choices"][0]["message"]["content"]
                    
                    # Extract usage stats
                    usage = result.get("usage", {})
                    usage_stats = QwenCloudUsage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        cost_cny=self._calculate_cost_cny(usage),
                        cost_jpy=self._calculate_cost_jpy(usage),
                        latency_ms=latency_ms
                    )
                    
                    # Update tracking
                    self.total_cost_jpy += usage_stats.cost_jpy
                    self.total_tokens_used += usage_stats.total_tokens
                    
                    logger.info(
                        f"✅ Kagemusha response: {usage_stats.output_tokens} tokens, "
                        f"¥{usage_stats.cost_jpy:.2f}, {latency_ms}ms"
                    )
                    
                    return content
                else:
                    raise Exception(f"Unexpected response format: {result}")
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Alibaba Cloud API error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Cloud Qwen API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"❌ Cloud Qwen generation failed: {e}")
            raise
    
    def _calculate_cost_cny(self, usage: Dict[str, int]) -> float:
        """Calculate cost in CNY"""
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        input_cost = (input_tokens / 1_000_000) * self.pricing_cny["input_per_1m"]
        output_cost = (output_tokens / 1_000_000) * self.pricing_cny["output_per_1m"]
        
        return input_cost + output_cost
    
    def _calculate_cost_jpy(self, usage: Dict[str, int]) -> float:
        """Calculate cost in JPY"""
        return self._calculate_cost_cny(usage) * self.cny_to_jpy
    
    async def health_check(self) -> bool:
        """
        Check if Alibaba Cloud API is accessible
        
        Returns:
            True if service is available
        """
        
        try:
            # Simple test request
            test_messages = [{"role": "user", "content": "test"}]
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": {"messages": test_messages},
                "parameters": {"max_tokens": 5}
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload
                )
                
                return response.status_code == 200
                
        except Exception as e:
            logger.warning(f"⚠️ Cloud Qwen health check failed: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics
        
        Returns:
            Dictionary with usage metrics
        """
        
        return {
            "calls_made": self.calls_made,
            "total_cost_jpy": round(self.total_cost_jpy, 2),
            "total_cost_cny": round(self.total_cost_jpy / self.cny_to_jpy, 2),
            "total_tokens_used": self.total_tokens_used,
            "average_cost_per_call_jpy": round(
                self.total_cost_jpy / self.calls_made if self.calls_made > 0 else 0, 2
            ),
            "activation_reasons": self.activation_reasons,
            "model": self.model,
            "context_capacity": "32768 tokens",
            "vs_local": {
                "context": "8x larger",
                "cost": "¥3 vs ¥0",
                "reliability": "Cloud-grade"
            }
        }
    
    def should_activate(
        self,
        local_failures: int = 0,
        context_size: int = 0,
        task_complexity: str = "medium"
    ) -> tuple[bool, str]:
        """
        Determine if Kagemusha should be activated
        
        Args:
            local_failures: Number of consecutive local Qwen failures
            context_size: Required context size in tokens
            task_complexity: Task complexity level
        
        Returns:
            Tuple of (should_activate, reason)
        """
        
        # Trigger 1: Local failures
        if local_failures >= 2:
            return True, "local_failure"
        
        # Trigger 2: Context overflow
        if context_size > 4096:
            return True, "context_overflow"
        
        # Trigger 3: Complex architectural tasks
        if task_complexity in ["complex", "strategic"]:
            return True, "complex_task"
        
        return False, "not_needed"
    
    def get_context_capacity_info(self) -> Dict[str, Any]:
        """Get information about context capacity advantages"""
        
        return {
            "local_qwen3": {
                "context_length": 4096,
                "optimal_for": ["Standard implementations", "File operations"],
                "limitations": ["Large codebases", "Architecture-wide changes"]
            },
            "cloud_qwen3_plus": {
                "context_length": 32768,
                "optimal_for": [
                    "Large codebase analysis",
                    "Multi-file refactoring",
                    "Architecture design",
                    "Complex dependencies"
                ],
                "advantage": "8x larger context"
            },
            "cost_tradeoff": {
                "local": "¥0 but limited",
                "cloud": "¥3 but powerful",
                "recommendation": "Use cloud for 15% of tasks requiring large context"
            }
        }


# Singleton instance management
_client_instance: Optional[AlibabaQwenClient] = None


def get_alibaba_qwen_client(api_key: str) -> AlibabaQwenClient:
    """
    Get or create singleton Alibaba Qwen client
    
    Args:
        api_key: Alibaba Cloud API key
    
    Returns:
        AlibabaQwenClient instance
    """
    global _client_instance
    
    if _client_instance is None:
        _client_instance = AlibabaQwenClient(api_key=api_key)
    
    return _client_instance
