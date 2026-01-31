"""
Bushidan Multi-Agent System v9.3 - LiteLLM Router with Fallbacks

Layer 1 Error Handling: Infrastructure-level resilience
- Automatic retries with exponential backoff
- Fallback from Taisho (local) to Karo (cloud) on failure
- Timeout management and connection pooling
- Circuit breaker pattern for API stability
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import httpx

from utils.logger import get_logger


logger = get_logger(__name__)


class ModelTier(Enum):
    """Model tier classification"""
    TAISHO = "taisho"  # Local Qwen3-Coder (primary for heavy tasks)
    KARO = "karo"      # Cloud Gemini/Groq (fallback)
    SHOGUN = "shogun"  # Claude (strategic only)


@dataclass
class ModelConfig:
    """Configuration for a model endpoint"""
    name: str
    tier: ModelTier
    endpoint: str
    model_name: str
    timeout: int = 120
    max_retries: int = 3
    cost_per_1k_tokens: float = 0.0
    is_local: bool = False


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for API endpoints"""
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "closed"  # closed, open, half_open
    failure_threshold: int = 5
    timeout_duration: int = 60  # seconds
    

class LiteLLMRouter:
    """
    Enhanced LiteLLM Router with automatic fallbacks and self-healing
    
    Features:
    1. Automatic retry with exponential backoff
    2. Fallback chain: Taisho (local) → Karo (cloud) → Shogun (strategic)
    3. Circuit breaker pattern to avoid hammering failed endpoints
    4. Connection pooling and timeout management
    5. Usage tracking and cost monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, ModelConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.usage_stats: Dict[str, int] = {"total_calls": 0, "fallbacks": 0, "retries": 0}
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize model configurations"""
        
        # Taisho (大将) - Local Qwen3-Coder-30B-A3B
        self.models["taisho-main"] = ModelConfig(
            name="taisho-main",
            tier=ModelTier.TAISHO,
            endpoint=self.config.get("qwen_api_base", "http://localhost:11434"),
            model_name="qwen3-coder-30b-a3b",
            timeout=120,  # Local can be slower
            max_retries=3,
            cost_per_1k_tokens=0.0,  # Free local inference
            is_local=True
        )
        
        # Karo (家老) - Cloud Gemini 2.0 Flash (fallback)
        self.models["karo-gemini"] = ModelConfig(
            name="karo-gemini",
            tier=ModelTier.KARO,
            endpoint="https://generativelanguage.googleapis.com",
            model_name="gemini-2.0-flash",
            timeout=30,
            max_retries=2,
            cost_per_1k_tokens=0.075,  # ~¥130/month estimated
            is_local=False
        )
        
        # Karo (家老) - Cloud Groq (speed fallback)
        self.models["karo-groq"] = ModelConfig(
            name="karo-groq",
            tier=ModelTier.KARO,
            endpoint="https://api.groq.com/openai/v1",
            model_name="llama-3.3-70b-versatile",
            timeout=15,  # Groq is very fast
            max_retries=2,
            cost_per_1k_tokens=0.0,  # Free tier
            is_local=False
        )
        
        # Initialize circuit breakers
        for model_name in self.models:
            self.circuit_breakers[model_name] = CircuitBreakerState()
        
        logger.info(f"✅ LiteLLM Router initialized with {len(self.models)} models")
    
    async def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.2,
        fallback_chain: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute completion with automatic fallback chain
        
        Args:
            model: Primary model to use ("taisho-main", "karo-gemini", etc.)
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            fallback_chain: Optional custom fallback chain (default: taisho→karo-groq→karo-gemini)
        
        Returns:
            Response dict with content, model used, and metadata
        """
        
        self.usage_stats["total_calls"] += 1
        
        # Default fallback chain: local→fast cloud→quality cloud
        if fallback_chain is None:
            fallback_chain = ["taisho-main", "karo-groq", "karo-gemini"]
        
        # Ensure primary model is in the chain
        if model not in fallback_chain:
            fallback_chain = [model] + fallback_chain
        
        last_error = None
        
        for attempt_model in fallback_chain:
            if attempt_model not in self.models:
                logger.warning(f"⚠️ Unknown model '{attempt_model}', skipping")
                continue
            
            model_config = self.models[attempt_model]
            
            # Check circuit breaker
            if not self._check_circuit_breaker(attempt_model):
                logger.warning(f"⚠️ Circuit breaker open for {attempt_model}, trying next")
                continue
            
            try:
                # Attempt completion with retries
                result = await self._attempt_completion_with_retry(
                    model_config=model_config,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Success - reset circuit breaker
                self._reset_circuit_breaker(attempt_model)
                
                # Log fallback usage
                if attempt_model != fallback_chain[0]:
                    self.usage_stats["fallbacks"] += 1
                    logger.info(f"✅ Fallback successful: {fallback_chain[0]} → {attempt_model}")
                
                return result
                
            except Exception as e:
                last_error = e
                self._record_failure(attempt_model)
                logger.warning(f"⚠️ Model {attempt_model} failed: {e}")
                
                # Try next in fallback chain
                continue
        
        # All models failed
        logger.error(f"❌ All models in fallback chain failed. Last error: {last_error}")
        raise Exception(f"All models failed. Last error: {last_error}")
    
    async def _attempt_completion_with_retry(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Attempt completion with exponential backoff retry
        """
        
        max_retries = model_config.max_retries
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                # Execute the actual API call
                result = await self._execute_completion(
                    model_config=model_config,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                return result
                
            except Exception as e:
                self.usage_stats["retries"] += 1
                
                if attempt < max_retries - 1:
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    raise
    
    async def _execute_completion(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Execute the actual API completion call
        """
        
        logger.info(f"🔄 Calling {model_config.name} ({model_config.model_name})")
        
        # For local Ollama/LiteLLM
        if model_config.is_local:
            return await self._call_local_completion(model_config, messages, max_tokens, temperature)
        
        # For cloud APIs (Gemini, Groq)
        return await self._call_cloud_completion(model_config, messages, max_tokens, temperature)
    
    async def _call_local_completion(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Call local Ollama via LiteLLM proxy"""
        
        payload = {
            "model": model_config.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=model_config.timeout) as client:
            response = await client.post(
                f"{model_config.endpoint}/v1/chat/completions",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return {
                "content": content,
                "model": model_config.name,
                "tier": model_config.tier.value,
                "cost": 0.0,  # Local is free
                "tokens": result.get("usage", {}).get("total_tokens", 0)
            }
    
    async def _call_cloud_completion(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Call cloud APIs (Gemini, Groq)
        
        Note: This is a simplified implementation. In production, use proper API clients.
        """
        
        # For Gemini
        if "gemini" in model_config.name:
            return await self._call_gemini(model_config, messages, max_tokens, temperature)
        
        # For Groq (OpenAI-compatible)
        if "groq" in model_config.name:
            return await self._call_groq(model_config, messages, max_tokens, temperature)
        
        raise NotImplementedError(f"Cloud API not implemented for {model_config.name}")
    
    async def _call_gemini(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Call Gemini API"""
        
        # Use existing GeminiClient
        from utils.gemini_client import GeminiClient
        
        api_key = self.config.get("gemini_api_key")
        if not api_key:
            raise Exception("Gemini API key not configured")
        
        client = GeminiClient(api_key=api_key)
        
        # Convert messages to prompt
        prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        content = await client.generate(
            prompt=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "content": content,
            "model": model_config.name,
            "tier": model_config.tier.value,
            "cost": len(content.split()) * model_config.cost_per_1k_tokens / 1000,
            "tokens": len(content.split())
        }
    
    async def _call_groq(
        self,
        model_config: ModelConfig,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Call Groq API (OpenAI-compatible)"""
        
        api_key = self.config.get("groq_api_key")
        if not api_key:
            raise Exception("Groq API key not configured")
        
        payload = {
            "model": model_config.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=model_config.timeout) as client:
            response = await client.post(
                f"{model_config.endpoint}/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                raise Exception(f"Groq API error {response.status_code}: {response.text}")
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return {
                "content": content,
                "model": model_config.name,
                "tier": model_config.tier.value,
                "cost": 0.0,  # Free tier
                "tokens": result.get("usage", {}).get("total_tokens", 0)
            }
    
    def _check_circuit_breaker(self, model_name: str) -> bool:
        """
        Check if circuit breaker allows requests
        
        Returns True if requests should proceed, False if circuit is open
        """
        
        breaker = self.circuit_breakers.get(model_name)
        if not breaker:
            return True
        
        current_time = time.time()
        
        # If circuit is open, check if timeout has passed
        if breaker.state == "open":
            if current_time - breaker.last_failure_time >= breaker.timeout_duration:
                # Try half-open state
                breaker.state = "half_open"
                logger.info(f"🔄 Circuit breaker for {model_name} entering half-open state")
                return True
            else:
                # Still in timeout
                return False
        
        return True
    
    def _record_failure(self, model_name: str):
        """Record a failure and potentially open circuit breaker"""
        
        breaker = self.circuit_breakers.get(model_name)
        if not breaker:
            return
        
        breaker.failure_count += 1
        breaker.last_failure_time = time.time()
        
        # Open circuit if threshold exceeded
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = "open"
            logger.warning(f"⚠️ Circuit breaker OPENED for {model_name} after {breaker.failure_count} failures")
    
    def _reset_circuit_breaker(self, model_name: str):
        """Reset circuit breaker after successful call"""
        
        breaker = self.circuit_breakers.get(model_name)
        if not breaker:
            return
        
        if breaker.state != "closed":
            logger.info(f"✅ Circuit breaker CLOSED for {model_name}")
        
        breaker.failure_count = 0
        breaker.state = "closed"
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get router usage statistics"""
        
        return {
            "total_calls": self.usage_stats["total_calls"],
            "fallbacks_used": self.usage_stats["fallbacks"],
            "retries_attempted": self.usage_stats["retries"],
            "circuit_breakers": {
                name: {
                    "state": breaker.state,
                    "failures": breaker.failure_count
                }
                for name, breaker in self.circuit_breakers.items()
            }
        }
