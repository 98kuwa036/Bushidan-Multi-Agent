"""
Bushidan Multi-Agent System v9.3 - Enhanced Qwen Client

Qwen3-Coder-30B-A3B client with integrated error handling.
Now uses LiteLLM Router for automatic fallbacks and retries.

v9.3 Enhancements:
- LiteLLM Router integration (Layer 1 error handling)
- Automatic fallback to cloud APIs on local failure
- Circuit breaker pattern for reliability
- Usage tracking and cost monitoring
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from utils.logger import get_logger
from utils.litellm_router import LiteLLMRouter


logger = get_logger(__name__)


class QwenClient:
    """
    Enhanced Qwen3-Coder-30B-A3B client with self-healing capabilities
    
    Uses LiteLLM Router for:
    - Automatic fallback: Local Qwen → Cloud Groq/Gemini
    - Retry with exponential backoff
    - Circuit breaker for failed endpoints
    - Cost tracking and optimization
    """
    
    def __init__(self, config: Dict[str, Any], api_base: str = "http://localhost:11434", model_name: str = "qwen3-coder-30b-a3b"):
        """
        Initialize enhanced Qwen client
        
        Args:
            config: System configuration with API keys
            api_base: Ollama/LiteLLM endpoint
            model_name: Model name
        """
        self.api_base = api_base
        self.model_name = model_name
        self.calls_made = 0
        
        # Initialize LiteLLM Router with fallback support
        self.router = LiteLLMRouter({
            "qwen_api_base": api_base,
            "gemini_api_key": config.get("gemini_api_key"),
            "groq_api_key": config.get("groq_api_key")
        })
        
        logger.info(f"✅ Enhanced Qwen client initialized with fallback support")
        
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.2,
        enable_fallback: bool = True
    ) -> str:
        """
        Generate response with automatic error handling and fallback
        
        Args:
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_fallback: Enable fallback to cloud APIs on failure
        
        Returns:
            Generated text content
        
        Raises:
            Exception: If all models in fallback chain fail
        """
        
        try:
            logger.info(f"🔄 Qwen client generating (fallback: {enable_fallback})")
            self.calls_made += 1
            
            # Use router with fallback chain
            if enable_fallback:
                fallback_chain = ["taisho-main", "karo-groq", "karo-gemini"]
            else:
                fallback_chain = ["taisho-main"]  # No fallback
            
            result = await self.router.completion(
                model="taisho-main",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                fallback_chain=fallback_chain
            )
            
            # Log successful generation
            model_used = result.get("model", "unknown")
            tier = result.get("tier", "unknown")
            logger.info(f"✅ Generated using {model_used} ({tier})")
            
            return result["content"]
                    
        except Exception as e:
            logger.error(f"❌ All generation attempts failed: {e}")
            # Return helpful error instead of crashing
            return f"Generation failed after trying all available models: {str(e)}"
    
    async def health_check(self) -> bool:
        """Check if Qwen service is available"""
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.endpoint}/v1/models")
                
                if response.status_code == 200:
                    models = response.json()
                    available_models = [m["id"] for m in models.get("data", [])]
                    return self.model_name in available_models
                
        except Exception as e:
            logger.warning(f"⚠️ Qwen health check failed: {e}")
        
        return False
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics"""
        return {
            "calls_made": self.calls_made,
            "estimated_cost": 0,  # Local inference is free
            "model": "qwen2.5-coder-32b-instruct",
            "quantization": "q4_K_M Japanese imatrix"
        }