"""
Bushidan Multi-Agent System v9.1 - Qwen Client

Qwen2.5-Coder-32B client wrapper for Ashigaru (Execution Layer).
Connects to local Ollama via LiteLLM proxy.
"""

import asyncio
import logging
from typing import List, Dict, Any

from utils.logger import get_logger


logger = get_logger(__name__)


class QwenClient:
    """
    Qwen2.5-Coder-32B client for Ashigaru execution tasks
    
    Uses local Ollama + LiteLLM proxy for cost-free inference.
    Japanese imatrix quantization for improved Japanese language support.
    """
    
    def __init__(self, endpoint: str = "http://localhost:8000"):
        self.endpoint = endpoint
        self.model_name = "qwen2.5-coder"
        self.calls_made = 0
        
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.2
    ) -> str:
        """
        Generate response using Qwen2.5-Coder via LiteLLM proxy
        
        Optimized for code generation and implementation tasks.
        """
        
        try:
            logger.info("🏃 Using Qwen2.5-Coder (Japanese imatrix)")
            self.calls_made += 1
            
            # Use OpenAI-compatible API via LiteLLM
            import httpx
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"API call failed with status {response.status_code}")
                    
        except ImportError:
            logger.error("❌ httpx library not installed")
            return "Qwen client error - httpx not available"
        except Exception as e:
            logger.error(f"❌ Qwen API call failed: {e}")
            # Return a helpful error message instead of crashing
            return f"Qwen execution failed: {str(e)}. Please check Ollama and LiteLLM services."
    
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