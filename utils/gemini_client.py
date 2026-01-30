"""
Bushidan Multi-Agent System v9.1 - Gemini Client

Gemini 2.0 Flash API wrapper for Karo (Tactical Layer).
"""

import asyncio
import logging
from typing import Optional

from utils.logger import get_logger


logger = get_logger(__name__)


class GeminiClient:
    """
    Gemini 2.0 Flash API client for Karo tactical coordination
    
    Optimized for speed, cost-efficiency, and Japanese language support.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.calls_made = 0
        
    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1000,
        temperature: float = 0.3
    ) -> str:
        """
        Generate response using Gemini 2.0 Flash
        
        Optimized for tactical coordination tasks.
        """
        
        try:
            logger.info("🏛️ Using Gemini 2.0 Flash API")
            self.calls_made += 1
            
            # Use google-generativeai library
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
            )
            
            return response.text
            
        except ImportError:
            logger.error("❌ google-generativeai library not installed")
            # Return fallback response
            return f"Gemini fallback response to: {prompt[:50]}..."
        except Exception as e:
            logger.error(f"❌ Gemini API call failed: {e}")
            # Return error response that won't break the system
            return f"Gemini API error - task delegation failed: {str(e)}"
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics"""
        return {
            "calls_made": self.calls_made,
            "estimated_cost": self.calls_made * 0.054,  # Rough estimate in JPY
            "model": "gemini-2.0-flash"
        }