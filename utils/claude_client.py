"""
Bushidan Multi-Agent System v9.1 - Claude Client

Claude API wrapper for Shogun (Strategic Layer).
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from utils.logger import get_logger


logger = get_logger(__name__)


class ClaudeClient:
    """
    Claude API client for Shogun strategic decisions
    
    Handles both Pro CLI and API calls with intelligent fallback.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pro_calls_used = 0
        self.pro_limit = 2000  # Monthly Pro limit
        
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.1
    ) -> str:
        """
        Generate response using Claude with Pro CLI priority
        
        Falls back to API if Pro limit exceeded.
        """
        
        try:
            # Try Pro CLI first if under limit
            if self.pro_calls_used < self.pro_limit:
                return await self._generate_pro_cli(messages, max_tokens)
            else:
                return await self._generate_api(messages, max_tokens, temperature)
                
        except Exception as e:
            logger.error(f"❌ Claude generation failed: {e}")
            # Fallback to API on Pro CLI failure
            if self.pro_calls_used < self.pro_limit:
                logger.info("🔄 Falling back to Claude API")
                return await self._generate_api(messages, max_tokens, temperature)
            raise
    
    async def _generate_pro_cli(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        """Generate using Claude Pro CLI"""
        
        # For now, simulate Pro CLI usage
        # In production, this would use actual claude-cli
        logger.info("🎌 Using Claude Pro CLI")
        self.pro_calls_used += 1
        
        # Simulate Pro CLI call
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Return placeholder - replace with actual claude-cli integration
        return f"Claude Pro CLI response to: {messages[0]['content'][:50]}..."
    
    async def _generate_api(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """Generate using Claude API"""
        
        logger.info("🌐 Using Claude API")
        
        try:
            # Use anthropic library for API calls
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.content[0].text
            
        except ImportError:
            logger.error("❌ anthropic library not installed")
            # Return fallback response
            return f"API fallback response to: {messages[0]['content'][:50]}..."
        except Exception as e:
            logger.error(f"❌ Claude API call failed: {e}")
            raise