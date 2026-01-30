"""
Bushidan Multi-Agent System v9.1 - Configuration Management

Simple configuration loading for the Universal Multi-LLM Framework.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from core.system_orchestrator import SystemConfig, SystemMode
from utils.logger import get_logger


logger = get_logger(__name__)


def load_config() -> SystemConfig:
    """Load system configuration from environment variables and files"""
    
    # Load .env file if it exists
    load_dotenv()
    
    # Required API keys
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if not claude_api_key:
        raise ValueError("CLAUDE_API_KEY environment variable is required")
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    
    # Optional tokens
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    notion_token = os.getenv("NOTION_TOKEN")
    
    # System mode
    mode_str = os.getenv("SYSTEM_MODE", "battalion").lower()
    try:
        mode = SystemMode(mode_str)
    except ValueError:
        logger.warning(f"Invalid system mode '{mode_str}', defaulting to battalion")
        mode = SystemMode.BATTALION
    
    # Service endpoints
    ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    litellm_endpoint = os.getenv("LITELLM_ENDPOINT", "http://localhost:8000")
    
    config = SystemConfig(
        mode=mode,
        claude_api_key=claude_api_key,
        gemini_api_key=gemini_api_key,
        tavily_api_key=tavily_api_key,
        slack_token=slack_token,
        notion_token=notion_token,
        ollama_endpoint=ollama_endpoint,
        litellm_endpoint=litellm_endpoint
    )
    
    logger.info(f"✅ Configuration loaded - Mode: {mode.value}")
    return config