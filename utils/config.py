"""
Bushidan Multi-Agent System v9.3.2 - Configuration Management

Enhanced configuration loading for the 4-Tier Hybrid Architecture.
Supports v9.3.2 features: Intelligent Routing, Prompt Caching, Power Optimization.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from core.system_orchestrator import SystemConfig, SystemMode
from utils.logger import get_logger


logger = get_logger(__name__)


def load_config() -> SystemConfig:
    """Load v9.3.2 system configuration from environment variables"""

    # Load .env file if it exists
    load_dotenv()

    # Required API keys
    claude_api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not claude_api_key:
        raise ValueError("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable is required")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    tavily_api_key = os.getenv("TAVILY_API_KEY", "")

    # v9.3.2: Additional API keys
    groq_api_key = os.getenv("GROQ_API_KEY")
    alibaba_api_key = os.getenv("ALIBABA_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

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

    # v9.3.2: Feature flags
    intelligent_routing = os.getenv("INTELLIGENT_ROUTING_ENABLED", "true").lower() == "true"
    prompt_caching = os.getenv("PROMPT_CACHING_ENABLED", "true").lower() == "true"
    power_optimization = os.getenv("POWER_OPTIMIZATION_ENABLED", "true").lower() == "true"

    config = SystemConfig(
        mode=mode,
        claude_api_key=claude_api_key,
        gemini_api_key=gemini_api_key,
        tavily_api_key=tavily_api_key,
        groq_api_key=groq_api_key,
        alibaba_api_key=alibaba_api_key,
        slack_token=slack_token,
        notion_token=notion_token,
        ollama_endpoint=ollama_endpoint,
        litellm_endpoint=litellm_endpoint,
        intelligent_routing_enabled=intelligent_routing,
        prompt_caching_enabled=prompt_caching,
        power_optimization_enabled=power_optimization
    )

    logger.info(f"✅ Configuration v9.3.2 loaded - Mode: {mode.value}")
    logger.info(f"  Intelligent Routing: {'✅' if intelligent_routing else '❌'}")
    logger.info(f"  Prompt Caching: {'✅' if prompt_caching else '❌'}")
    logger.info(f"  Power Optimization: {'✅' if power_optimization else '❌'}")

    return config