"""
Bushidan Multi-Agent System v11.5 - Configuration Management

v11.5 9層ハイブリッドアーキテクチャ対応
除外: Alibaba/Qwen/Kimi (中国企業)
採用: Anthropic/OpenAI/Mistral/xAI/Google/Meta/NVIDIA (西側企業のみ)
"""

import os
from dotenv import load_dotenv

from core.system_orchestrator import SystemConfig, SystemMode
from utils.logger import get_logger


logger = get_logger(__name__)


def load_config() -> SystemConfig:
    """Load v11.4 system configuration from environment variables"""

    load_dotenv()

    # Required API keys
    claude_api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not claude_api_key:
        raise ValueError("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable is required")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    tavily_api_key = os.getenv("TAVILY_API_KEY", "")

    # v11.5: API keys (西側企業のみ)
    groq_api_key = os.getenv("GROQ_API_KEY")                   # 家老-B Llama 3.3
    openai_api_key = os.getenv("OPENAI_API_KEY")               # 軍師 o3-mini
    mistral_api_key = os.getenv("MISTRAL_API_KEY")             # 参謀-A Mistral Large 3
    xai_api_key = os.getenv("XAI_API_KEY")                     # 参謀-B Grok 4.1 Fast

    # Optional tokens
    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    notion_token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY")

    # System mode
    mode_str = os.getenv("SYSTEM_MODE", "battalion").lower()
    try:
        mode = SystemMode(mode_str)
    except ValueError:
        logger.warning(f"Invalid system mode '{mode_str}', defaulting to battalion")
        mode = SystemMode.BATTALION

    # Service endpoints
    litellm_endpoint = os.getenv("LITELLM_ENDPOINT", "http://localhost:8000")
    llamacpp_endpoint = os.getenv("LLAMACPP_ENDPOINT", "http://192.168.11.239:8080")

    # Feature flags
    intelligent_routing = os.getenv("INTELLIGENT_ROUTING_ENABLED", "true").lower() == "true"
    prompt_caching = os.getenv("PROMPT_CACHING_ENABLED", "true").lower() == "true"
    power_optimization = os.getenv("POWER_OPTIMIZATION_ENABLED", "true").lower() == "true"

    config = SystemConfig(
        mode=mode,
        claude_api_key=claude_api_key,
        gemini_api_key=gemini_api_key,
        tavily_api_key=tavily_api_key,
        groq_api_key=groq_api_key,
        openai_api_key=openai_api_key,
        mistral_api_key=mistral_api_key,
        xai_api_key=xai_api_key,
        discord_token=discord_token,
        notion_token=notion_token,
        litellm_endpoint=litellm_endpoint,
        llamacpp_endpoint=llamacpp_endpoint,
        intelligent_routing_enabled=intelligent_routing,
        prompt_caching_enabled=prompt_caching,
        power_optimization_enabled=power_optimization,
    )

    logger.info(f"✅ Configuration v11.5 loaded - Mode: {mode.value}")
    logger.info(f"  Intelligent Routing: {'✅' if intelligent_routing else '❌'}")
    logger.info(f"  OpenAI (o3-mini): {'✅' if openai_api_key else '⚠️ 未設定'}")
    logger.info(f"  Mistral (Large 3): {'✅' if mistral_api_key else '⚠️ 未設定'}")
    logger.info(f"  xAI (Grok 4.1 Fast): {'✅' if xai_api_key else '⚠️ 未設定'}")
    logger.info(f"  Nemotron Local: {llamacpp_endpoint}")

    return config
