"""
Bushidan Multi-Agent System v9.1 - System Orchestrator

Simplified system coordination for the Universal Multi-LLM Framework.
Manages the 3-tier hierarchy: Shogun -> Karo -> Ashigaru
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from mcp.memory_mcp import MemoryMCP
from mcp.filesystem_mcp import FilesystemMCP
from mcp.git_mcp import GitMCP
from mcp.web_search_mcp import WebSearchMCP


logger = get_logger(__name__)


class SystemMode(Enum):
    """v9.1 Simplified system modes"""
    BATTALION = "battalion"  # Full system: Shogun + Karo + Ashigaru + All MCP
    COMPANY = "company"      # Slack mode: Karo + Ashigaru + Memory MCP  
    PLATOON = "platoon"      # HA OS mode: Ashigaru + Dynamic MCP


@dataclass
class SystemConfig:
    """v9.1 Configuration structure"""
    mode: SystemMode
    claude_api_key: str
    gemini_api_key: str
    tavily_api_key: str
    slack_token: Optional[str] = None
    notion_token: Optional[str] = None
    ollama_endpoint: str = "http://localhost:11434"
    litellm_endpoint: str = "http://localhost:8000"


class SystemOrchestrator:
    """
    v9.1 System Orchestrator - Simplified coordination layer
    
    Manages the 6 core components:
    1. Shogun (Strategic Layer)
    2. Karo (Tactical Layer) 
    3. Ashigaru (Execution Layer)
    4. Memory MCP
    5. Core MCPs (Filesystem, Git)
    6. Web Search MCP
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.mcps: Dict[str, Any] = {}
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize all system components"""
        logger.info("🔧 Initializing Bushidan v9.1 components...")
        
        try:
            # Initialize MCP servers
            await self._initialize_mcps()
            
            # Verify external dependencies
            await self._verify_dependencies()
            
            self.initialized = True
            logger.info("✅ System orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    async def _initialize_mcps(self) -> None:
        """Initialize Model Context Protocol servers"""
        
        # Core MCPs (always required)
        self.mcps["memory"] = MemoryMCP()
        self.mcps["filesystem"] = FilesystemMCP()
        self.mcps["git"] = GitMCP()
        
        # Web search MCP (if Tavily API available)
        if self.config.tavily_api_key:
            self.mcps["web_search"] = WebSearchMCP(self.config.tavily_api_key)
        
        # Initialize all MCP servers
        for name, mcp in self.mcps.items():
            await mcp.initialize()
            logger.info(f"✅ {name.title()} MCP initialized")
    
    async def _verify_dependencies(self) -> None:
        """Verify external service availability"""
        
        # Check Ollama availability
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.ollama_endpoint}/api/tags")
                if response.status_code == 200:
                    logger.info("✅ Ollama service available")
                else:
                    logger.warning("⚠️ Ollama service not responding")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify Ollama: {e}")
        
        # Check LiteLLM proxy
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.litellm_endpoint}/v1/models")
                if response.status_code == 200:
                    logger.info("✅ LiteLLM proxy available")
                else:
                    logger.warning("⚠️ LiteLLM proxy not responding")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify LiteLLM: {e}")
    
    def get_mcp(self, name: str) -> Optional[Any]:
        """Get MCP server by name"""
        return self.mcps.get(name)
    
    async def shutdown(self) -> None:
        """Graceful shutdown of all components"""
        logger.info("📴 Shutting down system orchestrator...")
        
        for name, mcp in self.mcps.items():
            try:
                await mcp.shutdown()
                logger.info(f"✅ {name.title()} MCP shutdown complete")
            except Exception as e:
                logger.warning(f"⚠️ Error shutting down {name} MCP: {e}")
        
        logger.info("✅ System orchestrator shutdown complete")