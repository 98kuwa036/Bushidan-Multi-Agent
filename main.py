#!/usr/bin/env python3
"""
Bushidan Multi-Agent System v9.1 - Main Entry Point

Universal Multi-LLM Framework based on Samurai hierarchy.
Simple, practical, and universal design philosophy.
"""

import asyncio
import logging
from typing import Optional

from core.shogun import Shogun
from core.system_orchestrator import SystemOrchestrator
from utils.config import load_config
from utils.logger import setup_logger


async def main() -> None:
    """Main entry point for Bushidan v9.1."""
    
    # Setup logging
    logger = setup_logger("bushidan_v9_1")
    logger.info("🏯 Bushidan Multi-Agent System v9.1 starting...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize System Orchestrator
        orchestrator = SystemOrchestrator(config)
        await orchestrator.initialize()
        
        # Initialize Shogun (Strategic Layer)
        shogun = Shogun(orchestrator)
        
        logger.info("✅ All systems initialized successfully")
        logger.info("🎌 将軍システム準備完了 - Ready for commands")
        
        # Start main event loop
        await shogun.start_service()
        
    except KeyboardInterrupt:
        logger.info("📴 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        raise
    finally:
        logger.info("🏯 Bushidan v9.1 shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())