#!/usr/bin/env python3
"""
Bushidan Multi-Agent System v9.3.2 - Main Entry Point

Universal Multi-LLM Framework based on Samurai hierarchy.
4-Tier Hybrid Architecture with Intelligent Routing.

v9.3.2 Features:
- Intelligent Router for optimal task delegation
- Prompt Caching for 90% cost reduction
- 3-tier fallback chain (99.5% reliability)
- Power-saving optimization
"""

import asyncio
import logging
import json
from typing import Optional

from core.shogun import Shogun
from core.system_orchestrator import SystemOrchestrator
from utils.config import load_config
from utils.logger import setup_logger


VERSION = "9.3.2"


def print_banner() -> None:
    """Print Bushidan v9.3.2 startup banner"""

    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║        🏯 Bushidan Multi-Agent System v{VERSION}              ║
║        "Universal Multi-LLM Framework"                       ║
╠══════════════════════════════════════════════════════════════╣
║  4-Tier Hierarchy:                                           ║
║    🎌 Shogun (Strategic)   - Claude Sonnet + Opus           ║
║    🏛️ Karo (Tactical)      - Groq + Gemini 3.0 Flash        ║
║    🏯 Taisho (Implementation) - Qwen3 + Kagemusha           ║
║    👣 Ashigaru (Execution)  - MCP Servers                   ║
╠══════════════════════════════════════════════════════════════╣
║  v9.3.2 Innovations:                                         ║
║    ⚡ Intelligent Routing (60% faster simple tasks)         ║
║    💾 Prompt Caching (90% cost reduction)                   ║
║    🔗 3-tier Fallback (99.5% reliability)                   ║
║    🔋 Power Optimization (¥200/month savings)               ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main() -> None:
    """Main entry point for Bushidan v9.3.2."""

    # Setup logging
    logger = setup_logger(f"bushidan_v{VERSION}")
    print_banner()
    logger.info(f"🏯 Bushidan Multi-Agent System v{VERSION} starting...")

    orchestrator = None
    shogun = None

    try:
        # Load configuration
        logger.info("📝 Loading configuration...")
        config = load_config()

        # Initialize System Orchestrator
        logger.info("🔧 Initializing System Orchestrator...")
        orchestrator = SystemOrchestrator(config)
        await orchestrator.initialize()

        # Initialize Shogun (Strategic Layer)
        logger.info("🎌 Initializing Shogun (Strategic Layer)...")
        shogun = Shogun(orchestrator)
        await shogun.initialize()

        # Print component status
        _print_component_status(orchestrator)

        logger.info("✅ All systems initialized successfully")
        logger.info("🎌 将軍システム v{} 準備完了 - Ready for commands".format(VERSION))
        logger.info("-" * 60)

        # Start main event loop
        await shogun.start_service()

    except KeyboardInterrupt:
        logger.info("📴 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        raise
    finally:
        # Print final statistics
        if shogun:
            try:
                _print_final_statistics(shogun, orchestrator)
            except Exception as e:
                logger.warning(f"⚠️ Could not print final statistics: {e}")

        # Shutdown orchestrator
        if orchestrator:
            await orchestrator.shutdown()

        logger.info(f"🏯 Bushidan v{VERSION} shutdown complete")


def _print_component_status(orchestrator: SystemOrchestrator) -> None:
    """Print status of all initialized components"""

    print("\n" + "=" * 60)
    print("Component Status")
    print("=" * 60)

    # AI Clients
    clients = {
        "Claude (Cached)": "claude_cached",
        "Groq": "groq",
        "Gemini 3.0 Flash": "gemini3",
        "Local Qwen3": "qwen3",
        "Kagemusha (Cloud Qwen3)": "alibaba_qwen",
        "Opus (Premium Review)": "opus"
    }

    print("\nAI Clients:")
    for name, key in clients.items():
        status = "✅" if orchestrator.get_client(key) else "❌"
        print(f"  {status} {name}")

    # MCP Servers
    mcps = ["memory", "filesystem", "git", "web_search"]
    print("\nMCP Servers:")
    for name in mcps:
        status = "✅" if orchestrator.get_mcp(name) else "❌"
        print(f"  {status} {name}")

    # Router
    router_status = "✅" if orchestrator.get_router() else "❌"
    print(f"\nIntelligent Router: {router_status}")

    # Features
    print("\nv9.3.2 Features:")
    print(f"  {'✅' if orchestrator.config.intelligent_routing_enabled else '❌'} Intelligent Routing")
    print(f"  {'✅' if orchestrator.config.prompt_caching_enabled else '❌'} Prompt Caching")
    print(f"  {'✅' if orchestrator.config.power_optimization_enabled else '❌'} Power Optimization")

    print("=" * 60 + "\n")


def _print_final_statistics(shogun: Shogun, orchestrator: SystemOrchestrator) -> None:
    """Print final statistics on shutdown"""

    print("\n" + "=" * 60)
    print("📊 Final Statistics")
    print("=" * 60)

    try:
        stats = shogun.get_statistics()

        # Routing stats
        routing = stats.get("routing_stats", {})
        print(f"\nTasks Processed: {routing.get('total_tasks', 0)}")
        print(f"Total Time: {routing.get('total_time_seconds', 0):.1f}s")
        print(f"Power Savings: {routing.get('power_savings', 0)} tasks via Groq")

        # Reviews
        reviews = stats.get("reviews_by_level", {})
        print(f"\nReviews:")
        print(f"  Basic: {reviews.get('basic', 0)}")
        print(f"  Detailed: {reviews.get('detailed', 0)}")
        print(f"  Premium (Opus): {reviews.get('premium', 0)}")

        # Complexity distribution
        by_complexity = routing.get("by_complexity", {})
        if by_complexity:
            print(f"\nBy Complexity:")
            for level, count in by_complexity.items():
                print(f"  {level}: {count}")

    except Exception as e:
        print(f"Could not collect statistics: {e}")

    print("=" * 60 + "\n")


async def quick_task(task_content: str) -> dict:
    """
    Quick task execution for CLI usage

    Usage:
        from main import quick_task
        result = await quick_task("Write a hello world function")
    """

    logger = setup_logger(f"bushidan_v{VERSION}")

    config = load_config()
    orchestrator = SystemOrchestrator(config)
    await orchestrator.initialize()

    shogun = Shogun(orchestrator)
    await shogun.initialize()

    from core.shogun import Task, TaskComplexity
    task = Task(content=task_content, complexity=TaskComplexity.MEDIUM)

    result = await shogun.process_task(task)

    await orchestrator.shutdown()

    return result


if __name__ == "__main__":
    asyncio.run(main())
