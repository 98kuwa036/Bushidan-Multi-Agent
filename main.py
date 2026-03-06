#!/usr/bin/env python3
"""
Bushidan Multi-Agent System v11.4 - Main Entry Point

武士団マルチエージェントシステム v11.4
脱中国企業・9層ハイブリッドアーキテクチャ
Universal Multi-LLM Framework based on Samurai hierarchy.

v11.4 Features:
- 脱中国企業: Alibaba/Qwen/Kimi/Moonshot 全排除
- 大元帥 (Daigensui): Claude Opus 4.5 - 最高難度・戦略設計
- 将軍 (Shogun): Claude Sonnet 4.6 - 高難度コーディング
- 軍師 (Gunshi): o3-mini (high) - 推論・設計・PDCA
- 参謀-A (Sanbo-A): GPT-5 - 汎用コーディング
- 参謀-B (Sanbo-B): Grok-code-fast-1 - 実装・バグ修正・高速
- 家老-A (Karo-A): Gemini Flash - 軽量タスク
- 家老-B (Karo-B): Llama 3.3 70B (Groq) - アルゴリズム特化
- 検校 (Kengyo): Gemini Flash Vision - マルチモーダル
- 隠密 (Onmitsu): Nemotron-3-Nano (Local) - 機密・超長文
"""

import asyncio
import logging
import json
import traceback
from typing import Optional

from core.system_orchestrator import SystemOrchestrator
from utils.config import load_config
from utils.logger import setup_logger


VERSION = "11.4"


def print_banner() -> None:
    """Print Bushidan v11.4 startup banner"""

    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║     武士団マルチエージェントシステム v{VERSION}                 ║
║     "脱中国企業・9層ハイブリッドアーキテクチャ + PDCA + BDI"   ║
╠══════════════════════════════════════════════════════════════╣
║  9層階層:                                                     ║
║    大元帥 (Daigensui)  - Claude Opus 4.5    最高難度・戦略   ║
║    将軍   (Shogun)     - Claude Sonnet 4.6  高難度コーディング║
║    軍師   (Gunshi)     - o3-mini (high)     推論・設計・PDCA ║
║    参謀-A (Sanbo-A)    - GPT-5              汎用コーディング ║
║    参謀-B (Sanbo-B)    - Grok-code-fast-1   実装・高速       ║
║    家老-A (Karo-A)     - Gemini Flash       軽量タスク       ║
║    家老-B (Karo-B)     - Llama 3.3 70B      アルゴリズム特化 ║
║    検校   (Kengyo)     - Gemini Flash Vision マルチモーダル  ║
║    隠密   (Onmitsu)    - Nemotron-3-Nano    機密・超長文     ║
╠══════════════════════════════════════════════════════════════╣
║  v11.4 新機能:                                                ║
║    脱中国企業: Alibaba/Qwen/Kimi/Moonshot 全排除             ║
║    大元帥新設: Claude Opus 4.5 (最高権限)                     ║
║    参謀新設: GPT-5 + Grok-code-fast-1 (実装二刀流)           ║
║    隠密新設: Nemotron-3-Nano (ローカル機密処理)              ║
║    2台体制: ProDesk(LLM専用) + EliteDesk(本陣)               ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main() -> None:
    """Main entry point for Bushidan v11.4."""

    # Setup logging
    logger = setup_logger(f"bushidan_v{VERSION}")
    print_banner()
    logger.info(f"Bushidan Multi-Agent System v{VERSION} starting...")
    logging.getLogger().setLevel(logging.DEBUG)

    orchestrator = None
    shogun = None

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()

        # Initialize System Orchestrator (Shogun and all tiers are initialized inside)
        logger.info("🔧 Initializing System Orchestrator...")
        orchestrator = SystemOrchestrator(config)
        await orchestrator.initialize()

        # Shogun is already initialized inside orchestrator._initialize_tiers()
        shogun = orchestrator.shogun

        # Print component status
        _print_component_status(orchestrator)

        logger.info("All systems initialized successfully")
        logger.info("v{} Ready for commands".format(VERSION))
        logger.info("-" * 60)

        # Start main event loop (tasks flow through LangGraph router via orchestrator)
        await shogun.start_service()

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise
    finally:
        # Print final statistics
        if shogun:
            try:
                _print_final_statistics(shogun, orchestrator)
            except Exception as e:
                logger.warning(f"Could not print final statistics: {e}")

        # Shutdown orchestrator
        if orchestrator:
            await orchestrator.shutdown()

        logger.info(f"Bushidan v{VERSION} shutdown complete")


def _print_component_status(orchestrator: SystemOrchestrator) -> None:
    """Print status of all initialized components"""

    print("\n" + "=" * 60)
    print("Component Status")
    print("=" * 60)

    # AI Clients
    clients = {
        "Claude Opus 4.5 (Daigensui)": "claude_opus",
        "Claude Sonnet 4.6 (Shogun)": "claude_cached",
        "o3-mini (Gunshi)": "o3_mini",
        "GPT-5 (Sanbo-A)": "gpt5",
        "Grok-code-fast-1 (Sanbo-B)": "grok_code",
        "Gemini Flash (Karo-A)": "gemini_flash",
        "Llama 3.3 70B / Groq (Karo-B)": "groq",
        "Gemini Flash Vision (Kengyo)": "gemini_flash_vision",
        "Nemotron-3-Nano (Onmitsu)": "nemotron",
    }

    print("\nAI Clients:")
    for name, key in clients.items():
        status = "OK" if orchestrator.get_client(key) else "N/A"
        print(f"  [{status}] {name}")

    # MCP Servers
    mcps = ["memory", "filesystem", "git", "web_search"]
    print("\nMCP Servers:")
    for name in mcps:
        status = "OK" if orchestrator.get_mcp(name) else "N/A"
        print(f"  [{status}] {name}")

    # Kengyo (Visual Debugger)
    kengyo = orchestrator.kengyo
    kengyo_status = "OK" if kengyo and kengyo.is_available() else "N/A"
    print(f"\nVisual Debugger:")
    print(f"  [{kengyo_status}] Kengyo - Gemini Flash Vision + Playwright")

    # Router
    router_status = "OK" if orchestrator.get_router() else "N/A"
    print(f"\nIntelligent Router: [{router_status}]")

    # Features
    print("\nv11.4 Features:")
    print(f"  [{'OK' if orchestrator.config.intelligent_routing_enabled else 'N/A'}] Intelligent Routing")
    print(f"  [{'OK' if orchestrator.config.prompt_caching_enabled else 'N/A'}] Prompt Caching")
    print(f"  [{'OK' if orchestrator.config.power_optimization_enabled else 'N/A'}] Power Optimization")

    print("=" * 60 + "\n")


def _print_final_statistics(shogun: Shogun, orchestrator: SystemOrchestrator) -> None:
    """Print final statistics on shutdown"""

    print("\n" + "=" * 60)
    print("Final Statistics")
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

    result = await orchestrator.process_task(task_content)

    await orchestrator.shutdown()

    return result


if __name__ == "__main__":
    asyncio.run(main())
