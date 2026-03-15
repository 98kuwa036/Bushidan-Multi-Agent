#!/usr/bin/env python3
"""
Bushidan Multi-Agent System v14 - Main Entry Point

武士団マルチエージェントシステム v14
10役職体制 + LangGraph v14 + MCP SDK + HITL + モバイルコンソール
"""

import asyncio
import logging

from utils.config import load_config
from utils.logger import setup_logger


VERSION = "14"


def print_banner() -> None:
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║     武士団マルチエージェントシステム v{VERSION}                    ║
║     "10役職体制 + LangGraph v14 + HITL + MCP SDK"           ║
╠══════════════════════════════════════════════════════════════╣
║  10役職:                                                      ║
║    大元帥  - Claude Opus 4.6    最高難度・戦略設計             ║
║    将軍    - Claude Sonnet 4.6  メインワーカー                 ║
║    軍師    - Mistral Large 3     深層推論・PDCA                 ║
║    参謀    - Mistral Large 3    汎用コーディング               ║
║    外事    - Command R+         RAG・外部情報                  ║
║    受付    - Command R          フォールバック                  ║
║    斥候    - Llama 3.3 (Groq)   高速Q&A                       ║
║    検校    - Gemini Vision      マルチモーダル                 ║
║    右筆    - ELYZA (Local)      日本語清書                     ║
║    隠密    - Nemotron (Local)   機密・ローカル                 ║
╠══════════════════════════════════════════════════════════════╣
║  v14: ノードタイムアウト / ヘルスチェック統合 / HITL            ║
║       MCP SDK / ClientRegistry / モバイルコンソール            ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main() -> None:
    logger = setup_logger(f"bushidan_v{VERSION}")
    print_banner()
    logger.info("Bushidan v%s starting...", VERSION)

    orchestrator = None

    try:
        config = load_config()

        from core.system_orchestrator import SystemOrchestrator
        orchestrator = SystemOrchestrator(config)
        await orchestrator.initialize()

        logger.info("v%s Ready for commands", VERSION)
        logger.info("-" * 60)

        # Keep running
        while True:
            await asyncio.sleep(3600)

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error("Critical error: %s", e)
        raise
    finally:
        if orchestrator:
            await orchestrator.shutdown()
        logger.info("Bushidan v%s shutdown complete", VERSION)


async def quick_task(task_content: str) -> dict:
    """Quick task execution for CLI usage"""
    config = load_config()
    from core.system_orchestrator import SystemOrchestrator
    orchestrator = SystemOrchestrator(config)
    await orchestrator.initialize()
    result = await orchestrator.process_task(task_content)
    await orchestrator.shutdown()
    return result


if __name__ == "__main__":
    asyncio.run(main())
