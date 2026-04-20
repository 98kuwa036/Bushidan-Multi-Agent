#!/usr/bin/env python3
"""
Bushidan Multi-Agent System v18 - Main Entry Point

武士団マルチエージェントシステム v18
12役職体制 + LangGraph + MCP SDK + HITL + 独自 Web UI
"""

import asyncio
import logging

from utils.config import load_config
from utils.logger import setup_logger


VERSION = "18"


def print_banner() -> None:
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║     武士団マルチエージェントシステム v{VERSION}                    ║
║     "10役職体制 + LangGraph v15 + HITL + MCP SDK"           ║
║     + 分散Claude処理 (Claude API Server)                    ║
╠══════════════════════════════════════════════════════════════╣
║  10役職:                                                      ║
║    大元帥  - Claude Opus 4.6 (CLI)  最高難度・戦略設計        ║
║    将軍    - Claude Sonnet 4.6 (CLI) メインワーカー           ║
║    軍師    - Mistral Large 3        深層推論・PDCA             ║
║    参謀    - Mistral Large 3        汎用コーディング           ║
║    外事    - Command A 03-2025      RAG・外部情報              ║
║    受付    - Gemini 3.1 Flash-Lite  フォールバック             ║
║    斥候    - Llama 3.3 (Groq)       高速Q&A                   ║
║    検校    - Gemini 3.1 Flash-Image マルチモーダル             ║
║    右筆    - Gemini 3.1 Flash-Lite  日本語清書                 ║
║    隠密    - Gemma3 27B/Nemotron 3  機密・ローカル             ║
╠══════════════════════════════════════════════════════════════╣
║  v15 更新: 分散Claude処理アーキテクチャ                        ║
║    ✓ Claude API Server (192.168.11.237:8070)                ║
║    ✓ Claude Pro CLI優先 → Anthropic API フォールバック       ║
║    ✓ バインドマウント経由でプロジェクトコンテキスト共有       ║
║    ✓ メモリ圧力低減 (bushidan-honjin)                        ║
║    ✓ ノードタイムアウト / ヘルスチェック統合 / HITL           ║
║    ✓ MCP SDK / ClientRegistry / モバイルコンソール           ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main() -> None:
    logger = setup_logger(f"bushidan_v{VERSION}")
    print_banner()
    logger.info("Bushidan v%s starting...", VERSION)

    orchestrator = None
    background_tasks = []

    try:
        config = load_config()

        # Claude API Server への接続確認
        import os
        claude_api_server = os.environ.get("CLAUDE_API_SERVER_URL")
        logger.info("Claude API Server: %s", claude_api_server or "(未設定)")

        from core.system_orchestrator import SystemOrchestrator
        orchestrator = SystemOrchestrator(config)
        await orchestrator.initialize()

        # ── DelegationWorker 起動 ─────────────────────────────────────
        try:
            from core.delegation_worker import DelegationWorker
            delegation_worker = DelegationWorker.get()
            delegation_task = asyncio.create_task(
                delegation_worker.start(), name="delegation_worker"
            )
            background_tasks.append(delegation_task)
            logger.info("✅ DelegationWorker タスク起動")
        except Exception as e:
            logger.warning("⚠️  DelegationWorker スキップ: %s", e)

        logger.info("v%s Ready for commands", VERSION)
        logger.info("=" * 60)
        logger.info("Claude処理チェーン:")
        logger.info("  1. リモート Claude API Server (%s)", claude_api_server or "未設定")
        logger.info("  2. Claude Pro CLI (優先)")
        logger.info("  3. Anthropic API (フォールバック)")
        logger.info("=" * 60)

        # Keep running
        while True:
            await asyncio.sleep(3600)

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error("Critical error: %s", e)
        raise
    finally:
        for task in background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
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
