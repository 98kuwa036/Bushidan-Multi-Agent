#!/usr/bin/env python3
"""
Omni-P4 Shogun-Hybrid CLI - IDEコンソールから呼び出せるインターフェース

Usage:
    shogun ask "ESP32-P4のDMA設定を教えて"
    shogun ask --agent taisho "このビルドエラーを分析して"
    shogun ask --category think "SPI DMAの設計方針を考えて"
    shogun mode                     # 現在のモード表示
    shogun mode a                   # Mode A (軍議) に切替
    shogun mode b                   # Mode B (進軍) に切替
    shogun status                   # システム状態表示
    shogun agents                   # エージェント一覧
    shogun health                   # Ollama接続チェック
    shogun models                   # ロード済みモデル一覧
    shogun unload                   # 全モデルアンロード
    shogun server                   # FastAPIサーバー起動
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shogun.core.controller import Controller, SystemMode
from shogun.core.task_queue import Task, TaskCategory
from shogun.providers.ollama import OllamaClient
from shogun.agents.factory import create_local_agents, create_cloud_agents

# Logging
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=LOG_FORMAT, level=level, stream=sys.stderr)


def find_config() -> str:
    """Find settings.yaml in expected locations."""
    candidates = [
        PROJECT_ROOT / "shogun" / "config" / "settings.yaml",
        Path("shogun/config/settings.yaml"),
        Path("config/settings.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("settings.yaml not found. Run from project root.")


async def build_controller() -> Controller:
    """Build and initialize the controller with all agents."""
    config_path = find_config()
    ctrl = Controller(config_path=config_path)

    # Register local agents
    local_agents = create_local_agents(ctrl.ollama, ctrl.config)
    for name, agent in local_agents.items():
        ctrl.register_agent(name, agent)

    # Register cloud agents (optional - requires API key)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            from shogun.providers.anthropic_api import AnthropicClient
            anthropic = AnthropicClient(api_key=api_key)
            cloud_agents = create_cloud_agents(anthropic, ctrl.config)
            for name, agent in cloud_agents.items():
                ctrl.register_agent(name, agent)
        except Exception as e:
            logging.warning("Cloud agents unavailable: %s", e)
    else:
        logging.info(
            "ANTHROPIC_API_KEY not set. Cloud agents (Shogun/Karo) disabled."
        )

    return ctrl


# ------------------------------------------------------------------
# CLI Commands
# ------------------------------------------------------------------

async def cmd_ask(args: argparse.Namespace) -> None:
    """Execute a task."""
    ctrl = await build_controller()
    try:
        result = await ctrl.ask(
            prompt=args.prompt,
            category=args.category,
            agent=args.agent or "",
        )
        print("\n" + "=" * 60)
        print(f"[{args.category.upper()}] 結果:")
        print("=" * 60)
        print(result)
        print("=" * 60)
    finally:
        await ctrl.shutdown()


async def cmd_mode(args: argparse.Namespace) -> None:
    """Show or switch mode."""
    ctrl = await build_controller()
    try:
        if args.target:
            target_map = {
                "a": SystemMode.MODE_A,
                "b": SystemMode.MODE_B,
                "cloud": SystemMode.CLOUD,
                "idle": SystemMode.IDLE,
            }
            target = target_map.get(args.target.lower())
            if not target:
                print(f"Unknown mode: {args.target}")
                print("Available: a, b, cloud, idle")
                return
            await ctrl.switch_mode(target)
            print(f"モード切替完了: {target.value}")
            if target == SystemMode.MODE_A:
                print("  -> 侍大将 (Taisho) ロード完了。軍議モード。")
            elif target == SystemMode.MODE_B:
                print("  -> 足軽隊 (Leader/Coder/Scout) ロード完了。進軍モード。")
        else:
            print(f"現在のモード: {ctrl.current_mode.value}")
    finally:
        await ctrl.shutdown()


async def cmd_status(args: argparse.Namespace) -> None:
    """Show system status."""
    ctrl = await build_controller()
    try:
        st = ctrl.status()
        print("=" * 50)
        print("  Omni-P4 Shogun-Hybrid System Status")
        print("=" * 50)
        print(f"  Mode        : {st['mode']}")
        print(f"  Agents      : {', '.join(st['agents_registered'])}")
        print(f"  Pending     : {st['pending_tasks']}")
        print(f"  Total Tasks : {st['total_tasks']}")

        # Ollama status
        healthy = await ctrl.ollama.health()
        print(f"  Ollama      : {'Online' if healthy else 'OFFLINE'}")

        if healthy:
            models = await ctrl.ollama.list_models()
            print(f"  Models (DL) : {len(models)}")
            for m in models:
                name = m.get("name", "unknown")
                size_gb = m.get("size", 0) / (1024**3)
                print(f"    - {name} ({size_gb:.1f} GB)")
        print("=" * 50)
    finally:
        await ctrl.shutdown()


async def cmd_agents(args: argparse.Namespace) -> None:
    """List all agents."""
    ctrl = await build_controller()
    try:
        print("=" * 70)
        print("  Omni-P4 Agent Registry")
        print("=" * 70)
        print(f"  {'Name':<12} {'Codename':<12} {'Tier':<8} {'Role'}")
        print("-" * 70)
        for name, agent in sorted(ctrl._agents.items()):
            print(
                f"  {name:<12} {agent.codename:<12} {agent.tier:<8} "
                f"{agent.role[:40]}..."
            )
        print("=" * 70)
    finally:
        await ctrl.shutdown()


async def cmd_health(args: argparse.Namespace) -> None:
    """Check Ollama connectivity."""
    ollama = OllamaClient()
    try:
        ok = await ollama.health()
        if ok:
            print("Ollama: Online")
            running = await ollama._get_client()
            resp = await running.get("/api/ps")
            loaded = resp.json().get("models", [])
            if loaded:
                print("Loaded models:")
                for m in loaded:
                    print(f"  - {m.get('name', '?')} (expires: {m.get('expires_at', '?')})")
            else:
                print("No models currently loaded.")
        else:
            print("Ollama: OFFLINE")
            print("Start with: ollama serve")
    finally:
        await ollama.close()


async def cmd_models(args: argparse.Namespace) -> None:
    """List models."""
    ollama = OllamaClient()
    try:
        models = await ollama.list_models()
        print(f"Available models ({len(models)}):")
        for m in models:
            name = m.get("name", "?")
            size_gb = m.get("size", 0) / (1024**3)
            modified = m.get("modified_at", "?")
            print(f"  {name:<40} {size_gb:>6.1f} GB  ({modified})")
    finally:
        await ollama.close()


async def cmd_unload(args: argparse.Namespace) -> None:
    """Unload all models."""
    ollama = OllamaClient()
    try:
        print("Unloading all models...")
        await ollama.unload_all()
        print("Done. All models unloaded from memory.")
    finally:
        await ollama.close()


async def cmd_server(args: argparse.Namespace) -> None:
    """Start FastAPI server."""
    import uvicorn
    from shogun.main import create_app

    app = create_app()
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def cmd_pipe(args: argparse.Namespace) -> None:
    """Read from stdin and process (for IDE pipe integration)."""
    if sys.stdin.isatty():
        print("No pipe input detected. Use: echo 'prompt' | shogun pipe")
        return
    prompt = sys.stdin.read().strip()
    if not prompt:
        print("Empty input.")
        return
    args.prompt = prompt
    args.category = args.category if hasattr(args, "category") else "code"
    args.agent = args.agent if hasattr(args, "agent") else ""
    await cmd_ask(args)


# ------------------------------------------------------------------
# Interactive REPL
# ------------------------------------------------------------------

async def cmd_repl(args: argparse.Namespace) -> None:
    """Interactive REPL mode."""
    ctrl = await build_controller()
    print("=" * 60)
    print("  Omni-P4 Shogun-Hybrid REPL")
    print("  Type 'help' for commands, 'quit' to exit")
    print("=" * 60)

    category = "code"
    agent = ""

    try:
        while True:
            try:
                prompt = input(f"\n[{category}]{f'({agent})' if agent else ''} > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSayonara!")
                break

            if not prompt:
                continue

            # REPL commands
            if prompt.lower() == "quit" or prompt.lower() == "exit":
                print("Sayonara!")
                break
            elif prompt.lower() == "help":
                print("""
Commands:
  /mode [a|b|cloud]  - Switch mode
  /cat <category>    - Set default category (recon/code/plan/think/strategy/critical)
  /agent <name>      - Set default agent (scout/coder/leader/taisho/karo/shogun)
  /agent             - Clear agent override
  /status            - Show system status
  /agents            - List agents
  quit / exit        - Exit REPL

Otherwise, type your prompt and press Enter.
""")
                continue
            elif prompt.startswith("/mode"):
                parts = prompt.split()
                if len(parts) > 1:
                    target_map = {
                        "a": SystemMode.MODE_A,
                        "b": SystemMode.MODE_B,
                        "cloud": SystemMode.CLOUD,
                    }
                    t = target_map.get(parts[1].lower())
                    if t:
                        await ctrl.switch_mode(t)
                        print(f"Mode: {t.value}")
                    else:
                        print(f"Unknown mode: {parts[1]}")
                else:
                    print(f"Current mode: {ctrl.current_mode.value}")
                continue
            elif prompt.startswith("/cat"):
                parts = prompt.split()
                if len(parts) > 1:
                    category = parts[1]
                    print(f"Category: {category}")
                else:
                    print(f"Current category: {category}")
                continue
            elif prompt.startswith("/agent"):
                parts = prompt.split()
                if len(parts) > 1:
                    agent = parts[1]
                    print(f"Agent: {agent}")
                else:
                    agent = ""
                    print("Agent override cleared.")
                continue
            elif prompt == "/status":
                st = ctrl.status()
                print(json.dumps(st, indent=2, ensure_ascii=False))
                continue
            elif prompt == "/agents":
                for n, a in sorted(ctrl._agents.items()):
                    print(f"  {n:<12} ({a.tier}) {a.codename}")
                continue

            # Execute prompt
            try:
                result = await ctrl.ask(
                    prompt=prompt,
                    category=category,
                    agent=agent or "",
                )
                print("\n" + "-" * 40)
                print(result)
                print("-" * 40)
            except Exception as e:
                print(f"Error: {e}")

    finally:
        await ctrl.shutdown()


# ------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shogun",
        description="Omni-P4 Shogun-Hybrid CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command")

    # ask
    p_ask = subparsers.add_parser("ask", help="Execute a task")
    p_ask.add_argument("prompt", help="Task prompt")
    p_ask.add_argument("-c", "--category", default="code",
                       choices=["recon", "code", "plan", "think", "strategy", "critical"])
    p_ask.add_argument("-a", "--agent", default="",
                       help="Force specific agent (scout/coder/leader/taisho/karo/shogun)")

    # mode
    p_mode = subparsers.add_parser("mode", help="Show/switch mode")
    p_mode.add_argument("target", nargs="?", help="a / b / cloud / idle")

    # status
    subparsers.add_parser("status", help="System status")

    # agents
    subparsers.add_parser("agents", help="List agents")

    # health
    subparsers.add_parser("health", help="Ollama health check")

    # models
    subparsers.add_parser("models", help="List Ollama models")

    # unload
    subparsers.add_parser("unload", help="Unload all models")

    # server
    p_server = subparsers.add_parser("server", help="Start API server")
    p_server.add_argument("--host", default="0.0.0.0")
    p_server.add_argument("--port", type=int, default=8400)

    # pipe
    p_pipe = subparsers.add_parser("pipe", help="Read from stdin")
    p_pipe.add_argument("-c", "--category", default="code")
    p_pipe.add_argument("-a", "--agent", default="")

    # repl
    subparsers.add_parser("repl", help="Interactive REPL mode")

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if not args.command:
        # Default: REPL mode
        asyncio.run(cmd_repl(args))
        return

    cmd_map = {
        "ask": cmd_ask,
        "mode": cmd_mode,
        "status": cmd_status,
        "agents": cmd_agents,
        "health": cmd_health,
        "models": cmd_models,
        "unload": cmd_unload,
        "server": cmd_server,
        "pipe": cmd_pipe,
        "repl": cmd_repl,
    }
    handler = cmd_map.get(args.command)
    if handler:
        asyncio.run(handler(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
