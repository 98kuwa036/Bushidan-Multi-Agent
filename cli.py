#!/usr/bin/env python3
"""将軍システム - CLI (IDE Console Interface)

IDEのコンソールから呼び出せるCLIインターフェース。

Usage:
    shogun repl                     # 対話モード
    shogun ask "prompt"             # タスク実行
    shogun ask -m company "prompt"  # 中隊モード (¥0)
    shogun ask -a taisho "prompt"   # エージェント指定
    shogun mode                     # 現在のモード表示
    shogun mode company             # 中隊モードに切替
    shogun status                   # システム状態
    shogun stats                    # コスト統計
    shogun health                   # ヘルスチェック
    shogun server                   # REST APIサーバー起動
    shogun pipe                     # stdin パイプ入力
    shogun maintenance run          # 月次メンテナンス実行
    shogun maintenance reports      # 過去レポート一覧
    shogun maintenance next         # 次回メンテナンス日
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_controller():
    from shogun.core.controller import Controller
    base_dir = str(Path(__file__).resolve().parent)
    return Controller(base_dir=base_dir)


# ─── Commands ───

async def cmd_ask(args) -> None:
    """Execute a task."""
    ctrl = get_controller()
    await ctrl.startup()

    prompt = " ".join(args.prompt)
    mode = args.mode or "battalion"
    agent = args.agent or ""

    try:
        result = await ctrl.process_task(
            prompt=prompt,
            mode=mode,
            force_agent=agent,
        )
        print(result)
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await ctrl.shutdown()


async def cmd_pipe(args) -> None:
    """Read from stdin and process."""
    ctrl = get_controller()
    await ctrl.startup()

    stdin_text = sys.stdin.read().strip()
    if not stdin_text:
        print("stdin is empty", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix or "以下の入力を分析してください"
    prompt = f"{prefix}:\n\n```\n{stdin_text}\n```"
    mode = args.mode or "company"  # Default: company for pipe

    try:
        result = await ctrl.process_task(prompt=prompt, mode=mode)
        print(result)
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await ctrl.shutdown()


async def cmd_repl(args) -> None:
    """Interactive REPL mode."""
    ctrl = get_controller()
    await ctrl.startup()

    current_mode = args.mode or "battalion"
    current_agent = ""

    print("=" * 60)
    print("  将軍システム v5.0 - 対話モード")
    print("=" * 60)
    print(f"  編成: {'大隊' if current_mode == 'battalion' else '中隊'}")
    print()
    print("  コマンド:")
    print("    /mode [battalion|company]  モード切替")
    print("    /agent [taisho|karo|shogun]  エージェント指定")
    print("    /agent                    エージェント指定解除")
    print("    /status                   ステータス表示")
    print("    /stats                    コスト統計")
    print("    /health                   ヘルスチェック")
    print("    quit / exit               終了")
    print("=" * 60)
    print()

    try:
        while True:
            try:
                mode_label = "大隊" if current_mode == "battalion" else "中隊"
                agent_label = f"→{current_agent}" if current_agent else ""
                prompt_str = f"[{mode_label}{agent_label}] > "
                line = input(prompt_str).strip()
            except EOFError:
                break

            if not line:
                continue

            if line.lower() in ("quit", "exit", "/quit", "/exit"):
                print("退陣。")
                break

            # REPL commands
            if line.startswith("/"):
                parts = line.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/mode":
                    if arg in ("battalion", "company", "大隊", "中隊"):
                        if arg == "大隊":
                            arg = "battalion"
                        elif arg == "中隊":
                            arg = "company"
                        current_mode = arg
                        label = "大隊" if arg == "battalion" else "中隊"
                        print(f"  → 編成変更: {label}")
                    else:
                        label = "大隊" if current_mode == "battalion" else "中隊"
                        print(f"  現在: {label}")
                        print("  使用: /mode battalion|company")

                elif cmd == "/agent":
                    if arg in ("taisho", "karo", "shogun", "侍大将", "家老", "将軍"):
                        name_map = {"侍大将": "taisho", "家老": "karo", "将軍": "shogun"}
                        current_agent = name_map.get(arg, arg)
                        print(f"  → エージェント固定: {current_agent}")
                    elif not arg:
                        current_agent = ""
                        print("  → エージェント固定解除 (自動選択)")
                    else:
                        print("  使用: /agent [taisho|karo|shogun]")

                elif cmd == "/status":
                    st = ctrl.get_status()
                    mode_label = "大隊" if st["mode"] == "battalion" else "中隊"
                    print(f"  モード: {mode_label}")
                    print(f"  待機タスク: {st['pending_tasks']}")
                    print(f"  総タスク数: {st['total_tasks']}")
                    ds = st["dashboard"]
                    print(f"  本日完了: {ds['completed_today']}")
                    print(f"  本日コスト: ¥{ds['total_cost_yen']:,}")

                elif cmd == "/stats":
                    print(ctrl.show_stats())

                elif cmd == "/health":
                    r1_ok = await ctrl.openvino.health()
                    cli_ok = await ctrl.claude_cli.check_available()
                    print(f"  侍大将 R1 (OpenVINO): {'✓' if r1_ok else '✗'}")
                    print(f"  Claude CLI (Pro版):    {'✓' if cli_ok else '✗'}")
                    api_ok = ctrl.api_provider is not None
                    print(f"  Anthropic API:        {'✓' if api_ok else '✗ (KEY未設定)'}")

                elif cmd == "/help":
                    print("  /mode [battalion|company]  モード切替")
                    print("  /agent [name]              エージェント指定")
                    print("  /status                    ステータス")
                    print("  /stats                     コスト統計")
                    print("  /health                    ヘルスチェック")
                    print("  quit                       終了")

                else:
                    print(f"  不明なコマンド: {cmd}")
                continue

            # Normal prompt
            try:
                result = await ctrl.process_task(
                    prompt=line,
                    mode=current_mode,
                    force_agent=current_agent,
                )
                print()
                print(result)
                print()
            except Exception as e:
                print(f"  エラー: {e}", file=sys.stderr)

    finally:
        await ctrl.shutdown()


async def cmd_status(args) -> None:
    ctrl = get_controller()
    st = ctrl.get_status()
    mode_label = "大隊" if st["mode"] == "battalion" else "中隊"
    print(f"モード: {mode_label}")
    print(f"待機タスク: {st['pending_tasks']}")
    print(f"総タスク数: {st['total_tasks']}")


async def cmd_health(args) -> None:
    ctrl = get_controller()
    r1_ok = await ctrl.openvino.health()
    cli_ok = await ctrl.claude_cli.check_available()
    api_ok = os.environ.get("ANTHROPIC_API_KEY", "") != ""

    print("ヘルスチェック:")
    print(f"  侍大将 R1 (OpenVINO): {'✓ 稼働中' if r1_ok else '✗ 停止'}")
    print(f"  Claude CLI (Pro版):   {'✓ 利用可能' if cli_ok else '✗ 未インストール'}")
    print(f"  Anthropic API:       {'✓ KEY設定済' if api_ok else '✗ KEY未設定'}")
    await ctrl.openvino.close()


async def cmd_stats(args) -> None:
    ctrl = get_controller()
    print(ctrl.show_stats())


async def cmd_mode(args) -> None:
    ctrl = get_controller()
    if args.target:
        from shogun.core.task_queue import DeploymentMode
        target = args.target
        if target in ("大隊", "battalion"):
            ctrl.current_mode = DeploymentMode.BATTALION
            print("→ 大隊モードに切替")
        elif target in ("中隊", "company"):
            ctrl.current_mode = DeploymentMode.COMPANY
            print("→ 中隊モードに切替")
        else:
            print(f"不明なモード: {target}")
    else:
        label = "大隊" if ctrl.current_mode.value == "battalion" else "中隊"
        print(f"現在のモード: {label}")


def cmd_server(args) -> None:
    """Start FastAPI server."""
    try:
        import uvicorn
        from shogun.main import create_app
        app = create_app()
        uvicorn.run(
            app,
            host=args.host or "0.0.0.0",
            port=args.port or 8080,
        )
    except ImportError as e:
        print(f"サーバー起動に必要なパッケージがありません: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_maintenance(args) -> None:
    """Monthly maintenance (反省会) commands."""
    from shogun.core.maintenance import MaintenanceManager

    manager = MaintenanceManager(base_dir=Path(__file__).resolve().parent)
    action = getattr(args, "action", "status")

    if action == "run":
        print("=" * 60)
        print("  将軍システム - 月次メンテナンス (反省会)")
        print("=" * 60)
        print()
        print("メンテナンス実行中...")
        print()

        report = manager.run_full_maintenance()

        # Summary
        summary = report["summary"]
        print(f"総チェック数: {summary['total']}")
        print(f"  ✓ 正常: {summary['passed']}")
        print(f"  ⚠ 警告: {summary['warnings']}")
        print(f"  ✗ エラー: {summary['errors']}")
        print()

        # Details
        for check_id, result in report["checks"].items():
            status = result.get("status", "unknown")
            emoji = {"ok": "✓", "warning": "⚠", "error": "✗"}.get(status, "?")
            name = result.get("name", check_id)
            msg = result.get("message", "")
            print(f"  {emoji} {name}: {msg}")

        print()
        print(f"レポート保存: reports/maintenance/maintenance_{report['timestamp']}.md")
        print()

    elif action == "reports":
        reports = manager.list_reports(limit=args.limit or 10)
        if not reports:
            print("過去のレポートがありません。")
            return

        print("過去のメンテナンスレポート:")
        print()
        for r in reports:
            s = r.get("summary", {})
            status = "✓" if s.get("errors", 0) == 0 else "✗"
            print(f"  {status} {r['date'][:10]}  passed:{s.get('passed',0)}  warnings:{s.get('warnings',0)}  errors:{s.get('errors',0)}")
            print(f"      → {r['file']}")

    elif action == "next":
        next_date = manager.get_next_maintenance_date()
        print(f"次回メンテナンス: {next_date.strftime('%Y-%m-%d %H:%M')}")

    elif action == "check":
        # Run single check
        check_name = args.check_name
        check_methods = {
            "llm": manager.check_llm_versions,
            "llm_versions": manager.check_llm_versions,
            "openvino": manager.check_openvino_model,
            "mcp": manager.check_mcp_servers,
            "health": manager.check_system_health,
            "logs": manager.cleanup_logs,
            "cost": manager.generate_cost_report,
        }
        if check_name in check_methods:
            result = check_methods[check_name]()
            print(f"{result.get('name', check_name)}:")
            print(f"  ステータス: {result.get('status', 'unknown')}")
            print(f"  結果: {result.get('message', '')}")
            if result.get("updates_available"):
                print("  更新可能:")
                for u in result["updates_available"]:
                    if isinstance(u, dict):
                        print(f"    - {u.get('package')}: {u.get('current')} → {u.get('latest')}")
                    else:
                        print(f"    - {u}")
        else:
            print(f"不明なチェック項目: {check_name}")
            print("利用可能: llm, openvino, mcp, health, logs, cost")

    else:
        # status (default)
        next_date = manager.get_next_maintenance_date()
        reports = manager.list_reports(limit=1)

        print("月次メンテナンス (反省会):")
        print(f"  次回予定: {next_date.strftime('%Y-%m-%d %H:%M')}")

        if reports:
            last = reports[0]
            s = last.get("summary", {})
            status = "正常" if s.get("errors", 0) == 0 else "要確認"
            print(f"  前回実行: {last['date'][:10]} ({status})")
        else:
            print("  前回実行: なし")

        print()
        print("コマンド:")
        print("  shogun maintenance run       # メンテナンス実行")
        print("  shogun maintenance reports   # 過去レポート一覧")
        print("  shogun maintenance check mcp # 個別チェック実行")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        prog="shogun",
        description="将軍システム v5.0 - 階層型ハイブリッドAI開発システム",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細ログ")
    sub = parser.add_subparsers(dest="command")

    # ask
    p_ask = sub.add_parser("ask", help="タスク実行")
    p_ask.add_argument("prompt", nargs="+", help="タスク内容")
    p_ask.add_argument("-m", "--mode", default="battalion",
                       help="battalion (大隊) / company (中隊)")
    p_ask.add_argument("-a", "--agent", default="",
                       help="エージェント指定: taisho/karo/shogun")

    # pipe
    p_pipe = sub.add_parser("pipe", help="stdin パイプ入力")
    p_pipe.add_argument("-m", "--mode", default="company")
    p_pipe.add_argument("-p", "--prefix", default="以下の入力を分析してください")

    # repl
    p_repl = sub.add_parser("repl", help="対話モード")
    p_repl.add_argument("-m", "--mode", default="battalion")

    # status
    sub.add_parser("status", help="システム状態")

    # health
    sub.add_parser("health", help="ヘルスチェック")

    # stats
    sub.add_parser("stats", help="コスト統計")

    # mode
    p_mode = sub.add_parser("mode", help="モード表示/切替")
    p_mode.add_argument("target", nargs="?", default="",
                        help="battalion/company")

    # server
    p_server = sub.add_parser("server", help="REST APIサーバー起動")
    p_server.add_argument("--host", default="0.0.0.0")
    p_server.add_argument("--port", type=int, default=8080)

    # maintenance (反省会)
    p_maint = sub.add_parser("maintenance", help="月次メンテナンス (反省会)")
    p_maint_sub = p_maint.add_subparsers(dest="action")
    p_maint_sub.add_parser("run", help="メンテナンス実行")
    p_maint_reports = p_maint_sub.add_parser("reports", help="過去レポート一覧")
    p_maint_reports.add_argument("-n", "--limit", type=int, default=10, help="表示件数")
    p_maint_sub.add_parser("next", help="次回メンテナンス日")
    p_maint_check = p_maint_sub.add_parser("check", help="個別チェック実行")
    p_maint_check.add_argument("check_name", help="チェック項目: llm, openvino, mcp, health, logs, cost")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        # Default: REPL
        args.command = "repl"
        args.mode = "battalion"

    if args.command == "server":
        cmd_server(args)
        return

    if args.command == "maintenance":
        cmd_maintenance(args)
        return

    # Async commands
    cmd_map = {
        "ask": cmd_ask,
        "pipe": cmd_pipe,
        "repl": cmd_repl,
        "status": cmd_status,
        "health": cmd_health,
        "stats": cmd_stats,
        "mode": cmd_mode,
    }

    fn = cmd_map.get(args.command)
    if fn:
        asyncio.run(fn(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
