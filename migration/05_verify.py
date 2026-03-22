#!/usr/bin/env python3
# ============================================================================
# 武士団マルチエージェント - 移行後確認スクリプト
# 実行場所: LXC コンテナ内
# 機能: サービス疎通・パッケージ・設定ファイル確認
# ============================================================================

import subprocess
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# ─── カラー定義 ───────────────────────────────────────────────────────────
class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# ─── ロギング関数 ──────────────────────────────────────────────────────────
def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{Color.BLUE}[{timestamp}]{Color.RESET} {msg}")

def success(msg):
    print(f"{Color.GREEN}[✓]{Color.RESET} {msg}")

def error(msg):
    print(f"{Color.RED}[✗]{Color.RESET} {msg}")

def warn(msg):
    print(f"{Color.YELLOW}[⚠]{Color.RESET} {msg}")

# ─── 実行ユーティリティ ────────────────────────────────────────────────────
def run_command(cmd, check=False):
    """コマンド実行ラッパー"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        if check and result.returncode != 0:
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None

# ─── チェック項目 ──────────────────────────────────────────────────────────
class MigrationVerifier:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def add_result(self, category, name, status, detail=""):
        self.results.append({
            'category': category,
            'name': name,
            'status': status,
            'detail': detail
        })

        if status == 'PASS':
            self.passed += 1
        elif status == 'FAIL':
            self.failed += 1
        else:
            self.warnings += 1

    # ─── チェック: ユーザー ───────────────────────────────────────────────
    def check_users(self):
        log("ユーザー確認中...")

        # claude ユーザー確認
        result = run_command("id claude", check=False)
        if result:
            self.add_result('Users', 'claude ユーザー', 'PASS', result)
            success(f"claude ユーザー: {result.split(':')[0]}")
        else:
            self.add_result('Users', 'claude ユーザー', 'FAIL', 'ユーザーが見つかりません')
            error("claude ユーザーが見つかりません")

        # sudoers 確認
        sudoers_file = "/etc/sudoers.d/claude"
        if os.path.exists(sudoers_file):
            self.add_result('Users', 'sudoers', 'PASS', f'{sudoers_file} 存在')
            success(f"sudoers ファイル: {sudoers_file}")
        else:
            self.add_result('Users', 'sudoers', 'WARN', f'{sudoers_file} 見つかりません')
            warn(f"sudoers ファイルが見つかりません: {sudoers_file}")

    # ─── チェック: Node.js ────────────────────────────────────────────────
    def check_nodejs(self):
        log("Node.js確認中...")

        node_version = run_command("node --version", check=False)
        npm_version = run_command("npm --version", check=False)

        if node_version and npm_version:
            self.add_result('Node.js', 'Node.js', 'PASS', node_version)
            self.add_result('Node.js', 'npm', 'PASS', npm_version)
            success(f"Node.js: {node_version}")
            success(f"npm: {npm_version}")
        else:
            status = 'FAIL' if not (node_version and npm_version) else 'WARN'
            if not node_version:
                self.add_result('Node.js', 'Node.js', status, 'インストール確認失敗')
                error("Node.js が見つかりません")
            if not npm_version:
                self.add_result('Node.js', 'npm', status, 'インストール確認失敗')
                error("npm が見つかりません")

    # ─── チェック: Python ─────────────────────────────────────────────────
    def check_python(self):
        log("Python確認中...")

        python_version = run_command("python3 --version", check=False)
        pip_version = run_command("pip3 --version", check=False)

        if python_version:
            self.add_result('Python', 'Python3', 'PASS', python_version)
            success(f"Python: {python_version}")
        else:
            self.add_result('Python', 'Python3', 'FAIL', 'インストール確認失敗')
            error("Python3 が見つかりません")

        if pip_version:
            self.add_result('Python', 'pip3', 'PASS', pip_version)
            success(f"pip: {pip_version}")
        else:
            self.add_result('Python', 'pip3', 'WARN', 'インストール確認失敗')
            warn("pip3 が見つかりません")

    # ─── チェック: PM2 ────────────────────────────────────────────────────
    def check_pm2(self):
        log("PM2確認中...")

        pm2_version = run_command("npm list -g pm2 2>/dev/null | grep pm2", check=False)

        if pm2_version:
            self.add_result('PM2', 'PM2', 'PASS', pm2_version.strip())
            success(f"PM2: インストール済み")
        else:
            self.add_result('PM2', 'PM2', 'WARN', 'インストール確認失敗')
            warn("PM2 が見つかりません")

    # ─── チェック: Claude CLI ─────────────────────────────────────────────
    def check_claude_cli(self):
        log("Claude CLI確認中...")

        claude_version = run_command("claude --version 2>/dev/null", check=False)
        which_claude = run_command("which claude", check=False)

        if claude_version or which_claude:
            status = 'PASS'
            detail = claude_version or f"位置: {which_claude}"
        else:
            status = 'WARN'
            detail = 'インストール確認失敗'

        self.add_result('CLI', 'Claude Code', status, detail)
        if status == 'PASS':
            success(f"Claude CLI: {detail}")
        else:
            warn(f"Claude CLI: {detail}")

    # ─── チェック: Python パッケージ ───────────────────────────────────────
    def check_python_packages(self):
        log("Python パッケージ確認中...")

        required_packages = [
            'anthropic',
            'langgraph',
            'langchain_core',
            'fastapi',
            'discord',
            'mcp'
        ]

        for pkg in required_packages:
            try:
                __import__(pkg)
                self.add_result('Python Packages', pkg, 'PASS', 'インストール済み')
                success(f"パッケージ {pkg}: OK")
            except ImportError:
                self.add_result('Python Packages', pkg, 'WARN', 'インストールされていません')
                warn(f"パッケージ {pkg}: 見つかりません")

    # ─── チェック: systemd サービス ──────────────────────────────────────
    def check_systemd_services(self):
        log("systemd サービス確認中...")

        services = [
            'bushidan-main.service',
            'bushidan-console.service',
            'bushidan-discord.service',
            'pm2-claude.service'
        ]

        for service in services:
            status_output = run_command(f"systemctl is-active {service} 2>/dev/null", check=False)
            is_enabled = run_command(f"systemctl is-enabled {service} 2>/dev/null", check=False)

            if status_output in ['active', 'running']:
                status = 'PASS'
                detail = f"稼働中 (enabled: {is_enabled})"
            elif status_output == 'inactive':
                status = 'WARN'
                detail = f"停止中 (enabled: {is_enabled})"
            else:
                status = 'WARN'
                detail = '状態確認失敗'

            self.add_result('Systemd', service, status, detail)
            if status == 'PASS':
                success(f"{service}: {detail}")
            else:
                warn(f"{service}: {detail}")

    # ─── チェック: ポート ──────────────────────────────────────────────────
    def check_ports(self):
        log("ポート確認中...")

        ports = {
            '8067': 'bushidan-console',
            '8066': 'bushidan-mattermost',
        }

        for port, service in ports.items():
            netstat_output = run_command(f"ss -tlnp 2>/dev/null | grep :{port}", check=False)

            if netstat_output:
                self.add_result('Ports', f':{port}', 'PASS', f'{service} リッスン中')
                success(f"ポート {port}: リッスン中")
            else:
                self.add_result('Ports', f':{port}', 'WARN', '応答なし')
                warn(f"ポート {port}: 応答なし（サービス未起動の場合あり）")

    # ─── チェック: ネットワーク ───────────────────────────────────────────
    def check_network(self):
        log("ネットワーク確認中...")

        # IP アドレス
        ip_output = run_command("ip -4 addr show eth0 2>/dev/null | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){3}'", check=False)

        if ip_output:
            self.add_result('Network', 'IP address', 'PASS', ip_output)
            success(f"IP address: {ip_output}")
        else:
            self.add_result('Network', 'IP address', 'WARN', '取得失敗')
            warn("IP address の取得に失敗")

        # DNS
        dns_output = run_command("cat /etc/resolv.conf | grep nameserver | head -1", check=False)
        if dns_output:
            self.add_result('Network', 'DNS', 'PASS', dns_output)
            success(f"DNS: {dns_output}")
        else:
            self.add_result('Network', 'DNS', 'WARN', '設定されていません')

    # ─── チェック: 設定ファイル ───────────────────────────────────────────
    def check_config_files(self):
        log("設定ファイル確認中...")

        config_files = [
            ('/home/claude/.bashrc', '.bashrc'),
            ('/home/claude/.gitconfig', '.gitconfig'),
            ('/home/claude/.npmrc', '.npmrc'),
            ('/home/claude/.ssh/known_hosts', '.ssh/known_hosts'),
            ('/home/claude/.pm2/dump.pm2', '.pm2/dump.pm2'),
            ('/home/claude/ecosystem.config.cjs', 'ecosystem.config.cjs'),
        ]

        for filepath, name in config_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                self.add_result('Config Files', name, 'PASS', f'{size} bytes')
                success(f"ファイル {name}: 存在 ({size} bytes)")
            else:
                self.add_result('Config Files', name, 'WARN', 'ファイルが見つかりません')
                warn(f"ファイル {name}: 見つかりません")

    # ─── チェック: ディレクトリ ───────────────────────────────────────────
    def check_directories(self):
        log("ディレクトリ確認中...")

        directories = [
            ('/home/claude/Bushidan-Multi-Agent', 'Bushidan-Multi-Agent'),
            ('/home/claude/Bushidan', 'Bushidan'),
            ('/home/claude/.claude', '.claude'),
        ]

        for dirpath, name in directories:
            if os.path.isdir(dirpath):
                item_count = len(os.listdir(dirpath))
                self.add_result('Directories', name, 'PASS', f'{item_count} items')
                success(f"ディレクトリ {name}: 存在 ({item_count} items)")
            else:
                self.add_result('Directories', name, 'WARN', 'ディレクトリが見つかりません')
                warn(f"ディレクトリ {name}: 見つかりません（別途コピー予定の場合は問題ありません）")

    # ─── チェック: .env ファイル ──────────────────────────────────────────
    def check_env_file(self):
        log(".env ファイル確認中...")

        env_path = '/home/claude/Bushidan-Multi-Agent/.env'

        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                env_content = f.read()
                key_count = len([line for line in env_content.split('\n') if '=' in line and not line.startswith('#')])

            self.add_result('Environment', '.env', 'PASS', f'{key_count} キーが定義済み')
            success(f".env: 存在 ({key_count} キー)")
        else:
            self.add_result('Environment', '.env', 'WARN', 'ファイルが見つかりません')
            warn(f".env: 見つかりません（別途コピーしてください）")

    # ─── レポート生成 ─────────────────────────────────────────────────────
    def print_report(self):
        print("\n")
        print("╔" + "═" * 70 + "╗")
        print("║" + " " * 70 + "║")
        print("║" + "  武士団マルチエージェント - 移行確認レポート".center(70) + "║")
        print("║" + " " * 70 + "║")
        print("╚" + "═" * 70 + "╝")

        # カテゴリ別集計
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'PASS': 0, 'WARN': 0, 'FAIL': 0}
            categories[cat][result['status']] += 1

        # カテゴリ別レポート
        for category in sorted(categories.keys()):
            stats = categories[category]
            print(f"\n【{category}】")
            for result in [r for r in self.results if r['category'] == category]:
                status_str = f"{Color.GREEN}✓ PASS{Color.RESET}" if result['status'] == 'PASS' else \
                             f"{Color.RED}✗ FAIL{Color.RESET}" if result['status'] == 'FAIL' else \
                             f"{Color.YELLOW}⚠ WARN{Color.RESET}"
                print(f"  {result['name']:30s} {status_str:20s} {result['detail']}")

        # 総合統計
        print(f"\n{'─' * 72}")
        print(f"総合結果: {Color.GREEN}PASS: {self.passed}{Color.RESET} | "
              f"{Color.YELLOW}WARN: {self.warnings}{Color.RESET} | "
              f"{Color.RED}FAIL: {self.failed}{Color.RESET}")

        # 判定
        if self.failed == 0:
            if self.warnings > 0:
                print(f"\n{Color.YELLOW}[判定] 移行完了（警告あり）{Color.RESET}")
                print("一部のコンポーネントが確認できませんが、機能上の問題はない見込みです。")
            else:
                print(f"\n{Color.GREEN}[判定] 移行完了（すべてOK）{Color.RESET}")
        else:
            print(f"\n{Color.RED}[判定] 移行に問題があります{Color.RESET}")
            print("上記のFAIL項目を確認・対応してください。")

        print(f"\n生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─── メイン処理 ────────────────────────────────────────────────────────────
def main():
    verifier = MigrationVerifier()

    log("================== 移行確認開始 ==================")

    verifier.check_users()
    verifier.check_nodejs()
    verifier.check_python()
    verifier.check_pm2()
    verifier.check_claude_cli()
    verifier.check_python_packages()
    verifier.check_systemd_services()
    verifier.check_ports()
    verifier.check_network()
    verifier.check_config_files()
    verifier.check_directories()
    verifier.check_env_file()

    verifier.print_report()

    # 終了コード
    sys.exit(0 if verifier.failed == 0 else 1)

if __name__ == '__main__':
    main()
