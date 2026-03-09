#!/usr/bin/env python3
"""
システムヘルスチェックスクリプト

Usage:
    python maintenance/check_system_health.py
    python maintenance/check_system_health.py --json
    python maintenance/check_system_health.py --detailed
"""
import os
import sys
import json
import argparse
import asyncio
import psutil
import socket
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# プロジェクトルート
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env読み込み
load_dotenv(project_root / ".env")


class SystemHealthChecker:
    """システムヘルスチェッカー"""

    def __init__(self):
        self.results = []
        self.warnings = []
        self.errors = []

    def check_disk_space(self) -> Dict[str, Any]:
        """ディスク容量チェック"""
        try:
            disk = psutil.disk_usage(str(project_root))
            free_gb = disk.free / (1024 ** 3)
            total_gb = disk.total / (1024 ** 3)
            percent_used = disk.percent

            status = "healthy"
            if percent_used > 90:
                status = "critical"
                self.errors.append(f"ディスク使用率が危険域: {percent_used}%")
            elif percent_used > 80:
                status = "warning"
                self.warnings.append(f"ディスク使用率が高い: {percent_used}%")

            return {
                "name": "Disk Space",
                "status": status,
                "details": {
                    "total_gb": round(total_gb, 2),
                    "free_gb": round(free_gb, 2),
                    "used_percent": percent_used
                }
            }
        except Exception as e:
            self.errors.append(f"ディスクチェック失敗: {str(e)}")
            return {
                "name": "Disk Space",
                "status": "error",
                "details": {"error": str(e)}
            }

    def check_memory(self) -> Dict[str, Any]:
        """メモリ使用量チェック"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            total_gb = memory.total / (1024 ** 3)
            percent_used = memory.percent

            status = "healthy"
            if percent_used > 90:
                status = "critical"
                self.errors.append(f"メモリ使用率が危険域: {percent_used}%")
            elif percent_used > 80:
                status = "warning"
                self.warnings.append(f"メモリ使用率が高い: {percent_used}%")

            return {
                "name": "Memory",
                "status": status,
                "details": {
                    "total_gb": round(total_gb, 2),
                    "available_gb": round(available_gb, 2),
                    "used_percent": percent_used
                }
            }
        except Exception as e:
            self.errors.append(f"メモリチェック失敗: {str(e)}")
            return {
                "name": "Memory",
                "status": "error",
                "details": {"error": str(e)}
            }

    def check_cpu(self) -> Dict[str, Any]:
        """CPU使用率チェック"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            status = "healthy"
            if cpu_percent > 90:
                status = "warning"
                self.warnings.append(f"CPU使用率が高い: {cpu_percent}%")

            return {
                "name": "CPU",
                "status": status,
                "details": {
                    "cpu_count": cpu_count,
                    "cpu_percent": cpu_percent
                }
            }
        except Exception as e:
            self.errors.append(f"CPUチェック失敗: {str(e)}")
            return {
                "name": "CPU",
                "status": "error",
                "details": {"error": str(e)}
            }

    def check_llamacpp_server(self) -> Dict[str, Any]:
        """llama.cpp サーバーチェック"""
        try:
            response = requests.get("http://127.0.0.1:8080/health", timeout=5)
            if response.status_code == 200:
                return {
                    "name": "llama.cpp Server",
                    "status": "healthy",
                    "details": {"endpoint": "http://127.0.0.1:8080", "response": "OK"}
                }
            else:
                self.warnings.append(f"llama.cpp サーバー応答異常: HTTP {response.status_code}")
                return {
                    "name": "llama.cpp Server",
                    "status": "warning",
                    "details": {"endpoint": "http://127.0.0.1:8080", "status_code": response.status_code}
                }
        except requests.exceptions.ConnectionError:
            self.warnings.append("llama.cpp サーバーが停止中")
            return {
                "name": "llama.cpp Server",
                "status": "stopped",
                "details": {"endpoint": "http://127.0.0.1:8080", "message": "Connection refused"}
            }
        except Exception as e:
            self.errors.append(f"llama.cpp チェック失敗: {str(e)}")
            return {
                "name": "llama.cpp Server",
                "status": "error",
                "details": {"error": str(e)}
            }

    def check_environment_variables(self) -> Dict[str, Any]:
        """環境変数チェック"""
        required_vars = [
            "CLAUDE_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ]

        optional_vars = [
            "GROQ_API_KEY",
            "KIMI_API_KEY",
            "TAVILY_API_KEY",
            "DISCORD_BOT_TOKEN",
            "NOTION_API_KEY",
            "GITHUB_TOKEN",
        ]

        missing_required = []
        missing_optional = []

        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)

        for var in optional_vars:
            if not os.getenv(var):
                missing_optional.append(var)

        status = "healthy"
        details = {
            "required_set": len(required_vars) - len(missing_required),
            "required_total": len(required_vars),
            "optional_set": len(optional_vars) - len(missing_optional),
            "optional_total": len(optional_vars)
        }

        if missing_required:
            status = "critical"
            self.errors.append(f"必須環境変数が未設定: {', '.join(missing_required)}")
            details["missing_required"] = missing_required

        if missing_optional:
            self.warnings.append(f"オプション環境変数が未設定: {', '.join(missing_optional)}")
            details["missing_optional"] = missing_optional

        return {
            "name": "Environment Variables",
            "status": status,
            "details": details
        }

    def check_config_files(self) -> Dict[str, Any]:
        """設定ファイルチェック"""
        config_files = [
            "config/settings.yaml",
            ".env",
            "bushidan_config.yaml",
        ]

        missing = []
        found = []

        for config in config_files:
            path = project_root / config
            if path.exists():
                found.append(config)
            else:
                missing.append(config)

        status = "healthy"
        if missing:
            status = "warning"
            self.warnings.append(f"設定ファイルが見つからない: {', '.join(missing)}")

        return {
            "name": "Config Files",
            "status": status,
            "details": {
                "found": found,
                "missing": missing if missing else []
            }
        }

    def check_log_directory(self) -> Dict[str, Any]:
        """ログディレクトリチェック"""
        log_dir = project_root / "logs"

        if not log_dir.exists():
            self.warnings.append("logsディレクトリが存在しません")
            return {
                "name": "Log Directory",
                "status": "warning",
                "details": {"exists": False}
            }

        # ログファイルサイズチェック
        total_size = 0
        file_count = 0

        for log_file in log_dir.glob("*.log"):
            total_size += log_file.stat().st_size
            file_count += 1

        total_size_mb = total_size / (1024 ** 2)

        status = "healthy"
        if total_size_mb > 1000:  # 1GB以上
            status = "warning"
            self.warnings.append(f"ログサイズが大きい: {total_size_mb:.2f}MB")

        return {
            "name": "Log Directory",
            "status": status,
            "details": {
                "exists": True,
                "file_count": file_count,
                "total_size_mb": round(total_size_mb, 2)
            }
        }

    def check_memory_database(self) -> Dict[str, Any]:
        """メモリデータベースチェック"""
        memory_files = [
            "shogun_memory.jsonl",
            "bushidan_memory.jsonl",
        ]

        found_files = []
        file_sizes = {}

        for mem_file in memory_files:
            path = project_root / mem_file
            if path.exists():
                found_files.append(mem_file)
                size_kb = path.stat().st_size / 1024
                file_sizes[mem_file] = round(size_kb, 2)

        status = "healthy"
        if not found_files:
            status = "warning"
            self.warnings.append("メモリデータベースファイルが見つかりません")

        return {
            "name": "Memory Database",
            "status": status,
            "details": {
                "files": found_files,
                "sizes_kb": file_sizes
            }
        }

    def check_python_packages(self) -> Dict[str, Any]:
        """必須Pythonパッケージチェック"""
        required_packages = [
            "anthropic",
            "google.generativeai",
            "groq",
            "requests",
            "psutil",
            "discord",
            "dotenv",
        ]

        installed = []
        missing = []

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                installed.append(package)
            except ImportError:
                missing.append(package)

        status = "healthy"
        if missing:
            status = "warning"
            self.warnings.append(f"パッケージが未インストール: {', '.join(missing)}")

        return {
            "name": "Python Packages",
            "status": status,
            "details": {
                "installed": len(installed),
                "missing": missing if missing else []
            }
        }

    async def run_all_checks(self, detailed: bool = False) -> Dict[str, Any]:
        """すべてのチェックを実行"""
        checks = [
            self.check_disk_space,
            self.check_memory,
            self.check_cpu,
            self.check_llamacpp_server,
            self.check_environment_variables,
            self.check_config_files,
            self.check_log_directory,
            self.check_memory_database,
            self.check_python_packages,
        ]

        results = []
        for check in checks:
            result = check()
            results.append(result)

        # 全体的なステータス判定
        overall_status = "healthy"
        critical_count = sum(1 for r in results if r["status"] == "critical")
        error_count = sum(1 for r in results if r["status"] == "error")
        warning_count = sum(1 for r in results if r["status"] == "warning")

        if critical_count > 0 or error_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "healthy": sum(1 for r in results if r["status"] == "healthy"),
                "warning": warning_count,
                "critical": critical_count,
                "error": error_count,
                "total": len(results)
            },
            "checks": results,
            "warnings": self.warnings if detailed else None,
            "errors": self.errors if detailed else None
        }


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="システムヘルスチェック")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    parser.add_argument("--detailed", "-d", action="store_true", help="詳細情報を含める")
    args = parser.parse_args()

    checker = SystemHealthChecker()
    result = await checker.run_all_checks(detailed=args.detailed)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=" * 80)
        print("システムヘルスチェック")
        print("=" * 80)
        print(f"実行時刻: {result['timestamp']}")
        print(f"全体ステータス: {result['overall_status'].upper()}")
        print()

        print("=" * 80)
        print("サマリー")
        print("=" * 80)
        summary = result['summary']
        print(f"正常: {summary['healthy']} / 警告: {summary['warning']} / "
              f"重大: {summary['critical']} / エラー: {summary['error']}")
        print()

        print("=" * 80)
        print("チェック結果")
        print("=" * 80)

        for check in result['checks']:
            status_symbol = {
                "healthy": "✓",
                "warning": "⚠️",
                "critical": "❌",
                "error": "✗",
                "stopped": "⏸️"
            }.get(check['status'], "?")

            print(f"{status_symbol} {check['name']}: {check['status'].upper()}")

            if args.detailed and check.get('details'):
                for key, value in check['details'].items():
                    print(f"  - {key}: {value}")
            print()

        if result.get('warnings'):
            print("=" * 80)
            print("警告")
            print("=" * 80)
            for warning in result['warnings']:
                print(f"⚠️  {warning}")
            print()

        if result.get('errors'):
            print("=" * 80)
            print("エラー")
            print("=" * 80)
            for error in result['errors']:
                print(f"❌ {error}")
            print()

    # 終了コード
    if result['overall_status'] == "critical":
        sys.exit(2)
    elif result['overall_status'] == "warning":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
