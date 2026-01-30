"""
将軍システム v7.0 - Monthly Maintenance (反省会)
=================================================
月次メンテナンス機能:
- LLMバージョン確認・更新
- システムヘルスチェック
- MCPサーバー更新
- ログクリーンアップ
- コストレポート生成
"""

import subprocess
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import logging
import yaml
import httpx

logger = logging.getLogger(__name__)


class MaintenanceManager:
    """月次メンテナンス（反省会）マネージャー"""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).parent.parent
        self.config = self._load_config()
        self.reports_dir = self.base_dir / self.config.get("reports", {}).get(
            "directory", "reports/maintenance"
        )
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """メンテナンス設定を読み込む"""
        config_path = self.base_dir / "config" / "settings.yaml"
        if config_path.exists():
            with open(config_path) as f:
                settings = yaml.safe_load(f)
                return settings.get("maintenance", {})
        return {}

    def run_full_maintenance(self) -> dict[str, Any]:
        """全メンテナンス項目を実行"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": timestamp,
            "date": datetime.now().isoformat(),
            "checks": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "warnings": 0,
                "errors": 0,
            },
        }

        checks = [
            ("llm_versions", self.check_llm_versions),
            ("openvino_model", self.check_openvino_model),
            ("mcp_servers", self.check_mcp_servers),
            ("system_health", self.check_system_health),
            ("log_cleanup", self.cleanup_logs),
            ("cost_report", self.generate_cost_report),
        ]

        for check_id, check_func in checks:
            report["summary"]["total"] += 1
            try:
                result = check_func()
                report["checks"][check_id] = result
                if result.get("status") == "ok":
                    report["summary"]["passed"] += 1
                elif result.get("status") == "warning":
                    report["summary"]["warnings"] += 1
                else:
                    report["summary"]["errors"] += 1
            except Exception as e:
                report["checks"][check_id] = {
                    "status": "error",
                    "error": str(e),
                }
                report["summary"]["errors"] += 1
                logger.error(f"Maintenance check failed: {check_id}: {e}")

        # レポート保存
        report_path = self.reports_dir / f"maintenance_{timestamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # サマリーMarkdown生成
        self._generate_markdown_report(report, timestamp)

        logger.info(f"Maintenance complete: {report['summary']}")
        return report

    def check_llm_versions(self) -> dict[str, Any]:
        """クラウドLLMバージョンを確認"""
        result = {
            "name": "LLMバージョン確認",
            "status": "ok",
            "current": {},
            "latest": {},
            "updates_available": [],
        }

        # 現在使用中のモデル
        config_path = self.base_dir / "config" / "settings.yaml"
        if config_path.exists():
            with open(config_path) as f:
                settings = yaml.safe_load(f)
                cloud = settings.get("cloud", {})
                result["current"]["shogun"] = cloud.get("shogun", {}).get(
                    "api_model", "unknown"
                )
                result["current"]["karo"] = cloud.get("karo", {}).get(
                    "api_model", "unknown"
                )

        # Claude CLI バージョン確認
        try:
            proc = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                result["current"]["claude_cli"] = proc.stdout.strip()
        except Exception:
            result["current"]["claude_cli"] = "not installed"

        # npm で最新版確認
        try:
            proc = subprocess.run(
                ["npm", "view", "@anthropic-ai/claude-code", "version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                latest = proc.stdout.strip()
                result["latest"]["claude_cli"] = latest
                if result["current"].get("claude_cli", "").find(latest) == -1:
                    result["updates_available"].append(
                        f"claude-cli: {result['current'].get('claude_cli')} → {latest}"
                    )
        except Exception:
            pass

        if result["updates_available"]:
            result["status"] = "warning"
            result["message"] = f"{len(result['updates_available'])}件の更新があります"
        else:
            result["message"] = "全て最新です"

        return result

    def check_openvino_model(self) -> dict[str, Any]:
        """侍大将 OpenVINO モデル確認"""
        result = {
            "name": "OpenVINOモデル確認",
            "status": "ok",
            "taisho_reachable": False,
            "model_info": {},
        }

        config_path = self.base_dir / "config" / "settings.yaml"
        taisho_url = "http://192.168.1.11:11434"
        if config_path.exists():
            with open(config_path) as f:
                settings = yaml.safe_load(f)
                taisho_url = settings.get("taisho", {}).get("url", taisho_url)

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{taisho_url}/")
                if resp.status_code == 200:
                    data = resp.json()
                    result["taisho_reachable"] = True
                    result["model_info"] = data
                    result["message"] = f"侍大将稼働中: {data.get('model', 'unknown')}"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"侍大将に接続できません: {e}"

        return result

    def check_mcp_servers(self, auto_update: bool = False) -> dict[str, Any]:
        """MCPサーバー（足軽）の更新確認"""
        result = {
            "name": "MCPサーバー更新確認",
            "status": "ok",
            "servers": [],
            "updates_available": [],
            "updated": [],
        }

        mcp_packages = [
            "@modelcontextprotocol/server-filesystem",
            "@modelcontextprotocol/server-github",
            "@modelcontextprotocol/server-fetch",
            "@modelcontextprotocol/server-memory",
            "@modelcontextprotocol/server-postgres",
            "@modelcontextprotocol/server-puppeteer",
            "@modelcontextprotocol/server-brave-search",
            "@modelcontextprotocol/server-slack",
        ]

        # 現在のバージョンと最新版を比較
        try:
            proc = subprocess.run(
                ["npm", "outdated", "-g", "--json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.stdout:
                outdated = json.loads(proc.stdout)
                for pkg in mcp_packages:
                    if pkg in outdated:
                        info = outdated[pkg]
                        result["updates_available"].append({
                            "package": pkg,
                            "current": info.get("current"),
                            "latest": info.get("latest"),
                        })
        except Exception as e:
            logger.warning(f"Failed to check npm outdated: {e}")

        # 自動更新が有効な場合
        check_config = next(
            (c for c in self.config.get("checks", []) if c["id"] == "mcp_servers"),
            {},
        )
        if auto_update or check_config.get("auto_update", False):
            if result["updates_available"]:
                try:
                    proc = subprocess.run(
                        ["npm", "update", "-g"] + mcp_packages,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if proc.returncode == 0:
                        result["updated"] = [u["package"] for u in result["updates_available"]]
                        result["message"] = f"{len(result['updated'])}件更新しました"
                except Exception as e:
                    result["status"] = "warning"
                    result["message"] = f"更新に失敗: {e}"

        if result["updates_available"] and not result["updated"]:
            result["status"] = "warning"
            result["message"] = f"{len(result['updates_available'])}件の更新があります"
        elif not result["updates_available"]:
            result["message"] = "全て最新です"

        return result

    def check_system_health(self) -> dict[str, Any]:
        """システム全体のヘルスチェック"""
        result = {
            "name": "システムヘルスチェック",
            "status": "ok",
            "components": {},
        }

        # Python venv
        venv_path = self.base_dir / ".venv"
        result["components"]["python_venv"] = {
            "status": "ok" if venv_path.exists() else "error",
            "path": str(venv_path),
        }

        # 設定ファイル
        settings_path = self.base_dir / "config" / "settings.yaml"
        result["components"]["settings"] = {
            "status": "ok" if settings_path.exists() else "error",
            "path": str(settings_path),
        }

        # MCP設定
        mcp_config_path = self.base_dir / "config" / "mcp_config.json"
        result["components"]["mcp_config"] = {
            "status": "ok" if mcp_config_path.exists() else "warning",
            "path": str(mcp_config_path),
        }

        # 環境変数
        env_path = self.base_dir / ".env"
        result["components"]["env_file"] = {
            "status": "ok" if env_path.exists() else "warning",
            "path": str(env_path),
        }

        # ディスク使用量
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.base_dir)
            free_gb = free / (1024**3)
            result["components"]["disk"] = {
                "status": "ok" if free_gb > 5 else "warning",
                "free_gb": round(free_gb, 2),
            }
        except Exception:
            pass

        # エラーチェック
        errors = [
            k for k, v in result["components"].items() if v.get("status") == "error"
        ]
        warnings = [
            k for k, v in result["components"].items() if v.get("status") == "warning"
        ]

        if errors:
            result["status"] = "error"
            result["message"] = f"エラー: {', '.join(errors)}"
        elif warnings:
            result["status"] = "warning"
            result["message"] = f"警告: {', '.join(warnings)}"
        else:
            result["message"] = "全コンポーネント正常"

        return result

    def cleanup_logs(self) -> dict[str, Any]:
        """古いログファイルのクリーンアップ"""
        result = {
            "name": "ログクリーンアップ",
            "status": "ok",
            "deleted_count": 0,
            "deleted_size_mb": 0,
        }

        retention_days = 30
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted_size = 0
        deleted_count = 0

        # ログディレクトリ
        log_dirs = [
            self.base_dir / "logs",
            self.base_dir / "queue" / "reports",
            Path("/var/log"),
        ]

        for log_dir in log_dirs:
            if not log_dir.exists():
                continue
            try:
                for log_file in log_dir.glob("*.log"):
                    if log_file.stat().st_mtime < cutoff.timestamp():
                        size = log_file.stat().st_size
                        log_file.unlink()
                        deleted_count += 1
                        deleted_size += size
            except Exception as e:
                logger.warning(f"Failed to cleanup {log_dir}: {e}")

        result["deleted_count"] = deleted_count
        result["deleted_size_mb"] = round(deleted_size / (1024 * 1024), 2)
        result["message"] = f"{deleted_count}ファイル ({result['deleted_size_mb']}MB) 削除"

        return result

    def generate_cost_report(self) -> dict[str, Any]:
        """月次コストレポート生成"""
        result = {
            "name": "月次コストレポート",
            "status": "ok",
            "period": {
                "start": (datetime.now().replace(day=1) - timedelta(days=1)).replace(
                    day=1
                ).strftime("%Y-%m-%d"),
                "end": (datetime.now().replace(day=1) - timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
            },
            "costs": {
                "pro_subscription_yen": 3000,
                "electricity_yen": 800,
                "api_usage_yen": 0,
            },
            "usage": {
                "shogun_calls": 0,
                "karo_calls": 0,
                "taisho_calls": 0,
            },
        }

        # ダッシュボードからコストデータ取得
        stats_path = self.base_dir / "status" / "stats.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                    result["costs"]["api_usage_yen"] = stats.get("api_cost_yen", 0)
                    result["usage"] = stats.get("usage", result["usage"])
            except Exception:
                pass

        total = sum(result["costs"].values())
        result["costs"]["total_yen"] = total
        result["message"] = f"月間コスト: ¥{total:,}"

        # 予算チェック
        budget = 6000
        if total > budget:
            result["status"] = "warning"
            result["message"] += f" (予算超過: +¥{total - budget:,})"

        return result

    def _generate_markdown_report(self, report: dict, timestamp: str) -> None:
        """Markdownレポート生成"""
        md_path = self.reports_dir / f"maintenance_{timestamp}.md"

        lines = [
            f"# 将軍システム 月次メンテナンス報告",
            f"",
            f"**実行日時:** {report['date']}",
            f"",
            f"## サマリー",
            f"",
            f"| 項目 | 件数 |",
            f"|------|------|",
            f"| 総チェック | {report['summary']['total']} |",
            f"| 正常 | {report['summary']['passed']} |",
            f"| 警告 | {report['summary']['warnings']} |",
            f"| エラー | {report['summary']['errors']} |",
            f"",
            f"## 詳細",
            f"",
        ]

        for check_id, check_result in report["checks"].items():
            status_emoji = {
                "ok": "✅",
                "warning": "⚠️",
                "error": "❌",
            }.get(check_result.get("status"), "❓")

            lines.append(f"### {status_emoji} {check_result.get('name', check_id)}")
            lines.append(f"")
            lines.append(f"**ステータス:** {check_result.get('status', 'unknown')}")
            if check_result.get("message"):
                lines.append(f"")
                lines.append(f"**結果:** {check_result['message']}")
            lines.append(f"")

        # 次回アクション
        lines.extend([
            f"## 次回アクション",
            f"",
        ])

        # 更新が必要な項目
        updates_needed = []
        for check_id, check_result in report["checks"].items():
            if check_result.get("updates_available"):
                updates_needed.extend(check_result["updates_available"])

        if updates_needed:
            lines.append(f"### 更新が必要な項目")
            lines.append(f"")
            for update in updates_needed:
                if isinstance(update, dict):
                    lines.append(f"- {update.get('package', update)}: {update.get('current')} → {update.get('latest')}")
                else:
                    lines.append(f"- {update}")
            lines.append(f"")
        else:
            lines.append(f"更新不要。全て最新状態です。")
            lines.append(f"")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report saved: {md_path}")

    def get_next_maintenance_date(self) -> datetime:
        """次回メンテナンス日時を取得"""
        schedule = self.config.get("schedule", {})
        day = schedule.get("day_of_month", 1)
        hour = schedule.get("hour", 9)
        minute = schedule.get("minute", 0)

        now = datetime.now()
        next_date = now.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)

        if next_date <= now:
            # 来月の同日
            if now.month == 12:
                next_date = next_date.replace(year=now.year + 1, month=1)
            else:
                next_date = next_date.replace(month=now.month + 1)

        return next_date

    def list_reports(self, limit: int = 10) -> list[dict]:
        """過去のレポート一覧"""
        reports = []
        for report_file in sorted(
            self.reports_dir.glob("maintenance_*.json"), reverse=True
        )[:limit]:
            try:
                with open(report_file) as f:
                    data = json.load(f)
                    reports.append({
                        "file": report_file.name,
                        "date": data.get("date"),
                        "summary": data.get("summary"),
                    })
            except Exception:
                pass
        return reports


def run_maintenance() -> dict:
    """メンテナンスを実行（CLIから呼び出し用）"""
    manager = MaintenanceManager()
    return manager.run_full_maintenance()
