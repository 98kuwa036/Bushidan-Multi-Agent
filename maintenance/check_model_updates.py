#!/usr/bin/env python3
"""
モデル更新チェッカー

最新のモデル情報を取得し、現在の設定と比較

Usage:
    python maintenance/check_model_updates.py
    python maintenance/check_model_updates.py --json
    python maintenance/check_model_updates.py --auto-update
"""
import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# プロジェクトルート
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env読み込み
load_dotenv(project_root / ".env")


class ModelUpdateChecker:
    """モデル更新チェッカー"""

    def __init__(self):
        self.current_models = self._load_current_models()
        self.latest_models = {}
        self.updates_available = []
        self.deprecated_models = []

    def _load_current_models(self) -> Dict[str, str]:
        """現在使用中のモデルを取得"""
        import yaml

        settings_path = project_root / "config" / "settings.yaml"

        if not settings_path.exists():
            return {}

        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)

            return {
                "claude_sonnet": settings.get("shogun", {}).get("model", "unknown"),
                "claude_opus": settings.get("shogun", {}).get("opus_model", "unknown"),
                "gemini_flash": settings.get("taisho", {}).get("final_defense", {}).get("model", "unknown"),
                "groq_llama": settings.get("karo", {}).get("dynamic_selection", {}).get("groq", {}).get("model", "unknown"),
            }
        except Exception as e:
            print(f"警告: 設定ファイル読み込みエラー: {e}")
            return {}

    def check_claude_models(self):
        """Claudeモデルの最新情報を確認"""
        print("Claude モデルチェック中...")

        try:
            # Anthropic公式ドキュメントから最新モデル情報を取得
            # 実際のAPIエンドポイントは存在しないため、既知の最新モデルを返す
            self.latest_models["claude_sonnet"] = "claude-sonnet-4-6"
            self.latest_models["claude_opus"] = "claude-opus-4-6"

            current_sonnet = self.current_models.get("claude_sonnet", "")
            current_opus = self.current_models.get("claude_opus", "")

            if current_sonnet != self.latest_models["claude_sonnet"]:
                self.updates_available.append({
                    "model": "Claude Sonnet",
                    "current": current_sonnet,
                    "latest": self.latest_models["claude_sonnet"],
                    "recommendation": "最新版へのアップデート推奨"
                })

            if current_opus != self.latest_models["claude_opus"]:
                self.updates_available.append({
                    "model": "Claude Opus",
                    "current": current_opus,
                    "latest": self.latest_models["claude_opus"],
                    "recommendation": "最新版へのアップデート推奨"
                })

            # 非推奨モデルチェック
            deprecated = ["claude-3-opus", "claude-3-sonnet", "claude-2"]
            if any(dep in current_sonnet or dep in current_opus for dep in deprecated):
                self.deprecated_models.append({
                    "model": "Claude 3/2系",
                    "status": "deprecated",
                    "action": "Claude 4.6へ移行してください"
                })

            print("✓ Claude モデルチェック完了")

        except Exception as e:
            print(f"❌ Claude モデルチェックエラー: {e}")

    def check_gemini_models(self):
        """Geminiモデルの最新情報を確認"""
        print("Gemini モデルチェック中...")

        try:
            # Gemini最新モデル
            self.latest_models["gemini_flash"] = "gemini-3-flash-preview"

            current_gemini = self.current_models.get("gemini_flash", "")

            if current_gemini != self.latest_models["gemini_flash"]:
                self.updates_available.append({
                    "model": "Gemini Flash",
                    "current": current_gemini,
                    "latest": self.latest_models["gemini_flash"],
                    "recommendation": "最新版へのアップデート推奨"
                })

            # 非推奨モデルチェック
            if "gemini-2.0" in current_gemini or "gemini-1." in current_gemini:
                self.deprecated_models.append({
                    "model": "Gemini 2.0/1.x系",
                    "status": "deprecated",
                    "action": "Gemini 3 Flashへ移行してください（2026年6月廃止予定）"
                })

            print("✓ Gemini モデルチェック完了")

        except Exception as e:
            print(f"❌ Gemini モデルチェックエラー: {e}")

    def check_other_models(self):
        """その他のモデル情報を確認"""
        print("その他のモデルチェック中...")

        try:
            # Groq Llama
            self.latest_models["groq_llama"] = "llama-3.3-70b-versatile"

            current_groq = self.current_models.get("groq_llama", "")

            if current_groq and current_groq != self.latest_models["groq_llama"]:
                self.updates_available.append({
                    "model": "Groq Llama",
                    "current": current_groq,
                    "latest": self.latest_models["groq_llama"],
                    "recommendation": "最新版へのアップデート推奨"
                })

            print("✓ その他のモデルチェック完了")

        except Exception as e:
            print(f"❌ その他のモデルチェックエラー: {e}")

    def generate_report(self) -> Dict[str, Any]:
        """レポート生成"""
        return {
            "current_models": self.current_models,
            "latest_models": self.latest_models,
            "updates_available": self.updates_available,
            "deprecated_models": self.deprecated_models,
            "summary": {
                "up_to_date": len(self.updates_available) == 0,
                "updates_count": len(self.updates_available),
                "deprecated_count": len(self.deprecated_models)
            }
        }

    def print_report(self, report: Dict[str, Any]):
        """レポート表示"""
        print("\n" + "=" * 80)
        print("モデル更新チェック結果")
        print("=" * 80)

        print("\n現在使用中のモデル:")
        print("-" * 80)
        for model_name, version in report["current_models"].items():
            print(f"  {model_name}: {version}")

        if report["updates_available"]:
            print("\n" + "=" * 80)
            print("利用可能なアップデート")
            print("=" * 80)
            for update in report["updates_available"]:
                print(f"\n📦 {update['model']}")
                print(f"  現在: {update['current']}")
                print(f"  最新: {update['latest']}")
                print(f"  推奨: {update['recommendation']}")
        else:
            print("\n✓ すべてのモデルが最新です")

        if report["deprecated_models"]:
            print("\n" + "=" * 80)
            print("⚠️  非推奨モデル")
            print("=" * 80)
            for deprecated in report["deprecated_models"]:
                print(f"\n⚠️  {deprecated['model']}")
                print(f"  ステータス: {deprecated['status']}")
                print(f"  アクション: {deprecated['action']}")

        print("\n" + "=" * 80)
        print("サマリー")
        print("=" * 80)
        summary = report["summary"]
        print(f"最新状態: {'✓ はい' if summary['up_to_date'] else '✗ いいえ'}")
        print(f"利用可能なアップデート: {summary['updates_count']}")
        print(f"非推奨モデル: {summary['deprecated_count']}")
        print()

    def run(self) -> Dict[str, Any]:
        """チェック実行"""
        print("=" * 80)
        print("モデル更新チェック")
        print("=" * 80)
        print()

        self.check_claude_models()
        self.check_gemini_models()
        self.check_other_models()

        return self.generate_report()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="モデル更新チェッカー")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    parser.add_argument("--auto-update", action="store_true", help="自動更新（未実装）")
    args = parser.parse_args()

    checker = ModelUpdateChecker()
    report = checker.run()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        checker.print_report(report)

    if args.auto_update:
        print("\n💡 自動更新機能は未実装です。")
        print("   手動で config/settings.yaml を更新してください。")

    # 終了コード
    if report["deprecated_models"]:
        sys.exit(2)  # 非推奨モデルあり
    elif report["updates_available"]:
        sys.exit(1)  # アップデートあり
    else:
        sys.exit(0)  # 最新


if __name__ == "__main__":
    main()
