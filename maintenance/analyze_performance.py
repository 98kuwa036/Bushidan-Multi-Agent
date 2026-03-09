#!/usr/bin/env python3
"""
パフォーマンス分析スクリプト

ログファイルからタスク処理時間、ルーティング決定、コストを分析

Usage:
    python maintenance/analyze_performance.py
    python maintenance/analyze_performance.py --days 7
    python maintenance/analyze_performance.py --json
"""
import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any, List

# プロジェクトルート
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PerformanceAnalyzer:
    """パフォーマンス分析クラス"""

    def __init__(self, log_dir: Path, days: int = 30):
        self.log_dir = log_dir
        self.days = days
        self.cutoff_date = datetime.now() - timedelta(days=days)

        # 統計データ
        self.task_times = defaultdict(list)  # complexity -> [times]
        self.routing_decisions = defaultdict(int)  # route -> count
        self.model_usage = defaultdict(int)  # model -> count
        self.costs = []
        self.errors = defaultdict(int)  # error_type -> count

    def parse_log_files(self):
        """ログファイルを解析"""
        if not self.log_dir.exists():
            print(f"警告: ログディレクトリが存在しません: {self.log_dir}")
            return

        for log_file in self.log_dir.glob("*.log"):
            self._parse_file(log_file)

    def _parse_file(self, file_path: Path):
        """個別のログファイルを解析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self._parse_line(line)
        except Exception as e:
            print(f"警告: ファイル解析エラー {file_path}: {e}")

    def _parse_line(self, line: str):
        """ログ行を解析"""
        # タスク完了時間
        # 例: "✅ 任務完了: 12.5秒"
        time_match = re.search(r'✅ 任務完了: ([\d.]+)秒', line)
        if time_match:
            elapsed_time = float(time_match.group(1))

            # 複雑度を推定
            complexity = "unknown"
            if "簡易" in line or "SIMPLE" in line:
                complexity = "simple"
            elif "MEDIUM" in line or "中程度" in line:
                complexity = "medium"
            elif "COMPLEX" in line or "複雑" in line:
                complexity = "complex"
            elif "STRATEGIC" in line or "戦略" in line:
                complexity = "strategic"

            self.task_times[complexity].append(elapsed_time)

        # ルーティング決定
        # 例: "⚡ 簡易任務 → Groq即応"
        if "Groq" in line and "即応" in line:
            self.routing_decisions["groq"] += 1
        elif "家老" in line and "采配" in line:
            self.routing_decisions["karo"] += 1
        elif "将軍自ら" in line:
            self.routing_decisions["shogun_direct"] += 1
        elif "軍師" in line or "Gunshi" in line:
            self.routing_decisions["gunshi"] += 1

        # モデル使用
        if "Claude" in line or "Sonnet" in line:
            self.model_usage["claude_sonnet"] += 1
        if "Opus" in line:
            self.model_usage["claude_opus"] += 1
        if "Gemini" in line:
            self.model_usage["gemini"] += 1
        if "Qwen" in line:
            self.model_usage["qwen"] += 1
        if "Groq" in line:
            self.model_usage["groq"] += 1
        if "Kimi" in line:
            self.model_usage["kimi"] += 1

        # エラー
        if "❌" in line or "ERROR" in line:
            if "API" in line:
                self.errors["api_error"] += 1
            elif "timeout" in line.lower():
                self.errors["timeout"] += 1
            elif "rate limit" in line.lower() or "レート制限" in line:
                self.errors["rate_limit"] += 1
            else:
                self.errors["other_error"] += 1

        # コスト（将来実装）
        cost_match = re.search(r'Cost: ¥([\d.]+)', line)
        if cost_match:
            cost = float(cost_match.group(1))
            self.costs.append(cost)

    def generate_report(self) -> Dict[str, Any]:
        """分析レポートを生成"""
        report = {
            "period": {
                "days": self.days,
                "start_date": self.cutoff_date.isoformat(),
                "end_date": datetime.now().isoformat()
            },
            "task_performance": self._analyze_task_performance(),
            "routing_statistics": self._analyze_routing(),
            "model_usage": dict(self.model_usage),
            "error_statistics": dict(self.errors),
            "cost_analysis": self._analyze_costs()
        }

        return report

    def _analyze_task_performance(self) -> Dict[str, Any]:
        """タスクパフォーマンス分析"""
        performance = {}

        for complexity, times in self.task_times.items():
            if not times:
                continue

            performance[complexity] = {
                "count": len(times),
                "avg_seconds": round(sum(times) / len(times), 2),
                "min_seconds": round(min(times), 2),
                "max_seconds": round(max(times), 2),
                "median_seconds": round(sorted(times)[len(times) // 2], 2)
            }

        return performance

    def _analyze_routing(self) -> Dict[str, Any]:
        """ルーティング統計分析"""
        total = sum(self.routing_decisions.values())

        if total == 0:
            return {"total": 0}

        routing = {
            "total": total,
            "breakdown": {}
        }

        for route, count in self.routing_decisions.items():
            routing["breakdown"][route] = {
                "count": count,
                "percentage": round((count / total) * 100, 2)
            }

        return routing

    def _analyze_costs(self) -> Dict[str, Any]:
        """コスト分析"""
        if not self.costs:
            return {
                "total_yen": 0,
                "avg_per_task_yen": 0,
                "note": "コストデータが記録されていません"
            }

        return {
            "total_yen": round(sum(self.costs), 2),
            "avg_per_task_yen": round(sum(self.costs) / len(self.costs), 2),
            "min_yen": round(min(self.costs), 2),
            "max_yen": round(max(self.costs), 2)
        }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="パフォーマンス分析")
    parser.add_argument("--days", type=int, default=30, help="分析対象期間（日数）")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    parser.add_argument("--log-dir", type=str, help="ログディレクトリパス")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else (project_root / "logs")

    analyzer = PerformanceAnalyzer(log_dir, days=args.days)
    analyzer.parse_log_files()
    report = analyzer.generate_report()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("=" * 80)
        print("パフォーマンス分析レポート")
        print("=" * 80)
        print(f"分析期間: {args.days}日間")
        print(f"開始: {report['period']['start_date']}")
        print(f"終了: {report['period']['end_date']}")
        print()

        # タスクパフォーマンス
        print("=" * 80)
        print("タスク処理時間統計")
        print("=" * 80)
        perf = report['task_performance']
        if perf:
            for complexity, stats in perf.items():
                print(f"\n{complexity.upper()}:")
                print(f"  タスク数: {stats['count']}")
                print(f"  平均: {stats['avg_seconds']}秒")
                print(f"  最小: {stats['min_seconds']}秒")
                print(f"  最大: {stats['max_seconds']}秒")
                print(f"  中央値: {stats['median_seconds']}秒")
        else:
            print("データなし")
        print()

        # ルーティング統計
        print("=" * 80)
        print("ルーティング統計")
        print("=" * 80)
        routing = report['routing_statistics']
        if routing.get('total', 0) > 0:
            print(f"総タスク数: {routing['total']}")
            print()
            for route, stats in routing['breakdown'].items():
                print(f"{route}: {stats['count']} ({stats['percentage']}%)")
        else:
            print("データなし")
        print()

        # モデル使用統計
        print("=" * 80)
        print("モデル使用統計")
        print("=" * 80)
        if report['model_usage']:
            for model, count in sorted(report['model_usage'].items(), key=lambda x: x[1], reverse=True):
                print(f"{model}: {count}回")
        else:
            print("データなし")
        print()

        # エラー統計
        print("=" * 80)
        print("エラー統計")
        print("=" * 80)
        if report['error_statistics']:
            for error_type, count in sorted(report['error_statistics'].items(), key=lambda x: x[1], reverse=True):
                print(f"{error_type}: {count}回")
        else:
            print("エラーなし")
        print()

        # コスト分析
        print("=" * 80)
        print("コスト分析")
        print("=" * 80)
        cost = report['cost_analysis']
        if cost.get('total_yen', 0) > 0:
            print(f"総コスト: ¥{cost['total_yen']}")
            print(f"タスクあたり平均: ¥{cost['avg_per_task_yen']}")
            print(f"最小: ¥{cost['min_yen']}")
            print(f"最大: ¥{cost['max_yen']}")
        else:
            print(cost.get('note', 'データなし'))
        print()


if __name__ == "__main__":
    main()
