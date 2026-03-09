#!/usr/bin/env python3
"""
メモリデータベースクリーンアップスクリプト

古いエントリの削除、重複データの統合、インデックス再構築

Usage:
    python maintenance/cleanup_memory.py --dry-run
    python maintenance/cleanup_memory.py --days 90
    python maintenance/cleanup_memory.py --backup
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# プロジェクトルート
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MemoryCleanup:
    """メモリデータベースクリーンアップ"""

    def __init__(self, days: int = 90, dry_run: bool = False):
        self.days = days
        self.dry_run = dry_run
        self.cutoff_date = datetime.now() - timedelta(days=days)

        self.memory_files = [
            project_root / "shogun_memory.jsonl",
            project_root / "bushidan_memory.jsonl",
        ]

        self.stats = {
            "total_entries": 0,
            "removed_old": 0,
            "removed_duplicates": 0,
            "final_entries": 0
        }

    def backup_files(self):
        """メモリファイルをバックアップ"""
        backup_dir = project_root / "backups" / "memory"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for mem_file in self.memory_files:
            if mem_file.exists():
                backup_path = backup_dir / f"{mem_file.stem}_{timestamp}.jsonl"
                shutil.copy2(mem_file, backup_path)
                print(f"✓ バックアップ作成: {backup_path}")

    def load_entries(self, file_path: Path) -> List[Dict[str, Any]]:
        """JSONLファイルからエントリを読み込み"""
        entries = []

        if not file_path.exists():
            return entries

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"警告: ファイル読み込みエラー {file_path}: {e}")

        return entries

    def save_entries(self, file_path: Path, entries: List[Dict[str, Any]]):
        """エントリをJSONLファイルに保存"""
        if self.dry_run:
            print(f"[DRY RUN] {len(entries)}エントリを{file_path}に保存（実際には保存されません）")
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"✓ 保存完了: {file_path} ({len(entries)}エントリ)")
        except Exception as e:
            print(f"❌ 保存失敗 {file_path}: {e}")

    def remove_old_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """古いエントリを削除"""
        filtered = []

        for entry in entries:
            timestamp_str = entry.get('timestamp')

            if not timestamp_str:
                # タイムスタンプがない場合は保持
                filtered.append(entry)
                continue

            try:
                # ISO形式またはUNIXタイムスタンプをサポート
                if isinstance(timestamp_str, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp_str)
                else:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                if timestamp >= self.cutoff_date:
                    filtered.append(entry)
                else:
                    self.stats["removed_old"] += 1
            except Exception:
                # パース失敗時は保持
                filtered.append(entry)

        return filtered

    def remove_duplicates(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重複エントリを削除"""
        seen = set()
        unique = []

        for entry in entries:
            # エントリの一意性を判定（content フィールドをキーとする）
            content = entry.get('content') or entry.get('task') or entry.get('category')

            if not content:
                unique.append(entry)
                continue

            # 辞書の場合は文字列化
            if isinstance(content, dict):
                content = json.dumps(content, sort_keys=True, ensure_ascii=False)

            key = str(content)[:200]  # 最初の200文字で判定

            if key not in seen:
                seen.add(key)
                unique.append(entry)
            else:
                self.stats["removed_duplicates"] += 1

        return unique

    def cleanup_file(self, file_path: Path):
        """個別ファイルのクリーンアップ"""
        if not file_path.exists():
            print(f"⚠️  ファイルが存在しません: {file_path}")
            return

        print(f"\n処理中: {file_path.name}")
        print("-" * 60)

        # エントリ読み込み
        entries = self.load_entries(file_path)
        self.stats["total_entries"] += len(entries)
        print(f"読み込み: {len(entries)}エントリ")

        # 古いエントリ削除
        entries = self.remove_old_entries(entries)
        print(f"古いエントリ削除後: {len(entries)}エントリ")

        # 重複削除
        entries = self.remove_duplicates(entries)
        print(f"重複削除後: {len(entries)}エントリ")

        self.stats["final_entries"] += len(entries)

        # 保存
        self.save_entries(file_path, entries)

    def run(self, backup: bool = False):
        """クリーンアップ実行"""
        print("=" * 80)
        print("メモリデータベース クリーンアップ")
        print("=" * 80)
        print(f"カットオフ日: {self.cutoff_date.strftime('%Y-%m-%d')} ({self.days}日以前)")
        print(f"モード: {'DRY RUN（実際には削除されません）' if self.dry_run else '実行'}")
        print()

        # バックアップ
        if backup and not self.dry_run:
            print("バックアップ作成中...")
            self.backup_files()
            print()

        # 各ファイルをクリーンアップ
        for mem_file in self.memory_files:
            self.cleanup_file(mem_file)

        # 統計表示
        print("\n" + "=" * 80)
        print("クリーンアップ統計")
        print("=" * 80)
        print(f"総エントリ数: {self.stats['total_entries']}")
        print(f"古いエントリ削除: {self.stats['removed_old']}")
        print(f"重複削除: {self.stats['removed_duplicates']}")
        print(f"最終エントリ数: {self.stats['final_entries']}")
        print(f"削減率: {((self.stats['total_entries'] - self.stats['final_entries']) / self.stats['total_entries'] * 100):.1f}%" if self.stats['total_entries'] > 0 else "N/A")
        print()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="メモリデータベースクリーンアップ")
    parser.add_argument("--days", type=int, default=90, help="保持期間（日数、デフォルト: 90）")
    parser.add_argument("--dry-run", action="store_true", help="実行せず、結果のみ表示")
    parser.add_argument("--backup", action="store_true", help="クリーンアップ前にバックアップ作成")
    args = parser.parse_args()

    cleanup = MemoryCleanup(days=args.days, dry_run=args.dry_run)
    cleanup.run(backup=args.backup)

    if args.dry_run:
        print("💡 実際にクリーンアップを実行する場合は --dry-run を外してください")
        print("💡 バックアップを作成する場合は --backup を追加してください")


if __name__ == "__main__":
    main()
