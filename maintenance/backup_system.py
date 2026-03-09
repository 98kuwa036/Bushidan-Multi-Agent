#!/usr/bin/env python3
"""
システムバックアップスクリプト

設定ファイル、メモリDB、重要データのバックアップ

Usage:
    python maintenance/backup_system.py
    python maintenance/backup_system.py --full
    python maintenance/backup_system.py --destination /path/to/backup
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# プロジェクトルート
project_root = Path(__file__).parent.parent


class SystemBackup:
    """システムバックアップ"""

    def __init__(self, full: bool = False, destination: str = None):
        self.full = full
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if destination:
            self.backup_root = Path(destination) / f"bushidan_backup_{self.timestamp}"
        else:
            self.backup_root = project_root / "backups" / f"system_{self.timestamp}"

        self.stats = {
            "files_backed_up": 0,
            "total_size_mb": 0,
            "errors": 0
        }

    def ensure_backup_directory(self):
        """バックアップディレクトリを作成"""
        self.backup_root.mkdir(parents=True, exist_ok=True)
        print(f"バックアップ先: {self.backup_root}")

    def backup_config_files(self):
        """設定ファイルをバックアップ"""
        print("\n" + "=" * 60)
        print("設定ファイルのバックアップ")
        print("=" * 60)

        config_files = [
            "config/settings.yaml",
            "bushidan_config.yaml",
            "config/mcp_config.yaml",
            "config/mcp_status_config.yaml",
            "config/interactive_config.yaml",
            ".env.example",  # .envは除外（機密情報のため）
        ]

        dest_dir = self.backup_root / "config"
        dest_dir.mkdir(exist_ok=True)

        for config_path in config_files:
            source = project_root / config_path
            if source.exists():
                dest = dest_dir / source.name
                try:
                    shutil.copy2(source, dest)
                    size_kb = source.stat().st_size / 1024
                    self.stats["files_backed_up"] += 1
                    self.stats["total_size_mb"] += size_kb / 1024
                    print(f"✓ {config_path} ({size_kb:.2f}KB)")
                except Exception as e:
                    print(f"❌ {config_path}: {e}")
                    self.stats["errors"] += 1
            else:
                print(f"⚠️  {config_path} (存在しません)")

    def backup_memory_databases(self):
        """メモリデータベースをバックアップ"""
        print("\n" + "=" * 60)
        print("メモリデータベースのバックアップ")
        print("=" * 60)

        memory_files = [
            "shogun_memory.jsonl",
            "bushidan_memory.jsonl",
        ]

        dest_dir = self.backup_root / "memory"
        dest_dir.mkdir(exist_ok=True)

        for mem_file in memory_files:
            source = project_root / mem_file
            if source.exists():
                dest = dest_dir / source.name
                try:
                    shutil.copy2(source, dest)
                    size_kb = source.stat().st_size / 1024
                    self.stats["files_backed_up"] += 1
                    self.stats["total_size_mb"] += size_kb / 1024
                    print(f"✓ {mem_file} ({size_kb:.2f}KB)")
                except Exception as e:
                    print(f"❌ {mem_file}: {e}")
                    self.stats["errors"] += 1
            else:
                print(f"⚠️  {mem_file} (存在しません)")

    def backup_documentation(self):
        """ドキュメントをバックアップ"""
        print("\n" + "=" * 60)
        print("ドキュメントのバックアップ")
        print("=" * 60)

        doc_files = [
            "README.md",
            "DISCORD_SETUP.md",
            "IMPLEMENTATION_GUIDE.md",
            "INTERACTIVE_MODE.md",
        ]

        dest_dir = self.backup_root / "docs"
        dest_dir.mkdir(exist_ok=True)

        for doc_file in doc_files:
            source = project_root / doc_file
            if source.exists():
                dest = dest_dir / source.name
                try:
                    shutil.copy2(source, dest)
                    size_kb = source.stat().st_size / 1024
                    self.stats["files_backed_up"] += 1
                    self.stats["total_size_mb"] += size_kb / 1024
                    print(f"✓ {doc_file} ({size_kb:.2f}KB)")
                except Exception as e:
                    print(f"❌ {doc_file}: {e}")
                    self.stats["errors"] += 1
            else:
                print(f"⚠️  {doc_file} (存在しません)")

    def backup_source_code(self):
        """ソースコードをバックアップ（フルバックアップ時のみ）"""
        if not self.full:
            return

        print("\n" + "=" * 60)
        print("ソースコードのバックアップ（フルバックアップ）")
        print("=" * 60)

        source_dirs = [
            "core",
            "utils",
            "bushidan",
            "maintenance",
            "interfaces",
        ]

        for source_dir in source_dirs:
            source_path = project_root / source_dir
            if source_path.exists():
                dest_path = self.backup_root / "source" / source_dir
                try:
                    shutil.copytree(source_path, dest_path,
                                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                    # ファイル数とサイズを計算
                    file_count = sum(1 for _ in dest_path.rglob("*") if _.is_file())
                    dir_size = sum(f.stat().st_size for f in dest_path.rglob("*") if f.is_file())
                    size_mb = dir_size / (1024 ** 2)
                    self.stats["files_backed_up"] += file_count
                    self.stats["total_size_mb"] += size_mb
                    print(f"✓ {source_dir}/ ({file_count}ファイル, {size_mb:.2f}MB)")
                except Exception as e:
                    print(f"❌ {source_dir}/: {e}")
                    self.stats["errors"] += 1
            else:
                print(f"⚠️  {source_dir}/ (存在しません)")

    def create_backup_manifest(self):
        """バックアップマニフェストを作成"""
        manifest_path = self.backup_root / "MANIFEST.txt"

        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                f.write(f"Bushidan Multi-Agent System Backup\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"Backup Date: {datetime.now().isoformat()}\n")
                f.write(f"Backup Type: {'Full' if self.full else 'Essential'}\n")
                f.write(f"Files Backed Up: {self.stats['files_backed_up']}\n")
                f.write(f"Total Size: {self.stats['total_size_mb']:.2f}MB\n")
                f.write(f"Errors: {self.stats['errors']}\n")
                f.write(f"\nContents:\n")

                # バックアップ内容をリスト化
                for item in sorted(self.backup_root.rglob("*")):
                    if item.is_file() and item != manifest_path:
                        rel_path = item.relative_to(self.backup_root)
                        f.write(f"  {rel_path}\n")

            print(f"\n✓ マニフェスト作成: {manifest_path}")
        except Exception as e:
            print(f"❌ マニフェスト作成失敗: {e}")

    def print_summary(self):
        """サマリーを表示"""
        print("\n" + "=" * 60)
        print("バックアップ統計")
        print("=" * 60)
        print(f"バックアップ先: {self.backup_root}")
        print(f"ファイル数: {self.stats['files_backed_up']}")
        print(f"総サイズ: {self.stats['total_size_mb']:.2f}MB")
        print(f"エラー: {self.stats['errors']}")
        print()

        if self.stats['errors'] == 0:
            print("✓ バックアップが正常に完了しました")
        else:
            print(f"⚠️  {self.stats['errors']}件のエラーが発生しました")

    def run(self):
        """バックアップ実行"""
        print("=" * 60)
        print("システムバックアップ")
        print("=" * 60)
        print(f"タイプ: {'フルバックアップ' if self.full else '基本バックアップ'}")
        print()

        self.ensure_backup_directory()
        self.backup_config_files()
        self.backup_memory_databases()
        self.backup_documentation()
        self.backup_source_code()
        self.create_backup_manifest()
        self.print_summary()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="システムバックアップ")
    parser.add_argument("--full", action="store_true", help="フルバックアップ（ソースコード含む）")
    parser.add_argument("--destination", type=str, help="バックアップ先ディレクトリ")
    args = parser.parse_args()

    backup = SystemBackup(full=args.full, destination=args.destination)
    backup.run()


if __name__ == "__main__":
    main()
