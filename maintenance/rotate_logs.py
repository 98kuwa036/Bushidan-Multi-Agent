#!/usr/bin/env python3
"""
ログローテーションスクリプト

古いログの圧縮とアーカイブ

Usage:
    python maintenance/rotate_logs.py --dry-run
    python maintenance/rotate_logs.py --days 30
    python maintenance/rotate_logs.py --compress
"""
import os
import sys
import gzip
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトルート
project_root = Path(__file__).parent.parent


class LogRotator:
    """ログローテーター"""

    def __init__(self, days: int = 30, compress: bool = False, dry_run: bool = False):
        self.days = days
        self.compress = compress
        self.dry_run = dry_run
        self.cutoff_date = datetime.now() - timedelta(days=days)

        self.log_dir = project_root / "logs"
        self.archive_dir = project_root / "backups" / "logs"

        self.stats = {
            "total_files": 0,
            "rotated_files": 0,
            "compressed_files": 0,
            "total_size_mb": 0,
            "saved_size_mb": 0
        }

    def ensure_directories(self):
        """必要なディレクトリを作成"""
        if not self.dry_run:
            self.archive_dir.mkdir(parents=True, exist_ok=True)

    def rotate_logs(self):
        """ログをローテーション"""
        if not self.log_dir.exists():
            print(f"⚠️  ログディレクトリが存在しません: {self.log_dir}")
            return

        print("=" * 80)
        print("ログローテーション")
        print("=" * 80)
        print(f"ログディレクトリ: {self.log_dir}")
        print(f"アーカイブ先: {self.archive_dir}")
        print(f"カットオフ日: {self.cutoff_date.strftime('%Y-%m-%d')}")
        print(f"圧縮: {'有効' if self.compress else '無効'}")
        print(f"モード: {'DRY RUN' if self.dry_run else '実行'}")
        print()

        for log_file in self.log_dir.glob("*.log"):
            self.stats["total_files"] += 1
            file_size_mb = log_file.stat().st_size / (1024 ** 2)
            self.stats["total_size_mb"] += file_size_mb

            # ファイルの最終更新日時を確認
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

            if mtime < self.cutoff_date:
                print(f"ローテーション対象: {log_file.name} ({file_size_mb:.2f}MB, {mtime.strftime('%Y-%m-%d')})")

                if not self.dry_run:
                    self._rotate_file(log_file, file_size_mb)
                else:
                    print(f"  [DRY RUN] アーカイブ予定")

                self.stats["rotated_files"] += 1
            else:
                print(f"保持: {log_file.name} ({file_size_mb:.2f}MB, {mtime.strftime('%Y-%m-%d')})")

    def _rotate_file(self, log_file: Path, original_size_mb: float):
        """個別ファイルをローテーション"""
        timestamp = datetime.now().strftime("%Y%m%d")
        dest_name = f"{log_file.stem}_{timestamp}.log"

        if self.compress:
            dest_name += ".gz"
            dest_path = self.archive_dir / dest_name

            try:
                with open(log_file, 'rb') as f_in:
                    with gzip.open(dest_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                compressed_size_mb = dest_path.stat().st_size / (1024 ** 2)
                saved_mb = original_size_mb - compressed_size_mb

                print(f"  ✓ 圧縮完了: {dest_path.name} ({compressed_size_mb:.2f}MB, {saved_mb:.2f}MB削減)")

                self.stats["compressed_files"] += 1
                self.stats["saved_size_mb"] += saved_mb

                # 元ファイルを削除
                log_file.unlink()
                print(f"  ✓ 元ファイル削除: {log_file.name}")

            except Exception as e:
                print(f"  ❌ 圧縮失敗: {e}")
        else:
            dest_path = self.archive_dir / dest_name

            try:
                shutil.move(str(log_file), str(dest_path))
                print(f"  ✓ 移動完了: {dest_path.name}")
            except Exception as e:
                print(f"  ❌ 移動失敗: {e}")

    def print_summary(self):
        """サマリーを表示"""
        print("\n" + "=" * 80)
        print("ローテーション統計")
        print("=" * 80)
        print(f"総ログファイル数: {self.stats['total_files']}")
        print(f"ローテーションファイル数: {self.stats['rotated_files']}")
        print(f"総サイズ: {self.stats['total_size_mb']:.2f}MB")

        if self.compress:
            print(f"圧縮ファイル数: {self.stats['compressed_files']}")
            print(f"節約サイズ: {self.stats['saved_size_mb']:.2f}MB")

    def run(self):
        """実行"""
        self.ensure_directories()
        self.rotate_logs()
        self.print_summary()

        if self.dry_run:
            print("\n💡 実際にローテーションを実行する場合は --dry-run を外してください")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="ログローテーション")
    parser.add_argument("--days", type=int, default=30, help="ローテーション対象期間（日数、デフォルト: 30）")
    parser.add_argument("--compress", "-c", action="store_true", help="ログを圧縮")
    parser.add_argument("--dry-run", action="store_true", help="実行せず、結果のみ表示")
    args = parser.parse_args()

    rotator = LogRotator(days=args.days, compress=args.compress, dry_run=args.dry_run)
    rotator.run()


if __name__ == "__main__":
    main()
