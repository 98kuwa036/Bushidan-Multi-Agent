#!/bin/bash
# 週次スキル進化タイマーをインストール（pct100 で実行）
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

mkdir -p "$SYSTEMD_USER_DIR"

cp "$SCRIPT_DIR/systemd/bushidan-evolution.service" "$SYSTEMD_USER_DIR/"
cp "$SCRIPT_DIR/systemd/bushidan-evolution.timer"   "$SYSTEMD_USER_DIR/"

systemctl --user daemon-reload
systemctl --user enable bushidan-evolution.timer
systemctl --user start  bushidan-evolution.timer

echo "✅ タイマー有効化完了"
systemctl --user list-timers bushidan-evolution.timer
