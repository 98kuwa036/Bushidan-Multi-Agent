#!/bin/bash

# ============================================================================
# 武士団マルチエージェントシステム v10.1
# Slack → Discord 移行アップデートスクリプト
# ============================================================================
#
# 機能:
# 1. リポジトリ最新化 (git pull)
# 2. 依存関係更新 (requirements.txt: slack-sdk → discord.py)
# 3. Python venv 再セットアップ
# 4. .env ファイル設定確認・更新
# 5. systemd サービス更新 (bushidan-slack → bushidan-discord)
# 6. 設定ファイル検証
#
# 用途: CT 100 (本陣) で実行
# 実行: bash scripts/update_from_slack_to_discord.sh
#
# ============================================================================

set -e

# ============ カラー定義 ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============ ユーティリティ関数 ============
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============ 前提条件確認 ============
check_prerequisites() {
    log_info "前提条件を確認中..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python3 がインストールされていません"
        exit 1
    fi

    if ! command -v git &> /dev/null; then
        log_error "git がインストールされていません"
        exit 1
    fi

    if [ ! -f .env ]; then
        log_warn ".env ファイルが見つかりません (スキップ: サンプルから作成してください)"
    fi

    log_success "前提条件確認完了"
}

# ============ ステップ 1: リポジトリ最新化 ============
update_repository() {
    log_info "ステップ 1/6: リポジトリを最新化中..."

    git fetch origin claude/complete-integration-refine-K59X0
    git pull origin claude/complete-integration-refine-K59X0

    log_success "リポジトリを最新化しました"
    log_info "最新コミット: $(git log -1 --oneline)"
}

# ============ ステップ 2: venv 再セットアップ ============
setup_venv() {
    log_info "ステップ 2/6: Python venv をセットアップ中..."

    # venv が存在しない場合は作成
    if [ ! -d .venv ]; then
        python3 -m venv .venv
        log_success "venv を作成しました"
    else
        log_info "既存の venv を使用します"
    fi

    # venv 有効化
    source .venv/bin/activate

    # pip 更新
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1

    log_success "venv セットアップ完了"
}

# ============ ステップ 3: 依存関係更新 ============
update_dependencies() {
    log_info "ステップ 3/6: 依存関係を更新中..."

    # venv 有効化
    source .venv/bin/activate

    # requirements.txt から インストール (slack-sdk は削除、discord.py が追加されている)
    pip install -r requirements.txt

    # 確認: discord.py がインストールされているか
    if python3 -c "import discord; print(f'discord.py {discord.__version__}' )" > /dev/null 2>&1; then
        log_success "discord.py がインストールされました"
    else
        log_error "discord.py のインストールに失敗しました"
        exit 1
    fi

    # 確認: slack-sdk がインストールされていないか
    if python3 -c "import slack_sdk" 2>/dev/null; then
        log_warn "slack-sdk はまだインストールされています (pip uninstall slack-sdk で削除してください)"
    else
        log_success "slack-sdk は削除されています"
    fi

    log_success "依存関係を更新しました"
}

# ============ ステップ 4: .env 設定確認 ============
check_env_config() {
    log_info "ステップ 4/6: .env 設定を確認中..."

    if [ ! -f .env ]; then
        log_warn ".env ファイルが見つかりません"
        log_info ".env.example をコピーして手動で設定してください:"
        log_info "  cp .env.example .env"
        log_info "  nano .env"
        return 0
    fi

    # DISCORD_BOT_TOKEN が設定されているか確認
    if grep -q "^DISCORD_BOT_TOKEN=" .env; then
        log_success "DISCORD_BOT_TOKEN は設定されています"
    else
        log_warn "DISCORD_BOT_TOKEN が .env に見つかりません"
        log_info "Discord Developer Portal からトークンを取得してください:"
        log_info "  1. https://discord.com/developers/applications"
        log_info "  2. Application → Bot → Reset Token をコピー"
        log_info "  3. .env の DISCORD_BOT_TOKEN=... に設定"
    fi

    # SLACK_BOT_TOKEN の古い設定があるか確認
    if grep -q "^SLACK_BOT_TOKEN=" .env; then
        log_warn "SLACK_BOT_TOKEN がまだ .env に存在します"
        log_info "以下で削除してください:"
        log_info "  sed -i '/^SLACK_BOT_TOKEN=/d' .env"
    else
        log_success "SLACK_BOT_TOKEN は削除されています"
    fi

    # SLACK_SIGNING_SECRET の古い設定があるか確認
    if grep -q "^SLACK_SIGNING_SECRET=" .env; then
        log_warn "SLACK_SIGNING_SECRET がまだ .env に存在します"
        log_info "以下で削除してください:"
        log_info "  sed -i '/^SLACK_SIGNING_SECRET=/d' .env"
    else
        log_success "SLACK_SIGNING_SECRET は削除されています"
    fi
}

# ============ ステップ 5: Discord Bot テスト ============
test_discord_bot() {
    log_info "ステップ 5/6: Discord Bot をテスト中..."

    source .venv/bin/activate

    # discord_bot.py が存在するか確認
    if [ ! -f bushidan/discord_bot.py ]; then
        log_error "bushidan/discord_bot.py が見つかりません"
        exit 1
    fi

    log_success "bushidan/discord_bot.py は正常です"

    # 簡易的な構文チェック
    if python3 -m py_compile bushidan/discord_bot.py 2>/dev/null; then
        log_success "Discord Bot の構文チェック完了"
    else
        log_error "Discord Bot に構文エラーがあります"
        exit 1
    fi

    # dotenv が正しく読み込まれるか確認
    if python3 -c "from pathlib import Path; from dotenv import load_dotenv; env_path = Path('.env'); load_dotenv(env_path) if env_path.exists() else None; print('dotenv OK')" > /dev/null 2>&1; then
        log_success "dotenv の読み込みは正常です"
    else
        log_error "dotenv の読み込みに失敗しました"
        exit 1
    fi
}

# ============ ステップ 6: systemd サービス更新 ============
update_systemd_service() {
    log_info "ステップ 6/6: systemd サービスを確認中..."

    SERVICE_FILE="/etc/systemd/system/bushidan-discord.service"
    OLD_SERVICE_FILE="/etc/systemd/system/bushidan-slack.service"

    # 古い Slack サービスが存在するか確認
    if [ -f "$OLD_SERVICE_FILE" ]; then
        log_warn "古い bushidan-slack.service が存在します"
        log_info "以下で停止・削除してください:"
        log_info "  sudo systemctl stop bushidan-slack"
        log_info "  sudo systemctl disable bushidan-slack"
        log_info "  sudo rm $OLD_SERVICE_FILE"
        log_info "  sudo systemctl daemon-reload"
    fi

    # 新しい Discord サービスが存在するか確認
    if [ ! -f "$SERVICE_FILE" ]; then
        log_warn "bushidan-discord.service がまだ作成されていません"
        log_info "以下のテンプレートを参考に作成してください:"
        cat << 'EOF'

[Unit]
Description=Bushidan Multi-Agent Discord Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/user/Bushidan-Multi-Agent
Environment="PATH=/home/user/Bushidan-Multi-Agent/.venv/bin"
ExecStart=/home/user/Bushidan-Multi-Agent/.venv/bin/python -m bushidan.discord_bot
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    else
        log_success "bushidan-discord.service は存在します"
        log_info "以下で起動してください:"
        log_info "  sudo systemctl start bushidan-discord"
        log_info "  sudo systemctl enable bushidan-discord"
    fi
}

# ============ クリーンアップ ============
cleanup() {
    log_info "クリーンアップ中..."

    # Slack 関連ファイルが削除されたか確認
    if [ -f bushidan/slack_bot.py ]; then
        log_error "bushidan/slack_bot.py がまだ存在しています"
        exit 1
    fi

    if [ -f integrations/slack_bot.py ]; then
        log_error "integrations/slack_bot.py がまだ存在しています"
        exit 1
    fi

    log_success "Slack 関連ファイルは削除されています"
}

# ============ メイン処理 ============
main() {
    echo ""
    echo "============================================================================"
    echo "武士団マルチエージェントシステム v10.1"
    echo "Slack → Discord 移行アップデート"
    echo "============================================================================"
    echo ""

    # 前提条件確認
    check_prerequisites
    echo ""

    # ステップ実行
    update_repository
    echo ""

    setup_venv
    echo ""

    update_dependencies
    echo ""

    check_env_config
    echo ""

    test_discord_bot
    echo ""

    update_systemd_service
    echo ""

    cleanup
    echo ""

    echo "============================================================================"
    echo -e "${GREEN}✓ アップデート完了！${NC}"
    echo "============================================================================"
    echo ""
    echo "次のステップ:"
    echo ""
    echo "1. .env ファイルで DISCORD_BOT_TOKEN を設定:"
    echo "   - Discord Developer Portal から取得"
    echo "   - https://discord.com/developers/applications"
    echo ""
    echo "2. Discord Bot を手動テスト:"
    echo "   source .venv/bin/activate"
    echo "   python -m bushidan.discord_bot"
    echo ""
    echo "3. systemd サービスとして起動:"
    echo "   sudo systemctl start bushidan-discord"
    echo "   sudo systemctl enable bushidan-discord"
    echo ""
    echo "4. Discord で @Bushidan コマンドをテスト:"
    echo "   @Bushidan こんにちは"
    echo ""
}

# スクリプト実行
main
