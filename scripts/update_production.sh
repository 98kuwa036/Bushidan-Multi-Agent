#!/bin/bash
# ============================================================================
# 武士団マルチエージェントシステム - 本番環境アップデートスクリプト
# dev → production 同期
#
# 開発環境: /home/claude/develop/Bushidan-Multi-Agent
# 本番環境: /home/claude/Bushidan-Multi-Agent
#
# 【設計方針】
#   - バックアップを必ず作成してからアップデート
#   - .env / logs / models / 実行時データは絶対に上書きしない
#   - ローカルLLM設定を状況に応じて切り替え可能
#   - --dry-run でプレビュー、--rollback でバックアップから復元
#
# 【ローカルLLM 運用状況】
#   現在: Qwen3-coder (メモリ増強前)
#   将来: Nemotron-3-Nano (メモリ増強後)
#   → --use-nemotron フラグで切り替え
#
# Usage:
#   ./scripts/update_production.sh               # 通常アップデート（Qwen3モード）
#   ./scripts/update_production.sh --dry-run     # プレビューのみ
#   ./scripts/update_production.sh --use-nemotron # Nemotronモードで適用（メモリ増強後）
#   ./scripts/update_production.sh --rollback    # 最新バックアップから復元
#   ./scripts/update_production.sh --list-backups # バックアップ一覧
# ============================================================================

set -euo pipefail

# ============================================================================
# 設定
# ============================================================================

DEV_DIR="/home/claude/develop/Bushidan-Multi-Agent"
PROD_DIR="/home/claude/Bushidan-Multi-Agent"
BACKUP_BASE="/home/claude/backups/bushidan"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${BACKUP_BASE}/${TIMESTAMP}"
LOG_FILE="${BACKUP_BASE}/update_${TIMESTAMP}.log"

# ローカルLLM設定
# Qwen3-coder (既存)
QWEN3_MODEL_PATH="${PROD_DIR}/models/qwen3/Qwen3-Coder-30B-Q4_K_M.gguf"
# Nemotron-3-Nano (新規)
NEMOTRON_MODEL_PATH="${PROD_DIR}/models/nemotron/Nemotron-3-Nano-Q4_K_M.gguf"
LLAMACPP_HOST="127.0.0.1"
LLAMACPP_PORT=8080

# 実行モードフラグ
DRY_RUN=0
USE_NEMOTRON=0
DO_ROLLBACK=0
LIST_BACKUPS=0
SKIP_PIP=0
NO_SERVICE_RESTART=0

# ============================================================================
# 保護対象（絶対に上書きしないファイル/ディレクトリ）
# ============================================================================

# rsync --exclude パターン（本番環境固有データ）
PRESERVE_PATTERNS=(
    # === 最重要: 絶対に触らない ===
    ".git/"                         # 本番 git リポジトリ（別管理）
    ".env"                          # APIキー（最重要）
    ".venv/"                        # Python仮想環境

    # === 実行時データ ===
    "logs/"                         # 実行ログ
    "models/"                       # LLMモデルファイル（大容量）
    "shogun_memory.jsonl"           # エージェント記憶
    "codings/"                      # 成果物
    "codings_summary_table.md"      # 成果物サマリー

    # === 本番固有設定 ===
    "bushidan_config.yaml"          # 本番ローカル設定
    "config/discord_llm_accounts.json"  # Discord アカウント設定
    "config/interactive_config.yaml"    # インタラクティブ設定
    "config/mcp_config.yaml"            # MCP 本番設定
    "config/mcp_status_config.yaml"     # MCP ステータス設定

    # === 本番固有ディレクトリ（開発版に存在しない） ===
    "src/"                          # 本番固有ソース
    "interfaces/"                   # 本番固有インターフェース
    "maintenance/"                  # メンテナンスデータ

    # === 本番固有ファイル ===
    "bushidan_mcp_status.py"
    "mcp_status_checker.py"
    "generate_coding_summary.py"
    "test_api_keys.py"
    "DISCORD_LLM_INTEGRATION.md"
    "DISCORD_SETUP.md"
    "IMPLEMENTATION_GUIDE.md"
    "INTERACTIVE_MODE.md"

    # === キャッシュ/一時 ===
    ".claude/"
    ".tmp/"
    "__pycache__/"
    "*.pyc"
    "*.pyo"
    ".pytest_cache/"
)

# ============================================================================
# ユーティリティ
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()         { echo -e "${NC}[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"; }
log_info()    { echo -e "${BLUE}[INFO]${NC}  $1" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $1" | tee -a "$LOG_FILE"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1" | tee -a "$LOG_FILE"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"; }
log_dry()     { echo -e "${CYAN}[DRY]${NC}   $1"; }

die() { log_error "$1"; exit 1; }

# ============================================================================
# 引数解析
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)        DRY_RUN=1 ;;
        --use-nemotron)   USE_NEMOTRON=1 ;;
        --rollback)       DO_ROLLBACK=1 ;;
        --list-backups)   LIST_BACKUPS=1 ;;
        --skip-pip)       SKIP_PIP=1 ;;
        --no-restart)     NO_SERVICE_RESTART=1 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  --dry-run         変更内容をプレビュー（実際には変更しない）"
            echo "  --use-nemotron    Nemotron-3-Nano を隠密として有効化（メモリ増強後）"
            echo "  --rollback        最新バックアップから本番を復元"
            echo "  --list-backups    バックアップ一覧を表示"
            echo "  --skip-pip        pip install をスキップ"
            echo "  --no-restart      サービス再起動をスキップ"
            echo ""
            echo "【デフォルト動作】"
            echo "  隠密ローカルLLM: Qwen3-coder (--use-nemotron なし)"
            exit 0
            ;;
        *) die "不明なオプション: $1 (--help で使用法確認)" ;;
    esac
    shift
done

# ============================================================================
# バックアップ一覧表示
# ============================================================================

if [ $LIST_BACKUPS -eq 1 ]; then
    echo ""
    echo "【バックアップ一覧】"
    if [ -d "$BACKUP_BASE" ]; then
        ls -lt "$BACKUP_BASE" | grep "^d" | awk '{print $NF}' | while read -r d; do
            local_size=$(du -sh "${BACKUP_BASE}/${d}" 2>/dev/null | cut -f1)
            echo "  ${d}  (${local_size})"
        done
    else
        echo "  バックアップなし"
    fi
    echo ""
    exit 0
fi

# ============================================================================
# ロールバック
# ============================================================================

if [ $DO_ROLLBACK -eq 1 ]; then
    echo ""
    echo -e "${BOLD}【ロールバック】${NC}"

    if [ ! -d "$BACKUP_BASE" ]; then
        die "バックアップディレクトリが存在しません: $BACKUP_BASE"
    fi

    # 最新バックアップを取得
    LATEST_BACKUP=$(ls -t "$BACKUP_BASE" | grep "^[0-9]" | head -1)
    if [ -z "$LATEST_BACKUP" ]; then
        die "バックアップが見つかりません"
    fi

    RESTORE_FROM="${BACKUP_BASE}/${LATEST_BACKUP}"
    echo "復元元: $RESTORE_FROM"
    read -p "本番環境をこのバックアップに戻しますか？ [y/N]: " -n 1 -r yn
    echo
    if [[ ! "$yn" =~ ^[Yy]$ ]]; then
        echo "キャンセルしました"
        exit 0
    fi

    mkdir -p "$BACKUP_BASE"
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"

    log_info "ロールバック開始: ${LATEST_BACKUP}"
    rsync -av --delete \
        --exclude=".env" \
        --exclude=".venv/" \
        --exclude="logs/" \
        --exclude="models/" \
        "${RESTORE_FROM}/" "${PROD_DIR}/" >> "$LOG_FILE" 2>&1
    log_success "ロールバック完了: ${PROD_DIR}"
    exit 0
fi

# ============================================================================
# メイン処理開始
# ============================================================================

mkdir -p "$BACKUP_BASE"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD}  武士団 v11.4 - 本番環境アップデート${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""

# モード表示
if [ $DRY_RUN -eq 1 ]; then
    echo -e "${CYAN}  モード: DRY RUN（実際の変更なし）${NC}"
else
    echo -e "${GREEN}  モード: 本番適用${NC}"
fi

if [ $USE_NEMOTRON -eq 1 ]; then
    echo -e "  ローカルLLM: ${GREEN}Nemotron-3-Nano (有効)${NC}"
else
    echo -e "  ローカルLLM: ${YELLOW}Qwen3-coder (Nemotron無効・メモリ増強待ち)${NC}"
fi
echo ""

# ============================================================================
# 前提条件チェック
# ============================================================================

log_info "前提条件チェック..."

[ -d "$DEV_DIR" ]  || die "開発環境が見つかりません: $DEV_DIR"
[ -d "$PROD_DIR" ] || die "本番環境が見つかりません: $PROD_DIR"
command -v rsync &>/dev/null || die "rsync がインストールされていません"
command -v python3 &>/dev/null || die "python3 がインストールされていません"

# 開発環境のバージョン確認
DEV_VERSION=$(grep -m1 'version:' "${DEV_DIR}/config/settings.yaml" 2>/dev/null | awk '{print $2}' | tr -d '"' || echo "不明")
PROD_VERSION=$(grep -m1 'version:' "${PROD_DIR}/config/settings.yaml" 2>/dev/null | awk '{print $2}' | tr -d '"' || echo "不明")

log_info "開発環境バージョン: v${DEV_VERSION}"
log_info "本番環境バージョン: v${PROD_VERSION}"

log_success "前提条件OK"

# ============================================================================
# 変更内容プレビュー
# ============================================================================

log_info "変更内容を確認中..."

# rsync の除外フラグ構築
RSYNC_EXCLUDES=()
for pattern in "${PRESERVE_PATTERNS[@]}"; do
    RSYNC_EXCLUDES+=("--exclude=${pattern}")
done

# 変更ファイル一覧（--dry-run で確認）
CHANGED_FILES=$(rsync -av --dry-run --delete \
    "${RSYNC_EXCLUDES[@]}" \
    "${DEV_DIR}/" "${PROD_DIR}/" 2>/dev/null \
    | grep -v "/$" \
    | grep -v "^sending\|^sent\|^total\|^building\|^created\|\.\/$" \
    | grep -v "^$" \
    | tail -n +2 \
    | head -n -3) || true

echo ""
echo "【更新対象ファイル】"
if [ -n "$CHANGED_FILES" ]; then
    echo "$CHANGED_FILES" | while read -r f; do
        if [[ "$f" == deleting* ]]; then
            echo -e "  ${RED}削除: ${f#deleting }${NC}"
        else
            if [ -f "${PROD_DIR}/${f}" ]; then
                echo -e "  ${YELLOW}更新: ${f}${NC}"
            else
                echo -e "  ${GREEN}新規: ${f}${NC}"
            fi
        fi
    done
else
    echo "  変更なし"
fi
echo ""

if [ $DRY_RUN -eq 1 ]; then
    log_dry "DRY RUN 完了。--dry-run を外して実行してください。"
    exit 0
fi

# ============================================================================
# 実行確認
# ============================================================================

echo -e "${YELLOW}【保護対象（上書きしない）】${NC}"
for p in "${PRESERVE_PATTERNS[@]}"; do
    echo "  $p"
done
echo ""

read -p "本番環境をアップデートしますか？ [y/N]: " -n 1 -r CONFIRM
echo
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "キャンセルしました"
    exit 0
fi

# ============================================================================
# ステップ1: バックアップ
# ============================================================================

echo ""
log_info "【Step 1/5】 本番環境をバックアップ中..."

mkdir -p "$BACKUP_DIR"

rsync -a \
    --exclude=".venv/" \
    --exclude="logs/" \
    --exclude="models/" \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    "${PROD_DIR}/" "${BACKUP_DIR}/" >> "$LOG_FILE" 2>&1

log_success "バックアップ完了: ${BACKUP_DIR}"

# ============================================================================
# ステップ2: ソースファイル同期
# ============================================================================

echo ""
log_info "【Step 2/5】 ソースファイルを同期中..."

rsync -av \
    "${RSYNC_EXCLUDES[@]}" \
    "${DEV_DIR}/" "${PROD_DIR}/" >> "$LOG_FILE" 2>&1

log_success "ソースファイル同期完了"

# ============================================================================
# ステップ3: ローカルLLM設定の適用
# ============================================================================

echo ""
log_info "【Step 3/5】 ローカルLLM設定を適用中..."

SETTINGS_FILE="${PROD_DIR}/config/settings.yaml"

if [ ! -f "$SETTINGS_FILE" ]; then
    log_warn "settings.yaml が見つかりません。スキップ"
else
    if [ $USE_NEMOTRON -eq 1 ]; then
        # Nemotron-3-Nano を有効化
        python3 - <<PYEOF
import re

with open("${SETTINGS_FILE}", "r") as f:
    content = f.read()

# llamacpp エンドポイントを Nemotron に設定
content = re.sub(
    r'(llamacpp_endpoint:\s*)["\']?http://[^"\'\n]+["\']?',
    r'\1"http://${LLAMACPP_HOST}:${LLAMACPP_PORT}"',
    content
)
content = re.sub(
    r'(llamacpp_model_path:\s*)["\']?[^"\'\n]+["\']?',
    r'\1"${NEMOTRON_MODEL_PATH}"',
    content
)
# use_llamacpp を true に
content = re.sub(r'(use_llamacpp:\s*)false', r'\1true', content)

with open("${SETTINGS_FILE}", "w") as f:
    f.write(content)

print("  Nemotron-3-Nano 設定を適用しました")
PYEOF
        log_success "ローカルLLM: Nemotron-3-Nano (有効化)"
    else
        # Qwen3-coder を使用（Nemotron は設定するが model_path を Qwen3 に向ける）
        python3 - <<PYEOF
import re

with open("${SETTINGS_FILE}", "r") as f:
    content = f.read()

# llamacpp エンドポイントを設定
content = re.sub(
    r'(llamacpp_endpoint:\s*)["\']?http://[^"\'\n]+["\']?',
    r'\1"http://${LLAMACPP_HOST}:${LLAMACPP_PORT}"',
    content
)
# model_path を Qwen3 に向ける
content = re.sub(
    r'(llamacpp_model_path:\s*)["\']?[^"\'\n]+["\']?',
    r'\1"${QWEN3_MODEL_PATH}"',
    content
)
# use_llamacpp を true に
content = re.sub(r'(use_llamacpp:\s*)false', r'\1true', content)

with open("${SETTINGS_FILE}", "w") as f:
    f.write(content)

print("  Qwen3-coder 設定を適用しました（Nemotron は無効）")
PYEOF
        log_success "ローカルLLM: Qwen3-coder (Nemotron は無効・メモリ増強後に --use-nemotron で切替)"
    fi
fi

# ============================================================================
# ステップ4: Python 依存関係の更新
# ============================================================================

echo ""
log_info "【Step 4/5】 Python 依存関係を確認中..."

if [ $SKIP_PIP -eq 1 ]; then
    log_info "pip install をスキップ (--skip-pip)"
else
    VENV_PIP="${PROD_DIR}/.venv/bin/pip"
    SYSTEM_PIP="pip3"

    if [ -f "$VENV_PIP" ]; then
        PIP_CMD="$VENV_PIP"
        log_info "仮想環境の pip を使用: ${VENV_PIP}"
    else
        PIP_CMD="$SYSTEM_PIP"
        log_warn "仮想環境が見つかりません。システム pip を使用"
    fi

    if $PIP_CMD install -r "${PROD_DIR}/requirements.txt" -q >> "$LOG_FILE" 2>&1; then
        log_success "pip install 完了"
    else
        log_warn "pip install で一部エラー (ログ確認: ${LOG_FILE})"
    fi
fi

# ============================================================================
# ステップ5: サービス再起動
# ============================================================================

echo ""
log_info "【Step 5/5】 サービス状態を確認中..."

if [ $NO_SERVICE_RESTART -eq 1 ]; then
    log_info "サービス再起動をスキップ (--no-restart)"
else
    # Bushidan サービスの再起動（systemd 使用時）
    SERVICES=("bushidan" "bushidan-bot" "bushidan-discord")
    RESTARTED=0

    for svc in "${SERVICES[@]}"; do
        if systemctl is-active --quiet "$svc" 2>/dev/null; then
            log_info "サービス再起動: $svc"
            if systemctl restart "$svc" 2>/dev/null; then
                log_success "再起動完了: $svc"
                RESTARTED=1
            else
                log_warn "再起動失敗: $svc (手動再起動が必要)"
            fi
        fi
    done

    if [ $RESTARTED -eq 0 ]; then
        log_info "稼働中のサービスなし（手動起動が必要な場合は python main.py を実行）"
    fi
fi

# ============================================================================
# 完了レポート
# ============================================================================

echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD}  アップデート完了${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""
echo -e "  ${GREEN}✅ 本番環境バージョン: v${DEV_VERSION}${NC}"
echo ""
echo "  【バックアップ】"
echo "    ${BACKUP_DIR}"
echo ""
echo "  【ローカルLLM (隠密)】"
if [ $USE_NEMOTRON -eq 1 ]; then
    echo -e "    ${GREEN}Nemotron-3-Nano: 有効${NC}"
    echo "    エンドポイント: http://${LLAMACPP_HOST}:${LLAMACPP_PORT}"
    echo "    モデル: ${NEMOTRON_MODEL_PATH}"
else
    echo -e "    ${YELLOW}Qwen3-coder: 有効（Nemotron: 無効）${NC}"
    echo "    エンドポイント: http://${LLAMACPP_HOST}:${LLAMACPP_PORT}"
    echo "    モデル: ${QWEN3_MODEL_PATH}"
    echo ""
    echo -e "    ${CYAN}※ メモリ増強後: ./scripts/update_production.sh --use-nemotron${NC}"
fi
echo ""
echo "  【ログ】"
echo "    ${LOG_FILE}"
echo ""
echo "  【ロールバック方法】"
echo "    ./scripts/update_production.sh --rollback"
echo ""
echo -e "${BOLD}=============================================${NC}"
