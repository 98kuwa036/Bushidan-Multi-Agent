#!/bin/bash
# ============================================================================
# 武士団マルチエージェント移行スクリプト - 共通設定
# Proxmox LXC → TrueNAS SCALE LXC
# ============================================================================

# ─── Proxmox 環境 ───────────────────────────────────────────────────────
PROXMOX_LXC_ID=100          # 要確認: pct list で確認
PROXMOX_HOST="192.168.11.231"  # Proxmox ホストIP
PROXMOX_SSH_USER="root"

# ─── TrueNAS 環境 ───────────────────────────────────────────────────────
TRUENAS_IP="192.168.11.231"     # ← 実行前に指定（例：192.168.11.240）
TRUENAS_SSH_USER="root"

# ─── LXC コンテナ設定 ───────────────────────────────────────────────────
LXC_NAME="bushidan-honjin"
LXC_IP="192.168.11.231"
LXC_GATEWAY="192.168.11.1"
LXC_DNS="8.8.8.8 8.8.4.4"
LXC_USER="claude"
LXC_UID=1000
LXC_GID=1000
LXC_SHELL="/bin/bash"

# ─── リソース設定 ───────────────────────────────────────────────────────
LXC_CPU_CORES=4
LXC_RAM_GB=4
LXC_DISK_GB=50

# ─── インストール対象バージョン ──────────────────────────────────────────
NODE_VERSION="20"           # Node.js v20.x (nodesource)
PYTHON_VERSION="3.10"       # Python 3.10
PM2_VERSION="latest"        # PM2
CLAUDE_CLI_VERSION="2.1.78"

# ─── パス設定 ───────────────────────────────────────────────────────────
WORK_DIR="/tmp/bushidan_migration"
MIGRATION_FILES_TAR="migration_files.tar.gz"
VZDUMP_FILE="bushidan-honin-100-backup.tar.zst"
BUSHIDAN_EXPORT_DIR="${WORK_DIR}/bushidan_export"
ROOTFS_DIR="${WORK_DIR}/rootfs"

# ─── オプション ─────────────────────────────────────────────────────────
# 言語・ロケール
LANG="ja_JP.UTF-8"
LC_ALL="ja_JP.UTF-8"

# ロギング
LOG_DIR="${WORK_DIR}/logs"
LOG_FILE="${LOG_DIR}/migration_$(date +%Y%m%d_%H%M%S).log"

# ─── 色定義 ─────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─── ユーティリティ関数 ──────────────────────────────────────────────────
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[✗ ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[⚠ WARN]${NC} $1" | tee -a "$LOG_FILE"
}

# ─── チェック関数 ──────────────────────────────────────────────────────
check_required_vars() {
    local missing=0
    if [ "$TRUENAS_IP" = "192.168.11.xxx" ]; then
        log_error "TRUENAS_IP が設定されていません。config.sh を編集してください。"
        missing=1
    fi
    if [ $missing -eq 1 ]; then
        return 1
    fi
    return 0
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "コマンド '$1' が見つかりません。"
        return 1
    fi
}

# ─── ディレクトリ初期化 ──────────────────────────────────────────────────
init_work_dir() {
    mkdir -p "$WORK_DIR" "$LOG_DIR"
    log "作業ディレクトリを初期化: $WORK_DIR"
}

export LOG_FILE
