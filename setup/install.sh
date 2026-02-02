#!/bin/bash
# =============================================================
# 武士団マルチエージェントシステム v9.4 - Controller Installation
# =============================================================
# CT 100 (本陣) で実行。Python + Node.js + MCP + CLI をセットアップ。
#
# Usage: bash install.sh
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "  武士団マルチエージェントシステム v9.4"
echo "  BDI Framework + llama.cpp CPU最適化"
echo "  Project: ${PROJECT_DIR}"
echo "============================================="

# --- [1] システムパッケージ ---
echo "[1/8] システムパッケージ..."
apt update && apt upgrade -y
apt install -y python3-pip python3-venv git curl wget build-essential cmake

# --- [2] Node.js ---
echo ""
echo "[2/8] Node.js 20..."
if command -v node &>/dev/null; then
    echo "  Node.js: $(node --version)"
else
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt install -y nodejs
    echo "  Node.js: $(node --version)"
fi

# --- [3] Claude CLI (npm) ---
echo ""
echo "[3/8] Claude CLI..."
if command -v claude &>/dev/null; then
    echo "  Claude CLI: $(claude --version 2>/dev/null || echo 'installed')"
else
    npm install -g @anthropic-ai/claude-code
    echo "  Claude CLI インストール完了"
fi

# --- [4] Python venv ---
echo ""
echo "[4/8] Python仮想環境..."
VENV_DIR="${PROJECT_DIR}/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  作成: ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"
echo "  依存パッケージ完了"

# --- [5] MCP サーバー群 ---
echo ""
echo "[5/8] MCP サーバー (足軽 × 8)..."
npm install -g \
    @modelcontextprotocol/server-filesystem \
    @modelcontextprotocol/server-github \
    @modelcontextprotocol/server-fetch \
    @modelcontextprotocol/server-memory \
    @modelcontextprotocol/server-postgres \
    @modelcontextprotocol/server-puppeteer \
    @modelcontextprotocol/server-brave-search \
    @modelcontextprotocol/server-slack
echo "  MCP × 8 インストール完了"

# --- [6] CLI ショートカット ---
echo ""
echo "[6/8] CLI ショートカット..."
cat > "${VENV_DIR}/bin/bushidan" << WRAPPER
#!/bin/bash
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="\$(dirname "\$SCRIPT_DIR")"
PROJECT_DIR="${PROJECT_DIR}"
source "\${VENV_DIR}/bin/activate"
cd "\$PROJECT_DIR"
python -m bushidan.cli "\$@"
WRAPPER
chmod +x "${VENV_DIR}/bin/bushidan"
echo "  CLI: ${VENV_DIR}/bin/bushidan"

# --- [7] 環境変数テンプレート ---
echo ""
echo "[7/8] 環境変数..."
ENV_FILE="${PROJECT_DIR}/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "${PROJECT_DIR}/.env.example" "$ENV_FILE"
    echo "  .env テンプレート作成。APIキーを設定してください。"
else
    echo "  .env 既に存在"
fi

# --- [8] Git自動同期 ---
echo ""
echo "[8/8] Git自動同期..."
SYNC_SCRIPT="${PROJECT_DIR}/setup/auto_sync.sh"
cat > "$SYNC_SCRIPT" << 'SYNC'
#!/bin/bash
# Git自動同期スクリプト v9.4
LOCAL_BASE="/home/claude"
REPO_NAME=$(basename "$(git -C "${PROJECT_DIR}" remote get-url origin 2>/dev/null | sed 's/\.git$//')" 2>/dev/null || echo "")
REPO_PATH="${LOCAL_BASE}/${REPO_NAME:-$(basename "$PROJECT_DIR")}"
LOG_FILE="/var/log/bushidan-sync.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

[ -d "$REPO_PATH" ] || { log "リポジトリ未検出: $REPO_PATH"; exit 1; }
cd "$REPO_PATH"

BRANCH=$(git branch --show-current)
git fetch origin >> "$LOG_FILE" 2>&1

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH" 2>/dev/null)

if [ "$LOCAL" != "$REMOTE" ]; then
    log "変更検出: リモートから更新"
    if ! git diff-index --quiet HEAD --; then
        log "ローカル変更あり: stash実行"
        git stash save "auto-stash $(date +%s)" >> "$LOG_FILE" 2>&1
    fi
    git pull origin "$BRANCH" >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        log "同期成功: $BRANCH"
    else
        log "エラー: pull失敗"
        exit 1
    fi
else
    log "同期不要: 最新状態"
fi
SYNC
chmod +x "$SYNC_SCRIPT"

# Add cron (5分ごと)
if ! crontab -l 2>/dev/null | grep -q "auto_sync"; then
    (crontab -l 2>/dev/null; echo "*/5 * * * * ${SYNC_SCRIPT}") | crontab -
    echo "  cron設定: 5分ごとにGit同期"
fi

echo ""
echo "============================================="
echo "  本陣セットアップ完了! (v9.4)"
echo ""
echo "  使い方:"
echo "    source ${VENV_DIR}/bin/activate"
echo "    bushidan health       # ヘルスチェック"
echo "    bushidan repl         # 対話モード"
echo "    bushidan ask 'Hello'  # タスク実行"
echo "    bushidan server       # APIサーバー起動"
echo ""
echo "  llama.cppセットアップ (CT 101用):"
echo "    ./scripts/setup_llamacpp_prodesck600.sh"
echo ""
echo "  PATH追加 (shell profile):"
echo "    export PATH=\"${VENV_DIR}/bin:\$PATH\""
echo "============================================="
