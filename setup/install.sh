#!/bin/bash
# =============================================================
# 武士団マルチエージェントシステム v9.4 - Controller Installation
# =============================================================
# CT 100 (本陣) で実行。Python + Node.js + MCP + CLI をセットアップ。
#
# Usage:
#   # Step 1: root でシステムパッケージをインストール
#   sudo bash install.sh --system
#
#   # Step 2: claude ユーザーでアプリをインストール
#   bash install.sh --user
#
#   # または一括実行（root で実行、内部でsu）
#   sudo bash install.sh
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CLAUDE_USER="claude"
CLAUDE_HOME="/home/${CLAUDE_USER}"

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo "============================================="
    echo "  武士団マルチエージェントシステム v9.4"
    echo "  BDI Framework + llama.cpp CPU最適化"
    echo "  Project: ${PROJECT_DIR}"
    echo "============================================="
}

install_system_packages() {
    echo ""
    echo "[1/3] システムパッケージ (root権限必要)..."

    if [ "$(id -u)" -ne 0 ]; then
        echo "  ⚠️ root権限が必要です。sudo で実行してください。"
        exit 1
    fi

    apt update && apt upgrade -y
    apt install -y python3-pip python3-venv git curl wget build-essential cmake

    # Node.js
    echo ""
    echo "[2/3] Node.js 20..."
    if command -v node &>/dev/null; then
        echo "  Node.js: $(node --version)"
    else
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
        apt install -y nodejs
        echo "  Node.js: $(node --version)"
    fi

    # Create claude user if not exists
    echo ""
    echo "[3/3] ユーザー設定..."
    if ! id "$CLAUDE_USER" &>/dev/null; then
        useradd -m -s /bin/bash "$CLAUDE_USER"
        echo "  ユーザー作成: $CLAUDE_USER"
    else
        echo "  ユーザー既存: $CLAUDE_USER"
    fi

    # Ensure project directory is accessible
    if [ -d "$PROJECT_DIR" ]; then
        chown -R ${CLAUDE_USER}:${CLAUDE_USER} "$PROJECT_DIR"
        echo "  プロジェクト所有権設定完了"
    fi

    echo ""
    echo "✅ システムパッケージ完了"
    echo ""
    echo "次のステップ: claude ユーザーでユーザーパッケージをインストール"
    echo "  su - $CLAUDE_USER"
    echo "  cd $PROJECT_DIR"
    echo "  bash setup/install.sh --user"
}

install_user_packages() {
    echo ""
    echo "【ユーザーパッケージインストール】"
    echo "  User: $(whoami)"
    echo "  Home: $HOME"
    echo ""

    # --- NPM ローカル設定 ---
    echo "[1/6] NPM ローカル設定..."
    NPM_GLOBAL="${HOME}/.npm-global"
    mkdir -p "$NPM_GLOBAL"
    npm config set prefix "$NPM_GLOBAL"

    # 古い認証トークンをクリア（Access token expired エラー対策）
    npm config delete _authToken 2>/dev/null || true
    npm config delete //registry.npmjs.org/:_authToken 2>/dev/null || true

    # PATH に追加（現在のセッション）
    export PATH="${NPM_GLOBAL}/bin:$PATH"

    # .bashrc に追加（永続化）
    if ! grep -q "npm-global" "${HOME}/.bashrc" 2>/dev/null; then
        cat >> "${HOME}/.bashrc" << 'NPMPATH'

# NPM global packages (user-local)
export NPM_GLOBAL="${HOME}/.npm-global"
export PATH="${NPM_GLOBAL}/bin:$PATH"
NPMPATH
        echo "  .bashrc にPATH追加"
    fi
    echo "  NPM prefix: $NPM_GLOBAL"

    # --- Claude CLI ---
    echo ""
    echo "[2/6] Claude CLI..."
    if command -v claude &>/dev/null || [ -f "${NPM_GLOBAL}/bin/claude" ]; then
        echo "  Claude CLI: インストール済み"
    else
        npm install -g @anthropic-ai/claude-code
        echo "  Claude CLI インストール完了"
    fi

    # --- MCP サーバー群 ---
    echo ""
    echo "[3/6] MCP サーバー (足軽 × 8)..."
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

    # --- Python venv ---
    echo ""
    echo "[4/6] Python仮想環境..."
    VENV_DIR="${PROJECT_DIR}/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        echo "  作成: ${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r "${PROJECT_DIR}/requirements.txt"
    echo "  依存パッケージ完了"

    # --- CLI ショートカット ---
    echo ""
    echo "[5/6] CLI ショートカット..."
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

    # --- 環境変数テンプレート ---
    echo ""
    echo "[6/6] 環境変数..."
    ENV_FILE="${PROJECT_DIR}/.env"
    if [ ! -f "$ENV_FILE" ]; then
        cp "${PROJECT_DIR}/.env.example" "$ENV_FILE"
        echo "  .env テンプレート作成。APIキーを設定してください。"
    else
        echo "  .env 既に存在"
    fi

    echo ""
    echo "============================================="
    echo "  ユーザーパッケージ完了! (v9.4)"
    echo ""
    echo "  使い方:"
    echo "    source ${VENV_DIR}/bin/activate"
    echo "    bushidan health       # ヘルスチェック"
    echo "    bushidan repl         # 対話モード"
    echo "    bushidan ask 'Hello'  # タスク実行"
    echo ""
    echo "  PATH (新しいシェルで自動適用):"
    echo "    export PATH=\"${NPM_GLOBAL}/bin:${VENV_DIR}/bin:\$PATH\""
    echo ""
    echo "  llama.cppセットアップ (CT 101用):"
    echo "    ./scripts/setup_llamacpp_prodesck600.sh"
    echo "============================================="
}

install_all() {
    # Root で実行された場合、システムパッケージ後にclaudeユーザーで続行
    if [ "$(id -u)" -eq 0 ]; then
        print_header
        install_system_packages

        echo ""
        echo "【claudeユーザーでユーザーパッケージをインストール中...】"
        echo ""

        # claude ユーザーとしてユーザーパッケージをインストール
        su - "$CLAUDE_USER" -c "cd $PROJECT_DIR && bash setup/install.sh --user"
    else
        # 非rootの場合、ユーザーパッケージのみ
        print_header
        install_user_packages
    fi
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --system    システムパッケージのみインストール (root必要)"
    echo "  --user      ユーザーパッケージのみインストール (claude user)"
    echo "  (no option) 全てインストール (root で実行推奨)"
    echo ""
    echo "推奨手順:"
    echo "  # 一括インストール"
    echo "  sudo bash install.sh"
    echo ""
    echo "  # または段階的に"
    echo "  sudo bash install.sh --system"
    echo "  su - claude"
    echo "  bash ~/Bushidan-Multi-Agent/setup/install.sh --user"
}

# ============================================================================
# Main
# ============================================================================

case "${1:-}" in
    --system)
        print_header
        install_system_packages
        ;;
    --user)
        print_header
        install_user_packages
        ;;
    --help|-h)
        show_usage
        ;;
    *)
        install_all
        ;;
esac
