#!/bin/bash
# ============================================================================
# 武士団マルチエージェントシステム v10.1 - 本陣セットアップ (CT 100)
# ============================================================================
#
# 【コア要件】必ず達成すべきこと:
#   1. Python仮想環境(.venv)の作成
#   2. requirements.txt の依存パッケージインストール
#   3. bushidan CLI ラッパーの作成
#
# 【最終結果】このスクリプト完了後の状態:
#   - .venv/bin/activate で Python 環境が有効化可能
#   - .venv/bin/bushidan コマンドで CLI 起動可能
#   - MCP サーバー設定ファイルが配置済み (npx経由で実行)
#
# 【動作保証】
#   - 必須プログラム (python3, node, npm) がインストール済みなら異常終了しない
#   - オプション機能の失敗はスキップし、最終レポートで報告
#
# Usage:
#   sudo bash install.sh --system   # Step 1: root でシステムパッケージ
#   bash install.sh --user          # Step 2: claude ユーザーでアプリ
#   sudo bash install.sh            # 一括実行
#
# ============================================================================

# エラー時に即座に終了しない (明示的にハンドリング)
# set -e は使わない

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CLAUDE_USER="claude"
CLAUDE_HOME="/home/${CLAUDE_USER}"

# ============================================================================
# 結果トラッキング
# ============================================================================

declare -A RESULTS
CORE_FAILED=0

mark_success() {
    RESULTS["$1"]="OK"
}

mark_failed() {
    RESULTS["$1"]="FAILED"
    if [[ "$2" == "core" ]]; then
        CORE_FAILED=1
    fi
}

mark_skipped() {
    RESULTS["$1"]="SKIPPED"
}

# ============================================================================
# ユーティリティ関数
# ============================================================================

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[OK] $1"
}

log_warning() {
    echo "[WARN] $1"
}

log_error() {
    echo "[ERROR] $1"
}

# ネットワーク操作のリトライ (最大4回、指数バックオフ)
retry_network() {
    local cmd="$1"
    local max_attempts=4
    local delay=2

    for ((i=1; i<=max_attempts; i++)); do
        if eval "$cmd"; then
            return 0
        fi
        if [ $i -lt $max_attempts ]; then
            log_warning "リトライ $i/$max_attempts (${delay}秒後)..."
            sleep $delay
            delay=$((delay * 2))
        fi
    done
    return 1
}

# ============================================================================
# 前提条件チェック
# ============================================================================

check_system_prerequisites() {
    log_info "システム前提条件を確認中..."
    local missing=()

    # 必須コマンド
    for cmd in python3 git curl; do
        if ! command -v $cmd &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "必須コマンドがありません: ${missing[*]}"
        log_info "Ubuntu/Debian: sudo apt install python3 git curl"
        return 1
    fi

    log_success "システム前提条件OK"
    return 0
}

check_user_prerequisites() {
    log_info "ユーザー前提条件を確認中..."
    local missing=()

    for cmd in python3 node npm; do
        if ! command -v $cmd &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "必須コマンドがありません: ${missing[*]}"
        log_info "先に --system オプションで実行してください"
        return 1
    fi

    log_success "ユーザー前提条件OK"
    return 0
}

# ============================================================================
# システムパッケージインストール (root権限)
# ============================================================================

install_system_packages() {
    echo ""
    echo "============================================="
    echo "  本陣セットアップ - システムパッケージ"
    echo "============================================="

    if [ "$(id -u)" -ne 0 ]; then
        log_error "root権限が必要です。sudo で実行してください。"
        return 1
    fi

    # --- 1. apt パッケージ ---
    log_info "[1/3] システムパッケージ..."
    if retry_network "apt update"; then
        mark_success "apt_update"
    else
        log_warning "apt update 失敗 (オフライン環境?)"
        mark_skipped "apt_update"
    fi

    # 必須パッケージ
    local packages="python3-pip python3-venv git curl wget build-essential cmake"
    if apt install -y $packages 2>/dev/null; then
        mark_success "apt_packages"
        log_success "apt パッケージインストール完了"
    else
        log_warning "一部パッケージのインストールに失敗"
        mark_failed "apt_packages"
    fi

    # --- 2. Node.js ---
    log_info "[2/3] Node.js..."
    if command -v node &>/dev/null; then
        log_success "Node.js: $(node --version) (既存)"
        mark_success "nodejs"
    else
        log_info "Node.js をインストール中..."
        if retry_network "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -" && \
           apt install -y nodejs; then
            log_success "Node.js: $(node --version)"
            mark_success "nodejs"
        else
            log_error "Node.js インストール失敗"
            mark_failed "nodejs" "core"
        fi
    fi

    # --- 3. ユーザー設定 ---
    log_info "[3/3] ユーザー設定..."
    if ! id "$CLAUDE_USER" &>/dev/null; then
        if useradd -m -s /bin/bash "$CLAUDE_USER"; then
            log_success "ユーザー作成: $CLAUDE_USER"
            mark_success "user_create"
        else
            log_error "ユーザー作成失敗"
            mark_failed "user_create" "core"
        fi
    else
        log_success "ユーザー既存: $CLAUDE_USER"
        mark_success "user_create"
    fi

    # プロジェクト所有権
    if [ -d "$PROJECT_DIR" ]; then
        chown -R ${CLAUDE_USER}:${CLAUDE_USER} "$PROJECT_DIR" 2>/dev/null || true
        log_success "プロジェクト所有権設定完了"
    fi

    return 0
}

# ============================================================================
# ユーザーパッケージインストール
# ============================================================================

install_user_packages() {
    echo ""
    echo "============================================="
    echo "  本陣セットアップ - ユーザーパッケージ"
    echo "  User: $(whoami)"
    echo "============================================="

    # 前提条件チェック
    if ! check_user_prerequisites; then
        return 1
    fi

    # --- 1. NPM ローカル設定 ---
    log_info "[1/6] NPM ローカル設定..."
    NPM_GLOBAL="${HOME}/.npm-global"
    mkdir -p "$NPM_GLOBAL"

    if npm config set prefix "$NPM_GLOBAL" 2>/dev/null; then
        mark_success "npm_config"
    else
        log_warning "npm config 設定失敗"
        mark_skipped "npm_config"
    fi

    # 古い認証トークンをクリア (エラーは無視)
    npm config delete _authToken 2>/dev/null || true
    npm config delete //registry.npmjs.org/:_authToken 2>/dev/null || true

    # PATH 設定
    export PATH="${NPM_GLOBAL}/bin:$PATH"

    if ! grep -q "npm-global" "${HOME}/.bashrc" 2>/dev/null; then
        cat >> "${HOME}/.bashrc" << 'NPMPATH'

# NPM global packages (user-local)
export NPM_GLOBAL="${HOME}/.npm-global"
export PATH="${NPM_GLOBAL}/bin:$PATH"
NPMPATH
        log_success ".bashrc にPATH追加"
    fi
    mark_success "npm_path"

    # --- 2. Claude CLI (オプション) ---
    log_info "[2/6] Claude CLI..."
    if command -v claude &>/dev/null || [ -f "${NPM_GLOBAL}/bin/claude" ]; then
        log_success "Claude CLI: インストール済み"
        mark_success "claude_cli"
    else
        if retry_network "npm install -g @anthropic-ai/claude-code 2>/dev/null"; then
            log_success "Claude CLI インストール完了"
            mark_success "claude_cli"
        else
            log_warning "Claude CLI インストール失敗 (オプション機能)"
            mark_skipped "claude_cli"
        fi
    fi

    # --- 3. MCP サーバー設定 ---
    log_info "[3/6] MCP サーバー設定..."
    MCP_CONFIG_DIR="${HOME}/.config/claude"
    mkdir -p "$MCP_CONFIG_DIR"

    # MCP設定ファイル (npx経由で実行時にダウンロード) - v10.1 Smithery対応
    cat > "${MCP_CONFIG_DIR}/mcp_servers.json" << 'MCPCONFIG'
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/claude"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp"]
    },
    "tavily": {
      "command": "npx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "${TAVILY_API_KEY}"
      }
    },
    "notion": {
      "command": "npx",
      "args": ["-y", "@notionhq/notion-mcp-server"],
      "env": {
        "NOTION_API_KEY": "${NOTION_API_KEY}"
      }
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git"]
    }
  }
}
MCPCONFIG

    log_success "MCP設定ファイル作成: ${MCP_CONFIG_DIR}/mcp_servers.json"
    mark_success "mcp_config"

    # --- 4. Python仮想環境 (コア) ---
    log_info "[4/6] Python仮想環境..."
    VENV_DIR="${PROJECT_DIR}/.venv"

    if [ ! -d "$VENV_DIR" ]; then
        if python3 -m venv "$VENV_DIR"; then
            log_success "作成: ${VENV_DIR}"
        else
            log_error "Python仮想環境の作成に失敗"
            mark_failed "python_venv" "core"
            return 1
        fi
    else
        log_success "既存: ${VENV_DIR}"
    fi
    mark_success "python_venv"

    # 仮想環境を有効化
    source "${VENV_DIR}/bin/activate"

    # pip アップグレード
    if pip install --upgrade pip 2>/dev/null; then
        mark_success "pip_upgrade"
    else
        log_warning "pip アップグレード失敗"
        mark_skipped "pip_upgrade"
    fi

    # 依存パッケージ (コア)
    if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
        if pip install -r "${PROJECT_DIR}/requirements.txt" 2>/dev/null; then
            log_success "依存パッケージインストール完了"
            mark_success "pip_requirements"
        else
            log_error "依存パッケージのインストールに失敗"
            mark_failed "pip_requirements" "core"
        fi
    else
        log_warning "requirements.txt が見つかりません"
        mark_skipped "pip_requirements"
    fi

    # --- 5. CLI ラッパー (コア) ---
    log_info "[5/6] CLI ラッパー..."
    cat > "${VENV_DIR}/bin/bushidan" << WRAPPER
#!/bin/bash
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="\$(dirname "\$SCRIPT_DIR")"
PROJECT_DIR="${PROJECT_DIR}"
source "\${VENV_DIR}/bin/activate"
cd "\$PROJECT_DIR"
python -m bushidan.cli "\$@"
WRAPPER

    if chmod +x "${VENV_DIR}/bin/bushidan"; then
        log_success "CLI: ${VENV_DIR}/bin/bushidan"
        mark_success "cli_wrapper"
    else
        log_error "CLI ラッパー作成失敗"
        mark_failed "cli_wrapper" "core"
    fi

    # --- 6. 環境変数テンプレート ---
    log_info "[6/6] 環境変数..."
    ENV_FILE="${PROJECT_DIR}/.env"
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "${PROJECT_DIR}/.env.example" ]; then
            cp "${PROJECT_DIR}/.env.example" "$ENV_FILE"
            log_success ".env テンプレート作成"
            mark_success "env_file"
        else
            log_warning ".env.example が見つかりません"
            mark_skipped "env_file"
        fi
    else
        log_success ".env 既に存在"
        mark_success "env_file"
    fi

    return 0
}

# ============================================================================
# 最終レポート
# ============================================================================

print_final_report() {
    local mode="$1"

    echo ""
    echo "============================================="
    echo "  セットアップ完了レポート ($mode)"
    echo "============================================="
    echo ""

    # 結果一覧
    echo "【処理結果】"
    for key in "${!RESULTS[@]}"; do
        local status="${RESULTS[$key]}"
        case "$status" in
            OK)      echo "  [OK]      $key" ;;
            FAILED)  echo "  [FAILED]  $key" ;;
            SKIPPED) echo "  [SKIP]    $key" ;;
        esac
    done

    echo ""

    # コア機能チェック
    if [ $CORE_FAILED -eq 1 ]; then
        echo "【状態】コア機能に失敗があります"
        echo ""
        echo "  必須機能が正しくインストールされていません。"
        echo "  エラーメッセージを確認し、再実行してください。"
        return 1
    else
        echo "【状態】正常完了"
    fi

    if [ "$mode" == "system" ]; then
        echo ""
        echo "【次のステップ】"
        echo "  su - $CLAUDE_USER"
        echo "  cd $PROJECT_DIR"
        echo "  bash setup/install.sh --user"
    elif [ "$mode" == "user" ]; then
        echo ""
        echo "【使い方】"
        echo "  source ${PROJECT_DIR}/.venv/bin/activate"
        echo "  bushidan health       # ヘルスチェック"
        echo "  bushidan repl         # 対話モード"
        echo ""
        echo "【MCP サーバー】(npx経由で実行時にダウンロード)"
        echo "  - filesystem (ファイル操作)"
        echo "  - memory (知識グラフ)"
        echo "  - github (リポジトリ操作)"
        echo "  - sequential-thinking (推論)"
        echo "  - notion (Notion連携)"
        echo ""
        echo "  設定: ~/.config/claude/mcp_servers.json"
        echo ""
        echo "【llama.cpp セットアップ (CT 101用)】"
        echo "  ./scripts/setup_llamacpp_prodesck600.sh"
    fi

    echo ""
    echo "============================================="

    return 0
}

# ============================================================================
# メイン処理
# ============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --system    システムパッケージのみ (root必要)"
    echo "  --user      ユーザーパッケージのみ (claude user)"
    echo "  (no option) 全てインストール (root で実行推奨)"
    echo ""
    echo "【コア要件】"
    echo "  1. Python仮想環境(.venv)の作成"
    echo "  2. requirements.txt の依存パッケージ"
    echo "  3. bushidan CLI ラッパー"
    echo ""
    echo "【最終結果】"
    echo "  - .venv/bin/activate で環境有効化"
    echo "  - bushidan コマンドで CLI 起動"
    echo "  - MCP サーバー設定ファイル配置"
}

install_all() {
    if [ "$(id -u)" -eq 0 ]; then
        echo ""
        echo "============================================="
        echo "  武士団マルチエージェントシステム v10.1"
        echo "  本陣セットアップ (一括インストール)"
        echo "============================================="

        install_system_packages
        print_final_report "system"

        echo ""
        echo "【claudeユーザーでユーザーパッケージをインストール中...】"
        echo ""

        # claude ユーザーとしてユーザーパッケージをインストール
        su - "$CLAUDE_USER" -c "cd $PROJECT_DIR && bash setup/install.sh --user"
    else
        echo ""
        echo "============================================="
        echo "  武士団マルチエージェントシステム v10.1"
        echo "  本陣セットアップ (ユーザーパッケージのみ)"
        echo "============================================="

        install_user_packages
        print_final_report "user"
    fi
}

# ============================================================================
# エントリーポイント
# ============================================================================

case "${1:-}" in
    --system)
        install_system_packages
        print_final_report "system"
        ;;
    --user)
        install_user_packages
        print_final_report "user"
        ;;
    --help|-h)
        show_usage
        ;;
    *)
        install_all
        ;;
esac

# 終了コード: コア機能が失敗した場合のみ 1
exit $CORE_FAILED
