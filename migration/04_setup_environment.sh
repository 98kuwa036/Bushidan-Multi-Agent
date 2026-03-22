#!/bin/bash
# ============================================================================
# TrueNAS SCALE LXC コンテナ内でのセットアップ
# 実行場所: LXC コンテナ内部 (/tmp/04_setup_environment.sh として配置済み)
# 実行ユーザー: root (初期状態)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# config.sh がない場合はインラインで設定
if [ ! -f "${SCRIPT_DIR}/config.sh" ]; then
    # 簡易ロギング関数
    log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"; }
    log_success() { echo "✓ $1"; }
    log_error() { echo "✗ ERROR: $1" >&2; }
    log_warn() { echo "⚠ WARN: $1"; }
else
    source "${SCRIPT_DIR}/config.sh"
fi

# ─── セクション 1: ユーザー・sudoers設定 ────────────────────────────────
setup_users_and_sudoers() {
    log "========== セクション 1: ユーザー・sudoers 設定 =========="

    # claude ユーザー作成
    if ! id -u claude &> /dev/null; then
        log "claude ユーザーを作成中..."
        useradd -m -u 1000 -s /bin/bash -G sudo claude
        # パスワードなしに設定（rsa鍵認証のみ）
        passwd -d claude
        log_success "claude ユーザー作成完了"
    else
        log "claude ユーザーは既に存在します"
    fi

    # sudoers 設定
    if [ -f "/tmp/migration_staging/etc/sudoers.d/claude" ]; then
        log "sudoers ファイルを配置中..."
        cp /tmp/migration_staging/etc/sudoers.d/claude /etc/sudoers.d/
        chmod 440 /etc/sudoers.d/claude
        log_success "sudoers 配置完了"
    else
        log_warn "sudoers ファイルが見つかりません。デフォルト設定を使用します。"
        echo "claude ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/claude
        chmod 440 /etc/sudoers.d/claude
    fi
}

# ─── セクション 2: パッケージインストール ────────────────────────────────
install_packages() {
    log "========== セクション 2: パッケージインストール =========="

    log "apt を更新中..."
    apt-get update
    apt-get upgrade -y

    log "必須パッケージをインストール中..."
    apt-get install -y \
        build-essential \
        curl \
        wget \
        git \
        jq \
        unzip \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
        net-tools \
        dnsutils \
        htop \
        vim \
        nano \
        openssh-client

    log_success "パッケージインストール完了"
}

# ─── セクション 3: Node.js インストール ────────────────────────────────
install_nodejs() {
    log "========== セクション 3: Node.js v20 インストール =========="

    if command -v node &> /dev/null; then
        local version=$(node --version)
        log "Node.js $version は既にインストール済みです"
        return 0
    fi

    log "NodeSource リポジトリキーをインストール中..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - || \
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash -

    log "Node.js v20 をインストール中..."
    apt-get install -y nodejs

    log "バージョン確認..."
    node --version
    npm --version

    log_success "Node.js インストール完了"
}

# ─── セクション 4: Python環境 ────────────────────────────────────────
setup_python() {
    log "========== セクション 4: Python環境セットアップ =========="

    # Python 3.10 確認・インストール
    if ! command -v python3.10 &> /dev/null; then
        log "Python 3.10 をインストール中..."
        apt-get install -y python3.10 python3.10-venv python3-pip
    fi

    # python3 -> python3.10 のシンボリックリンク確認
    if ! command -v python3 &> /dev/null || [ "$(python3 --version | awk '{print $2}' | cut -d. -f1,2)" != "3.10" ]; then
        log "python3 シンボリックリンクを設定中..."
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
    fi

    python3 --version
    pip3 --version

    log_success "Python環境セットアップ完了"
}

# ─── セクション 5: npm グローバル設定 ────────────────────────────────────
setup_npm_global() {
    log "========== セクション 5: npm グローバル設定 =========="

    # claude ホームディレクトリ準備
    local claude_home="/home/claude"
    mkdir -p "$claude_home/.npm-global"
    chown -R claude:claude "$claude_home"

    # .npmrc 配置
    if [ -f "/tmp/migration_staging/.npmrc" ]; then
        log ".npmrc を配置中..."
        cp /tmp/migration_staging/.npmrc "$claude_home/"
        chown claude:claude "$claude_home/.npmrc"
    else
        log ".npmrc を生成中..."
        echo "prefix=$claude_home/.npm-global" > "$claude_home/.npmrc"
        chown claude:claude "$claude_home/.npmrc"
    fi

    # PM2 グローバルインストール (claude ユーザーで実行)
    log "PM2 をインストール中..."
    sudo -u claude npm install -g pm2

    log_success "npm グローバル設定完了"
}

# ─── セクション 6: Claude CLI インストール ──────────────────────────────
install_claude_cli() {
    log "========== セクション 6: Claude CLI インストール =========="

    # pip経由でインストール
    log "Claude CLI をインストール中..."
    pip3 install --upgrade claude-code || {
        log_warn "pip install 失敗。バイナリダウンロードを試行..."
        # フォールバック: バイナリダウンロード
        curl -fsSL https://github.com/anthropics/claude-code/releases/download/v2.1.78/claude-code-linux-x64 \
            -o /usr/local/bin/claude
        chmod +x /usr/local/bin/claude
    }

    if command -v claude &> /dev/null; then
        log_success "Claude CLI インストール完了: $(claude --version 2>/dev/null || echo 'version unknown')"
    else
        log_warn "Claude CLI インストールに失敗しました。後で手動インストールしてください。"
    fi
}

# ─── セクション 7: ファイル配置 ──────────────────────────────────────────
deploy_configuration_files() {
    log "========== セクション 7: ファイル配置 =========="

    local migration_tar="/tmp/migration_files.tar.gz"
    local staging="/tmp/migration_staging"

    # アーカイブ展開
    if [ -f "$migration_tar" ]; then
        log "移行ファイルを展開中..."
        mkdir -p "$staging"
        tar -xzf "$migration_tar" -C "$staging"
        log_success "展開完了"
    else
        log_error "移行ファイルが見つかりません: $migration_tar"
        return 1
    fi

    local claude_home="/home/claude"

    # .gitconfig
    if [ -f "$staging/.gitconfig" ]; then
        log "配置: .gitconfig"
        cp "$staging/.gitconfig" "$claude_home/"
        chown claude:claude "$claude_home/.gitconfig"
    fi

    # .ssh/known_hosts
    if [ -f "$staging/.ssh/known_hosts" ]; then
        log "配置: .ssh/known_hosts"
        mkdir -p "$claude_home/.ssh"
        cp "$staging/.ssh/known_hosts" "$claude_home/.ssh/"
        chmod 600 "$claude_home/.ssh/known_hosts"
        chown -R claude:claude "$claude_home/.ssh"
    fi

    # .pm2/dump.pm2
    if [ -f "$staging/.pm2/dump.pm2" ]; then
        log "配置: .pm2/dump.pm2"
        mkdir -p "$claude_home/.pm2"
        cp "$staging/.pm2/dump.pm2" "$claude_home/.pm2/"
        chown -R claude:claude "$claude_home/.pm2"
    fi

    # .claude ディレクトリ
    if [ -d "$staging/.claude" ]; then
        log "配置: .claude/ (Claude CLI設定)"
        mkdir -p "$claude_home/.claude"
        cp -r "$staging/.claude/"* "$claude_home/.claude/" 2>/dev/null || true
        chown -R claude:claude "$claude_home/.claude"
    fi

    # cclog
    if [ -d "$staging/cclog" ]; then
        log "配置: cclog/"
        cp -r "$staging/cclog" "$claude_home/"
        chown -R claude:claude "$claude_home/cclog"
    fi

    # .bashrc カスタム部分
    if [ -f "$staging/.bashrc_custom" ]; then
        log ".bashrc カスタム部分を追記中..."
        echo "" >> "$claude_home/.bashrc"
        echo "# ─── カスタム設定（移行分） ───" >> "$claude_home/.bashrc"
        cat "$staging/.bashrc_custom" >> "$claude_home/.bashrc"
        chown claude:claude "$claude_home/.bashrc"
    fi

    # ecosystem.config.cjs (claude-code-ui エントリ削除版)
    if [ -f "$staging/ecosystem.config.cjs.modified" ]; then
        log "配置: ecosystem.config.cjs (修正版)"
        cp "$staging/ecosystem.config.cjs.modified" "$claude_home/ecosystem.config.cjs"
        chown claude:claude "$claude_home/ecosystem.config.cjs"
    fi

    log_success "ファイル配置完了"
}

# ─── セクション 8: systemd サービス設定 ──────────────────────────────────
setup_systemd_services() {
    log "========== セクション 8: systemd サービス設定 =========="

    local staging="/tmp/migration_staging"
    local systemd_src="$staging/etc/systemd/system"

    if [ ! -d "$systemd_src" ]; then
        log_warn "systemd サービスファイルが見つかりません。スキップします。"
        return 0
    fi

    # サービスファイルをコピー
    for service_file in "$systemd_src"/*.service; do
        if [ -f "$service_file" ]; then
            local filename=$(basename "$service_file")
            log "配置: $filename"
            cp "$service_file" "/etc/systemd/system/$filename"
        fi
    done

    # daemon-reload
    log "systemctl daemon-reload を実行中..."
    systemctl daemon-reload

    # サービス有効化
    log "サービスを有効化中..."
    for service in bushidan-console bushidan-main bushidan-discord pm2-claude; do
        if systemctl list-unit-files | grep -q "^${service}\.service"; then
            log "  - $service"
            systemctl enable "$service"
        fi
    done

    log_success "systemd サービス設定完了"
}

# ─── セクション 9: Python venv 再構築 ──────────────────────────────────
rebuild_python_venv() {
    log "========== セクション 9: Python venv 再構築 =========="

    local bushidan_dir="/home/claude/Bushidan-Multi-Agent"

    if [ ! -d "$bushidan_dir" ]; then
        log_warn "Bushidan-Multi-Agent ディレクトリが見つかりません。スキップします。"
        log "このディレクトリは別途 rsync でコピーしてください。"
        return 0
    fi

    if [ ! -f "$bushidan_dir/requirements.txt" ]; then
        log_warn "requirements.txt が見つかりません。venv 再構築をスキップします。"
        return 0
    fi

    log "venv を再構築中... ($bushidan_dir/.venv)"
    cd "$bushidan_dir"

    # 既存venvの削除（ノードバイナリの互換性問題回避）
    if [ -d ".venv" ]; then
        log "既存の .venv を削除中..."
        rm -rf .venv
    fi

    # venv 作成
    python3 -m venv .venv
    source .venv/bin/activate

    # pip アップグレード
    pip install --upgrade pip setuptools wheel

    # requirements インストール
    if [ -f "requirements.txt" ]; then
        log "requirements.txt をインストール中..."
        pip install -r requirements.txt
    fi

    # 所有者設定
    chown -R claude:claude "$bushidan_dir/.venv"

    log_success "Python venv 再構築完了"
}

# ─── セクション 10: PM2 設定 ────────────────────────────────────────────
setup_pm2() {
    log "========== セクション 10: PM2 設定 =========="

    local bushidan_dir="/home/claude/Bushidan-Multi-Agent"
    local ecosystem_file="$bushidan_dir/ecosystem.config.cjs"

    if [ ! -f "$ecosystem_file" ]; then
        log_warn "ecosystem.config.cjs が見つかりません。PM2 スキップします。"
        return 0
    fi

    log "PM2 でプロセスを起動中..."
    sudo -u claude bash -c "cd '$bushidan_dir' && npm run pm2:start 2>/dev/null || pm2 start ecosystem.config.cjs"

    # PM2 save
    log "PM2 設定を保存中..."
    sudo -u claude pm2 save

    log_success "PM2 設定完了"
}

# ─── メイン処理 ────────────────────────────────────────────────────────
main() {
    cat << EOF
╔════════════════════════════════════════════════════╗
║ TrueNAS SCALE LXC セットアップ (コンテナ内)      ║
╚════════════════════════════════════════════════════╝

実行中...

EOF

    # 全セクション実行
    setup_users_and_sudoers || exit 1
    install_packages || exit 1
    install_nodejs || exit 1
    setup_python || exit 1
    setup_npm_global || exit 1
    install_claude_cli || exit 1
    deploy_configuration_files || exit 1
    setup_systemd_services || exit 1
    rebuild_python_venv || exit 1
    setup_pm2 || exit 1

    cat << EOF

╔════════════════════════════════════════════════════╗
║ セットアップ完了！                                ║
╚════════════════════════════════════════════════════╝

次のステップ:

1. Bushidan-Multi-Agent と Bushidan をコピー:
   rsync -avz /path/to/Bushidan-Multi-Agent /home/claude/
   rsync -avz /path/to/Bushidan /home/claude/

2. .env ファイルをコピー:
   scp /path/to/.env /home/claude/Bushidan-Multi-Agent/

3. サービスを起動:
   systemctl start bushidan-main
   systemctl start bushidan-console
   systemctl start bushidan-discord
   pm2 start ecosystem.config.cjs

4. 動作確認:
   python3 /tmp/05_verify.py

5. ログ確認:
   journalctl -u bushidan-main -f
   pm2 logs

EOF
}

main "$@"
