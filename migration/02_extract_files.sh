#!/bin/bash
# ============================================================================
# vzdump アーカイブから必要ファイルを抽出
# 実行場所: 任意（TrueNAS或いは中継ホスト）
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ─── 前提条件チェック ────────────────────────────────────────────────────
check_prerequisites() {
    log "前提条件をチェック中..."

    check_command "zstd" || {
        log_error "zstd コマンドが必要です。インストール: apt install zstd"
        exit 1
    }

    check_command "tar" || {
        log_error "tar コマンドが見つかりません。"
        exit 1
    }

    if [ ! -f "$BUSHIDAN_EXPORT_DIR/$VZDUMP_FILE" ]; then
        log_error "vzdump ファイルが見つかりません: $BUSHIDAN_EXPORT_DIR/$VZDUMP_FILE"
        log "ファイルを指定するか、TrueNAS側にコピーしてください。"
        exit 1
    fi

    log_success "前提条件OK"
}

# ─── rootfs 展開 ───────────────────────────────────────────────────────
extract_rootfs() {
    log "rootfs を展開中..."

    mkdir -p "$ROOTFS_DIR"

    # vzdump ファイルの構造確認
    log "vzdump ファイル構造を確認中..."
    zstd -d --stdout "$BUSHIDAN_EXPORT_DIR/$VZDUMP_FILE" 2>/dev/null | \
        tar -tf - | head -20 | tee -a "$LOG_FILE"

    # rootfs 抽出
    log "zstd 解凍 + tar 展開中..."
    zstd -d --stdout "$BUSHIDAN_EXPORT_DIR/$VZDUMP_FILE" 2>/dev/null | \
        tar -xf - -C "$ROOTFS_DIR" 2>&1 | tee -a "$LOG_FILE"

    if [ -d "$ROOTFS_DIR/rootfs" ]; then
        log_success "rootfs 展開完了: $ROOTFS_DIR/rootfs"
    else
        log_error "rootfs が見つかりません。アーカイブ構造を確認してください。"
        ls -la "$ROOTFS_DIR" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# ─── 必要ファイル抽出 ───────────────────────────────────────────────────
collect_migration_files() {
    log "必要なファイルを抽出中..."

    local rootfs="$ROOTFS_DIR/rootfs"
    local staging_dir="${WORK_DIR}/staging"
    mkdir -p "$staging_dir"

    # ホームディレクトリ配下のファイル
    log "  - .gitconfig"
    [ -f "$rootfs/home/claude/.gitconfig" ] && cp "$rootfs/home/claude/.gitconfig" "$staging_dir/"

    log "  - .npmrc"
    [ -f "$rootfs/home/claude/.npmrc" ] && cp "$rootfs/home/claude/.npmrc" "$staging_dir/"

    log "  - .ssh/known_hosts"
    mkdir -p "$staging_dir/.ssh"
    [ -f "$rootfs/home/claude/.ssh/known_hosts" ] && \
        cp "$rootfs/home/claude/.ssh/known_hosts" "$staging_dir/.ssh/"

    log "  - .pm2/dump.pm2"
    mkdir -p "$staging_dir/.pm2"
    [ -f "$rootfs/home/claude/.pm2/dump.pm2" ] && \
        cp "$rootfs/home/claude/.pm2/dump.pm2" "$staging_dir/.pm2/"

    log "  - .claude/settings.json"
    mkdir -p "$staging_dir/.claude"
    [ -f "$rootfs/home/claude/.claude/settings.json" ] && \
        cp "$rootfs/home/claude/.claude/settings.json" "$staging_dir/.claude/"

    log "  - .claude/policy-limits.json"
    [ -f "$rootfs/home/claude/.claude/policy-limits.json" ] && \
        cp "$rootfs/home/claude/.claude/policy-limits.json" "$staging_dir/.claude/"

    log "  - cclog/ (ディレクトリ)"
    if [ -d "$rootfs/home/claude/cclog" ]; then
        cp -r "$rootfs/home/claude/cclog" "$staging_dir/"
    fi

    log "  - .bashrc (カスタム部分)"
    if [ -f "$rootfs/home/claude/.bashrc" ]; then
        # カスタム行（NPM_GLOBAL以降）を抽出
        sed -n '/NPM_GLOBAL/,$p' "$rootfs/home/claude/.bashrc" > "$staging_dir/.bashrc_custom" 2>/dev/null || true
    fi

    # ecosystem.config.cjs の修正版を生成
    log "  - ecosystem.config.cjs (claude-code-ui エントリ削除版)"
    if [ -f "$rootfs/home/claude/ecosystem.config.cjs" ]; then
        # claude-code-ui ブロック削除版を生成
        sed '/name: .claude-code-ui/,/},$/d' "$rootfs/home/claude/ecosystem.config.cjs" \
            > "$staging_dir/ecosystem.config.cjs.modified"
    fi

    # systemd サービスファイル
    log "  - systemd service files"
    mkdir -p "$staging_dir/etc/systemd/system"

    [ -f "$rootfs/etc/systemd/system/bushidan-console.service" ] && \
        cp "$rootfs/etc/systemd/system/bushidan-console.service" "$staging_dir/etc/systemd/system/"

    [ -f "$rootfs/etc/systemd/system/bushidan-main.service" ] && \
        cp "$rootfs/etc/systemd/system/bushidan-main.service" "$staging_dir/etc/systemd/system/"

    [ -f "$rootfs/etc/systemd/system/bushidan-discord.service" ] && \
        cp "$rootfs/etc/systemd/system/bushidan-discord.service" "$staging_dir/etc/systemd/system/"

    [ -f "$rootfs/etc/systemd/system/pm2-claude.service" ] && \
        cp "$rootfs/etc/systemd/system/pm2-claude.service" "$staging_dir/etc/systemd/system/"

    # sudoers ファイル
    log "  - /etc/sudoers.d/claude"
    mkdir -p "$staging_dir/etc/sudoers.d"
    if [ -f "$rootfs/etc/sudoers.d/claude" ]; then
        cp "$rootfs/etc/sudoers.d/claude" "$staging_dir/etc/sudoers.d/"
        chmod 440 "$staging_dir/etc/sudoers.d/claude"
    fi

    log_success "ファイル抽出完了"
}

# ─── アーカイブ作成 ─────────────────────────────────────────────────────
create_migration_archive() {
    log "移行用アーカイブを作成中..."

    local staging_dir="${WORK_DIR}/staging"
    local archive_path="${WORK_DIR}/${MIGRATION_FILES_TAR}"

    cd "$staging_dir"
    tar -czf "$archive_path" \
        .gitconfig .npmrc .bashrc_custom ecosystem.config.cjs.modified \
        .ssh .pm2 .claude cclog \
        etc/ \
        2>&1 | tee -a "$LOG_FILE"

    if [ -f "$archive_path" ]; then
        local size=$(du -h "$archive_path" | cut -f1)
        log_success "アーカイブ完成: $archive_path (サイズ: $size)"
        echo "$archive_path"
    else
        log_error "アーカイブ作成に失敗しました。"
        exit 1
    fi
}

# ─── メイン処理 ────────────────────────────────────────────────────────
main() {
    log "=========================================="
    log "rootfs 抽出・ファイル集約開始"
    log "=========================================="

    init_work_dir

    # 引数からベースディレクトリ指定可能
    if [ $# -gt 0 ]; then
        BUSHIDAN_EXPORT_DIR="$1"
        log "引数から指定: BUSHIDAN_EXPORT_DIR=$BUSHIDAN_EXPORT_DIR"
    fi

    check_prerequisites
    extract_rootfs
    collect_migration_files
    local archive=$(create_migration_archive)

    log "=========================================="
    log_success "抽出完了"
    log "次のステップ:"
    log "  1. TrueNAS ホスト上へ以下をコピー:"
    log "     $archive"
    log "  2. 03_create_lxc.sh を実行"
    log "=========================================="

    # クリーンアップ確認
    log_warn "一時ファイルはそのままです。必要に応じて削除してください: $WORK_DIR"
}

main "$@"
