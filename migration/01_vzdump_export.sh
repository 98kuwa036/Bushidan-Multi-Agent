#!/bin/bash
# ============================================================================
# Proxmox LXC をエクスポート (vzdump → アーカイブ + TrueNAS転送)
# 実行場所: Proxmox ホスト
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ─── 前提条件チェック ────────────────────────────────────────────────────
check_prerequisites() {
    log "前提条件をチェック中..."

    check_command "pct" || {
        log_error "このスクリプトは Proxmox ホスト上で実行してください。"
        exit 1
    }

    check_command "vzdump" || {
        log_error "vzdump コマンドが見つかりません。"
        exit 1
    }

    check_command "ssh" || {
        log_error "ssh コマンドが見つかりません。"
        exit 1
    }

    # LXC ID の確認
    if ! pct status "$PROXMOX_LXC_ID" &> /dev/null; then
        log_error "LXC ID $PROXMOX_LXC_ID が見つかりません。"
        log "実行中のLXC一覧:"
        pct list
        exit 1
    fi

    log_success "前提条件OK"
}

# ─── LXC 停止 ────────────────────────────────────────────────────────
stop_lxc() {
    log "LXC コンテナを停止中..."

    if pct status "$PROXMOX_LXC_ID" | grep -q "running"; then
        log_warn "LXC $PROXMOX_LXC_ID は起動中です。一時停止を開始します。"
        pct suspend "$PROXMOX_LXC_ID"
        sleep 5
        log_success "LXC 一時停止完了"
    else
        log "LXC はすでに停止しています。"
    fi
}

# ─── LXC 再開 ────────────────────────────────────────────────────────
resume_lxc() {
    log "LXC コンテナを再開中..."
    if pct status "$PROXMOX_LXC_ID" | grep -q "suspended"; then
        pct resume "$PROXMOX_LXC_ID"
        sleep 5
        log_success "LXC 再開完了"
    fi
}

# ─── vzdump でアーカイブ作成 ─────────────────────────────────────────
create_backup() {
    log "vzdump でアーカイブを作成中..."

    local backup_file="/var/lib/vz/dump/${VZDUMP_FILE}"

    # 既存ファイル削除
    if [ -f "$backup_file" ]; then
        log_warn "既存のバックアップファイルを削除します: $backup_file"
        rm -f "$backup_file"
    fi

    # vzdump 実行 (zstd圧縮、スナップショット使用)
    log "vzdump を実行中... (圧縮可能なため数分かかります)"
    vzdump lxc "$PROXMOX_LXC_ID" \
        --compress zstd \
        --dumpdir "/var/lib/vz/dump" \
        --notes "Bushidan-Multi-Agent migration to TrueNAS SCALE" \
        --tmpdir "$WORK_DIR"

    if [ -f "$backup_file" ]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log_success "バックアップ完成: $backup_file (サイズ: $size)"
        echo "$backup_file"
    else
        log_error "バックアップファイルが見つかりません: $backup_file"
        exit 1
    fi
}

# ─── TrueNAS への転送準備 ──────────────────────────────────────────────
transfer_to_truenas() {
    local backup_file="$1"

    log "バックアップファイルをTrueNASへ転送中..."
    log "対象: $TRUENAS_IP"

    # SSH キーベース認証で接続可能か確認
    if ! ssh -o ConnectTimeout=5 "${TRUENAS_SSH_USER}@${TRUENAS_IP}" "echo 'SSH接続OK'" &> /dev/null; then
        log_error "TrueNAS ($TRUENAS_IP) への SSH 接続に失敗しました。"
        log "以下を確認してください:"
        log "  1. TrueNAS IP アドレスが正しいか"
        log "  2. SSH キーベース認証が設定済みか"
        log "  3. ファイアウォール設定が正しいか"
        exit 1
    fi

    # TrueNAS側の作業ディレクトリ作成
    ssh "${TRUENAS_SSH_USER}@${TRUENAS_IP}" "mkdir -p /tmp/bushidan_migration/backup"

    # scp で転送 (以下の場合、rsync 推奨)
    log "ファイル転送開始... (大きなファイルの場合は時間がかかります)"
    local start_time=$(date +%s)

    scp "$backup_file" "${TRUENAS_SSH_USER}@${TRUENAS_IP}:/tmp/bushidan_migration/backup/"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "転送完了 (所要時間: ${duration}秒)"

    # SHA256 チェックサム生成・検証
    log "チェックサムを検証中..."
    local local_checksum=$(sha256sum "$backup_file" | awk '{print $1}')
    log "ローカル checksum: $local_checksum"

    local remote_checksum=$(ssh "${TRUENAS_SSH_USER}@${TRUENAS_IP}" \
        "sha256sum /tmp/bushidan_migration/backup/$(basename "$backup_file")" | awk '{print $1}')
    log "リモート checksum: $remote_checksum"

    if [ "$local_checksum" = "$remote_checksum" ]; then
        log_success "チェックサム検証OK - ファイル完全性確認"
    else
        log_error "チェックサム不一致 - 転送エラーの可能性があります"
        exit 1
    fi
}

# ─── メイン処理 ────────────────────────────────────────────────────────
main() {
    log "=========================================="
    log "Proxmox LXC エクスポート開始"
    log "=========================================="

    init_work_dir
    check_prerequisites

    log "LXC ID: $PROXMOX_LXC_ID"
    log "ホスト名: $LXC_NAME"

    # エクスポート実行
    stop_lxc
    local backup_file=$(create_backup)
    resume_lxc

    # TrueNAS転送
    transfer_to_truenas "$backup_file"

    log "=========================================="
    log_success "エクスポート完了"
    log "次のステップ:"
    log "  1. TrueNAS ホスト上で 02_extract_files.sh を実行"
    log "  2. 出力の migration_files.tar.gz を確認"
    log "=========================================="
}

main "$@"
