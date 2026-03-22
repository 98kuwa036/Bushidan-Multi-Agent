#!/bin/bash
# ============================================================================
# TrueNAS SCALE で新しい LXC コンテナを作成・初期化
# 実行場所: TrueNAS SCALE ホスト
# 前提: incus または lxc コマンドが利用可能
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ─── 前提条件チェック ────────────────────────────────────────────────────
check_prerequisites() {
    log "前提条件をチェック中..."

    # incus or lxc コマンド確認
    if command -v incus &> /dev/null; then
        LXC_CMD="incus"
        log_success "incus コマンドを使用します"
    elif command -v lxc &> /dev/null; then
        LXC_CMD="lxc"
        log_success "lxc コマンドを使用します"
    else
        log_error "incus or lxc コマンドが見つかりません。"
        log "TrueNAS SCALE 24.10+ では incus がデフォルトです。"
        exit 1
    fi

    check_command "tar" || exit 1
    check_command "ssh" || exit 1

    # 移行ファイルの確認
    if [ ! -f "${WORK_DIR}/${MIGRATION_FILES_TAR}" ]; then
        log_error "移行ファイルが見つかりません: ${WORK_DIR}/${MIGRATION_FILES_TAR}"
        exit 1
    fi

    log_success "前提条件OK"
}

# ─── LXC イメージ準備 ───────────────────────────────────────────────────
prepare_image() {
    log "LXC イメージを準備中..."

    # Ubuntu 22.04 イメージを確認・取得
    local image_name="ubuntu-22.04"

    # 既存イメージ確認
    if $LXC_CMD image list | grep -q "$image_name"; then
        log "イメージ $image_name は既に利用可能です"
        return 0
    fi

    log "イメージ $image_name を取得中... (初回のみ)"
    $LXC_CMD image import images:ubuntu/22.04/cloud --alias "$image_name" 2>&1 | tee -a "$LOG_FILE"

    if $LXC_CMD image list | grep -q "$image_name"; then
        log_success "イメージ取得完了"
    else
        log_error "イメージの取得に失敗しました。"
        exit 1
    fi
}

# ─── LXC コンテナ作成 ────────────────────────────────────────────────────
create_lxc_container() {
    log "LXC コンテナを作成中..."
    log "  名前: $LXC_NAME"
    log "  IP: $LXC_IP"
    log "  CPU: $LXC_CPU_CORES, RAM: ${LXC_RAM_GB}GB, Disk: ${LXC_DISK_GB}GB"

    # コンテナ存在確認
    if $LXC_CMD list | grep -q "$LXC_NAME"; then
        log_warn "コンテナ $LXC_NAME は既に存在します。削除してから再作成します。"
        log "削除確認..."
        read -p "削除して進めますか？ (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $LXC_CMD delete -f "$LXC_NAME" 2>&1 | tee -a "$LOG_FILE"
            sleep 3
        else
            log_warn "中止しました。"
            return 1
        fi
    fi

    # コンテナ作成
    $LXC_CMD launch "ubuntu:22.04" "$LXC_NAME" \
        --config limits.cpu="$LXC_CPU_CORES" \
        --config limits.memory="${LXC_RAM_GB}GB" \
        2>&1 | tee -a "$LOG_FILE"

    # 起動待機
    log "コンテナの起動を待機中..."
    local attempts=0
    while [ $attempts -lt 30 ]; do
        if $LXC_CMD exec "$LXC_NAME" -- ping -c 1 8.8.8.8 &> /dev/null; then
            log_success "コンテナ起動完了"
            return 0
        fi
        sleep 1
        ((attempts++))
    done

    log_error "コンテナの起動がタイムアウトしました。"
    return 1
}

# ─── ネットワーク設定 ────────────────────────────────────────────────────
configure_network() {
    log "ネットワークを設定中..."

    # 静的 IP 設定 (Netplan経由)
    log "静的IP設定: $LXC_IP"

    # cloud-init を無効化
    $LXC_CMD exec "$LXC_NAME" -- cloud-init disable

    # netplan 設定
    cat > /tmp/99-custom-network.yaml << EOF
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - $LXC_IP/24
      gateway4: $LXC_GATEWAY
      nameservers:
        addresses:
          - $LXC_DNS

EOF

    $LXC_CMD file push /tmp/99-custom-network.yaml \
        "$LXC_NAME"/etc/netplan/99-custom.yaml

    $LXC_CMD exec "$LXC_NAME" -- netplan apply

    # IP 確認
    sleep 2
    local ip=$($LXC_CMD exec "$LXC_NAME" -- ip -4 addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    if [ "$ip" = "${LXC_IP%/*}" ]; then
        log_success "ネットワーク設定OK: $ip"
    else
        log_warn "IP設定確認: $ip (期待値: ${LXC_IP%/*})"
    fi
}

# ─── 移行ファイルのコピー ───────────────────────────────────────────────
copy_migration_files() {
    log "移行ファイルをコンテナにコピー中..."

    local src="${WORK_DIR}/${MIGRATION_FILES_TAR}"
    local dst="/tmp/migration_files.tar.gz"

    $LXC_CMD file push "$src" "$LXC_NAME$dst"

    log_success "ファイルコピー完了"
}

# ─── セットアップスクリプト配置 ────────────────────────────────────────
deploy_setup_script() {
    log "セットアップスクリプトをコンテナにコピー中..."

    # 04_setup_environment.sh をコピー
    $LXC_CMD file push "${SCRIPT_DIR}/04_setup_environment.sh" \
        "$LXC_NAME"/tmp/04_setup_environment.sh

    $LXC_CMD file push "${SCRIPT_DIR}/05_verify.py" \
        "$LXC_NAME"/tmp/05_verify.py

    # 実行権限付与
    $LXC_CMD exec "$LXC_NAME" -- chmod +x /tmp/04_setup_environment.sh /tmp/05_verify.py

    log_success "セットアップスクリプト配置完了"
}

# ─── セットアップ実行の案内 ──────────────────────────────────────────────
print_next_steps() {
    cat << EOF

========================================
LXC コンテナ作成完了
========================================

コンテナ名: $LXC_NAME
IP: $LXC_IP
CPU: $LXC_CPU_CORES cores
RAM: ${LXC_RAM_GB}GB
Disk: ${LXC_DISK_GB}GB

次のステップ:

【方法 1】スクリプトで自動セットアップ:
  $LXC_CMD exec $LXC_NAME -- bash /tmp/04_setup_environment.sh

【方法 2】手動でコンテナにログイン:
  $LXC_CMD exec $LXC_NAME -- bash
  # コンテナ内で以下を実行
  bash /tmp/04_setup_environment.sh

移行ファイルは以下に配置済み:
  /tmp/$MIGRATION_FILES_TAR

セットアップ後の確認:
  $LXC_CMD exec $LXC_NAME -- python3 /tmp/05_verify.py

========================================

注意事項:
- Bushidan-Multi-Agent と Bushidan ディレクトリは別途 rsync でコピーしてください
- .env ファイル (APIキー含む) は別途セキュアにコピーしてください

例:
  rsync -avz /path/to/Bushidan-Multi-Agent root@$LXC_IP:/home/claude/
  scp /path/to/.env root@$LXC_IP:/home/claude/Bushidan-Multi-Agent/

EOF
}

# ─── メイン処理 ────────────────────────────────────────────────────────
main() {
    log "=========================================="
    log "TrueNAS SCALE LXC コンテナ作成開始"
    log "=========================================="

    init_work_dir
    check_required_vars || exit 1
    check_prerequisites

    prepare_image
    create_lxc_container || exit 1
    configure_network
    copy_migration_files
    deploy_setup_script

    print_next_steps

    log_success "LXC 作成スクリプト完了"
}

main "$@"
