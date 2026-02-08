#!/bin/bash
# ============================================================================
# 武士団マルチエージェントシステム v10.1 - Proxmox コンテナセットアップ
# ============================================================================
#
# 【コア要件】必ず達成すべきこと:
#   1. CT 100 (本陣) の作成: 10GB - System orchestration
#   2. CT 101 (Qwen3) の作成: 45GB - llama.cpp + モデル
#
# 【最終結果】このスクリプト完了後の状態:
#   - CT 100: Ubuntu + Python + Node.js + MCP 設定
#   - CT 101: Ubuntu + build-essential + llama.cpp 準備
#   - 両コンテナで claude ユーザーが使用可能
#
# 【動作保証】
#   - Proxmox VE 環境で pct コマンドが使用可能なら異常終了しない
#   - 既存コンテナがあればスキップ
#   - ネットワーク設定はカスタマイズ可能
#
# Prerequisites:
#   - Proxmox VE 8.x
#   - Ubuntu 22.04/24.04 template
#   - 55GB+ storage
#
# Usage:
#   ./setup/proxmox_setup_v94.sh              # 対話モード
#   ./setup/proxmox_setup_v94.sh --auto       # 自動モード
#   ./setup/proxmox_setup_v94.sh --ct100-only # CT 100 のみ
#   ./setup/proxmox_setup_v94.sh --ct101-only # CT 101 のみ
#
# ============================================================================

# エラー時に即座に終了しない
# set -e は使わない

# ============================================================================
# 設定 (必要に応じてカスタマイズ)
# ============================================================================

# Container IDs
CT_HONIN=100      # 本陣 (Main base)
CT_QWEN3=101      # Qwen3 LLM container

# Storage
STORAGE="local-lvm"  # Change to your storage
TEMPLATE="local:vztmpl/ubuntu-22.04-standard_22.04-1_amd64.tar.zst"

# Network
BRIDGE="vmbr0"
GATEWAY="192.168.11.1"  # Change to your gateway
NETMASK="24"
DNS_PRIMARY="8.8.8.8"    # Google DNS
DNS_SECONDARY="8.8.4.4"  # Google DNS backup

# CT 100 (本陣) - Orchestration only
CT100_IP="192.168.11.231"  # Change to your IP
CT100_HOSTNAME="bushidan-honin"
CT100_DISK="10"   # GB
CT100_RAM="2048"  # MB
CT100_CORES="2"

# CT 101 (Qwen3) - LLM inference
CT101_IP="192.168.11.232"  # Change to your IP
CT101_HOSTNAME="bushidan-qwen3"
CT101_DISK="45"   # GB (17GB model + system)
CT101_RAM="24576" # MB (24GB for model loading)
CT101_CORES="6"   # HP ProDesk 600: i5-8500 (6C/6T)

# Root password
ROOT_PASSWORD="bushidan2024"

# 実行モード
AUTO_MODE=0
CT100_ONLY=0
CT101_ONLY=0

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

# ============================================================================
# 前提条件チェック
# ============================================================================

check_proxmox() {
    log_info "Proxmox 環境を確認中..."

    if ! command -v pct &>/dev/null; then
        log_error "pct コマンドが見つかりません"
        log_info "このスクリプトは Proxmox VE ホストで実行してください"
        return 1
    fi

    if ! command -v pvesm &>/dev/null; then
        log_error "pvesm コマンドが見つかりません"
        return 1
    fi

    mark_success "proxmox_env"
    log_success "Proxmox 環境検出"
    return 0
}

check_template() {
    log_info "テンプレートを確認中..."

    if pveam list local 2>/dev/null | grep -q "ubuntu-22.04"; then
        mark_success "template"
        log_success "Ubuntu 22.04 テンプレート検出"
        return 0
    fi

    log_warning "Ubuntu 22.04 テンプレートが見つかりません"

    if [ $AUTO_MODE -eq 1 ]; then
        log_info "テンプレートをダウンロード中..."
        if pveam download local ubuntu-22.04-standard_22.04-1_amd64.tar.zst 2>/dev/null; then
            mark_success "template"
            log_success "テンプレートダウンロード完了"
            return 0
        else
            log_error "テンプレートダウンロード失敗"
            mark_failed "template" "core"
            return 1
        fi
    else
        log_info "以下のコマンドでダウンロードできます:"
        echo "  pveam download local ubuntu-22.04-standard_22.04-1_amd64.tar.zst"
        mark_skipped "template"
        return 1
    fi
}

# ============================================================================
# コンテナ作成
# ============================================================================

create_container() {
    local CTID=$1
    local HOSTNAME=$2
    local IP=$3
    local DISK=$4
    local RAM=$5
    local CORES=$6
    local DESC=$7

    log_info "CT $CTID ($DESC) を作成中..."

    # 既存チェック
    if pct status $CTID &>/dev/null; then
        log_warning "CT $CTID は既に存在します。スキップ"
        mark_skipped "ct${CTID}_create"
        return 0
    fi

    # コンテナ作成
    if pct create $CTID $TEMPLATE \
        --hostname $HOSTNAME \
        --password "$ROOT_PASSWORD" \
        --storage $STORAGE \
        --rootfs ${STORAGE}:${DISK} \
        --memory $RAM \
        --cores $CORES \
        --net0 name=eth0,bridge=$BRIDGE,ip=${IP}/${NETMASK},gw=$GATEWAY \
        --nameserver "${DNS_PRIMARY} ${DNS_SECONDARY}" \
        --features nesting=1 \
        --unprivileged 1 \
        --start 0 2>/dev/null; then
        mark_success "ct${CTID}_create"
        log_success "CT $CTID 作成完了"
        return 0
    else
        log_error "CT $CTID 作成失敗"
        mark_failed "ct${CTID}_create" "core"
        return 1
    fi
}

# ============================================================================
# CT 100 (本陣) 設定
# ============================================================================

configure_ct100() {
    log_info "CT $CT_HONIN (本陣) を設定中..."

    # コンテナ起動
    if ! pct status $CT_HONIN 2>/dev/null | grep -q "running"; then
        pct start $CT_HONIN 2>/dev/null || true
        sleep 5
    fi

    # 基本セットアップ
    local setup_result
    setup_result=$(pct exec $CT_HONIN -- bash -c '
        # パッケージ更新
        apt update && apt upgrade -y 2>/dev/null

        # 基本パッケージ
        apt install -y python3-pip python3-venv git curl wget sudo ca-certificates gnupg 2>/dev/null

        # Node.js インストール (MCP サーバー用)
        if ! command -v node &>/dev/null; then
            mkdir -p /etc/apt/keyrings
            curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg 2>/dev/null
            echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list
            apt update 2>/dev/null
            apt install -y nodejs 2>/dev/null
            echo "NODEJS_INSTALLED"
        fi

        # claude ユーザー作成
        if ! id claude &>/dev/null; then
            useradd -m -s /bin/bash claude
            echo "claude ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/claude
            echo "USER_CREATED"
        else
            echo "USER_EXISTS"
        fi
    ' 2>&1)

    if echo "$setup_result" | grep -q "USER_"; then
        mark_success "ct100_base"
        log_success "CT $CT_HONIN 基本設定完了"
    else
        log_warning "CT $CT_HONIN 基本設定に問題がありました"
        mark_failed "ct100_base"
    fi

    # リポジトリクローン
    pct exec $CT_HONIN -- su - claude -c '
        if [ ! -d ~/Bushidan-Multi-Agent ]; then
            git clone https://github.com/98kuwa036/Bushidan-Multi-Agent.git ~/Bushidan-Multi-Agent 2>/dev/null
        fi
    ' 2>/dev/null || true

    log_success "CT $CT_HONIN (本陣) 設定完了"
    mark_success "ct100_config"
}

# ============================================================================
# CT 101 (Qwen3) 設定
# ============================================================================

configure_ct101() {
    log_info "CT $CT_QWEN3 (Qwen3) を設定中..."

    # コンテナ起動
    if ! pct status $CT_QWEN3 2>/dev/null | grep -q "running"; then
        pct start $CT_QWEN3 2>/dev/null || true
        sleep 5
    fi

    # 基本セットアップ
    local setup_result
    setup_result=$(pct exec $CT_QWEN3 -- bash -c '
        # パッケージ更新
        apt update && apt upgrade -y 2>/dev/null

        # ビルド用パッケージ
        apt install -y build-essential cmake git curl wget python3-pip 2>/dev/null

        # claude ユーザー作成
        if ! id claude &>/dev/null; then
            useradd -m -s /bin/bash claude
            echo "claude ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/claude
            echo "USER_CREATED"
        else
            echo "USER_EXISTS"
        fi
    ' 2>&1)

    if echo "$setup_result" | grep -q "USER_"; then
        mark_success "ct101_base"
        log_success "CT $CT_QWEN3 基本設定完了"
    else
        log_warning "CT $CT_QWEN3 基本設定に問題がありました"
        mark_failed "ct101_base"
    fi

    # リポジトリクローン
    pct exec $CT_QWEN3 -- su - claude -c '
        if [ ! -d ~/Bushidan-Multi-Agent ]; then
            git clone https://github.com/98kuwa036/Bushidan-Multi-Agent.git ~/Bushidan-Multi-Agent 2>/dev/null
        fi
        chmod +x ~/Bushidan-Multi-Agent/scripts/setup_llamacpp_prodesck600.sh 2>/dev/null
    ' 2>/dev/null || true

    log_success "CT $CT_QWEN3 (Qwen3) 基本設定完了"
    mark_success "ct101_config"
}

# ============================================================================
# 最終レポート
# ============================================================================

print_final_report() {
    echo ""
    echo "============================================="
    echo "  Proxmox セットアップ - 完了レポート"
    echo "============================================="
    echo ""

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

    if [ $CORE_FAILED -eq 1 ]; then
        echo "【状態】コア機能に失敗があります"
        return 1
    fi

    echo "【状態】正常完了"
    echo ""
    echo "【コンテナ一覧】"
    echo "  CT $CT_HONIN (本陣): ${CT100_IP}"
    echo "    Disk: ${CT100_DISK}GB, RAM: ${CT100_RAM}MB, CPU: ${CT100_CORES} cores"
    echo "    役割: System orchestration"
    echo ""
    echo "  CT $CT_QWEN3 (Qwen3): ${CT101_IP}"
    echo "    Disk: ${CT101_DISK}GB, RAM: ${CT101_RAM}MB, CPU: ${CT101_CORES} cores"
    echo "    役割: llama.cpp + Qwen3-Coder-30B"
    echo ""
    echo "【次のステップ】"
    echo ""
    echo "  1. CT $CT_HONIN で本陣セットアップ:"
    echo "     pct enter $CT_HONIN"
    echo "     su - claude"
    echo "     cd ~/Bushidan-Multi-Agent"
    echo "     bash setup/install.sh --user"
    echo ""
    echo "  2. CT $CT_QWEN3 で llama.cpp セットアップ:"
    echo "     pct enter $CT_QWEN3"
    echo "     su - claude"
    echo "     cd ~/Bushidan-Multi-Agent"
    echo "     ./scripts/setup_llamacpp_prodesck600.sh"
    echo ""
    echo "【ストレージ使用量】"
    echo "  CT $CT_HONIN: ~3GB (OS + Python + MCP)"
    echo "  CT $CT_QWEN3: ~35GB (OS + llama.cpp + Model 17GB)"
    echo "  合計: ~38GB (余裕: ~17GB)"
    echo ""
    echo "============================================="
}

# ============================================================================
# メイン処理
# ============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --auto        自動モード (確認なしで実行)"
    echo "  --ct100-only  CT 100 (本陣) のみ作成"
    echo "  --ct101-only  CT 101 (Qwen3) のみ作成"
    echo "  --help        このヘルプを表示"
    echo ""
    echo "【コア要件】"
    echo "  1. CT 100 (本陣): 10GB - System orchestration"
    echo "  2. CT 101 (Qwen3): 45GB - llama.cpp + モデル"
    echo ""
    echo "【設定変更】"
    echo "  スクリプト冒頭の設定セクションを編集してください:"
    echo "  - CT100_IP, CT101_IP: IPアドレス"
    echo "  - GATEWAY: デフォルトゲートウェイ"
    echo "  - STORAGE: ストレージ名"
    echo "  - ROOT_PASSWORD: rootパスワード"
}

run_full_setup() {
    echo ""
    echo "============================================="
    echo "  武士団 v10.1 - Proxmox セットアップ"
    echo "  BDI + Kimi K2.5 + 検校 + MCP権限マトリクス"
    echo "============================================="
    echo ""

    # 前提条件チェック
    if ! check_proxmox; then
        CORE_FAILED=1
        print_final_report
        exit 1
    fi

    if ! check_template; then
        CORE_FAILED=1
        print_final_report
        exit 1
    fi

    echo ""
    log_info "コンテナを作成中..."
    echo ""

    # CT 100 (本陣)
    if [ $CT101_ONLY -eq 0 ]; then
        create_container $CT_HONIN "$CT100_HOSTNAME" "$CT100_IP" "$CT100_DISK" "$CT100_RAM" "$CT100_CORES" "本陣"
    fi

    # CT 101 (Qwen3)
    if [ $CT100_ONLY -eq 0 ]; then
        create_container $CT_QWEN3 "$CT101_HOSTNAME" "$CT101_IP" "$CT101_DISK" "$CT101_RAM" "$CT101_CORES" "Qwen3"
    fi

    echo ""
    log_info "コンテナを設定中..."
    echo ""

    # 設定
    if [ $CT101_ONLY -eq 0 ]; then
        configure_ct100
    fi

    if [ $CT100_ONLY -eq 0 ]; then
        configure_ct101
    fi

    print_final_report
}

# ============================================================================
# エントリーポイント
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            AUTO_MODE=1
            shift
            ;;
        --ct100-only)
            CT100_ONLY=1
            shift
            ;;
        --ct101-only)
            CT101_ONLY=1
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            log_error "不明なオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

run_full_setup

exit $CORE_FAILED
