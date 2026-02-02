#!/bin/bash
# ============================================================================
# Bushidan Multi-Agent System v9.4 - Proxmox Container Setup
# ============================================================================
#
# Creates 2 LXC containers for v9.4 architecture:
#   CT 100 (本陣): 10GB - System orchestration, no model
#   CT 101 (Qwen3): 45GB - llama.cpp + Qwen3-Coder-30B (~17GB model)
#
# Prerequisites:
#   - Proxmox VE 8.x
#   - Ubuntu 22.04/24.04 template downloaded
#   - Sufficient storage (55GB+ total)
#
# Usage:
#   chmod +x setup/proxmox_setup_v94.sh
#   ./setup/proxmox_setup_v94.sh
#
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Configuration
# ============================================================================

# Container IDs
CT_HONIN=100      # 本陣 (Main base)
CT_QWEN3=101      # Qwen3 LLM container

# Storage
STORAGE="local-lvm"  # Change to your storage
TEMPLATE="local:vztmpl/ubuntu-22.04-standard_22.04-1_amd64.tar.zst"

# Network
BRIDGE="vmbr0"
GATEWAY="192.168.1.1"  # Change to your gateway
NETMASK="24"

# CT 100 (本陣) - Orchestration only
CT100_IP="192.168.1.100"  # Change to your IP
CT100_HOSTNAME="bushidan-honin"
CT100_DISK="10"   # GB
CT100_RAM="2048"  # MB
CT100_CORES="2"

# CT 101 (Qwen3) - LLM inference
CT101_IP="192.168.1.101"  # Change to your IP
CT101_HOSTNAME="bushidan-qwen3"
CT101_DISK="45"   # GB (17GB model + system)
CT101_RAM="24576" # MB (24GB for model loading)
CT101_CORES="8"   # All CPU cores for inference

# Root password (change this!)
ROOT_PASSWORD="bushidan2024"

# ============================================================================
# Functions
# ============================================================================

check_proxmox() {
    if ! command -v pct &> /dev/null; then
        log_error "pct command not found. Run this on Proxmox host."
        exit 1
    fi
    log_success "Proxmox environment detected"
}

check_template() {
    if ! pveam list local | grep -q "ubuntu-22.04"; then
        log_warning "Ubuntu 22.04 template not found. Downloading..."
        pveam download local ubuntu-22.04-standard_22.04-1_amd64.tar.zst
    fi
    log_success "Template available"
}

create_container() {
    local CTID=$1
    local HOSTNAME=$2
    local IP=$3
    local DISK=$4
    local RAM=$5
    local CORES=$6

    log_info "Creating container CT $CTID ($HOSTNAME)..."

    # Check if container exists
    if pct status $CTID &> /dev/null; then
        log_warning "Container CT $CTID already exists. Skipping..."
        return
    fi

    # Create container
    pct create $CTID $TEMPLATE \
        --hostname $HOSTNAME \
        --password "$ROOT_PASSWORD" \
        --storage $STORAGE \
        --rootfs ${STORAGE}:${DISK} \
        --memory $RAM \
        --cores $CORES \
        --net0 name=eth0,bridge=$BRIDGE,ip=${IP}/${NETMASK},gw=$GATEWAY \
        --features nesting=1 \
        --unprivileged 1 \
        --start 0

    log_success "Container CT $CTID created"
}

configure_ct100() {
    log_info "Configuring CT $CT_HONIN (本陣)..."

    # Start container
    pct start $CT_HONIN
    sleep 5

    # Update and install base packages
    pct exec $CT_HONIN -- bash -c "
        apt update && apt upgrade -y
        apt install -y python3-pip python3-venv git curl wget sudo

        # Create user
        useradd -m -s /bin/bash claude || true
        echo 'claude ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/claude

        # Clone repository
        su - claude -c 'git clone https://github.com/98kuwa036/Bushidan-Multi-Agent.git ~/Bushidan-Multi-Agent || true'

        # Run install script
        su - claude -c 'cd ~/Bushidan-Multi-Agent && bash setup/install.sh'
    "

    log_success "CT $CT_HONIN (本陣) configured"
}

configure_ct101() {
    log_info "Configuring CT $CT_QWEN3 (Qwen3)..."

    # Start container
    pct start $CT_QWEN3
    sleep 5

    # Update and install packages for llama.cpp build
    pct exec $CT_QWEN3 -- bash -c "
        apt update && apt upgrade -y
        apt install -y build-essential cmake git curl wget python3-pip

        # Create user
        useradd -m -s /bin/bash claude || true
        echo 'claude ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/claude

        # Clone repository
        su - claude -c 'git clone https://github.com/98kuwa036/Bushidan-Multi-Agent.git ~/Bushidan-Multi-Agent || true'

        # Run llama.cpp setup
        su - claude -c 'cd ~/Bushidan-Multi-Agent && chmod +x scripts/setup_llamacpp_prodesck600.sh'
    "

    log_success "CT $CT_QWEN3 (Qwen3) base configured"
    log_info "To complete llama.cpp setup, run inside CT $CT_QWEN3:"
    echo "  pct enter $CT_QWEN3"
    echo "  su - claude"
    echo "  cd ~/Bushidan-Multi-Agent && ./scripts/setup_llamacpp_prodesck600.sh"
}

show_summary() {
    echo ""
    echo "============================================================================"
    echo " Bushidan v9.4 - Proxmox Setup Complete"
    echo "============================================================================"
    echo ""
    echo "【コンテナ一覧】"
    echo "  CT $CT_HONIN (本陣): ${CT100_IP}"
    echo "    - Disk: ${CT100_DISK}GB"
    echo "    - RAM: ${CT100_RAM}MB"
    echo "    - CPU: ${CT100_CORES} cores"
    echo "    - 役割: System orchestration"
    echo ""
    echo "  CT $CT_QWEN3 (Qwen3): ${CT101_IP}"
    echo "    - Disk: ${CT101_DISK}GB"
    echo "    - RAM: ${CT101_RAM}MB"
    echo "    - CPU: ${CT101_CORES} cores"
    echo "    - 役割: llama.cpp + Qwen3-Coder-30B"
    echo ""
    echo "【次のステップ】"
    echo ""
    echo "  1. CT $CT_QWEN3 で llama.cpp セットアップを完了:"
    echo "     pct enter $CT_QWEN3"
    echo "     su - claude"
    echo "     cd ~/Bushidan-Multi-Agent"
    echo "     ./scripts/setup_llamacpp_prodesck600.sh"
    echo ""
    echo "  2. CT $CT_HONIN で API キーを設定:"
    echo "     pct enter $CT_HONIN"
    echo "     su - claude"
    echo "     cd ~/Bushidan-Multi-Agent"
    echo "     vim .env"
    echo ""
    echo "  3. システム起動:"
    echo "     # CT $CT_QWEN3 で llama.cpp サーバー起動"
    echo "     sudo systemctl start bushidan-llamacpp"
    echo ""
    echo "     # CT $CT_HONIN で Bushidan 起動"
    echo "     source .venv/bin/activate"
    echo "     python main.py"
    echo ""
    echo "【ストレージ使用量】"
    echo "  CT $CT_HONIN: ~3GB (OS + Python + MCP)"
    echo "  CT $CT_QWEN3: ~35GB (OS + llama.cpp + Model 17GB)"
    echo "  合計: ~38GB (余裕: ~17GB)"
    echo ""
    echo "============================================================================"
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo ""
    echo "============================================================================"
    echo " Bushidan Multi-Agent System v9.4 - Proxmox Setup"
    echo " BDI Framework + llama.cpp CPU最適化"
    echo "============================================================================"
    echo ""

    check_proxmox
    check_template

    echo ""
    log_info "Creating containers..."
    echo ""

    # Create containers
    create_container $CT_HONIN "$CT100_HOSTNAME" "$CT100_IP" "$CT100_DISK" "$CT100_RAM" "$CT100_CORES"
    create_container $CT_QWEN3 "$CT101_HOSTNAME" "$CT101_IP" "$CT101_DISK" "$CT101_RAM" "$CT101_CORES"

    echo ""
    log_info "Configuring containers..."
    echo ""

    # Configure containers
    configure_ct100
    configure_ct101

    show_summary
}

# Run
main "$@"
