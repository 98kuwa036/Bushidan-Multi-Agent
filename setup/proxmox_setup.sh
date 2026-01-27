#!/bin/bash
# =============================================================
# Omni-P4 Shogun-Hybrid - Proxmox LXC セットアップスクリプト
# =============================================================
# ProDesk 600 G4 (Core i5-8500, 24GB RAM, Proxmox VE)
#
# CT 100: 本陣 (Controller)   - 1GB RAM
# CT 101: 侍大将ノ間 (Mode A) - 22GB RAM
# CT 102: 足軽長屋 (Mode B)   - 18GB RAM
#
# Usage: bash proxmox_setup.sh [STORAGE] [TEMPLATE]
# =============================================================

set -euo pipefail

STORAGE="${1:-local-lvm}"
TEMPLATE="${2:-local:vztmpl/debian-12-standard_12.7-1_amd64.tar.zst}"
BRIDGE="vmbr0"
GATEWAY="192.168.1.1"
NAMESERVER="1.1.1.1"

echo "============================================="
echo "  Omni-P4 Shogun-Hybrid Proxmox Setup"
echo "============================================="

# --- CT 100: 本陣 (Controller) ---
echo ""
echo "[1/3] Creating CT 100: 本陣 (Controller)..."
pct create 100 "$TEMPLATE" \
  --hostname honin-controller \
  --memory 1024 \
  --swap 512 \
  --cores 2 \
  --rootfs "${STORAGE}:8" \
  --net0 "name=eth0,bridge=${BRIDGE},ip=192.168.1.10/24,gw=${GATEWAY}" \
  --nameserver "$NAMESERVER" \
  --features nesting=1 \
  --unprivileged 1 \
  --start 0 \
  --onboot 1 \
  --description "Omni-P4 本陣: Task queue, mode switching, API gateway"

echo "  CT 100 created."

# --- CT 101: 侍大将ノ間 (Mode A) ---
echo ""
echo "[2/3] Creating CT 101: 侍大将ノ間 (Mode A - Deep Thinking)..."
pct create 101 "$TEMPLATE" \
  --hostname taisho-chamber \
  --memory 22528 \
  --swap 2048 \
  --cores 6 \
  --rootfs "${STORAGE}:30" \
  --net0 "name=eth0,bridge=${BRIDGE},ip=192.168.1.11/24,gw=${GATEWAY}" \
  --nameserver "$NAMESERVER" \
  --features nesting=1 \
  --unprivileged 1 \
  --start 0 \
  --onboot 0 \
  --description "Omni-P4 侍大将ノ間: DeepSeek-R1-14B-JP (Q8_0, 32k ctx)"

echo "  CT 101 created."

# --- CT 102: 足軽長屋 (Mode B) ---
echo ""
echo "[3/3] Creating CT 102: 足軽長屋 (Mode B - Action)..."
pct create 102 "$TEMPLATE" \
  --hostname ashigaru-barracks \
  --memory 18432 \
  --swap 2048 \
  --cores 6 \
  --rootfs "${STORAGE}:30" \
  --net0 "name=eth0,bridge=${BRIDGE},ip=192.168.1.12/24,gw=${GATEWAY}" \
  --nameserver "$NAMESERVER" \
  --features nesting=1 \
  --unprivileged 1 \
  --start 0 \
  --onboot 0 \
  --description "Omni-P4 足軽長屋: Hermes-8B + Qwen-Coder-7B + Qwen-1.5B"

echo "  CT 102 created."

echo ""
echo "============================================="
echo "  All containers created successfully!"
echo ""
echo "  Next steps:"
echo "  1. Start CT 100: pct start 100"
echo "  2. Install Ollama on CT 101 & 102:"
echo "     pct exec 101 -- bash /path/to/ollama_setup.sh"
echo "     pct exec 102 -- bash /path/to/ollama_setup.sh"
echo "  3. Install Controller on CT 100:"
echo "     pct exec 100 -- bash /path/to/install.sh"
echo "============================================="
