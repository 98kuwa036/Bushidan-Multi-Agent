#!/bin/bash
# =============================================================
# Omni-P4 将軍システム v5.0 - Proxmox LXC Setup
# =============================================================
# Proxmoxホスト上で実行。CT 100 (本陣) と CT 101 (侍大将) を作成。
#
# Usage:
#   bash proxmox_setup.sh [storage]
#   bash proxmox_setup.sh local-lvm
# =============================================================

set -euo pipefail

STORAGE="${1:-local-lvm}"
TEMPLATE="local:vztmpl/ubuntu-22.04-standard_22.04-1_amd64.tar.zst"

echo "============================================="
echo "  Omni-P4 将軍システム - Proxmox Setup"
echo "  Storage: ${STORAGE}"
echo "============================================="

# --- HugePages 設定 (20GB) ---
echo "[1/6] HugePages 設定..."
if ! grep -q "vm.nr_hugepages" /etc/sysctl.conf; then
    echo "vm.nr_hugepages = 10240" >> /etc/sysctl.conf
    sysctl -p
    echo "  HugePages: 10240 (20GB)"
else
    echo "  HugePages: 既に設定済み"
fi

# --- CT 100: 本陣 (Controller) ---
echo ""
echo "[2/6] CT 100: 本陣 作成..."
if pct status 100 &>/dev/null; then
    echo "  CT 100 既に存在。スキップ。"
else
    pct create 100 "$TEMPLATE" \
        --hostname honmaru-control \
        --memory 2048 \
        --cores 2 \
        --rootfs "${STORAGE}:20" \
        --net0 name=eth0,bridge=vmbr0,ip=192.168.1.10/24,gw=192.168.1.1 \
        --onboot 1
    echo "  CT 100 作成完了"
fi

# --- CT 101: 侍大将R1 (OpenVINO) ---
echo ""
echo "[3/6] CT 101: 侍大将R1 作成..."
if pct status 101 &>/dev/null; then
    echo "  CT 101 既に存在。スキップ。"
else
    pct create 101 "$TEMPLATE" \
        --hostname taisho-openvino \
        --memory 20480 \
        --swap 0 \
        --cores 6 \
        --rootfs "${STORAGE}:50" \
        --net0 name=eth0,bridge=vmbr0,ip=192.168.1.11/24,gw=192.168.1.1 \
        --onboot 1
    echo "  CT 101 作成完了"
fi

# --- 最適化設定 ---
echo ""
echo "[4/6] 最適化設定..."

# CT 101: CPU最適化
pct set 101 --cputype host 2>/dev/null || true
echo "  CT 101: cputype=host"

# CT 101: メモリ固定 (balloon無効)
if ! grep -q "balloon" /etc/pve/lxc/101.conf 2>/dev/null; then
    echo "lxc.cgroup2.memory.max = 21474836480" >> /etc/pve/lxc/101.conf
fi

# HugePages マウント
if ! grep -q "hugepages" /etc/pve/lxc/101.conf 2>/dev/null; then
    echo "lxc.mount.entry: /dev/hugepages dev/hugepages none bind,create=dir 0 0" \
        >> /etc/pve/lxc/101.conf
    echo "  CT 101: HugePages マウント追加"
fi

# --- 起動 ---
echo ""
echo "[5/6] コンテナ起動..."
pct start 100 2>/dev/null || echo "  CT 100: 既に稼働中"
pct start 101 2>/dev/null || echo "  CT 101: 既に稼働中"

sleep 3

# --- 確認 ---
echo ""
echo "[6/6] 状態確認..."
pct list | grep -E "^(100|101)"

echo ""
echo "============================================="
echo "  Proxmox Setup 完了!"
echo ""
echo "  次のステップ:"
echo "    1. CT 101 で OpenVINO セットアップ:"
echo "       pct enter 101"
echo "       bash /path/to/shogun/setup/openvino_setup.sh"
echo ""
echo "    2. CT 100 で本陣セットアップ:"
echo "       pct enter 100"
echo "       bash /path/to/shogun/setup/install.sh"
echo "============================================="
