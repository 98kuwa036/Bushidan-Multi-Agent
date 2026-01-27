#!/bin/bash
# =============================================================
# Omni-P4 Shogun-Hybrid - Ollama Setup Script
# =============================================================
# Run inside CT 101 (Mode A) or CT 102 (Mode B) to install
# Ollama and download required models.
#
# Usage:
#   bash ollama_setup.sh [mode_a|mode_b|all]
# =============================================================

set -euo pipefail

MODE="${1:-all}"

echo "============================================="
echo "  Ollama Setup - Mode: ${MODE}"
echo "============================================="

# --- Install Ollama ---
if ! command -v ollama &> /dev/null; then
    echo "[1] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Ollama installed."
else
    echo "[1] Ollama already installed: $(ollama --version)"
fi

# --- Start Ollama service ---
echo "[2] Starting Ollama service..."
if systemctl is-active --quiet ollama 2>/dev/null; then
    echo "  Already running."
else
    # Try systemd first, fallback to direct
    if systemctl start ollama 2>/dev/null; then
        echo "  Started via systemd."
    else
        echo "  Starting directly..."
        nohup ollama serve > /var/log/ollama.log 2>&1 &
        sleep 3
        echo "  Started."
    fi
fi

# Wait for ready
echo "[3] Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/ > /dev/null 2>&1; then
        echo "  Ollama is ready."
        break
    fi
    sleep 1
done

# --- Download Models ---
echo ""
echo "[4] Downloading models for mode: ${MODE}"

download_model() {
    local model="$1"
    local desc="$2"
    echo "  Downloading: ${desc} (${model})..."
    ollama pull "$model"
    echo "  Done: ${model}"
}

if [ "$MODE" = "mode_a" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "--- Mode A Models (侍大将) ---"
    # Note: For Q8_0, you may need to import from GGUF manually:
    # ollama create taisho-r1-14b-jp -f /path/to/taisho.Modelfile
    download_model "cyberagent/deepseek-r1-distill-qwen-14b-japanese" "侍大将 (Taisho) - DeepSeek-R1-14B-JP"
fi

if [ "$MODE" = "mode_b" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "--- Mode B Models (足軽隊) ---"
    download_model "nous-hermes2:latest" "足軽頭 (Leader) - Hermes-3-8B"
    download_model "qwen2.5-coder:7b" "技術兵 (Coder) - Qwen2.5-Coder-7B"
    download_model "qwen2.5:1.5b" "小者 (Scout) - Qwen2.5-1.5B"
fi

echo ""
echo "============================================="
echo "  Model download complete!"
echo ""
echo "  Installed models:"
ollama list
echo ""
echo "  To create custom Modelfiles:"
echo "  ollama create taisho-r1-14b-jp -f config/modelfiles/taisho.Modelfile"
echo "  ollama create leader-hermes-8b -f config/modelfiles/leader.Modelfile"
echo "  ollama create coder-qwen-7b -f config/modelfiles/coder.Modelfile"
echo "  ollama create scout-qwen-1.5b -f config/modelfiles/scout.Modelfile"
echo "============================================="
