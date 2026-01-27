#!/bin/bash
# =============================================================
# Omni-P4 Shogun-Hybrid - Controller Installation Script
# =============================================================
# Install the Python controller on CT 100 (or any host).
#
# Usage: bash install.sh
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "  Omni-P4 Shogun-Hybrid Installation"
echo "  Project: ${PROJECT_DIR}"
echo "============================================="

# --- Python Check ---
echo "[1] Checking Python..."
if command -v python3 &> /dev/null; then
    PY="python3"
elif command -v python &> /dev/null; then
    PY="python"
else
    echo "  Python not found. Installing..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y python3 python3-pip python3-venv
    elif command -v pacman &> /dev/null; then
        pacman -S --noconfirm python python-pip
    elif command -v emerge &> /dev/null; then
        emerge --ask n dev-lang/python
    fi
    PY="python3"
fi
echo "  Python: $($PY --version)"

# --- Virtual Environment ---
echo ""
echo "[2] Setting up virtual environment..."
VENV_DIR="${PROJECT_DIR}/.venv"
if [ ! -d "$VENV_DIR" ]; then
    $PY -m venv "$VENV_DIR"
    echo "  Created: ${VENV_DIR}"
else
    echo "  Already exists: ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

# --- Install Dependencies ---
echo ""
echo "[3] Installing dependencies..."
pip install -r "${PROJECT_DIR}/requirements.txt"

# --- Create CLI symlink ---
echo ""
echo "[4] Creating CLI shortcut..."
CLI_SCRIPT="${PROJECT_DIR}/shogun/cli.py"
chmod +x "$CLI_SCRIPT"

# Create wrapper script
cat > "${VENV_DIR}/bin/shogun" << 'WRAPPER_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$VENV_DIR")"
source "${VENV_DIR}/bin/activate"
cd "$PROJECT_DIR"
python -m shogun.cli "$@"
WRAPPER_EOF
chmod +x "${VENV_DIR}/bin/shogun"
echo "  CLI installed: ${VENV_DIR}/bin/shogun"

# --- Environment Setup ---
echo ""
echo "[5] Environment configuration..."
ENV_FILE="${PROJECT_DIR}/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'ENV_EOF'
# Omni-P4 Shogun-Hybrid Environment
# Set your Anthropic API key for cloud agents (Shogun/Karo)
# ANTHROPIC_API_KEY=sk-ant-xxxxx

# Ollama endpoint (default: localhost)
# OLLAMA_BASE_URL=http://localhost:11434
ENV_EOF
    echo "  Created .env template: ${ENV_FILE}"
    echo "  Edit this file to set your ANTHROPIC_API_KEY"
else
    echo "  .env already exists."
fi

echo ""
echo "============================================="
echo "  Installation complete!"
echo ""
echo "  Quick start:"
echo "    source ${VENV_DIR}/bin/activate"
echo "    shogun health       # Check Ollama"
echo "    shogun status       # System status"
echo "    shogun repl         # Interactive mode"
echo "    shogun ask 'Hello'  # Quick ask"
echo "    shogun server       # Start API server"
echo ""
echo "  For IDE integration, add to shell profile:"
echo "    export PATH=\"${VENV_DIR}/bin:\$PATH\""
echo "============================================="
