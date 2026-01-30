#!/bin/bash
# setup_r1_japanese.sh - Enhanced Japanese R1 Setup for Shogun System v7.0
#
# Sets up cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese with OpenVINO
# For Taisho (侍大将) on CT 101 (192.168.1.11)
#
# Usage: Run inside CT 101 (侍大将専用LXC)
# Requires: 20GB RAM, 6 CPU cores, HugePages configured
#
# Enhanced with comprehensive error handling and recovery mechanisms

set -euo pipefail  # Enhanced error handling

# Global variables for error handling
LOG_FILE="/var/log/taisho-setup.log"
BACKUP_DIR="/opt/backup-$(date +%Y%m%d-%H%M%S)"
RECOVERY_INFO_FILE="/opt/recovery_info.json"

# Enhanced logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Color codes
    local RED='\033[0;31m'
    local GREEN='\033[0;32m'
    local YELLOW='\033[1;33m'
    local BLUE='\033[0;34m'
    local NC='\033[0m' # No Color
    
    case $level in
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "INFO")
            echo -e "${GREEN}[INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        *)
            echo "[$timestamp] $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Error handler function
error_handler() {
    local exit_code=$?
    local line_number=$1
    
    log "ERROR" "Setup failed at line $line_number with exit code $exit_code"
    log "ERROR" "Last command: $BASH_COMMAND"
    
    # Save recovery information
    cat > "$RECOVERY_INFO_FILE" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "exit_code": $exit_code,
    "failed_line": $line_number,
    "failed_command": "$BASH_COMMAND",
    "backup_dir": "$BACKUP_DIR",
    "log_file": "$LOG_FILE"
}
EOF
    
    log "INFO" "Recovery information saved to $RECOVERY_INFO_FILE"
    log "INFO" "Log file available at $LOG_FILE"
    
    # Attempt automatic recovery for common issues
    attempt_recovery $exit_code
    
    exit $exit_code
}

# Set error trap
trap 'error_handler ${LINENO}' ERR

# Recovery function for common issues
attempt_recovery() {
    local exit_code=$1
    
    log "INFO" "Attempting automatic recovery for exit code $exit_code..."
    
    case $exit_code in
        130)
            log "INFO" "Process interrupted by user - cleanup not needed"
            ;;
        1)
            log "INFO" "General error - checking for partial installations"
            cleanup_partial_installation
            ;;
        2)
            log "INFO" "Command not found - checking package installation"
            check_missing_packages
            ;;
        *)
            log "WARN" "No specific recovery procedure for exit code $exit_code"
            ;;
    esac
}

# Cleanup partial installations
cleanup_partial_installation() {
    log "INFO" "Cleaning up partial installation..."
    
    # Stop any running services
    if systemctl is-active --quiet taisho-japanese-r1 2>/dev/null; then
        systemctl stop taisho-japanese-r1 || true
    fi
    
    # Remove incomplete model downloads
    if [ -d "/opt/models/deepseek-r1-japanese" ] && [ ! -f "/opt/models/deepseek-r1-japanese/.download_complete" ]; then
        log "WARN" "Removing incomplete model download"
        rm -rf /opt/models/deepseek-r1-japanese
    fi
    
    log "INFO" "Partial installation cleanup complete"
}

# Check for missing packages
check_missing_packages() {
    log "INFO" "Checking for missing packages..."
    
    local required_packages=("python3" "python3-pip" "python3-venv" "git" "curl" "wget")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        log "WARN" "Missing packages detected: ${missing_packages[*]}"
        log "INFO" "Attempting to install missing packages..."
        apt update && apt install -y "${missing_packages[@]}" || {
            log "ERROR" "Failed to install missing packages"
            return 1
        }
    fi
}

# Create backup of existing installation
create_backup() {
    if [ -d "/opt/openvino-env" ] || [ -d "/opt/models" ] || [ -f "/opt/taisho_server.py" ]; then
        log "INFO" "Creating backup of existing installation..."
        mkdir -p "$BACKUP_DIR"
        
        [ -d "/opt/openvino-env" ] && cp -r /opt/openvino-env "$BACKUP_DIR/" || true
        [ -d "/opt/models" ] && cp -r /opt/models "$BACKUP_DIR/" || true  
        [ -f "/opt/taisho_server.py" ] && cp /opt/taisho_server.py "$BACKUP_DIR/" || true
        
        log "INFO" "Backup created at $BACKUP_DIR"
    fi
}

# Enhanced system check with detailed reporting
check_system_requirements() {
    log "INFO" "Performing enhanced system requirements check..."
    
    local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    local cpu_cores=$(nproc)
    local disk_space_gb=$(df /opt --output=avail -BG | tail -n1 | tr -d 'G')
    
    log "INFO" "System specifications:"
    log "INFO" "  RAM: ${mem_gb}GB (required: 20GB)"
    log "INFO" "  CPU cores: ${cpu_cores} (required: 4)"
    log "INFO" "  Disk space: ${disk_space_gb}GB (required: 50GB)"
    log "INFO" "  Architecture: $(uname -m)"
    
    local requirements_met=true
    
    if [ "$mem_gb" -lt 20 ]; then
        log "ERROR" "Insufficient RAM: ${mem_gb}GB < 20GB required"
        requirements_met=false
    fi
    
    if [ "$cpu_cores" -lt 4 ]; then
        log "ERROR" "Insufficient CPU cores: ${cpu_cores} < 4 required"
        requirements_met=false
    fi
    
    if [ "$disk_space_gb" -lt 50 ]; then
        log "ERROR" "Insufficient disk space: ${disk_space_gb}GB < 50GB required"
        requirements_met=false
    fi
    
    if [ "$(uname -m)" != "x86_64" ]; then
        log "WARN" "Architecture $(uname -m) may not be fully supported"
    fi
    
    if [ "$requirements_met" = false ]; then
        log "ERROR" "System requirements not met - aborting installation"
        return 1
    fi
    
    log "INFO" "✅ All system requirements satisfied"
    return 0
}

# Main execution starts here
main() {
    # Initialize logging
    mkdir -p "$(dirname "$LOG_FILE")"
    log "INFO" "========================================="
    log "INFO" "🏯 侍大将 (Taisho) - 日本語R1 セットアップ"
    log "INFO" "========================================="
    log "INFO" "Model: cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
    log "INFO" "Target: CT 101 (192.168.1.11)"
    log "INFO" "OpenVINO: INT8 quantization for optimal performance"
    log "INFO" "Enhanced with comprehensive error handling"
    log "INFO" "========================================="

    # Create backup if needed
    create_backup

    # Check if running in correct container
    if [ "$(hostname)" != "taisho-r1-japanese" ]; then
        log "WARN" "Expected hostname 'taisho-r1-japanese', got '$(hostname)'"
        read -p "Continue anyway? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "INFO" "Installation cancelled by user"
            exit 0
        fi
    fi

    # Enhanced system requirements check
    log "INFO" "[1/9] Enhanced system requirements check"
    check_system_requirements

    # Update system
    log "INFO" "[2/9] System update"
    log "DEBUG" "Updating package lists..."
    apt update || {
        log "ERROR" "Failed to update package lists"
        return 1
    }
    
    log "DEBUG" "Installing required packages..."
    apt install -y python3 python3-pip python3-venv git curl wget htop build-essential || {
        log "ERROR" "Failed to install required packages"
        return 1
    }
    log "INFO" "✅ System packages installed"

    # Create virtual environment
    log "INFO" "[3/9] Python environment setup"
    cd /opt
    
    if [ -d "openvino-env" ]; then
        log "WARN" "Existing virtual environment found - removing"
        rm -rf openvino-env
    fi
    
    log "DEBUG" "Creating Python virtual environment..."
    python3 -m venv openvino-env || {
        log "ERROR" "Failed to create virtual environment"
        return 1
    }
    
    log "DEBUG" "Activating virtual environment..."
    source openvino-env/bin/activate || {
        log "ERROR" "Failed to activate virtual environment"
        return 1
    }
    log "INFO" "✅ Python virtual environment ready"

    # Install OpenVINO and dependencies
    log "INFO" "[4/9] OpenVINO installation"
    log "DEBUG" "Upgrading pip..."
    pip install --upgrade pip || {
        log "ERROR" "Failed to upgrade pip"
        return 1
    }
    
    log "DEBUG" "Installing OpenVINO and ML dependencies..."
    pip install openvino==2024.0.0 openvino-dev optimum[openvino] transformers torch || {
        log "ERROR" "Failed to install OpenVINO dependencies"
        return 1
    }
    
    log "DEBUG" "Installing web server and utility dependencies..."
    pip install flask requests psutil huggingface_hub || {
        log "ERROR" "Failed to install server dependencies"
        return 1
    }

    log "INFO" "✅ OpenVINO installation complete"

    # Create model directory
    log "INFO" "[5/9] Model download and setup"
    log "DEBUG" "Creating model directory..."
    mkdir -p /opt/models || {
        log "ERROR" "Failed to create model directory"
        return 1
    }
    cd /opt/models

    # Download Japanese R1 model with progress tracking
    log "INFO" "🏗️ Downloading cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
    log "INFO" "Model size: ~2.8GB - This may take 10-30 minutes depending on network speed..."
    
    # Check if model already exists and is complete
    if [ -f "deepseek-r1-japanese/.download_complete" ]; then
        log "INFO" "Model already downloaded - skipping"
    else
        # Remove any partial download
        [ -d "deepseek-r1-japanese" ] && rm -rf deepseek-r1-japanese
        
        log "DEBUG" "Starting model download..."
        huggingface-cli download \
          cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese \
          --local-dir deepseek-r1-japanese \
          --local-dir-use-symlinks False || {
            log "ERROR" "Model download failed"
            return 1
        }
        
        # Mark download as complete
        touch deepseek-r1-japanese/.download_complete
        log "INFO" "✅ Model download complete"
    fi

# Convert to OpenVINO format with INT8 quantization
echo "[6/8] OpenVINO INT8 quantization"
echo "🔧 Converting to optimized INT8 format (30-60 minutes)..."
echo "This process uses significant CPU and memory resources."

optimum-cli export openvino \
  --model deepseek-r1-japanese \
  --task text-generation \
  --weight-format int8 \
  --group-size 128 \
  --ratio 0.8 \
  --output deepseek-r1-japanese-int8

echo "✅ INT8 quantization complete"

# Create inference server
echo "[7/8] Inference server setup"
cat > /opt/taisho_server.py << 'PYTHON'
#!/usr/bin/env python3
"""Taisho (侍大将) Inference Server - Japanese R1

High-performance inference server for DeepSeek R1 Japanese model.
Optimized for deep thinking and Japanese language processing.
"""

import os
import time
import logging
from flask import Flask, request, jsonify
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import psutil
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variables
model = None
tokenizer = None
model_stats = {
    'requests_processed': 0,
    'total_tokens_generated': 0,
    'average_response_time': 0,
    'start_time': time.time(),
    'model_loaded': False
}

def load_model():
    """Load the Japanese R1 model with OpenVINO optimization."""
    global model, tokenizer
    
    logger.info("🏯 侍大将システム起動中...")
    logger.info("Loading DeepSeek-R1-Distill-Qwen-14B-Japanese (INT8)")
    
    model_path = "/opt/models/deepseek-r1-japanese-int8"
    tokenizer_path = "/opt/models/deepseek-r1-japanese"
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load optimized model
        logger.info("Loading INT8 optimized model...")
        model = OVModelForCausalLM.from_pretrained(
            model_path,
            device="CPU",
            ov_config={
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1", 
                "INFERENCE_PRECISION_HINT": "u8",
                "KV_CACHE_PRECISION": "u8"
            }
        )
        
        model_stats['model_loaded'] = True
        logger.info("✅ Japanese R1 model loaded successfully")
        logger.info("🎯 Ready for deep thinking and Japanese inference")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate response using Japanese R1."""
    global model, tokenizer, model_stats
    
    if not model or not tokenizer:
        return jsonify({'error': '侍大将モデル未初期化'}), 500
    
    data = request.json
    prompt = data.get('prompt', '')
    system = data.get('system', '')
    max_tokens = data.get('max_tokens', 1000)
    temperature = data.get('temperature', 0.6)
    
    if not prompt:
        return jsonify({'error': 'プロンプトが必要です'}), 400
    
    start_time = time.time()
    
    try:
        # Format prompt for Japanese R1
        if system:
            full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        logger.info(f"🤔 [思考開始] {len(prompt)}文字のプロンプト処理")
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        
        # Generate with <think> support
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Calculate statistics
        end_time = time.time()
        response_time = end_time - start_time
        tokens_generated = len(outputs[0]) - input_length
        
        # Update stats
        model_stats['requests_processed'] += 1
        model_stats['total_tokens_generated'] += tokens_generated
        model_stats['average_response_time'] = (
            (model_stats['average_response_time'] * (model_stats['requests_processed'] - 1) + response_time) /
            model_stats['requests_processed']
        )
        
        logger.info(
            f"💡 [思考完了] {response_time:.1f}秒, {tokens_generated}トークン, "
            f"{tokens_generated/response_time:.1f} tok/s"
        )
        
        return jsonify({
            'response': response,
            'stats': {
                'response_time_seconds': response_time,
                'tokens_generated': tokens_generated,
                'tokens_per_second': tokens_generated / response_time if response_time > 0 else 0,
                'thinking_indicators': response.count('<think>'),
                'japanese_detected': any(ord(char) > 0x3040 for char in response)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        return jsonify({'error': f'推論エラー: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    system_info = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'uptime_hours': (time.time() - model_stats['start_time']) / 3600
    }
    
    return jsonify({
        'status': 'healthy' if model_stats['model_loaded'] else 'loading',
        'model': 'DeepSeek-R1-Japanese',
        'version': 'v7.0',
        'agent': '侍大将',
        'system': system_info,
        'stats': model_stats
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    """Detailed statistics."""
    return jsonify({
        'model_stats': model_stats,
        'system_stats': {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total // (1024**3),
            'memory_available_gb': psutil.virtual_memory().available // (1024**3),
            'load_average': os.getloadavg()
        },
        'performance': {
            'requests_per_hour': model_stats['requests_processed'] / ((time.time() - model_stats['start_time']) / 3600) if model_stats['requests_processed'] > 0 else 0,
            'average_tokens_per_request': model_stats['total_tokens_generated'] / model_stats['requests_processed'] if model_stats['requests_processed'] > 0 else 0
        }
    })

if __name__ == '__main__':
    # Load model in separate thread to avoid blocking
    model_thread = threading.Thread(target=load_model)
    model_thread.start()
    
    logger.info("🌸 侍大将サーバー起動 (ポート: 11434)")
    logger.info("Japanese R1 inference server starting...")
    
    # Start Flask server
    app.run(
        host='0.0.0.0',
        port=11434,
        debug=False,
        threaded=True
    )
PYTHON

chmod +x /opt/taisho_server.py

# Create systemd service
echo "[8/8] System service setup"
cat > /etc/systemd/system/taisho-japanese-r1.service << 'EOF'
[Unit]
Description=Taisho (侍大将) Japanese R1 Inference Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt
Environment=PYTHONPATH=/opt/openvino-env/lib/python3.10/site-packages
ExecStartPre=/bin/bash -c 'source /opt/openvino-env/bin/activate'
ExecStart=/opt/openvino-env/bin/python /opt/taisho_server.py
Restart=on-failure
RestartSec=10
TimeoutStartSec=300

# Optimization settings
Environment=OMP_NUM_THREADS=6
Environment=KMP_AFFINITY=granularity=fine,compact,1,0
Environment=OPENVINO_LOG_LEVEL=WARNING

# Memory settings
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable taisho-japanese-r1
systemctl start taisho-japanese-r1

# Wait for service to start
echo "⏳ Waiting for service to start (may take 2-3 minutes for model loading)..."
sleep 10

# Test the service
echo "🧪 Testing Japanese R1 inference..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/health > /dev/null; then
        echo "✅ Service is responding"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 5
done

# Final test with Japanese input
echo "🗾 Testing Japanese processing..."
response=$(curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "こんにちは、私は侍大将です。",
    "system": "日本語で簡潔に回答してください。",
    "max_tokens": 50
  }' | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('response', 'Error: ' + str(data)))" 2>/dev/null || echo "Error during test")

echo "Response: $response"

echo ""
echo "========================================"
echo "🎉 侍大将 (Japanese R1) セットアップ完了!"
echo "========================================"
echo "Model: cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
echo "Quantization: INT8 (optimized for performance)"
echo "Server: http://192.168.1.11:11434"
echo "Service: taisho-japanese-r1.service"
echo ""
echo "Management commands:"
echo "  systemctl status taisho-japanese-r1"
echo "  systemctl restart taisho-japanese-r1"
echo "  journalctl -u taisho-japanese-r1 -f"
echo ""
echo "Health check:"
echo "  curl http://192.168.1.11:11434/api/health"
echo ""
echo "Test generation:"
echo '  curl -X POST http://192.168.1.11:11434/api/generate \'
echo '    -H "Content-Type: application/json" \'
echo '    -d "{\"prompt\": \"ESP32のI2S設定について教えてください\", \"max_tokens\": 200}"'
echo ""
    log "INFO" "🏯 侍大将は深い思考(<think>)で皆様をお支えします!"
    log "INFO" "========================================"
    
    # Final success message
    log "INFO" "✅ Setup completed successfully!"
    log "INFO" "All logs saved to: $LOG_FILE"
    
    return 0
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
    exit $?
fi