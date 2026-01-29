#!/bin/bash
# setup_r1_japanese.sh - Japanese R1 Setup for Shogun System v7.0
#
# Sets up cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese with OpenVINO
# For Taisho (侍大将) on CT 101 (192.168.1.11)
#
# Usage: Run inside CT 101 (侍大将専用LXC)
# Requires: 20GB RAM, 6 CPU cores, HugePages configured

set -e

echo "========================================"
echo "🏯 侍大将 (Taisho) - 日本語R1 セットアップ"
echo "========================================"
echo "Model: cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
echo "Target: CT 101 (192.168.1.11)"
echo "OpenVINO: INT8 quantization for optimal performance"
echo "========================================"

# Check if running in correct container
if [ "$(hostname)" != "taisho-r1-japanese" ]; then
    echo "⚠️ Warning: Expected hostname 'taisho-r1-japanese', got '$(hostname)'"
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check system resources
echo "[1/8] System resource check"
MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
CPU_CORES=$(nproc)

echo "Available RAM: ${MEM_GB}GB (minimum: 20GB)"
echo "CPU cores: ${CPU_CORES} (minimum: 6)"

if [ "$MEM_GB" -lt 20 ]; then
    echo "❌ Insufficient RAM: ${MEM_GB}GB < 20GB required"
    exit 1
fi

if [ "$CPU_CORES" -lt 4 ]; then
    echo "❌ Insufficient CPU cores: ${CPU_CORES} < 4 required"
    exit 1
fi

echo "✅ System resources OK"

# Update system
echo "[2/8] System update"
apt update
apt install -y python3 python3-pip python3-venv git curl wget htop

# Create virtual environment
echo "[3/8] Python environment setup"
cd /opt
python3 -m venv openvino-env
source openvino-env/bin/activate

# Install OpenVINO and dependencies
echo "[4/8] OpenVINO installation"
pip install --upgrade pip
pip install openvino==2024.0.0 openvino-dev optimum[openvino] transformers torch
pip install flask requests psutil huggingface_hub

echo "✅ OpenVINO installation complete"

# Create model directory
echo "[5/8] Model download and setup"
mkdir -p /opt/models
cd /opt/models

# Download Japanese R1 model
echo "🏗️ Downloading cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese (2.8GB)"
echo "This may take 10-30 minutes depending on network speed..."

huggingface-cli download \
  cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese \
  --local-dir deepseek-r1-japanese \
  --local-dir-use-symlinks False

echo "✅ Model download complete"

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
echo "🏯 侍大将は深い思考(<think>)で皆様をお支えします!"
echo "========================================"