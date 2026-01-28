#!/bin/bash
# =============================================================
# 将軍システム v5.0 - OpenVINO R1 Setup
# =============================================================
# CT 101 (侍大将) 内で実行。
# DeepSeek-R1-Distill-Qwen-14B を OpenVINO INT8 に変換。
#
# Usage:
#   pct enter 101
#   bash openvino_setup.sh
# =============================================================

set -euo pipefail

MODEL_DIR="/opt/openvino/models/deepseek-r1-14b-int8"
SERVER_DIR="/opt/openvino"

echo "============================================="
echo "  侍大将 R1 - OpenVINO Setup"
echo "============================================="

# --- 基本パッケージ ---
echo "[1/6] 基本パッケージインストール..."
apt update && apt upgrade -y
apt install -y build-essential cmake python3-pip python3-venv git wget curl

# --- Python venv ---
echo ""
echo "[2/6] Python環境セットアップ..."
python3 -m venv /opt/openvino/venv
source /opt/openvino/venv/bin/activate

pip install --upgrade pip
pip install \
    openvino==2024.0.0 \
    "openvino-dev[pytorch,onnx]==2024.0.0" \
    "optimum[openvino]" \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    numpy \
    flask

# --- モデル変換 ---
echo ""
echo "[3/6] モデル変換 (DeepSeek-R1-14B → OpenVINO INT8)..."
if [ -d "$MODEL_DIR" ]; then
    echo "  モデル既に存在: ${MODEL_DIR}"
else
    mkdir -p "$MODEL_DIR"
    cat > /tmp/convert_r1.py << 'PYTHON'
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

print("[1/3] モデルダウンロード...")
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

print("[2/3] OpenVINO INT8変換...")
model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    load_in_8bit=True,
    compile=False
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("[3/3] 保存...")
save_dir = "/opt/openvino/models/deepseek-r1-14b-int8"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"完了: {save_dir}")
PYTHON

    python3 /tmp/convert_r1.py
    echo "  変換完了"
fi

# --- 推論サーバー ---
echo ""
echo "[4/6] 推論サーバー配置..."
cat > "${SERVER_DIR}/openvino_server.py" << 'PYTHON'
"""OpenVINO R1 Inference Server - Ollama互換API"""

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import openvino as ov
from flask import Flask, request, jsonify
import time

app = Flask(__name__)


class OpenVINOR1:
    def __init__(self):
        print("[起動] OpenVINO R1...")

        self.core = ov.Core()
        self.core.set_property("CPU", {
            "INFERENCE_NUM_THREADS": 6,
            "INFERENCE_PRECISION_HINT": "u8",
            "PERFORMANCE_HINT": "LATENCY",
            "KV_CACHE_PRECISION": "u8",
            "CPU_BIND_THREAD": "YES",
        })

        self.model = OVModelForCausalLM.from_pretrained(
            "/opt/openvino/models/deepseek-r1-14b-int8",
            device="CPU",
            ov_config={
                "PERFORMANCE_HINT": "LATENCY",
                "KV_CACHE_PRECISION": "u8",
            },
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "/opt/openvino/models/deepseek-r1-14b-int8"
        )

        print("[起動] 完了")

    def generate(self, prompt, max_tokens=512, temperature=0.6):
        start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start
        result = response[len(prompt):]
        tok_count = len(self.tokenizer.encode(result))
        tok_per_sec = tok_count / elapsed if elapsed > 0 else 0
        print(f"[R1] {tok_count} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")
        return result


r1 = OpenVINOR1()


@app.route("/")
def index():
    return jsonify({"status": "ok", "model": "taisho-openvino"})


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    result = r1.generate(
        data.get("prompt", ""),
        data.get("max_tokens", 512),
        data.get("temperature", 0.6),
    )
    return jsonify({
        "model": "taisho-openvino",
        "response": result,
        "done": True,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11434)
PYTHON

echo "  サーバースクリプト配置完了"

# --- systemd サービス ---
echo ""
echo "[5/6] systemd サービス設定..."
cat > /etc/systemd/system/openvino-r1.service << 'EOF'
[Unit]
Description=OpenVINO R1 Inference Server (侍大将)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/openvino
ExecStart=/opt/openvino/venv/bin/python3 /opt/openvino/openvino_server.py
Restart=on-failure
RestartSec=10
CPUAffinity=0 1 2 3 4 5
MemoryMax=18G
Environment="OPENVINO_ENABLE_HUGE_PAGES=1"

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable openvino-r1
systemctl start openvino-r1

echo "  サービス起動完了"

# --- 動作確認 ---
echo ""
echo "[6/6] 動作確認..."
sleep 5

if curl -s http://localhost:11434/ | grep -q "ok"; then
    echo "  ✓ R1 サーバー稼働中"

    echo "  推論テスト..."
    RESULT=$(curl -s -X POST http://localhost:11434/api/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"Hello, I am","max_tokens":20}')
    echo "  ✓ 応答: ${RESULT}"
else
    echo "  ✗ R1 サーバー起動失敗"
    echo "  ログ確認: journalctl -u openvino-r1 -n 50"
fi

echo ""
echo "============================================="
echo "  侍大将 R1 Setup 完了!"
echo ""
echo "  エンドポイント: http://192.168.1.11:11434"
echo "  ログ: journalctl -u openvino-r1 -f"
echo "============================================="
