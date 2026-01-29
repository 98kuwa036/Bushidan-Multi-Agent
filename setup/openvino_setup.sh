#!/bin/bash
# =============================================================
# 将軍システム v7.0 - OpenVINO R1 Setup (CyberAgent Japanese Edition)
# =============================================================
# CT 101 (侍大将) 内で実行。
# CyberAgent DeepSeek-R1-Distill-Qwen-14B-Japanese を OpenVINO INT8 に変換。
#
# Hardware: HP ProDesk 600 G4 (Core i5-8500 / 24GB RAM)
# Optimization: CPU Latency Focused
#
# Usage:
#   pct enter 101
#   bash openvino_setup.sh
# =============================================================

set -euo pipefail

# モデル名をディレクトリ名に反映
MODEL_ID="cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
MODEL_DIR="/opt/openvino/models/deepseek-r1-14b-jp-int8"
SERVER_DIR="/opt/openvino"

echo "============================================="
echo "  侍大将 R1 - OpenVINO Setup (JP Optimized)"
echo "============================================="

# --- [1/6] 基本パッケージ ---
echo "[1/6] 基本パッケージインストール..."
apt update && apt upgrade -y
# libpython3-dev等はビルド時に必須な場合があるため念のため追加
apt install -y build-essential cmake python3-pip python3-venv git wget curl libpython3-dev

# --- [2/6] Python環境セットアップ ---
echo ""
echo "[2/6] Python環境セットアップ..."
# 既存環境があれば再作成
if [ -d "/opt/openvino/venv" ]; then
    rm -rf /opt/openvino/venv
fi

python3 -m venv /opt/openvino/venv
source /opt/openvino/venv/bin/activate

pip install --upgrade pip
# nncfはモデル圧縮(INT8化)に必須。accelerateもHuggingFace連携で必要。
pip install \
    openvino>=2024.0.0 \
    "openvino-dev>=2024.0.0" \
    "optimum[openvino,nncf]" \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    numpy \
    flask

# --- [3/6] モデル変換 ---
echo ""
echo "[3/6] モデル変換 (${MODEL_ID} → OpenVINO INT8)..."
# 変換済みかチェック
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/openvino_model.xml" ]; then
    echo "  モデル既に存在: ${MODEL_DIR} (スキップ)"
else
    mkdir -p "$MODEL_DIR"
    cat > /tmp/convert_r1.py << PYTHON
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import nncf

print("[1/3] モデルダウンロード & INT8変換ロード中...")
# i5-8500はAVX2対応。INT8量子化でメモリ帯域幅のボトルネックを緩和し高速化する
model_id = "${MODEL_ID}"
save_dir = "${MODEL_DIR}"

# export=True と load_in_8bit=True で変換と圧縮を同時に行う
model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,
    load_in_8bit=True, 
    compile=False,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("[2/3] ディスクへの保存...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"完了: {save_dir}")
PYTHON

    # メモリ不足で落ちないよう、変換時は一時的にswapなどが有効であることを期待
    python3 /tmp/convert_r1.py
    echo "  変換完了"
fi

# --- [4/6] 推論サーバー ---
echo ""
echo "[4/6] 推論サーバー配置 (i5-8500 最適化)..."
cat > "${SERVER_DIR}/openvino_server.py" << 'PYTHON'
"""OpenVINO R1 Inference Server (CyberAgent/JP) - CPU Optimized"""

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import openvino as ov
from flask import Flask, request, jsonify
import time
import os

app = Flask(__name__)

# モデルパス
MODEL_PATH = "/opt/openvino/models/deepseek-r1-14b-jp-int8"

class OpenVINOR1:
    def __init__(self):
        print("[起動] OpenVINO Core初期化 (i5-8500 Optimized)...")
        
        # Core i5-8500 (6 Cores/6 Threads) 用設定
        # NUM_STREAMS=1: レイテンシ重視（1つの推論を全コアで全力で終わらせる）
        # INFERENCE_NUM_THREADS=6: 物理コア数に合わせる
        self.core = ov.Core()
        
        # 頻繁な再コンパイルを防ぐキャッシュ設定
        os.makedirs("/opt/openvino/cache", exist_ok=True)
        self.core.set_property({"CACHE_DIR": "/opt/openvino/cache"})

        print(f"[起動] モデル読み込み: {MODEL_PATH}")
        self.model = OVModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device="CPU",
            ov_config={
                "PERFORMANCE_HINT": "LATENCY",    # スループットより応答速度優先
                "NUM_STREAMS": "1",               # 1リクエストを最速で返す設定
                "INFERENCE_NUM_THREADS": 6,       # i5-8500の物理コア数
                "KV_CACHE_PRECISION": "u8",       # メモリ節約
                "ENABLE_CPU_PINNING": "YES",      # スレッドをコアに固定
            },
            compile=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("[起動] 完了")

    def generate(self, prompt, max_tokens=1024, temperature=0.7):
        start = time.time()
        
        # CyberAgentモデル推奨のフォーマットがある場合はここで調整可能だが
        # 基本は生のプロンプトか、呼び出し側(Shogun)で整形されたものを期待する
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1, # 日本語の繰り返し防止に少し強めに入れる
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # プロンプト部分を除去して応答のみ抽出
        # (tokenizerのdecode挙動により、単純なスライスがずれる場合があるため安全策)
        if response.startswith(prompt):
            result = response[len(prompt):]
        else:
            result = response # promptが含まれていない場合（モデルによる）

        elapsed = time.time() - start
        
        # 統計
        tok_count = len(self.tokenizer.encode(result))
        tok_per_sec = tok_count / elapsed if elapsed > 0 else 0
        print(f"[R1] {tok_count} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")
        
        return result

r1 = OpenVINOR1()

@app.route("/")
def index():
    return jsonify({"status": "ok", "model": "taisho-openvino-jp"})

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    result = r1.generate(
        data.get("prompt", ""),
        data.get("max_tokens", 1024),
        data.get("temperature", 0.7),
    )
    return jsonify({
        "model": "taisho-openvino-jp",
        "response": result,
        "done": True,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11434)
PYTHON

echo "  サーバースクリプト配置完了"

# --- [5/6] systemd サービス ---
echo ""
echo "[5/6] systemd サービス設定..."
cat > /etc/systemd/system/openvino-r1.service << 'EOF'
[Unit]
Description=OpenVINO R1 Inference Server (侍大将 - CyberAgent)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/openvino
ExecStart=/opt/openvino/venv/bin/python3 /opt/openvino/openvino_server.py
Restart=on-failure
RestartSec=10
# i5-8500: Core 0-5
CPUAffinity=0 1 2 3 4 5
# 24GB RAM搭載のため、20GBまで許可（OS分4GB残し）
MemoryMax=20G
Environment="OPENVINO_ENABLE_HUGE_PAGES=1"

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable openvino-r1
systemctl restart openvino-r1

echo "  サービス起動完了"

# --- [6/6] 動作確認 ---
echo ""
echo "[6/6] 動作確認 (30秒待機)..."
# モデルロードに時間がかかるため長めに待機
sleep 30

if curl -s http://localhost:11434/ | grep -q "ok"; then
    echo "  ✓ R1 サーバー稼働中"

    echo "  推論テスト (自己紹介)..."
    RESULT=$(curl -s -X POST http://localhost:11434/api/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"自己紹介をしてください。","max_tokens":50}')
    echo "  ✓ 応答サンプル: ${RESULT}"
else
    echo "  ✗ R1 サーバー起動失敗 または 起動中"
    echo "  ログ確認: journalctl -u openvino-r1 -n 50 -f"
fi

echo ""
echo "============================================="
echo "  侍大将 R1 Setup (JP) 完了!"
echo "  Hardware: ProDesk 600 G4 (i5-8500)"
echo ""
echo "  エンドポイント: http://<IP>:11434"
echo "  ログ: journalctl -u openvino-r1 -f"
echo "============================================="
