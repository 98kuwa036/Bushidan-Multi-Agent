#!/bin/bash
# ProDesk 600 G4 (Core i5-8500) 向け Qwen ローカル環境セットアップ
# CPU推論専用・プライバシー保護・コスト削減用

set -e

echo "=== Qwen Local LLM Setup for CPU Inference ==="

# 1. llama.cpp のビルド（OpenBLAS最適化）
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp

    # CPU最適化ビルド（AVX2サポート確認）
    make clean
    LLAMA_OPENBLAS=1 make -j6

    cd ..
else
    echo "llama.cpp already exists. Updating..."
    cd llama.cpp && git pull && cd ..
fi

# 2. モデルのダウンロード（Qwen2.5-Coder-7B Q4_K_M）
MODEL_DIR="./models/qwen"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/qwen2.5-coder-7b-instruct-q4_k_m.gguf" ]; then
    echo "Downloading Qwen2.5-Coder-7B-Instruct (Q4_K_M)..."
    wget -O "$MODEL_DIR/qwen2.5-coder-7b-instruct-q4_k_m.gguf" \
        "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
else
    echo "Model already downloaded."
fi

# 3. OpenAI互換サーバーの起動スクリプト作成
cat > start_qwen_server.sh << 'EOF'
#!/bin/bash
# Qwen ローカルサーバー起動（OpenAI API互換）

MODEL_PATH="./models/qwen/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

./llama.cpp/llama-server \
  -m "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8080 \
  -c 8192 \
  -t 5 \
  --mlock \
  --temp 0.7 \
  --top-p 0.9 \
  --top-k 40 \
  --min-p 0.05 \
  --repeat-penalty 1.1 \
  --chat-template qwen \
  --log-disable

echo "Qwen server running at http://127.0.0.1:8080"
echo "OpenAI-compatible endpoint: http://127.0.0.1:8080/v1"
EOF

chmod +x start_qwen_server.sh

# 4. systemd サービス作成（バックグラウンド常駐用）
cat > qwen-local.service << EOF
[Unit]
Description=Qwen Local LLM Server (CPU Inference)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/start_qwen_server.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Start server manually:  ./start_qwen_server.sh"
echo "2. Or install as service:  sudo cp qwen-local.service /etc/systemd/system/"
echo "                           sudo systemctl enable qwen-local"
echo "                           sudo systemctl start qwen-local"
echo ""
echo "Server endpoint: http://127.0.0.1:8080/v1 (OpenAI compatible)"
