# Bushidan v9.4 - llama.cpp セットアップガイド

## 概要

v9.4ではローカルLLMバックエンドをOllamaからllama.cppに変更しました。これによりHP ProDesk 600などのCPU環境で最適化された推論が可能になります。

## なぜllama.cppか？

| 項目 | Ollama | llama.cpp |
|------|--------|-----------|
| **CPU最適化** | 一般的 | 高度（AVX2, FMA, NEON） |
| **メモリ制御** | 自動 | mlock/mmap対応 |
| **設定柔軟性** | 限定的 | フルコントロール |
| **依存関係** | Go + llama.cpp | ネイティブC++ |
| **オーバーヘッド** | やや多い | 最小限 |
| **HP ProDesk 600** | 動作 | 最適化済み |

## ハードウェア要件

### HP ProDesk 600 推奨スペック

| コンポーネント | 最小 | 推奨 |
|---------------|------|------|
| **CPU** | i5-10500 (6C/12T) | i7-10700 (8C/16T) |
| **RAM** | 16GB DDR4 | 32GB DDR4 |
| **ストレージ** | 50GB空き | SSD 100GB空き |
| **OS** | Ubuntu 20.04 | Ubuntu 22.04 |

### 期待性能

- **モデル**: Qwen3-Coder-30B-A3B-instruct-q4_k_m.gguf (約17GB)
- **推論速度**: 15-25 tokens/秒 (i7-10700)
- **コンテキスト**: 4096 tokens
- **電力**: 約65-95W

## クイックスタート

### 1. 自動セットアップ

```bash
# セットアップスクリプト実行
chmod +x scripts/setup_llamacpp_prodesck600.sh
./scripts/setup_llamacpp_prodesck600.sh
```

セットアップオプション:
1. フルセットアップ（ビルド + モデル + サービス）
2. llama.cppのみビルド
3. モデルのみダウンロード
4. サービスのみ作成
5. 検証のみ

### 2. サーバー起動

```bash
# 手動起動
./scripts/start_llamacpp.sh

# またはsystemdサービス
sudo systemctl start bushidan-llamacpp
sudo systemctl enable bushidan-llamacpp  # 自動起動
```

### 3. 動作確認

```bash
# ヘルスチェック
curl http://127.0.0.1:8080/health

# 簡単なテスト
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## 手動セットアップ

### 1. 依存関係インストール

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake git curl wget
```

### 2. llama.cppビルド

```bash
# クローン
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# CPU最適化ビルド
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_NATIVE=ON \
    -DLLAMA_AVX2=ON \
    -DLLAMA_FMA=ON \
    -DLLAMA_F16C=ON \
    -DLLAMA_BUILD_SERVER=ON

make -j$(nproc)
```

### 3. モデルダウンロード

```bash
mkdir -p ~/Bushidan-Multi-Agent/models
cd ~/Bushidan-Multi-Agent/models

# Hugging Faceからダウンロード
wget https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/qwen3-coder-30b-a3b-instruct-q4_k_m.gguf
```

### 4. サーバー起動

```bash
./llama.cpp/build/bin/llama-server \
    -m models/Qwen3-Coder-30B-A3B-instruct-q4_k_m.gguf \
    -c 4096 \
    -t 8 \
    -b 512 \
    --parallel 1 \
    --host 127.0.0.1 \
    --port 8080 \
    --mlock \
    --mmap
```

## 設定パラメータ

### CPU最適化パラメータ

| パラメータ | HP ProDesk 600推奨 | 説明 |
|-----------|-------------------|------|
| `-t` (threads) | 8 | 使用スレッド数（物理コア数推奨） |
| `-c` (context) | 4096 | コンテキストサイズ（速度優先で縮小） |
| `-b` (batch) | 512 | バッチサイズ（CPU最適値） |
| `--parallel` | 1 | 並列リクエスト数（安定性優先） |
| `--mlock` | 有効 | メモリロック（スワップ防止） |
| `--mmap` | 有効 | メモリマップファイル |

### Bushidan設定 (SystemConfig)

```python
from core.system_orchestrator import SystemConfig, SystemMode

config = SystemConfig(
    mode=SystemMode.BATTALION,
    claude_api_key="...",
    gemini_api_key="...",
    tavily_api_key="...",

    # llama.cpp設定
    llamacpp_endpoint="http://127.0.0.1:8080",
    llamacpp_model_path="models/Qwen3-Coder-30B-A3B-instruct-q4_k_m.gguf",
    llamacpp_threads=8,
    llamacpp_context_size=4096,
    llamacpp_batch_size=512,
    llamacpp_mlock=True,
    use_llamacpp=True  # llama.cpp使用（Falseでレガシーのollama）
)
```

## トラブルシューティング

### メモリ不足

**症状**: サーバー起動後にOOM killerで終了

**解決策**:
```bash
# スワップ追加
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# または、より小さい量子化モデルを使用
# Q3_K_M (~12GB) または Q2_K (~9GB)
```

### 遅い推論速度

**症状**: 10 tokens/秒以下

**解決策**:
1. スレッド数を物理コア数に合わせる
2. mlock有効化（メモリスワップ防止）
3. 他のアプリケーションを終了
4. CPUガバナーをperformanceに設定:
   ```bash
   sudo cpupower frequency-set -g performance
   ```

### API接続エラー

**症状**: Bushidan起動時に「llama.cpp server not responding」

**解決策**:
```bash
# サーバー状態確認
sudo systemctl status bushidan-llamacpp

# ログ確認
journalctl -u bushidan-llamacpp -n 50

# ポート確認
ss -tlnp | grep 8080
```

## 性能ベンチマーク

### HP ProDesk 600 (i7-10700, 32GB RAM)

| テスト | 結果 |
|-------|------|
| 初回ロード時間 | 約45秒 |
| 短いプロンプト (100 tokens) | 18-22 tok/s |
| 長いプロンプト (2000 tokens) | 15-18 tok/s |
| ピークRAM使用量 | 約19GB |
| 平均CPU使用率 | 85-95% |

### 他環境との比較

| 環境 | 推論速度 | 備考 |
|------|---------|------|
| HP ProDesk 600 (i7) | 18 tok/s | CPU推論 |
| RTX 3060 (12GB) | 45 tok/s | GPU推論、Q4_K_M一部オフロード |
| RTX 4090 (24GB) | 85 tok/s | フルGPU推論 |
| Apple M2 Pro | 25 tok/s | Metal GPU |

## API リファレンス

### エンドポイント

| エンドポイント | 説明 |
|---------------|------|
| `GET /health` | ヘルスチェック |
| `POST /completion` | ネイティブ補完API |
| `POST /v1/chat/completions` | OpenAI互換チャットAPI |
| `GET /v1/models` | 利用可能モデル一覧 |

### OpenAI互換API例

```python
import httpx

async def chat_with_qwen3():
    url = "http://127.0.0.1:8080/v1/chat/completions"

    payload = {
        "model": "qwen3-coder-30b",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to check if a number is prime."}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=payload)
        result = response.json()
        return result["choices"][0]["message"]["content"]
```

## 関連ドキュメント

- [v9.4 Implementation Guide](./v9.4_IMPLEMENTATION_GUIDE.md)
- [README](../README.md)
- [llama.cpp公式ドキュメント](https://github.com/ggerganov/llama.cpp)
- [Qwen3モデル](https://huggingface.co/Qwen)
