# 武士団マルチエージェントシステム v10.1 - クイックスタート

Discord 経由で武士団システムを使い始めるまでの最短手順です。

---

## 前提条件

- Proxmox VE でコンテナ作成済み
  - CT 100 (本陣 192.168.11.231): システム調整
  - CT 101 (Qwen3 192.168.11.232): llama.cpp 推論サーバー
- 各APIキー取得済み（[API_SETUP_GUIDE.md](./API_SETUP_GUIDE.md) 参照）

---

## Step 1: CT 101 - llama-server 設定

### systemd サービスを設定・起動

```bash
# CT 101 上で (root)
nano /etc/systemd/system/bushidan-llamacpp.service
```

内容:
```ini
[Unit]
Description=Bushidan llama.cpp Server (Qwen3-Coder-30B)
After=network.target

[Service]
Type=simple
User=claude
WorkingDirectory=/home/claude
ExecStart=/home/claude/llama.cpp/build/bin/llama-server \
    -m /home/claude/Bushidan-Multi-Agent/models/qwen3/Qwen3-Coder-30B-Q4_K_M.gguf \
    -c 4096 \
    -t 6 \
    -b 512 \
    --parallel 1 \
    --host 0.0.0.0 \
    --port 8080 \
    --mlock \
    --mmap
Restart=on-failure
RestartSec=10
MemoryMax=20G
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable bushidan-llamacpp
sudo systemctl start bushidan-llamacpp
```

### 動作確認

```bash
# CT 101 ローカル
curl http://127.0.0.1:8080/health

# CT 100 から
curl http://192.168.11.232:8080/health
# → {"status":"ok"}
```

---

## Step 2: CT 100 - 環境変数設定

```bash
pct enter 100
su - claude
cd ~/Bushidan-Multi-Agent
nano .env
```

### 最小構成 (.env)

```bash
# ===== 必須 =====
CLAUDE_API_KEY=sk-ant-api03-xxxxx
GEMINI_API_KEY=AIzaSyxxxxx

# ===== Discord =====
DISCORD_BOT_TOKEN=MTIzNDU2Nzg5...

# ===== llama.cpp =====
LLAMACPP_ENDPOINT=http://192.168.11.232:8080
```

### 推奨構成

```bash
CLAUDE_API_KEY=sk-ant-api03-xxxxx
GEMINI_API_KEY=AIzaSyxxxxx
DISCORD_BOT_TOKEN=MTIzNDU2Nzg5...
LLAMACPP_ENDPOINT=http://192.168.11.232:8080
GROQ_API_KEY=gsk_xxxxx          # Simple タスク高速化 (無料)
TAVILY_API_KEY=tvly-xxxxx       # Web検索 (無料枠)
OPENROUTER_API_KEY=sk-or-v1-... # Gunshi (Qwen3-Coder-Next)
QWEN3_CODER_NEXT_API_KEY=sk-or-v1-...
QWEN3_CODER_NEXT_PROVIDER=openrouter
ALIBABA_API_KEY=sk-or-v1-...   # Kagemusha (Qwen3-Plus) via OpenRouter
KIMI_API_KEY=sk-xxxxx           # 傭兵 (並列実行)
KIMI_PROVIDER=moonshot
```

---

## Step 3: Discord Bot 設定

### 3-1: Discord Application を作成

1. [discord.com/developers/applications](https://discord.com/developers/applications) にアクセス
2. **New Application** → 名前: `Bushidan` → **Create**
3. 左メニュー → **Bot** → **Add Bot** (または Reset Token)
4. **Reset Token** → 表示されたトークンをコピー
5. `.env` に追記:
   ```bash
   DISCORD_BOT_TOKEN=ここにトークンを貼り付け
   ```

### 3-2: Intents を有効化 (必須)

1. 左メニュー → **Bot** → **Privileged Gateway Intents**
2. 以下を全て **ON**:
   - `MESSAGE CONTENT INTENT` ← **必須**
   - `SERVER MEMBERS INTENT`
3. **Save Changes**

### 3-3: Bot をサーバーに招待

1. 左メニュー → **OAuth2** → **URL Generator**
2. **Scopes**: `bot` にチェック
3. **Bot Permissions**: `Send Messages`, `Read Message History`, `Add Reactions`
4. 生成された URL をブラウザで開いてサーバーを選択して招待

---

## Step 4: Discord Bot 起動

### discord.py をインストール

```bash
cd ~/Bushidan-Multi-Agent
source .venv/bin/activate
pip install discord.py
```

### 手動起動 (テスト用)

```bash
python -m bushidan.discord_bot
```

ログで確認:
```
🏯 武士団 Discord Bot 起動完了: Bushidan#1234 (ID: 123456789)
```

### systemd サービス化 (本番用)

```bash
sudo tee /etc/systemd/system/bushidan-discord.service << 'EOF'
[Unit]
Description=Bushidan Discord Bot
After=network.target

[Service]
Type=simple
User=claude
WorkingDirectory=/home/claude/Bushidan-Multi-Agent
Environment=PATH=/home/claude/Bushidan-Multi-Agent/.venv/bin:/usr/bin
ExecStart=/home/claude/Bushidan-Multi-Agent/.venv/bin/python -m bushidan.discord_bot
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bushidan-discord
sudo systemctl start bushidan-discord

# ログ確認
sudo journalctl -u bushidan-discord -f
```

---

## Step 5: Discord から動作確認

### Bot をチャンネルに追加

サーバーに招待済みであれば自動でメッセージを受信します。

### テストメッセージ

```
@Bushidan こんにちは
```

Bot が応答すれば成功です。

### 使用例

```
@Bushidan このPythonコードのバグを修正して:
def add(a, b):
    return a - b

@Bushidan 新しいREST APIエンドポイントを設計して。ユーザー一覧取得のGET。

@Bushidan このエラーの原因を調べて: [エラーログ]
```

---

## トラブルシューティング

### Discord Bot が応答しない

1. **MESSAGE CONTENT INTENT が OFF になっている**
   - Discord Developer Portal → Bot → Privileged Gateway Intents
   - `MESSAGE CONTENT INTENT` を ON に変更

2. **ログ確認**
   ```bash
   sudo journalctl -u bushidan-discord -f
   ```

3. **Bot がサーバーに招待されているか確認**
   - Discord Developer Portal → OAuth2 → URL Generator で再招待

4. **DISCORD_BOT_TOKEN が正しいか確認**
   ```bash
   grep DISCORD_BOT_TOKEN ~/Bushidan-Multi-Agent/.env
   ```

### llama-server に接続できない

1. **CT 101 でサービス確認**
   ```bash
   systemctl status bushidan-llamacpp
   curl http://127.0.0.1:8080/health
   ```

2. **ファイアウォール確認**
   ```bash
   sudo ufw allow 8080
   ```

3. **IP アドレス確認**
   ```bash
   ip addr show eth0
   ```

### モデル読み込みが遅い

初回起動時は17GBのモデルをメモリにロードするため数分かかります。
```bash
journalctl -u bushidan-llamacpp -f
```

---

## ネットワーク構成図

```
┌──────────────────────────────────────────────────────────────┐
│                      Proxmox Host                             │
│                                                               │
│  ┌───────────────────┐         ┌──────────────────┐          │
│  │    CT 100         │         │    CT 101        │          │
│  │    (本陣)          │         │    (Qwen3)       │          │
│  │  192.168.11.231   │  HTTP   │  192.168.11.232  │          │
│  │                   │ ──────▶ │                  │          │
│  │  Discord Bot      │  :8080  │  llama-server    │          │
│  │                   │         │  Port: 8080      │          │
│  └───────────────────┘         └──────────────────┘          │
│            │                                                  │
└────────────│─────────────────────────────────────────────────┘
             │
             ▼ HTTPS (WebSocket Gateway)
     ┌───────────────┐
     │  Discord API  │
     │  Claude API   │
     │  Gemini API   │
     │  Groq API     │
     │  OpenRouter   │
     └───────────────┘
```

---

## 次のステップ

- [API_SETUP_GUIDE.md](./API_SETUP_GUIDE.md) - 各APIの詳細設定
- [agent_instructions/](./agent_instructions/) - 各役職の動作仕様
- [../config/mcp_permissions.yaml](../config/mcp_permissions.yaml) - MCP権限設定
