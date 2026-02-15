# 武士団マルチエージェントシステム v10.1 - クイックスタート

Slack経由で武士団システムを使い始めるまでの最短手順です。

---

## 前提条件

- Proxmox VE でコンテナ作成済み
  - CT 100 (本陣): システム調整
  - CT 101 (Qwen3): llama.cpp 推論サーバー
- 各APIキー取得済み（[API_SETUP_GUIDE.md](./API_SETUP_GUIDE.md) 参照）

---

## Step 1: CT 101 - llama-server 設定

### 外部接続を許可

```bash
# Proxmox ホストから
pct enter 101
su - claude

# start_llamacpp.sh を編集
nano ~/Bushidan-Multi-Agent/scripts/start_llamacpp.sh
```

以下を変更:
```bash
# 変更前
--host 127.0.0.1 \

# 変更後 (外部からの接続を許可)
--host 0.0.0.0 \
```

### systemd サービス設定も更新

```bash
sudo nano /etc/systemd/system/bushidan-llamacpp.service
```

同様に `--host 127.0.0.1` を `--host 0.0.0.0` に変更。

### サービス再起動

```bash
sudo systemctl daemon-reload
sudo systemctl restart bushidan-llamacpp
sudo systemctl status bushidan-llamacpp
```

### 動作確認

```bash
# ローカルで確認
curl http://127.0.0.1:8080/health

# CT 100 から確認 (別ターミナル)
curl http://<CT101のIP>:8080/health
```

正常なら `{"status":"ok"}` が返ります。

---

## Step 2: CT 100 - 環境変数設定

```bash
# Proxmox ホストから
pct enter 100
su - claude
cd ~/Bushidan-Multi-Agent

# .env ファイルを編集
nano .env
```

### 最小構成 (.env)

```bash
# ===== 必須 =====
CLAUDE_API_KEY=sk-ant-api03-xxxxx
GEMINI_API_KEY=AIzaSyxxxxx

# ===== Slack =====
SLACK_BOT_TOKEN=xoxb-xxxxx
SLACK_SIGNING_SECRET=xxxxx

# ===== llama.cpp =====
LLAMACPP_ENDPOINT=http://192.168.11.231:8080
```

### 推奨構成 (パフォーマンス向上)

```bash
# ===== 必須 =====
CLAUDE_API_KEY=sk-ant-api03-xxxxx
GEMINI_API_KEY=AIzaSyxxxxx

# ===== Slack =====
SLACK_BOT_TOKEN=xoxb-xxxxx
SLACK_SIGNING_SECRET=xxxxx

# ===== llama.cpp =====
LLAMACPP_ENDPOINT=http://192.168.11.231:8080

# ===== 推奨 (無料) =====
GROQ_API_KEY=gsk_xxxxx
TAVILY_API_KEY=tvly-xxxxx
```

---

## Step 3: Slack App Token 取得 (Socket Mode)

Request URL の設定は**不要**です。Socket Mode を使用するため WebSocket 接続で動作します。

### 3-1: Socket Mode を有効化

1. [Slack API Apps](https://api.slack.com/apps) にアクセス
2. 対象のアプリを選択
3. 左メニュー → **Socket Mode** → **Enable Socket Mode** をオン

### 3-2: App-Level Token を生成

1. 左メニュー → **Basic Information** → **App-Level Tokens**
2. **Generate Token and Scopes** をクリック
3. Token Name: `bushidan-socket` など任意の名前
4. Scope: **`connections:write`** を追加
5. **Generate** をクリック
6. 表示される `xapp-1-...` トークンをコピー

### 3-3: .env に追加

```bash
# CT 100 で
nano ~/Bushidan-Multi-Agent/.env
```

以下の行を追記:
```bash
SLACK_APP_TOKEN=xapp-1-XXXXXXXXX-XXXXXXXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### 3-4: Event Subscriptions の設定

1. 左メニュー → **Event Subscriptions** → **Enable Events** をオン
2. **Request URL は入力不要** (Socket Mode の場合)
3. **Subscribe to bot events** に以下を追加:
   - `app_mention`
   - `message.channels`（オプション）
4. **Save Changes**

### 3-5: Bot Scopes の確認

左メニュー → **OAuth & Permissions** → **Scopes** に以下があることを確認:
- `app_mentions:read`
- `chat:write`
- `channels:history`

不足していれば追加して **Reinstall to Workspace** を実行。

---

## Step 4: Slack Bot 起動

### 手動起動 (テスト用)

```bash
cd ~/Bushidan-Multi-Agent
source .venv/bin/activate
python -m bushidan.slack_bot
```

### systemd サービス化 (本番用)

> **注意**: systemd サービス化する前に `SLACK_APP_TOKEN` を `.env` に設定してください。

```bash
# サービスファイル作成
sudo tee /etc/systemd/system/bushidan-slack.service << 'EOF'
[Unit]
Description=Bushidan Slack Bot
After=network.target

[Service]
Type=simple
User=claude
WorkingDirectory=/home/claude/Bushidan-Multi-Agent
Environment=PATH=/home/claude/Bushidan-Multi-Agent/.venv/bin:/usr/bin
ExecStart=/home/claude/Bushidan-Multi-Agent/.venv/bin/python -m bushidan.slack_bot
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 有効化・起動
sudo systemctl daemon-reload
sudo systemctl enable bushidan-slack
sudo systemctl start bushidan-slack

# ログ確認
sudo journalctl -u bushidan-slack -f
```

---

## Step 4: Slack から動作確認

### Bot をチャンネルに招待

1. Slack で使用するチャンネルを開く
2. `/invite @Bushidan` を入力

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

@Bushidan 新しいREST APIエンドポイントを追加して。ユーザー一覧を取得するGETエンドポイント。

@Bushidan このエラーの原因を調べて: [エラーログ]
```

---

## トラブルシューティング

### Slack Bot が応答しない

1. **Event Subscriptions の URL 確認**
   - Slack App 設定 → Event Subscriptions
   - Request URL が正しいか確認
   - ステータスが "Verified" か確認

2. **ログ確認**
   ```bash
   sudo journalctl -u bushidan-slack -f
   ```

3. **Bot がチャンネルに招待されているか確認**
   - `/invite @Bushidan` を実行

### llama-server に接続できない

1. **CT 101 でサービス確認**
   ```bash
   systemctl status bushidan-llamacpp
   curl http://127.0.0.1:8080/health
   ```

2. **ファイアウォール確認**
   ```bash
   # CT 101 で
   sudo ufw status
   # 必要なら
   sudo ufw allow 8080
   ```

3. **IP アドレス確認**
   ```bash
   ip addr show eth0
   ```

### モデル読み込みが遅い

初回起動時は17GBのモデルをメモリにロードするため数分かかります。
```bash
# ログでロード状況を確認
journalctl -u bushidan-llamacpp -f
```

---

## ネットワーク構成図

```
┌─────────────────────────────────────────────────────────────┐
│                      Proxmox Host                            │
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │    CT 100        │         │    CT 101        │          │
│  │    (本陣)         │         │    (Qwen3)       │          │
│  │                  │  HTTP   │                  │          │
│  │  Slack Bot  ────────────────▶ llama-server   │          │
│  │  Port: 3000     │  :8080  │  Port: 8080      │          │
│  │                  │         │                  │          │
│  │  IP: .100        │         │  IP: .101/.231   │          │
│  └──────────────────┘         └──────────────────┘          │
│           │                                                  │
└───────────│──────────────────────────────────────────────────┘
            │
            ▼ HTTPS
    ┌───────────────┐
    │   Slack API   │
    │   Claude API  │
    │   Gemini API  │
    │   Groq API    │
    └───────────────┘
```

---

## 次のステップ

- [API_SETUP_GUIDE.md](./API_SETUP_GUIDE.md) - 各APIの詳細設定
- [agent_instructions/](./agent_instructions/) - 各役職の動作仕様
- [../config/mcp_permissions.yaml](../config/mcp_permissions.yaml) - MCP権限設定
