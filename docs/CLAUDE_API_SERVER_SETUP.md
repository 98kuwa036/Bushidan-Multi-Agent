# Claude API Server セットアップガイド

## 概要

`claude-dedicated` LXC (192.168.11.237) に Claude API Server を構築し、`bushidan-honjin` LXC からリモート経由で Claude を呼び出します。

**メリット:**
- メモリ分離：bushidan-honjin のメモリ圧迫を回避
- スケーラビリティ：複数の LXC から API Server を共有可能
- 優先順位：Claude Pro CLI → Anthropic API のフォールバック機構

## セットアップ手順

### 1. claude-dedicated LXC の初期設定 (Proxmox)

```bash
# 軽量ディストロ（Alpine Linux など）で新規 LXC 作成
# IP: 192.168.11.237

# LXC 内でのセットアップ
apk update
apk add python3 py3-pip curl git
```

### 2. Claude CLI のインストール (claude-dedicated LXC 内)

```bash
# Claude CLI インストール
curl -fsSL https://claude.ai/install.sh | bash

# または npm でインストール
npm install -g claude@latest

# バージョン確認
/home/claude/.local/bin/claude --version
```

### 3. Python 環境の構築 (claude-dedicated LXC 内)

```bash
# Python3.10+ インストール
apk add python3.13 py3.13-pip py3.13-venv

# 仮想環境作成
cd /opt
python3.13 -m venv venv
source venv/bin/activate

# Flask インストール
pip install flask httpx
```

### 4. Claude API Server のセットアップ (claude-dedicated LXC 内)

```bash
# bushidan-honjin からファイルをコピー or Git クローン
# claude_api_server.py を /opt に配置

# 環境変数設定
export ANTHROPIC_API_KEY="sk-ant-..."  # フォールバック用
export CLAUDE_API_PORT=8070

# サーバー起動
python3.13 /opt/claude_api_server.py

# または systemd で自動起動設定
# （後述）
```

### 5. systemd サービス設定 (オプション, claude-dedicated LXC 内)

```bash
# /etc/systemd/system/claude-api-server.service を作成

sudo tee /etc/systemd/system/claude-api-server.service > /dev/null << 'EOF'
[Unit]
Description=Claude API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt
Environment="ANTHROPIC_API_KEY=sk-ant-..."
Environment="CLAUDE_API_PORT=8070"
ExecStart=/opt/venv/bin/python3 /opt/claude_api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable claude-api-server
sudo systemctl start claude-api-server

# ステータス確認
sudo systemctl status claude-api-server
```

### 6. bushidan-honjin LXC での環境変数設定

```bash
# bushidan-honjin LXC の .env に追加

# Claude API Server (claude-dedicated LXC)
CLAUDE_API_SERVER_URL=http://192.168.11.237:8070
CLAUDE_API_TIMEOUT=60
```

### 7. 動作確認

**ヘルスチェック:**
```bash
# bushidan-honjin から
curl http://192.168.11.237:8070/health
# → {"status": "ok", "service": "claude-api-server"}
```

**API ステータス:**
```bash
curl http://192.168.11.237:8070/api/status
# → {
#      "status": "ok",
#      "claude_cli_available": true,
#      "anthropic_api_available": true,
#      "cli_path": "/home/claude/.local/bin/claude"
#    }
```

**テストリクエスト:**
```bash
curl -X POST http://192.168.11.237:8070/api/claude \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, Claude!",
    "system": "You are a helpful assistant.",
    "max_tokens": 100
  }'

# → {
#      "content": "Hello! I'm Claude, an AI assistant...",
#      "model": "claude-pro-cli",
#      "source": "cli",
#      "error": null
#    }
```

## トラブルシューティング

### Claude CLI が見つからない
```bash
# claude-dedicated LXC で確認
which claude
# 表示されない場合は再インストール
curl -fsSL https://claude.ai/install.sh | bash
```

### API Server が起動しない
```bash
# ポート確認
netstat -tuln | grep 8070

# ファイアウォール確認
ufw allow 8070

# ログ確認
journalctl -u claude-api-server -f
```

### bushidan-honjin から接続できない
```bash
# bushidan-honjin から疎通確認
ping 192.168.11.237
curl http://192.168.11.237:8070/health

# ネットワーク設定確認
ip route show
```

### フォールバックが頻発する場合
- Claude CLI のライセンス・クォータ確認
- Anthropic API キーの有効期限確認
- ログレベル変更: `DEBUG=1 python3 claude_api_server.py`

## アーキテクチャ図

```
bushidan-honjin LXC (192.168.11.230)
       ↓ HTTP POST
       ↓ /api/claude
       ↓
claude-dedicated LXC (192.168.11.237)
   ├─ Claude Pro CLI (優先)
   │   └─ [Proプラン枠で実行]
   │
   ├─ Anthropic API (フォールバック)
   │   └─ ANTHROPIC_API_KEY 使用
   │
   └─ Flask API Server (8070)
       └─ JSON レスポンス返却
```

## 参考リンク

- [Claude CLI インストール](https://claude.ai/)
- [Anthropic API ドキュメント](https://docs.anthropic.com/)
- [Flask ドキュメント](https://flask.palletsprojects.com/)
