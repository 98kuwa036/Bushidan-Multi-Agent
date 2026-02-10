# 武士団マルチエージェントシステム v10.1 - API セットアップガイド

このガイドでは、武士団システムで使用する各APIの取得方法とセットアップ手順を説明します。

---

## 目次

1. [Slack Bot Token](#1-slack-bot-token)
2. [Claude API (Anthropic)](#2-claude-api-anthropic)
3. [Gemini API (Google)](#3-gemini-api-google)
4. [Tavily API](#4-tavily-api)
5. [Groq API](#5-groq-api)
6. [Kimi K2.5 API (Moonshot)](#6-kimi-k25-api-moonshot)
7. [Alibaba Cloud (Qwen3-Coder-Next)](#7-alibaba-cloud-qwen3-coder-next)
8. [Notion API](#8-notion-api)
9. [環境変数の設定](#9-環境変数の設定)

---

## 1. Slack Bot Token

Slack経由で武士団システムを操作するために必要です。

### 公式ドキュメント
- https://api.slack.com/docs/quickstart
- https://api.slack.com/authentication/basics

### Step 1: Slack App 作成

1. [Slack API](https://api.slack.com/apps) にアクセス
2. **Create New App** をクリック
3. **From scratch** を選択
4. 以下を入力:
   - **App Name**: `Bushidan` (任意の名前)
   - **Pick a workspace**: 使用するワークスペースを選択
5. **Create App** をクリック

### Step 2: Bot Token Scopes 設定

1. 左メニューから **OAuth & Permissions** を選択
2. **Scopes** セクションまでスクロール
3. **Bot Token Scopes** で **Add an OAuth Scope** をクリック
4. 以下のスコープを追加:

```
app_mentions:read      - @メンション検知
channels:history       - パブリックチャンネルのメッセージ読み取り
channels:read          - チャンネル情報取得
chat:write             - メッセージ送信
files:read             - ファイル読み取り
files:write            - ファイルアップロード
groups:history         - プライベートチャンネルのメッセージ読み取り
groups:read            - プライベートチャンネル情報取得
im:history             - DMのメッセージ読み取り
im:read                - DM情報取得
im:write               - DM送信
users:read             - ユーザー情報取得
```

### Step 3: Event Subscriptions 設定

1. 左メニューから **Event Subscriptions** を選択
2. **Enable Events** を **On** に切り替え
3. **Request URL** を設定:
   ```
   http://<あなたのサーバーIP>:3000/slack/events
   ```
   - ローカル開発の場合は [ngrok](https://ngrok.com/) を使用
4. **Subscribe to bot events** で以下を追加:
   ```
   app_mention        - @メンションイベント
   message.channels   - パブリックチャンネルメッセージ
   message.groups     - プライベートチャンネルメッセージ
   message.im         - DMメッセージ
   ```
5. **Save Changes** をクリック

### Step 4: App をワークスペースにインストール

1. 左メニューから **Install App** を選択
2. **Install to Workspace** をクリック
3. 権限を確認し **許可する** をクリック
4. 表示される **Bot User OAuth Token** をコピー
   - `xoxb-` で始まる文字列

### Step 5: Signing Secret 取得

1. 左メニューから **Basic Information** を選択
2. **App Credentials** セクションの **Signing Secret** をコピー

### 取得するトークン

| 環境変数 | 値の形式 | 取得場所 |
|---------|---------|---------|
| `SLACK_BOT_TOKEN` | `xoxb-...` | OAuth & Permissions |
| `SLACK_SIGNING_SECRET` | 32文字の英数字 | Basic Information |

---

## 2. Claude API (Anthropic)

将軍 (Shogun) 層で使用するClaude Sonnet/Opus APIです。

### 公式ドキュメント
- https://docs.anthropic.com/en/docs/initial-setup
- https://console.anthropic.com/

### Step 1: アカウント作成

1. [Anthropic Console](https://console.anthropic.com/) にアクセス
2. **Sign Up** でアカウント作成
3. メール認証を完了

### Step 2: API Key 作成

1. ログイン後、左メニューから **API Keys** を選択
2. **Create Key** をクリック
3. キーに名前を付ける（例: `bushidan-production`）
4. **Create Key** をクリック
5. 表示されるAPIキーをコピー
   - `sk-ant-` で始まる文字列
   - **一度しか表示されないので必ず保存**

### Step 3: 支払い設定

1. 左メニューから **Billing** を選択
2. クレジットカードを登録
3. 使用量上限を設定（推奨）

### 料金目安 (2024年時点)

| モデル | 入力 (1M tokens) | 出力 (1M tokens) |
|--------|-----------------|-----------------|
| Claude Sonnet 4.5 | $3 | $15 |
| Claude Opus 4 | $15 | $75 |

### 取得するトークン

| 環境変数 | 値の形式 |
|---------|---------|
| `CLAUDE_API_KEY` | `sk-ant-api03-...` |

---

## 3. Gemini API (Google)

最終防衛線 (Gemini 3 Flash) として使用します。

### 公式ドキュメント
- https://ai.google.dev/gemini-api/docs/quickstart
- https://aistudio.google.com/

### Step 1: Google AI Studio にアクセス

1. [Google AI Studio](https://aistudio.google.com/) にアクセス
2. Googleアカウントでログイン

### Step 2: API Key 作成

1. 左メニューから **Get API key** を選択
2. **Create API key** をクリック
3. プロジェクトを選択（または新規作成）
4. 生成されたAPIキーをコピー

### 無料枠

- **15 RPM** (リクエスト/分)
- **1,500 RPD** (リクエスト/日)
- **1,000,000 TPM** (トークン/分)

### 取得するトークン

| 環境変数 | 値の形式 |
|---------|---------|
| `GEMINI_API_KEY` | `AIzaSy...` |

---

## 4. Tavily API

Web検索MCPで使用します。

### 公式ドキュメント
- https://docs.tavily.com/docs/welcome
- https://tavily.com/

### Step 1: アカウント作成

1. [Tavily](https://tavily.com/) にアクセス
2. **Get Started** または **Sign Up** をクリック
3. メールアドレスでアカウント作成

### Step 2: API Key 取得

1. ダッシュボードにログイン
2. **API Keys** セクションでキーをコピー
   - `tvly-` で始まる文字列

### 無料枠

- **1,000 検索/月** (無料)
- 超過分は $0.01/検索

### 取得するトークン

| 環境変数 | 値の形式 |
|---------|---------|
| `TAVILY_API_KEY` | `tvly-...` |

---

## 5. Groq API

Simple タスクの高速処理に使用します（Llama 3.3 70B）。

### 公式ドキュメント
- https://console.groq.com/docs/quickstart
- https://console.groq.com/

### Step 1: アカウント作成

1. [Groq Console](https://console.groq.com/) にアクセス
2. **Sign Up** でアカウント作成
3. メール認証を完了

### Step 2: API Key 作成

1. ダッシュボードにログイン
2. 左メニューから **API Keys** を選択
3. **Create API Key** をクリック
4. キーに名前を付けて作成
5. 表示されるAPIキーをコピー
   - `gsk_` で始まる文字列

### 無料枠

- **14,400 リクエスト/日**
- **速度制限**: 30 リクエスト/分

### 取得するトークン

| 環境変数 | 値の形式 |
|---------|---------|
| `GROQ_API_KEY` | `gsk_...` |

---

## 6. Kimi K2.5 API (Moonshot)

傭兵 (Yohei) として並列サブタスク実行に使用します。

### 公式ドキュメント
- https://platform.moonshot.cn/docs/
- https://platform.moonshot.cn/

### Step 1: アカウント作成

1. [Moonshot Platform](https://platform.moonshot.cn/) にアクセス
2. アカウント登録（中国の電話番号が必要な場合あり）
3. または [OpenRouter](https://openrouter.ai/) 経由で使用

### Step 2: API Key 取得 (Moonshot直接)

1. ダッシュボードにログイン
2. **API管理** から新しいキーを作成
3. APIキーをコピー

### Step 2 (代替): OpenRouter 経由

OpenRouterを使用すると、中国の電話番号なしでKimi K2.5にアクセスできます。

1. [OpenRouter](https://openrouter.ai/) にアクセス
2. アカウント作成
3. **Keys** からAPIキーを作成
4. モデル名: `moonshot/kimi-k2.5`

### 取得するトークン

| 環境変数 | 値の形式 | 備考 |
|---------|---------|------|
| `KIMI_API_KEY` | プロバイダによる | |
| `KIMI_PROVIDER` | `moonshot` or `openrouter` | デフォルト: moonshot |

---

## 7. Alibaba Cloud (Qwen3-Coder-Next)

軍師 (Gunshi) 層のAPIと影武者 (Kagemusha) で使用します。

### 公式ドキュメント
- https://help.aliyun.com/zh/model-studio/developer-reference/
- https://dashscope.console.aliyun.com/

### Step 1: Alibaba Cloud アカウント作成

1. [Alibaba Cloud](https://www.alibabacloud.com/) にアクセス
2. アカウント作成（クレジットカード必要）
3. 実名認証を完了

### Step 2: DashScope API Key 取得

1. [DashScope Console](https://dashscope.console.aliyun.com/) にアクセス
2. **API-KEY管理** を選択
3. **新しいAPI-KEYを作成** をクリック
4. APIキーをコピー

### 料金目安

| モデル | 入力 (1K tokens) | 出力 (1K tokens) |
|--------|-----------------|-----------------|
| Qwen3-Coder-Next | ¥0.002 | ¥0.006 |
| Qwen3-Plus | ¥0.004 | ¥0.012 |

### 取得するトークン

| 環境変数 | 値の形式 |
|---------|---------|
| `ALIBABA_API_KEY` | `sk-...` |
| `QWEN3_CODER_NEXT_API_KEY` | `sk-...` (同じキー) |

---

## 8. Notion API

長期記憶ストレージとして使用します（オプション）。

### 公式ドキュメント
- https://developers.notion.com/docs/getting-started
- https://www.notion.so/my-integrations

### Step 1: Integration 作成

1. [Notion Integrations](https://www.notion.so/my-integrations) にアクセス
2. **New integration** をクリック
3. 以下を入力:
   - **Name**: `Bushidan`
   - **Associated workspace**: 使用するワークスペース
   - **Capabilities**:
     - Read content
     - Update content
     - Insert content
4. **Submit** をクリック

### Step 2: Internal Integration Secret 取得

1. 作成したIntegrationの詳細ページを開く
2. **Internal Integration Secret** をコピー
   - `secret_` で始まる文字列

### Step 3: データベースと連携

1. Notionでデータベースを作成
2. データベースページの **...** メニューから **Add connections** を選択
3. 作成した `Bushidan` Integrationを選択

### 取得するトークン

| 環境変数 | 値の形式 |
|---------|---------|
| `NOTION_API_KEY` | `secret_...` |

---

## 9. 環境変数の設定

### .env ファイルの作成

CT 100 (本陣) で以下のコマンドを実行:

```bash
cd ~/Bushidan-Multi-Agent
cp .env.example .env
nano .env
```

### 完全な .env 設定例

```bash
# =============================================================================
# 武士団マルチエージェントシステム v10.1 - 環境変数設定
# =============================================================================

# ===== 必須 (最低限これだけあれば動作) =====

# Claude API (将軍層)
CLAUDE_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Gemini API (最終防衛線)
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ===== Slack 連携 (Slack経由で使用する場合) =====

SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
SLACK_SIGNING_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ===== llama.cpp (ローカル推論) =====

# CT 101 のIPアドレスを設定
LLAMACPP_ENDPOINT=http://192.168.11.231:8080

# ===== 推奨 (パフォーマンス向上) =====

# Groq API (Simple タスク高速化 - 無料)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Tavily API (Web検索 - 1000回/月無料)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ===== オプション (追加機能) =====

# Kimi K2.5 (傭兵 - 並列サブタスク実行)
KIMI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
KIMI_PROVIDER=moonshot  # or openrouter

# Alibaba Cloud (軍師 + 影武者)
ALIBABA_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
QWEN3_CODER_NEXT_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Notion (長期記憶)
NOTION_API_KEY=secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ===== llama.cpp 詳細設定 (通常は変更不要) =====

LLAMACPP_THREADS=6
LLAMACPP_CONTEXT_SIZE=4096
LLAMACPP_BATCH_SIZE=512
```

### 設定の検証

```bash
# 環境変数が正しく読み込まれるか確認
source .venv/bin/activate
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('CLAUDE_API_KEY:', 'SET' if os.getenv('CLAUDE_API_KEY') else 'NOT SET')"
```

---

## トラブルシューティング

### API キーが認識されない

```bash
# .env ファイルの権限を確認
ls -la .env

# 改行コードを確認 (Windows で編集した場合)
file .env

# Unix 形式に変換
dos2unix .env
```

### Slack Bot が応答しない

1. Event Subscriptions の Request URL が正しいか確認
2. Bot がチャンネルに招待されているか確認
3. ログを確認:
   ```bash
   journalctl -u bushidan-slack -f
   ```

### llama.cpp に接続できない

```bash
# CT 101 で llama-server が起動しているか確認
curl http://192.168.11.231:8080/health

# ファイアウォール確認
sudo ufw status
```

---

## 料金まとめ (月額目安)

| サービス | 無料枠 | 推定月額 (通常使用) |
|---------|--------|-------------------|
| Slack | 無料 | ¥0 |
| Claude API | - | ¥3,000〜5,000 |
| Gemini API | 無料枠大 | ¥0〜100 |
| Tavily | 1,000回/月 | ¥0 |
| Groq | 無料 | ¥0 |
| Kimi K2.5 | - | ¥30〜100 |
| Alibaba Cloud | - | ¥50〜100 |
| **合計** | - | **¥3,000〜5,500** |

---

## 次のステップ

1. `.env` ファイルを設定
2. CT 101 で llama-server を起動
3. Slack Bot を起動:
   ```bash
   cd ~/Bushidan-Multi-Agent
   source .venv/bin/activate
   python -m bushidan.slack_bot
   ```
4. Slack で `@Bushidan こんにちは` を送信してテスト
