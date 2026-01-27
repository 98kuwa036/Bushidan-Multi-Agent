# Omni-P4 Shogun-Hybrid "Long Context" v3.0

ESP32-P4低レイヤー開発を支援する、クラウド/ローカルハイブリッド型の階層的AI開発システム。

## 概要

```
☁️ 天空層 (API)          🏰 地上層 (Proxmox / Ollama)
┌─────────────────┐      ┌──────────────────────────────┐
│ Shogun (Opus4.5)│      │ Mode A: 軍議                  │
│ Karo (Sonnet4.5)│      │   └─ Taisho (14B Q8 / 32k)   │
│   最終決裁のみ   │      │                                │
└────────┬────────┘      │ Mode B: 進軍                  │
         │ escalation    │   ├─ Leader (8B Q6 / 32k)     │
         ▼               │   ├─ Coder  (7B Q6 / 32k)     │
  ┌──────────────┐       │   └─ Scout  (1.5B Q8 / 4k)    │
  │ CT100 本陣    │◄─────┤                                │
  │ (Controller)  │       │ 排他運用: AとBは同時に動かない │
  └──────────────┘       └──────────────────────────────┘
```

### 特徴

1. **ハイブリッド頭脳**: Claude 4.5 API + ローカルOllama LLM
2. **排他的モード切替**: 24GB RAMを最大活用（Mode A: 思考 / Mode B: 実働）
3. **自動エスカレーション**: Scout → Coder → Leader → Taisho → Karo → Shogun
4. **Long Context**: ローカル足軽に32kトークンの記憶

## クイックスタート

### 1. インストール

```bash
cd /path/to/codings
bash shogun/setup/install.sh
source shogun/.venv/bin/activate
```

### 2. Ollama準備

```bash
# Ollamaインストール (まだの場合)
curl -fsSL https://ollama.com/install.sh | sh

# モデルダウンロード
bash shogun/setup/ollama_setup.sh all
```

### 3. 使用

```bash
# ヘルスチェック
shogun health

# 対話モード (REPL)
shogun repl

# 直接質問
shogun ask "ESP32-P4のSPI DMAの設定方法は？"

# カテゴリ指定
shogun ask -c think "DMAバッファのアライメント設計を考えて"

# エージェント指定
shogun ask -a taisho "このビルドエラーを分析して"

# REST APIサーバー起動
shogun server --port 8400
```

## CLI コマンド一覧

| コマンド | 説明 |
|---------|------|
| `shogun repl` | 対話モード (デフォルト) |
| `shogun ask <prompt>` | タスク実行 |
| `shogun ask -c <cat> <prompt>` | カテゴリ指定で実行 |
| `shogun ask -a <agent> <prompt>` | エージェント指定で実行 |
| `shogun mode` | 現在のモード表示 |
| `shogun mode a` | Mode A (軍議) に切替 |
| `shogun mode b` | Mode B (進軍) に切替 |
| `shogun status` | システム状態表示 |
| `shogun agents` | エージェント一覧 |
| `shogun health` | Ollama接続チェック |
| `shogun models` | ダウンロード済みモデル一覧 |
| `shogun unload` | 全モデルをメモリから解放 |
| `shogun server` | REST APIサーバー起動 |
| `shogun pipe` | stdin からパイプ入力 |

### REPL内コマンド

```
/mode [a|b|cloud]  - モード切替
/cat <category>    - デフォルトカテゴリ変更
/agent <name>      - エージェント固定
/agent             - エージェント固定解除
/status            - ステータス表示
/agents            - エージェント一覧
quit               - 終了
```

### カテゴリ

| カテゴリ | 説明 | デフォルトエージェント |
|---------|------|---------------------|
| `recon` | 偵察 (ログ収集・検索) | Scout |
| `code` | 実装 (コーディング) | Coder |
| `plan` | 設計 (タスク分解) | Leader |
| `think` | 深考 (複雑な推論) | Taisho |
| `strategy` | 戦略 (実装方針) | Karo |
| `critical` | 最終決裁 | Shogun |

## REST API

サーバー起動: `shogun server --port 8400`

| Endpoint | Method | 説明 |
|----------|--------|------|
| `/` | GET | システム情報 |
| `/ask` | POST | タスク投入 |
| `/status` | GET | ステータス |
| `/mode` | POST | モード切替 |
| `/agents` | GET | エージェント一覧 |
| `/tasks` | GET | タスク一覧 |
| `/tasks/{id}` | GET | タスク詳細 |
| `/health` | GET | ヘルスチェック |
| `/unload` | POST | 全モデル解放 |

### API使用例

```bash
# タスク投入
curl -X POST http://localhost:8400/ask \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "ESP32-P4のGPIO設定コードを書いて", "category": "code"}'

# モード切替
curl -X POST http://localhost:8400/mode \
  -H 'Content-Type: application/json' \
  -d '{"mode": "a"}'

# ステータス確認
curl http://localhost:8400/status
```

## IDE連携

### VS Code タスク設定

`.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Shogun: Ask",
      "type": "shell",
      "command": "shogun ask '${input:prompt}'",
      "problemMatcher": []
    },
    {
      "label": "Shogun: REPL",
      "type": "shell",
      "command": "shogun repl",
      "isBackground": true,
      "problemMatcher": []
    },
    {
      "label": "Shogun: Status",
      "type": "shell",
      "command": "shogun status",
      "problemMatcher": []
    }
  ],
  "inputs": [
    {
      "id": "prompt",
      "type": "promptString",
      "description": "Task prompt"
    }
  ]
}
```

### パイプ統合

```bash
# ビルドログを分析
cat build.log | shogun pipe -c recon

# エラーをCoderに投入
grep "error" build.log | shogun pipe -c code -a coder
```

## アーキテクチャ

### エージェント階層

```
☁️ 将軍 (Shogun) ─── Claude Opus 4.5 (最高意思決定者)
☁️ 家老 (Karo)   ─── Claude Sonnet 4.5 (実装参謀)
─────────────────── API / Local 境界 ───────────────────
🏰 侍大将 (Taisho) ── DeepSeek-R1-14B-JP Q8 / 32k (設計・推論)
⚔️ 足軽頭 (Leader) ── Hermes-3-8B Q6 / 32k (現場監督)
⚔️ 技術兵 (Coder)  ── Qwen2.5-Coder-7B Q6 / 32k (実装職人)
⚔️ 小者   (Scout)  ── Qwen2.5-1.5B Q8 / 4k (斥候)
```

### メモリ配分 (24GB)

```
Mode A (軍議):
  CT100 Controller: 1GB
  CT101 Taisho:     22GB (14B Q8 = ~15GB + 32k Context)
  CT102:            停止

Mode B (進軍):
  CT100 Controller: 1GB
  CT101:            停止
  CT102 Ashigaru:   18GB
    ├─ Leader:  6.6GB
    ├─ Coder:   5.9GB
    ├─ Scout:   1.8GB
    └─ Buffer:  ~4GB (Context用)
```

### タスク実行フロー

```
1. タスク受信
   │
2. カテゴリ判定 → デフォルトエージェント選択
   │
3. モード切替 (必要時)
   │  ├─ Mode A: Taisho ロード (22GB)
   │  ├─ Mode B: Leader+Coder+Scout ロード (18GB)
   │  └─ Cloud: ローカルモデル不要
   │
4. エージェント実行
   │
5. 結果判定
   ├─ 成功 → 完了
   └─ 失敗 → エスカレーション
              └─ Scout → Coder → Leader → Taisho → Karo → Shogun
```

## Proxmoxセットアップ

```bash
# Proxmoxホストで実行
bash shogun/setup/proxmox_setup.sh local-lvm

# 各CTでOllamaセットアップ
pct exec 101 -- bash /path/to/shogun/setup/ollama_setup.sh mode_a
pct exec 102 -- bash /path/to/shogun/setup/ollama_setup.sh mode_b

# コントローラーインストール
pct exec 100 -- bash /path/to/shogun/setup/install.sh
```

## 環境変数

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `ANTHROPIC_API_KEY` | Anthropic API キー (Cloud必須) | - |
| `OLLAMA_BASE_URL` | Ollama エンドポイント | `http://localhost:11434` |

## ディレクトリ構成

```
shogun/
├── __init__.py
├── cli.py                  # CLI エントリーポイント
├── main.py                 # FastAPI サーバー
├── requirements.txt
├── pyproject.toml
├── README.md               # このファイル
│
├── core/
│   ├── __init__.py
│   ├── controller.py       # モード切替 & オーケストレーション
│   ├── task_queue.py       # タスクキュー管理
│   └── escalation.py       # エスカレーション制御
│
├── agents/
│   ├── __init__.py
│   ├── base.py             # エージェント基底クラス (Local/Cloud)
│   └── factory.py          # エージェント生成ファクトリ
│
├── providers/
│   ├── __init__.py
│   ├── ollama.py           # Ollama REST API クライアント
│   └── anthropic_api.py    # Anthropic Messages API クライアント
│
├── config/
│   ├── settings.yaml       # システム設定
│   └── modelfiles/
│       ├── taisho.Modelfile
│       ├── leader.Modelfile
│       ├── coder.Modelfile
│       └── scout.Modelfile
│
├── setup/
│   ├── proxmox_setup.sh    # Proxmox LXC 作成
│   ├── ollama_setup.sh     # Ollama + モデル導入
│   └── install.sh          # Controller インストール
│
├── queue/                  # タスクキュー永続化
├── status/                 # ステータスファイル
└── logs/                   # ログディレクトリ
```
