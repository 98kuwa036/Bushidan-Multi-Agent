# 将軍システム Hybrid v5.0

クラウド/ローカルハイブリッド型の階層的AI開発システム。
月額¥4,000-6,000で最高品質の開発環境を実現。

## アーキテクチャ

```
┌──────────────────────────────────────────────────┐
│ 将軍 (Shogun) ── Claude Opus 4.5                 │
│ 最終決裁・戦略判断  ¥1,350/回  月1-5回           │  ☁️ Cloud
│                                                  │  (claude-cli Pro版
│ 家老 (Karo) ── Claude Sonnet 4.5                 │   → API fallback)
│ 作業割振り・複雑実装  ¥280/回  月10-20回         │
└──────────────────────────────────────────────────┘
                       ↑ エスカレーション
━━━━━━━━━━━━━━━ API / Local 境界 ━━━━━━━━━━━━━━━
                       ↓ 指示
┌──────────────────────────────────────────────────┐
│ 侍大将 (Taisho) ── DeepSeek-R1-14B OpenVINO     │  🏰 CT 101
│ 設計・推論・足軽の意見統一  ¥0  月900+回         │  (192.168.1.11)
└──────────────────────────────────────────────────┘
                       ↓ MCP指示
┌──────────────────────────────────────────────────┐
│ 足軽 × 8 ── MCPサーバー群                        │  ⚔️ CT 100
│ 1.filesystem 2.github 3.fetch 4.memory           │  (192.168.1.10)
│ 5.postgres 6.puppeteer 7.brave-search 8.slack    │
└──────────────────────────────────────────────────┘
```

## 部隊編成

### 大隊モード (Battalion) - デフォルト
- **全階層利用可能**: 将軍 + 家老 + 侍大将 + 足軽×8
- 複雑度に応じて自動ルーティング
- Simple/Medium → 侍大将 (¥0)
- Complex → 侍大将分析 → 家老実装 (¥280)
- Strategic → 将軍決裁 (¥1,350)

### 中隊モード (Company) - コスト優先
- **侍大将 + 足軽のみ**: API不使用 ¥0
- Simple/Mediumタスク専用
- 能力超過時は大隊モード推奨を通知

## クイックスタート

### インストール
```bash
cd /path/to/codings
bash shogun/setup/install.sh
source shogun/.venv/bin/activate
```

### IDE コンソールから使う
```bash
# 対話モード (REPL)
shogun repl

# 直接タスク実行
shogun ask "ESP32-P4のSPI DMAの設定方法は？"

# 中隊モード (¥0)
shogun ask -m company "I2Sバッファサイズを決めて"

# エージェント指定
shogun ask -a taisho "このビルドエラーを分析して"

# パイプ入力 (ログ解析)
cat build.log | shogun pipe

# APIサーバー起動
shogun server --port 8080

# ヘルスチェック
shogun health
```

### REPL コマンド
```
/mode [battalion|company]  モード切替
/agent [taisho|karo|shogun] エージェント固定
/agent                     自動選択に戻す
/status                    ステータス
/stats                     コスト統計
/health                    ヘルスチェック
quit                       終了
```

## REST API

サーバー起動: `shogun server --port 8080`

| Endpoint | Method | 説明 |
|----------|--------|------|
| `/api/task` | POST | タスク投入 (大隊/中隊) |
| `/api/ask` | POST | HA OS音声用 (中隊固定) |
| `/api/status` | GET | システム状態 |
| `/api/stats` | GET | コスト統計 |
| `/api/health` | GET | ヘルスチェック |
| `/api/mode` | POST | モード切替 |
| `/api/mcp` | GET | MCP (足軽) 状態 |
| `/api/dashboard` | GET | 戦況報告 |

```bash
# 大隊モード
curl -X POST http://192.168.1.10:8080/api/task \
  -H 'Content-Type: application/json' \
  -d '{"task": "Spotify統合実装", "mode": "battalion"}'

# 中隊モード
curl -X POST http://192.168.1.10:8080/api/task \
  -d '{"task": "I2Sバッファサイズ", "mode": "company"}'

# HA OS音声 (中隊固定)
curl -X POST http://192.168.1.10:8080/api/ask \
  -d '{"question": "I2Sバッファは何サンプル推奨？"}'
```

## インターフェース

| 経路 | 使用頻度 | 部隊 | コスト |
|------|---------|------|--------|
| エディタ拡張 (Pro版) | 95% | - | ¥0 (Pro月額内) |
| Slack | 3% | 大隊/中隊 | 状況依存 |
| HA OS 音声 | 2% | 中隊固定 | ¥0 |
| API直接 | <1% | 指定可能 | 状況依存 |

## ハードウェア構成

| 機器 | スペック | 役割 |
|------|---------|------|
| HP ProDesk 600 G4 | i5-8500, 24GB, Proxmox | CT100 本陣 + CT101 侍大将 |
| Raspberry Pi 4B | 8GB, HA OS | 音声アシスタント, 中隊トリガー |
| メインPC | macOS/Linux/Windows | 日常開発 (エディタ拡張) |

## コスト構成

```
固定費:
  Claude Pro: ¥3,000/月
  電力 (24h稼働): ¥800/月

変動費:
  Anthropic API: ¥1,000-3,000/月
  (中隊モード活用で大幅削減)

合計: ¥4,000-6,000/月
```

## ディレクトリ構成

```
shogun/
├── cli.py                  CLI (REPL + コマンド)
├── main.py                 FastAPI サーバー
├── core/
│   ├── controller.py       本陣 (大隊/中隊制御)
│   ├── task_queue.py       YAML タスクキュー
│   ├── escalation.py       エスカレーション連鎖
│   ├── complexity.py       複雑度判定エンジン
│   ├── dashboard.py        戦況報告 (dashboard.md)
│   └── mcp_manager.py      MCP (足軽×8) 管理
├── providers/
│   ├── claude_cli.py       Claude CLI (Pro版) ラッパー
│   ├── anthropic_api.py    Anthropic API (フォールバック)
│   └── openvino_client.py  OpenVINO R1 クライアント
├── integrations/
│   ├── slack_bot.py        Slack 10ボット統合
│   └── ha_interface.py     HA OS 音声連携
├── instructions/           エージェント役割定義
├── config/
│   ├── settings.yaml       システム設定
│   └── mcp_config.json     MCP サーバー定義
├── setup/
│   ├── proxmox_setup.sh    Proxmox LXC 作成
│   ├── openvino_setup.sh   R1 モデル変換 + サーバー
│   └── install.sh          本陣インストール
├── queue/                  タスクキュー (YAML)
│   ├── tasks/              足軽別タスクファイル
│   └── reports/            足軽別レポート
├── status/                 dashboard.md
├── context/                プロジェクトコンテキスト
└── templates/              テンプレート
```

## 構築手順

1. **Proxmox**: `bash setup/proxmox_setup.sh local-lvm`
2. **侍大将 (CT101)**: `pct enter 101 && bash setup/openvino_setup.sh`
3. **本陣 (CT100)**: `pct enter 100 && bash setup/install.sh`
4. **環境変数**: `.env` にAPIキー設定
5. **動作確認**: `shogun health`

## 参考

- [multi-agent-shogun](https://github.com/yohey-w/multi-agent-shogun) - 設計思想の原典
