# Maintenance Scripts

武士団マルチエージェントシステムのメンテナンス用スクリプト集

## 📋 目次

- [APIキーチェック](#1-apiキーチェック)
- [システムヘルスチェック](#2-システムヘルスチェック)
- [パフォーマンス分析](#3-パフォーマンス分析)
- [メモリクリーンアップ](#4-メモリクリーンアップ)
- [ログローテーション](#5-ログローテーション)
- [システムバックアップ](#6-システムバックアップ)
- [モデル更新チェック](#7-モデル更新チェック)
- [Discord LLMアカウント設定](#8-discord-llmアカウント設定)

---

## 1. APIキーチェック

### `check_api_keys.py`

.envに設定されている全APIキーの有効性を一括チェックします。

#### 使い方

```bash
# 基本的な使い方
python maintenance/check_api_keys.py

# JSON形式で出力（CI/CDやスクリプト連携用）
python maintenance/check_api_keys.py --json
```

#### チェック対象API

1. **Claude Sonnet 4.6** - 将軍のメインモデル
2. **Gemini 3 Flash Preview** - 最終防衛線
3. **OpenRouter API** - 軍師・影武者
4. **Kimi API** - 傭兵（並列実行）
5. **Groq API** - Simple タスク高速処理
6. **Tavily API** - Web検索
7. **Discord Bot Token** - Discord連携
8. **Notion API** - 長期記憶
9. **GitHub Token** - Git操作

---

## 2. システムヘルスチェック

### `check_system_health.py`

システム全体の健全性を診断します。

#### 使い方

```bash
# 基本チェック
python maintenance/check_system_health.py

# 詳細情報を含む
python maintenance/check_system_health.py --detailed

# JSON出力
python maintenance/check_system_health.py --json
```

#### チェック項目

- **ディスク容量** - 空き容量、使用率
- **メモリ** - 使用率、空き容量
- **CPU** - 使用率、コア数
- **llama.cpp サーバー** - 稼働状態
- **環境変数** - 必須・オプション変数の設定状況
- **設定ファイル** - 存在確認
- **ログディレクトリ** - サイズ、ファイル数
- **メモリデータベース** - ファイル存在、サイズ
- **Pythonパッケージ** - 必須パッケージの確認

#### 終了コード

- **0**: 正常
- **1**: 警告あり
- **2**: 重大な問題あり

---

## 3. パフォーマンス分析

### `analyze_performance.py`

ログファイルからタスク処理時間、ルーティング決定、コストを分析します。

#### 使い方

```bash
# 過去30日間を分析
python maintenance/analyze_performance.py

# 過去7日間を分析
python maintenance/analyze_performance.py --days 7

# JSON出力
python maintenance/analyze_performance.py --json

# カスタムログディレクトリ
python maintenance/analyze_performance.py --log-dir /path/to/logs
```

#### 分析内容

- **タスク処理時間統計** - 複雑度別の平均・最小・最大・中央値
- **ルーティング統計** - 各ルートの使用回数と割合
- **モデル使用統計** - モデル別の使用回数
- **エラー統計** - エラータイプ別の発生回数
- **コスト分析** - 総コスト、タスクあたり平均

---

## 4. メモリクリーンアップ

### `cleanup_memory.py`

メモリデータベースから古いエントリを削除し、重複を統合します。

#### 使い方

```bash
# 実行前に確認（DRY RUN）
python maintenance/cleanup_memory.py --dry-run

# 90日以前のエントリを削除
python maintenance/cleanup_memory.py --days 90

# バックアップを作成してからクリーンアップ
python maintenance/cleanup_memory.py --days 90 --backup

# 30日以前のエントリを削除（バックアップあり）
python maintenance/cleanup_memory.py --days 30 --backup
```

#### 処理内容

1. 古いエントリの削除（指定日数以前）
2. 重複エントリの削除
3. 統計情報の表示

#### 対象ファイル

- `shogun_memory.jsonl`
- `bushidan_memory.jsonl`

---

## 5. ログローテーション

### `rotate_logs.py`

古いログファイルをアーカイブ・圧縮します。

#### 使い方

```bash
# 実行前に確認（DRY RUN）
python maintenance/rotate_logs.py --dry-run

# 30日以前のログをアーカイブ
python maintenance/rotate_logs.py --days 30

# 圧縮してアーカイブ
python maintenance/rotate_logs.py --days 30 --compress

# 7日以前のログを圧縮アーカイブ
python maintenance/rotate_logs.py --days 7 --compress
```

#### 処理内容

1. 指定日数以前のログファイルを検出
2. `backups/logs/` ディレクトリにアーカイブ
3. オプションで gzip 圧縮
4. 元ファイルを削除

---

## 6. システムバックアップ

### `backup_system.py`

重要なシステムファイルをバックアップします。

#### 使い方

```bash
# 基本バックアップ（設定・メモリDB・ドキュメント）
python maintenance/backup_system.py

# フルバックアップ（ソースコード含む）
python maintenance/backup_system.py --full

# カスタム保存先
python maintenance/backup_system.py --destination /path/to/backup
```

#### バックアップ対象

**基本バックアップ:**
- 設定ファイル（config/, bushidan_config.yaml等）
- メモリデータベース（.jsonl）
- ドキュメント（README.md等）

**フルバックアップ:**
- 上記 + ソースコード（core/, utils/, bushidan/, maintenance/）

#### バックアップ先

- デフォルト: `backups/system_YYYYMMDD_HHMMSS/`
- カスタム: `--destination` で指定

---

## 7. モデル更新チェック

### `check_model_updates.py`

使用中のLLMモデルと最新版を比較し、更新を通知します。

#### 使い方

```bash
# 更新チェック
python maintenance/check_model_updates.py

# JSON出力
python maintenance/check_model_updates.py --json
```

#### チェック内容

- **現在のモデルバージョン** - settings.yamlから取得
- **最新モデルバージョン** - 既知の最新版と比較
- **非推奨モデル警告** - 廃止予定モデルの検出
- **更新推奨** - アップデート可能なモデルの表示

#### 終了コード

- **0**: すべて最新
- **1**: アップデートあり
- **2**: 非推奨モデル使用中

---

## 8. Discord LLMアカウント設定

### `setup_discord_llm_accounts.py`

各LLMエージェント用のDiscordウェブフックを作成し、スレッド内での個別会話を実現します。

#### 使い方

```bash
# ウェブフック作成（チャンネルID: 1234567890）
python maintenance/setup_discord_llm_accounts.py --create-webhooks 1234567890

# アカウント一覧表示
python maintenance/setup_discord_llm_accounts.py --list

# ウェブフックテスト送信
python maintenance/setup_discord_llm_accounts.py --test

# 統合コード例を表示
python maintenance/setup_discord_llm_accounts.py --code
```

#### 作成されるアカウント

1. **🎌 将軍 (Shogun)** - Claude Sonnet 4.6
2. **📋 軍師 (Gunshi)** - Qwen3-Coder-Next
3. **👔 家老 (Karo)** - Gemini 3 Flash
4. **⚔️ 大将 (Taisho)** - Local Qwen3
5. **🗡️ 傭兵 (Yohei)** - Kimi K2.5
6. **👁️ 検校 (Kengyo)** - Kimi Vision
7. **⚡ 足軽-Groq (Ashigaru)** - Llama 3.3 70B

#### 設定ファイル

作成されたウェブフック情報は以下に保存されます:
- `config/discord_llm_accounts.json`

#### 統合方法

`--code` オプションで統合コード例を表示できます。

---

## CI/CD統合例

### GitHub Actions

```yaml
name: Daily Maintenance
on:
  schedule:
    - cron: '0 0 * * *'  # 毎日0時
  workflow_dispatch:

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Health Check
        run: python maintenance/check_system_health.py --json
      - name: Model Update Check
        run: python maintenance/check_model_updates.py --json
```

---

## 定期実行の推奨スケジュール

| スクリプト | 頻度 | 目的 |
|-----------|------|------|
| check_api_keys.py | 毎日 | APIキーの有効性確認 |
| check_system_health.py | 毎日 | システム健全性監視 |
| analyze_performance.py | 毎週 | パフォーマンス傾向分析 |
| cleanup_memory.py | 毎月 | メモリDB最適化 |
| rotate_logs.py | 毎月 | ログディスク節約 |
| backup_system.py | 毎週 | データ保護 |
| check_model_updates.py | 毎週 | 最新モデル追従 |

---

## トラブルシューティング

### Kimi APIのレート制限

```bash
# .envファイルに以下を追加してOpenRouter経由で使用
KIMI_PROVIDER=openrouter
```

### Google Gemini APIの非推奨警告

```bash
# 新しいパッケージに移行（将来）
pip install google-genai
```

### ディスク容量不足

```bash
# ログローテーションと圧縮
python maintenance/rotate_logs.py --days 7 --compress

# メモリクリーンアップ
python maintenance/cleanup_memory.py --days 30 --backup
```

---

## 実行環境

### 前提条件

- Python 3.10+
- 仮想環境の有効化
- `.env` ファイルの設定

### 依存パッケージ

```bash
pip install anthropic google-generativeai groq requests python-dotenv psutil discord.py aiohttp pyyaml
```

---

## 貢献

メンテナンススクリプトの追加や改善のアイデアがあれば、issueやPRを作成してください。

## ライセンス

武士団マルチエージェントシステムと同じライセンスに従います。
