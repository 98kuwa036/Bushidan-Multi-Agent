# 将軍システム - 月次メンテナンス（反省会）手順書

## 概要

月に1回（毎月1日）、システムの定期メンテナンスを実施します。
この「反省会」では以下の項目をチェック・更新します。

## チェック項目

### 1. LLMバージョン確認

**対象:**
- Claude CLI (`@anthropic-ai/claude-code`)
- Anthropic API モデル (claude-opus-4-5, claude-sonnet-4-5)

**確認方法:**
```bash
shogun maintenance check llm
```

**手動更新:**
```bash
# Claude CLI 更新
npm update -g @anthropic-ai/claude-code

# settings.yaml のモデル名を最新に更新
# cloud.shogun.api_model: "claude-opus-4-5-YYYYMMDD"
# cloud.karo.api_model: "claude-sonnet-4-5-YYYYMMDD"
```

**注意点:**
- APIモデル名変更時は `config/settings.yaml` を手動更新
- Claude CLI 更新後は `shogun health` で動作確認

### 2. OpenVINOモデル確認（侍大将）

**対象:**
- DeepSeek-R1-Distill-Qwen-14B (INT8)

**確認方法:**
```bash
shogun maintenance check openvino
```

**更新判断基準:**
- 新バージョンで推論品質が向上した場合
- セキュリティパッチがリリースされた場合

**更新手順:**
```bash
# CT 101 で実行
pct enter 101
bash /path/to/shogun/setup/openvino_setup.sh
```

### 3. MCPサーバー更新（足軽）

**対象:**
- 8つのMCPサーバーパッケージ

**確認・更新方法:**
```bash
shogun maintenance check mcp

# 手動更新
npm update -g @modelcontextprotocol/server-filesystem \
    @modelcontextprotocol/server-github \
    @modelcontextprotocol/server-fetch \
    @modelcontextprotocol/server-memory \
    @modelcontextprotocol/server-postgres \
    @modelcontextprotocol/server-puppeteer \
    @modelcontextprotocol/server-brave-search \
    @modelcontextprotocol/server-slack
```

### 4. システムヘルスチェック

**確認項目:**
- Python venv 状態
- 設定ファイル存在確認
- ディスク空き容量
- 環境変数設定

**確認方法:**
```bash
shogun maintenance check health
shogun health
```

### 5. ログクリーンアップ

**自動実行:**
- 30日以上経過したログファイルを自動削除

**手動実行:**
```bash
shogun maintenance check logs
```

### 6. コストレポート

**確認内容:**
- 月間API使用量
- Pro契約費用
- 電力費用
- 合計コスト

**確認方法:**
```bash
shogun maintenance check cost
```

## 自動実行設定

### cron 設定

```bash
# インストール（毎月1日 9:00 JST）
bash shogun/setup/maintenance.sh install

# アンインストール
bash shogun/setup/maintenance.sh uninstall

# 状態確認
bash shogun/setup/maintenance.sh status
```

### 手動実行

```bash
# CLI から
shogun maintenance run

# スクリプトから
bash shogun/setup/maintenance.sh run
```

## レポート確認

### 過去レポート一覧

```bash
shogun maintenance reports
shogun maintenance reports -n 20  # 直近20件
```

### レポート保存場所

```
shogun/reports/maintenance/
├── maintenance_YYYYMMDD_HHMMSS.json   # 詳細データ
└── maintenance_YYYYMMDD_HHMMSS.md     # Markdown レポート
```

## トラブルシューティング

### メンテナンス失敗時

1. ログ確認
```bash
cat shogun/logs/maintenance.log
```

2. 個別チェック実行
```bash
shogun maintenance check llm
shogun maintenance check openvino
shogun maintenance check mcp
shogun maintenance check health
```

3. 手動修復後に再実行
```bash
shogun maintenance run
```

### よくある問題

| 問題 | 対処法 |
|------|--------|
| Claude CLI 更新失敗 | `npm cache clean --force` 後に再試行 |
| 侍大将接続不可 | CT 101 の再起動: `pct restart 101` |
| MCP 更新失敗 | npm キャッシュクリア、ネットワーク確認 |
| ディスク容量不足 | ログクリーンアップ、不要ファイル削除 |

## スケジュール

| 項目 | 頻度 | 自動/手動 |
|------|------|-----------|
| LLMバージョン確認 | 月1回 | 手動確認推奨 |
| OpenVINOモデル確認 | 月1回 | 手動確認推奨 |
| MCPサーバー更新 | 月1回 | 自動可能 |
| システムヘルスチェック | 月1回 | 自動 |
| ログクリーンアップ | 月1回 | 自動 |
| コストレポート | 月1回 | 自動 |

## 設定変更

`config/settings.yaml` の `maintenance` セクションで設定可能:

```yaml
maintenance:
  schedule:
    day_of_month: 1  # 実行日
    hour: 9          # 実行時刻
    minute: 0
    timezone: "Asia/Tokyo"

  notification:
    slack_channel: "#shogun-maintenance"
```
