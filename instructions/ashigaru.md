# 足軽 (Ashigaru) - 役割定義書

## 役職
**足軽 × 8** - MCPサーバー群 (ツール実行層)

## 実体
MCPサーバー (各50-150MB)。LLMではなくツール実行エージェント。
侍大将の指示に従い、特定の機能を実行する。

## 編成

| ID | 名前 | MCP Server | 役割 |
|----|------|-----------|------|
| 1 | 足軽1番 | filesystem | ファイル操作（読み書き・検索） |
| 2 | 足軽2番 | github | Git/GitHub操作（commit, PR, issue） |
| 3 | 足軽3番 | fetch | Web情報取得（URL fetch, scraping） |
| 4 | 足軽4番 | memory | 長期記憶（Knowledge Graph） |
| 5 | 足軽5番 | postgres | データベース操作（SQL実行） |
| 6 | 足軽6番 | puppeteer | ブラウザ自動化（E2Eテスト） |
| 7 | 足軽7番 | brave-search | Web検索（情報収集） |
| 8 | 足軽8番 | slack | チーム連携（メッセージ投稿） |

## 動作環境
- CT 100 (192.168.1.10) でホスト
- Node.js 20+ で実行
- npmグローバルインストール

## RACE-001 ルール
- 各足軽は自分専用のファイルのみ読み書き
- `queue/tasks/ashigaru{N}.yaml` は足軽Nのみが読む
- `queue/reports/ashigaru{N}_report.yaml` は足軽Nのみが書く
- 複数の足軽が同じファイルを同時に操作してはならない

## レポート形式
```yaml
worker_id: ashigaru1
task_id: subtask_001
timestamp: "2026-01-28T10:15:00"
status: done          # done | failed | blocked
result:
  summary: "完了でござる"
  files_modified:
    - "/path/to/file"
  notes: ""
skill_candidate:
  found: false        # 必須フィールド
  name: null
  description: null
```

## コスト
¥0 (ローカル実行)
