# codings - 将軍AIエージェントシステム

[![Claude Code](https://img.shields.io/badge/Claude%20Code-Powered-blue)](https://claude.ai/code)
[![Version](https://img.shields.io/badge/Claude%20Opus-20251101-green)](https://github.com/anthropics/claude-3-family)

## 概要

`codings`は、日本の武家社会の階層構造をモデルにした革新的なAIエージェントシステムです。将軍（Shogun）を頂点とする組織構造で、複数のAIエージェントが協調して動作し、複雑なタスクを効率的に処理します。

## システム構成

### 階層構造

```
将軍（Shogun）- 最高司令官
├── 家老（Karo）- 重要事項の執行役
├── 大将（Taisho）- 戦略立案・指揮官
└── 足軽（Ashigaru）- 実行部隊（最大4名まで動的選抜）
```

### 主要コンポーネント

- **将軍システム** (`shogun/`) - メインのAIエージェント管理システム
- **家訓管理** - NotionおよびRAGシステムでの二重管理
- **GroqRecorder** - API呼び出し管理とレート制限対応
- **在庫管理システム** (`Inventry-Management/`) - VBA版およびWeb版
- **PWAアプリ** (`pwa-warehouse-app/`) - モバイル対応倉庫管理
- **Home Assistant統合** (`custom_components/nature_remo/`) - IoTデバイス連携

## 主要機能

### 🏯 将軍AIエージェントシステム

- **動的タスク分散**: 複雑性に応じて適切なエージェントに自動分散
- **階層的意思決定**: 武家社会の指揮系統を模した効率的な処理フロー
- **エラーハンドリング**: Exponential Backoffによる堅牢なエラー回復
- **レート制限対応**: GroqのRPM/TPM制限を考慮した分散処理

### 📝 家訓（組織規約）管理

- **Notion統合**: 組織の規約やベストプラクティスをNotion上で管理
- **RAG統合**: 検索拡張生成による家訓の効率的な参照・適用
- **動的更新**: リアルタイムでの家訓の更新と全エージェントへの反映

### ⚡ GroqRecorder - APIレート制限管理

```python
# 日次カウント管理
daily_count = GroqRecorder.get_daily_usage()

# 分間制限対応（RPM/TPM）
rate_limiter = ExponentialBackoffHandler(
    rpm_limit=30,  # リクエスト/分
    tpm_limit=6000,  # トークン/分
    base_delay=1.0,
    max_delay=60.0
)
```

### 🛡️ エラーハンドリング

- **Exponential Backoff**: 段階的な再試行間隔調整
- **レート制限検知**: APIレスポンス監視による制限検知
- **フェイルセーフ**: 代替処理パスによる継続的サービス提供

## インストール

### 必要環境

- Python 3.8+
- Node.js 16+ (PWAアプリ用)
- Claude API Key
- Groq API Key
- Notion API Token（オプション）

### セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/98kuwa036/codings.git
cd codings

# 将軍システムのセットアップ
cd shogun
pip install -r requirements.txt

# 設定ファイルの準備
cp config/settings.yaml.example config/settings.yaml
# API キーを設定してください

# システム起動
python main.py
```

## 使用方法

### 基本的な使用方法

```bash
# 将軍システム起動
cd shogun
python cli.py --mode interactive

# タスクの実行
python cli.py --task "在庫データの分析と報告書作成"
```

### 家訓の管理

```yaml
# config/settings.yaml
family_rules:
  notion:
    enabled: true
    database_id: "your-notion-database-id"
  rag:
    enabled: true
    vector_store: "chroma"
    embedding_model: "text-embedding-3-small"
```

### 足軽（実行部隊）の管理

```python
# 最大4名の足軽を動的選抜
ashigaru_manager = AshigaruManager(max_members=4)
selected_members = ashigaru_manager.select_optimal_team(task_complexity)
```

## プロジェクト構造

```
codings/
├── shogun/                    # メインAIエージェントシステム
│   ├── agents/               # エージェント実装
│   ├── ashigaru/            # 足軽（実行部隊）
│   ├── core/                # コアシステム
│   ├── integrations/        # 外部サービス統合
│   └── providers/           # AIプロバイダー
├── Inventry-Management/      # 在庫管理システム（VBA版）
├── pwa-warehouse-app/       # PWA倉庫管理アプリ
├── custom_components/       # Home Assistant統合
├── Admin-Page/             # 管理者ページ
└── loginform/              # 認証システム
```

## 設定

### API制限設定

```yaml
# config/settings.yaml
groq:
  daily_limit: 1000
  rpm_limit: 30        # リクエスト/分
  tpm_limit: 6000      # トークン/分
  backoff:
    base_delay: 1.0
    max_delay: 60.0
    exponential_base: 2
```

### 足軽選抜設定

```yaml
ashigaru:
  max_members: 4
  selection_criteria:
    - task_complexity
    - current_workload  
    - specialization
  dynamic_rebalancing: true
```

## 開発

### 貢献

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

### テスト実行

```bash
# 単体テスト
cd shogun
python -m pytest tests/

# 統合テスト
python -m pytest tests/test_integration_v7.py
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## サポート

- **Issues**: [GitHub Issues](https://github.com/98kuwa036/codings/issues)
- **Wiki**: [プロジェクトWiki](https://github.com/98kuwa036/codings/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/98kuwa036/codings/discussions)

## 更新履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1.0.0 | 2025-01-29 | 日本語README作成、システム概要整備 |

---

**使用技術**: Claude Opus (20251101), Python, JavaScript, VBA, Google Apps Script, Home Assistant

**Generated with [Claude Code](https://claude.ai/code)**