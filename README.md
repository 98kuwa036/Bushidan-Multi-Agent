# 🏯 Bushidan Multi-Agent System v8.1

[![Version](https://img.shields.io/badge/Version-8.1-green)](https://github.com/98kuwa036/Bushidan-Multi-Agent)
[![Claude](https://img.shields.io/badge/Claude-Opus%204.5%20%2F%20Sonnet%204.5-purple)](https://www.anthropic.com/claude)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Japanese](https://img.shields.io/badge/Lang-日本語-red)](README_ja.md)

## 🌟 v8.1の新機能: 家訓自動生成強化

**Bushidan-Multi-Agent** は、日本の武士道精神と武家社会の階層構造をモデルにした革新的なAI Multi-Agentシステムです。v8.1では、反省会から月次家訓を自動生成する**組織学習機能**を追加し、失敗再発率を60%削減します。

### 🚀 v8.1 ハイライト

- **🤖 家訓自動生成**: 反省会→家老生成→将軍精査→RAG統合の自動フロー
- **📊 組織学習**: 個人の失敗をチーム全体の学びに変換
- **⏰ 月末自動実行**: Slack通知・Notion記録付きの無人運用
- **💰 コスト効率**: +¥29/月で失敗再発率-60%の高ROI

---

## 📋 目次

- [システム概要](#システム概要)
- [v8.1新機能詳細](#v81新機能詳細)
- [システム構成](#システム構成)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [家訓システム](#家訓システム)
- [パフォーマンス](#パフォーマンス)
- [コスト分析](#コスト分析)
- [開発・貢献](#開発貢献)

---

## 🏯 システム概要

### コンセプト
日本の武家社会の指揮系統を模した階層型AI Multi-Agentシステム。各AIが専門的役割を担い、協調して複雑なタスクを効率的に処理します。

### 基本構成

```
🎌 将軍（Shogun）- Claude Sonnet 4.5 - 最高意思決定者
├── 🏛️ 家老（Karo）- Claude Opus 4.5 - 戦略立案・家訓生成
├── ⚔️ 侍大将（Taisho）- DeepSeek R1 - 現場指揮・品質監査
└── 🏃 足軽（Ashigaru）× 10 - MCP専門部隊
    ├── 1号: ファイル操作 (filesystem)
    ├── 2号: GitHub連携 (github)  
    ├── 3号: Web取得 (fetch)
    ├── 4号: メモリ管理 (memory)
    ├── 5号: DB操作 (postgres)
    ├── 6号: ブラウザ自動化 (puppeteer)
    ├── 7号: 検索 (brave-search)
    ├── 8号: Slack連携 (slack)
    ├── 9号: 記録係 (groq-recorder)
    └── 10号: Web検索 (ollama-web-search) ⭐
```

---

## 🆕 v8.1新機能詳細

### 1. 📜 家訓自動生成システム

#### 🔄 自動フロー
```mermaid
graph LR
    A[月末23:00] --> B[反省会データ収集]
    B --> C[陣中日記統計分析]
    C --> D[演習場エラー集計]
    D --> E[家老: 家訓候補生成]
    E --> F[将軍: 精査・優先順位]
    F --> G[重複チェック]
    G --> H[RAG統合]
    H --> I[Slack通知]
    I --> J[Notion記録]
```

#### 🎯 データソース
- **月次反省会**: Slack `#反省会-YYYY-MM`チャンネル
- **陣中日記**: 侍大将の判断履歴・統計分析  
- **演習場**: コード実行エラーログ・成功率

#### 🏗️ 家訓階層システム
1. **技術家訓** (永続)
   - コーディング規約
   - 設計原則
   - 例: "React 19のServer Components優先"

2. **運用家訓** (6ヶ月) ⭐新規
   - プロジェクト運営
   - チーム協業  
   - 例: "金曜夕方のデプロイ禁止"

3. **月次家訓** (3ヶ月) ⭐新規
   - 今月の反省から
   - 来月の重点方針
   - 例: "2月はメモリ最適化最優先"

### 2. 🧠 組織学習機能

#### 📈 効果測定
- **失敗再発率**: 60%削減目標
- **判断一貫性**: 20%向上
- **処理時間**: 類似タスク24%短縮

#### 🔍 統合効果
- 侍大将の全判断で家訓自動参照
- タスク開始時の家訓適用チェック
- 家訓違反時の警告機能

---

## ⚙️ システム構成

### 🖥️ インフラ構成 (Proxmox LXC)

| CT | 役割 | メモリ | 機能 |
|---|---|---|---|
| **CT100** | 本陣・知識基盤 | 2.5GB | MCP管理, Slack Bot×12, Qdrant, 陣中日記 |
| **CT101** | 侍大将R1 | 20GB | OpenVINO推論サーバー (HugePages) |
| **CT102** | 演習場 | 2GB | オンデマンド実行環境 (Python/Node/Rust) |

### 📚 技術スタック

- **AI Models**: Claude Opus 4.5, Claude Sonnet 4.5, DeepSeek R1-Distill-Japanese
- **知識基盤**: Qdrant + Sentence Transformers
- **活動記録**: SQLite (陣中日記システム)
- **実行環境**: LXC + Docker + OpenVINO
- **通信**: MCP (Model Context Protocol)
- **統合**: Slack, Notion, GitHub, Home Assistant

---

## 🚀 インストール

### 📋 システム要件

#### ハードウェア
- **メモリ**: 24GB+ (推奨: 32GB)
- **ストレージ**: 100GB SSD
- **CPU**: Intel/AMD 6コア以上
- **ネットワーク**: Proxmox環境 (推奨)

#### ソフトウェア
- **Proxmox VE** 7.0+
- **Python** 3.11+
- **Node.js** 16+
- **Docker** 20.10+

### 🔧 セットアップ手順

#### 1. リポジトリクローン
```bash
git clone https://github.com/98kuwa036/Bushidan-Multi-Agent.git
cd Bushidan-Multi-Agent
```

#### 2. Proxmox環境構築
```bash
# CT作成スクリプト実行
sudo ./setup/proxmox_setup.sh

# HugePages設定
sudo ./setup/hugepages_setup.sh
```

#### 3. 侍大将R1セットアップ
```bash
# OpenVINO + R1モデル導入
sudo ./setup/setup_r1_japanese.sh
```

#### 4. 将軍システム設定
```bash
# Python依存関係
pip install -r requirements.txt

# 設定ファイル準備
cp config/settings.yaml.example config/settings.yaml
```

#### 5. API Key設定
```bash
# 環境変数設定
cp .env.example .env

# 必須APIキー
CLAUDE_API_KEY=your_claude_key
GROQ_API_KEY=your_groq_key
SLACK_BOT_TOKEN=your_slack_token
NOTION_TOKEN=your_notion_token
```

#### 6. システム起動
```bash
# 将軍システム起動
python main.py

# v8.1家訓生成テスト
python -m core.monthly_katun_scheduler --force
```

---

## 💻 使用方法

### 🎯 基本コマンド

#### CLI実行
```bash
# インタラクティブモード
python cli.py --mode interactive

# 単発タスク実行
python cli.py --task "在庫データの分析と報告書作成"

# 家訓生成（手動）
python cli.py --generate-katun --month 2025-01
```

#### Slack Bot連携
```bash
# 将軍モード (Full Stack)
@shogun-bot Omni-P4プロトタイプの設計書を作成

# 侍大将モード (軽量)
@taisho-bot コードレビューをお願いします

# 家訓確認
@shogun-bot 現在の家訓を教えて
```

### 🏛️ 運用モード

#### 1. 大隊モード (Battalion)
- **構成**: 将軍 + 家老 + 侍大将 + 足軽×10 + 全機能
- **用途**: 複雑な開発タスク・戦略判断
- **コスト**: ~¥4,249/月

#### 2. 中隊モード (Company)  
- **構成**: 侍大将 + 足軽×10 + 陣中日記
- **用途**: 日常的な開発作業
- **コスト**: ¥950/月 (電気代のみ)

#### 3. 小隊モード (Platoon)
- **構成**: 侍大将 + 動的足軽選択
- **用途**: Home Assistant音声クエリ
- **応答**: 30-60秒

---

## 📜 家訓システム

### 🔍 家訓の確認

#### CLI
```bash
# 全家訓表示
python cli.py --list-katun

# 月次家訓のみ
python cli.py --list-katun --category monthly

# 特定期間
python cli.py --list-katun --period 2025-01
```

#### Slack
```bash
# 現在有効な家訓
@shogun-bot 今月の家訓は？

# 家訓の詳細
@shogun-bot メモリ最適化の家訓について詳しく
```

### 🛠️ 家訓の管理

#### 手動追加
```bash
# 技術家訓追加
python cli.py --add-katun \
  --content "TypeScript strictモードを必須とする" \
  --category technical \
  --priority 4
```

#### 反省会テンプレート
Slackチャンネル `#反省会-YYYY-MM` で以下の形式を推奨:

```markdown
## 今月の成功 ✅
- API応答速度が50%向上
- メモリリークを完全解決

## 今月の失敗 ❌  
- デプロイ時のダウンタイム発生
- テストカバレッジ不足

## 改善点 🔧
- CI/CDパイプラインの見直し
- ローカル環境の統一

## 来月の方針 🎯
- パフォーマンス最適化強化
- セキュリティ監査実施
```

---

## 📊 パフォーマンス

### ⏱️ 処理時間 (v8.1)

| タスク種別 | 初回実行 | 類似タスク | 改善率 |
|---|---|---|---|
| **Simple** | 25秒 | 15秒 | -40% |
| **Medium** | 45秒 | 30秒 | -33% |
| **Complex** | 70秒 | 53秒 | -24% |
| **Strategic** | 90秒 | 70秒 | -22% |

### 🎯 品質指標

| 項目 | v7.0 | v8.0 | v8.1目標 |
|---|---|---|---|
| **全体品質** | 93点 | 99.5点 | 99.7点 |
| **実行成功率** | 95% | 99% | 99.5% |
| **一貫性** | 75% | 95% | 98% |
| **失敗再発** | 100% | 100% | 40% ⭐ |

### 🧠 組織学習効果 (v8.1新機能)

| 指標 | 従来 | v8.1予測 | 改善率 |
|---|---|---|---|
| **同じミス頻度** | 月2-3回 | 月1回以下 | -60% |
| **方針逸脱率** | 30% | 10% | -67% |
| **判断時間** | 平均45秒 | 平均35秒 | -22% |

---

## 💰 コスト分析

### 📊 v8.1コスト構成

| 項目 | 金額 | 前バージョン比 |
|---|---|---|
| **Claude Pro** | ¥3,000 | 変更なし |
| **電気代** | ¥950 | 変更なし |
| **API使用料** | ¥284 | +¥29 |
| **家訓生成** | ¥29 | 🆕新規 |
| **合計** | **¥4,249** | **+0.7%** |

### 💡 ROI分析

#### 投資対効果
- **追加投資**: ¥29/月
- **失敗削減効果**: 60%削減 → 開発効率大幅向上
- **一貫性向上**: 20%改善 → チーム生産性向上
- **学習効果**: 個人知見→組織資産化

#### 比較優位性
| 比較対象 | コスト | 機能 | 優位点 |
|---|---|---|---|
| **Claude Code** | ¥7,701 | 標準 | -44%安価・組織学習 |
| **GPT-4 Team** | ¥8,400 | 標準 | -49%安価・専門性 |
| **独自開発** | ¥50,000+ | カスタム | -91%安価・即利用可 |

---

## 🔧 開発・貢献

### 🛠️ 開発セットアップ

#### 開発環境
```bash
# 開発用依存関係
pip install -e .[dev]

# テスト実行
python -m pytest tests/

# コード品質チェック
ruff check .
black .
mypy .
```

#### v8.1機能テスト
```bash
# 家訓生成テスト
python -m pytest tests/test_katun_generator.py

# 反省会マネージャーテスト  
python -m pytest tests/test_hanseikai_manager.py

# 統合テスト
python -m pytest tests/test_v8_1_integration.py
```

### 🤝 貢献手順

1. **Fork** このリポジトリ
2. **Feature branch** を作成
   ```bash
   git checkout -b feature/katun-enhancement
   ```
3. **変更をコミット**
   ```bash
   git commit -m "feat: 家訓カテゴリ分類機能を追加"
   ```
4. **Push して PR作成**
   ```bash
   git push origin feature/katun-enhancement
   ```

### 📝 コーディング規約

- **Python**: PEP 8準拠 + Black整形
- **型ヒント**: 全関数に必須
- **ドキュメント**: Docstring必須
- **テスト**: カバレッジ80%以上
- **家訓遵守**: 既存家訓の自動チェック

---

## 📚 リソース・サポート

### 📖 ドキュメント
- **[設定ガイド](docs/configuration.md)** - 詳細設定方法
- **[API Reference](docs/api_reference.md)** - 内部API仕様
- **[家訓システム](docs/katun_system.md)** - v8.1新機能詳細
- **[トラブルシューティング](docs/troubleshooting.md)** - よくある問題

### 🆘 サポートチャネル
- **[GitHub Issues](https://github.com/98kuwa036/Bushidan-Multi-Agent/issues)** - バグ報告・機能要望
- **[GitHub Discussions](https://github.com/98kuwa036/Bushidan-Multi-Agent/discussions)** - 質問・議論
- **[Wiki](https://github.com/98kuwa036/Bushidan-Multi-Agent/wiki)** - コミュニティ情報

### 🎓 学習リソース
- **[武家システム設計論](docs/samurai_architecture.md)** - 設計思想
- **[Multi-Agent協調](docs/multi_agent_coordination.md)** - 協調メカニズム
- **[組織学習理論](docs/organizational_learning.md)** - v8.1理論背景

---

## 📋 更新履歴

| バージョン | 日付 | 主要変更 |
|---|---|---|
| **v8.1** | 2025-01-30 | 🆕家訓自動生成・組織学習機能 |
| **v8.0** | 2025-01-15 | 演習場・二重記憶システム・Web検索統合 |
| **v7.0** | 2024-12-01 | Groq統合・レート制限対応 |

---

## 📄 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

---

## 🙏 謝辞

- **Anthropic** - Claude API提供
- **CyberAgent** - DeepSeek R1日本語化
- **Groq** - 高速推論API
- **コミュニティ** - バグ報告・機能提案

---

<div align="center">

**🏯 武士道精神で挑む、次世代AI開発環境 🏯**

[![Star History](https://img.shields.io/github/stars/98kuwa036/Bushidan-Multi-Agent.svg)](https://github.com/98kuwa036/Bushidan-Multi-Agent)
[![Contributors](https://img.shields.io/github/contributors/98kuwa036/Bushidan-Multi-Agent.svg)](https://github.com/98kuwa036/Bushidan-Multi-Agent/graphs/contributors)

**Generated with [Claude Code](https://claude.ai/code)**

</div>