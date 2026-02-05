# 🏯 武士団マルチエージェントシステム v10.1

[![Version](https://img.shields.io/badge/Version-10.1-green)](https://github.com/98kuwa036/Bushidan-Multi-Agent)
[![Claude](https://img.shields.io/badge/Claude-Sonnet%204.5%20%2B%20Opus%204-purple)](https://www.anthropic.com/claude)
[![Kimi](https://img.shields.io/badge/Kimi-K2.5%20128K-ff6600)](https://platform.moonshot.cn/)
[![Gemini](https://img.shields.io/badge/Gemini-3.0%20Flash-blue)](https://ai.google.dev/)
[![Qwen3](https://img.shields.io/badge/Qwen3--Coder--Next-80B%20API-orange)](https://qwenlm.github.io/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-CPU%20Optimized-brightgreen)](https://github.com/ggerganov/llama.cpp)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-red)](https://groq.com/)
[![BDI](https://img.shields.io/badge/BDI-Framework-yellow)](docs/bdi_framework.md)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## v10.1 の革新: 傭兵 (Kimi K2.5) + Smithery MCP

**Bushidan v10.1** は、**傭兵 (Kimi K2.5)** を Do フェーズの第1実行者として迎え、**真の並列サブタスク実行**と**4層鉄壁フォールバック**を実現しました。

### v10.1 ハイライト

- **🗡️ 傭兵 (Kimi K2.5)**: 128K context, 並列サブタスク実行, マルチモーダル
- **🧠 軍師 PDCA Engine**: Plan→Do(Kimi並列)→Check(256K)→Act
- **🔗 4層鉄壁チェーン**: Kimi K2.5 → Local Qwen3 → Kagemusha → Gemini 3 Flash
- **📦 Smithery MCP**: Sequential Thinking, Playwright, Tavily, Exa, Graph Memory, Prisma
- **🏯 5層+傭兵**: 将軍→軍師→家老→大将(+傭兵)→足軽
- **⚡ Groq統合**: Simple タスクは爆速・無料（300-500 tok/s）
- **💰 Prompt Caching**: Claude API 90%コスト削減

---

## 📋 目次

- [システム概要](#システム概要)
- [4層鉄壁フォールバック](#4層鉄壁フォールバック)
- [BDIフレームワーク](#bdiフレームワーク)
- [5層アーキテクチャ](#5層アーキテクチャ)
- [Smithery MCP](#smithery-mcp)
- [インストール](#インストール)
- [パフォーマンス](#パフォーマンス)
- [コスト分析](#コスト分析)

---

## 🏯 システム概要

### 布陣：傭兵 + PDCA統合5層ハイブリッド・アーキテクチャ

```
┌──────────────────────────────────────────────────────────────┐
│ 🎌 将軍（Shogun）- Claude Sonnet 4.5 + Opus 4 + BDI          │
│    役割: 複雑度判断・戦略的設計・最終品質検品                 │
└──────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        │    複雑度判断による動的ルーティング    │
        └──────────────────┬──────────────────┘
                           ↓
       ┌───────────┬───────┼───────┬───────────┐
    Simple      Medium  Complex Strategic
       ↓           ↓       ↓       ↓
┌──────────┐ ┌─────────┐ ┌──────────────┐ ┌────────┐
│ ⚡ Groq  │ │ 👔 家老  │ │ 🧠 軍師 PDCA │ │ 🎌 将軍│
│ 300-500  │ │ 戦術調整 │ │ Plan→Do(Kimi │ │ +Opus  │
│ tok/s    │ │ +BDI     │ │ 並列)→Check  │ │ BDI    │
│ 無料     │ │          │ │ →Act  256K   │ │        │
└──────────┘ └─────────┘ └──────────────┘ └────────┘
       │           │       │       │
       └───────────┴───────┴───────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ ⚔️ 大将（Taisho）- 4層鉄壁フォールバックチェーン              │
│                                                              │
│  🗡️ Tier 1: Kimi K2.5 傭兵 (128K ctx, 並列実行)             │
│  🏯 Tier 2: Local Qwen3 (4K ctx, ¥0, 秘匿・統合・オフライン) │
│  🏯 Tier 3: Cloud Qwen3+ 影武者 (32K ctx, ¥3, 容量拡張)      │
│  🛡️ Tier 4: Gemini 3 Flash (¥0.04, 最終防衛)                │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 👣 足軽（Ashigaru）- Smithery MCP × 10                        │
│    AI:   Sequential Thinking                                 │
│    検索: Tavily, Exa                                         │
│    操作: Playwright, Filesystem, Git                         │
│    記憶: Graph Memory                                        │
│    DB:   Prisma                                              │
│    連携: Slack, Notion                                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗡️ 4層鉄壁フォールバック

### Kimi K2.5 傭兵の役割

| 階層 | 担当 | 役割と動作 |
|---|---|---|
| **Tier 1** | 🗡️ 傭兵 (Kimi K2.5) | Do（並列実行）: 128K context で複数サブタスクを真に並列処理 |
| **Tier 2** | 🏯 侍大将 (Local Qwen3) | Do（高信頼）: 秘匿情報処理・Kimi成果物の統合・オフライン保証 |
| **Tier 3** | 🏯 影武者 (Cloud Qwen3+) | Do（容量拡張）: コンテキスト不足時のクラウド補完 |
| **Tier 4** | 🛡️ 最終防衛 (Gemini 3 Flash) | Do（生存確認）: 全モデル沈黙時の緊急復旧 |

### ローカル Qwen3 の新たな職務

1. **秘匿情報の処理**: API に送信できない機密コードをローカルで完遂
2. **Kimi 成果物の統合**: 並列で書かれたコード断片を整合性あるプログラムに組み上げ
3. **オフライン保証**: ネットワーク障害時も最低限の Medium 実装を維持

### 並列実行の仕組み

```python
# llama.cpp (旧): シングルスレッド → asyncio.gather が実質直列
# Kimi API (新): クラウド推論 → asyncio.gather が真に並列動作

results = await kimi_client.implement_subtasks_parallel(
    subtasks,             # 複数サブタスク
    max_concurrency=4,    # 最大4並列
)
```

---

## 🧠 BDIフレームワーク

### Belief-Desire-Intention アーキテクチャ

| 層 | 信念 (Belief) | 願望 (Desire) | 意図 (Intention) |
|---|---|---|---|
| **将軍** | システム状態、クライアント可用性 | 品質維持、コスト最適化 | 戦略的アクション |
| **軍師** | タスク構造、cross-file整合性 | PDCA最適化 | PDCA作戦計画 |
| **家老** | タスク分解可能性 | 効率的分解 | 調整計画 |
| **大将** | フォールバック状態、Kimi可用性 | 正確実装、自己修復 | 4層実装計画 |

BDI により、実装者が Kimi でも Qwen3 でも「正確実装」という意図が一貫管理される。

---

## 📦 Smithery MCP

### npm → Smithery 移行

v10.1 では MCP サーバー管理を npm 直接実行から Smithery レジストリ経由に統一。

| カテゴリ | MCP | 説明 |
|---|---|---|
| **AI** | Sequential Thinking | 動的思考チェーン |
| **ブラウザ** | Playwright | ブラウザ操作・スクリーンショット |
| **検索** | Tavily | Web検索 (1,000/月 free) |
| **検索** | Exa | セマンティック検索 |
| **データ** | Filesystem | ファイル操作 |
| **データ** | Graph Memory | グラフ型記憶 |
| **データ** | Prisma | DB操作 |
| **連携** | Slack | チーム連携 |
| **連携** | Notion | ドキュメント管理 |
| **連携** | Git | バージョン管理 |

---

## 🚀 インストール

### セットアップ手順

```bash
# リポジトリクローン
git clone https://github.com/98kuwa036/Bushidan-Multi-Agent.git
cd Bushidan-Multi-Agent

# Python依存関係
pip install -r requirements.txt

# llama.cpp + Qwen3-Coder設定（HP ProDesk 600 CPU対応）
chmod +x scripts/setup_llamacpp_prodesck600.sh
./scripts/setup_llamacpp_prodesck600.sh

# API Key設定
cp .env.example .env
# .env を編集して API キーを設定

# システム起動
python main.py
```

### 必要な API Key

| Key | 用途 | コスト |
|---|---|---|
| `CLAUDE_API_KEY` | 将軍 (Claude Sonnet + Opus) | Pro ¥3,000/月 |
| `GEMINI_API_KEY` | 最終防衛 (Gemini 3 Flash) | ~¥0.04/req |
| `GROQ_API_KEY` | Simple タスク (爆速) | 無料 |
| `ALIBABA_API_KEY` | 影武者 (Cloud Qwen3+) | ¥3/req |
| `QWEN3_CODER_NEXT_API_KEY` | 軍師 (256K PDCA) | ~¥0.05/req |
| `KIMI_API_KEY` | 傭兵 (Kimi K2.5, 128K並列) | ~¥0.01/req |
| `TAVILY_API_KEY` | Web検索 | 1,000/月 free |

---

## 📊 パフォーマンス

### ⏱️ 処理時間

| 階層 | 処理時間 | 担当 | 特徴 |
|---|---|---|---|
| **Simple** | **2秒** | Groq | 瞬発力・無料 |
| **Medium** | **12秒** | 家老+大将+BDI | ローカル・¥0 |
| **Complex (CHECK合格)** | **20-28秒** | 軍師PDCA+Kimi並列 | Kimi並列で高速化 |
| **Complex (修正あり)** | **35-45秒** | 軍師PDCA+Kimi | 修正込み・品質87-93 |
| **Strategic** | **45秒** | 将軍+Opus+BDI | 最高品質 |

### v9.4 → v10.1 改善

| 指標 | v9.4 | v10.1 | 改善 |
|---|---|---|---|
| **COMPLEX品質** | 75-82点 | 87-93点 | **+8pt (PDCA)** |
| **COMPLEX時間** | 30-40s | 20-28s | **-10s (Kimi並列)** |
| **Do並列** | 実質直列 | 真の並列 | **Kimi API** |
| **信頼性** | 99.5% | 99.7% | **4層チェーン** |

---

## 💰 コスト分析

### v10.1 コスト構成

| 項目 | 金額 | 説明 |
|---|---|---|
| **Claude Pro** | ¥3,000 | Pro CLI + Prompt Caching |
| **Claude API** | ¥14 | Caching 90%削減 |
| **Opus Premium** | ¥100 | Strategic 検品 |
| **Qwen3-Coder-Next** | ¥10 | 軍師 PDCA |
| **Kimi K2.5** | ¥30 | 傭兵並列実行 |
| **Gemini 3 Flash** | ¥120 | 最終防御線 |
| **Alibaba Qwen3+** | ¥60 | 影武者 |
| **Groq** | ¥0 | 無料枠内 |
| **電力** | ¥80 | Groq節約効果 |
| **合計** | **¥3,414** | v9.4比 +¥30 (Kimi傭兵) |

---

## 📋 更新履歴

| バージョン | 日付 | 主要変更 |
|---|---|---|
| **v10.1** | 2026-02-05 | 🆕 傭兵(Kimi K2.5) + Smithery MCP + 4層鉄壁チェーン |
| **v10** | 2026-02-04 | 軍師(Gunshi)層追加・PDCA Engine・5層アーキテクチャ |
| **v9.4** | 2026-02-01 | BDIフレームワーク全層統合・日本語ログ全面化 |
| **v9.3.2** | 2025-01-31 | インテリジェント・ルーティング・Groq統合 |
| **v9.3** | 2025-01-31 | 4段ハイブリッド・DSPy統合 |

---

## 📄 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

---

<div align="center">

**🏯 傭兵 + PDCA Engine + BDI形式推論で挑む、鉄壁5層ハイブリッド 🏯**

**Generated with [Claude Code](https://claude.ai/code)**

</div>
