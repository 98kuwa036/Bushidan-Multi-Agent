# 🏯 武士団マルチエージェントシステム v11.5

[![Version](https://img.shields.io/badge/Version-11.5-brightgreen)](https://github.com/98kuwa036/Bushidan-Multi-Agent)
[![Claude](https://img.shields.io/badge/Claude-Opus%204.5%20%2B%20Sonnet%204.6-purple)](https://www.anthropic.com/claude)
[![OpenAI](https://img.shields.io/badge/OpenAI-o3--mini%20%2B%20GPT--5-412991)](https://openai.com/)
[![xAI](https://img.shields.io/badge/xAI-Grok--code--fast--1-1DA1F2)](https://x.ai/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-blue)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-red)](https://groq.com/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron--3--Nano%20Local-76B900)](https://www.nvidia.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-MCP%20Integrated-orange)](https://langchain-ai.github.io/langgraph/)
[![Notion](https://img.shields.io/badge/Notion-Active%20Knowledge-black)](https://notion.so/)
[![Mattermost](https://img.shields.io/badge/Mattermost-Bot%20%2B%20MCP-0058CC)](https://mattermost.com/)
[![BDI](https://img.shields.io/badge/BDI-Framework-yellow)](docs/bdi_framework.md)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## v11.5 の革新: LangGraph × MCP × Notion 密結合

**Bushidan v11.5** は v11.4 の9層アーキテクチャを土台に、**LangGraph SDK** による **MCP ツール認識ルーティング** と **Notion 積極活用** を実装しました。

### v11.5 ハイライト

- **🔗 LangGraph MCP 密結合**: `fetch_context` ノードで MCP ツール一覧を取得し、ルーティング判断に活用
- **📖 Notion 積極活用**: 家訓・直近タスクをルーター判断に注入、全タスクを自動永続化
- **🧠 軍師 PDCA ルート**: COMPLEX/STRATEGIC タスクを o3-mini → 参謀A/B に自動委譲
- **🔧 Taisho MCP チェーン**: GitHub/tavily/playwright ツール存在時に最適な MCP チェーンを実行
- **💬 Mattermost 統合**: Bot (@メンション応答) + MCP サーバー (エージェントが Mattermost を操作)
- **🤖 進捗報告**: エージェントが処理中タスクの進捗を Mattermost にリアルタイム投稿

### v11.4 から継承

- **🚫 脱中国企業**: Qwen3・Kimi K2.5 完全排除、西側連合モデルのみ
- **👑 大元帥 (Claude Opus 4.5)**: SWE-Bench 80.9%、最高難度・戦略設計
- **🎌 将軍 (Claude Sonnet 4.6)**: SWE-Bench 72.7%、高難度コーディング専任
- **🧠 軍師 (o3-mini high)**: 推論特化 PDCA エンジン
- **⚡ 参謀-B (Grok-code-fast-1)**: 実装特化・240tok/s 超高速
- **🔧 家老-B (Llama 3.3 70B)**: HumanEval 88.4%、Groq 無料
- **🥷 隠密 (Nemotron-3-Nano)**: NVIDIA製ローカルLLM、機密処理・オフライン対応

---

## 📋 目次

- [システム概要](#システム概要)
- [v11.5 LangGraph MCP 密結合](#v115-langgraph-mcp-密結合)
- [Notion 積極活用](#notion-積極活用)
- [Mattermost 統合](#mattermost-統合)
- [9層アーキテクチャ詳細](#9層アーキテクチャ詳細)
- [脱中国企業の理由](#脱中国企業の理由)
- [軍師 PDCA エンジン](#軍師-pdca-エンジン)
- [BDIフレームワーク](#bdiフレームワーク)
- [ハードウェア構成](#ハードウェア構成)
- [インストール](#インストール)
- [パフォーマンス](#パフォーマンス)
- [コスト分析](#コスト分析)
- [更新履歴](#更新履歴)

---

---

## 🔗 v11.5 LangGraph MCP 密結合

### StateGraph v11.5 — ツール認識 + Notion コンテキスト注入

```
[START]
  ↓
[analyze]         タスク複雑度・マルチステップ・アクション種別を検出
  ↓
[fetch_context]   並列取得:
                    ├─ MCPManager.list_tools()  → 実行中ツール一覧
                    └─ Notion.get_routing_context() → 家訓 + 直近タスク
  ↓
[route_decision]  ツール認識 + Notion 知識参照 + 複雑度ベースの5経路分岐
  ↓
  ├─ [groq_qa]           Simple Q&A  → 家老-B Groq (即応・無料)
  ├─ [gunshi_pdca]       Complex     → 軍師 o3-mini PDCA → 参謀A/B
  ├─ [gemini_autonomous] Multi-step  → Gemini Flash 自律実行
  ├─ [taisho_mcp]        Tool chain  → Taisho + MCP ツール連携
  └─ [karo_default]      Default     → 家老 既存ルーティング
  ↓
[persist_notion]  fire-and-forget: 全タスク結果を Notion へ自動保存
  ↓
[END]
```

### MCP ツール認識ルーティング

```python
# fetch_context ノードが取得する情報
available_tools = ["read_file", "write_file", "github", "search", ...]
notion_context  = "【家訓】...\n【直近タスク】..."

# route_decision がツール存在を見て経路を決定
if "github" in tools and "git" in content:
    → taisho_mcp  (GitHub MCP ツールを活用)

if "tavily" in tools and "search" in content:
    → taisho_mcp  (Tavily 検索ツールを活用)

if complexity in ("complex", "strategic"):
    → gunshi_pdca (軍師 o3-mini PDCA へ委譲)
```

### MCPManager v11.5 — TOOL_REGISTRY

```python
# 実行中サーバーのツール一覧を返す (LangGraph ルーター用)
manager.list_tools()
# → {"github": ["create_issue", "push_files", ...],
#    "tavily": ["search", "search_context"],
#    "mattermost": ["post_message", "submit_task", ...]}

manager.get_server_for_tool("search")  # → "tavily"
manager.is_tool_available("github")   # → True/False
```

---

## 📖 Notion 積極活用

### v11.5 での役割変化

| v11.4 (受動的) | v11.5 (能動的) |
|---|---|
| 手動で `save_knowledge_entry()` を呼ぶ | 全タスク完了後に **自動保存** |
| タイトルのみ返す search_knowledge() | **本文全文**を取得できる `get_page_content()` |
| ページ作成のみ | **既存ページの更新**にも対応 |
| ルーターは Notion を参照しない | **家訓・直近タスクをルーター判断に注入** |

### v11.5 新メソッド

```python
# LangGraph fetch_context ノードから呼ばれる
context = await notion.get_routing_context()
# → "【家訓】コストより品質を優先する...\n【直近タスク】- [軍師] git push..."

# LangGraph persist_notion ノードから呼ばれる (fire-and-forget)
page_id = await notion.auto_save_task_result(
    task="Dockerfile を書いて",
    result="FROM python:3.12...",
    metadata={"route": "gunshi_pdca", "agent_role": "軍師", "execution_time": 18.3}
)

# 家訓ページを本文まで取得
content = await notion.get_page_content(page_id)  # ブロック全文

# 実行中タスクのステータスを更新
await notion.update_entry(page_id, status="進行中")
```

### Notion に自動保存される情報

| フィールド | 内容 |
|---|---|
| **Title** | `[エージェント役職] タスク先頭60文字` |
| **Type** | `タスク完了` |
| **Agent** | `軍師` / `家老-B` / `参謀` など |
| **Tags** | 使用した MCP ツール名 |
| **Body** | タスク全文 + 実行結果 + ルート・実行時間 |

---

## 💬 Mattermost 統合

### 2コンポーネント構成

```
┌─────────────────────────────────────────────┐
│  bushidan/mattermost_bot.py (Bot)           │
│  ┌─────────────────────────────────────────┐│
│  │ WebSocket @メンション受信               ││
│  │ → SystemOrchestrator → 9層処理         ││
│  │ → スレッドに返答 (受領通知→結果上書き)  ││
│  └─────────────────────────────────────────┘│
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  mcp/mattermost_mcp_server.py (MCP Server) │
│  Bushidan AI エージェントが呼び出すツール群  │
│  ┌─────────────────────────────────────────┐│
│  │ Mattermost 操作 (7ツール)               ││
│  │   post_message / search_messages        ││
│  │   add_reaction / create_channel ...     ││
│  │                                         ││
│  │ 武士団連携 (3ツール)                    ││
│  │   submit_task → 9層システムへ委譲       ││
│  │   get_bushidan_status → 全層状態確認    ││
│  │   report_agent_progress → 進捗報告      ││
│  └─────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

### Bot コマンド

| コマンド | 説明 |
|---|---|
| `@Bushidan <タスク>` | 9層システムにタスク投入 |
| `@Bushidan !mode` | 現在のモード表示 |
| `@Bushidan !mode battalion` | 大隊モード（全9層）に切替 |
| `@Bushidan !status` | 全9層のオンライン状態表示 |
| `@Bushidan !help` | ヘルプ表示 |

### エージェント進捗報告（ASCII プログレスバー）

```
🧠 [軍師] task_abc123
[████████████░░░░░░░░] 60%
Plan フェーズ完了。参謀-A/B への Do フェーズを開始します。
```

### 環境変数

```bash
MATTERMOST_URL=chat.example.com
MATTERMOST_TOKEN=<bot-access-token>
MATTERMOST_PORT=443         # 省略可
MATTERMOST_SCHEME=https     # 省略可
```

### 起動

```bash
# Bot 起動
python -m bushidan.mattermost_bot

# MCP サーバー起動 (stdio)
python -m mcp.mattermost_mcp_server
```

---

## 🏯 システム概要

### 布陣：9層ハイブリッドアーキテクチャ（西側連合）

```
┌──────────────────────────────────────────────────────────────┐
│ 👑 大元帥（Daigensui）- Claude Opus 4.5                       │
│    SWE-Bench 80.9% | 最高難度・戦略設計・最終品質保証          │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 🎌 将軍（Shogun）- Claude Sonnet 4.6                          │
│    SWE-Bench 72.7% | 高難度コーディング・設計実装              │
└──────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        │    複雑度判断による動的ルーティング    │
        └──────────────────┬──────────────────┘
                           ↓
   ┌──────────┬────────────┼────────────┬──────────┐
Simple     Medium      Complex     Strategic  Coding
   ↓           ↓           ↓           ↓          ↓
┌──────┐  ┌───────┐  ┌──────────┐  ┌────────┐  ┌───────┐
│ 🔧   │  │ ⚡    │  │ 🧠 軍師  │  │ 👑 大  │  │ 🎌 将 │
│家老-B│  │参謀-B │  │ o3-mini  │  │ 元帥   │  │  軍   │
│Groq  │  │Grok  │  │ PDCA     │  │ Opus   │  │Sonnet │
│無料  │  │240t/s│  │ Engine   │  │ 4.5    │  │  4.6  │
└──────┘  └───────┘  └──────────┘  └────────┘  └───────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 📋 参謀-A (GPT-5) + ⚡ 参謀-B (Grok-code-fast-1)             │
│    PDCA Do フェーズ: GPT-5汎用 + Grok高速 並列実装             │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────┬────────────────────────────────────┐
│ 🏮 家老-A (Gemini Flash) │ 🔧 家老-B (Llama 3.3 70B / Groq)  │
│    軽量タスク・最終防衛   │    Simple タスク・無料・300-500t/s  │
└─────────────────────────┴────────────────────────────────────┘
                           ↓
┌─────────────────────────┬────────────────────────────────────┐
│ 👁️ 検校 (Gemini Vision)  │ 🥷 隠密 (Nemotron-3-Nano Local)    │
│    ビジュアル品質検証     │    機密処理・オフライン・¥0         │
└─────────────────────────┴────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ 👣 足軽（Ashigaru）- Smithery MCP × 10                        │
│    AI: Sequential Thinking | 検索: Tavily, Exa               │
│    操作: Playwright, Filesystem, Git | 記憶: Graph Memory     │
│    DB: Prisma | 連携: Mattermost MCP, Notion, Discord        │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎖️ 9層アーキテクチャ詳細

| 層 | 役職 | モデル | 特徴 | コスト/req |
|---|---|---|---|---|
| **1** | 👑 大元帥 | Claude Opus 4.5 | SWE-Bench 80.9%・最高品質 | ¥10-50 |
| **2** | 🎌 将軍 | Claude Sonnet 4.6 | SWE-Bench 72.7%・高難度コーディング | ¥0-5 |
| **3** | 🧠 軍師 | o3-mini (high) | 推論特化・PDCA エンジン | ¥0.05-0.20 |
| **4a** | 📋 参謀-A | GPT-5 | 汎用コーディング・最新 OpenAI | ¥0.01-0.10 |
| **4b** | ⚡ 参謀-B | Grok-code-fast-1 | 実装特化・240tok/s | ¥0.01-0.05 |
| **5a** | 🏮 家老-A | Gemini 2.5 Flash | 軽量・超低コスト | ¥0.01 |
| **5b** | 🔧 家老-B | Llama 3.3 70B (Groq) | HumanEval 88.4%・無料 | ¥0 |
| **6** | 👁️ 検校 | Gemini Flash Vision | マルチモーダル・ビジュアル検証 | ¥0.01 |
| **7** | 🥷 隠密 | Nemotron-3-Nano (Local) | 機密・オフライン・ローカル | ¥0 |

### 設計原則：役割分離

> **重要**: 軍師（o3-mini）と参謀-B（Grok）は **役割分離** であり、置き換えではない。
>
> - **軍師 o3-mini**: 推論・計画・検証（PDCA の Plan/Check/Act）
> - **参謀-B Grok**: 実装・コード生成（PDCA の Do フェーズ）
>
> 両者が協調することで「考える頭脳」と「動く手足」が分離され、品質と速度を両立する。

---

## 🚫 脱中国企業の理由

v11.4 では以下の中国企業サービスを **完全排除** しました。

| 除外サービス | 理由 |
|---|---|
| Kimi K2.5 (Moonshot AI) | 中国企業・データ主権リスク |
| Qwen3-Coder-Next (Alibaba) | 中国企業・API信頼性 |
| DashScope / Alibaba Cloud | 中国企業・規制リスク |

**代替（西側連合）:**
- OpenAI (o3-mini, GPT-5) - 米国
- xAI (Grok-code-fast-1) - 米国
- NVIDIA (Nemotron-3-Nano) - 米国（ローカル）
- Google (Gemini Flash) - 米国
- Anthropic (Claude Opus 4.5, Sonnet 4.6) - 米国
- Meta (Llama 3.3 70B via Groq) - 米国

---

## 🧠 軍師 PDCA エンジン

### o3-mini (high) による高精度 PDCA サイクル

```
┌─────────────────────────────────┐
│ Plan (o3-mini high)              │ ← タスク分析・サブタスク分解（最大5）
└─────────────────┬───────────────┘
                  ↓
┌─────────────────────────────────┐
│ Do（参謀A/B 並列実装）           │ ← GPT-5 (汎用) + Grok (高速) 協調
│  独立サブタスク → 参謀-B 並列    │
│  依存サブタスク → 参謀-A 逐次    │
└─────────────────┬───────────────┘
                  ↓
┌─────────────────────────────────┐
│ Check (o3-mini high + 検校)      │ ← テキスト検証(80%) + ビジュアル(20%)
│  軍師: 要件充足・整合性・品質     │
│  検校: UI/UXレイアウト・表示品質  │
└─────────────────┬───────────────┘
                  ↓
┌─────────────────────────────────┐
│ Act (o3-mini high)               │ ← 不合格サブタスクに修正指示
│  最大1回修正ループ               │
└─────────────────────────────────┘
```

### PDCA フォールバックチェーン

| 複雑度 | Primary | Fallback |
|---|---|---|
| **Simple** | 家老-B (Groq Llama 3.3) | 家老-A (Gemini Flash) |
| **Medium** | 参謀-B (Grok-code-fast-1) | 参謀-A (GPT-5) → 家老-A |
| **Complex** | 軍師 o3-mini PDCA | 参謀-A GPT-5 → 参謀-B Grok → 家老-A |
| **Strategic** | 大元帥 Claude Opus 4.5 | 将軍 Claude Sonnet 4.6 |

---

## 👁️ 検校 ビジュアルデバッガー

PDCA Check フェーズで **Gemini Flash Vision + Playwright MCP** による UI ビジュアル検証を実施。

```
PDCA Check フェーズ:

  ┌─────────────────────────────────┐
  │ 🔍 テキスト検証 (軍師 o3-mini)  │  ← 要件充足・整合性・コード品質
  │    重み: 80%                     │
  └─────────────┬───────────────────┘
                │
  ┌─────────────▼───────────────────┐
  │ 👁️ ビジュアル検証 (検校)        │  ← UI/UX レイアウト・表示品質
  │    重み: 20%                     │
  │  Playwright → Screenshot         │
  │  Gemini Vision → Analysis        │
  │  Desktop + Mobile 自動チェック   │
  └─────────────────────────────────┘
```

| カテゴリ | チェック内容 |
|---|---|
| **レイアウト** | 要素の重なり、はみ出し、不正な空白 |
| **テキスト** | フォントサイズ、コントラスト比、可読性 |
| **操作性** | ボタン・リンクの視認性と操作可能性 |
| **レスポンシブ** | Desktop (1280px) / Tablet (768px) / Mobile (375px) |
| **リグレッション** | Before/After 比較による意図しない変更検出 |

---

## 🧠 BDIフレームワーク

### Belief-Desire-Intention アーキテクチャ

| 層 | 役職 | 信念 (Belief) | 願望 (Desire) | 意図 (Intention) |
|---|---|---|---|---|
| **1** | 大元帥 | システム全体状態・リスク | 最高品質・セキュリティ保証 | 戦略的判断・最終承認 |
| **2** | 将軍 | コード品質・クライアント状態 | 高品質コード・効率実装 | 実装計画・コーディング |
| **3** | 軍師 | タスク構造・cross-file整合性 | PDCA最適化・推論精度 | PDCA作戦計画 |
| **4a/b** | 参謀A/B | 実装要件・サブタスク状態 | 正確実装・高速完了 | 実装計画 |
| **5a/b** | 家老A/B | タスク分解・軽量処理 | 効率的処理・コスト削減 | 調整・実行計画 |
| **6** | 検校 | UI状態・ビューポート情報 | ビジュアル品質保証 | スクリーンショット解析 |
| **7** | 隠密 | 機密データ・ローカル状態 | 安全処理・機密保持 | ローカル完結計画 |

---

## 🖥️ ハードウェア構成

### HP ProDesk 600 G4（2台構成）

| 機器 | 役割 | スペック |
|---|---|---|
| **EliteDesk (本陣)** | システム統括・オーケストレーション | i5-8500, 16GB DDR4 |
| **ProDesk 600 (隠密)** | ローカルLLM専用機 | i5-8500, **32GB DDR4** |

### ProDesk 600 RAM 構成（v11.4 確定）
- **既存**: 16GB (8GB × 2)
- **増設**: 8GB × 2 **購入済み**
- **合計**: **32GB DDR4**（Nemotron-3-Nano Q4_K_M ~21GB に十分）

### Nemotron-3-Nano 運用仕様

| 項目 | 内容 |
|---|---|
| **モデル** | Nemotron-3-Nano-30B-A3B |
| **量子化** | Q4_K_M (~21GB) |
| **バックエンド** | llama.cpp (CPU最適化) |
| **推論速度** | 15-25 tok/s (CPU) |
| **コンテキスト** | 8192 tokens |
| **エンドポイント** | http://192.168.11.232:8080 |
| **特記** | NVIDIA製・脱中国・機密処理対応 |

---

## 📦 Smithery MCP

| カテゴリ | MCP | 説明 |
|---|---|---|
| **AI** | Sequential Thinking | 動的思考チェーン・軍師専属 |
| **ブラウザ** | Playwright | ブラウザ操作・スクリーンショット・検校専属 |
| **検索** | Tavily | Web検索 (1,000/月 free) |
| **検索** | Exa | セマンティック検索 |
| **データ** | Filesystem | ファイル操作 |
| **データ** | Graph Memory | グラフ型記憶 |
| **データ** | Prisma | DB操作 |
| **連携** | Mattermost MCP | チャット連携・エージェント進捗報告 (v11.5) |
| **連携** | Discord | チーム連携 |
| **連携** | Notion | 知識ベース・自動タスク永続化 (v11.5 積極活用) |
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

# Nemotron-3-Nano セットアップ（HP ProDesk 600）
chmod +x setup/setup_nemotron.sh
./setup/setup_nemotron.sh

# API Key設定
cp .env.example .env
# .env を編集して API キーを設定

# システム起動
python main.py
```

### 必要な API Key

| Key | 役職 | コスト |
|---|---|---|
| `ANTHROPIC_API_KEY` | 大元帥 (Opus 4.5) + 将軍 (Sonnet 4.6) | Pro ¥3,000/月 |
| `OPENAI_API_KEY` | 軍師 (o3-mini) + 参謀-A (GPT-5) | ~¥0.05-0.20/req |
| `XAI_API_KEY` | 参謀-B (Grok-code-fast-1) | ~¥0.01-0.05/req |
| `GEMINI_API_KEY` | 家老-A + 検校 (Gemini Flash Vision) | ~¥0.01/req |
| `GROQ_API_KEY` | 家老-B (Llama 3.3 70B) | **無料** |
| `TAVILY_API_KEY` | 足軽・Web検索 | 1,000/月 free |
| `MATTERMOST_URL` + `MATTERMOST_TOKEN` | Mattermost Bot + MCP (v11.5) | 無料 |
| `DISCORD_TOKEN` | Discord Bot (オプション) | 無料 |
| `NOTION_TOKEN` + `NOTION_DB_ID` | Notion 知識ベース (v11.5 積極活用) | 無料 |

### ローカルLLM（隠密・Nemotron-3-Nano）

```bash
# HP ProDesk 600 で実行
./setup/setup_nemotron.sh

# llama.cpp サーバー起動（自動起動設定）
./llama.cpp/build/bin/llama-server \
  --model models/nemotron/Nemotron-3-Nano-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --threads 6 --ctx-size 8192
```

---

## 📊 パフォーマンス

### ⏱️ 処理時間目標

| 複雑度 | 処理時間 | ルーティング | コスト |
|---|---|---|---|
| **Simple** | **2秒** | 家老-B Groq (Llama 3.3) | ¥0 |
| **Medium** | **8-12秒** | 参謀-B Grok-code-fast-1 | ¥0.01-0.05 |
| **Complex (合格)** | **20-30秒** | 軍師 PDCA + 参謀並列 | ¥0.10-0.30 |
| **Complex (修正)** | **35-50秒** | 軍師 PDCA + Act ループ | ¥0.20-0.50 |
| **Strategic** | **45秒** | 大元帥 Opus 4.5 | ¥10-50 |
| **Coding** | **15-30秒** | 将軍 Sonnet 4.6 | ¥0-5 |
| **Confidential** | **15-25秒** | 隠密 Nemotron Local | ¥0 |

### v10.1 → v11.4 改善

| 指標 | v10.1 | v11.4 | 改善 |
|---|---|---|---|
| **参謀速度** | Kimi K2.5 | Grok 240tok/s | **高速化** |
| **信頼性** | 中国企業依存 | 西側連合 | **リスク排除** |
| **ローカルLLM** | Qwen3 (中国) | Nemotron-3-Nano (NVIDIA) | **脱中国** |
| **推論エンジン** | Qwen3-Coder-Next | o3-mini (high) | **推論精度向上** |

---

## 💰 コスト分析

### v11.4 月次コスト見積り

| 項目 | 金額 | 説明 |
|---|---|---|
| **Claude Pro** | ¥3,000 | Pro CLI + Prompt Caching |
| **Claude API** | ¥140 | Prompt Caching 90%削減 |
| **Opus Premium** | ¥200 | Strategic 大元帥 |
| **OpenAI (o3-mini + GPT-5)** | ¥200 | 軍師 PDCA + 参謀-A |
| **xAI Grok** | ¥50 | 参謀-B 高速実装 |
| **Gemini Flash** | ¥80 | 家老-A + 検校 |
| **Groq** | ¥0 | 家老-B 無料枠内 |
| **電力** | ¥80 | ローカルLLM電気代 |
| **合計** | **¥3,750** | v10.1 比 +¥300（Kimi→Grok+o3-mini） |

---

## 🔐 セキュリティ

| 機能 | 説明 |
|---|---|
| **機密処理分離** | 秘匿データは隠密（ローカルLLM）で処理、API送信なし |
| **西側連合限定** | 中国企業 API への一切の送信を排除 |
| **MCP権限マトリクス** | 役職別 MCP アクセス制御（exclusive / primary / secondary / forbidden） |
| **セキュリティKW監視** | auth, payment, database, credential 等の検知で大元帥へエスカレーション |
| **Prompt Caching** | Claude API キャッシング（コスト削減 + セッション安全性） |

---

## 📋 更新履歴

| バージョン | 日付 | 主要変更 |
|---|---|---|
| **v11.5** | 2026-03-03 | 🆕 LangGraph × MCP × Notion 密結合・Mattermost Bot + MCP サーバー |
| **v11.4** | 2026-03-03 | 🆕 脱中国企業 + 9層アーキテクチャ（大元帥・参謀A/B・家老A/B・隠密） |
| **v10.1** | 2026-02-05 | 傭兵(Kimi K2.5) + Smithery MCP + 4層鉄壁チェーン |
| **v10.0** | 2026-02-04 | 軍師(Gunshi)層追加・PDCA Engine・5層アーキテクチャ |
| **v9.4** | 2026-02-01 | BDIフレームワーク全層統合・日本語ログ全面化 |
| **v9.3** | 2025-01-31 | インテリジェント・ルーティング・Groq統合 |

---

## 📄 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

---

<div align="center">

**🏯 西側連合 9層ハイブリッド × LangGraph MCP × Notion 積極活用 × o3-mini PDCA 🏯**

**Generated with [Claude Code](https://claude.ai/code)**

</div>
