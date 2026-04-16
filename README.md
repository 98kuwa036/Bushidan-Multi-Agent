# 🏯 武士団マルチエージェントシステム v18

[![Version](https://img.shields.io/badge/Version-18.0-brightgreen)](https://github.com/98kuwa036/Bushidan-Multi-Agent)
[![Claude](https://img.shields.io/badge/Claude-Opus%204.6%20%2B%20Sonnet%204.6-purple)](https://www.anthropic.com/claude)
[![Gemini](https://img.shields.io/badge/Gemini-3.1%20Flash%20%2F%20Flash--Lite-blue)](https://ai.google.dev/)
[![Cerebras](https://img.shields.io/badge/Cerebras-Gemma2%209B-teal)](https://cloud.cerebras.ai/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B%20%2F%203B-red)](https://groq.com/)
[![Cohere](https://img.shields.io/badge/Cohere-Command%20A%20%2F%20R-coral)](https://cohere.com/)
[![Mistral](https://img.shields.io/badge/Mistral-Small-orange)](https://mistral.ai/)
[![Local LLM](https://img.shields.io/badge/Local-Gemma4%20MoE%20%2B%20Nemotron-76B900)](https://ai.google.dev/gemma)
[![LangGraph](https://img.shields.io/badge/LangGraph-PostgreSQL%20Checkpoint-orange)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Console%208067-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

> 戦国時代の軍事組織をモデルにした階層型マルチエージェントAIシステム。  
> 11の専門役職と複数のLLMが協調し、複雑なタスクを自律的に処理する。

---

## 🎖️ 役職体制（11役職）

| 役職 | キー | モデル | 担当領域 |
|------|------|--------|---------|
| **受付** | uketuke | Gemini 3.1 Flash-Lite | 意図分類・ルーティング判断 |
| **外事** | gaiji | Cohere Command R | 外部情報取得・RAG |
| **斥候** | seppou | Llama 3.3 70B (Groq) | Web 検索・情報収集 |
| **軍師** | gunshi | Cohere Command A | 戦略立案・複合タスク分解 |
| **参謀** | sanbo | Gemini Flash Preview | コード実行支援・HITL承認制御 |
| **将軍** | shogun | Claude Sonnet 4.6 | 汎用高精度処理・ロードマップ生成 |
| **大元帥** | daigensui | Claude Opus 4.6 | 最高難度タスク・監査処理 |
| **目付** | metsuke | Mistral Small | 中難度処理・品質検査 |
| **検校** | kengyo | Gemini 3.1 Flash Image | 画像解析・マルチモーダル |
| **右筆** | yuhitsu | Gemma4 MoE (Local) → Gemini Flash-Lite (fallback) | 日本語清書・文章生成 |
| **隠密** | onmitsu | Nemotron (Local) → Gemma4 (Local) | 機密処理（クラウド不可） |

> `claude_fallback` — Anthropic API への最終フォールバック（全役職共通）

---

## 🏗️ インフラ構成

```
┌─────────────────────────────────────────────────────────────────┐
│                       ホームネットワーク (Proxmox VE)             │
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │  pct237     │   │  pct100     │   │  pct239             │   │
│  │ .237        │   │ .231        │   │  .239               │   │
│  │             │   │             │   │                     │   │
│  │ Claude Code │──▶│ Bushidan    │◀──│  Gemma4 MoE 27B     │   │
│  │ 作業環境    │   │ コンソール  │   │  Nemotron 3         │   │
│  │ :8070       │   │ :8067       │   │  :8082              │   │
│  └─────────────┘   └──────┬──────┘   └─────────────────────┘   │
│                            │                                    │
│              ┌─────────────┼─────────────┐                     │
│              ▼             ▼             ▼                     │
│       ┌────────────┐              ┌──────────────┐            │
│       │  pct105    │              │  Cloud APIs  │            │
│       │ PostgreSQL │              │  Anthropic   │            │
│       │    :5432   │              │  Google      │            │
│       │            │              │  Groq        │            │
│       └────────────┘              │  Cohere      │            │
│                                   │  Mistral     │            │
│                                   │  Cerebras    │            │
│                                   └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

| コンテナ | IP | 役割 |
|---------|-----|------|
| pct100 | 192.168.11.231 | Bushidan アプリ本体・Web コンソール |
| pct105 | 192.168.11.236 | PostgreSQL 17（LangGraph チェックポイント） |
| pct237 | 192.168.11.237 | Claude API Server・開発環境 |
| pct239 | 192.168.11.239 | ローカル LLM サーバー |

---

## 🔄 LangGraph ルーティングアーキテクチャ

```
                  ユーザー入力
                       │
              ┌────────▼────────┐
              │  受付 (uketuke) │  Gemini Flash-Lite
              │  意図分類・判断  │  SemanticRouter 事前チェック
              └────────┬────────┘
                       │  route 決定
        ┌──────────────┼──────────────────────┐
        │              │              │       │
   ┌────▼───┐    ┌─────▼──┐    ┌─────▼──┐  ┌─▼────────┐
   │ 外事   │    │  軍師  │    │  将軍  │  │ 大元帥   │
   │ RAG   │    │  戦略  │    │ Sonnet │  │  Opus    │
   └────┬───┘    └─────┬──┘    └─────┬──┘  └─┬────────┘
        │              │              │       │
        └──────────────▼──────────────┘───────┘
                       │
              ┌────────▼────────┐
              │ human_interrupt │  HITL ノード
              │ (破壊的操作検出) │  Web UI で承認/却下
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   PostgreSQL    │
              │  LangGraph      │  スレッド別会話永続化
              │   Saver         │
              └─────────────────┘
```

---

## ✨ 主要機能

### 🌐 Web コンソール
- **エンドポイント**: `http://192.168.11.231:8067`
- bcrypt ハッシュ認証（24h セッション管理）
- WebSocket リアルタイムチャット + SSE（Server-Sent Events）
- スレッド別会話管理・タグ・検索・日時フィルタ
- ロードマップ表示・並列結果カード
- マークダウンレンダリング + コードシンタックスハイライト
- テーマカラー切替・ダークモード・コンパクトモード

### 🔧 メンテナンスページ
| タブ | 機能 |
|------|------|
| ログ | コンソール / メンテナンス / 監査 YAML / journalctl をタブ切替 |
| アップデート | ① 変更確認 → ② サンドボックス検証（venv 分離 + インポート + pytest）→ ③ 本番適用 |
| パッケージ | pip list / 更新可能パッケージ確認 / 個別アップグレード |
| サービス | systemd ユーザーサービス（起動・停止・再起動） |
| システム | CPU / メモリ / ディスク / キャッシュ状態 / スキル進化手動実行 |

### 🤚 HITL（Human-in-the-Loop）
- LangGraph `human_interrupt` ノードでグラフ処理を一時停止
- 参謀（sanbo）が `rm -rf`・`git push --force` 等の破壊的操作を正規表現で検出
- Web UI に承認カードを表示（✅ 承認 / ❌ 却下 / ✏️ 修正指示）
- `/api/resume` エンドポイントで処理を再開

### 🤖 ローカル LLM（pct239）
- Gemma4 MoE 27B + Nemotron 3 を排他制御で切替運用
- `LocalModelManager` シングルトン + 分散ロック + aiohttp セッション永続化
- 右筆: ローカル優先 → Gemini Flash-Lite クラウドフォールバック
- 隠密: ローカル 2段フォールバック（Nemotron → Gemma4）、機密性のためクラウド不可

### 📊 スキル自動進化
- 会話履歴から繰り返しパターンを自動検出 → スキル候補生成（`utils/skill_tracker.py`）
- 監査ログ（YAML）から実行効果を分析（`core/audit.py`）
- `POST /api/v18/evolve` で手動トリガー

### ⚡ 高速筆耕パイプライン（3段構え）

軽量モデルを組み合わせた爆速生成ライン（`core/processors/fast_generation.py`）：

```
Stage 1: Cerebras (gemma2-9b-it)
         ↓ 日本語対応 9B・ウェーハスケール超高速推論
         ↓ 荒削りドラフト生成 (~80%完成)

Stage 2: Groq (llama-3.2-3b-preview)
         ↓ 超軽量 3B・爆速整形
         ↓ 構造・冗長性の改善

Stage 3: Haiku (claude-haiku-4-5-20251001)
         ↓ 最終清書・マークダウン整形・日本語品質仕上げ
         ↓ 完成出力
```

各ステージは独立してフォールバック可能。Cerebras 障害時は Haiku 直接生成に切替。

---

### ⚡ パフォーマンス最適化
- SemanticRouter 事前チェック（CONFIDENT 閾値以上で LLM ルーティングをスキップ）
- キャッシュキー最適化（`MD5(message)` でスレッド横断キャッシュ共有）
- 長大コンテキスト事前圧縮（12 ターン超で目付/節刀が要約生成）
- Anthropic Batch API ユーティリティ（非リアルタイム処理を 50% コスト削減）

---

## 📁 ディレクトリ構成

```
Bushidan-Multi-Agent/
├── core/
│   ├── langgraph_router.py     # LangGraph StateGraph + 全ルーティングロジック (~2100行)
│   ├── system_orchestrator.py
│   └── audit.py
├── roles/                      # 各役職の実装（11ファイル）
│   ├── base.py                 # BaseRole・RoleResult（HITL フィールド含む）
│   ├── daigensui.py            # 大元帥 (Claude Opus 4.6)
│   ├── shogun.py               # 将軍 (Claude Sonnet 4.6)
│   ├── sanbo.py                # 参謀 + 破壊的操作 HITL
│   ├── yuhitsu.py              # 右筆 (Gemma4 → Gemini FL fallback)
│   ├── onmitsu.py              # 隠密 (ローカル限定 2段フォールバック)
│   └── ...
├── utils/
│   ├── local_model_manager.py  # ローカル LLM 排他制御 + aiohttp 永続化
│   ├── skill_tracker.py        # スキル観察・候補生成
│   ├── client_registry.py      # LLM クライアントシングルトン
│   ├── cache_manager.py        # レスポンスキャッシュ
│   ├── anthropic_batch.py      # Batch API ユーティリティ
│   └── semantic_router.py      # SemanticRouter
├── console/
│   ├── app.py                  # FastAPI サーバー（全エンドポイント）
│   ├── auth.py                 # bcrypt 認証・セッション管理
│   ├── maintenance.py          # メンテナンス機能バックエンド
│   └── static/
│       ├── index.html          # シングルページ Web UI
│       └── style.css
├── integrations/
│   ├── notion/                 # Notion 統合（ローカルインデックス）
│   └── mcp/                    # MCP SDK ツールレジストリ
├── tests/
│   ├── unit/                   # ユニットテスト（auth 等）
│   └── performance/
├── notebooks/                  # JupyterLab 分析ノートブック
├── audit/                      # YAML 監査ログ格納
└── skills/                     # 承認済みスキル格納
```

---

## 🔐 セキュリティ設計

- Web コンソール: bcrypt ハッシュ認証（平文パスワードは `.env` に保存しない）
- セッショントークン: `secrets.token_urlsafe(32)`、24h TTL
- ローカルネットワーク内専用運用（LAN 外への公開なし）
- 隠密役職: ローカル LLM 専用（機密データのクラウド送信を構造的に排除）
- 環境変数: 全クレデンシャルを `.env` 管理（コードへのハードコード禁止）

---

## 🛠️ 技術スタック

| カテゴリ | 技術 |
|---------|------|
| **エージェントフレームワーク** | LangGraph 1.x + AsyncPostgresSaver |
| **DB** | PostgreSQL 17 + psycopg3 |
| **Web フレームワーク** | FastAPI + uvicorn |
| **認証** | bcrypt 5.x + secrets |
| **LLM クライアント** | anthropic, google-generativeai, groq, cohere, mistralai, cerebras-cloud-sdk, aiohttp |
| **Web UI** | Vanilla JS + marked.js + highlight.js + DOMPurify |
| **通信プロトコル** | WebSocket（チャット）、SSE（メンテナンス更新）、REST |
| **監視** | Prometheus + Grafana (pct100:9090/3000) |
| **テスト** | pytest + pytest-asyncio |

---

## 📈 バージョン履歴

| バージョン | 主な変更 |
|-----------|---------|
| **v18** | bcrypt 認証・メンテナンスページ・HITL Web UI・スキル進化・Gemini レビュー対応（セキュリティ強化・fire-and-forget 修正・ユニットテスト） |
| **v17** | PostgreSQL 永続化・スレッド管理・タグ・マークダウン・ロール別フィルタ・履歴検索・体感速度改善 |
| **v16** | LangGraph v2 移行・Gemma4 MoE 統合・WebSocket ハング修正・Nemotron 修復 |
| **v15** | 分散 Claude 処理アーキテクチャ（pct237 Claude API Server）・10役職体制 |
| **v14** | LangGraph HITL・MCP SDK・モバイルコンソール初版・ClientRegistry |

---

## 📝 ライセンス

MIT License — © 2025 98kuwa036
