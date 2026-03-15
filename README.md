# 🏯 武士団マルチエージェントシステム v14

[![Version](https://img.shields.io/badge/Version-14.0-brightgreen)](https://github.com/98kuwa036/Bushidan-Multi-Agent)
[![Claude](https://img.shields.io/badge/Claude-Opus%204.6%20%2B%20Sonnet%204.6-purple)](https://www.anthropic.com/claude)
[![OpenAI](https://img.shields.io/badge/OpenAI-o3--mini%20high-412991)](https://openai.com/)
[![Cohere](https://img.shields.io/badge/Cohere-Command%20R%20%2B%20R%2B-coral)](https://cohere.com/)
[![Mistral](https://img.shields.io/badge/Mistral-Large%203-orange)](https://mistral.ai/)
[![Gemini](https://img.shields.io/badge/Gemini-3%20Flash%20Vision-blue)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-red)](https://groq.com/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron%20Local-76B900)](https://www.nvidia.com/)
[![ELYZA](https://img.shields.io/badge/ELYZA-Local%20Japanese-009688)](https://elyza.ai/)
[![LangGraph](https://img.shields.io/badge/LangGraph-HITL%20%2B%20MCP%20SDK-orange)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## v14 の革新

**Bushidan v14** は v11.5 の9層アーキテクチャを完全リビルドし、**10役職体制** + **LangGraph HITL** + **MCP SDK** + **モバイルコンソール** を実現しました。

### v14 ハイライト

- **🏯 10役職体制**: 受付・外事・斥候・軍師・参謀・将軍・大元帥・右筆・検校・隠密
- **🔄 LangGraph HITL**: Human-in-the-Loop 中断ノード（3分岐: human / loop / done）
- **⏱️ ノードタイムアウト**: 役職別タイムアウト設定 + フォールバックマップ
- **🔌 MCP SDK**: MCPToolRegistry シングルトン + 役職別アクセス制御（ACL）
- **💰 Claude Pro CLI 優先戦略**: 大元帥・将軍は Pro CLI (無制限) → API フォールバック
- **📱 モバイルコンソール**: FastAPI + WebSocket（port 8067）
- **🔧 ClientRegistry**: シングルトン, role_key → BaseLLMClient 遅延初期化・キャッシュ
- **🩺 ヘルスチェック統合**: ルーティング判断時にクライアント死活確認 + 自動フォールバック
- **📝 context_summary**: リレー間コンテキスト要約でトークン節約

---

## 📋 目次

- [10役職体制](#10役職体制)
- [LangGraph v14 StateGraph](#langgraph-v14-stategraph)
- [HITL（Human-in-the-Loop）](#hitlhuman-in-the-loop)
- [MCP SDK・権限マトリクス](#mcp-sdk権限マトリクス)
- [モバイルコンソール](#モバイルコンソール)
- [Claude Code CLI フォールバック](#claude-code-cli-フォールバック)
- [ClientRegistry アーキテクチャ](#clientregistry-アーキテクチャ)
- [インストール](#インストール)
- [パフォーマンス・タイムアウト設定](#パフォーマンスタイムアウト設定)
- [ハードウェア構成](#ハードウェア構成)
- [更新履歴](#更新履歴)

---

## 🎖️ 10役職体制

```
╔══════════════════════════════════════════════════════════════╗
║     武士団マルチエージェントシステム v14                         ║
║     "10役職体制 + LangGraph v14 + HITL + MCP SDK"           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  👑 大元帥  - Claude Opus 4.6    最高難度・戦略設計            ║
║  🎌 将軍    - Claude Sonnet 4.6  メインワーカー                ║
║  🧠 軍師    - o3-mini (high)     深層推論・PDCA                ║
║  📋 参謀    - Mistral Large 3    汎用コーディング              ║
║  🌏 外事    - Command R+         RAG・外部情報収集             ║
║  🔔 受付    - Command R          フォールバック処理            ║
║  ⚡ 斥候    - Llama 3.3 (Groq)   高速Q&A                      ║
║  👁️ 検校    - Gemini 3 Flash     マルチモーダル検証            ║
║  ✍️ 右筆    - ELYZA (Local)      日本語清書・整形              ║
║  🥷 隠密    - Nemotron (Local)   機密・オフライン処理          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### 役職詳細

| 役職 | ロールキー | モデル | 企業 | 特徴 | コスト |
|---|---|---|---|---|---|
| 👑 大元帥 | `daigensui` | Claude Opus 4.6 | Anthropic | 最高難度・戦略設計・最終エスカレーション | ¥10-50/req |
| 🎌 将軍 | `shogun` | Claude Sonnet 4.6 | Anthropic | メインワーカー・高難度コーディング | ¥0-5/req |
| 🧠 軍師 | `gunshi` | o3-mini (high) | OpenAI | 深層推論・PDCA エンジン | ¥0.05-0.20/req |
| 📋 参謀 | `sanbo` | Mistral Large 3 | Mistral AI | 汎用コーディング・MCP連携 | ¥0.01-0.10/req |
| 🌏 外事 | `gaiji` | Command R+ | Cohere | RAG・外部情報・Retrieval特化 | ¥0.01/req |
| 🔔 受付 | `uketuke` | Command R | Cohere | デフォルトフォールバック | ¥0.003/req |
| ⚡ 斥候 | `seppou` | Llama 3.3 70B (Groq) | Meta / Groq | 高速Q&A・300-500tok/s | **¥0 (無料)** |
| 👁️ 検校 | `kengyo` | Gemini 3 Flash Vision | Google | マルチモーダル・UI/UX検証 | ¥0.01/req |
| ✍️ 右筆 | `yuhitsu` | ELYZA (Local) | ELYZA/Meta | 日本語清書・オフライン | **¥0** |
| 🥷 隠密 | `onmitsu` | Nemotron-3-Nano (Local) | NVIDIA | 機密処理・オフライン | **¥0** |

### 設計原則: 役割分離

> **大元帥・将軍**: Pro CLI 優先 → API フォールバック（コスト最適化）
>
> **外事 (Command R+)**: Cohere の Retrieval 特化モデル。RAG・Notion検索・外部情報収集に最適
>
> **受付 (Command R)**: 軽量フォールバック。他役職がすべて応答不能な際の最終砦
>
> **右筆・隠密**: 完全ローカル実行。機密データは一切外部送信しない

---

## 🔄 LangGraph v14 StateGraph

```
[START]
  ↓
[analyze_intent]    タスク複雑度・種別分析
  ↓
[notion_retrieve]   Notion RAG 検索・コンテキスト注入
  ↓
[route_decision]    ルーティング判断 (10役職 + ヘルスチェック)
  ↓
┌─────────────────────────────────────────────────────┐
│                    10 ルーティング先                   │
│  groq_qa       → 斥候    (高速Q&A)                  │
│  gunshi_pdca   → 軍師    (深層推論・PDCA)            │
│  gaiji_rag     → 外事    (RAG・外部情報)             │
│  taisho_mcp    → 参謀    (コーディング+MCP)          │
│  yuhitsu_jp    → 右筆    (日本語清書)                │
│  karo_default  → 受付    (フォールバック)            │
│  onmitsu_local → 隠密    (機密・ローカル)            │
│  kengyo_vision → 検校    (画像解析)                  │
│  shogun_direct → 将軍    (メインワーカー)            │
│  daigensui_exec→ 大元帥  (最終エスカレーション)      │
└─────────────────────────────────────────────────────┘
  ↓
[check_followup]    3分岐ジャッジ
  ├─ "human" → [human_interrupt]  (HITL 中断・ユーザー確認待ち)
  ├─ "loop"  → [notion_retrieve]  (自律ループ継続)
  └─ "done"  → [notion_store]     (Notion に結果保存・完了)
```

### フォールバックマップ（障害時の自動切替）

| 障害ノード | フォールバック先 |
|---|---|
| `groq_qa` | `karo_default` (受付) |
| `gunshi_pdca` | `shogun_direct` (将軍) |
| `gaiji_rag` | `karo_default` (受付) |
| `taisho_mcp` | `shogun_direct` (将軍) |
| `yuhitsu_jp` | `karo_default` (受付) |
| `onmitsu_local` | `karo_default` (受付) |
| `kengyo_vision` | `shogun_direct` (将軍) |
| `daigensui_direct` | `shogun_direct` (将軍) |
| `shogun_direct` | `karo_default` (受付) |

---

## 🤝 HITL（Human-in-the-Loop）

v14 では `check_followup` ノードが応答を解析し、3方向に分岐します。

```python
# check_followup が判定する3分岐
"human" → ユーザー確認が必要（危険な操作・重要な意思決定）
"loop"  → 自律継続が可能（追加情報収集・多段階タスク）
"done"  → 完了（Notion保存 → END）
```

### HITL フロー

```
[実行ノード完了]
       ↓
[check_followup]
       ├─ requires_human_approval: true
       │         ↓
       │   [human_interrupt]
       │         ↓
       │   ユーザーがチャットで応答（最大300秒待機）
       │         ↓
       │   [route_decision へ再入力]
       │
       ├─ requires_followup: true
       │         ↓
       │   [notion_retrieve] ← ループ継続
       │
       └─ 完了
               ↓
         [notion_store] → END
```

### モバイルコンソールでのHITL操作

ブラウザ（またはスマートフォン）から `http://[host]:8067` にアクセスし、HITL 中断時にリアルタイム応答できます。

---

## 🔌 MCP SDK・権限マトリクス

### MCPToolRegistry（シングルトン）

```python
from core.mcp_sdk import MCPToolRegistry

registry = MCPToolRegistry.get()

# 役職が利用可能なツール一覧を取得
tools = registry.get_tools_for_role("shogun")

# ツール呼び出し（権限チェック付き）
result = await registry.call_tool("github", "create_issue", {...}, role="shogun")
```

### MCPPermissionLevel（ACL）

| レベル | 説明 |
|---|---|
| `exclusive` | その役職専用（他は使用不可） |
| `primary` | 主要アクセス権 |
| `secondary` | 補助的アクセス権 |
| `readonly` | 読み取り専用 |
| `delegated` | 上位役職から委譲 |
| `forbidden` | アクセス禁止 |

### 主要MCPサーバー

| カテゴリ | MCP | 主な利用役職 |
|---|---|---|
| **AI** | Sequential Thinking | 軍師（exclusive） |
| **ブラウザ** | Playwright | 検校（exclusive） |
| **検索** | Tavily, Exa | 外事・参謀 |
| **ファイル** | Filesystem, Git | 将軍・参謀 |
| **記憶** | Graph Memory | 大元帥・将軍 |
| **DB** | Prisma | 参謀 |
| **連携** | Mattermost MCP | 全役職 |
| **連携** | Notion | 全役職 |
| **連携** | Discord | 将軍・参謀 |

---

## 📱 モバイルコンソール

### FastAPI + WebSocket サーバー（port 8067）

```bash
# コンソール起動
uvicorn console.app:app --host 0.0.0.0 --port 8067
```

ブラウザから `http://192.168.11.230:8067` でアクセス。スマートフォンからも操作可能。

### エンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/api/login` | ログイン（CONSOLE_PASSWORD 認証） |
| `POST` | `/api/chat` | チャット（同期） |
| `GET` | `/api/models` | 利用可能モデル一覧 |
| `GET` | `/api/history` | 会話履歴 |
| `GET` | `/api/health` | ヘルスチェック |
| `WS` | `/ws/chat` | WebSocket チャット（HITL対応） |

---

## 💻 Claude Code CLI フォールバック

### 大元帥・将軍の Proプラン優先戦略

v14 の最大の特徴が、**大元帥（Opus 4.6）** と **将軍（Sonnet 4.6）** が **Claude Code CLI** を使用する仕様です。

```
Web UI (モバイルコンソール @ port 8067)
    ↓
LangGraph Router v14
    ↓
大元帥 / 将軍 ロール
    ↓
[Claude Code CLI]  (`claude -p` コマンド)
    ├─ ✅ Pro無制限枠で実行成功
    │       ↓
    │   [結果を Web UI に返却]
    │
    └─ ❌ CLI失敗 / タイムアウト / エラー
            ↓
        [Anthropic API フォールバック]  (有料)
            ↓
        [結果を Web UI に返却]
```

### 実装詳細

#### `utils/claude_cli_client.py`

```python
# Claude Code CLI を優先実行
from utils.claude_cli_client import call_claude_with_fallback

result = await call_claude_with_fallback(
    prompt="複雑なコーディングタスク",
    model="claude-sonnet-4-6",
    api_key=ANTHROPIC_API_KEY,  # フォールバック用のみ
    system="あなたは将軍です"
)
```

**フロー:**
1. `claude -p <prompt>` で Pro CLI を実行（無制限枠）
2. CLI が成功 → 出力を返却 ✅
3. CLI が失敗・エラー・タイムアウト → Anthropic API にフォールバック ⚠️

#### 自動フォールバック判定

CLI が以下の場合、自動的に API にフォールバック：
- `Usage limit reached` — Pro枠超過
- `Rate limit` — レート制限
- `authentication failed` — 認証エラー
- `timeout` — 120秒以上の応答遅延
- 空応答 — 出力なし
- 非ゼロ終了コード — CLI異常終了

### コスト最適化

```
月間シナリオ:
┌─────────────────────────────────────────────┐
│ Claude Pro ¥3,000/月                        │
│  ├─ 無制限使用: 大元帥・将軍 (Pro CLI)      │
│  └─ テキスト処理: 自由に実行可能             │
│                                              │
│ Anthropic API ¥150/月 (予備)                │
│  ├─ Pro CLI 失敗時のみ発動                   │
│  └─ Prompt Caching で 90% コスト削減        │
└─────────────────────────────────────────────┘

結果: Pro CLI 優先 → API フォールバック により、
      Pro¥3,000枠を最大活用 + API コスト最小化
```

### Web UI での実行例

モバイルコンソール（port 8067）で大元帥に指示：

```
【Web UI】
入力: "複雑なTypeScript型定義を設計して"
モデル選択: "大元帥 (Claude Opus 4.6)"

【バックエンド処理】
1. 将軍ロール → claude_cli_client.py
2. Claude Code CLI実行: claude -p "複雑なTypeScript..."
3. Pro枠で無制限実行 ✅
4. 結果を Web UI にリアルタイム返却

【モバイル画面】
"型定義設計が完了しました。interface User { ... }"
```

### Pro CLI 必須環境

Claude Code CLI を使うには以下が必須です：

```bash
# Claude Code CLI のセットアップ
# https://claude.ai/code から Claude Code をインストール
# ターミナルで `claude` コマンドが使える状態

# 確認
$ claude --version
Claude 0.1.0

# Pro APIキーを設定
$ claude config set api-key YOUR_ANTHROPIC_API_KEY
```

---

---

## 🔧 ClientRegistry アーキテクチャ

### シングルトン・遅延初期化パターン

```python
# 全ロールが共通で使用するクライアント取得
from utils.client_registry import ClientRegistry

registry = ClientRegistry.get()           # シングルトン取得
client = registry.get_client("shogun")    # 遅延初期化・キャッシュ
response = await client.generate(
    messages=[{"role": "user", "content": "タスク"}],
    system="システムプロンプト"
)
```

### 8 つの Adapter クラス

| Adapter | 担当役職 | モデル |
|---|---|---|
| `_CohereAdapter` | 受付・外事 | Command R / Command R+ |
| `_GroqAdapter` | 斥候 | Llama 3.3 70B |
| `_O3MiniAdapter` | 軍師 | o3-mini (high) |
| `_MistralAdapter` | 参謀 | Mistral Large 3 |
| `_ClaudeAdapter` | 将軍・大元帥 | Sonnet 4.6 / Opus 4.6 |
| `_Gemini3Adapter` | 検校 | Gemini 3 Flash Vision |
| `_ElyzaAdapter` | 右筆 | ELYZA Local (Nemotron fallback) |
| `_NemotronAdapter` | 隠密 | Nemotron-3-Nano Local |

### BaseLLMClient ABC

```python
class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list,
        system: str = "",
        max_tokens: int = 2048,
        **kwargs,
    ) -> str: ...

    async def health_check(self) -> bool:
        return True  # デフォルト: 常時正常
```

---

## 🚀 インストール

### セットアップ手順

```bash
# リポジトリクローン
git clone https://github.com/98kuwa036/Bushidan-Multi-Agent.git
cd Bushidan-Multi-Agent

# Python 依存関係
pip install -r requirements.txt

# API Key 設定
cp .env.example .env
# .env を編集して API キーを設定

# システム起動
python main.py
```

### モバイルコンソール起動（別ターミナル）

```bash
uvicorn console.app:app --host 0.0.0.0 --port 8067
```

### PM2 での常駐起動

```bash
# メインシステム
pm2 start "python main.py" --name bushidan-main

# モバイルコンソール
pm2 start "uvicorn console.app:app --host 0.0.0.0 --port 8067" --name bushidan-console

pm2 save
```

### 必要な API Key

| Key | 役職 | コスト |
|---|---|---|
| `ANTHROPIC_API_KEY` | 大元帥 (Opus 4.6) + 将軍 (Sonnet 4.6) | Pro ¥3,000/月 + API |
| `OPENAI_API_KEY` | 軍師 (o3-mini high) | ~¥0.05-0.20/req |
| `MISTRAL_API_KEY` | 参謀 (Mistral Large 3) | ~¥0.01-0.10/req |
| `COHERE_API_KEY` | 外事 (Command R+) + 受付 (Command R) | ~¥0.003-0.01/req |
| `GEMINI_API_KEY` | 検校 (Gemini 3 Flash Vision) | ~¥0.01/req |
| `GROQ_API_KEY` | 斥候 (Llama 3.3 70B) | **無料** |
| `TAVILY_API_KEY` | MCP・Web検索 | 1,000/月 free |
| `NOTION_TOKEN` + `NOTION_DB_ID` | 知識ベース・タスク永続化 | 無料 |
| `MATTERMOST_URL` + `MATTERMOST_TOKEN` | Mattermost Bot + MCP | 無料 |
| `CONSOLE_PASSWORD` | モバイルコンソール認証 | — |

### ローカルLLM（右筆・隠密）

```bash
# HP ProDesk 600 (192.168.11.239) で実行
# ELYZA (右筆)
./setup/setup_local_llm.sh --model elyza

# Nemotron-3-Nano (隠密)
./setup/setup_nemotron.sh

# llama.cpp サーバー起動
./llama.cpp/build/bin/llama-server \
  --model models/nemotron/Nemotron-3-Nano-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --threads 6 --ctx-size 8192
```

---

## ⏱️ パフォーマンス・タイムアウト設定

### ノード別タイムアウト

| ノード | 役職 | タイムアウト | フォールバック |
|---|---|---|---|
| `groq_qa` | 斥候 | **30秒** | 受付 |
| `karo_default` | 受付 | 60秒 | — |
| `gaiji_rag` | 外事 | 60秒 | 受付 |
| `taisho_mcp` | 参謀 | 60秒 | 将軍 |
| `kengyo_vision` | 検校 | 60秒 | 将軍 |
| `gunshi_pdca` | 軍師 | **90秒** | 将軍 |
| `yuhitsu_jp` | 右筆 | 90秒 | 受付 |
| `shogun_direct` | 将軍 | **120秒** | 受付 |
| `onmitsu_local` | 隠密 | 120秒 | 受付 |
| `daigensui_direct` | 大元帥 | **180秒** | 将軍 |

### 処理時間目標

| 複雑度 | 処理時間 | ルーティング | コスト |
|---|---|---|---|
| **Simple Q&A** | **2-5秒** | 斥候 Groq | ¥0 |
| **Fallback** | **5-10秒** | 受付 Command R | ¥0.003 |
| **RAG / 外部情報** | **10-20秒** | 外事 Command R+ | ¥0.01 |
| **コーディング (MCP)** | **15-30秒** | 参謀 Mistral | ¥0.05 |
| **複雑推論 (PDCA)** | **30-60秒** | 軍師 o3-mini | ¥0.10-0.30 |
| **メインワーカー** | **15-30秒** | 将軍 Sonnet 4.6 | ¥0-5 |
| **戦略設計** | **45-90秒** | 大元帥 Opus 4.6 | ¥10-50 |
| **日本語清書** | **20-40秒** | 右筆 ELYZA Local | ¥0 |
| **機密処理** | **30-60秒** | 隠密 Nemotron Local | ¥0 |

---

## 🔐 セキュリティ

| 機能 | 説明 |
|---|---|
| **機密処理分離** | 秘匿データは隠密（ローカルLLM）で処理、外部API送信なし |
| **右筆ローカル実行** | 日本語データをローカルELYZAで処理 |
| **MCP権限マトリクス** | 役職別MCPアクセス制御（exclusive / primary / secondary / forbidden） |
| **HITL承認フロー** | 危険な操作は human_interrupt ノードで一時停止・ユーザー確認 |
| **ヘルスチェック** | 各クライアントの5分キャッシュ死活確認・自動フォールバック |
| **モバイルコンソール認証** | CONSOLE_PASSWORD によるセッション認証 |

---

## 🖥️ ハードウェア構成

### HP ProDesk / EliteDesk 2台構成

| 機器 | IP | 役割 | スペック |
|---|---|---|---|
| **EliteDesk (本陣)** | 192.168.11.230 | システム統括・オーケストレーション・モバイルコンソール | i5-8500, 16GB DDR4 |
| **ProDesk 600 (隠密)** | 192.168.11.239 | ローカルLLM専用機（ELYZA + Nemotron） | i5-8500, **32GB DDR4** |

### ローカルLLM 運用仕様

| モデル | 役職 | 量子化 | RAM使用量 | 推論速度 | エンドポイント |
|---|---|---|---|---|---|
| ELYZA Llama3 Japanese | 右筆 | Q4_K_M | ~8GB | 10-20tok/s | http://192.168.11.239:8081 |
| Nemotron-3-Nano-30B | 隠密 | Q4_K_M | ~21GB | 15-25tok/s | http://192.168.11.239:8080 |

---

## 💰 月次コスト見積り

| 項目 | 金額 | 説明 |
|---|---|---|
| **Claude Pro** | ¥3,000 | Pro CLI + Prompt Caching（大元帥・将軍） |
| **Claude API** | ¥150 | Prompt Caching 90%削減 |
| **OpenAI (o3-mini)** | ¥200 | 軍師 PDCA |
| **Mistral Large 3** | ¥100 | 参謀 コーディング |
| **Cohere** | ¥50 | 外事 (R+) + 受付 (R) |
| **Gemini Flash** | ¥80 | 検校 マルチモーダル |
| **Groq** | ¥0 | 斥候 無料枠内 |
| **電力** | ¥80 | ローカルLLM 2モデル |
| **合計** | **¥3,660** | v11.5 比 -¥90（Grok除外・Mistral/Cohere追加） |

---

## 📁 プロジェクト構造

```
Bushidan-Multi-Agent/
├── main.py                      # CLIエントリーポイント v14
├── core/
│   ├── langgraph_router.py      # LangGraph v14 (HITL + タイムアウト + ヘルスチェック)
│   ├── system_orchestrator.py   # 簡素化オーケストレーター (~200行)
│   ├── state.py                 # BushidanState (HITL + context_summary)
│   ├── mcp_permissions.py       # MCP権限マトリクス
│   └── mcp_sdk.py               # MCPToolRegistry シングルトン
├── roles/
│   ├── base.py                  # BaseRole v14 (ClientRegistry連携)
│   ├── daigensui.py             # 大元帥
│   ├── shogun.py                # 将軍
│   ├── gunshi.py                # 軍師
│   ├── sanbo.py                 # 参謀
│   ├── gaiji.py                 # 外事
│   ├── uketuke.py               # 受付
│   ├── seppou.py                # 斥候 (Groq)
│   ├── kengyo.py                # 検校
│   ├── yuhitsu.py               # 右筆
│   └── onmitsu.py               # 隠密
├── utils/
│   ├── client_registry.py       # ClientRegistry シングルトン (10ロール)
│   ├── base_client.py           # BaseLLMClient ABC
│   ├── claude_cli_client.py     # Claude Pro CLI + APIフォールバック
│   └── ...                      # 各クライアント実装
├── console/
│   ├── app.py                   # FastAPI モバイルコンソール (port 8067)
│   └── static/                  # HTML/CSS
├── config/
│   ├── mcp_permissions.yaml     # MCP権限マトリクス設定
│   └── settings.yaml            # システム設定
└── docs/
    └── agent_instructions/      # 各役職の指示書
```

---

## 📋 更新履歴

| バージョン | 日付 | 主要変更 |
|---|---|---|
| **v14** | 2026-03-14 | 🆕 10役職体制・LangGraph HITL・MCP SDK・ClientRegistry・モバイルコンソール |
| **v11.5** | 2026-03-03 | LangGraph × MCP × Notion 密結合・Mattermost Bot + MCP |
| **v11.4** | 2026-03-03 | 脱中国企業 + 9層アーキテクチャ（大元帥・参謀A/B・家老A/B・隠密） |
| **v10.1** | 2026-02-05 | 傭兵(Kimi K2.5) + Smithery MCP + 4層鉄壁チェーン |
| **v10.0** | 2026-02-04 | 軍師(Gunshi)層追加・PDCA Engine・5層アーキテクチャ |
| **v9.4** | 2026-02-01 | BDIフレームワーク全層統合・日本語ログ全面化 |

---

## 📄 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

---

<div align="center">

**🏯 武士団 v14 — 10役職体制 × LangGraph HITL × MCP SDK × モバイルコンソール 🏯**

**Generated with [Claude Code](https://claude.ai/code)**

</div>
