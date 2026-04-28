"""
console/app.py — 武士団モバイルコンソール FastAPI サーバー v15（分散Claude処理対応）

エンドポイント:
  POST /api/login          — ログイン (CONSOLE_PASSWORD)
  POST /api/chat           — チャット (同期)
  GET  /api/models         — 利用可能モデル一覧
  GET  /api/history        — 会話履歴
  GET  /api/health         — ヘルスチェック
  WS   /ws/chat            — WebSocket チャット

起動:
  uvicorn console.app:app --host 0.0.0.0 --port 8067

v15: 分散Claude処理 + 10役職体制
  - Claude API Server (192.168.11.237:8070) 統合
  - Claude Pro CLI優先 → Anthropic API フォールバック
  - LangGraph Router v15 統合
"""

import asyncio
import datetime
import json
import re
import os
import time
import uuid
from typing import Optional
import httpx
from pathlib import Path

from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.logger import get_logger

# ── Prometheus メトリクス ────────────────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    )
    _HAS_PROMETHEUS = True

    _req_total = Counter(
        "bushidan_requests_total",
        "Total chat requests",
        ["source", "route"],
    )
    _req_errors = Counter(
        "bushidan_errors_total",
        "Total chat errors",
        ["source"],
    )
    _req_duration = Histogram(
        "bushidan_request_duration_seconds",
        "Request duration in seconds",
        ["route"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )
    _active_ws = Gauge("bushidan_active_websockets", "Active WebSocket connections")

except ImportError:
    _HAS_PROMETHEUS = False

# ── 環境変数読み込み (API キー等) ────────────────────────────────────────
load_dotenv()

logger = get_logger(__name__)

_HTTP_CLIENT: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=180.0, write=10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _HTTP_CLIENT


app = FastAPI(title="武士団コンソール", version="18")

# CORS対応（ローカルLLMサーバーのプロキシ用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 静的ファイル ────────────────────────────────────────────────────────
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ── PostgreSQL 接続設定 ────────────────────────────────────────────────────
POSTGRES_URL = os.environ.get("POSTGRES_URL", "")
if not POSTGRES_URL:
    raise RuntimeError("POSTGRES_URL が未設定です。.env を確認してください。")

# ── バックグラウンドタスク管理 ─────────────────────────────────────────────
_bg_tasks: set = set()


def _fire(coro, *, name: str = None) -> "asyncio.Task":
    """fire-and-forget タスクを起動し GC による中断を防ぐ"""
    t = asyncio.create_task(coro, name=name)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return t


# ── ロール有効/無効管理 ───────────────────────────────────────────────────
_DISABLED_ROLES: set = set()  # 無効化されたロールキーのセット（メモリ管理）

# ── Startup Event: DB スキーマ初期化 + LLM Gemma4 切り替え ──────────────
@app.on_event("startup")
async def startup_init_db_and_switch():
    """Console 起動時: DB スキーマ作成 + Gemma4 に自動切り替え (v17)"""
    # Phase 1: DB スキーマ初期化
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                # threads テーブル
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS threads (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT '新しい会話',
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        is_archived BOOLEAN DEFAULT FALSE
                    );
                """)
                # messages テーブル
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        thread_id TEXT REFERENCES threads(id) ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        agent_role TEXT,
                        model TEXT,
                        execution_time FLOAT,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                # tags カラム追加（タグ機能用）
                await cur.execute("""
                    ALTER TABLE threads ADD COLUMN IF NOT EXISTS tags TEXT DEFAULT '[]';
                """)
                # インデックス
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);")
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC);")
                # system_config テーブル (key-value 設定ストア)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS system_config (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                logger.info("✅ Console startup: DB schema initialized")

                # Phase 1.5: テストデータ初期化
                # 既存のテストスレッドを削除
                await cur.execute("""
                    DELETE FROM threads WHERE id LIKE 'test-%'
                """)

                test_data = [
                    ("test-001", "AIチャットボット構築", [
                        ("user", "AIチャットボットを構築したいのですが、どんな技術スタックを使うべきですか？", None, None),
                        ("assistant", "AIチャットボット構築には以下の技術スタックがおすすめです:\n\n1. **LLM選択**: Claude、GPT-4、Mistral\n2. **フレームワーク**: LangChain、LlamaIndex、HuggingFace\n3. **デプロイ**: FastAPI、Docker、Kubernetes", "gunshi", "Claude Sonnet"),
                        ("user", "実装時の注意点はありますか？", None, None),
                        ("assistant", "実装時の注意点:\n- トークン管理\n- レイテンシー最適化\n- エラーハンドリング\n- ロギング・監視", "gunshi", "Claude Sonnet"),
                    ]),
                    ("test-002", "データベース最適化", [
                        ("user", "PostgreSQLのクエリパフォーマンス改善方法を教えてください", None, None),
                        ("assistant", "PostgreSQL クエリ最適化:\n\n- インデックス設計\n- クエリプラン分析\n- バッチ処理\n- コネクションプーリング", "sanbo", "Mistral Large"),
                        ("user", "大規模データセットでの対応は？", None, None),
                        ("assistant", "大規模データ対応:\n- シャーディング\n- キャッシング (Redis)\n- パーティショニング\n- 非同期処理", "sanbo", "Mistral Large"),
                    ]),
                    ("test-003", "React フォーム構築", [
                        ("user", "Reactでバリデーション付きフォームを実装したいです", None, None),
                        ("assistant", "推奨方法: React Hook Form + Zod\n- 軽量で高速\n- TypeScript対応\n- 優れたバリデーション", "daigensui", "Claude Opus"),
                    ]),
                    ("test-004", "セキュリティ監査", [
                        ("user", "本番環境でのセキュリティチェックリストを教えてください", None, None),
                        ("assistant", "セキュリティチェックリスト:\n- SQLインジェクション対策\n- XSS対策\n- CSRF対策\n- TLS 1.3有効化", "shogun", "Claude Sonnet"),
                    ]),
                    ("test-005", "モバイルアプリ設計", [
                        ("user", "モバイルアプリのアーキテクチャを教えてください", None, None),
                        ("assistant", "推奨アーキテクチャ:\n- フロントエンド: React Native/Flutter\n- バックエンド: REST/GraphQL\n- 認証: OAuth 2.0", "uketuke", "Llama 3.3 70B (Groq)"),
                        ("user", "パフォーマンス最適化は？", None, None),
                        ("assistant", "最適化方法:\n- バンドル削減\n- 画像最適化\n- 遅延ロード\n- APIバッチ化", "uketuke", "Llama 3.3 70B (Groq)"),
                    ]),
                    ("test-006", "クラウド移行", [
                        ("user", "AWS移行戦略を教えてください", None, None),
                        ("assistant", "AWS移行フェーズ:\n1. 現状分析\n2. マイグレーション計画\n3. パイロット実行\n4. 本番移行\n\nサービス: EC2、RDS、CloudFront", "uketuke", "Llama 3.3 70B (Groq)"),
                    ]),
                ]

                for idx, (thread_id, title, messages) in enumerate(test_data):
                    await cur.execute("SELECT id FROM threads WHERE id = %s", (thread_id,))
                    if not await cur.fetchone():
                        days_ago = len(test_data) - idx
                        created_at = datetime.datetime.now() - datetime.timedelta(days=days_ago)
                        await cur.execute(
                            "INSERT INTO threads (id, title, created_at, updated_at, tags) VALUES (%s, %s, %s, %s, %s)",
                            (thread_id, title, created_at, created_at, "[]")
                        )
                        for role, content, agent_role, model in messages:
                            await cur.execute(
                                "INSERT INTO messages (thread_id, role, content, agent_role, model, execution_time) VALUES (%s, %s, %s, %s, %s, %s)",
                                (thread_id, role, content, agent_role, model, 0.5 if agent_role else None)
                            )
                logger.info("✅ Console startup: Test data initialized")
    except Exception as e:
        logger.warning(f"⚠️ DB initialization failed: {e}")

    # Phase 2: LLM サーバー Gemma4 に切り替え
    try:
        llm_url = os.environ.get("LOCAL_LLM_URL")
        if not llm_url:
            logger.warning("LOCAL_LLM_URL 未設定: LLM 切り替えスキップ")
        else:
            # 3回リトライ
            switched = False
            for attempt in range(3):
                try:
                    resp = await _get_http_client().post(
                        f"{llm_url}/switch/gemma", timeout=10.0
                    )
                    resp.raise_for_status()
                    logger.info("✅ Console startup: LLM server switched to Gemma4 MoE")
                    switched = True
                    break
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2)
                        continue
                    logger.warning(f"⚠️ Failed to switch LLM to Gemma4 after 3 attempts: {e}")
            _ = switched  # 結果はログ済み
    except Exception as e:
        logger.warning(f"⚠️ LLM auto-switch failed: {e}")

    # Phase 3: LangGraph Router 事前初期化（初回チャットのレイテンシを削減）
    _fire(_prewarm_router(), name="router_prewarm")
    # Phase 4: スキル自動進化バックグラウンドループ起動（デフォルト無効）
    _fire(_skill_evolution_loop(), name="skill_evolution_loop")


async def _prewarm_router():
    """LangGraph Router を起動時にバックグラウンド初期化する。
    DB 接続が確立されるまでポーリングしてから Router を初期化する。
    """
    # DB が応答するまで最大30秒ポーリング（sleep(N) マジックナンバーを避ける）
    for _attempt in range(10):
        try:
            import psycopg
            async with await psycopg.AsyncConnection.connect(
                POSTGRES_URL, autocommit=True, connect_timeout=3
            ) as _c:
                await _c.execute("SELECT 1")
            break  # DB 応答確認できたら即抜ける
        except Exception:
            await asyncio.sleep(3)
    try:
        await _get_router()
        logger.info("🔥 Router pre-warm 完了 — 初回チャットは即時応答")
    except Exception as e:
        logger.warning("⚠️ Router pre-warm 失敗 (遅延初期化にフォールバック): %s", e)


# ── スキル自動進化ループ ──────────────────────────────────────────────────
_EVOLUTION_CHECK_INTERVAL = 3600  # 1時間ごとにチェック


async def _get_evolution_config() -> dict:
    """system_config からスキル進化の設定を取得する。"""
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT key, value FROM system_config WHERE key LIKE 'skill_evolution_%'"
                )
                rows = await cur.fetchall()
                cfg = {r[0]: r[1] for r in rows}
        return {
            "enabled": cfg.get("skill_evolution_auto_enabled", "false").lower() == "true",
            "interval_hours": int(cfg.get("skill_evolution_interval_hours", "24")),
            "min_observations": int(cfg.get("skill_evolution_min_observations", "10")),
        }
    except Exception as e:
        logger.warning("⚠️ evolution config read failed: %s", e)
        return {"enabled": False, "interval_hours": 24, "min_observations": 10}


async def _should_run_evolution(cfg: dict) -> tuple[bool, str]:
    """進化を実行すべきか判定する。(実行可否, 理由) を返す。"""
    if not cfg["enabled"]:
        return False, "自動実行が無効です"
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                # 最終実行時刻を取得
                await cur.execute(
                    "SELECT run_at FROM skill_evolution_log ORDER BY run_at DESC LIMIT 1"
                )
                row = await cur.fetchone()
                last_run = row[0] if row else None

                if last_run:
                    elapsed_hours = (datetime.datetime.now() - last_run).total_seconds() / 3600
                    if elapsed_hours < cfg["interval_hours"]:
                        remaining = cfg["interval_hours"] - elapsed_hours
                        return False, f"前回実行から {elapsed_hours:.1f}h 経過 (次回まで {remaining:.1f}h)"

                # 最終実行以降の新規 observation 数を確認
                since = last_run or datetime.datetime.now() - datetime.timedelta(days=365)
                await cur.execute(
                    "SELECT COUNT(*) FROM skill_observations WHERE created_at > %s", (since,)
                )
                count_row = await cur.fetchone()
                new_obs = count_row[0] if count_row else 0
                if new_obs < cfg["min_observations"]:
                    return False, f"新規観測数 {new_obs} < 閾値 {cfg['min_observations']}"
                return True, f"実行条件充足 (新規観測: {new_obs}件)"
    except Exception as e:
        return False, f"判定エラー: {e}"


async def _skill_evolution_loop():
    """スキル自動進化バックグラウンドループ。デフォルトは無効状態で起動。"""
    await asyncio.sleep(60)  # 起動直後の実行を避ける
    while True:
        try:
            cfg = await _get_evolution_config()
            should_run, reason = await _should_run_evolution(cfg)
            if should_run:
                logger.info("🧬 スキル自動進化: 実行開始 (%s)", reason)
                from core.skill import evolve_skills_from_audit
                result = await evolve_skills_from_audit(days=cfg["interval_hours"] // 24 or 1)
                logger.info("🧬 スキル自動進化: 完了 %s", result)
            else:
                logger.debug("🧬 スキル自動進化: スキップ (%s)", reason)
        except Exception as e:
            logger.warning("⚠️ スキル自動進化ループエラー: %s", e)
        await asyncio.sleep(_EVOLUTION_CHECK_INTERVAL)


# ── ルーター (遅延初期化) ──────────────────────────────────────────────
_router = None
_router_lock = asyncio.Lock()


async def _get_router():
    global _router
    if _router is not None:
        return _router
    async with _router_lock:
        if _router is not None:
            return _router
        from core.langgraph_router import LangGraphRouter
        _router = LangGraphRouter()
        await _router.initialize()
        logger.info("🔗 Console: LangGraph Router v15 初期化完了")
        return _router


# ── モデル定義 ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    password: str

class ChatRequest(BaseModel):
    message: str
    model: str = "auto"         # "auto" or role_key
    thread_id: str = ""

class ChatResponse(BaseModel):
    response: str
    agent_role: str
    handled_by: str
    route: str
    execution_time: float
    thread_id: str

class LLMTestRequest(BaseModel):
    endpoint: str
    body: dict

class ResumeRequest(BaseModel):
    thread_id: str
    response: str  # ユーザーの Go サイン or テキスト応答

class ThreadUpdateRequest(BaseModel):
    title: Optional[str] = None
    tags: Optional[str] = None  # JSON配列文字列: '["#React","#進行中"]'

# ── 認証ヘルパー ──────────────────────────────────────────────────────

def _check_auth(token: str = ""):
    from console.auth import validate_session
    if not validate_session(token):
        raise HTTPException(status_code=401, detail="Unauthorized")

# ── エンドポイント ────────────────────────────────────────────────────

@app.get("/")
async def index():
    """モバイルUI (index.html)"""
    html_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return JSONResponse({"message": "武士団コンソール v15"})


@app.post("/api/login")
async def login(req: LoginRequest):
    """セッション認証"""
    from console.auth import check_password, create_session
    if not check_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    token = create_session()
    return {"token": token}


@app.post("/api/chat")
async def chat(req: ChatRequest, token: str = ""):
    """同期チャット"""
    _check_auth(token)
    router = await _get_router()
    # 無効化されたロールは auto にフォールバック
    effective_role = req.model if req.model not in _DISABLED_ROLES else "auto"
    forced_role = effective_role if effective_role != "auto" else None
    thread_id = req.thread_id or str(uuid.uuid4())[:8]
    t0 = time.time()

    try:
        result = await router.process_message(
            message=req.message,
            thread_id=thread_id,
            source="console",
            forced_role=forced_role,
        )
    except Exception:
        if _HAS_PROMETHEUS:
            _req_errors.labels(source="console").inc()
        raise

    route = result.get("route", "unknown")
    elapsed = time.time() - t0
    if _HAS_PROMETHEUS:
        _req_total.labels(source="console", route=route).inc()
        _req_duration.labels(route=route).observe(elapsed)

    return ChatResponse(
        response=result.get("response", ""),
        agent_role=result.get("agent_role", ""),
        handled_by=result.get("handled_by", ""),
        route=route,
        execution_time=result.get("execution_time", elapsed),
        thread_id=thread_id,
    )


@app.get("/api/models")
async def models():
    """利用可能モデル一覧"""
    model_list = [
        {"key": "auto",      "name": "🔀 自動ルーティング",                             "emoji": "🔀"},
        {"key": "all",       "name": "🏯 全員会議（一斉送信）",                          "emoji": "🏯"},
        {"key": "daigensui", "name": "👑 大元帥（最終判断・監査）— Claude Opus 4.6",      "emoji": "👑"},
        {"key": "shogun",    "name": "🎌 将軍（計画立案・指揮）— Claude Sonnet 4.6",     "emoji": "🎌"},
        {"key": "gunshi",    "name": "🧠 軍師（汎用処理・推論）— Command A",              "emoji": "🧠"},
        {"key": "metsuke",   "name": "🔎 目付（要約・整形・軽量推論）— Mistral Small",   "emoji": "🔎"},
        {"key": "sanbo",     "name": "📋 参謀（ツール実行・コーディング）— Gemini Flash", "emoji": "📋"},
        {"key": "gaiji",     "name": "🌏 外事（外部情報・RAG）— Command R",              "emoji": "🌏"},
        {"key": "uketuke",   "name": "🚪 受付（Q&A・雑談・コード）— Llama 3.3 70B (Groq)", "emoji": "🚪"},
        {"key": "kengyo",    "name": "👁️ 検校（画像解析）— Gemini 3.1 Flash Image",    "emoji": "👁️"},
        {"key": "yuhitsu",   "name": "✍️ 右筆（日本語処理）— Gemma4 MoE Local",         "emoji": "✍️"},
        {"key": "onmitsu",   "name": "🥷 隠密（機密データ処理）— Nemotron Local",         "emoji": "🥷"},
    ]
    return {"models": model_list}


@app.get("/api/settings/claude-fallback")
async def get_claude_fallback(token: str = ""):
    """現在のClaudeインシデントフォールバックモードを返す"""
    _check_auth(token)
    from utils.claude_cli_client import get_incident_fallback_mode
    return {"mode": get_incident_fallback_mode()}


class FallbackModeRequest(BaseModel):
    mode: str  # "gemini" または "bedrock"

@app.post("/api/settings/claude-fallback")
async def set_claude_fallback(req: FallbackModeRequest, token: str = ""):
    """Claudeインシデント時のフォールバックモードを切り替える。
    mode: "gemini" (デフォルト: Gemini 3.1 Pro) / "bedrock" (AWS Bedrock)
    """
    _check_auth(token)
    if req.mode not in ("gemini", "bedrock"):
        return {"error": "mode は 'gemini' または 'bedrock' を指定してください"}
    from utils.claude_cli_client import set_incident_fallback_mode
    set_incident_fallback_mode(req.mode)
    return {"mode": req.mode, "message": f"フォールバックモードを {req.mode} に変更しました"}


@app.post("/api/resume")
async def resume(req: ResumeRequest, token: str = ""):
    """
    HITL Go サイン — human_interrupt で停止中のグラフを再開する。

    Request:
      { "thread_id": "abc12345", "response": "GoサインまたはYes/No/テキスト" }
    """
    _check_auth(token)
    router = await _get_router()
    result = await router.resume(
        thread_id=req.thread_id,
        human_response=req.response,
    )
    return result


@app.get("/api/history")
async def history(thread_id: str = "", token: str = ""):
    """指定スレッドの会話履歴を返す"""
    _check_auth(token)
    if not thread_id:
        return {"history": []}
    router = await _get_router()
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = await router._compiled.aget_state(config)
        hist = state.values.get("conversation_history", []) if state else []
        return {"history": hist, "thread_id": thread_id}
    except Exception as e:
        logger.debug("history 取得失敗: %s", e)
        return {"history": [], "thread_id": thread_id}


@app.get("/api/llm-status")
async def llm_status(token: str = ""):
    """ローカルLLM排他制御ステータス"""
    _check_auth(token)
    try:
        from utils.local_model_manager import LocalModelManager
        return await LocalModelManager.get().get_status()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/llm/test")
async def llm_test(req: LLMTestRequest, token: str = ""):
    """ローカルLLMサーバーのプロキシエンドポイント（CORS対応） v16"""
    _check_auth(token)
    try:
        llm_url = os.environ.get("LOCAL_LLM_URL", "")
        endpoint = req.endpoint or "/generate/gemma"
        client = _get_http_client()
        response = await client.post(f"{llm_url}{endpoint}", json=req.body)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        try:
            error_data = e.response.json()
        except Exception:
            error_data = {}
        return JSONResponse(
            {"error": error_data.get("error") or f"LLM server returned {e.response.status_code}"},
            status_code=e.response.status_code,
        )
    except httpx.TimeoutException:
        return JSONResponse({"error": "LLM server timeout"}, status_code=504)
    except Exception as e:
        logger.error("LLM test proxy error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=503)


@app.post("/api/llm/switch/{model}")
async def llm_switch(model: str, token: str = ""):
    """ローカルLLMサーバーのモデル切り替え (Gemma4 ↔ Nemotron) v16"""
    _check_auth(token)
    if model not in ["gemma", "nemotron"]:
        return JSONResponse({"error": "Invalid model. Use 'gemma' or 'nemotron'"}, status_code=400)
    try:
        llm_url = os.environ.get("LOCAL_LLM_URL", "")
        client = _get_http_client()
        response = await client.post(f"{llm_url}/switch/{model}", timeout=60.0)
        response.raise_for_status()
        result = response.json()
        logger.info("LLM model switched to: %s", model)
        if result.get("success"):
            try:
                from utils.local_model_manager import LocalModelManager
                LocalModelManager.get()._active_model = model
            except Exception as sync_err:
                logger.warning("LocalModelManager sync failed (non-fatal): %s", sync_err)
        return result
    except httpx.HTTPStatusError as e:
        try:
            error_data = e.response.json()
        except Exception:
            error_data = {}
        return JSONResponse(
            {"error": error_data.get("error") or f"Switch failed: {e.response.status_code}"},
            status_code=e.response.status_code,
        )
    except httpx.TimeoutException:
        return JSONResponse({"error": "LLM switch timeout"}, status_code=504)
    except Exception as e:
        logger.error("LLM switch error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=503)


@app.get("/metrics")
async def metrics():
    """Prometheus メトリクス エンドポイント"""
    if not _HAS_PROMETHEUS:
        return JSONResponse({"error": "prometheus_client not installed"}, status_code=503)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/health")
async def health():
    """ヘルスチェック"""
    from utils.client_registry import ClientRegistry
    registry = ClientRegistry.get()
    result: dict = {
        "status": "ok",
        "version": "18",
        "roles": registry.available_roles,
    }
    # PostgreSQL 接続状態
    if _router is not None:
        result["postgres"] = _router.pg_status
    else:
        result["postgres"] = {"status": "initializing", "error": "", "saver": "none"}
    # ローカルLLM排他制御ステータス
    try:
        from utils.local_model_manager import LocalModelManager
        llm_status = await LocalModelManager.get().get_status()
        result["local_llm"] = {
            "active_model": llm_status.get("active_model", "?"),
            "lock_held":    llm_status.get("lock_held", False),
            "lock_holder":  llm_status.get("lock_holder", ""),
            "lock_seconds": llm_status.get("lock_seconds", 0),
        }
    except Exception as e:
        logger.debug("local_llm status 取得失敗: %s", e)
        result["local_llm"] = {"active_model": "unknown", "lock_held": False}
    return result


# ── v17: Thread/Message CRUD API (PostgreSQL backed) ────────────────────────

@app.get("/api/threads")
async def get_threads(limit: int = 50, offset: int = 0, token: str = ""):
    """スレッド一覧を取得 (最新順)"""
    _check_auth(token)
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT id, title, created_at, updated_at, tags FROM threads
                       WHERE NOT is_archived
                       ORDER BY updated_at DESC
                       LIMIT %s OFFSET %s""",
                    (limit, offset)
                )
                rows = await cur.fetchall()
                threads = [
                    {
                        "id": row[0],
                        "title": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                        "updated_at": row[3].isoformat() if row[3] else None,
                        "tags": row[4] if row[4] else "[]",
                    }
                    for row in rows
                ]
                return {"threads": threads}
    except Exception as e:
        logger.error(f"get_threads error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/threads")
async def create_thread(title: str = "新しい会話", token: str = ""):
    """新しいスレッドを作成"""
    _check_auth(token)
    try:
        import psycopg
        thread_id = str(uuid.uuid4())
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO threads (id, title) VALUES (%s, %s)""",
                    (thread_id, title)
                )
                return {"thread_id": thread_id, "title": title}
    except Exception as e:
        logger.error(f"create_thread error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.put("/api/threads/{thread_id}")
async def update_thread(thread_id: str, body: ThreadUpdateRequest, token: str = ""):
    """スレッドのタイトルまたはタグを更新"""
    _check_auth(token)
    try:
        import psycopg
        updates = []
        params = []

        if body.title is not None:
            updates.append("title = %s")
            params.append(body.title)
        if body.tags is not None:
            updates.append("tags = %s")
            params.append(body.tags)

        if not updates:
            return {"thread_id": thread_id}

        updates.append("updated_at = NOW()")
        params.append(thread_id)

        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE threads SET {', '.join(updates)} WHERE id = %s",
                    params
                )
        return {"thread_id": thread_id}
    except Exception as e:
        logger.error(f"update_thread error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/threads/all")
async def delete_all_threads(token: str = ""):
    """全スレッドをアーカイブ（論理削除）"""
    _check_auth(token)
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute("UPDATE threads SET is_archived = TRUE WHERE NOT is_archived")
                return {"deleted": cur.rowcount}
    except Exception as e:
        logger.error(f"delete_all_threads error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/settings/roles")
async def get_role_settings(token: str = ""):
    """ロールの有効/無効状態を返す"""
    _check_auth(token)
    from utils.client_registry import ClientRegistry
    all_roles = ClientRegistry.get().available_roles
    return {"roles": {r: r not in _DISABLED_ROLES for r in all_roles}}


class RoleEnabledRequest(BaseModel):
    enabled: bool


@app.post("/api/settings/roles/{role_key}")
async def set_role_enabled(role_key: str, req: RoleEnabledRequest, token: str = ""):
    """ロールの有効/無効を切り替える"""
    _check_auth(token)
    if req.enabled:
        _DISABLED_ROLES.discard(role_key)
    else:
        _DISABLED_ROLES.add(role_key)
    logger.info("ロール %s: %s", role_key, "有効" if req.enabled else "無効")
    return {"role": role_key, "enabled": req.enabled}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str, token: str = ""):
    """スレッドを削除 (アーカイブ)"""
    _check_auth(token)
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """UPDATE threads SET is_archived = TRUE
                       WHERE id = %s""",
                    (thread_id,)
                )
                return {"thread_id": thread_id, "archived": True}
    except Exception as e:
        logger.error(f"delete_thread error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/threads/{thread_id}/messages")
async def get_messages(thread_id: str, limit: int = 100, offset: int = 0, token: str = ""):
    """スレッドのメッセージを取得"""
    _check_auth(token)
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT id, role, content, agent_role, model, execution_time, created_at
                       FROM messages
                       WHERE thread_id = %s
                       ORDER BY created_at ASC
                       LIMIT %s OFFSET %s""",
                    (thread_id, limit, offset)
                )
                rows = await cur.fetchall()
                messages = [
                    {
                        "id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "agent_role": row[3],
                        "model": row[4],
                        "execution_time": row[5],
                        "created_at": row[6].isoformat() if row[6] else None,
                    }
                    for row in rows
                ]
                return {"thread_id": thread_id, "messages": messages}
    except Exception as e:
        logger.error(f"get_messages error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


class MessageRequest(BaseModel):
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    agent_role: Optional[str] = None
    model: Optional[str] = None
    execution_time: Optional[float] = None


@app.post("/api/threads/{thread_id}/messages")
async def save_message(thread_id: str, msg: MessageRequest, token: str = ""):
    """メッセージをスレッドに保存"""
    _check_auth(token)
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO messages (thread_id, role, content, agent_role, model, execution_time)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (thread_id, msg.role, msg.content, msg.agent_role, msg.model, msg.execution_time)
                )
                return {"status": "saved", "thread_id": thread_id}
    except Exception as e:
        logger.error(f"save_message error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


class RewindRequest(BaseModel):
    checkpoint_id: str
    new_message: Optional[str] = None
    new_role: Optional[str] = None


@app.get("/api/threads/{thread_id}/checkpoints")
async def get_checkpoints(thread_id: str, token: str = ""):
    """LangGraph チェックポイント履歴を返す（タイムトラベル用）"""
    _check_auth(token)
    try:
        router = await _get_router()
        checkpointer = getattr(router, "_checkpointer", None)
        if checkpointer is None:
            return JSONResponse({"error": "Checkpointer not available"}, status_code=503)

        config = {"configurable": {"thread_id": thread_id}}
        history = []
        async for state in router._compiled.aget_state_history(config):
            ts = state.created_at if hasattr(state, "created_at") else None
            checkpoint_id = None
            if state.config and "configurable" in state.config:
                checkpoint_id = state.config["configurable"].get("checkpoint_id")
            vals = state.values or {}
            history.append({
                "checkpoint_id": checkpoint_id,
                "created_at": str(ts) if ts else None,
                "message": vals.get("message", ""),
                "response": vals.get("response", ""),
                "handled_by": vals.get("handled_by", ""),
                "agent_role": vals.get("agent_role", ""),
            })
        return {"thread_id": thread_id, "checkpoints": history}
    except Exception as e:
        logger.error(f"get_checkpoints error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/threads/{thread_id}/rewind")
async def rewind_thread(thread_id: str, req: RewindRequest, token: str = ""):
    """指定チェックポイントに巻き戻して再実行（タイムトラベル）"""
    _check_auth(token)
    try:
        router = await _get_router()
        if router._compiled is None:
            return JSONResponse({"error": "Router not initialized"}, status_code=503)

        # チェックポイントを指定して状態を復元
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": req.checkpoint_id,
            }
        }
        state = await router._compiled.aget_state(config)
        if state is None:
            return JSONResponse({"error": "Checkpoint not found"}, status_code=404)

        # 復元した状態から再実行するメッセージを決定
        rerun_message = req.new_message or state.values.get("message", "")
        rerun_role = req.new_role or state.values.get("agent_role", "auto")

        # 新しいスレッドIDで再実行（元スレッドは保持）
        new_thread_id = f"{thread_id}_rw_{req.checkpoint_id[:8]}"
        new_config = {"configurable": {"thread_id": new_thread_id}}

        result_response = ""
        result_handled_by = ""
        async for event in router._compiled.astream_events(
            {"message": rerun_message, "forced_role": rerun_role if rerun_role != "auto" else None},
            config=new_config,
            version="v2",
        ):
            if event.get("event") == "on_chain_end":
                output = event.get("data", {}).get("output") or {}
                if isinstance(output, dict) and output.get("response"):
                    result_response = output["response"]
                    result_handled_by = output.get("handled_by", "")

        return {
            "thread_id": new_thread_id,
            "original_thread_id": thread_id,
            "checkpoint_id": req.checkpoint_id,
            "response": result_response,
            "handled_by": result_handled_by,
        }
    except Exception as e:
        logger.error(f"rewind_thread error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


class SandboxRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: float = 10.0


@app.post("/api/sandbox/run")
async def sandbox_run(req: SandboxRequest, token: str = ""):
    """Python コードをサンドボックスで安全に実行"""
    _check_auth(token)
    try:
        from utils.code_sandbox import run_code
        if req.timeout > 30:
            req.timeout = 30.0
        result = await run_code(req.code, timeout=req.timeout, language=req.language)
        return result
    except Exception as e:
        logger.error(f"sandbox_run error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """WebSocket チャット"""
    token = ws.query_params.get("token", "")
    from console.auth import validate_session
    if not validate_session(token):
        await ws.close(code=4001, reason="Unauthorized")
        return
    await ws.accept()
    if _HAS_PROMETHEUS:
        _active_ws.inc()
    router = await _get_router()
    thread_id = str(uuid.uuid4())[:8]

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "message")

            if data.get("thread_id"):
                thread_id = data["thread_id"]

            # ── HITL 再開リクエスト ─────────────────────────────────
            if msg_type == "hitl_resume":
                human_response = data.get("response", "")
                await ws.send_json({"type": "thinking", "thread_id": thread_id})
                t0 = time.time()
                try:
                    result = await router.resume(
                        thread_id=thread_id,
                        human_response=human_response,
                    )
                except Exception:
                    if _HAS_PROMETHEUS:
                        _req_errors.labels(source="console_ws").inc()
                    raise
                await _send_result(ws, result, thread_id, t0)
                continue

            # ── 通常メッセージ ──────────────────────────────────────
            message  = data.get("message", "")
            model    = data.get("model", "auto")
            streaming = data.get("stream", True)  # デフォルトでストリーミング有効

            await ws.send_json({"type": "thinking", "thread_id": thread_id})
            t0 = time.time()

            # ── 全員会議モード ──────────────────────────────────────
            if model == "all":
                await _broadcast_message(ws, router, message, thread_id, t0)
                continue

            # 無効化されたロールは auto にフォールバック
            if model in _DISABLED_ROLES:
                model = "auto"
            forced_role = model if model != "auto" else None

            # ── ストリーミングモード ─────────────────────────────────
            if streaming and hasattr(router, "stream_message"):
                try:
                    # ストリーミング全体にタイムアウト (300秒)
                    async def stream_with_timeout():
                        try:
                            stream_gen = router.stream_message(
                                message=message,
                                thread_id=thread_id,
                                source="console_ws",
                                forced_role=forced_role,
                            )
                            # asyncio.timeout() は async generator に対しても正しく動作する
                            async with asyncio.timeout(300):
                                async for chunk in stream_gen:
                                    yield chunk
                        except asyncio.TimeoutError:
                            logger.warning("⏱️ WebSocket stream_message timeout: thread=%s", thread_id[:8])
                            yield {
                                "type": "error",
                                "message": "ストリーミング処理がタイムアウトしました (300秒)",
                            }

                    async for chunk in stream_with_timeout():
                        await ws.send_json({**chunk, "thread_id": thread_id})
                except Exception as e:
                    if _HAS_PROMETHEUS:
                        _req_errors.labels(source="console_ws").inc()
                    await ws.send_json({"type": "error", "message": str(e), "thread_id": thread_id})
                continue

            # ── 非ストリーミング (フォールバック) ───────────────────
            try:
                # ── タイムアウト: 180秒 ──
                result = await asyncio.wait_for(
                    router.process_message(
                        message=message,
                        thread_id=thread_id,
                        source="console_ws",
                        forced_role=forced_role,
                    ),
                    timeout=180
                )
            except asyncio.TimeoutError:
                logger.warning("⏱️ WebSocket process_message timeout: thread=%s", thread_id[:8])
                await ws.send_json({
                    "type": "error",
                    "message": "処理がタイムアウトしました (180秒)",
                    "thread_id": thread_id,
                })
                continue
            except Exception:
                if _HAS_PROMETHEUS:
                    _req_errors.labels(source="console_ws").inc()
                raise

            await _send_result(ws, result, thread_id, t0)

    except WebSocketDisconnect:
        logger.debug("WebSocket 切断: thread=%s", thread_id)
    except Exception as e:
        logger.error("WebSocket エラー: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception as send_err:
            logger.debug("WS error送信失敗: %s", send_err)
    finally:
        if _HAS_PROMETHEUS:
            _active_ws.dec()


async def _send_result(ws: WebSocket, result: dict, thread_id: str, t0: float):
    """LangGraph 結果を WebSocket に送信。HITL の場合は hitl_question を送出。"""
    route = result.get("route", "unknown")
    elapsed = time.time() - t0
    if _HAS_PROMETHEUS:
        _req_total.labels(source="console_ws", route=route).inc()
        _req_duration.labels(route=route).observe(elapsed)

    # HITL 中断検出: dialog_status か human_question で判定
    dialog_status  = result.get("dialog_status", "")
    human_question = result.get("human_question", "")
    if dialog_status == "waiting_for_human" or human_question:
        await ws.send_json({
            "type":          "hitl_question",
            "question":      human_question or "確認が必要です。続行しますか？",
            "agent_role":    result.get("agent_role", ""),
            "thread_id":     thread_id,
            "execution_time": result.get("execution_time", elapsed),
        })
        return

    await ws.send_json({
        "type":           "response",
        "response":       result.get("response", ""),
        "agent_role":     result.get("agent_role", ""),
        "handled_by":     result.get("handled_by", ""),
        "route":          route,
        "execution_time": result.get("execution_time", elapsed),
        "thread_id":      thread_id,
    })


async def _broadcast_message(ws: WebSocket, router, message: str, thread_id: str, t0: float):
    """全役職に並列送信し、完了した順に WebSocket へ送信する"""
    BROADCAST_ROLES = [
        "daigensui", "shogun", "gunshi", "sanbo", "gaiji",
        "metsuke", "kengyo", "yuhitsu", "onmitsu",
    ]

    async def run_one(role: str):
        try:
            # ── ロール別タイムアウト: 通常 120秒 ──
            result = await asyncio.wait_for(
                router.process_message(
                    message=message,
                    thread_id=f"broadcast_{role}",
                    source="console_ws_broadcast",
                    forced_role=role,
                ),
                timeout=120
            )
            return role, result
        except asyncio.TimeoutError:
            logger.warning("⏱️ broadcast timeout: role=%s", role)
            return role, {
                "status": "failed",
                "error": "タイムアウト",
                "response": f"❌ {role} のタイムアウト (120秒)",
                "agent_role": role,
                "route": "broadcast",
                "execution_time": 120.0,
            }
        except Exception as e:
            logger.error("broadcast error for %s: %s", role, e)
            return role, {
                "status": "failed",
                "error": str(e),
                "response": f"❌ {role} エラー: {e}",
                "agent_role": role,
                "route": "broadcast",
                "execution_time": 0,
            }

    tasks = [asyncio.create_task(run_one(r)) for r in BROADCAST_ROLES]
    completed = 0
    failed = 0
    for fut in asyncio.as_completed(tasks):
        try:
            role, result = await fut
            await _send_result(ws, result, thread_id, t0)
            if result.get("status") == "completed":
                completed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error("broadcast error: %s", e)
            failed += 1

    await ws.send_json({
        "type":      "system",
        "message":   f"🏯 全員会議完了 (成功{completed}名/{failed}名失敗、計{completed+failed}/{len(BROADCAST_ROLES)})",
        "thread_id": thread_id,
    })


# ── v18: YAML 監査ログ API ───────────────────────────────────────────────

@app.get("/api/audit-logs")
async def get_audit_logs(days: int = 7, token: str = ""):
    """直近 N 日分の YAML 監査ログ一覧を返す"""
    _check_auth(token)
    try:
        from utils.audit_log import list_audit_logs
        logs = list_audit_logs(days=days)
        return {"logs": logs, "count": len(logs)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/audit-logs/content")
async def get_audit_log_content(path: str, token: str = ""):
    """指定パスの YAML ファイル内容を返す"""
    _check_auth(token)
    import os as _os
    try:
        from utils.audit_log import _AUDIT_DIR
        # パストラバーサル防止: audit ディレクトリ以下のみ許可
        abs_path = _os.path.realpath(path)
        if not Path(abs_path).is_relative_to(_AUDIT_DIR.resolve()):
            raise HTTPException(status_code=403, detail="アクセス拒否")
        if not _os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="ファイルが見つかりません")
        content = Path(abs_path).read_text(encoding="utf-8")
        return {"path": path, "content": content}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── v18: スキル提案 API ──────────────────────────────────────────────────

@app.get("/api/skill-proposals")
async def get_skill_proposals(status: str = "pending", token: str = ""):
    """スキル候補一覧を返す (status: pending | approved | dismissed)"""
    _check_auth(token)
    try:
        from utils.skill_tracker import list_proposals
        proposals = await list_proposals(status=status)
        return {"proposals": proposals, "count": len(proposals)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


class SkillApproveRequest(BaseModel):
    skill_name: str
    route_hint: str
    system_prompt_hint: str = ""


@app.post("/api/skill-proposals/{proposal_id}/approve")
async def approve_skill_proposal(proposal_id: str, req: SkillApproveRequest, token: str = ""):
    """スキル候補を承認し、skills/*.yaml に書き出す"""
    _check_auth(token)
    try:
        from utils.skill_tracker import approve_proposal
        result = await approve_proposal(
            proposal_id=proposal_id,
            skill_name=req.skill_name,
            route_hint=req.route_hint,
            system_prompt_hint=req.system_prompt_hint,
        )
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/skill-proposals/{proposal_id}/dismiss")
async def dismiss_skill_proposal(proposal_id: str, token: str = ""):
    """スキル候補を却下する"""
    _check_auth(token)
    try:
        from utils.skill_tracker import dismiss_proposal
        result = await dismiss_proposal(proposal_id)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/skills")
async def get_approved_skills(token: str = ""):
    """承認済みスキル一覧を返す"""
    _check_auth(token)
    try:
        from utils.skill_tracker import list_proposals
        skills = await list_proposals(status="approved")
        return {"skills": skills, "count": len(skills)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 進化提案 API ─────────────────────────────────────────────────────────────

@app.get("/api/evolution-proposals")
async def get_evolution_proposals(status: str = "pending", token: str = ""):
    """週次進化サイクルで生成された型付き提案書一覧"""
    _check_auth(token)
    try:
        from utils.evolution_proposals import list_evolution_proposals
        proposals = await list_evolution_proposals(status=status)
        return {"proposals": proposals, "count": len(proposals)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/evolution-proposals/{proposal_id}/approve")
async def approve_evolution_proposal(proposal_id: str, token: str = ""):
    """進化提案を承認して適用する"""
    _check_auth(token)
    try:
        from utils.evolution_proposals import approve_evolution_proposal as _approve
        result = await _approve(proposal_id)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/evolution-proposals/{proposal_id}/dismiss")
async def dismiss_evolution_proposal(proposal_id: str, token: str = ""):
    """進化提案を却下する"""
    _check_auth(token)
    try:
        from utils.evolution_proposals import dismiss_evolution_proposal as _dismiss
        result = await _dismiss(proposal_id)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/evolution-reports/latest")
async def get_latest_evolution_report(token: str = ""):
    """最新の週次レポートを返す"""
    _check_auth(token)
    from pathlib import Path
    latest = Path.home() / "bushidan_reports" / "evolution" / "latest.txt"
    if not latest.exists():
        return JSONResponse({"error": "レポートがまだありません"}, status_code=404)
    return {"report": latest.read_text(encoding="utf-8")}


# ── v18: KPI サマリーメトリクス API ─────────────────────────────────────────

@app.get("/api/metrics/summary")
async def get_metrics_summary(token: str = ""):
    """直近24時間のKPIサマリーを返す（ヘッダースパークライン用）"""
    _check_auth(token)
    if not POSTGRES_URL:
        return JSONResponse({"error": "POSTGRES_URL未設定"}, status_code=503)
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
            async with conn.cursor() as cur:
                # 直近24時間の観測データが存在するか確認
                await cur.execute(
                    "SELECT to_regclass('public.skill_observations')"
                )
                if not (await cur.fetchone())[0]:
                    return {"available": False}

                await cur.execute("""
                    SELECT
                        COUNT(*)                                                 AS total,
                        ROUND(AVG(CASE WHEN success THEN 1.0 ELSE 0 END) * 100, 1) AS success_rate,
                        ROUND(AVG(execution_time)::numeric, 0)                  AS avg_ms,
                        SUM(CASE WHEN used_fallback THEN 1 ELSE 0 END)          AS fallback_count,
                        SUM(CASE WHEN NOT COALESCE(success, TRUE) THEN 1 ELSE 0 END) AS error_count
                    FROM skill_observations
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                row = await cur.fetchone()
                if not row or not row[0]:
                    return {"available": True, "total": 0, "success_rate": None,
                            "avg_ms": None, "fallback_count": 0, "error_count": 0}

                # 直近6時間のロール別集計（スパークライン用）
                await cur.execute("""
                    SELECT role_used, COUNT(*) as cnt
                    FROM skill_observations
                    WHERE created_at >= NOW() - INTERVAL '6 hours'
                    GROUP BY role_used
                    ORDER BY cnt DESC
                    LIMIT 5
                """)
                top_roles = [{"role": r[0], "count": r[1]} for r in await cur.fetchall()]

        return {
            "available":    True,
            "total":        row[0],
            "success_rate": float(row[1]) if row[1] is not None else None,
            "avg_ms":       float(row[2]) if row[2] is not None else None,
            "fallback_count": int(row[3] or 0),
            "error_count":  int(row[4] or 0),
            "top_roles":    top_roles,
        }
    except Exception as e:
        logger.debug("metrics/summary error: %s", e)
        return {"available": False, "error": str(e)}


# ── v18 監査ログヘルパー ──────────────────────────────────────────────────

async def _write_v18_audit(**kwargs) -> None:
    """v18 監査ログをサイレントに書き込む"""
    try:
        from core.audit import write_pipeline_audit
        await write_pipeline_audit(**kwargs)
    except Exception as e:
        logger.debug("v18 audit write failed: %s", e)


# ── v18: Phase 1 パイプライン ─────────────────────────────────────────────

# v18 Phase 1 プロセッサ (遅延初期化)
_v18_karasu: Optional[object] = None
_v18_uchu: Optional[object] = None
_v18_notion: Optional[object] = None
_v18_router_proc: Optional[object] = None
_v18_init_lock = asyncio.Lock()


async def _get_v18_processors():
    global _v18_karasu, _v18_uchu, _v18_notion, _v18_router_proc
    if _v18_karasu is not None:
        return _v18_karasu, _v18_uchu, _v18_notion, _v18_router_proc
    async with _v18_init_lock:
        if _v18_karasu is not None:
            return _v18_karasu, _v18_uchu, _v18_notion, _v18_router_proc
        from core.processors import KarasuProcessor, UchuProcessor, NotionSearchProcessor, RouteDecisionProcessor
        _v18_karasu = KarasuProcessor()
        _v18_uchu = UchuProcessor()
        _v18_notion = NotionSearchProcessor()
        _v18_router_proc = RouteDecisionProcessor()
        logger.info("✅ v18 Phase 1 プロセッサ初期化完了")
        return _v18_karasu, _v18_uchu, _v18_notion, _v18_router_proc


async def _run_phase1_pipeline(message: str):
    """Phase 1 パイプライン実行: Karasu → Uchu → NotionSearch → RouteDecision"""
    karasu_p, uchu_p, notion_p, route_p = await _get_v18_processors()

    # 並列実行: Karasu (外部検索) と Uchu (intent 分類) を同時に
    from core.models.notion_index import SkillSearchQuery
    from core.models.routing import RoutingInput

    karasu_task = asyncio.create_task(karasu_p.process(message))
    uchu_task = asyncio.create_task(uchu_p.process(message))

    karasu_out, uchu_out = await asyncio.gather(karasu_task, uchu_task)

    # Uchu に Karasu の結果を注入してリキャリブレーション（外部知識があれば）
    if karasu_out.search_results and not uchu_out.requires_external_knowledge:
        uchu_out_refined = await uchu_p.process(message, karasu=karasu_out)
        # 信頼度が上がった場合のみ採用
        if uchu_out_refined.confidence > uchu_out.confidence:
            uchu_out = uchu_out_refined

    # NotionSearch: スキル DB から関連スキルを検索
    skill_query = SkillSearchQuery.from_uchu(uchu_out)
    notion_out = await notion_p.process(skill_query)

    # RouteDecision: デシジョンツリー
    routing_input = RoutingInput(
        user_input=message,
        intent=uchu_out,
        search=notion_out,
        karasu=karasu_out,
    )
    route_decision = await route_p.aprocess(routing_input)

    return karasu_out, uchu_out, notion_out, route_decision


class V2ChatRequest(BaseModel):
    message: str
    model: str = "auto"
    thread_id: str = ""
    use_phase1: bool = True  # False にすると v17 互換モード


class V2ChatResponse(BaseModel):
    response: str
    agent_role: str
    handled_by: str
    route: str
    execution_time: float
    thread_id: str
    # Phase 1 メタデータ
    complexity: str = ""
    intent_type: str = ""
    suggested_role: str = ""
    notion_score: float = 0.0
    karasu_results: int = 0
    pipeline_ms: float = 0.0


@app.post("/api/v2/chat", response_model=V2ChatResponse)
async def v2_chat(req: V2ChatRequest, token: str = ""):
    """v18 Phase 1 パイプライン統合チャット"""
    _check_auth(token)
    thread_id = req.thread_id or str(uuid.uuid4())[:8]
    t0 = time.time()

    # Phase 1 パイプライン実行
    forced_role = None
    complexity = ""
    intent_type = ""
    suggested_role = ""
    notion_score = 0.0
    karasu_results = 0
    pipeline_ms = 0.0

    if req.use_phase1 and req.model == "auto":
        try:
            karasu_out, uchu_out, notion_out, route_decision = await _run_phase1_pipeline(req.message)
            pipeline_ms = route_decision.processing_time_ms
            # Enum の場合は .value で str に変換
            complexity = route_decision.complexity.value if hasattr(route_decision.complexity, "value") else str(route_decision.complexity)
            intent_type = route_decision.intent_type.value if hasattr(route_decision.intent_type, "value") else str(route_decision.intent_type)
            suggested_role = route_decision.selected_role
            notion_score = notion_out.relevance_score
            karasu_results = len(karasu_out.search_results)

            # Phase 1 が決定したロールをそのまま使用
            if route_decision.selected_role != "auto":
                forced_role = route_decision.selected_role

            logger.info(
                "v18 Phase1: complexity=%s intent=%s role=%s (conf=%.2f) notion=%.3f karasu=%d",
                complexity, intent_type, suggested_role,
                route_decision.confidence, notion_score, karasu_results,
            )
        except Exception as e:
            logger.warning("v18 Phase1 pipeline error (fallback to v17): %s", e)
    else:
        # 明示的ロール指定または Phase 1 無効
        effective_role = req.model if req.model not in _DISABLED_ROLES else "auto"
        forced_role = effective_role if effective_role != "auto" else None

    # LangGraph ルーター実行
    router = await _get_router()
    try:
        result = await router.process_message(
            message=req.message,
            thread_id=thread_id,
            source="console_v2",
            forced_role=forced_role,
        )
    except Exception:
        if _HAS_PROMETHEUS:
            _req_errors.labels(source="console_v2").inc()
        raise

    route = result.get("route", "unknown")
    elapsed = time.time() - t0
    if _HAS_PROMETHEUS:
        _req_total.labels(source="console_v2", route=route).inc()
        _req_duration.labels(route=route).observe(elapsed)

    final_response = result.get("response", "")
    final_agent_role = result.get("agent_role", "")

    # v18 監査ログ書き込み（非同期・ノンブロッキング）
    _fire(
        _write_v18_audit(
            thread_id=thread_id,
            user_input=req.message,
            agent_role=final_agent_role,
            response=final_response,
            execution_time_ms=(time.time() - t0) * 1000,
            pipeline_ms=pipeline_ms,
            stage=route,
            complexity=complexity,
            intent_type=intent_type,
            selected_role=suggested_role,
            notion_score=notion_score,
            karasu_results=karasu_results,
        ),
        name="audit_write",
    )

    return V2ChatResponse(
        response=final_response,
        agent_role=final_agent_role,
        handled_by=result.get("handled_by", ""),
        route=route,
        execution_time=result.get("execution_time", elapsed),
        thread_id=thread_id,
        complexity=complexity,
        intent_type=intent_type,
        suggested_role=suggested_role,
        notion_score=notion_score,
        karasu_results=karasu_results,
        pipeline_ms=pipeline_ms,
    )


@app.get("/api/v18/cache/status")
async def v18_cache_status(token: str = ""):
    """v18 キャッシュバックエンドのステータス"""
    _check_auth(token)
    try:
        from utils.cache_manager import CacheManager
        return await CacheManager.instance().get_status()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/v18/evolve")
async def v18_evolve_skills(days: int = 7, token: str = ""):
    """スキル自動進化サイクルを手動実行"""
    _check_auth(token)
    try:
        from core.skill import evolve_skills_from_audit
        result = await evolve_skills_from_audit(days=days)
        return result
    except Exception as e:
        logger.error("v18 evolve error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/v18/audit-logs")
async def v18_audit_logs(days: int = 3, token: str = ""):
    """v18 時刻別 YAML 監査ログ一覧"""
    _check_auth(token)
    try:
        from core.audit import AuditLogger
        logs = AuditLogger.get().list_logs(days=days)
        return {"logs": logs, "count": len(logs)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/v2/pipeline/analyze")
async def v2_pipeline_analyze(req: V2ChatRequest, token: str = ""):
    """Phase 1 パイプライン診断 — LangGraph を呼ばずに分析結果のみ返す"""
    _check_auth(token)
    try:
        karasu_out, uchu_out, notion_out, route_decision = await _run_phase1_pipeline(req.message)
        return {
            "karasu": {
                "keywords": karasu_out.keywords,
                "search_results": len(karasu_out.search_results),
                "search_query": karasu_out.search_query,
                "fallback_used": karasu_out.fallback_used,
                "confidence": karasu_out.confidence,
                "reasoning": karasu_out.search_reasoning,
                "processing_ms": karasu_out.processing_time_ms,
            },
            "uchu": {
                "complexity": uchu_out.complexity.value,
                "intent_type": uchu_out.intent_type.value,
                "has_image": uchu_out.has_image,
                "has_secret_keyword": uchu_out.has_secret_keyword,
                "primary_topic": uchu_out.primary_topic,
                "sub_topics": uchu_out.sub_topics,
                "requires_external_knowledge": uchu_out.requires_external_knowledge,
                "requires_code_execution": uchu_out.requires_code_execution,
                "suggested_role": uchu_out.suggested_role,
                "confidence": uchu_out.confidence,
                "reasoning": uchu_out.reasoning,
            },
            "notion": {
                "suggested_role": notion_out.suggested_role,
                "relevance_score": notion_out.relevance_score,
                "results_count": len(notion_out.results),
                "top_skills": [
                    {
                        "name": s.name,
                        "role": s.recommended_role,
                        "score": round(s.relevance_score, 4),
                    }
                    for s in notion_out.results[:3]
                ],
                "cache_hit": notion_out.cache_hit,
                "reasoning": notion_out.search_reasoning,
                "processing_ms": notion_out.processing_time_ms,
            },
            "routing": {
                "selected_role": route_decision.selected_role,
                "confidence": route_decision.confidence,
                "fallback_role": route_decision.fallback_role,
                "decision_path": route_decision.decision_tree_path,
                "reasoning": route_decision.decision_reasoning,
                "content_type": route_decision.content_type,
                "processing_ms": route_decision.processing_time_ms,
            },
        }
    except Exception as e:
        logger.error("v2 pipeline analyze error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════════
# メンテナンス API
# ═══════════════════════════════════════════════════════════════

class ServiceActionRequest(BaseModel):
    service: str
    action:  str  # start / stop / restart / status

class PackageUpgradeRequest(BaseModel):
    package: str

class ConflictFixRequest(BaseModel):
    conflicts: str  # pip check の出力テキスト

class ApplyFixRequest(BaseModel):
    packages: list[str]  # ["name==version", ...]


@app.get("/api/maintenance/logs")
async def maintenance_logs(
    source: str = "console",
    lines: int = 300,
    date: str = "",
    token: str = "",
):
    """ログ取得 (source: console / audit / maintenance / journal-{service})"""
    _check_auth(token)
    from console.maintenance import (
        get_log_console, get_log_maintenance, get_audit_log, get_log_journal,
    )
    if source == "console":
        text = await asyncio.to_thread(get_log_console, lines)
    elif source == "audit":
        text = await asyncio.to_thread(get_audit_log, date or None)
    elif source == "maintenance":
        text = await asyncio.to_thread(get_log_maintenance, lines)
    elif source.startswith("journal-"):
        text = await asyncio.to_thread(get_log_journal, source[len("journal-"):], lines)
    else:
        raise HTTPException(status_code=400, detail=f"不明なソース: {source}")
    return JSONResponse({"text": text, "source": source})


@app.get("/api/maintenance/audit-dates")
async def maintenance_audit_dates(token: str = ""):
    """監査ログが存在する日付一覧"""
    _check_auth(token)
    from console.maintenance import get_audit_dates
    return JSONResponse({"dates": await asyncio.to_thread(get_audit_dates)})


@app.get("/api/maintenance/system")
async def maintenance_system(token: str = ""):
    """システム情報 (CPU・メモリ・ディスク・サービス・git)"""
    _check_auth(token)
    from console.maintenance import get_system_info
    return JSONResponse(await asyncio.to_thread(get_system_info))


@app.get("/api/maintenance/packages")
async def maintenance_packages(token: str = ""):
    """インストール済みパッケージ一覧"""
    _check_auth(token)
    from console.maintenance import get_packages
    return JSONResponse({"packages": await asyncio.to_thread(get_packages)})


@app.get("/api/maintenance/packages/outdated")
async def maintenance_packages_outdated(token: str = ""):
    """更新可能なパッケージ一覧 (時間がかかる場合あり)"""
    _check_auth(token)
    from console.maintenance import get_outdated_packages
    pkgs = await get_outdated_packages()
    return JSONResponse({"outdated": pkgs})


@app.post("/api/maintenance/packages/upgrade")
async def maintenance_package_upgrade(req: PackageUpgradeRequest, token: str = ""):
    """単一パッケージをアップグレード"""
    _check_auth(token)
    from console.maintenance import upgrade_package
    result = await upgrade_package(req.package)
    return JSONResponse(result)


@app.post("/api/maintenance/packages/suggest-fix")
async def maintenance_suggest_fix(req: ConflictFixRequest, token: str = ""):
    """依存競合を LLM (Mistral Small) に分析させて解決案を返す"""
    _check_auth(token)
    if not req.conflicts.strip():
        return JSONResponse({"packages": [], "explanation": "競合なし"})
    try:
        from utils.client_registry import ClientRegistry
        client = ClientRegistry.get().get_client("metsuke")
        if not client:
            return JSONResponse({"error": "LLMクライアント未初期化"}, status_code=503)

        prompt = (
            "以下は `pip check` の出力です。依存関係の競合を解決するために "
            "インストールすべきパッケージのバージョンを JSON で返してください。\n\n"
            f"```\n{req.conflicts[:2000]}\n```\n\n"
            "返答は必ず次の JSON **のみ** で返してください。余分なテキスト不要。\n"
            '{"packages": ["package_name==x.y.z", ...], "explanation": "1行の説明"}\n\n'
            "注意:\n"
            "- 競合を起こしているパッケージの **どちらか** を互換バージョンに下げる提案を優先\n"
            "- インストール済みパッケージ名と完全一致させること (例: pydantic-core ではなく pydantic_core)\n"
            "- 解決不可能な場合は packages を空配列にして explanation に理由を書く"
        )
        raw = await client.generate(
            messages=[{"role": "user", "content": prompt}],
            system="あなたは Python パッケージ依存関係の専門家です。JSON のみ返してください。",
            max_tokens=300,
        )
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return JSONResponse({"error": "LLM応答のパース失敗", "raw": raw[:200]}, status_code=500)
        result = json.loads(m.group())
        # パッケージ名のサニタイズ: name==version 形式のみ許可
        _valid = re.compile(r'^[a-zA-Z0-9_\-\.]+==[\d\w\.\-\+]+$')
        result["packages"] = [p for p in result.get("packages", []) if _valid.match(p)]
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/maintenance/packages/apply-fix")
async def maintenance_apply_fix(req: ApplyFixRequest, token: str = ""):
    """LLM 提案のパッケージをバリデーション後にインストール"""
    _check_auth(token)
    _valid = __import__('re').compile(r'^[a-zA-Z0-9_\-\.]+==[\d\w\.\-\+]+$')
    invalid = [p for p in req.packages if not _valid.match(p)]
    if invalid:
        return JSONResponse({"success": False, "output": f"無効なパッケージ指定: {invalid}"})
    from console.maintenance import upgrade_package_list
    result = await upgrade_package_list(req.packages)
    return JSONResponse(result)


@app.get("/api/maintenance/update/stream")
async def maintenance_update_stream(stage: str = "check", token: str = ""):
    """
    アップデートをストリーミング実行 (SSE text/event-stream)
    stage: check / sandbox / apply
    """
    _check_auth(token)
    if stage not in ("check", "sandbox", "apply"):
        raise HTTPException(status_code=400, detail="stage は check / sandbox / apply のいずれか")
    from console.maintenance import stream_update

    async def _gen():
        async for chunk in stream_update(stage):
            yield chunk
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/maintenance/service")
async def maintenance_service(req: ServiceActionRequest, token: str = ""):
    """サービス管理 (start / stop / restart / status)"""
    _check_auth(token)
    from console.maintenance import service_action
    result = await service_action(req.service, req.action)
    return JSONResponse(result)


# ── スキル自動進化 設定 API ──────────────────────────────────────────────

class EvolutionAutoConfig(BaseModel):
    enabled: bool
    interval_hours: int = 24
    min_observations: int = 10


@app.get("/api/v18/evolution-auto-config")
async def get_evolution_auto_config(token: str = ""):
    """スキル自動進化の現在設定と状態を返す"""
    _check_auth(token)
    cfg = await _get_evolution_config()
    should_run, reason = await _should_run_evolution(cfg)
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT run_at, candidates_created, skills_activated FROM skill_evolution_log ORDER BY run_at DESC LIMIT 1"
                )
                row = await cur.fetchone()
                last_run_info = {
                    "run_at": row[0].isoformat() if row and row[0] else None,
                    "candidates_created": row[1] if row else 0,
                    "skills_activated": row[2] if row else 0,
                } if row else None
    except Exception as e:
        last_run_info = None
        logger.warning("evolution config status error: %s", e)
    return {
        "config": cfg,
        "status": {"will_run_next_check": should_run, "reason": reason},
        "last_run": last_run_info,
    }


@app.post("/api/v18/evolution-auto-config")
async def update_evolution_auto_config(req: EvolutionAutoConfig, token: str = ""):
    """スキル自動進化の設定を更新する"""
    _check_auth(token)
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                updates = {
                    "skill_evolution_auto_enabled": str(req.enabled).lower(),
                    "skill_evolution_interval_hours": str(req.interval_hours),
                    "skill_evolution_min_observations": str(req.min_observations),
                }
                for key, value in updates.items():
                    await cur.execute("""
                        INSERT INTO system_config (key, value, updated_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                    """, (key, value))
        logger.info("🧬 スキル自動進化設定更新: enabled=%s interval=%dh min_obs=%d",
                    req.enabled, req.interval_hours, req.min_observations)
        return {"success": True, "config": req.dict()}
    except Exception as e:
        logger.error("evolution config update error: %s", e)
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.on_event("shutdown")
async def _shutdown_http_client():
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        await _HTTP_CLIENT.aclose()
        _HTTP_CLIENT = None
