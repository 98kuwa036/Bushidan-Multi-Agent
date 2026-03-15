"""
console/app.py — 武士団モバイルコンソール FastAPI サーバー v14

エンドポイント:
  POST /api/login          — ログイン (CONSOLE_PASSWORD)
  POST /api/chat           — チャット (同期)
  GET  /api/models         — 利用可能モデル一覧
  GET  /api/history        — 会話履歴
  GET  /api/health         — ヘルスチェック
  WS   /ws/chat            — WebSocket チャット

起動:
  uvicorn console.app:app --host 0.0.0.0 --port 8067
"""

import asyncio
import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="武士団コンソール", version="14")

# ── 静的ファイル ────────────────────────────────────────────────────────
import os
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

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
        logger.info("🔗 Console: LangGraph Router v14 初期化完了")
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
    return JSONResponse({"message": "武士団コンソール v14"})


@app.post("/api/login")
async def login(req: LoginRequest):
    """セッション認証"""
    from console.auth import check_password, create_session
    if not check_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    token = create_session()
    return {"token": token}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """同期チャット"""
    router = await _get_router()
    forced_role = req.model if req.model != "auto" else None
    thread_id = req.thread_id or str(uuid.uuid4())[:8]

    result = await router.process_message(
        message=req.message,
        thread_id=thread_id,
        source="console",
        forced_role=forced_role,
    )

    return ChatResponse(
        response=result.get("response", ""),
        agent_role=result.get("agent_role", ""),
        handled_by=result.get("handled_by", ""),
        route=result.get("route", ""),
        execution_time=result.get("execution_time", 0),
        thread_id=thread_id,
    )


@app.get("/api/models")
async def models():
    """利用可能モデル一覧"""
    from utils.client_registry import ClientRegistry
    registry = ClientRegistry.get()
    model_list = [
        {"key": "auto", "name": "自動ルーティング", "emoji": "🔀"},
        {"key": "daigensui", "name": "大元帥 (Claude Opus 4.6)", "emoji": "⚔️"},
        {"key": "shogun", "name": "将軍 (Claude Sonnet 4.6)", "emoji": "🏯"},
        {"key": "gunshi", "name": "軍師 (o3-mini)", "emoji": "📜"},
        {"key": "sanbo", "name": "参謀 (Mistral Large 3)", "emoji": "🗡️"},
        {"key": "gaiji", "name": "外事 (Command R+)", "emoji": "🌐"},
        {"key": "uketuke", "name": "受付 (Command R)", "emoji": "🚪"},
        {"key": "seppou", "name": "斥候 (Llama 3.3 Groq)", "emoji": "🏹"},
        {"key": "kengyo", "name": "検校 (Gemini Vision)", "emoji": "👁️"},
        {"key": "yuhitsu", "name": "右筆 (ELYZA)", "emoji": "🖊️"},
        {"key": "onmitsu", "name": "隠密 (Nemotron Local)", "emoji": "🥷"},
    ]
    return {"models": model_list}


@app.get("/api/health")
async def health():
    """ヘルスチェック"""
    from utils.client_registry import ClientRegistry
    registry = ClientRegistry.get()
    return {
        "status": "ok",
        "version": "14",
        "roles": registry.available_roles,
    }


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """WebSocket チャット"""
    await ws.accept()
    router = await _get_router()
    thread_id = str(uuid.uuid4())[:8]

    try:
        while True:
            data = await ws.receive_json()
            message = data.get("message", "")
            model = data.get("model", "auto")
            forced_role = model if model != "auto" else None

            if data.get("thread_id"):
                thread_id = data["thread_id"]

            # 処理開始通知
            await ws.send_json({"type": "thinking", "thread_id": thread_id})

            result = await router.process_message(
                message=message,
                thread_id=thread_id,
                source="console_ws",
                forced_role=forced_role,
            )

            await ws.send_json({
                "type": "response",
                "response": result.get("response", ""),
                "agent_role": result.get("agent_role", ""),
                "handled_by": result.get("handled_by", ""),
                "route": result.get("route", ""),
                "execution_time": result.get("execution_time", 0),
                "thread_id": thread_id,
            })

    except WebSocketDisconnect:
        logger.debug("WebSocket 切断: thread=%s", thread_id)
    except Exception as e:
        logger.error("WebSocket エラー: %s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
