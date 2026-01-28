"""Omni-P4 将軍システム - FastAPI Server

REST API for CT 100 (本陣).
Slack Bot, HA OS, メインPC全てのインターフェースが本陣経由。

Run: shogun server --port 8080
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shogun.core.controller import Controller

logger = logging.getLogger("shogun.server")

_ctrl: Controller | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ctrl
    logger.info("[本陣] サーバー起動...")
    base_dir = str(Path(__file__).resolve().parent)
    _ctrl = Controller(base_dir=base_dir)
    await _ctrl.startup()
    logger.info("[本陣] サーバー準備完了")
    yield
    logger.info("[本陣] サーバー停止...")
    if _ctrl:
        await _ctrl.shutdown()
    _ctrl = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Omni-P4 将軍システム",
        description="階層型ハイブリッドAI開発システム v5.0",
        version="5.0.0",
        lifespan=lifespan,
    )

    # --- Models ---

    class TaskRequest(BaseModel):
        task: str = Field(..., description="タスク内容")
        mode: str = Field("battalion", description="battalion / company")
        agent: str = Field("", description="エージェント指定 (空=自動)")

    class TaskResponse(BaseModel):
        task_id: str
        status: str
        agent: str
        mode: str
        complexity: str
        cost_yen: int
        result: str

    class AskRequest(BaseModel):
        question: str = Field(..., description="質問 (HA OS音声用)")

    class AskResponse(BaseModel):
        answer: str
        mode: str
        cost: int

    # --- Routes ---

    @app.get("/")
    async def root():
        return {
            "system": "Omni-P4 将軍システム",
            "version": "5.0.0",
            "modes": ["battalion", "company"],
        }

    @app.post("/api/task", response_model=TaskResponse)
    async def create_task(req: TaskRequest):
        """タスク投入 (大隊 / 中隊)."""
        if not _ctrl:
            raise HTTPException(503, "System not ready")

        try:
            result = await _ctrl.process_task(
                prompt=req.task,
                mode=req.mode,
                force_agent=req.agent,
            )
        except Exception as e:
            raise HTTPException(500, str(e))

        # Get task info
        tasks = _ctrl.queue.get_all_tasks()
        task = tasks[-1] if tasks else None

        return TaskResponse(
            task_id=task.id if task else "unknown",
            status=task.status.value if task else "done",
            agent=task.assigned_agent if task else "",
            mode=req.mode,
            complexity=task.complexity.value if task else "simple",
            cost_yen=task.cost_yen if task else 0,
            result=result,
        )

    @app.post("/api/ask", response_model=AskResponse)
    async def ask_company(req: AskRequest):
        """HA OS音声コマンド受付 (中隊モード固定, ¥0)."""
        if not _ctrl:
            raise HTTPException(503, "System not ready")

        try:
            result = await _ctrl.process_task(
                prompt=req.question,
                mode="company",
            )
        except Exception as e:
            raise HTTPException(500, str(e))

        # Strip <think> blocks for voice response
        answer = result
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[1].strip()

        return AskResponse(answer=answer, mode="company", cost=0)

    @app.get("/api/status")
    async def status():
        if not _ctrl:
            raise HTTPException(503, "System not ready")
        return _ctrl.get_status()

    @app.get("/api/stats")
    async def stats():
        if not _ctrl:
            raise HTTPException(503, "System not ready")
        return _ctrl.stats

    @app.get("/api/health")
    async def health():
        if not _ctrl:
            return {"status": "starting"}
        r1_ok = await _ctrl.openvino.health()
        cli_ok = await _ctrl.claude_cli.check_available()
        return {
            "status": "ok" if r1_ok else "degraded",
            "taisho_r1": r1_ok,
            "claude_cli": cli_ok,
            "api_fallback": _ctrl.api_provider is not None,
            "mode": _ctrl.current_mode.value,
        }

    @app.post("/api/mode")
    async def switch_mode(mode: str):
        if not _ctrl:
            raise HTTPException(503, "System not ready")
        from shogun.core.task_queue import DeploymentMode
        try:
            _ctrl.current_mode = DeploymentMode(mode)
        except ValueError:
            raise HTTPException(400, f"Unknown mode: {mode}")
        label = "大隊" if mode == "battalion" else "中隊"
        return {"mode": mode, "label": label}

    @app.get("/api/mcp")
    async def mcp_status():
        if not _ctrl:
            raise HTTPException(503, "System not ready")
        return {"servers": _ctrl.mcp.get_status()}

    @app.get("/api/dashboard")
    async def dashboard():
        if not _ctrl:
            raise HTTPException(503, "System not ready")
        return _ctrl.dashboard.get_summary()

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)
