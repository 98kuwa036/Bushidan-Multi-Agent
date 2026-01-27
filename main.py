"""
Omni-P4 Shogun-Hybrid - FastAPI Server

REST API for task submission, status monitoring, and agent management.
Run: shogun server --port 8400
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shogun.core.controller import Controller, SystemMode
from shogun.core.task_queue import Task, TaskCategory, TaskStatus
from shogun.agents.factory import create_local_agents, create_cloud_agents

logger = logging.getLogger("shogun.server")

# Global controller reference
_ctrl: Controller | None = None


async def get_controller() -> Controller:
    global _ctrl
    if _ctrl is None:
        raise RuntimeError("Controller not initialized")
    return _ctrl


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global _ctrl
    logger.info("Starting Shogun-Hybrid server...")

    config_path = str(PROJECT_ROOT / "shogun" / "config" / "settings.yaml")
    _ctrl = Controller(config_path=config_path)

    # Register local agents
    local_agents = create_local_agents(_ctrl.ollama, _ctrl.config)
    for name, agent in local_agents.items():
        _ctrl.register_agent(name, agent)

    # Register cloud agents (optional)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            from shogun.providers.anthropic_api import AnthropicClient
            anthropic = AnthropicClient(api_key=api_key)
            cloud_agents = create_cloud_agents(anthropic, _ctrl.config)
            for name, agent in cloud_agents.items():
                _ctrl.register_agent(name, agent)
        except Exception as e:
            logger.warning("Cloud agents unavailable: %s", e)

    # Load pending tasks
    await _ctrl.queue.load_from_disk()

    logger.info("Server ready.")
    yield

    # Shutdown
    logger.info("Shutting down server...")
    await _ctrl.shutdown()
    _ctrl = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Omni-P4 Shogun-Hybrid",
        description="階層型ハイブリッドAI開発システム",
        version="3.0.0",
        lifespan=lifespan,
    )

    # --- Request/Response Models ---

    class AskRequest(BaseModel):
        prompt: str = Field(..., description="タスク内容")
        category: str = Field("code", description="カテゴリ: recon/code/plan/think/strategy/critical")
        agent: str = Field("", description="エージェント指定 (空欄=自動選択)")
        context: dict = Field(default_factory=dict, description="追加コンテキスト")

    class AskResponse(BaseModel):
        task_id: str
        status: str
        agent: str
        result: str

    class ModeRequest(BaseModel):
        mode: str = Field(..., description="a / b / cloud / idle")

    class StatusResponse(BaseModel):
        mode: str
        agents_registered: list[str]
        pending_tasks: int
        total_tasks: int
        ollama_healthy: bool

    class TaskResponse(BaseModel):
        id: str
        prompt: str
        category: str
        status: str
        assigned_agent: str
        result: str
        error: str
        escalation_count: int

    # --- Routes ---

    @app.get("/")
    async def root():
        return {"system": "Omni-P4 Shogun-Hybrid", "version": "3.0.0"}

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest):
        """Submit a task for execution."""
        ctrl = await get_controller()
        try:
            cat = TaskCategory(req.category)
        except ValueError:
            raise HTTPException(400, f"Unknown category: {req.category}")

        task = Task(prompt=req.prompt, category=cat, context=req.context)
        if req.agent:
            task.assigned_agent = req.agent
        await ctrl.queue.enqueue(task)

        try:
            result = await ctrl.dispatch(task)
        except Exception as e:
            raise HTTPException(500, str(e))

        return AskResponse(
            task_id=task.id,
            status=task.status.value,
            agent=task.assigned_agent,
            result=result,
        )

    @app.get("/status", response_model=StatusResponse)
    async def status():
        """Get system status."""
        ctrl = await get_controller()
        st = ctrl.status()
        healthy = await ctrl.ollama.health()
        return StatusResponse(
            mode=st["mode"],
            agents_registered=st["agents_registered"],
            pending_tasks=st["pending_tasks"],
            total_tasks=st["total_tasks"],
            ollama_healthy=healthy,
        )

    @app.post("/mode")
    async def switch_mode(req: ModeRequest):
        """Switch operating mode."""
        ctrl = await get_controller()
        mode_map = {
            "a": SystemMode.MODE_A,
            "b": SystemMode.MODE_B,
            "cloud": SystemMode.CLOUD,
            "idle": SystemMode.IDLE,
        }
        target = mode_map.get(req.mode.lower())
        if not target:
            raise HTTPException(400, f"Unknown mode: {req.mode}")

        await ctrl.switch_mode(target)
        return {"mode": target.value, "message": f"Switched to {target.value}"}

    @app.get("/agents")
    async def list_agents():
        """List registered agents."""
        ctrl = await get_controller()
        agents = []
        for name, agent in sorted(ctrl._agents.items()):
            agents.append({
                "name": name,
                "codename": agent.codename,
                "tier": agent.tier,
                "role": agent.role[:80],
            })
        return {"agents": agents}

    @app.get("/tasks")
    async def list_tasks():
        """List all tasks."""
        ctrl = await get_controller()
        tasks = ctrl.queue.get_all_tasks()
        return {
            "tasks": [
                TaskResponse(
                    id=t.id,
                    prompt=t.prompt[:200],
                    category=t.category.value,
                    status=t.status.value,
                    assigned_agent=t.assigned_agent,
                    result=t.result[:500] if t.result else "",
                    error=t.error,
                    escalation_count=t.escalation_count,
                ).dict()
                for t in tasks
            ]
        }

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get a specific task."""
        ctrl = await get_controller()
        task = ctrl.queue.get_task(task_id)
        if not task:
            raise HTTPException(404, f"Task not found: {task_id}")
        return task.to_dict()

    @app.get("/health")
    async def health():
        """Health check."""
        ctrl = await get_controller()
        ollama_ok = await ctrl.ollama.health()
        return {
            "status": "ok" if ollama_ok else "degraded",
            "ollama": ollama_ok,
            "mode": ctrl.current_mode.value,
        }

    @app.post("/unload")
    async def unload_all():
        """Unload all models from memory."""
        ctrl = await get_controller()
        await ctrl.ollama.unload_all()
        ctrl.current_mode = SystemMode.IDLE
        return {"message": "All models unloaded", "mode": "idle"}

    return app


# For direct execution: python -m shogun.main
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8400)
