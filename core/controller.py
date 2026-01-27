"""Controller - モード切替 & オーケストレーション"""

import asyncio
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from shogun.core.task_queue import Task, TaskQueue, TaskCategory, TaskStatus
from shogun.core.escalation import (
    get_default_agent,
    get_next_escalation,
    get_required_mode,
    should_escalate,
    build_escalation_context,
)
from shogun.providers.ollama import OllamaClient

logger = logging.getLogger("shogun.controller")


class SystemMode(str, Enum):
    IDLE = "idle"
    MODE_A = "mode_a"    # 軍議 (Deep Thinking) - Taisho only
    MODE_B = "mode_b"    # 進軍 (Action) - Ashigaru squad
    CLOUD = "cloud"      # Cloud API (no local models needed)


class Controller:
    """Central orchestrator for the Shogun-Hybrid system.

    Manages mode switching between Mode A (Taisho) and Mode B (Ashigaru),
    dispatches tasks to the appropriate agent, and handles escalation.
    """

    def __init__(self, config_path: str = "shogun/config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.ollama = OllamaClient(
            base_url=self.config["network"]["ollama_base_url"]
        )
        self.current_mode = SystemMode.IDLE
        self.queue = TaskQueue(queue_dir="shogun/queue")
        self._agents: dict[str, Any] = {}
        self._running = False
        self._mode_lock = asyncio.Lock()

    @staticmethod
    def _load_config(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return yaml.safe_load(p.read_text())

    # ------------------------------------------------------------------
    # Agent Registry
    # ------------------------------------------------------------------

    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent instance."""
        self._agents[name] = agent
        logger.info("Agent registered: %s", name)

    def get_agent(self, name: str) -> Any:
        return self._agents.get(name)

    # ------------------------------------------------------------------
    # Mode Switching
    # ------------------------------------------------------------------

    async def switch_mode(self, target: SystemMode) -> None:
        """Switch between Mode A / Mode B / Cloud.

        This is the critical operation that manages 24GB RAM:
        - Mode A: Unload all ashigaru, load Taisho (22GB)
        - Mode B: Unload Taisho, load Leader + Coder + Scout (18GB)
        - Cloud: Unload all local models
        """
        async with self._mode_lock:
            if self.current_mode == target:
                logger.debug("Already in %s, no switch needed", target.value)
                return

            logger.info(
                "=== MODE SWITCH: %s -> %s ===",
                self.current_mode.value, target.value
            )
            start = time.monotonic()

            # Step 1: Unload current mode's models
            await self._unload_current_mode()

            # Step 2: Load target mode's models
            if target == SystemMode.MODE_A:
                await self._load_mode_a()
            elif target == SystemMode.MODE_B:
                await self._load_mode_b()
            # Cloud mode needs no local model loading

            self.current_mode = target
            elapsed = time.monotonic() - start
            logger.info(
                "=== MODE SWITCH COMPLETE: %s (%.1fs) ===",
                target.value, elapsed
            )

    async def _unload_current_mode(self) -> None:
        """Unload all models for the current mode."""
        if self.current_mode == SystemMode.MODE_A:
            cfg = self.config["local"]["mode_a"]["taisho"]
            await self.ollama.unload_model(cfg["model"])
        elif self.current_mode == SystemMode.MODE_B:
            for agent_key in ("leader", "coder", "scout"):
                cfg = self.config["local"]["mode_b"][agent_key]
                await self.ollama.unload_model(cfg["model"])

    async def _load_mode_a(self) -> None:
        """Load Taisho model."""
        cfg = self.config["local"]["mode_a"]["taisho"]
        await self.ollama.load_model(cfg["model"], keep_alive="30m")

    async def _load_mode_b(self) -> None:
        """Load all Ashigaru models."""
        for agent_key in ("leader", "coder", "scout"):
            cfg = self.config["local"]["mode_b"][agent_key]
            await self.ollama.load_model(cfg["model"], keep_alive="30m")

    # ------------------------------------------------------------------
    # Task Dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, task: Task) -> str:
        """Dispatch a task to the appropriate agent.

        Returns the result string.
        """
        # Determine agent
        if not task.assigned_agent:
            task.assigned_agent = get_default_agent(task.category)

        agent_name = task.assigned_agent
        required_mode = get_required_mode(agent_name)

        # Switch mode if necessary
        if required_mode == "a":
            await self.switch_mode(SystemMode.MODE_A)
        elif required_mode == "b":
            await self.switch_mode(SystemMode.MODE_B)
        elif required_mode == "cloud":
            await self.switch_mode(SystemMode.CLOUD)

        # Get agent
        agent = self.get_agent(agent_name)
        if not agent:
            raise RuntimeError(f"Agent not found: {agent_name}")

        # Execute
        logger.info(
            "[Dispatch] Task %s -> %s (%s)",
            task.id, agent_name, task.category.value
        )
        task.status = TaskStatus.IN_PROGRESS
        try:
            result = await agent.execute(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            await self.queue.complete(task.id, result)
            return result
        except Exception as e:
            error_msg = str(e)
            task.error = error_msg
            task.status = TaskStatus.FAILED
            await self.queue.fail(task.id, error_msg)
            logger.error("[Dispatch] Task %s failed: %s", task.id, error_msg)

            # Attempt escalation
            if should_escalate(task):
                return await self._escalate_task(task)
            raise

    async def _escalate_task(self, task: Task) -> str:
        """Escalate a failed task to the next agent in the chain."""
        next_agent = get_next_escalation(task.assigned_agent)
        if not next_agent:
            raise RuntimeError(
                f"Task {task.id}: Escalation exhausted. Even Shogun cannot solve this."
            )

        await self.queue.escalate(task.id)
        context = build_escalation_context(task)

        # Create escalated task
        escalated = Task(
            prompt=f"{context}\n\n---\n\n{task.prompt}",
            category=task.category,
            assigned_agent=next_agent,
            parent_id=task.id,
            escalation_count=task.escalation_count,
            context=task.context,
        )
        await self.queue.enqueue(escalated)

        logger.info(
            "[Escalation] %s -> %s (task %s -> %s)",
            task.assigned_agent, next_agent, task.id, escalated.id
        )
        return await self.dispatch(escalated)

    # ------------------------------------------------------------------
    # Quick Execute (CLI shortcut)
    # ------------------------------------------------------------------

    async def ask(
        self,
        prompt: str,
        category: str = "code",
        agent: str = "",
    ) -> str:
        """Quick single-shot execution from CLI."""
        cat = TaskCategory(category)
        task = Task(prompt=prompt, category=cat)
        if agent:
            task.assigned_agent = agent
        await self.queue.enqueue(task)
        return await self.dispatch(task)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Get current system status."""
        return {
            "mode": self.current_mode.value,
            "agents_registered": list(self._agents.keys()),
            "pending_tasks": self.queue.get_pending_count(),
            "total_tasks": len(self.queue.get_all_tasks()),
        }

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down controller...")
        await self._unload_current_mode()
        await self.ollama.close()
        logger.info("Controller shutdown complete.")
