"""Task Queue - タスク管理キュー"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("shogun.queue")


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class TaskCategory(str, Enum):
    RECON = "recon"       # 偵察
    CODE = "code"         # 実装
    PLAN = "plan"         # 設計
    THINK = "think"       # 深考
    STRATEGY = "strategy" # 戦略
    CRITICAL = "critical" # 最終決裁


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str = ""
    category: TaskCategory = TaskCategory.CODE
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str = ""
    result: str = ""
    error: str = ""
    context: dict = field(default_factory=dict)
    parent_id: str = ""
    escalation_count: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d["category"] = TaskCategory(d["category"])
        d["status"] = TaskStatus(d["status"])
        return cls(**d)


class TaskQueue:
    """Async task queue with persistence."""

    def __init__(self, queue_dir: str = "queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._queue: asyncio.Queue[Task] = asyncio.Queue()
        self._tasks: dict[str, Task] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, task: Task) -> str:
        """Add a task to the queue."""
        async with self._lock:
            self._tasks[task.id] = task
            await self._queue.put(task)
            self._persist(task)
            logger.info(
                "[Queue] Enqueued: %s (%s) -> %s",
                task.id, task.category.value, task.prompt[:50]
            )
        return task.id

    async def dequeue(self) -> Task:
        """Get next pending task."""
        task = await self._queue.get()
        task.status = TaskStatus.IN_PROGRESS
        self._persist(task)
        return task

    async def complete(self, task_id: str, result: str) -> None:
        """Mark a task as completed."""
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = time.time()
                self._persist(task)
                logger.info("[Queue] Completed: %s", task_id)

    async def fail(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.FAILED
                task.error = error
                self._persist(task)
                logger.warning("[Queue] Failed: %s - %s", task_id, error)

    async def escalate(self, task_id: str) -> None:
        """Mark a task for escalation."""
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.ESCALATED
                task.escalation_count += 1
                self._persist(task)
                logger.info(
                    "[Queue] Escalated: %s (count=%d)",
                    task_id, task.escalation_count
                )

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def get_pending_count(self) -> int:
        return self._queue.qsize()

    def _persist(self, task: Task) -> None:
        """Save task state to disk."""
        path = self.queue_dir / f"{task.id}.json"
        path.write_text(json.dumps(task.to_dict(), ensure_ascii=False, indent=2))

    async def load_from_disk(self) -> int:
        """Restore pending tasks from disk on startup."""
        count = 0
        for path in sorted(self.queue_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                task = Task.from_dict(data)
                if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                    task.status = TaskStatus.PENDING
                    self._tasks[task.id] = task
                    await self._queue.put(task)
                    count += 1
                else:
                    self._tasks[task.id] = task
            except Exception as e:
                logger.error("Failed to load task %s: %s", path.name, e)
        logger.info("[Queue] Loaded %d pending tasks from disk", count)
        return count
