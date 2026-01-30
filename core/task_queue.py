"""Task Queue - YAMLベースのタスクキュー管理

本家 multi-agent-shogun に倣い、YAML ファイルでエージェント間通信を行う。
ファイル分離: 各足軽は自分専用のファイルのみ読み書き（RACE-001対策）。
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("shogun.task_queue")


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"
    ESCALATED = "escalated"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    STRATEGIC = "strategic"


class DeploymentMode(str, Enum):
    BATTALION = "battalion"  # 大隊: 家老+将軍+侍大将+足軽
    COMPANY = "company"      # 中隊: 侍大将+足軽 only (¥0)


@dataclass
class Task:
    """Single task unit."""

    prompt: str
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    status: TaskStatus = TaskStatus.PENDING
    complexity: Complexity = Complexity.SIMPLE
    mode: DeploymentMode = DeploymentMode.BATTALION
    assigned_agent: str = ""
    result: str = ""
    error: str = ""
    escalation_count: int = 0
    context: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    cost_yen: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "status": self.status.value,
            "complexity": self.complexity.value,
            "mode": self.mode.value,
            "assigned_agent": self.assigned_agent,
            "result": self.result,
            "error": self.error,
            "escalation_count": self.escalation_count,
            "context": self.context,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "cost_yen": self.cost_yen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            status=TaskStatus(d.get("status", "pending")),
            complexity=Complexity(d.get("complexity", "simple")),
            mode=DeploymentMode(d.get("mode", "battalion")),
            assigned_agent=d.get("assigned_agent", ""),
            result=d.get("result", ""),
            error=d.get("error", ""),
            escalation_count=d.get("escalation_count", 0),
            context=d.get("context", {}),
            created_at=d.get("created_at", ""),
            completed_at=d.get("completed_at", ""),
            cost_yen=d.get("cost_yen", 0),
        )


class TaskQueue:
    """YAML-based task queue (本家互換).

    File layout:
        queue/shogun_to_karo.yaml      将軍→家老
        queue/tasks/ashigaru{N}.yaml   家老→足軽N (per-worker)
        queue/reports/ashigaru{N}_report.yaml  足軽N→家老 (per-worker)
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.queue_dir = self.base_dir / "queue"
        self.tasks_dir = self.queue_dir / "tasks"
        self.reports_dir = self.queue_dir / "reports"
        self._tasks: dict[str, Task] = {}

        # Ensure directories
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

    def enqueue(self, task: Task) -> None:
        """Add task to queue."""
        self._tasks[task.id] = task
        self._write_shogun_queue()
        logger.info("[Queue] Enqueued: %s (%s)", task.id, task.complexity.value)

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def get_pending(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]

    def update_task(self, task: Task) -> None:
        self._tasks[task.id] = task
        self._write_shogun_queue()

    def complete_task(self, task_id: str, result: str, cost_yen: int = 0) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.DONE
            task.result = result
            task.cost_yen = cost_yen
            task.completed_at = datetime.now().isoformat()
            self._write_shogun_queue()

    def fail_task(self, task_id: str, error: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.error = error
            self._write_shogun_queue()

    # --- YAML file I/O (本家互換) ---

    def _write_shogun_queue(self) -> None:
        """Write queue/shogun_to_karo.yaml."""
        path = self.queue_dir / "shogun_to_karo.yaml"
        data = {
            "queue": [t.to_dict() for t in self._tasks.values()],
        }
        path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False))

    def write_ashigaru_task(self, worker_id: int, task: Task) -> None:
        """Write per-worker task file (RACE-001 safe)."""
        path = self.tasks_dir / f"ashigaru{worker_id}.yaml"
        data = {
            "worker_id": f"ashigaru{worker_id}",
            "task_id": task.id,
            "description": task.prompt,
            "status": "assigned",
            "assigned_at": datetime.now().isoformat(),
        }
        path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False))
        logger.info("[Queue] Assigned to ashigaru%d: %s", worker_id, task.id)

    def write_ashigaru_report(
        self, worker_id: int, task_id: str, status: str,
        summary: str, files_modified: list[str] | None = None,
        skill_candidate: dict | None = None,
    ) -> None:
        """Write per-worker report file (RACE-001 safe)."""
        path = self.reports_dir / f"ashigaru{worker_id}_report.yaml"
        data = {
            "worker_id": f"ashigaru{worker_id}",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "result": {
                "summary": summary,
                "files_modified": files_modified or [],
            },
            "skill_candidate": skill_candidate or {"found": False},
        }
        path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False))

    def read_ashigaru_report(self, worker_id: int) -> dict | None:
        """Read a worker's report."""
        path = self.reports_dir / f"ashigaru{worker_id}_report.yaml"
        if not path.exists():
            return None
        return yaml.safe_load(path.read_text())

    def reset_all_workers(self) -> None:
        """Reset all worker files to idle state (startup)."""
        for i in range(1, 9):
            task_path = self.tasks_dir / f"ashigaru{i}.yaml"
            task_path.write_text(yaml.dump({
                "worker_id": f"ashigaru{i}",
                "task_id": None,
                "description": None,
                "status": "idle",
            }, allow_unicode=True))

            report_path = self.reports_dir / f"ashigaru{i}_report.yaml"
            report_path.write_text(yaml.dump({
                "worker_id": f"ashigaru{i}",
                "task_id": None,
                "timestamp": None,
                "status": "idle",
                "result": None,
                "skill_candidate": {"found": False},
            }, allow_unicode=True))

        # Clear main queue
        (self.queue_dir / "shogun_to_karo.yaml").write_text(
            yaml.dump({"queue": []}, allow_unicode=True)
        )
        logger.info("[Queue] All workers reset to idle")

    def load_from_disk(self) -> None:
        """Load tasks from shogun_to_karo.yaml."""
        path = self.queue_dir / "shogun_to_karo.yaml"
        if not path.exists():
            return
        try:
            data = yaml.safe_load(path.read_text())
            if data and "queue" in data:
                for td in data["queue"]:
                    task = Task.from_dict(td)
                    self._tasks[task.id] = task
                logger.info("[Queue] Loaded %d tasks from disk", len(self._tasks))
        except Exception as e:
            logger.warning("[Queue] Failed to load: %s", e)
