"""Escalation Engine - エスカレーション制御"""

import logging
from typing import Any

from shogun.core.task_queue import Task, TaskCategory, TaskStatus

logger = logging.getLogger("shogun.escalation")

# Escalation chain: lower index = lower tier
ESCALATION_CHAIN = [
    "scout",    # 小者 (1.5B)
    "coder",    # 技術兵 (7B)
    "leader",   # 足軽頭 (8B)
    "taisho",   # 侍大将 (14B)  -- mode switch required
    "karo",     # 家老 (Cloud)
    "shogun",   # 将軍 (Cloud)
]

# Category -> default starting agent
CATEGORY_DEFAULT = {
    TaskCategory.RECON: "scout",
    TaskCategory.CODE: "coder",
    TaskCategory.PLAN: "leader",
    TaskCategory.THINK: "taisho",
    TaskCategory.STRATEGY: "karo",
    TaskCategory.CRITICAL: "shogun",
}

# Agent -> required mode
AGENT_MODE = {
    "scout": "b",
    "coder": "b",
    "leader": "b",
    "taisho": "a",
    "karo": "cloud",
    "shogun": "cloud",
}


def get_default_agent(category: TaskCategory) -> str:
    """Get the default agent for a task category."""
    return CATEGORY_DEFAULT.get(category, "leader")


def get_next_escalation(current_agent: str) -> str | None:
    """Get the next agent in the escalation chain.

    Returns None if already at the top (Shogun).
    """
    try:
        idx = ESCALATION_CHAIN.index(current_agent)
    except ValueError:
        logger.warning("Unknown agent: %s, defaulting to leader", current_agent)
        return "leader"

    if idx + 1 < len(ESCALATION_CHAIN):
        next_agent = ESCALATION_CHAIN[idx + 1]
        logger.info("Escalation: %s -> %s", current_agent, next_agent)
        return next_agent

    logger.warning("Already at top of chain (Shogun). Cannot escalate further.")
    return None


def get_required_mode(agent: str) -> str:
    """Get the operating mode required for an agent."""
    return AGENT_MODE.get(agent, "b")


def should_escalate(task: Task, max_retries: int = 2) -> bool:
    """Determine if a failed task should be escalated."""
    if task.status != TaskStatus.FAILED:
        return False
    if task.escalation_count >= len(ESCALATION_CHAIN) - 1:
        return False
    return True


def build_escalation_context(task: Task) -> str:
    """Build context message for the escalation target."""
    parts = [
        f"## エスカレーション報告 (Task: {task.id})",
        f"**元の指示**: {task.prompt}",
        f"**カテゴリ**: {task.category.value}",
        f"**前任エージェント**: {task.assigned_agent}",
        f"**エスカレーション回数**: {task.escalation_count}",
    ]
    if task.error:
        parts.append(f"**エラー内容**: {task.error}")
    if task.result:
        parts.append(f"**途中経過**: {task.result}")
    return "\n".join(parts)
