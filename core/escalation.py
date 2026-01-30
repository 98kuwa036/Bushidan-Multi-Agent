"""Escalation Engine - エスカレーション制御

エスカレーション連鎖:
  足軽(MCP) → 侍大将(R1) → 家老(Sonnet) → 将軍(Opus)

中隊モードでは侍大将が上限。能力超過時に大隊モード推奨を通知。
"""

import logging
from shogun.core.task_queue import Task, TaskStatus, Complexity

logger = logging.getLogger("shogun.escalation")

# Escalation chain (lower index = lower tier)
ESCALATION_CHAIN = [
    "taisho",   # 侍大将 (R1, ローカル)
    "karo",     # 家老 (Sonnet, 作業割振り)
    "shogun",   # 将軍 (Opus, 最終決裁)
]

# Complexity → default handler
COMPLEXITY_HANDLER = {
    Complexity.SIMPLE: "taisho",
    Complexity.MEDIUM: "taisho",
    Complexity.COMPLEX: "karo",       # 家老が将軍に渡す前に判断
    Complexity.STRATEGIC: "shogun",   # 将軍が最終決裁
}

# Agent → required cost (yen)
AGENT_COST = {
    "taisho": 0,
    "karo": 280,
    "shogun": 1350,
}


def get_handler(complexity: Complexity) -> str:
    """Get the default handler agent for a given complexity."""
    return COMPLEXITY_HANDLER.get(complexity, "taisho")


def get_next_escalation(current_agent: str) -> str | None:
    """Get the next agent in the escalation chain.

    Returns None if already at the top (Shogun).
    """
    try:
        idx = ESCALATION_CHAIN.index(current_agent)
    except ValueError:
        return "taisho"  # Default: start from taisho

    if idx + 1 < len(ESCALATION_CHAIN):
        next_agent = ESCALATION_CHAIN[idx + 1]
        logger.info("エスカレーション: %s → %s", current_agent, next_agent)
        return next_agent

    logger.warning("将軍（最上位）。これ以上のエスカレーション不可。")
    return None


def should_escalate(task: Task) -> bool:
    """Determine if a failed/blocked task should be escalated."""
    if task.status not in (TaskStatus.FAILED, TaskStatus.BLOCKED):
        return False
    if task.escalation_count >= len(ESCALATION_CHAIN) - 1:
        return False
    return True


def can_handle_in_company_mode(complexity: Complexity) -> bool:
    """Check if the task can be handled in company mode (中隊).

    中隊 = 侍大将 + 足軽のみ。Simple/Medium のみ対応。
    """
    return complexity in (Complexity.SIMPLE, Complexity.MEDIUM)


def build_escalation_context(task: Task) -> str:
    """Build context message for escalation target."""
    parts = [
        "## エスカレーション報告",
        f"**任務ID**: {task.id}",
        f"**元の指示**: {task.prompt}",
        f"**複雑度**: {task.complexity.value}",
        f"**前任**: {task.assigned_agent}",
        f"**エスカレーション回数**: {task.escalation_count}",
    ]
    if task.error:
        parts.append(f"**エラー**: {task.error}")
    if task.result:
        parts.append(f"**途中経過**:\n{task.result[:2000]}")
    return "\n".join(parts)


def build_taisho_analysis_prompt(task: Task) -> str:
    """Build prompt for Taisho to analyze before escalation to Karo/Shogun."""
    return f"""あなたは侍大将です。以下の任務を分析し、上位への報告をまとめてください。
実装は家老・将軍が行います。あなたは分析と方針提案のみ行ってください。

## 任務
{task.prompt}

## 指示
1. <think>タグで徹底的に思考してください
2. 問題の本質を特定してください
3. 推奨する解決方針を提案してください
4. 足軽（MCPツール）で集めた情報があれば整理してください
"""
