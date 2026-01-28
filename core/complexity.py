"""Complexity Estimator - タスク複雑度判定エンジン

simple   → 侍大将のみ (¥0)
medium   → 侍大将のみ (¥0)
complex  → 侍大将分析 → 将軍実装 (¥280)
strategic → 家老決裁 (¥1,350)
"""

import re
from shogun.core.task_queue import Complexity


# Keyword patterns for complexity estimation
STRATEGIC_KEYWORDS = [
    "優先順位", "予算", "戦略", "方針", "ビジネス", "市場",
    "アーキテクチャ全体", "プロジェクト方針", "ロードマップ",
    "priority", "budget", "strategy", "architecture decision",
]

COMPLEX_KEYWORDS = [
    "統合", "10ファイル", "外部api", "大規模", "リファクタリング",
    "全体リファクタ", "新機能追加", "spotify統合", "認証",
    "データベース移行", "マルチスレッド", "並行処理",
    "integration", "refactor", "multi-file", "authentication",
    "database migration", "concurrent",
]

MEDIUM_KEYWORDS = [
    "3ファイル", "4ファイル", "5ファイル", "実装",
    "テスト作成", "バグ修正", "ドライバ", "関数追加",
    "implement", "test", "fix", "driver", "add function",
]

# Heuristic: count actionable items (e.g., numbered lists, bullet points)
ACTION_ITEM_PATTERN = re.compile(r"^\s*[\d\-\*・]\s*", re.MULTILINE)


def estimate_complexity(task: str) -> Complexity:
    """Estimate task complexity from the prompt text.

    Escalation order:
        simple → medium → complex → strategic

    Returns:
        Complexity enum value.
    """
    task_lower = task.lower()
    task_combined = task + task_lower

    # Strategic check
    if any(kw in task_combined for kw in STRATEGIC_KEYWORDS):
        return Complexity.STRATEGIC

    # Complex check
    if any(kw in task_combined for kw in COMPLEX_KEYWORDS):
        return Complexity.COMPLEX

    # Count action items (more items = more complex)
    action_items = len(ACTION_ITEM_PATTERN.findall(task))
    if action_items >= 5:
        return Complexity.COMPLEX

    # Medium check
    if any(kw in task_combined for kw in MEDIUM_KEYWORDS):
        return Complexity.MEDIUM

    if action_items >= 3:
        return Complexity.MEDIUM

    # Length heuristic
    if len(task) > 500:
        return Complexity.MEDIUM

    return Complexity.SIMPLE


def estimated_cost_yen(complexity: Complexity) -> int:
    """Estimate API cost in yen for a given complexity."""
    costs = {
        Complexity.SIMPLE: 0,       # 侍大将のみ
        Complexity.MEDIUM: 0,       # 侍大将のみ
        Complexity.COMPLEX: 280,    # 将軍 Sonnet
        Complexity.STRATEGIC: 1350, # 家老 Opus
    }
    return costs.get(complexity, 0)
