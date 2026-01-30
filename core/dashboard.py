"""Dashboard - 戦況報告 (本家 multi-agent-shogun 互換)

dashboard.md を生成・更新する。
本家ルール: dashboard の更新権限は家老のみ（本システムではControllerが代行）。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("shogun.dashboard")


class Dashboard:
    """Markdown dashboard generator (dashboard.md)."""

    def __init__(self, base_dir: str):
        self.path = Path(base_dir) / "status" / "dashboard.md"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._action_required: list[str] = []
        self._in_progress: list[str] = []
        self._completed: list[dict] = []
        self._skill_candidates: list[dict] = []
        self._questions: list[str] = []

    def init(self) -> None:
        """Initialize a fresh dashboard."""
        self._action_required = []
        self._in_progress = []
        self._completed = []
        self._skill_candidates = []
        self._questions = []
        self._write()
        logger.info("[Dashboard] Initialized: %s", self.path)

    def add_action_required(self, item: str) -> None:
        self._action_required.append(item)
        self._write()

    def add_in_progress(self, item: str) -> None:
        self._in_progress.append(item)
        self._write()

    def remove_in_progress(self, item: str) -> None:
        self._in_progress = [x for x in self._in_progress if x != item]
        self._write()

    def add_completed(self, task_id: str, mission: str, result: str, cost: int = 0) -> None:
        self._completed.append({
            "time": datetime.now().strftime("%H:%M"),
            "task_id": task_id,
            "mission": mission[:60],
            "result": result[:80],
            "cost": cost,
        })
        self._write()

    def add_skill_candidate(self, name: str, description: str) -> None:
        self._skill_candidates.append({"name": name, "description": description})
        self._write()

    def add_question(self, question: str) -> None:
        self._questions.append(question)
        self._write()

    def _write(self) -> None:
        """Write dashboard.md."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# 📊 戦況報告 (Battle Status Report)",
            f"最終更新: {now}",
            "",
        ]

        # 🚨 要対応
        lines.append("## 🚨 要対応 - 殿のご判断をお待ちしております")
        if self._action_required:
            for item in self._action_required:
                lines.append(f"- {item}")
        else:
            lines.append("_なし_")
        lines.append("")

        # 🔄 進行中
        lines.append("## 🔄 進行中")
        if self._in_progress:
            for item in self._in_progress:
                lines.append(f"- {item}")
        else:
            lines.append("_なし_")
        lines.append("")

        # ✅ 戦果
        lines.append("## ✅ 本日の戦果")
        lines.append("| 時刻 | 任務ID | 任務 | 結果 | コスト |")
        lines.append("|------|--------|------|------|--------|")
        if self._completed:
            for c in self._completed:
                cost_str = f"¥{c['cost']:,}" if c["cost"] > 0 else "¥0"
                lines.append(
                    f"| {c['time']} | {c['task_id']} | {c['mission']} | {c['result']} | {cost_str} |"
                )
        lines.append("")

        # 🎯 スキル化候補
        lines.append("## 🎯 スキル化候補")
        if self._skill_candidates:
            for sc in self._skill_candidates:
                lines.append(f"- **{sc['name']}**: {sc['description']}")
        else:
            lines.append("_なし_")
        lines.append("")

        # ❓ 伺い事項
        lines.append("## ❓ 伺い事項")
        if self._questions:
            for q in self._questions:
                lines.append(f"- {q}")
        else:
            lines.append("_なし_")
        lines.append("")

        self.path.write_text("\n".join(lines), encoding="utf-8")

    def get_summary(self) -> dict:
        """Get dashboard summary as dict."""
        total_cost = sum(c.get("cost", 0) for c in self._completed)
        return {
            "action_required": len(self._action_required),
            "in_progress": len(self._in_progress),
            "completed_today": len(self._completed),
            "total_cost_yen": total_cost,
            "skill_candidates": len(self._skill_candidates),
            "questions": len(self._questions),
        }
