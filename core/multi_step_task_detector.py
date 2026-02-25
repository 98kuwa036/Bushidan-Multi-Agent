"""
Bushidan Multi-Agent System v10.1 - Multi-Step Task Detector
複合タスク（複数ステップ）を検出するモジュール
"""

import re
from typing import List, Tuple
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MultiStepAnalysis:
    """複合タスク分析結果"""

    is_multi_step: bool
    reason: str = ""
    steps: List[str] = None
    step_count: int = 0
    pipeline_detected: bool = False
    confidence: float = 0.0  # 0.0-1.0


class MultiStepTaskDetector:
    """複合タスク（複数ステップ）を検出"""

    # 複合タスクを示すキーワード（日本語）
    MULTI_STEP_KEYWORDS_JA = (
        "して",  # 最重要：「クローンして確認して」
        "および",  # 「クローン及び確認」
        "それから",  # 「クローンしてそれから確認」
        "その後",  # 「クローン、その後確認」
        "そして",  # 「クローンしてそして編集」
        "〜した後",  # 「クローンした後に確認」
        "〜後に",  # 「クローン後に確認」
        "〜で",  # 「クローンで確認して」
        "の内容を",  # 「codingsレポジトリの内容をまとめて」→ 内容読取+処理
    )

    # 複合タスクを示すキーワード（英語）
    MULTI_STEP_KEYWORDS_EN = (
        "and then",  # 「clone and then check」
        "after",  # 「after cloning, check」
        "then",  # 「clone, then check」
        "next",  # 「clone, next check」
        "followed by",  # 「clone followed by check」
    )

    # パイプライン的記号
    PIPELINE_SYMBOLS = (
        "→",  # 矢印
        "=>",  # 矢印（ASCII）
        "->",  # 矢印（ASCII）
        "|",  # パイプ（シェル的）
        ">>",  # 連鎖
    )

    # アクションキーワード（単発か複合か判定用）
    ACTION_KEYWORDS = (
        "clone",
        "クローン",
        "pull",
        "push",
        "fetch",
        "install",
        "インストール",
        "run",
        "実行",
        "execute",
        "build",
        "delete",
        "削除",
        "create",
        "作成",
        "edit",
        "編集",
        "update",
        "更新",
        "check",
        "確認",
        "inspect",
        "検査",
        "verify",
        "validate",
        "test",
        "テスト",
        "review",
        "確認",
        "modify",
        "変更",
        "summarize",
        "まとめ",
        "analyze",
        "分析",
        "optimize",
        "最適化",
        "format",
        "フォーマット",
    )

    def __init__(self):
        pass

    def analyze(self, task_content: str) -> MultiStepAnalysis:
        """複合タスクかどうかを分析"""

        logger.debug(f"複合タスク分析開始: {task_content[:50]}...")

        # 複合ステップを示すキーワード検出
        multi_step_kw_count = self._count_multi_step_keywords(task_content)
        has_multi_step_kw = multi_step_kw_count >= 2

        # パイプライン記号検出
        has_pipeline = self._detect_pipeline_symbols(task_content)

        # アクションキーワード複数検出
        action_count = self._count_action_keywords(task_content)
        has_multiple_actions = action_count >= 2

        # ステップ分解
        steps = self._extract_steps(task_content)
        step_count = len(steps)

        # 複合タスク判定
        is_multi_step = False
        confidence = 0.0
        reason = ""

        # Multi-step keywords take priority (highest confidence)
        if multi_step_kw_count >= 1:  # 複合キーワードが1個以上 = 複合タスク
            is_multi_step = True
            confidence = min(0.95, 0.5 + multi_step_kw_count * 0.25)
            reason = f"Multi-step keyword detected ({multi_step_kw_count}x)"
        # Multiple action keywords also indicate multi-step
        elif has_multiple_actions:
            is_multi_step = True
            confidence = 0.75
            reason = f"Multiple action keywords detected ({action_count}x)"

        if has_pipeline:
            is_multi_step = True
            confidence = max(confidence, 0.9)
            reason = f"{reason} + Pipeline symbol detected"

        if step_count >= 2:
            is_multi_step = True
            confidence = max(confidence, 0.85)
            reason = f"{reason} + {step_count} steps extracted"

        # 信頼度が 0 なら分析失敗
        if confidence == 0.0:
            reason = "Not multi-step task"

        analysis = MultiStepAnalysis(
            is_multi_step=is_multi_step,
            reason=reason,
            steps=steps,
            step_count=step_count,
            pipeline_detected=has_pipeline,
            confidence=confidence,
        )

        logger.debug(
            f"分析結果: {analysis.is_multi_step}, 信頼度: {analysis.confidence:.2f}, 理由: {analysis.reason}"
        )

        return analysis

    def _count_multi_step_keywords(self, content: str) -> int:
        """複合ステップキーワードの個数を数える"""
        content_lower = content.lower()
        count = 0

        for kw in self.MULTI_STEP_KEYWORDS_JA:
            count += content_lower.count(kw)

        for kw in self.MULTI_STEP_KEYWORDS_EN:
            count += content_lower.count(kw)

        return count

    def _detect_pipeline_symbols(self, content: str) -> bool:
        """パイプライン記号を検出"""
        for symbol in self.PIPELINE_SYMBOLS:
            if symbol in content:
                logger.debug(f"Pipeline symbol detected: {symbol}")
                return True
        return False

    def _count_action_keywords(self, content: str) -> int:
        """アクションキーワードの個数を数える"""
        content_lower = content.lower()
        count = 0

        for kw in self.ACTION_KEYWORDS:
            # 単語境界で検出（部分一致を避ける）
            if re.search(r"\b" + re.escape(kw) + r"\b", content_lower):
                count += 1

        return count

    def _extract_steps(self, content: str) -> List[str]:
        """ステップを抽出（句点や記号で分割）"""
        steps = []

        # パターン1：句点で分割
        if "。" in content:
            parts = content.split("。")
            steps.extend([p.strip() for p in parts if p.strip()])

        # パターン2：パイプライン記号で分割
        for symbol in self.PIPELINE_SYMBOLS:
            if symbol in content:
                parts = content.split(symbol)
                steps.extend([p.strip() for p in parts if p.strip()])
                break

        # パターン3：複合キーワードで分割（して の場合）
        if "して" in content and len(steps) < 2:
            parts = re.split(r"([^。、]*して)", content)
            temp_steps = []
            for part in parts:
                if part.strip() and part != "して":
                    temp_steps.append(part.strip().replace("して", ""))
            if len(temp_steps) > len(steps):
                steps = temp_steps

        # ステップ数が 2 以上かつ各ステップが短すぎないかチェック
        steps = [s for s in steps if len(s) > 3]  # 3文字以上のステップのみ
        steps = steps[:10]  # 最大10ステップ

        return steps

    def print_analysis(self, analysis: MultiStepAnalysis) -> str:
        """分析結果を出力"""
        output = "🔍 複合タスク分析結果\n"
        output += f"  複合タスク: {'✅ はい' if analysis.is_multi_step else '❌ いいえ'}\n"
        output += f"  信頼度: {analysis.confidence:.1%}\n"
        output += f"  理由: {analysis.reason}\n"

        if analysis.steps:
            output += f"  ステップ数: {analysis.step_count}\n"
            output += "  ステップ:\n"
            for i, step in enumerate(analysis.steps, 1):
                output += f"    {i}. {step[:50]}{'...' if len(step) > 50 else ''}\n"

        return output
