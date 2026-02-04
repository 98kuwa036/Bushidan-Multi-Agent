"""
Bushidan Multi-Agent System v10 - 軍師 (Gunshi) Layer

5層ハイブリッドアーキテクチャの作戦立案層。
将軍の戦略的判断を受け、具体的な作戦計画に変換して家老に指示する。

Model: Qwen3-Coder-Next 80B-A3B (API)
- 256K コンテキスト: コードベース全体を俯瞰
- SWE-Bench 70.6%: 高度なコード理解
- Non-thinking mode: 即断即決

BDI Framework:
- Beliefs: コードベース構造、タスク複雑度、チーム能力
- Desires: 正確な作戦立案、コード品質保証、効率的分解
- Intentions: 作戦計画の生成と実行監督

v10 新機能:
- 作戦立案 (plan_strategy): 複雑タスクの実行計画
- コード監査 (audit_code): 実装結果のレビュー
- タスク分解 (decompose): サブタスクへの分割
- 将軍→軍師→家老の指揮系統確立
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from utils.logger import get_logger


logger = get_logger(__name__)


class GunshiMode(Enum):
    """軍師の動作モード"""
    STRATEGY = "strategy"    # 作戦立案
    AUDIT = "audit"          # コード監査
    DECOMPOSE = "decompose"  # タスク分解
    ADVISE = "advise"        # 助言


@dataclass
class GunshiResult:
    """軍師の処理結果"""
    mode: GunshiMode
    content: str
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: str = ""
    confidence: float = 0.0
    elapsed_seconds: float = 0.0
    tokens_used: int = 0


class Gunshi:
    """
    軍師 (Gunshi) - 作戦立案層

    将軍が「何をやるか」を決め、軍師が「どうやるか」を設計する。
    256K コンテキストでコードベース全体を俯瞰し、
    複雑タスクを実装可能な単位に分解して家老・大将に指示する。

    位置: 将軍 → 【軍師】 → 家老 → 大将 → 足軽

    担当:
    - COMPLEX タスクの作戦立案
    - コードアーキテクチャの監査
    - 実装結果のレビュー
    - 技術的意思決定の助言
    """

    VERSION = "10.0"

    def __init__(self, orchestrator):
        """
        軍師の初期化

        Args:
            orchestrator: SystemOrchestrator reference
        """
        self.orchestrator = orchestrator
        self.client = None
        self.initialized = False

        # BDI state
        self.bdi_enabled = True
        self.beliefs: Dict[str, Any] = {}
        self.desires = {
            "accurate_planning": {"priority": 0.95, "type": "achievement"},
            "code_quality": {"priority": 0.9, "type": "maintenance"},
            "efficient_decomposition": {"priority": 0.85, "type": "optimization"},
            "risk_mitigation": {"priority": 0.8, "type": "maintenance"},
        }

        # Statistics
        self._stats = {
            "total_tasks": 0,
            "strategies_planned": 0,
            "audits_completed": 0,
            "decompositions": 0,
            "advices_given": 0,
            "fallbacks_to_karo": 0,
        }

        logger.info("🧠 軍師（作戦立案層）初期化開始...")

    async def initialize(self) -> None:
        """軍師の初期化"""
        # Qwen3-Coder-Next クライアント取得
        self.client = self.orchestrator.get_client("qwen3_coder_next")

        if self.client:
            logger.info("🧠 軍師初期化完了 (Qwen3-Coder-Next API)")
            self.initialized = True
        else:
            logger.warning("⚠️ 軍師クライアント未設定 (複雑タスクは家老に直接委譲)")
            self.initialized = False

    async def process_complex_task(
        self,
        task_content: str,
        complexity: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GunshiResult:
        """
        複雑タスクの処理 (将軍からの委譲)

        軍師が利用可能な場合: 作戦を立案してから家老に委譲
        軍師が利用不可の場合: 家老に直接フォールバック

        Args:
            task_content: タスク内容
            complexity: タスク複雑度
            context: コンテキスト情報

        Returns:
            GunshiResult with strategy plan
        """
        self._stats["total_tasks"] += 1

        if not self.initialized or not self.client:
            logger.info("🧠 軍師不在 → 家老に直接委譲")
            self._stats["fallbacks_to_karo"] += 1
            return GunshiResult(
                mode=GunshiMode.ADVISE,
                content=task_content,
                confidence=0.0,
            )

        start_time = asyncio.get_event_loop().time()

        try:
            # タスク複雑度に応じたモード選択
            mode = self._select_mode(task_content, complexity)

            if mode == GunshiMode.STRATEGY:
                result = await self._plan_strategy(task_content, context)
            elif mode == GunshiMode.DECOMPOSE:
                result = await self._decompose_task(task_content, context)
            elif mode == GunshiMode.AUDIT:
                result = await self._audit(task_content, context)
            else:
                result = await self._advise(task_content, context)

            elapsed = asyncio.get_event_loop().time() - start_time
            result.elapsed_seconds = elapsed

            logger.info(
                f"🧠 軍師処理完了: {mode.value} / {elapsed:.2f}s / "
                f"confidence: {result.confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"❌ 軍師処理失敗: {e}")
            self._stats["fallbacks_to_karo"] += 1
            return GunshiResult(
                mode=GunshiMode.ADVISE,
                content=task_content,
                confidence=0.0,
            )

    def _select_mode(self, task_content: str, complexity: str) -> GunshiMode:
        """タスクに応じた動作モード選択"""
        task_lower = task_content.lower()

        # レビュー・監査キーワード
        audit_keywords = [
            "レビュー", "review", "監査", "audit", "チェック", "check",
            "品質", "quality", "セキュリティ", "security"
        ]
        if any(kw in task_lower for kw in audit_keywords):
            return GunshiMode.AUDIT

        # 分解キーワード
        decompose_keywords = [
            "分解", "decompose", "分割", "split", "サブタスク", "subtask",
            "ステップ", "step"
        ]
        if any(kw in task_lower for kw in decompose_keywords):
            return GunshiMode.DECOMPOSE

        # 複雑タスク → 作戦立案
        if complexity in ("complex", "strategic"):
            return GunshiMode.STRATEGY

        return GunshiMode.ADVISE

    async def _plan_strategy(
        self,
        task_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GunshiResult:
        """作戦立案"""
        self._stats["strategies_planned"] += 1

        result_text = await self.client.plan_strategy(
            task_content=task_content,
            codebase_context=context.get("codebase", "") if context else None
        )

        return GunshiResult(
            mode=GunshiMode.STRATEGY,
            content=result_text,
            confidence=0.85,
        )

    async def _decompose_task(
        self,
        task_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GunshiResult:
        """タスク分解"""
        self._stats["decompositions"] += 1

        constraints = context.get("constraints", "") if context else None

        result_text = await self.client.decompose_task(
            complex_task=task_content,
            constraints=constraints
        )

        return GunshiResult(
            mode=GunshiMode.DECOMPOSE,
            content=result_text,
            confidence=0.8,
        )

    async def _audit(
        self,
        task_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GunshiResult:
        """コード監査"""
        self._stats["audits_completed"] += 1

        result_text = await self.client.review_code(
            code=task_content,
            review_focus=context.get("review_focus", "品質・セキュリティ") if context else "品質・セキュリティ"
        )

        return GunshiResult(
            mode=GunshiMode.AUDIT,
            content=result_text,
            confidence=0.9,
        )

    async def _advise(
        self,
        task_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GunshiResult:
        """助言"""
        self._stats["advices_given"] += 1

        messages = [{"role": "user", "content": task_content}]
        result_text = await self.client.generate(messages)

        return GunshiResult(
            mode=GunshiMode.ADVISE,
            content=result_text,
            confidence=0.75,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """統計取得"""
        return {
            "version": self.VERSION,
            "initialized": self.initialized,
            "model": "Qwen3-Coder-Next 80B-A3B (API)",
            "bdi_enabled": self.bdi_enabled,
            **self._stats
        }

    def get_bdi_state(self) -> Dict[str, Any]:
        """BDI状態取得"""
        return {
            "beliefs": self.beliefs,
            "desires": self.desires,
            "current_intentions": []
        }
