"""
Bushidan Multi-Agent System v11.4 - 軍師 (Gunshi) PDCA Operation Layer

8層ハイブリッドアーキテクチャの作戦立案・実行監督層。
将軍の戦略的判断を受け、PDCAサイクルで複雑タスクを完遂する。

PDCA Operation Cycle:
  Plan  (作戦立案): reasoning_effort=highでタスク分析 → サブタスク分解
  Do    (作戦実行): サブタスクを参謀A/B(GPT-5/Grok)に委譲 → 並列実行
  Check (戦果検証): 全実装結果を一括レビュー + 検校ビジュアル検証
  Act   (修正指示): 不合格サブタスクに修正指示 → 再実行 (最大1回)

Reasoning Effort Strategy (各フェーズで最適化):
  Plan:   reasoning_effort="high" (正確かつ多角的な計画)
  Do:     参謀A/B default (実装層に委任)
  Check:  reasoning_effort="high" (厳密な検証、見逃し防止)
  Act:    reasoning_effort="high" (精密な修正指示)

Model: o3-mini (reasoning_effort=high)
- 推論特化モデル: Plan/Check/Actの精度に直結
- reasoning_effort制御: temperatureではなくreasoning_effortで精度調整

Position: 大元帥 → 将軍 → 【軍師】→ 参謀A/B → 家老A/B → 検校 → 隠密 → 足軽
"""

import asyncio
import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from utils.logger import get_logger


logger = get_logger(__name__)


# ==================== Data Structures ====================

class GunshiMode(Enum):
    """軍師の動作モード"""
    STRATEGY = "strategy"    # 作戦立案
    AUDIT = "audit"          # コード監査
    DECOMPOSE = "decompose"  # タスク分解
    ADVISE = "advise"        # 助言


class OperationPhase(Enum):
    """PDCA作戦フェーズ"""
    PLAN = "plan"
    DO = "do"
    CHECK = "check"
    ACT = "act"
    COMPLETED = "completed"


@dataclass
class SubTask:
    """分解されたサブタスク"""
    id: str
    description: str
    focused_context: str = ""
    estimated_tokens: int = 2000
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    # Execution results
    result: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    execution_time: float = 0.0
    executed_by: str = ""


@dataclass
class VerificationVerdict:
    """サブタスク別検証結果"""
    subtask_id: str
    passed: bool
    issues: List[str] = field(default_factory=list)
    fix_instructions: str = ""


@dataclass
class OperationResult:
    """PDCA作戦の最終結果"""
    task_content: str
    plan_summary: str
    subtasks: List[SubTask]
    verification_passed: bool
    quality_score: float
    corrections_applied: int
    total_elapsed_seconds: float
    phase_times: Dict[str, float] = field(default_factory=dict)
    final_output: str = ""
    error: Optional[str] = None


@dataclass
class GunshiResult:
    """軍師の処理結果 (legacy互換 + PDCA)"""
    mode: GunshiMode
    content: str
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: str = ""
    confidence: float = 0.0
    elapsed_seconds: float = 0.0
    tokens_used: int = 0
    operation_result: Optional[OperationResult] = None


# ==================== Gunshi PDCA Engine ====================

class Gunshi:
    """
    軍師 (Gunshi) - PDCA作戦立案・実行監督エンジン

    ■ 問題認識:
    複数ファイルにまたがる COMPLEX タスクでは整合性が保てない。
    参謀に適切な粒度で委譲し、全体を俯瞰して検証する必要がある。

    ■ 解決策: PDCA で Gunshi (o3-mini) の推論能力を最大活用
    Plan:  Gunshi が全体を俯瞰してサブタスクに分解 (各≤4000tok)
    Do:    各サブタスクを参謀A(GPT-5)/参謀B(Grok) に委譲 → 並列実行
    Check: 全結果を Gunshi に戻して一括検証 (cross-file整合性)
           + 検校(Kengyo)がUI関連タスクのビジュアル検証を実施
    Act:   不合格に具体的修正指示 → 参謀が再実装 (最大1回)

    ■ Reasoning Effort 戦略 (o3-miniはtemperatureではなくreasoning_effortを使用):
    Plan  = "high"  (正確な分析、多角的な計画)
    Check = "high"  (見逃し防止、厳密判定)
    Act   = "high"  (正確な修正指示)

    ■ 制約 (暴走防止):
    - サブタスク最大5個 (過度な分解は品質低下を招く)
    - 修正ループ最大1回 (無限ループ防止)
    - サブタスク context ≤ 4000 tokens (参謀のコンテキストに収める)
    """

    VERSION = "11.4"
    MAX_SUBTASKS = 5
    MAX_CORRECTION_LOOPS = 1
    SUBTASK_TOKEN_LIMIT = 4000

    # Reasoning effort per phase (o3-mini uses reasoning_effort instead of temperature)
    REASONING_EFFORT_PLAN = "high"
    REASONING_EFFORT_CHECK = "high"
    REASONING_EFFORT_ACT = "high"

    def __init__(self, orchestrator):
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
            "total_operations": 0,
            "pdca_cycles_completed": 0,
            "subtasks_delegated": 0,
            "subtasks_completed": 0,
            "verifications_passed": 0,
            "verifications_failed": 0,
            "corrections_applied": 0,
            "fallbacks_to_karo": 0,
            "avg_quality_score": 0.0,
            "total_plan_time": 0.0,
            "total_do_time": 0.0,
            "total_check_time": 0.0,
        }

        logger.info("🧠 軍師（PDCA作戦エンジン）初期化開始...")

    async def initialize(self) -> None:
        """軍師の初期化"""
        self.client = self.orchestrator.get_client("o3mini")

        if self.client:
            logger.info("🧠 軍師初期化完了 (o3-mini reasoning_effort=high + PDCA Engine)")
            self.initialized = True
        else:
            logger.warning("⚠️ 軍師クライアント未設定 (複雑タスクは家老に直接委譲)")
            self.initialized = False

    # ==================== PDCA Main Entry ====================

    async def execute_operation(
        self,
        task_content: str,
        complexity: str = "complex",
        context: Optional[Dict[str, Any]] = None
    ) -> OperationResult:
        """
        PDCA作戦サイクル実行 (メインエントリーポイント)

        将軍から委譲された COMPLEX タスクを
        Plan → Do → Check → Act のサイクルで完遂する。

        Args:
            task_content: タスク内容
            complexity: タスク複雑度
            context: コンテキスト情報 (codebase等)

        Returns:
            OperationResult with plan, results, verification, quality score
        """
        self._stats["total_operations"] += 1
        operation_start = asyncio.get_event_loop().time()
        phase_times: Dict[str, float] = {}

        # Report to Discord if reporter available
        reporter = self.orchestrator.get_reporter()
        task_id = self.orchestrator.get_task_id()
        if reporter and task_id:
            await reporter.report_start(
                task_id,
                "gunshi",
                "軍師がPDCA作戦サイクルを開始します"
            )

        # Gunshi不在 → 家老にフォールバック
        if not self.initialized or not self.client:
            logger.info("🧠 軍師不在 → 家老に直接委譲")
            self._stats["fallbacks_to_karo"] += 1
            return OperationResult(
                task_content=task_content,
                plan_summary="",
                subtasks=[],
                verification_passed=False,
                quality_score=0.0,
                corrections_applied=0,
                total_elapsed_seconds=0.0,
                error="gunshi_unavailable"
            )

        try:
            # ===== Phase 1: PLAN (作戦立案) =====
            logger.info("📋 Phase 1/4: PLAN (作戦立案)...")
            if reporter and task_id:
                await reporter.report_progress(
                    task_id,
                    "📋 Phase 1/4: PLAN - 作戦立案中...",
                    0.0
                )
            t0 = asyncio.get_event_loop().time()

            plan_summary, subtasks = await self._phase_plan(task_content, context)

            phase_times["plan"] = asyncio.get_event_loop().time() - t0
            self._stats["total_plan_time"] += phase_times["plan"]
            logger.info(
                f"📋 PLAN完了: {len(subtasks)}個のサブタスク "
                f"({phase_times['plan']:.1f}s)"
            )

            # ===== Phase 2: DO (作戦実行) =====
            logger.info(f"⚔️ Phase 2/4: DO (作戦実行 - {len(subtasks)}タスク)...")
            if reporter and task_id:
                await reporter.report_progress(
                    task_id,
                    f"⚔️ Phase 2/4: DO - {len(subtasks)}個のサブタスクを実行中...",
                    0.25
                )
                # Report delegation to Taisho
                await reporter.report_delegation(
                    task_id,
                    "gunshi",
                    "taisho",
                    f"{len(subtasks)}個のサブタスクを大将に委譲"
                )
            t0 = asyncio.get_event_loop().time()

            subtasks = await self._phase_do(subtasks)

            phase_times["do"] = asyncio.get_event_loop().time() - t0
            self._stats["total_do_time"] += phase_times["do"]
            completed = sum(1 for st in subtasks if st.status == "completed")
            self._stats["subtasks_completed"] += completed
            logger.info(
                f"⚔️ DO完了: {completed}/{len(subtasks)} 成功 "
                f"({phase_times['do']:.1f}s)"
            )

            # ===== Phase 3: CHECK (戦果検証 + ビジュアル検証) =====
            logger.info("🔍 Phase 3/4: CHECK (戦果検証)...")
            if reporter and task_id:
                await reporter.report_progress(
                    task_id,
                    "🔍 Phase 3/4: CHECK - 戦果検証中...",
                    0.50
                )
            t0 = asyncio.get_event_loop().time()

            passed, quality_score, verdicts = await self._phase_check(
                task_content, plan_summary, subtasks
            )

            # 検校 (Kengyo) ビジュアル検証
            visual_result = await self._phase_check_visual(
                task_content, subtasks, context
            )
            if visual_result:
                passed, quality_score = self._merge_visual_verdict(
                    passed, quality_score, visual_result
                )

            phase_times["check"] = asyncio.get_event_loop().time() - t0
            self._stats["total_check_time"] += phase_times["check"]
            visual_tag = " + 検校" if visual_result else ""
            logger.info(
                f"🔍 CHECK完了{visual_tag}: {'合格' if passed else '不合格'} "
                f"(品質 {quality_score:.0%}, {phase_times['check']:.1f}s)"
            )

            # ===== Phase 4: ACT (修正指示) - 不合格時のみ =====
            corrections_applied = 0
            if not passed and self.MAX_CORRECTION_LOOPS > 0:
                failed_verdicts = [v for v in verdicts if not v.passed]
                if failed_verdicts:
                    logger.info(
                        f"🔧 Phase 4/4: ACT ({len(failed_verdicts)}件修正)..."
                    )
                    if reporter and task_id:
                        await reporter.report_progress(
                            task_id,
                            f"🔧 Phase 4/4: ACT - {len(failed_verdicts)}件の修正指示を発行中...",
                            0.75
                        )
                    t0 = asyncio.get_event_loop().time()

                    subtasks, corrections_applied = await self._phase_act(
                        subtasks, verdicts
                    )

                    phase_times["act"] = asyncio.get_event_loop().time() - t0
                    self._stats["corrections_applied"] += corrections_applied
                    logger.info(
                        f"🔧 ACT完了: {corrections_applied}件修正 "
                        f"({phase_times['act']:.1f}s)"
                    )

                    # 再検証
                    if corrections_applied > 0:
                        t0 = asyncio.get_event_loop().time()
                        passed, quality_score, _ = await self._phase_check(
                            task_content, plan_summary, subtasks
                        )
                        phase_times["recheck"] = (
                            asyncio.get_event_loop().time() - t0
                        )
                        logger.info(
                            f"🔍 RE-CHECK: {'合格' if passed else '不合格'} "
                            f"(品質 {quality_score:.0%})"
                        )

            # ===== 結果統合 =====
            total_elapsed = asyncio.get_event_loop().time() - operation_start
            final_output = self._consolidate_results(subtasks)

            # 統計更新
            self._stats["pdca_cycles_completed"] += 1
            if passed:
                self._stats["verifications_passed"] += 1
            else:
                self._stats["verifications_failed"] += 1
            n = self._stats["pdca_cycles_completed"]
            self._stats["avg_quality_score"] = (
                (self._stats["avg_quality_score"] * (n - 1) + quality_score) / n
            )

            logger.info(
                f"🏯 PDCA作戦完了: 品質={quality_score:.0%} / "
                f"修正={corrections_applied}回 / 合計={total_elapsed:.1f}s"
            )

            # Report completion to Discord
            if reporter and task_id:
                await reporter.report_complete(
                    task_id,
                    f"PDCA作戦完了 (品質スコア: {quality_score:.0%}, 修正: {corrections_applied}回, 所要時間: {total_elapsed:.1f}秒)"
                )

            return OperationResult(
                task_content=task_content,
                plan_summary=plan_summary,
                subtasks=subtasks,
                verification_passed=passed,
                quality_score=quality_score,
                corrections_applied=corrections_applied,
                total_elapsed_seconds=total_elapsed,
                phase_times=phase_times,
                final_output=final_output
            )

        except Exception as e:
            total_elapsed = asyncio.get_event_loop().time() - operation_start
            logger.error(f"❌ PDCA作戦失敗: {e}")
            self._stats["fallbacks_to_karo"] += 1
            return OperationResult(
                task_content=task_content,
                plan_summary="",
                subtasks=[],
                verification_passed=False,
                quality_score=0.0,
                corrections_applied=0,
                total_elapsed_seconds=total_elapsed,
                phase_times=phase_times,
                error=str(e)
            )

    # ==================== Phase 1: PLAN ====================

    async def _phase_plan(
        self,
        task_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[SubTask]]:
        """
        Phase 1: PLAN (作戦立案 + タスク分解)

        Gunshi API (o3-mini, reasoning_effort=high) を使用。
        タスクを分析し、参謀のコンテキストに収まる
        サブタスクに分解する。

        Returns:
            (plan_summary, list_of_subtasks)
        """
        codebase_ctx = ""
        if context and context.get("codebase"):
            codebase_ctx = context["codebase"]

        plan_prompt = (
            "あなたは軍師（作戦立案者）です。\n"
            "以下の複雑タスクを分析し、実行計画を立案してください。\n\n"
            f"## タスク\n{task_content}\n\n"
        )
        if codebase_ctx:
            plan_prompt += f"## コードベース情報\n{codebase_ctx}\n\n"

        plan_prompt += (
            "## 制約\n"
            f"- サブタスクは最大{self.MAX_SUBTASKS}個\n"
            f"- 各サブタスクの説明は{self.SUBTASK_TOKEN_LIMIT}トークン以内"
            "（実装者のコンテキスト窓が4096トークンのため）\n"
            "- 各サブタスクは可能な限り独立させる（並列実行のため）\n"
            "- 依存関係がある場合は明示する\n\n"
            "## 出力形式 (JSON)\n"
            "```json\n"
            "{\n"
            '  "plan_summary": "作戦計画の概要（1-2文）",\n'
            '  "subtasks": [\n'
            "    {\n"
            '      "id": "ST-1",\n'
            '      "description": "具体的な実装指示'
            '（実装者がこれだけ見て作業できる粒度）",\n'
            '      "focused_context": "このサブタスクに必要な情報のみ",\n'
            '      "estimated_tokens": 2000,\n'
            '      "dependencies": [],\n'
            '      "priority": 1\n'
            "    }\n"
            "  ],\n"
            '  "risks": ["リスク1"],\n'
            '  "success_criteria": "成功基準"\n'
            "}\n"
            "```"
        )

        messages = [{"role": "user", "content": plan_prompt}]
        result_text = await self.client.generate(
            messages=messages,
            reasoning_effort=self.REASONING_EFFORT_PLAN,
            max_tokens=4096
        )

        plan_data = self._parse_json_response(result_text)

        if not plan_data or "subtasks" not in plan_data:
            # JSON解析失敗 → 単一タスクにフォールバック
            logger.warning("⚠️ 作戦計画のJSON解析失敗 → 単一タスクとして実行")
            return (
                result_text[:200],
                [SubTask(id="ST-1", description=task_content, priority=1)]
            )

        plan_summary = plan_data.get("plan_summary", "")
        subtasks = []
        for i, st_data in enumerate(
            plan_data["subtasks"][:self.MAX_SUBTASKS]
        ):
            subtasks.append(SubTask(
                id=st_data.get("id", f"ST-{i+1}"),
                description=st_data.get("description", ""),
                focused_context=st_data.get("focused_context", ""),
                estimated_tokens=st_data.get("estimated_tokens", 2000),
                dependencies=st_data.get("dependencies", []),
                priority=st_data.get("priority", i + 1)
            ))

        return plan_summary, subtasks

    # ==================== Phase 2: DO ====================

    async def _phase_do(self, subtasks: List[SubTask]) -> List[SubTask]:
        """
        Phase 2: DO (作戦実行)

        サブタスクの実行先:
        - 参謀A (GPT-5) / 参謀B (Grok): クラウドAPIで真の並列実行
        - フォールバック: GPT-5 → Grok → Gemini Flash

        依存関係処理:
        - 独立タスク → 参謀A/Bに振り分けて並列実行
        - 依存タスク → 依存解決後に順次実行
        """
        # 依存関係で分類
        independent = [st for st in subtasks if not st.dependencies]
        dependent = [st for st in subtasks if st.dependencies]

        # 独立タスクの並列実行
        if independent:
            sanbo_a = self.orchestrator.get_client("gpt5")
            sanbo_b = self.orchestrator.get_client("grok")
            if (sanbo_a or sanbo_b) and len(independent) > 1:
                # 参謀A/B で真の並列実行
                await self._execute_parallel_with_sanbo(
                    independent, sanbo_a, sanbo_b
                )
            else:
                # 単一タスクまたは参謀不在 → 個別実行
                await asyncio.gather(
                    *[self._execute_single_subtask(st) for st in independent],
                    return_exceptions=True
                )

        # 依存タスクを順次実行
        completed_ids = {
            st.id for st in subtasks if st.status == "completed"
        }
        for st in dependent:
            dep_ids = set(st.dependencies)
            if dep_ids.issubset(completed_ids):
                await self._execute_single_subtask(st)
                if st.status == "completed":
                    completed_ids.add(st.id)
            else:
                missing = dep_ids - completed_ids
                st.status = "failed"
                st.result = f"依存タスク未完了: {missing}"

        self._stats["subtasks_delegated"] += len(subtasks)
        return subtasks

    async def _execute_parallel_with_sanbo(
        self,
        subtasks: List[SubTask],
        sanbo_a,
        sanbo_b
    ) -> None:
        """
        参謀A (GPT-5) / 参謀B (Grok) で複数サブタスクを真に並列実行

        クラウド API なので asyncio.gather が実際に並列動作する。
        サブタスクを参謀A/Bに交互に振り分けて負荷分散する。
        """
        logger.info(
            f"⚔️ 参謀A/B 並列実行: {len(subtasks)} サブタスク"
        )

        start_time = asyncio.get_event_loop().time()

        async def _execute_with_client(st: SubTask, client, client_name: str):
            """指定クライアントでサブタスクを実行"""
            prompt = st.description
            if st.focused_context:
                prompt = (
                    f"## コンテキスト\n{st.focused_context}\n\n"
                    f"## タスク\n{st.description}"
                )
            try:
                st.status = "running"
                task_start = asyncio.get_event_loop().time()
                messages = [{
                    "role": "user",
                    "content": f"以下のタスクを実装してください。\n\n{prompt}"
                }]
                result_text = await client.generate(
                    messages=messages,
                    max_tokens=8192
                )
                st.result = result_text
                st.status = "completed"
                st.executed_by = f"sanbo/{client_name}"
                st.execution_time = (
                    asyncio.get_event_loop().time() - task_start
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ 参謀({client_name}) failed for {st.id}: {e}"
                )
                st.status = "failed"
                st.result = str(e)
                st.executed_by = f"sanbo/{client_name}/failed"

        # サブタスクを参謀A/Bに交互に振り分け
        tasks = []
        for i, st in enumerate(subtasks):
            if i % 2 == 0 and sanbo_a:
                tasks.append(_execute_with_client(st, sanbo_a, "gpt5"))
            elif sanbo_b:
                tasks.append(_execute_with_client(st, sanbo_b, "grok"))
            elif sanbo_a:
                tasks.append(_execute_with_client(st, sanbo_a, "gpt5"))
            else:
                st.status = "failed"
                st.result = "参謀クライアント不在"

        await asyncio.gather(*tasks, return_exceptions=True)

        # 失敗したサブタスクをフォールバック実行
        for st in subtasks:
            if st.status == "failed":
                logger.warning(
                    f"⚠️ 参謀失敗 → フォールバック実行: {st.id}"
                )
                await self._execute_single_subtask(st)

        total_time = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"✅ 参謀 並列実行完了: {total_time:.1f}s "
            f"({len([s for s in subtasks if s.status == 'completed'])}"
            f"/{len(subtasks)} 成功)"
        )

    async def _execute_single_subtask(self, subtask: SubTask) -> None:
        """
        単一サブタスクの実行

        フォールバックチェーン:
        1. 参謀A (GPT-5)
        2. 参謀B (Grok)
        3. Gemini Flash (緊急パス)
        """
        subtask.status = "running"
        start_time = asyncio.get_event_loop().time()

        # 実装者向けプロンプト構築
        impl_prompt = subtask.description
        if subtask.focused_context:
            impl_prompt = (
                f"## コンテキスト\n{subtask.focused_context}\n\n"
                f"## タスク\n{subtask.description}"
            )

        messages = [{
            "role": "user",
            "content": f"以下のタスクを実装してください。\n\n{impl_prompt}"
        }]

        # フォールバックチェーン: GPT-5 → Grok → Gemini Flash
        fallback_chain = [
            ("gpt5", "参謀A/GPT-5"),
            ("grok", "参謀B/Grok"),
            ("gemini_flash", "Gemini Flash"),
        ]

        for client_key, client_label in fallback_chain:
            client = self.orchestrator.get_client(client_key)
            if not client:
                continue
            try:
                result_text = await client.generate(
                    messages=messages,
                    max_tokens=8192
                )
                subtask.result = result_text
                subtask.status = "completed"
                subtask.executed_by = f"sanbo/{client_key}"
                subtask.execution_time = (
                    asyncio.get_event_loop().time() - start_time
                )
                return
            except Exception as e:
                logger.warning(
                    f"⚠️ {client_label} 実行失敗 ({subtask.id}): {e}"
                )
                continue

        # 全参謀失敗 → Gunshi API で自力実装 (緊急パス)
        logger.warning(
            f"⚠️ 参謀全滅 → Gunshi API自力実装 ({subtask.id})"
        )
        try:
            result_text = await self.client.generate(
                messages=messages,
                reasoning_effort=self.REASONING_EFFORT_ACT,
                max_tokens=4096
            )
            subtask.result = result_text
            subtask.status = "completed"
            subtask.executed_by = "gunshi_api"
        except Exception as e:
            logger.error(
                f"❌ Gunshi API自力実装も失敗 ({subtask.id}): {e}"
            )
            subtask.status = "failed"
            subtask.result = str(e)
            subtask.executed_by = "gunshi_api/failed"

        subtask.execution_time = (
            asyncio.get_event_loop().time() - start_time
        )

    # ==================== Phase 3: CHECK ====================

    async def _phase_check(
        self,
        original_task: str,
        plan_summary: str,
        subtasks: List[SubTask]
    ) -> Tuple[bool, float, List[VerificationVerdict]]:
        """
        Phase 3: CHECK (戦果検証)

        Gunshi API (o3-mini, reasoning_effort=high) で全結果を一括レビュー。
        全サブタスク結果を投入し、cross-file 整合性・品質を検証する。

        これが Gunshi の最大の価値:
        参謀は個別タスクしか見えないが、
        Gunshi は全サブタスクの結果を同時に見て整合性を確認できる。
        """
        # 全結果を統合して検証プロンプトを構築
        results_block = ""
        for st in subtasks:
            results_block += f"\n### {st.id}\n"
            results_block += f"指示: {st.description[:200]}\n"
            results_block += f"状態: {st.status} (実行: {st.executed_by})\n"
            if st.result:
                # 結果を適度に切り詰め (256K以内に収める)
                result_preview = st.result[:8000]
                if len(st.result) > 8000:
                    result_preview += f"\n... (以下 {len(st.result)-8000} chars 省略)"
                results_block += f"結果:\n{result_preview}\n"
            results_block += "---\n"

        verify_prompt = (
            "あなたは軍師（検証官）です。"
            "作戦の実行結果を厳密に検証してください。\n\n"
            f"## 元のタスク\n{original_task}\n\n"
            f"## 作戦計画\n{plan_summary}\n\n"
            f"## 実行結果\n{results_block}\n\n"
            "## 検証基準\n"
            "1. 元のタスクの要件を全て満たしているか\n"
            "2. サブタスク間の整合性（import, 型, インターフェース）\n"
            "3. コード品質（バグ, エッジケース, セキュリティ）\n"
            "4. 実装の完全性（未実装・TODO が残っていないか）\n\n"
            "## 出力形式 (JSON)\n"
            "```json\n"
            "{\n"
            '  "overall_passed": true,\n'
            '  "quality_score": 0.85,\n'
            '  "subtask_verdicts": [\n'
            "    {\n"
            '      "id": "ST-1",\n'
            '      "passed": true,\n'
            '      "issues": [],\n'
            '      "fix_instructions": ""\n'
            "    }\n"
            "  ],\n"
            '  "summary": "検証結果の概要"\n'
            "}\n"
            "```"
        )

        messages = [{"role": "user", "content": verify_prompt}]
        result_text = await self.client.generate(
            messages=messages,
            reasoning_effort=self.REASONING_EFFORT_CHECK,
            max_tokens=4096
        )

        verify_data = self._parse_json_response(result_text)

        if not verify_data:
            logger.warning("⚠️ 検証結果のJSON解析失敗 → 中間スコアで通過扱い")
            return True, 0.70, []

        overall_passed = verify_data.get("overall_passed", True)
        quality_score = min(1.0, max(0.0,
            verify_data.get("quality_score", 0.70)
        ))

        verdicts = []
        for v in verify_data.get("subtask_verdicts", []):
            verdicts.append(VerificationVerdict(
                subtask_id=v.get("id", ""),
                passed=v.get("passed", True),
                issues=v.get("issues", []),
                fix_instructions=v.get("fix_instructions", "")
            ))

        return overall_passed, quality_score, verdicts

    # ==================== Phase 3.5: CHECK Visual (検校) ====================

    async def _phase_check_visual(
        self,
        task_content: str,
        subtasks: List[SubTask],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Phase 3 拡張: 検校 (Kengyo) によるビジュアル検証

        UI 関連タスクの場合のみ実行。
        検校が未初期化、またはタスクに UI 要素がない場合はスキップ。

        Returns:
            ビジュアル検証結果 or None (スキップ時)
        """
        kengyo = self._get_kengyo()
        if not kengyo or not kengyo.is_available():
            return None

        subtask_results = [
            {
                "id": st.id,
                "description": st.description,
                "status": st.status,
                "result": st.result or "",
            }
            for st in subtasks
        ]

        result = await kengyo.check_phase_visual_verify(
            task_content=task_content,
            subtask_results=subtask_results,
            context=context,
        )
        return result

    def _get_kengyo(self):
        """検校 (Kengyo) インスタンス取得"""
        if hasattr(self.orchestrator, '_kengyo') and self.orchestrator._kengyo:
            return self.orchestrator._kengyo
        return None

    def _merge_visual_verdict(
        self,
        text_passed: bool,
        text_quality: float,
        visual_result: Dict[str, Any],
    ) -> tuple:
        """
        テキスト検証結果とビジュアル検証結果を統合

        ビジュアル検証は補助的な位置づけ:
        - テキスト合格 + ビジュアル不合格(重大) → 不合格に引き下げ
        - テキスト不合格 + ビジュアル合格 → 不合格のまま
        - 品質スコアはビジュアルを 20% で加重
        """
        visual_passed = visual_result.get("overall_passed", True)
        critical = visual_result.get("critical_issues", 0)

        # テキスト合格でもビジュアルに重大問題があれば不合格
        merged_passed = text_passed
        if text_passed and not visual_passed and critical > 0:
            merged_passed = False
            logger.info(
                f"👁️ 検校が重大問題を検出 ({critical}件) → 不合格に引き下げ"
            )

        # 品質スコア: テキスト 80% + ビジュアル 20%
        # ビジュアルスコアは issue 数で概算
        total_issues = visual_result.get("total_issues", 0)
        visual_score = max(0.0, 1.0 - (total_issues * 0.1))
        merged_quality = (text_quality * 0.8) + (visual_score * 0.2)
        merged_quality = min(1.0, max(0.0, merged_quality))

        return merged_passed, merged_quality

    # ==================== Phase 4: ACT ====================

    async def _phase_act(
        self,
        subtasks: List[SubTask],
        verdicts: List[VerificationVerdict]
    ) -> Tuple[List[SubTask], int]:
        """
        Phase 4: ACT (修正指示 → 再実行)

        不合格サブタスクに対して、Gunshi が具体的な修正指示を付けて
        参謀に再実装させる。最大1ループ。
        """
        corrections = 0
        verdict_map = {v.subtask_id: v for v in verdicts}

        for subtask in subtasks:
            verdict = verdict_map.get(subtask.id)
            if not verdict or verdict.passed:
                continue
            if not verdict.fix_instructions and not verdict.issues:
                continue

            issues_text = "\n".join(f"- {issue}" for issue in verdict.issues)
            correction_prompt = (
                "以下の実装に問題が見つかりました。修正してください。\n\n"
                f"## 元のタスク\n{subtask.description}\n\n"
                f"## 現在の実装\n{subtask.result}\n\n"
                f"## 問題点\n{issues_text}\n\n"
                f"## 修正指示\n{verdict.fix_instructions}\n\n"
                "修正後の完全な実装を出力してください。"
            )

            corrected = await self._try_execute(correction_prompt, subtask.id)

            if corrected:
                subtask.result = corrected
                subtask.status = "completed"
                corrections += 1
                logger.info(f"🔧 {subtask.id} 修正適用完了")

        return subtasks, corrections

    async def _try_execute(
        self, prompt: str, subtask_id: str
    ) -> Optional[str]:
        """
        修正実行: Taisho に委譲 (Taisho未初期化時のみ Gunshi API)

        Taisho 内部で Kimi→Local→Kagemusha→Gemini の4層フォールバックが動く。
        Gunshi API は Taisho が未初期化の場合のみ使用する緊急パス。
        """
        taisho = self.orchestrator.taisho
        if taisho:
            try:
                from core.taisho import ImplementationTask, ImplementationMode
                task = ImplementationTask(
                    content=prompt,
                    mode=ImplementationMode.STANDARD,
                    context={"from_gunshi": True, "correction": True}
                )
                result = await taisho.execute_implementation(task)
                if result.get("status") == "completed":
                    return result.get("result", "")
                logger.warning(
                    f"⚠️ Taisho修正失敗 ({subtask_id}): "
                    f"{result.get('error', 'non-completed status')}"
                )
                return None
            except Exception as e:
                logger.error(f"❌ Taisho修正例外 ({subtask_id}): {e}")
                return None

        # Taisho 未初期化 → Gunshi API 緊急パス
        logger.warning(f"⚠️ Taisho未初期化 → Gunshi API修正 ({subtask_id})")
        try:
            messages = [{"role": "user", "content": prompt}]
            return await self.client.generate(
                messages=messages,
                temperature=self.TEMP_ACT,
                max_tokens=4096
            )
        except Exception as e:
            logger.error(f"❌ Gunshi API修正も失敗 ({subtask_id}): {e}")
            return None

    # ==================== Helpers ====================

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """レスポンスからJSON部分を抽出・解析"""
        # 1. 直接 parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Markdown code block から抽出
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        # 3. テキスト中の JSON object を探索
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _consolidate_results(self, subtasks: List[SubTask]) -> str:
        """全サブタスクの結果を統合"""
        parts = []
        for st in subtasks:
            if st.status == "completed" and st.result:
                parts.append(f"## {st.id}\n{st.result}")
            elif st.status == "failed":
                parts.append(
                    f"## {st.id} [FAILED]\n{st.result or 'No result'}"
                )
        return "\n\n".join(parts)

    # ==================== Legacy Compatibility ====================

    async def process_complex_task(
        self,
        task_content: str,
        complexity: str,
        context: Optional[Dict[str, Any]] = None
    ) -> GunshiResult:
        """Legacy entry point → PDCA operation にリダイレクト"""
        operation = await self.execute_operation(
            task_content, complexity, context
        )

        return GunshiResult(
            mode=GunshiMode.STRATEGY,
            content=operation.final_output or operation.plan_summary,
            subtasks=[
                {
                    "id": st.id,
                    "description": st.description,
                    "status": st.status,
                    "executed_by": st.executed_by,
                }
                for st in operation.subtasks
            ],
            risk_assessment="",
            confidence=operation.quality_score,
            elapsed_seconds=operation.total_elapsed_seconds,
            operation_result=operation
        )

    # ==================== Statistics & BDI ====================

    def get_statistics(self) -> Dict[str, Any]:
        """統計取得"""
        return {
            "version": self.VERSION,
            "engine": "PDCA (Plan-Do-Check-Act)",
            "initialized": self.initialized,
            "model": "Qwen3-Coder-Next 80B-A3B (API)",
            "bdi_enabled": self.bdi_enabled,
            "config": {
                "max_subtasks": self.MAX_SUBTASKS,
                "max_correction_loops": self.MAX_CORRECTION_LOOPS,
                "subtask_token_limit": self.SUBTASK_TOKEN_LIMIT,
                "temp_plan": self.TEMP_PLAN,
                "temp_check": self.TEMP_CHECK,
                "temp_act": self.TEMP_ACT,
            },
            **self._stats
        }

    def get_bdi_state(self) -> Dict[str, Any]:
        """BDI状態取得"""
        return {
            "beliefs": self.beliefs,
            "desires": self.desires,
            "current_intentions": []
        }
