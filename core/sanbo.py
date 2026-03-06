"""
Bushidan Multi-Agent System v11.4 - Sanbo (参謀: 実装実行層)

参謀層は軍師 PDCA の Do フェーズにおける主力実装エンジン。
GPT-5（参謀-A）と Grok-code-fast-1（参謀-B）の2モデル協調実装。

v11.4 参謀構成:
- 参謀-A (Sanbo-A): GPT-5 - 汎用コーディング・複雑なロジック
- 参謀-B (Sanbo-B): Grok-code-fast-1 - 実装特化・高速（~240 tok/s）

設計原則:
- 軍師（o3-mini）との役割分離: 軍師＝推論・計画, 参謀＝実装・コード生成
- 独立サブタスク → 参謀-B 並列実行（高速化）
- 依存サブタスク → 参謀-A 逐次実行（品質保証）
- フォールバック: 参謀-B → 参謀-A → 家老-A
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import get_logger

if TYPE_CHECKING:
    from core.system_orchestrator import SystemOrchestrator

logger = get_logger(__name__)


class SanboRole(Enum):
    """参謀の役割"""
    SANBO_A = "sanbo_a"  # GPT-5 汎用
    SANBO_B = "sanbo_b"  # Grok 高速


class SubtaskDependency(Enum):
    """サブタスク依存関係"""
    INDEPENDENT = "independent"    # 独立: 並列実行可
    DEPENDENT = "dependent"        # 依存: 逐次実行必要
    SEQUENTIAL = "sequential"      # 逐次: 前タスク完了後


@dataclass
class SanboSubtask:
    """参謀が実装するサブタスク"""
    id: str
    content: str
    dependency_type: SubtaskDependency = SubtaskDependency.INDEPENDENT
    depends_on: List[str] = field(default_factory=list)
    priority: int = 1
    assigned_role: Optional[SanboRole] = None
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class SanboStats:
    """参謀使用統計"""
    total_subtasks: int = 0
    sanbo_a_tasks: int = 0
    sanbo_b_tasks: int = 0
    parallel_executions: int = 0
    sequential_executions: int = 0
    fallback_to_karo_a: int = 0
    total_execution_time: float = 0.0
    average_subtask_time: float = 0.0


class Sanbo:
    """
    参謀 (Sanbo) - 実装実行層 v11.4

    軍師 PDCA の Do フェーズにおけるメイン実装エンジン。

    参謀-A (GPT-5):
    - 汎用コーディング・複雑なビジネスロジック
    - 依存関係があるサブタスクの逐次実行
    - コードレビュー・品質チェック補助

    参謀-B (Grok-code-fast-1):
    - 実装特化・高速コーディング (~240 tok/s)
    - 独立サブタスクの並列実行
    - バグ修正・パッチ適用

    フォールバックチェーン:
    参謀-B → 参謀-A → 家老-A (Gemini Flash)
    """

    VERSION = "11.4"

    def __init__(self, orchestrator: "SystemOrchestrator"):
        self.orchestrator = orchestrator
        self.stats = SanboStats()

        # クライアント参照（遅延初期化）
        self._gpt5_client = None   # 参謀-A
        self._grok_client = None   # 参謀-B
        self._gemini_client = None  # 家老-A フォールバック

        self._initialized = False

    async def initialize(self) -> None:
        """参謀層クライアントを初期化"""
        logger.info("⚔️ 参謀層 初期化開始...")

        # 参謀-A: GPT-5
        self._gpt5_client = self.orchestrator.get_client("gpt5")
        if self._gpt5_client:
            logger.info("✅ 参謀-A (GPT-5) クライアント取得完了")
        else:
            logger.warning("⚠️ 参謀-A (GPT-5) 未利用可 - OpenAI API キー要確認")

        # 参謀-B: Grok-code-fast-1
        self._grok_client = self.orchestrator.get_client("grok")
        if self._grok_client:
            logger.info("✅ 参謀-B (Grok-code-fast-1) クライアント取得完了")
        else:
            logger.warning("⚠️ 参謀-B (Grok) 未利用可 - xAI API キー要確認")

        # 家老-A フォールバック: Gemini Flash
        self._gemini_client = (
            self.orchestrator.get_client("gemini3")
            or self.orchestrator.get_client("gemini")
        )
        if self._gemini_client:
            logger.info("✅ 家老-A (Gemini Flash) フォールバック設定完了")

        self._initialized = True
        logger.info("⚔️ 参謀層 初期化完了")

    async def execute_subtasks(
        self,
        subtasks: List[SanboSubtask],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        サブタスク群を実行（独立→並列, 依存→逐次）

        軍師 PDCA の Do フェーズで呼び出される主要メソッド。

        Args:
            subtasks: 実行するサブタスクリスト
            context: 共有コンテキスト（ファイル情報等）

        Returns:
            {subtask_id: result} の辞書
        """
        if not self._initialized:
            await self.initialize()

        results: Dict[str, str] = {}
        start_time = time.time()

        logger.info(f"⚔️ 参謀: {len(subtasks)} サブタスク実行開始")

        # サブタスクを依存関係で分類
        independent = [t for t in subtasks if t.dependency_type == SubtaskDependency.INDEPENDENT]
        dependent = [t for t in subtasks if t.dependency_type != SubtaskDependency.INDEPENDENT]

        # 独立タスク: 参謀-B で並列実行
        if independent:
            logger.info(f"⚡ 独立タスク {len(independent)} 件 → 参謀-B 並列実行")
            parallel_results = await self._execute_parallel(independent, context)
            results.update(parallel_results)
            self.stats.parallel_executions += 1

        # 依存タスク: 参謀-A で逐次実行
        if dependent:
            logger.info(f"📋 依存タスク {len(dependent)} 件 → 参謀-A 逐次実行")
            sequential_results = await self._execute_sequential(dependent, results, context)
            results.update(sequential_results)
            self.stats.sequential_executions += 1

        elapsed = time.time() - start_time
        self.stats.total_subtasks += len(subtasks)
        self.stats.total_execution_time += elapsed

        if self.stats.total_subtasks > 0:
            self.stats.average_subtask_time = self.stats.total_execution_time / self.stats.total_subtasks

        logger.info(f"⚔️ 参謀: 全サブタスク完了 ({elapsed:.1f}s)")
        return results

    async def _execute_parallel(
        self,
        subtasks: List[SanboSubtask],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """独立サブタスクを参謀-B (Grok) で並列実行"""
        tasks = [
            self._execute_single(subtask, SanboRole.SANBO_B, context)
            for subtask in subtasks
        ]

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for subtask, result in zip(subtasks, results_list):
            if isinstance(result, Exception):
                logger.warning(f"参謀-B 失敗 [{subtask.id}]: {result} → 参謀-A にフォールバック")
                # フォールバック: 参謀-A で再実行
                try:
                    fallback_result = await self._execute_single(
                        subtask, SanboRole.SANBO_A, context
                    )
                    results[subtask.id] = fallback_result
                except Exception as e2:
                    logger.error(f"参謀-A フォールバック失敗 [{subtask.id}]: {e2}")
                    results[subtask.id] = f"[ERROR] {str(e2)}"
            else:
                results[subtask.id] = result

        return results

    async def _execute_sequential(
        self,
        subtasks: List[SanboSubtask],
        previous_results: Dict[str, str],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """依存サブタスクを参謀-A (GPT-5) で逐次実行"""
        results = {}

        # 依存順にソート（トポロジカルソート簡易版）
        ordered = self._sort_by_dependency(subtasks)

        for subtask in ordered:
            # 依存タスクの結果をコンテキストに追加
            dep_context = dict(context or {})
            for dep_id in subtask.depends_on:
                dep_result = previous_results.get(dep_id) or results.get(dep_id)
                if dep_result:
                    dep_context[f"dependency_{dep_id}"] = dep_result

            try:
                result = await self._execute_single(
                    subtask, SanboRole.SANBO_A, dep_context
                )
                results[subtask.id] = result
                previous_results[subtask.id] = result  # 後続タスクの参照用
            except Exception as e:
                logger.error(f"参謀-A 逐次実行失敗 [{subtask.id}]: {e}")
                results[subtask.id] = f"[ERROR] {str(e)}"

        return results

    async def _execute_single(
        self,
        subtask: SanboSubtask,
        preferred_role: SanboRole,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        単一サブタスクを指定役割で実行

        Args:
            subtask: 実行するサブタスク
            preferred_role: 優先する参謀役割
            context: コンテキスト

        Returns:
            実装結果テキスト
        """
        start_time = time.time()

        # プロンプト構築
        prompt = self._build_implementation_prompt(subtask, context)

        messages = [
            {
                "role": "system",
                "content": (
                    "あなたは高品質なコード実装エンジンです。"
                    "指定された要件を正確に実装してください。"
                    "動作するコードを出力してください。"
                )
            },
            {"role": "user", "content": prompt}
        ]

        # 役割に応じてクライアント選択
        client = None
        role_used = None

        if preferred_role == SanboRole.SANBO_B and self._grok_client:
            client = self._grok_client
            role_used = "参謀-B (Grok)"
        elif preferred_role == SanboRole.SANBO_A and self._gpt5_client:
            client = self._gpt5_client
            role_used = "参謀-A (GPT-5)"
        elif self._gpt5_client:
            client = self._gpt5_client
            role_used = "参謀-A (GPT-5) [フォールバック]"
        elif self._grok_client:
            client = self._grok_client
            role_used = "参謀-B (Grok) [フォールバック]"
        elif self._gemini_client:
            client = self._gemini_client
            role_used = "家老-A (Gemini) [最終防衛]"
            self.stats.fallback_to_karo_a += 1

        if not client:
            raise RuntimeError("参謀: 利用可能なクライアントがありません")

        logger.info(f"⚔️ {role_used}: サブタスク [{subtask.id}] 実行中...")

        result = await client.generate(messages)

        elapsed = time.time() - start_time
        subtask.execution_time = elapsed
        subtask.result = result

        if preferred_role == SanboRole.SANBO_A:
            self.stats.sanbo_a_tasks += 1
        else:
            self.stats.sanbo_b_tasks += 1

        logger.info(f"✅ {role_used}: [{subtask.id}] 完了 ({elapsed:.1f}s)")
        return result

    def _build_implementation_prompt(
        self,
        subtask: SanboSubtask,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """実装プロンプトを構築"""
        prompt_parts = [f"## 実装要求\n\n{subtask.content}"]

        if context:
            # 依存タスク結果を含める
            dep_results = {
                k: v for k, v in context.items()
                if k.startswith("dependency_")
            }
            if dep_results:
                prompt_parts.append("\n## 前タスクの実装結果\n")
                for dep_id, dep_result in dep_results.items():
                    task_id = dep_id.replace("dependency_", "")
                    prompt_parts.append(f"### {task_id}\n```\n{dep_result}\n```")

        return "\n".join(prompt_parts)

    def _sort_by_dependency(self, subtasks: List[SanboSubtask]) -> List[SanboSubtask]:
        """依存関係に基づいてサブタスクをトポロジカルソート（簡易版）"""
        sorted_tasks = []
        remaining = list(subtasks)
        processed_ids = set()

        max_iterations = len(subtasks) * 2
        iterations = 0

        while remaining and iterations < max_iterations:
            iterations += 1
            for task in remaining[:]:
                # 全依存タスクが処理済みかチェック
                if all(dep in processed_ids for dep in task.depends_on):
                    sorted_tasks.append(task)
                    processed_ids.add(task.id)
                    remaining.remove(task)

        # 循環依存等で残ったタスクをそのまま追加
        sorted_tasks.extend(remaining)
        return sorted_tasks

    async def implement_medium_task(
        self,
        task_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Medium タスクの直接実装（参謀-B 優先）

        軍師を経由しない Medium タスクの直接処理。
        参謀-B (Grok) を第1候補として使用。

        Args:
            task_content: 実装するタスク内容
            context: コンテキスト

        Returns:
            実装結果
        """
        if not self._initialized:
            await self.initialize()

        subtask = SanboSubtask(
            id="medium_task",
            content=task_content,
            dependency_type=SubtaskDependency.INDEPENDENT
        )

        result = await self._execute_single(subtask, SanboRole.SANBO_B, context)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """参謀層の使用統計を取得"""
        return {
            "tier": "sanbo",
            "version": self.VERSION,
            "clients": {
                "sanbo_a_gpt5": self._gpt5_client is not None,
                "sanbo_b_grok": self._grok_client is not None,
                "fallback_gemini": self._gemini_client is not None,
            },
            "total_subtasks": self.stats.total_subtasks,
            "sanbo_a_tasks": self.stats.sanbo_a_tasks,
            "sanbo_b_tasks": self.stats.sanbo_b_tasks,
            "parallel_executions": self.stats.parallel_executions,
            "sequential_executions": self.stats.sequential_executions,
            "fallback_to_karo_a": self.stats.fallback_to_karo_a,
            "total_execution_time": round(self.stats.total_execution_time, 1),
            "average_subtask_time": round(self.stats.average_subtask_time, 1),
        }

    def is_available(self) -> bool:
        """参謀層が利用可能か（少なくとも1クライアントが必要）"""
        return bool(self._gpt5_client or self._grok_client)
