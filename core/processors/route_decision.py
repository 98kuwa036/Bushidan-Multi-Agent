"""
武士団 v18 — route_decision (RouteDecision) プロセッサ
Phase 1 の集約情報から最適ロールを決定するデシジョンツリー
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from utils.logger import get_logger
from core.models.routing import KNOWN_ROLES, RoutingDecision, RoutingInput
from core.models.uchu import ComplexityLevel, IntentType

logger = get_logger(__name__)


# ─── ロール選出ルール定義 ─────────────────────────────────────────────────────

# (条件チェッカー, ロール, 信頼度, パス名)
_RULE_TYPE = Tuple[str, str, float]


def _rule_image(ri: RoutingInput) -> Optional[_RULE_TYPE]:
    if ri.intent.has_image:
        return ("image_detected", "kengyo", 0.97)
    return None


def _rule_secret(ri: RoutingInput) -> Optional[_RULE_TYPE]:
    if ri.intent.has_secret_keyword:
        return ("secret_keyword", "onmitsu", 0.97)
    return None


def _rule_explicit_role(ri: RoutingInput) -> Optional[_RULE_TYPE]:
    """Uchu が suggest したロールを高信頼で採用"""
    role = ri.intent.suggested_role
    if role and role in KNOWN_ROLES and role != "auto":
        return ("uchu_suggested", role, 0.90)
    return None


def _rule_notion_high_confidence(ri: RoutingInput) -> Optional[_RULE_TYPE]:
    """pgvector スキル検索が高スコアでマッチ"""
    if (ri.search.relevance_score >= 0.82
            and ri.search.suggested_role != "auto"
            and ri.search.suggested_role in KNOWN_ROLES):
        return ("notion_high", ri.search.suggested_role, 0.88)
    return None


def _rule_notion_medium_confidence(ri: RoutingInput) -> Optional[_RULE_TYPE]:
    """pgvector スキル検索がそこそこマッチ"""
    if (ri.search.relevance_score >= 0.65
            and ri.search.suggested_role != "auto"
            and ri.search.suggested_role in KNOWN_ROLES):
        return ("notion_medium", ri.search.suggested_role, 0.75)
    return None


def _rule_intent_complexity(ri: RoutingInput) -> Optional[_RULE_TYPE]:
    """Intent + Complexity の組み合わせルール"""
    it = ri.intent.intent_type
    cx = ri.intent.complexity

    # コード系
    if it == IntentType.CODE:
        if cx in (ComplexityLevel.COMPLEX, ComplexityLevel.STRATEGIC):
            return ("code_complex", "sanbo", 0.80)
        return ("code_normal", "yuhitsu", 0.78)

    # 分析系
    if it == IntentType.ANALYSIS:
        if cx in (ComplexityLevel.COMPLEX, ComplexityLevel.STRATEGIC):
            return ("analysis_complex", "gunshi", 0.80)
        return ("analysis_normal", "gunshi", 0.72)

    # 外部知識が必要
    if it == IntentType.RAG or ri.intent.requires_external_knowledge:
        return ("rag_needed", "gaiji", 0.78)

    # タスク系（ドキュメント等）
    if it == IntentType.TASK:
        if cx == ComplexityLevel.STRATEGIC:
            return ("task_strategic", "shogun", 0.82)
        if cx == ComplexityLevel.COMPLEX:
            return ("task_complex", "metsuke", 0.75)
        return ("task_normal", "yuhitsu", 0.70)

    # Q&A 系
    if it == IntentType.QA:
        if cx in (ComplexityLevel.COMPLEX, ComplexityLevel.STRATEGIC):
            return ("qa_complex", "gunshi", 0.72)
        if cx == ComplexityLevel.SIMPLE:
            return ("qa_simple", "uketuke", 0.80)
        return ("qa_normal", "uketuke", 0.70)

    return None


def _rule_complexity_fallback(ri: RoutingInput) -> Optional[_RULE_TYPE]:
    """複雑度だけでの最後のフォールバック"""
    cx = ri.intent.complexity
    mapping = {
        ComplexityLevel.SIMPLE: ("cx_simple", "uketuke", 0.65),
        ComplexityLevel.LOW_MEDIUM: ("cx_low_medium", "uketuke", 0.62),
        ComplexityLevel.MEDIUM: ("cx_medium", "yuhitsu", 0.60),
        ComplexityLevel.COMPLEX: ("cx_complex", "gunshi", 0.65),
        ComplexityLevel.STRATEGIC: ("cx_strategic", "shogun", 0.70),
    }
    return mapping.get(cx)


# ルールチェーン（上から順に評価、最初にマッチしたものを採用）
_RULES = [
    _rule_image,
    _rule_secret,
    _rule_explicit_role,
    _rule_notion_high_confidence,
    _rule_notion_medium_confidence,
    _rule_intent_complexity,
    _rule_complexity_fallback,
]


def _determine_content_type(ri: RoutingInput) -> str:
    """テキスト / コードの判定"""
    if ri.intent.intent_type == IntentType.CODE:
        return "code"
    if ri.intent.requires_code_execution:
        return "code"
    return "text"


class RouteDecisionProcessor:
    """route_decision プロセッサ"""

    def process(self, ri: RoutingInput) -> RoutingDecision:
        """デシジョンツリーでロールを決定（同期）"""
        start = time.time()
        path: List[str] = []

        selected_role = "auto"
        confidence = 0.5
        matched_path = "default_auto"

        for rule_fn in _RULES:
            result = rule_fn(ri)
            path.append(rule_fn.__name__)
            if result:
                matched_path, selected_role, confidence = result
                break

        # 外部知識ありの場合は gaiji を優先（intent ルールより後で上書き）
        if (ri.karasu and ri.karasu.search_results
                and ri.intent.requires_external_knowledge
                and selected_role not in ("kengyo", "onmitsu", "shogun")):
            if confidence < 0.82:
                selected_role = "gaiji"
                matched_path += "+karasu_injection"
                confidence = max(confidence, 0.73)

        # 最終バリデーション
        if selected_role not in KNOWN_ROLES:
            selected_role = "auto"
            confidence = 0.4

        content_type = _determine_content_type(ri)

        # 推論テキスト
        reasoning_parts = [
            f"intent={ri.intent.intent_type.value}",
            f"complexity={ri.intent.complexity.value}",
            f"notion_score={ri.search.relevance_score:.3f}",
            f"notion_role={ri.search.suggested_role}",
            f"rule={matched_path}",
        ]
        if ri.karasu:
            reasoning_parts.append(f"karasu_results={len(ri.karasu.search_results)}")

        elapsed_ms = (time.time() - start) * 1000
        logger.debug(
            "RouteDecision: %.1fms role=%s conf=%.2f path=%s",
            elapsed_ms, selected_role, confidence, matched_path,
        )

        return RoutingDecision(
            selected_role=selected_role,
            confidence=confidence,
            fallback_role="auto",
            decision_tree_path=path,
            decision_reasoning=" | ".join(reasoning_parts),
            processing_time_ms=elapsed_ms,
            intent_type=ri.intent.intent_type.value,
            complexity=ri.intent.complexity.value,
            content_type=content_type,
        )

    async def aprocess(self, ri: RoutingInput) -> RoutingDecision:
        """非同期ラッパー（LangGraph ノードから呼べるよう）"""
        return self.process(ri)
