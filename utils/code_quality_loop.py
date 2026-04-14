"""utils/code_quality_loop.py — Groq × Gemini Flash コード品質ループ

フロー:
  1. Groq が生成したコードを Gemini Flash（参謀）がレビュー
  2. 指摘あり → Groq が修正 → 再レビュー
  3. 「LGTM」or 指摘0件 → 完了
  4. 複雑なロジックエラー検知時 → 軍師（Command A）にエスカレーション

最大イテレーション: 3回（超えたら現状コードで返す）
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ロジック系エスカレーションキーワード
_ESCALATION_KWS = [
    "ロジックエラー", "アルゴリズム", "設計", "アーキテクチャ", "根本的",
    "logic error", "algorithm", "design flaw", "fundamental",
]

_LGTM_PATTERNS = [
    r"\blgtm\b", r"問題なし", r"指摘なし", r"問題ありません",
    r"修正不要", r"良好", r"パスします", r"合格",
]


def _is_lgtm(review: str) -> bool:
    lower = review.lower().strip()
    if len(lower) < 30:  # 短い応答はLGTMとみなす
        return True
    return any(re.search(p, lower) for p in _LGTM_PATTERNS)


def _needs_escalation(review: str) -> bool:
    return any(kw in review for kw in _ESCALATION_KWS)


def _extract_code(text: str, language: str = "") -> Optional[str]:
    """レスポンスからコードブロックを抽出"""
    # 言語指定ありで検索
    if language:
        m = re.search(rf'```{re.escape(language)}\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # 言語問わず最初のコードブロック
    m = re.search(r'```\w*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


async def run_review_loop(
    code: str,
    requirements: str,
    language: str,
    groq_client,
    review_client,          # Gemini Flash（参謀）
    escalation_client,      # Command A（軍師）
    max_iterations: int = 3,
) -> tuple[str, list[dict], int]:
    """
    コード品質ループを実行する。

    Returns:
        (最終コード, レビュー履歴, 実施ラウンド数)
    """
    current_code = code
    history = []

    for i in range(max_iterations):
        round_num = i + 1
        logger.info("🔄 コードレビュー ラウンド %d/%d", round_num, max_iterations)

        # ── レビュー ──────────────────────────────────────────────────
        should_escalate = (
            i > 0
            and bool(history)
            and escalation_client is not None
            and _needs_escalation(history[-1]["review"])
        )
        reviewer = escalation_client if should_escalate else review_client
        reviewer_name = "軍師（Command A）" if should_escalate else "参謀（Gemini Flash）"

        review_resp = await reviewer.generate(
            messages=[{"role": "user", "content": (
                f"以下の{language}コードをレビューしてください。\n\n"
                f"【要件】\n{requirements}\n\n"
                f"【コード】\n```{language}\n{current_code}\n```\n\n"
                "問題点があれば箇条書きで具体的に指摘してください。"
                "問題がなければ「LGTM」とだけ答えてください。"
            )}],
            system=(
                "あなたはコードレビュアーです。"
                "セキュリティ・バグ・パフォーマンス・可読性の観点でレビューしてください。"
                "問題がなければ必ず「LGTM」とだけ答えてください。余分な説明不要。"
            ),
            max_tokens=1024,
        )

        history.append({
            "round": round_num,
            "reviewer": reviewer_name,
            "review": review_resp,
            "lgtm": _is_lgtm(review_resp),
        })
        logger.info("📋 ラウンド%d レビュー結果 [%s]: %s...", round_num, reviewer_name, review_resp[:80])

        if _is_lgtm(review_resp):
            logger.info("✅ コードレビュー合格 (%d ラウンド)", round_num)
            break

        if i == max_iterations - 1:
            logger.info("⚠️ 最大ラウンド到達 — 現状コードで返却")
            break

        # ── Groq で修正 ───────────────────────────────────────────────
        fix_resp = await groq_client.generate(
            messages=[{"role": "user", "content": (
                f"以下のコードにレビュー指摘があります。修正してください。\n\n"
                f"【元のコード】\n```{language}\n{current_code}\n```\n\n"
                f"【指摘事項】\n{review_resp}\n\n"
                f"修正後のコードのみを返してください。説明不要。"
            )}],
            system="コード修正専門家。修正済みコードのみ返す。余分な説明は不要。",
            max_tokens=2048,
        )

        fixed = _extract_code(fix_resp, language)
        if fixed:
            current_code = fixed
            logger.info("🔧 ラウンド%d 修正完了", round_num)
        else:
            logger.warning("⚠️ ラウンド%d: コード抽出失敗 — 元コードを維持", round_num)

    return current_code, history, len(history)


def format_review_summary(history: list[dict]) -> str:
    """レビュー履歴をユーザー向けサマリーに整形"""
    if not history:
        return ""

    rounds = len(history)
    final = history[-1]

    if final["lgtm"]:
        if rounds == 1:
            summary = "✅ **初回レビューで合格**"
        else:
            summary = f"✅ **{rounds}ラウンドでレビュー合格**"
    else:
        summary = f"⚠️ **{rounds}ラウンド実施（未解決の指摘あり）**"

    details = []
    for h in history:
        status = "✅ LGTM" if h["lgtm"] else f"📋 指摘あり"
        details.append(f"  Round {h['round']} [{h['reviewer']}]: {status}")

    return summary + "\n" + "\n".join(details)
