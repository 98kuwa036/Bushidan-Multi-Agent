#!/usr/bin/env python3
"""
武士団 週次スキル進化バッチ
systemd タイマーから毎週日曜 03:00 に実行される。

処理フロー:
  1. SkillEvolutionEngine で監査ログを分析・候補生成
  2. 目付（統計集計）: ロール別成功率・fallback率・レイテンシを数値化
  3. 軍師（Mistral）: 統計から型付き提案書を生成（JSON）
  4. 大元帥（ルール判定）: リスクレベルを評価して最終提案書を確定
  5. evolution_proposals テーブルに保存
  6. レポートファイル保存 + Webhook 通知（設定時のみ）
"""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass


# ── 目付: 統計集計 ────────────────────────────────────────────────────────────

async def _collect_stats() -> dict:
    """直近14日の成功率・フォールバック率・平均レイテンシを集計"""
    postgres_url = os.environ.get("POSTGRES_URL", "")
    if not postgres_url:
        return {}

    # 新カラムが確実に存在するようスキーマを保証する
    from utils.skill_tracker import ensure_schema as _ensure_tracker_schema
    await _ensure_tracker_schema()

    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(postgres_url, autocommit=True) as conn:
            async with conn.cursor() as cur:
                # success/had_hitl/used_fallback は ensure_schema() で追加済み
                await cur.execute("""
                    SELECT
                        role_used,
                        COUNT(*)                                                   AS total,
                        SUM(CASE WHEN success IS TRUE THEN 1 ELSE 0 END)          AS success_n,
                        SUM(CASE WHEN had_hitl IS TRUE THEN 1 ELSE 0 END)         AS hitl_n,
                        SUM(CASE WHEN used_fallback IS TRUE THEN 1 ELSE 0 END)    AS fallback_n,
                        ROUND(AVG(execution_time)::numeric, 2)                    AS avg_sec
                    FROM skill_observations
                    WHERE created_at >= NOW() - INTERVAL '14 days'
                    GROUP BY role_used
                    ORDER BY total DESC
                """)
                rows = await cur.fetchall()
                role_stats = {}
                for role, total, sc, hc, flc, avg_sec in rows:
                    role_stats[role] = {
                        "total":        total,
                        "success_rate": round((sc or 0) / total, 3) if total else 0.0,
                        "hitl_rate":    round((hc or 0) / total, 3) if total else 0.0,
                        "fallback_rate": round((flc or 0) / total, 3) if total else 0.0,
                        "avg_sec":      float(avg_sec or 0),
                    }

                # id (pattern_hash) も取得して提案時のターゲット指定に使う
                await cur.execute("""
                    SELECT id, keywords, typical_role, occurrence_count,
                           COALESCE(success_count, 0),
                           COALESCE(failure_count, 0),
                           COALESCE(fallback_count, 0),
                           COALESCE(avg_execution_ms, 0)
                    FROM skill_candidates
                    WHERE status = 'pending'
                      AND occurrence_count >= 5
                    ORDER BY COALESCE(failure_count, 0) DESC,
                             COALESCE(fallback_count, 0) DESC
                    LIMIT 10
                """)
                problem_rows = await cur.fetchall()
                problem_patterns = []
                for cid, kw_json, role, occ, sc2, fc2, flc2, avg_ms in problem_rows:
                    try:
                        kws = json.loads(kw_json) if kw_json else []
                    except Exception:
                        kws = []
                    total2 = (sc2 or 0) + (fc2 or 0)
                    problem_patterns.append({
                        "candidate_id":  cid,            # ← proposal の target に使う
                        "keywords":      kws[:5],
                        "role":          role,
                        "occurrence":    occ,
                        "success_rate":  round(sc2 / total2, 2) if total2 else 0.0,
                        "fallback_rate": round((flc2 or 0) / occ, 2) if occ else 0.0,
                        "avg_ms":        avg_ms or 0,
                    })
        return {"role_stats": role_stats, "problem_patterns": problem_patterns}
    except Exception as e:
        print(f"[metsuke] 集計エラー: {e}", file=sys.stderr)
        return {}


# ── 軍師: 型付き提案書生成 ────────────────────────────────────────────────────

_GUNSHI_SYSTEM = """あなたは武士団マルチエージェントシステムの軍師です。
与えられたシステム統計データを分析し、ルーティング改善のための提案書を JSON 形式で出力してください。

出力は必ず以下の JSON 配列のみを返してください（説明文は不要）:
[
  {
    "type": "routing_hint" | "threshold_adjust" | "new_skill" | "prune_skill",
    "target": "対象のID・ファイルパス・設定キー",
    "current_value": "現在の値",
    "proposed_value": "提案する値",
    "evidence": {"key": "value の証拠データ"},
    "rationale": "提案の根拠（1〜2文）"
  }
]

提案のルール:
- 証拠データが十分でない場合は提案しない（sample_count < 5 は無視）
- routing_hint: 成功率 < 0.6 かつ別のロールが頻繁に fallback 先になっている場合
- new_skill: 出現回数 >= 10 かつ 成功率 >= 0.7 のパターンを正式スキルに昇格
- 最大5件まで
"""


async def _gunshi_propose(stats: dict) -> list[dict]:
    """軍師（Mistral）が統計から型付き提案書を生成する"""
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    if not mistral_key:
        print("[gunshi] MISTRAL_API_KEY 未設定 — ルールベース提案にフォールバック")
        return _rule_based_proposals(stats)

    stats_text = json.dumps(stats, ensure_ascii=False, indent=2)
    try:
        from utils.mistral_client import MistralClient
        client = MistralClient(api_key=mistral_key)
        response = await asyncio.wait_for(
            client.generate(
                messages=[{"role": "user", "content": f"以下のシステム統計を分析して提案書を生成してください:\n\n{stats_text}"}],
                system=_GUNSHI_SYSTEM,
                model="mistral-large-latest",
                max_tokens=2000,
                temperature=0.2,
            ),
            timeout=60.0,
        )
        # JSON 部分を抽出
        text = response.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            proposals = json.loads(text[start:end])
            print(f"[gunshi] {len(proposals)} 件の提案書生成")
            return proposals
    except asyncio.TimeoutError:
        print("[gunshi] タイムアウト — ルールベース提案にフォールバック", file=sys.stderr)
    except Exception as e:
        print(f"[gunshi] エラー: {e} — ルールベース提案にフォールバック", file=sys.stderr)

    return _rule_based_proposals(stats)


def _rule_based_proposals(stats: dict) -> list[dict]:
    """LLM不使用のルールベース提案（フォールバック兼デフォルト）"""
    proposals = []
    role_stats = stats.get("role_stats", {})
    problem_patterns = stats.get("problem_patterns", [])

    # 成功率が低くフォールバック多発のロール → 最も成功率が高いロールへの変更を提案
    # target は typical_role（DBカラム名）と一致させ、_apply_proposal 側で WHERE typical_role で処理する
    for role, s in role_stats.items():
        if s["total"] >= 5 and s["success_rate"] < 0.6 and s["fallback_rate"] > 0.3:
            # フォールバック先として最も成功率が高いロールを探す
            best_alt = max(
                ((r, v) for r, v in role_stats.items() if r != role),
                key=lambda x: x[1]["success_rate"],
                default=(None, None),
            )
            proposed_role = best_alt[0] if best_alt[0] else "gunshi"
            proposals.append({
                "type":           "routing_hint",
                "target":         role,           # typical_role 値（WHERE typical_role = %s で使う）
                "current_value":  role,
                "proposed_value": proposed_role,
                "evidence":       {
                    "success_rate":  s["success_rate"],
                    "fallback_rate": s["fallback_rate"],
                    "sample_count":  s["total"],
                },
                "rationale": f"{role} の成功率が {s['success_rate']*100:.0f}% と低く、fallback が {s['fallback_rate']*100:.0f}% 発生しています。",
            })

    # 高頻度・高成功率パターンをスキル昇格
    # target は candidate_id (pattern_hash) を使う
    for p in problem_patterns:
        if p["occurrence"] >= 10 and p["success_rate"] >= 0.7:
            proposals.append({
                "type":           "new_skill",
                "target":         p["candidate_id"],   # ← DBの id (pattern_hash)
                "current_value":  "pending",
                "proposed_value": "approved",
                "evidence":       {
                    "keywords":     p["keywords"],
                    "occurrence":   p["occurrence"],
                    "success_rate": p["success_rate"],
                },
                "rationale": f"キーワード「{'・'.join(p['keywords'][:3])}」は {p['occurrence']} 回出現し成功率 {p['success_rate']*100:.0f}% — 正式スキルへの昇格を推奨。",
            })

    return proposals[:5]


# ── 大元帥: リスク判定 ────────────────────────────────────────────────────────

def _daigensui_assess(proposals: list[dict], stats: dict) -> list[dict]:
    """大元帥（ルールベース）が各提案のリスクレベルを判定して確定提案書を返す"""
    week = datetime.date.today().strftime("%Y-W%W")
    assessed = []
    for p in proposals:
        risk = _assess_risk(p, stats)
        pid = hashlib.md5(
            f"{week}:{p.get('type')}:{p.get('target')}".encode()
        ).hexdigest()[:12]
        assessed.append({
            **p,
            "id":         f"evprop-{pid}",
            "risk_level": risk,
            "week_label": week,
        })
    return assessed


def _assess_risk(proposal: dict, stats: dict) -> str:
    ptype = proposal.get("type", "")
    evidence = proposal.get("evidence", {})
    sample_count = evidence.get("sample_count", evidence.get("occurrence", 0))

    if sample_count < 5:
        return "high"

    if ptype == "prune_skill":
        return "medium"

    if ptype == "threshold_adjust":
        # 閾値変更は常に medium
        return "medium"

    if ptype == "routing_hint":
        success_rate = evidence.get("success_rate", 1.0)
        if success_rate < 0.4 and sample_count >= 10:
            return "low"    # 明確に問題あり
        return "medium"

    if ptype == "new_skill":
        if evidence.get("success_rate", 0) >= 0.8 and sample_count >= 15:
            return "low"
        return "medium"

    return "medium"


# ── レポートとレポートファイル ────────────────────────────────────────────────

def _build_report(evolution_result: dict, stats: dict, proposals: list[dict]) -> str:
    lines = ["📊 **武士団 週次進化レポート**\n"]

    lines.append("**スキル進化サイクル結果:**")
    lines.append(f"  ・分析パターン数: {evolution_result.get('patterns_analyzed', 0)}")
    lines.append(f"  ・新規候補: {evolution_result.get('new_candidates', 0)}")
    lines.append(f"  ・処理時間: {evolution_result.get('elapsed_ms', 0):.0f}ms\n")

    role_stats = stats.get("role_stats", {})
    if role_stats:
        lines.append("**ロール別成功率 (直近14日):**")
        for role, s in sorted(role_stats.items(), key=lambda x: x[1]["total"], reverse=True):
            bar = "✅" if s["success_rate"] >= 0.8 else ("⚠️" if s["success_rate"] >= 0.5 else "❌")
            lines.append(
                f"  {bar} {role}: {s['success_rate']*100:.0f}%成功 "
                f"({s['total']}件, avg {s['avg_sec']:.1f}s, fallback {s['fallback_rate']*100:.0f}%)"
            )
        lines.append("")

    if proposals:
        lines.append(f"**今週の提案書 ({len(proposals)}件):**")
        risk_label = {"low": "低", "medium": "中", "high": "高"}
        for i, p in enumerate(proposals, 1):
            risk = risk_label.get(p.get("risk_level", "medium"), "中")
            lines.append(f"  [{i}] リスク:{risk} | {p.get('type')} → {p.get('target')}")
            lines.append(f"       {p.get('rationale', '')}")
        lines.append("\n👉 GET /api/evolution-proposals で一覧確認")
        lines.append("👉 POST /api/evolution-proposals/{id}/approve で承認")
    else:
        lines.append("**今週の提案書: なし**（データ蓄積中）")

    return "\n".join(lines)


def _save_report_file(report: str) -> Path:
    report_dir = Path.home() / "bushidan_reports" / "evolution"
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = f"report-{datetime.date.today().isoformat()}.txt"
    path = report_dir / filename
    path.write_text(report, encoding="utf-8")
    latest = report_dir / "latest.txt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(filename)
    return path


async def _send_webhook(body: str) -> None:
    webhook_url = os.environ.get("BATCH_NOTIFY_WEBHOOK", "")
    if not webhook_url:
        return
    try:
        import aiohttp
        payload = {"type": "weekly_evolution_report", "body": body}
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                print(f"[webhook] status={resp.status}")
    except Exception as e:
        print(f"[webhook] スキップ: {e}", file=sys.stderr)


# ── メイン ───────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=== 武士団 週次スキル進化バッチ 開始 ===")

    # 1. SkillEvolutionEngine 実行
    from core.skill.skill_evolution import evolve_skills_from_audit
    evolution_result = await evolve_skills_from_audit(days=14)
    print(f"[evolution] {evolution_result}")

    # 2. 目付: 統計集計
    stats = await _collect_stats()
    print(f"[metsuke] ロール数: {len(stats.get('role_stats', {}))}, 問題パターン: {len(stats.get('problem_patterns', []))}")

    # 3. 軍師: 型付き提案書生成
    raw_proposals = await _gunshi_propose(stats)
    print(f"[gunshi] 提案書: {len(raw_proposals)} 件")

    # 4. 大元帥: リスク判定
    proposals = _daigensui_assess(raw_proposals, stats)
    low_risk  = [p for p in proposals if p.get("risk_level") == "low"]
    mid_risk  = [p for p in proposals if p.get("risk_level") == "medium"]
    print(f"[daigensui] 低リスク: {len(low_risk)} 件, 中リスク: {len(mid_risk)} 件")

    # 5. DB 保存
    from utils.evolution_proposals import save_proposals
    saved = await save_proposals(proposals)
    print(f"[db] 保存: {saved} 件")

    # 6. レポート生成・保存
    report = _build_report(evolution_result, stats, proposals)
    saved_path = _save_report_file(report)
    print(f"\n--- レポート ({saved_path}) ---")
    print(report)
    print("---\n")

    # 7. Webhook 通知（設定時のみ）
    await _send_webhook(report)

    print("=== 完了 ===")


if __name__ == "__main__":
    asyncio.run(main())
