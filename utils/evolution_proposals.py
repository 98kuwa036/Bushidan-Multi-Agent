"""
utils/evolution_proposals.py — 型付き進化提案の管理

週次バッチが生成した提案書を PostgreSQL に保存・取得・適用する。

提案の種類 (type):
  routing_hint      — skills/*.yaml の route_hint を変更
  threshold_adjust  — SemanticRouter の閾値を±0.05 調整
  new_skill         — skill_candidates を pending→承認
  prune_skill       — 30日未使用スキルを削除
"""
from __future__ import annotations

import datetime
import json
import os

import logging
logger = logging.getLogger(__name__)

POSTGRES_URL = os.environ.get("POSTGRES_URL", "")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evolution_proposals (
    id              TEXT PRIMARY KEY,
    proposal_type   TEXT NOT NULL,
    target          TEXT NOT NULL,
    current_value   TEXT DEFAULT '',
    proposed_value  TEXT DEFAULT '',
    evidence        TEXT DEFAULT '{}',
    risk_level      TEXT DEFAULT 'low',
    rationale       TEXT DEFAULT '',
    status          TEXT DEFAULT 'pending',
    created_at      TIMESTAMP DEFAULT NOW(),
    resolved_at     TIMESTAMP,
    applied_at      TIMESTAMP,
    week_label      TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_evprop_status ON evolution_proposals(status);
CREATE INDEX IF NOT EXISTS idx_evprop_week   ON evolution_proposals(week_label);
"""


async def ensure_schema() -> None:
    if not POSTGRES_URL:
        return
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                for stmt in _SCHEMA_SQL.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        await cur.execute(stmt)
    except Exception as e:
        logger.warning("evolution_proposals schema init 失敗: %s", e)


async def save_proposals(proposals: list[dict]) -> int:
    """バッチ生成の提案書を保存（同一IDは重複スキップ）"""
    if not proposals or not POSTGRES_URL:
        return 0
    await ensure_schema()
    try:
        import psycopg
        count = 0
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                for p in proposals:
                    await cur.execute(
                        "SELECT id FROM evolution_proposals WHERE id = %s", (p["id"],)
                    )
                    if await cur.fetchone():
                        continue
                    await cur.execute(
                        """INSERT INTO evolution_proposals
                           (id, proposal_type, target, current_value, proposed_value,
                            evidence, risk_level, rationale, week_label)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (
                            p["id"],
                            p.get("type", ""),
                            p.get("target", ""),
                            json.dumps(p.get("current_value", ""), ensure_ascii=False),
                            json.dumps(p.get("proposed_value", ""), ensure_ascii=False),
                            json.dumps(p.get("evidence", {}), ensure_ascii=False),
                            p.get("risk_level", "low"),
                            p.get("rationale", ""),
                            p.get("week_label", ""),
                        ),
                    )
                    count += 1
        return count
    except Exception as e:
        logger.error("save_proposals 失敗: %s", e)
        return 0


async def list_evolution_proposals(status: str = "pending") -> list[dict]:
    if not POSTGRES_URL:
        return []
    await ensure_schema()
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT id, proposal_type, target, current_value, proposed_value,
                              evidence, risk_level, rationale, status, created_at, week_label
                       FROM evolution_proposals
                       WHERE status = %s
                       ORDER BY created_at DESC LIMIT 50""",
                    (status,),
                )
                rows = await cur.fetchall()
        result = []
        for r in rows:
            result.append({
                "id":             r[0],
                "type":           r[1],
                "target":         r[2],
                "current_value":  _try_json(r[3]),
                "proposed_value": _try_json(r[4]),
                "evidence":       _try_json(r[5]),
                "risk_level":     r[6],
                "rationale":      r[7],
                "status":         r[8],
                "created_at":     r[9].isoformat() if r[9] else None,
                "week_label":     r[10],
            })
        return result
    except Exception as e:
        logger.warning("list_evolution_proposals 失敗: %s", e)
        return []


async def approve_evolution_proposal(proposal_id: str) -> dict:
    """提案を承認して適用する"""
    if not POSTGRES_URL:
        return {"ok": False, "error": "POSTGRES_URL未設定"}
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT proposal_type, target, proposed_value FROM evolution_proposals WHERE id = %s",
                    (proposal_id,),
                )
                row = await cur.fetchone()
                if not row:
                    return {"ok": False, "error": "提案が見つかりません"}

                ptype, target, proposed_raw = row
                proposed = _try_json(proposed_raw)

                applied = await _apply_proposal(ptype, target, proposed)

                await cur.execute(
                    "UPDATE evolution_proposals SET status='approved', resolved_at=NOW(), applied_at=NOW() WHERE id = %s",
                    (proposal_id,),
                )

        logger.info("✅ 進化提案 承認・適用: %s (%s → %s)", proposal_id, ptype, target)
        return {"ok": True, "applied": applied}
    except Exception as e:
        logger.error("approve_evolution_proposal 失敗: %s", e)
        return {"ok": False, "error": str(e)}


async def dismiss_evolution_proposal(proposal_id: str) -> dict:
    if not POSTGRES_URL:
        return {"ok": False, "error": "POSTGRES_URL未設定"}
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE evolution_proposals SET status='dismissed', resolved_at=NOW() WHERE id = %s",
                    (proposal_id,),
                )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _apply_proposal(ptype: str, target: str, proposed) -> str:
    """提案を実際のシステムに適用する"""
    if ptype == "routing_hint":
        # target は典型ロール名（例: "sanbo"）
        # 1) 稼働中の skills/*.yaml の route_hint を書き換える（最優先）
        # 2) pending な skill_candidates の typical_role も更新する
        from pathlib import Path
        import re as _re

        skills_dir = Path(os.environ.get("SKILLS_DIR", str(Path(__file__).parent.parent / "skills")))
        yaml_updated = 0
        if skills_dir.exists():
            for skill_file in skills_dir.glob("skill-*.yaml"):
                try:
                    text = skill_file.read_text(encoding="utf-8")
                    # route_hint: "sanbo" の行を書き換える
                    new_text = _re.sub(
                        r'^(route_hint:\s*")[^"]*(")',
                        lambda m: m.group(0),  # まず全マッチを取得
                        text,
                        flags=_re.MULTILINE,
                    )
                    # route_hint の値が target と一致するファイルのみ更新
                    if _re.search(
                        rf'^route_hint:\s*["\']?{_re.escape(target)}["\']?\s*$',
                        text, _re.MULTILINE
                    ):
                        new_text = _re.sub(
                            rf'^(route_hint:\s*)["\']?{_re.escape(target)}["\']?(\s*)$',
                            rf'\g<1>"{proposed}"\2',
                            text,
                            flags=_re.MULTILINE,
                        )
                        skill_file.write_text(new_text, encoding="utf-8")
                        yaml_updated += 1
                except Exception:
                    pass

        # インメモリキャッシュを再ロード
        try:
            from utils.skill_tracker import _load_skills
            _load_skills()
        except Exception:
            pass

        # pending 候補の typical_role も更新
        db_updated = 0
        try:
            import psycopg
            async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """UPDATE skill_candidates
                           SET typical_role = %s
                           WHERE typical_role = %s AND status = 'pending'""",
                        (proposed, target),
                    )
                    db_updated = cur.rowcount
        except Exception as e:
            logger.warning("routing_hint DB update 失敗: %s", e)

        return f"routing_hint: yaml={yaml_updated}件, db_pending={db_updated}件 ({target} → {proposed})"

    elif ptype == "threshold_adjust":
        from pathlib import Path
        import json as _json
        kpath = Path(__file__).parent.parent / "config" / "semantic_knowledge.json"
        try:
            if kpath.exists():
                data = _json.loads(kpath.read_text(encoding="utf-8"))
            else:
                data = {"version": 1, "roles": {}}
            data.setdefault("thresholds", {})[target] = proposed
            data["updated_at"] = datetime.date.today().isoformat()
            kpath.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"threshold {target} → {proposed}"
        except Exception as e:
            return f"threshold update 失敗: {e}"

    elif ptype == "new_skill":
        # target は candidate の id (pattern_hash)
        # skill_tracker.approve_proposal() を使うことで
        # DB更新 + skills/*.yaml 書き出し + インメモリ再ロードを一括実行する
        try:
            import psycopg
            # スキル名と典型ロールを DB から取得
            async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT skill_name, typical_role, keywords FROM skill_candidates WHERE id = %s",
                        (target,),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return f"skill_candidate not found: {target}"
                    skill_name_db, typical_role, kw_json = row

            skill_name = skill_name_db or f"自動スキル_{target[:8]}"
            route_hint = typical_role or "gunshi"

            from utils.skill_tracker import approve_proposal
            result = await approve_proposal(
                proposal_id=target,
                skill_name=skill_name,
                route_hint=route_hint,
                system_prompt_hint="",
            )
            if result.get("ok"):
                return f"skill approved + YAML written: {result.get('skill_file', '')}"
            return f"skill approval 失敗: {result.get('error', 'unknown')}"
        except Exception as e:
            return f"skill approval 失敗: {e}"

    elif ptype == "prune_skill":
        from pathlib import Path
        skill_path = Path(target)
        if skill_path.exists():
            skill_path.unlink()
            return f"pruned: {skill_path.name}"
        return f"already removed: {target}"

    return f"unknown type: {ptype}"


def _try_json(val):
    if val is None:
        return val
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return val
