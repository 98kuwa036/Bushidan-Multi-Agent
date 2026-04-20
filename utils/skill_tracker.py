"""utils/skill_tracker.py — スキル自動進化システム

Multi-Agent Shogun の「ボトムアップスキル発見」を Bushidan に統合。

フロー:
  1. チャット実行後に observe() を呼ぶ
  2. PostgreSQL の skill_observations テーブルに記録
  3. 類似パターンが SKILL_PROPOSE_THRESHOLD 回以上溜まると
     skill_candidates に候補を作成
  4. /api/skill-proposals で一覧取得
  5. 承認 → skills/*.yaml に書き出し + ルーティングヒント有効化

パターン検出方式:
  - シンプルなキーワード頻度 (TF-IDF 不要)
  - 名詞・動詞の重要語句 (N-gram ハッシュ) でクラスタリング
  - 同一ハッシュが threshold 回以上 → 候補生成
"""

import asyncio
import datetime
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

POSTGRES_URL = os.environ.get("POSTGRES_URL")
if not POSTGRES_URL:
    raise RuntimeError("環境変数 POSTGRES_URL が未設定です。.env を確認してください。")

# 候補生成のしきい値 (同一パターンが N 回以上で提案)
SKILL_PROPOSE_THRESHOLD = int(os.environ.get("SKILL_PROPOSE_THRESHOLD", "3"))

# 承認済みスキルの保存先
_SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", "/mnt/Bushidan-Multi-Agent/skills"))

# 観察対象の最小メッセージ長 (短い挨拶等はスキップ)
_MIN_OBS_LEN = 15


# ── キーワード抽出 ─────────────────────────────────────────────────────

_STOP_WORDS = frozenset([
    "て", "に", "を", "は", "が", "で", "の", "と", "も", "から", "まで",
    "ます", "です", "ください", "お", "ご", "ある", "いる", "する", "なる",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "it", "for", "on", "with", "as", "at", "by",
    "please", "help", "me", "my", "i", "you", "can", "this", "that",
])


def _extract_keywords(text: str, max_kw: int = 8) -> list[str]:
    """テキストから主要キーワードを抽出 (最大 max_kw 件)"""
    # 記号・句読点を除去
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = cleaned.split()
    # ストップワード除去 + 短すぎるトークン除去
    kws = [t for t in tokens if len(t) >= 3 and t not in _STOP_WORDS]
    # 出現頻度でソート (多い順)
    from collections import Counter
    freq = Counter(kws)
    return [w for w, _ in freq.most_common(max_kw)]


def _pattern_hash(keywords: list[str]) -> str:
    """キーワードリストから再現可能なハッシュを生成"""
    normalized = sorted(set(keywords))[:5]  # 上位5語でハッシュ
    key = "|".join(normalized)
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ── DB スキーマ初期化 ─────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS skill_observations (
    id          SERIAL PRIMARY KEY,
    thread_id   TEXT,
    message     TEXT,
    role_used   TEXT,
    pattern_hash TEXT,
    keywords    TEXT,           -- JSON array
    execution_time FLOAT,
    success     BOOLEAN DEFAULT TRUE,
    error_msg   TEXT DEFAULT '',
    had_hitl    BOOLEAN DEFAULT FALSE,
    used_fallback BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP DEFAULT NOW()
);

ALTER TABLE skill_observations ADD COLUMN IF NOT EXISTS success BOOLEAN DEFAULT TRUE;
ALTER TABLE skill_observations ADD COLUMN IF NOT EXISTS error_msg TEXT DEFAULT '';
ALTER TABLE skill_observations ADD COLUMN IF NOT EXISTS had_hitl BOOLEAN DEFAULT FALSE;
ALTER TABLE skill_observations ADD COLUMN IF NOT EXISTS used_fallback BOOLEAN DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS skill_candidates (
    id              TEXT PRIMARY KEY,  -- pattern_hash
    pattern_hash    TEXT UNIQUE NOT NULL,
    keywords        TEXT,              -- JSON array
    sample_messages TEXT,              -- JSON array (最大5件)
    typical_role    TEXT,
    occurrence_count INT DEFAULT 0,
    success_count   INT DEFAULT 0,
    failure_count   INT DEFAULT 0,
    hitl_count      INT DEFAULT 0,
    fallback_count  INT DEFAULT 0,
    avg_execution_ms FLOAT DEFAULT 0.0,
    status          TEXT DEFAULT 'pending',  -- pending|approved|dismissed
    proposed_at     TIMESTAMP DEFAULT NOW(),
    approved_at     TIMESTAMP,
    skill_name      TEXT,
    skill_file      TEXT
);

ALTER TABLE skill_candidates ADD COLUMN IF NOT EXISTS success_count INT DEFAULT 0;
ALTER TABLE skill_candidates ADD COLUMN IF NOT EXISTS failure_count INT DEFAULT 0;
ALTER TABLE skill_candidates ADD COLUMN IF NOT EXISTS hitl_count INT DEFAULT 0;
ALTER TABLE skill_candidates ADD COLUMN IF NOT EXISTS fallback_count INT DEFAULT 0;
ALTER TABLE skill_candidates ADD COLUMN IF NOT EXISTS avg_execution_ms FLOAT DEFAULT 0.0;

CREATE INDEX IF NOT EXISTS idx_obs_hash ON skill_observations(pattern_hash);
CREATE INDEX IF NOT EXISTS idx_obs_created ON skill_observations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cand_status ON skill_candidates(status);
"""


async def ensure_schema() -> None:
    """DB スキーマを初期化する (起動時に1回呼ぶ)"""
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True) as conn:
            async with conn.cursor() as cur:
                for stmt in _SCHEMA_SQL.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        await cur.execute(stmt)
        logger.info("✅ SkillTracker: DB schema ready")
    except Exception as e:
        logger.warning("SkillTracker schema init 失敗: %s", e)


# ── 観察・集計 ──────────────────────────────────────────────────────

async def observe(
    thread_id: str,
    message: str,
    role_used: str,
    execution_time: float,
    success: bool = True,
    error: str = "",
    had_hitl: bool = False,
    used_fallback: bool = False,
) -> None:
    """
    チャット実行後に呼ぶ。パターンを記録し、候補生成を検討する。

    Args:
        thread_id:      スレッド ID
        message:        ユーザーのメッセージ
        role_used:      実際に処理したロール (handled_by または agent_role)
        execution_time: 応答時間 (秒)
        success:        タスクが正常完了したか
        error:          エラーメッセージ（あれば）
        had_hitl:       Human-in-the-loop が発生したか
        used_fallback:  フォールバックが発動したか
    """
    if not message or len(message) < _MIN_OBS_LEN:
        return

    try:
        keywords = _extract_keywords(message)
        if not keywords:
            return

        phash = _pattern_hash(keywords)

        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
            async with conn.cursor() as cur:
                # 観察記録
                await cur.execute(
                    """INSERT INTO skill_observations
                       (thread_id, message, role_used, pattern_hash, keywords, execution_time,
                        success, error_msg, had_hitl, used_fallback)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        thread_id,
                        message[:300],
                        role_used,
                        phash,
                        json.dumps(keywords, ensure_ascii=False),
                        round(execution_time, 2),
                        success,
                        error[:200] if error else "",
                        had_hitl,
                        used_fallback,
                    ),
                )

                # 同一ハッシュの件数確認
                await cur.execute(
                    "SELECT COUNT(*) FROM skill_observations WHERE pattern_hash = %s",
                    (phash,),
                )
                row = await cur.fetchone()
                count = row[0] if row else 0

                if count >= SKILL_PROPOSE_THRESHOLD:
                    await _maybe_create_candidate(
                        cur, phash, keywords, role_used, count,
                        success=success, had_hitl=had_hitl,
                        used_fallback=used_fallback, execution_time=execution_time,
                    )

            await conn.commit()

    except Exception as e:
        logger.error("SkillTracker.observe 失敗: %s", e)


async def _maybe_create_candidate(
    cur, phash: str, keywords: list[str], role: str, count: int,
    success: bool = True, had_hitl: bool = False, used_fallback: bool = False,
    execution_time: float = 0.0,
) -> None:
    """候補が未存在 or pending の場合、件数・成功率を更新/挿入する"""
    await cur.execute(
        "SELECT id, status, success_count, failure_count, hitl_count, fallback_count, avg_execution_ms FROM skill_candidates WHERE pattern_hash = %s",
        (phash,),
    )
    existing = await cur.fetchone()

    if existing:
        if existing[1] == "dismissed":
            return
        sc = (existing[2] or 0) + (1 if success else 0)
        fc = (existing[3] or 0) + (0 if success else 1)
        hc = (existing[4] or 0) + (1 if had_hitl else 0)
        flc = (existing[5] or 0) + (1 if used_fallback else 0)
        prev_avg = existing[6] or 0.0
        new_avg = (prev_avg * (count - 1) + execution_time * 1000) / count if count > 0 else 0.0
        await cur.execute(
            """UPDATE skill_candidates
               SET occurrence_count = %s, success_count = %s, failure_count = %s,
                   hitl_count = %s, fallback_count = %s, avg_execution_ms = %s
               WHERE pattern_hash = %s""",
            (count, sc, fc, hc, flc, round(new_avg, 1), phash),
        )
    else:
        await cur.execute(
            """SELECT message FROM skill_observations
               WHERE pattern_hash = %s ORDER BY created_at DESC LIMIT 5""",
            (phash,),
        )
        samples = [r[0] for r in await cur.fetchall()]

        await cur.execute(
            """INSERT INTO skill_candidates
               (id, pattern_hash, keywords, sample_messages, typical_role,
                occurrence_count, success_count, failure_count, hitl_count,
                fallback_count, avg_execution_ms, status, proposed_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', NOW())""",
            (
                phash, phash,
                json.dumps(keywords, ensure_ascii=False),
                json.dumps(samples, ensure_ascii=False),
                role, count,
                1 if success else 0,
                0 if success else 1,
                1 if had_hitl else 0,
                1 if used_fallback else 0,
                round(execution_time * 1000, 1),
            ),
        )
        logger.info(
            "💡 スキル候補を提案: hash=%s keywords=%s (出現%d回, 成功=%s)",
            phash, keywords[:3], count, success,
        )


# ── 候補一覧・承認 API ─────────────────────────────────────────────

async def list_proposals(status: str = "pending") -> list[dict]:
    """スキル候補一覧を返す"""
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """SELECT id, keywords, sample_messages, typical_role,
                              occurrence_count, status, proposed_at, skill_name, skill_file
                       FROM skill_candidates
                       WHERE status = %s
                       ORDER BY occurrence_count DESC, proposed_at DESC
                       LIMIT 50""",
                    (status,),
                )
                rows = await cur.fetchall()

        result = []
        for r in rows:
            result.append({
                "id":              r[0],
                "keywords":        json.loads(r[1]) if r[1] else [],
                "sample_messages": json.loads(r[2]) if r[2] else [],
                "typical_role":    r[3],
                "occurrence_count": r[4],
                "status":          r[5],
                "proposed_at":     r[6].isoformat() if r[6] else None,
                "skill_name":      r[7],
                "skill_file":      r[8],
            })
        return result
    except Exception as e:
        logger.warning("list_proposals 失敗: %s", e)
        return []


async def approve_proposal(
    proposal_id: str,
    skill_name: str,
    route_hint: str,
    system_prompt_hint: str = "",
) -> dict:
    """
    スキル候補を承認し、skills/*.yaml に書き出す。

    Args:
        proposal_id:        候補 ID (pattern_hash)
        skill_name:         スキル名 (例: "PostgreSQL最適化相談")
        route_hint:         推奨ルーティング先 (例: "gunshi")
        system_prompt_hint: システムプロンプトへの追加インジェクション

    Returns:
        {"ok": True, "skill_file": "..."}
    """
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT keywords, sample_messages, typical_role, occurrence_count "
                    "FROM skill_candidates WHERE id = %s",
                    (proposal_id,),
                )
                row = await cur.fetchone()
                if not row:
                    return {"ok": False, "error": "候補が見つかりません"}

                keywords    = json.loads(row[0]) if row[0] else []
                samples     = json.loads(row[1]) if row[1] else []
                typical_role = row[2]
                count        = row[3]

                # skills/ ディレクトリに YAML ファイルを書き出す
                skill_file = _write_skill_yaml(
                    proposal_id=proposal_id,
                    skill_name=skill_name,
                    keywords=keywords,
                    route_hint=route_hint or typical_role,
                    system_prompt_hint=system_prompt_hint,
                    sample_messages=samples,
                    occurrence_count=count,
                )

                # DB を approved に更新
                await cur.execute(
                    """UPDATE skill_candidates
                       SET status='approved', approved_at=NOW(),
                           skill_name=%s, skill_file=%s
                       WHERE id = %s""",
                    (skill_name, skill_file, proposal_id),
                )
            await conn.commit()

        # インメモリキャッシュを再読み込み
        _load_skills()

        logger.info("✅ スキル承認: '%s' → %s", skill_name, skill_file)
        return {"ok": True, "skill_file": skill_file, "skill_name": skill_name}

    except Exception as e:
        logger.error("approve_proposal 失敗: %s", e)
        return {"ok": False, "error": str(e)}


async def dismiss_proposal(proposal_id: str) -> dict:
    """スキル候補を却下する"""
    try:
        import psycopg
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE skill_candidates SET status='dismissed' WHERE id = %s",
                    (proposal_id,),
                )
            await conn.commit()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── スキルファイル書き出し ─────────────────────────────────────────

def _write_skill_yaml(
    proposal_id: str,
    skill_name: str,
    keywords: list[str],
    route_hint: str,
    system_prompt_hint: str,
    sample_messages: list[str],
    occurrence_count: int,
) -> str:
    """承認されたスキルを YAML ファイルに書き出す。ファイルパスを返す。"""
    _SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\-]", "_", skill_name)[:40]
    filename = f"skill-{proposal_id[:8]}-{safe_name}.yaml"
    path = _SKILLS_DIR / filename

    now = datetime.datetime.now().isoformat()
    lines = [
        f"# 武士団 Bushidan v18 — 自動生成スキル",
        f"# 承認日時: {now}",
        f"# 出現回数: {occurrence_count}",
        f"---",
        f"id: \"{proposal_id}\"",
        f"name: \"{skill_name}\"",
        f"created_at: \"{now}\"",
        f"occurrence_count: {occurrence_count}",
        f"",
        f"# このスキルが発動するキーワード (いずれか1つ以上含む場合)",
        f"trigger_keywords:",
    ]
    for kw in keywords:
        lines.append(f'  - "{kw}"')

    lines += [
        f"",
        f"# 推奨ルーティング先ロール",
        f"route_hint: \"{route_hint}\"",
        f"",
        f"# システムプロンプトへの追加インジェクション (空白可)",
        f"system_prompt_hint: |",
    ]
    if system_prompt_hint.strip():
        for line in system_prompt_hint.strip().split("\n"):
            lines.append(f"  {line}")
    else:
        lines.append("  # (なし)")

    lines += [
        f"",
        f"# 学習に使ったサンプルメッセージ",
        f"sample_messages:",
    ]
    for msg in sample_messages[:3]:
        safe_msg = msg.replace('"', "'")[:120]
        lines.append(f'  - "{safe_msg}"')

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("📝 スキルファイル書き出し: %s", path)
    return str(path)


# ── 承認済みスキルの読み込みとルーティングヒント ───────────────────────

_LOADED_SKILLS: list[dict] = []


def _load_skills() -> None:
    """skills/ ディレクトリからすべての承認済みスキルを読み込む"""
    global _LOADED_SKILLS
    skills = []
    if not _SKILLS_DIR.exists():
        _LOADED_SKILLS = skills
        return
    for f in _SKILLS_DIR.glob("skill-*.yaml"):
        try:
            text = f.read_text(encoding="utf-8")
            skill: dict = {}
            in_keywords = False
            for line in text.split("\n"):
                if line.startswith("id:"):
                    skill["id"] = line.split(":", 1)[1].strip().strip('"')
                    in_keywords = False
                elif line.startswith("name:"):
                    skill["name"] = line.split(":", 1)[1].strip().strip('"')
                    in_keywords = False
                elif line.startswith("route_hint:"):
                    skill["route_hint"] = line.split(":", 1)[1].strip().strip('"')
                    in_keywords = False
                elif line.startswith("trigger_keywords:"):
                    in_keywords = True
                elif in_keywords and line.startswith("  - "):
                    skill.setdefault("trigger_keywords", []).append(
                        line.strip().lstrip("- ").strip('"')
                    )
                elif line.strip() and not line.startswith(" "):
                    in_keywords = False
            if skill.get("id") and skill.get("trigger_keywords"):
                skills.append(skill)
        except Exception as e:
            logger.debug("スキルファイル読み込み失敗 %s: %s", f.name, e)
    _LOADED_SKILLS = skills
    logger.info("🎯 承認済みスキル読み込み: %d件", len(skills))


def get_route_hint(message: str) -> Optional[str]:
    """
    メッセージに合致するスキルがあれば推奨ルーティング先を返す。
    LangGraph の _route_decision() で参照する。

    Returns:
        role_key (例: "gunshi") or None
    """
    if not _LOADED_SKILLS:
        return None
    msg_lower = message.lower()
    for skill in _LOADED_SKILLS:
        if any(kw.lower() in msg_lower for kw in skill.get("trigger_keywords", [])):
            route = skill.get("route_hint")
            if route:
                logger.info(
                    "🎯 スキルヒット: '%s' → %s", skill.get("name", "?"), route
                )
                return route
    return None


# 起動時にスキルをロード
_load_skills()
