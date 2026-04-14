"""
武士団 v18 — Phase 4 スキル自動進化エンジン

フロー:
  1. v18 監査ログ (core/audit) からパターンを分析
  2. 高頻度パターンを skill_candidates テーブルに登録
  3. 管理画面で承認後 → notion_skills テーブルに投入 + embedding 生成
  4. pgvector 検索が次回から強化される

Notion API 統合:
  - NOTION_API_KEY が設定されていれば Notion DB に書き込み
  - 未設定の場合は PostgreSQL のみ更新
"""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

POSTGRES_URL = os.environ.get("POSTGRES_URL", "")
if not POSTGRES_URL:
    logger.error("POSTGRES_URL が未設定です。.env を確認してください。")

_AUDIT_ROOT = Path(
    os.environ.get("AUDIT_DIR", "/mnt/Bushidan-Multi-Agent/audit")
) / "v18"

# スキル候補のしきい値
_CANDIDATE_THRESHOLD = int(os.environ.get("SKILL_EVOLVE_THRESHOLD", "5"))

# DB スキーマ初期化 SQL
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS skill_evolution_log (
    id SERIAL PRIMARY KEY,
    run_at TIMESTAMP DEFAULT NOW(),
    patterns_found INTEGER DEFAULT 0,
    candidates_created INTEGER DEFAULT 0,
    skills_activated INTEGER DEFAULT 0,
    run_duration_ms FLOAT DEFAULT 0,
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS notion_skills (
    skill_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    category TEXT DEFAULT '',
    recommended_role TEXT DEFAULT 'auto',
    role_confidence FLOAT DEFAULT 0.7,
    trigger_keywords TEXT[] DEFAULT '{}',
    embedding vector(1536),
    source TEXT DEFAULT 'manual',   -- 'manual' | 'evolved' | 'notion'
    notion_page_id TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_notion_skills_embedding
    ON notion_skills USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_notion_skills_active
    ON notion_skills (is_active) WHERE is_active;

CREATE UNIQUE INDEX IF NOT EXISTS idx_notion_skills_name
    ON notion_skills (name);
"""


@dataclass
class SkillPattern:
    """監査ログから抽出されたパターン"""
    pattern_hash: str
    keywords: List[str]
    intent_types: List[str]         # 出現したintent_type一覧
    agent_roles: List[str]          # 実際に使われたロール一覧
    complexity_levels: List[str]
    occurrence_count: int = 0
    avg_notion_score: float = 0.0


@dataclass
class EvolvedSkill:
    """進化によって生成されたスキル候補"""
    name: str
    description: str
    category: str
    recommended_role: str
    role_confidence: float
    trigger_keywords: List[str]
    source_pattern: str             # pattern_hash
    occurrence_count: int


class SkillEvolutionEngine:
    """スキル自動進化エンジン"""

    def __init__(self) -> None:
        self._openai_key = os.getenv("OPENAI_API_KEY", "")
        self._notion_key = os.getenv("NOTION_API_KEY", "")
        self._notion_db_id = os.getenv("NOTION_SKILLS_DB_ID", "")
        self._openai = None

    def _get_openai(self):
        if self._openai is None and self._openai_key:
            try:
                from openai import AsyncOpenAI
                self._openai = AsyncOpenAI(api_key=self._openai_key)
            except ImportError:
                pass
        return self._openai

    async def _ensure_schema(self) -> None:
        """必要テーブルの作成"""
        try:
            import psycopg
            # pgvector extension ensure
            async with await psycopg.AsyncConnection.connect(
                POSTGRES_URL, autocommit=True
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    for stmt in _SCHEMA_SQL.strip().split(";"):
                        stmt = stmt.strip()
                        if stmt:
                            try:
                                await cur.execute(stmt)
                            except Exception as e:
                                logger.debug("schema stmt skipped (%s): %.60s", type(e).__name__, stmt)
        except Exception as e:
            logger.warning("_ensure_schema error: %s", e)

    # ── 監査ログからパターン抽出 ─────────────────────────────────────────

    def _parse_audit_yaml_line(self, line: str) -> Optional[dict]:
        """YAML行からキー・値を簡易抽出"""
        m = re.match(r'\s+(\w+):\s*(.*)', line)
        if m:
            return {"key": m.group(1), "value": m.group(2).strip().strip('"')}
        return None

    async def analyze_audit_logs(self, days: int = 7) -> List[SkillPattern]:
        """監査ログを分析してパターンを抽出"""
        patterns: Dict[str, SkillPattern] = {}

        if not _AUDIT_ROOT.exists():
            return []

        cutoff = datetime.date.today() - datetime.timedelta(days=days)

        for day_dir in sorted(_AUDIT_ROOT.iterdir()):
            if not day_dir.is_dir():
                continue
            try:
                day = datetime.date.fromisoformat(day_dir.name)
            except ValueError:
                continue
            if day < cutoff:
                continue

            for yaml_file in day_dir.glob("bushidan-hour-*.yaml"):
                try:
                    content = yaml_file.read_text(encoding="utf-8")
                    await self._extract_patterns_from_yaml(content, patterns)
                except Exception as e:
                    logger.debug("yaml parse error %s: %s", yaml_file.name, e)

        return list(patterns.values())

    async def _extract_patterns_from_yaml(
        self, content: str, patterns: Dict[str, SkillPattern]
    ) -> None:
        """YAML コンテンツからパターンを抽出"""
        # シンプルな行パース
        current_entry: Dict[str, str] = {}
        for line in content.split("\n"):
            parsed = self._parse_audit_yaml_line(line)
            if parsed:
                current_entry[parsed["key"]] = parsed["value"]

            # エントリ境界検出（空行またはネストの終わり）
            if line.strip() == "" and current_entry:
                await self._process_entry(current_entry, patterns)
                current_entry = {}

        if current_entry:
            await self._process_entry(current_entry, patterns)

    async def _process_entry(
        self, entry: Dict[str, str], patterns: Dict[str, SkillPattern]
    ) -> None:
        """1エントリをパターンに集約"""
        user_input = entry.get("user_input_summary", "")
        intent_type = entry.get("intent_type", "qa")
        agent_role = entry.get("agent_role", "auto")
        complexity = entry.get("complexity", "medium")
        try:
            notion_score = float(entry.get("notion_score", "0"))
        except ValueError:
            notion_score = 0.0

        if len(user_input) < 10:
            return

        # キーワード抽出（簡易）
        keywords = _simple_keywords(user_input)
        if not keywords:
            return

        ph = _pattern_hash(keywords, intent_type)

        if ph not in patterns:
            patterns[ph] = SkillPattern(
                pattern_hash=ph,
                keywords=keywords,
                intent_types=[intent_type],
                agent_roles=[agent_role],
                complexity_levels=[complexity],
                occurrence_count=1,
                avg_notion_score=notion_score,
            )
        else:
            p = patterns[ph]
            p.occurrence_count += 1
            if intent_type not in p.intent_types:
                p.intent_types.append(intent_type)
            if agent_role not in p.agent_roles:
                p.agent_roles.append(agent_role)
            if complexity not in p.complexity_levels:
                p.complexity_levels.append(complexity)
            # 移動平均
            p.avg_notion_score = (
                p.avg_notion_score * (p.occurrence_count - 1) + notion_score
            ) / p.occurrence_count

    # ── スキル候補生成 ───────────────────────────────────────────────────

    async def generate_candidates(
        self, patterns: List[SkillPattern]
    ) -> List[EvolvedSkill]:
        """高頻度パターンからスキル候補を生成"""
        candidates: List[EvolvedSkill] = []

        high_freq = [p for p in patterns if p.occurrence_count >= _CANDIDATE_THRESHOLD]
        logger.info("SkillEvolution: %d patterns, %d above threshold", len(patterns), len(high_freq))

        for pattern in high_freq:
            # 最頻出ロールを推奨ロールに
            role_freq: Dict[str, int] = {}
            for r in pattern.agent_roles:
                role_freq[r] = role_freq.get(r, 0) + 1
            recommended_role = max(role_freq, key=role_freq.get)
            role_confidence = role_freq[recommended_role] / len(pattern.agent_roles)

            # カテゴリ推定
            category = _infer_category(pattern.intent_types, pattern.keywords)

            # スキル名 (キーワード上位3語)
            name = " / ".join(pattern.keywords[:3])

            candidates.append(EvolvedSkill(
                name=name,
                description=f"自動検出パターン: {', '.join(pattern.keywords[:5])}",
                category=category,
                recommended_role=recommended_role,
                role_confidence=round(role_confidence, 3),
                trigger_keywords=pattern.keywords[:8],
                source_pattern=pattern.pattern_hash,
                occurrence_count=pattern.occurrence_count,
            ))

        return candidates

    # ── PostgreSQL への保存 ──────────────────────────────────────────────

    async def save_candidates_to_db(self, candidates: List[EvolvedSkill]) -> int:
        """skill_candidates テーブルに候補を保存（重複スキップ）"""
        if not candidates:
            return 0

        try:
            import psycopg
            count = 0
            async with await psycopg.AsyncConnection.connect(
                POSTGRES_URL, autocommit=True
            ) as conn:
                async with conn.cursor() as cur:
                    # skill_candidates テーブルが存在しない場合は作成
                    await cur.execute("""
                        CREATE TABLE IF NOT EXISTS skill_candidates (
                            id TEXT PRIMARY KEY,
                            pattern TEXT,
                            keywords TEXT[],
                            route_hint TEXT DEFAULT 'auto',
                            system_prompt_hint TEXT DEFAULT '',
                            occurrence_count INTEGER DEFAULT 1,
                            status TEXT DEFAULT 'pending',
                            proposed_at TIMESTAMP DEFAULT NOW(),
                            resolved_at TIMESTAMP,
                            skill_name TEXT DEFAULT '',
                            category TEXT DEFAULT '',
                            role_confidence FLOAT DEFAULT 0.7
                        )
                    """)

                    for c in candidates:
                        cid = f"evolved-{c.source_pattern}"
                        # 既存チェック
                        await cur.execute(
                            "SELECT id FROM skill_candidates WHERE id = %s",
                            (cid,),
                        )
                        if await cur.fetchone():
                            # 出現回数のみ更新
                            await cur.execute(
                                "UPDATE skill_candidates SET occurrence_count = %s WHERE id = %s",
                                (c.occurrence_count, cid),
                            )
                            continue

                        await cur.execute(
                            """
                            INSERT INTO skill_candidates
                                (id, pattern, keywords, route_hint, occurrence_count,
                                 status, skill_name, category, role_confidence)
                            VALUES (%s, %s, %s, %s, %s, 'pending', %s, %s, %s)
                            """,
                            (
                                cid,
                                c.source_pattern,
                                c.trigger_keywords,
                                c.recommended_role,
                                c.occurrence_count,
                                c.name,
                                c.category,
                                c.role_confidence,
                            ),
                        )
                        count += 1

            return count
        except Exception as e:
            logger.error("save_candidates_to_db error: %s", e)
            return 0

    # ── 承認済みスキルを notion_skills テーブルに投入 ───────────────────

    async def activate_approved_skills(self) -> int:
        """approved 状態のスキル候補を notion_skills に投入して embedding を生成"""
        try:
            import psycopg
            from pgvector.psycopg import register_vector

            activated = 0
            async with await psycopg.AsyncConnection.connect(
                POSTGRES_URL, autocommit=True
            ) as conn:
                await register_vector(conn)
                async with conn.cursor() as cur:
                    # 承認済みで未投入のもの (notion_skills.name で重複チェック)
                    await cur.execute("""
                        SELECT id, skill_name, pattern, keywords, route_hint,
                               system_prompt_hint, category, role_confidence
                        FROM skill_candidates
                        WHERE status = 'approved'
                          AND skill_name NOT IN (
                              SELECT name FROM notion_skills
                              WHERE source = 'evolved'
                          )
                    """)
                    rows = await cur.fetchall()

                    for row in rows:
                        (cid, name, pattern, kws, role, sp_hint, category, conf) = row
                        description = sp_hint or f"進化スキル: {name}"

                        # Embedding 生成
                        embed_text = f"{name} {description} {' '.join(kws or [])}"
                        embedding = await self._get_embedding(embed_text)

                        if embedding:
                            vec_literal = "[" + ",".join(str(v) for v in embedding) + "]"
                            await cur.execute(
                                """
                                INSERT INTO notion_skills
                                    (name, description, category, recommended_role,
                                     role_confidence, trigger_keywords, embedding, source)
                                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, 'evolved')
                                ON CONFLICT (name) DO NOTHING
                                """,
                                (name, description, category or "auto",
                                 role or "auto", conf or 0.7, kws or [], vec_literal),
                            )
                        else:
                            await cur.execute(
                                """
                                INSERT INTO notion_skills
                                    (name, description, category, recommended_role,
                                     role_confidence, trigger_keywords, source)
                                VALUES (%s, %s, %s, %s, %s, %s, 'evolved')
                                ON CONFLICT (name) DO NOTHING
                                """,
                                (name, description, category or "auto",
                                 role or "auto", conf or 0.7, kws or []),
                            )
                        activated += 1

            return activated
        except Exception as e:
            logger.error("activate_approved_skills error: %s", e)
            return 0

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """OpenAI embedding 取得"""
        client = self._get_openai()
        if client is None:
            return None
        try:
            resp = await asyncio.wait_for(
                client.embeddings.create(model="text-embedding-3-small", input=text[:8000]),
                timeout=8.0,
            )
            if not resp.data:
                logger.warning("embedding: OpenAI returned empty data")
                return None
            return resp.data[0].embedding
        except Exception as e:
            logger.warning("embedding error: %s", e)
            return None

    # ── Notion DB 同期 ───────────────────────────────────────────────────

    async def sync_to_notion(self, skill_id: int) -> bool:
        """notion_skills レコードを Notion API で DB に同期"""
        if not self._notion_key or not self._notion_db_id:
            logger.debug("Notion API 未設定、スキップ")
            return False

        try:
            import psycopg
            import urllib.request

            # スキル取得
            async with await psycopg.AsyncConnection.connect(
                POSTGRES_URL, autocommit=True
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT name, description, category, recommended_role FROM notion_skills WHERE skill_id = %s",
                        (skill_id,),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return False
                    name, desc, category, role = row

            # Notion API 呼び出し
            payload = json.dumps({
                "parent": {"database_id": self._notion_db_id},
                "properties": {
                    "Name": {"title": [{"text": {"content": name}}]},
                    "Description": {"rich_text": [{"text": {"content": desc or ""}}]},
                    "Category": {"select": {"name": category or "General"}},
                    "Role": {"select": {"name": role or "auto"}},
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.notion.com/v1/pages",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._notion_key}",
                    "Content-Type": "application/json",
                    "Notion-Version": "2022-06-28",
                },
                method="POST",
            )

            def _do_request():
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            result = await asyncio.get_event_loop().run_in_executor(None, _do_request)
            notion_page_id = result.get("id", "")

            if notion_page_id:
                async with await psycopg.AsyncConnection.connect(
                    POSTGRES_URL, autocommit=True
                ) as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "UPDATE notion_skills SET notion_page_id = %s WHERE skill_id = %s",
                            (notion_page_id, skill_id),
                        )
                logger.info("Notion 同期完了: %s → %s", name, notion_page_id)
                return True

        except Exception as e:
            logger.warning("sync_to_notion error: %s", e)
        return False

    # ── フルサイクル実行 ─────────────────────────────────────────────────

    async def run_evolution_cycle(self, days: int = 7) -> dict:
        """スキル進化サイクルをフル実行"""
        t0 = time.time()
        await self._ensure_schema()

        # 1. 監査ログ分析
        patterns = await self.analyze_audit_logs(days=days)

        # 2. 候補生成
        candidates = await self.generate_candidates(patterns)

        # 3. DB 保存
        new_candidates = await self.save_candidates_to_db(candidates)

        # 4. 承認済みスキルを有効化
        activated = await self.activate_approved_skills()

        elapsed_ms = (time.time() - t0) * 1000

        # 5. 実行ログ
        try:
            import psycopg
            async with await psycopg.AsyncConnection.connect(
                POSTGRES_URL, autocommit=True
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO skill_evolution_log
                            (patterns_found, candidates_created, skills_activated, run_duration_ms)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (len(patterns), new_candidates, activated, round(elapsed_ms, 1)),
                    )
        except Exception as e:
            logger.warning("evolution_log insert failed: %s", e)

        result = {
            "patterns_analyzed": len(patterns),
            "high_freq_patterns": len(candidates),
            "new_candidates": new_candidates,
            "skills_activated": activated,
            "elapsed_ms": round(elapsed_ms, 1),
        }
        logger.info(
            "SkillEvolution: patterns=%d candidates=%d activated=%d (%.0fms)",
            len(patterns), new_candidates, activated, elapsed_ms,
        )
        return result


# ── ユーティリティ ────────────────────────────────────────────────────────

_STOP_WORDS_SIMPLE = frozenset([
    "て", "に", "を", "は", "が", "で", "の", "と", "も", "から",
    "ます", "です", "する", "なる", "ある", "いる",
    "the", "a", "an", "is", "are", "to", "of", "in", "for",
    "please", "help", "me", "i", "you", "can", "this", "that",
])


def _simple_keywords(text: str, max_kw: int = 6) -> List[str]:
    """シンプルなキーワード抽出"""
    from collections import Counter
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in cleaned.split() if len(t) >= 3 and t not in _STOP_WORDS_SIMPLE]
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(max_kw)]


def _pattern_hash(keywords: List[str], intent_type: str) -> str:
    normalized = sorted(set(keywords[:4])) + [intent_type]
    key = "|".join(normalized)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _infer_category(intent_types: List[str], keywords: List[str]) -> str:
    """インテントとキーワードからカテゴリを推定"""
    if "code" in intent_types:
        return "コーディング"
    if "analysis" in intent_types:
        return "分析"
    if "rag" in intent_types:
        return "情報検索"
    if "task" in intent_types:
        return "タスク管理"
    # キーワードベース
    kw_str = " ".join(keywords).lower()
    if any(w in kw_str for w in ["python", "javascript", "react", "api", "docker"]):
        return "コーディング"
    if any(w in kw_str for w in ["分析", "データ", "統計", "グラフ"]):
        return "分析"
    return "一般"


# ── エントリポイント関数 ──────────────────────────────────────────────────

async def evolve_skills_from_audit(days: int = 7) -> dict:
    """外部から呼び出せるスキル進化エントリポイント"""
    engine = SkillEvolutionEngine()
    return await engine.run_evolution_cycle(days=days)
