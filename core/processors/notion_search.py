"""
武士団 v18 — notion_index (NotionSearch) プロセッサ
OpenAI text-embedding-3-small + pgvector K-NN でスキル検索
Redis 7日間キャッシュ（利用不可時はスキップ）
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import List, Optional

from utils.logger import get_logger
from core.models.notion_index import NotionIndexOutput, SkillItem, SkillSearchQuery

logger = get_logger(__name__)

POSTGRES_URL = os.getenv("POSTGRES_URL", "")
if not POSTGRES_URL:
    logger.error("POSTGRES_URL が未設定です。.env を確認してください。")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_EMBED_CACHE_TTL = 7 * 24 * 3600  # 7 days
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM = 1536


class NotionSearchProcessor:
    """notion_index プロセッサ — pgvector K-NN スキル検索"""

    TIMEOUT_SEC = 6.0

    def __init__(self) -> None:
        self._openai_key = os.getenv("OPENAI_API_KEY", "")
        self._openai = None
        self._redis = None
        self._redis_ok: Optional[bool] = None  # None = 未確認
        self._redis_init_lock = asyncio.Lock()  # 初期化の競合状態防止

    # ── OpenAI クライアント ──────────────────────────────────────────────
    def _get_openai(self):
        if self._openai is None and self._openai_key:
            try:
                from openai import AsyncOpenAI
                self._openai = AsyncOpenAI(api_key=self._openai_key)
            except ImportError:
                logger.warning("openai package not installed")
        return self._openai

    # ── Redis クライアント ───────────────────────────────────────────────
    async def _get_redis_async(self):
        """Redis クライアントを非同期で初期化（競合状態防止）"""
        if self._redis_ok is False:
            return None
        if self._redis is not None:
            return self._redis
        async with self._redis_init_lock:
            # ロック取得後に再確認（double-checked locking）
            if self._redis is not None:
                return self._redis
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(REDIS_URL, decode_responses=False)
                self._redis_ok = True
            except Exception as e:
                logger.warning("Redis init failed: %s", e)
                self._redis_ok = False
        return self._redis

    def _get_redis(self):
        """同期版（既存コードとの互換性用）"""
        if self._redis_ok is False:
            return None
        return self._redis

    async def close(self) -> None:
        """Redis 接続を解放"""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None
            self._redis_ok = None

    # ── キャッシュ用キー ─────────────────────────────────────────────────
    @staticmethod
    def _cache_key(text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()[:24]
        return f"emb:{digest}"

    # ── 埋め込み取得（Redis キャッシュ付き）────────────────────────────
    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        cache_key = self._cache_key(text)

        # キャッシュ確認
        redis = await self._get_redis_async()
        if redis:
            try:
                cached = await redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.debug("Redis get error: %s", e)

        # OpenAI API 呼び出し
        client = self._get_openai()
        if client is None:
            logger.warning("OpenAI client unavailable")
            return None

        try:
            resp = await asyncio.wait_for(
                client.embeddings.create(
                    model=_EMBED_MODEL,
                    input=text[:8000],  # max 8k chars
                ),
                timeout=self.TIMEOUT_SEC,
            )
            if not resp.data:
                logger.warning("OpenAI embedding returned empty data")
                return None
            embedding = resp.data[0].embedding

            # キャッシュ保存（redisは上で取得済み）
            if redis:
                try:
                    await redis.set(cache_key, json.dumps(embedding), ex=_EMBED_CACHE_TTL)
                except Exception as e:
                    logger.debug("Redis set error: %s", e)

            return embedding

        except asyncio.TimeoutError:
            logger.warning("OpenAI embedding timeout")
            return None
        except Exception as e:
            logger.warning("OpenAI embedding error: %s", e)
            return None

    # ── pgvector K-NN 検索 ───────────────────────────────────────────────
    async def _search_skills(
        self, embedding: List[float], top_k: int = 5
    ) -> List[SkillItem]:
        try:
            import psycopg
            from pgvector.psycopg import register_vector

            conn_str = POSTGRES_URL.replace("postgresql://", "postgresql+psycopg://") \
                if "+psycopg" not in POSTGRES_URL else POSTGRES_URL
            # psycopg3 用: プレフィックス不要
            conn_str = POSTGRES_URL

            vec_literal = "[" + ",".join(str(v) for v in embedding) + "]"

            async with await psycopg.AsyncConnection.connect(
                conn_str, autocommit=True
            ) as conn:
                await register_vector(conn)
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT
                            skill_id, name, description, category,
                            recommended_role, role_confidence,
                            1 - (embedding <-> %s::vector) AS relevance_score,
                            trigger_keywords
                        FROM notion_skills
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <-> %s::vector ASC
                        LIMIT %s
                        """,
                        (vec_literal, vec_literal, top_k),
                    )
                    rows = await cur.fetchall()

            items: List[SkillItem] = []
            for row in rows:
                (skill_id, name, desc, category,
                 rec_role, role_conf, relevance, trigger_kws) = row
                items.append(SkillItem(
                    skill_id=skill_id,
                    name=name,
                    description=desc or "",
                    category=category or "",
                    recommended_role=rec_role or "auto",
                    role_confidence=float(role_conf or 0.7),
                    relevance_score=max(0.0, float(relevance or 0.0)),
                    trigger_keywords=trigger_kws or [],
                ))
            return items

        except Exception as e:
            logger.error("pgvector search error: %s", type(e).__name__)
            return []

    # ── フォールバック: キーワードマッチ ───────────────────────────────
    async def _keyword_search(
        self, query_text: str, top_k: int = 5
    ) -> List[SkillItem]:
        """埋め込み失敗時の LIKE ベース検索"""
        try:
            import psycopg

            words = [w for w in query_text.split()[:5] if len(w) >= 2]
            if not words:
                return []

            # パラメータ化クエリで SQL インジェクションを防止
            where_parts = []
            params: List = []
            for w in words:
                where_parts.append("(name ILIKE %s OR description ILIKE %s)")
                params.extend([f"%{w}%", f"%{w}%"])
            where_clause = " OR ".join(where_parts)
            params.append(top_k)

            async with await psycopg.AsyncConnection.connect(
                POSTGRES_URL, autocommit=True
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        SELECT skill_id, name, description, category,
                               recommended_role, role_confidence, trigger_keywords
                        FROM notion_skills
                        WHERE {where_clause}
                        LIMIT %s
                        """,
                        params,
                    )
                    rows = await cur.fetchall()

            items: List[SkillItem] = []
            for i, row in enumerate(rows):
                (skill_id, name, desc, category, rec_role, role_conf, trigger_kws) = row
                items.append(SkillItem(
                    skill_id=skill_id,
                    name=name,
                    description=desc or "",
                    category=category or "",
                    recommended_role=rec_role or "auto",
                    role_confidence=float(role_conf or 0.7),
                    relevance_score=0.5 - i * 0.05,  # 疑似スコア
                    trigger_keywords=trigger_kws or [],
                ))
            return items

        except Exception as e:
            logger.error("keyword search error: %s", type(e).__name__)
            return []

    # ── メイン process ───────────────────────────────────────────────────
    async def process(self, query: SkillSearchQuery) -> NotionIndexOutput:
        start = time.time()

        # 検索テキスト構築
        search_text = query.primary_topic
        if query.sub_topics:
            search_text += " " + " ".join(query.sub_topics[:3])
        search_text = search_text.strip()[:500]

        reasoning_parts: List[str] = [f"検索テキスト: 「{search_text[:80]}」"]
        cache_hit = False

        # 埋め込み取得
        embedding = await self._get_embedding(search_text)

        results: List[SkillItem] = []
        if embedding:
            results = await self._search_skills(embedding, top_k=query.top_k)
            reasoning_parts.append(f"pgvector K-NN 検索: {len(results)} 件")

            # キャッシュヒット確認（Redis から即座に返った場合）
            if self._redis_ok:
                cache_hit = True  # 埋め込みキャッシュが使われた可能性
        else:
            # フォールバック: キーワード検索
            results = await self._keyword_search(search_text, top_k=query.top_k)
            reasoning_parts.append(f"キーワード検索 (embedding失敗): {len(results)} 件")

        # 推奨ロール決定（最上位スキルから）
        suggested_role = "auto"
        top_relevance = 0.0
        if results:
            top = results[0]
            top_relevance = top.relevance_score
            if top_relevance >= 0.7:
                suggested_role = top.recommended_role
                reasoning_parts.append(
                    f"最上位スキル「{top.name}」(relevance={top_relevance:.3f}) → {suggested_role}"
                )
            else:
                reasoning_parts.append(
                    f"最上位スキル relevance={top_relevance:.3f} < 0.7 → auto"
                )
        else:
            reasoning_parts.append("スキルマッチなし → auto")

        elapsed_ms = (time.time() - start) * 1000
        logger.debug(
            "NotionSearch: %.1fms results=%d suggested=%s",
            elapsed_ms, len(results), suggested_role,
        )

        return NotionIndexOutput(
            results=results,
            suggested_role=suggested_role,
            relevance_score=top_relevance,
            search_query_used=search_text,
            cache_hit=cache_hit,
            search_reasoning=" | ".join(reasoning_parts),
            processing_time_ms=elapsed_ms,
        )
