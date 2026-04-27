"""
core/router/mixins/checkpointer.py — PostgreSQL チェックポインター管理 Mixin
"""
import asyncio
from typing import TYPE_CHECKING
from utils.logger import get_logger
from core.router.constants import POSTGRES_URL

logger = get_logger(__name__)

if TYPE_CHECKING:
    from langgraph.checkpoint.memory import MemorySaver


class CheckpointerMixin:
    """PostgresSaver 初期化・再接続・ヘルスチェックを担当"""

    if TYPE_CHECKING:
        _pg_status: str
        _pg_error: str
        _pool: object
        _checkpointer: object
        _memory_fallback: "MemorySaver | None"
        _compiled: object
        _compiled_fast: object

    async def _init_checkpointer(self):
        """PostgresSaver を初期化。失敗時は MemorySaver にフォールバック。"""
        try:
            import psycopg
            from psycopg_pool import AsyncConnectionPool
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with await asyncio.wait_for(
                psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True),
                timeout=10,
            ) as setup_conn:
                setup_saver = AsyncPostgresSaver(setup_conn)
                await asyncio.wait_for(setup_saver.setup(), timeout=15)

            self._pool = AsyncConnectionPool(
                conninfo=POSTGRES_URL,
                min_size=1,
                max_size=5,
                open=False,
                reconnect_timeout=30.0,
                kwargs={"autocommit": True},
            )
            await asyncio.wait_for(self._pool.open(wait=True), timeout=20)

            checkpointer = AsyncPostgresSaver(self._pool)
            self._pg_status = "connected"
            self._pg_error = ""
            logger.info("✅ PostgresSaver 初期化完了 (%s)", POSTGRES_URL.split("@")[-1])
            return checkpointer

        except Exception as e:
            self._pg_status = "disconnected"
            self._pg_error = str(e)
            logger.warning("⚠️  PostgresSaver 失敗 → MemorySaver にフォールバック: %s", e)
            if self._pool:
                try:
                    await self._pool.close()
                except Exception:
                    pass
                self._pool = None
            from langgraph.checkpoint.memory import MemorySaver
            fallback = MemorySaver()
            self._memory_fallback = fallback
            return fallback

    async def _background_pg_reconnect(self) -> None:
        """PostgreSQL 再接続ループ（60秒間隔）。成功時はグラフを再コンパイル。"""
        logger.info("🔄 PostgreSQL 再接続ループ開始 (60秒間隔)")
        while self._pg_status != "connected":
            await asyncio.sleep(60)
            self._pg_status = "reconnecting"
            logger.info("🔄 PostgreSQL 再接続試行中...")
            try:
                import psycopg
                from psycopg_pool import AsyncConnectionPool
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                async with await asyncio.wait_for(
                    psycopg.AsyncConnection.connect(POSTGRES_URL, autocommit=True),
                    timeout=10,
                ) as setup_conn:
                    setup_saver = AsyncPostgresSaver(setup_conn)
                    await asyncio.wait_for(setup_saver.setup(), timeout=15)

                new_pool = AsyncConnectionPool(
                    conninfo=POSTGRES_URL,
                    min_size=1, max_size=5,
                    open=False,
                    reconnect_timeout=30.0,
                    kwargs={"autocommit": True},
                )
                await asyncio.wait_for(new_pool.open(wait=True), timeout=20)
                new_saver = AsyncPostgresSaver(new_pool)

                migrated = 0
                if self._memory_fallback is not None:
                    storage = getattr(self._memory_fallback, "storage", {})
                    for thread_id, checkpoints in storage.items():
                        for checkpoint_id, (checkpoint, metadata) in checkpoints.items():
                            try:
                                cfg = {"configurable": {"thread_id": thread_id,
                                                         "checkpoint_id": checkpoint_id}}
                                await new_saver.aput(cfg, checkpoint, metadata, {})
                                migrated += 1
                            except Exception as me:
                                logger.warning("⚠️ チェックポイント移行失敗 %s/%s: %s",
                                               thread_id, checkpoint_id, me)
                    if migrated:
                        logger.info("✅ MemorySaver → PostgreSQL 移行: %d件", migrated)

                if self._pool:
                    try:
                        await self._pool.close()
                    except Exception:
                        pass
                self._pool = new_pool
                self._checkpointer = new_saver
                self._memory_fallback = None
                self._compiled = self._build_graph().compile(checkpointer=self._checkpointer)

                self._pg_status = "connected"
                self._pg_error = ""
                logger.info("✅ PostgreSQL 再接続成功。移行: %d件", migrated)

            except Exception as e:
                self._pg_status = "disconnected"
                self._pg_error = str(e)
                logger.warning("⚠️ PostgreSQL 再接続失敗: %s", e)

    @property
    def pg_status(self) -> dict:
        return {
            "status": self._pg_status,
            "error":  self._pg_error,
            "saver":  type(self._checkpointer).__name__ if self._checkpointer else "none",
        }

    async def _background_health_check(self) -> None:
        """5分間隔で全ロールのヘルスチェックを実行する。"""
        await asyncio.sleep(10)
        while True:
            try:
                from utils.client_registry import ClientRegistry
                results = await ClientRegistry.get().health_check_all()
                unhealthy = [k for k, v in results.items() if not v]
                if unhealthy:
                    logger.warning("🏥 unhealthy ロール: %s", unhealthy)
                else:
                    logger.debug("🏥 全ロール healthy")
            except Exception as e:
                logger.debug("🏥 ヘルスチェック失敗: %s", e)
            await asyncio.sleep(300)
