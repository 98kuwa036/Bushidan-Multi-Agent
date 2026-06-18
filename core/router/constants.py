"""
core/router/constants.py — LangGraph Router 定数・モジュールレベル関数
"""
import asyncio
import os
from utils.logger import get_logger

logger = get_logger(__name__)

POSTGRES_URL: str = os.environ.get("POSTGRES_URL", "")

NODE_TIMEOUTS: dict[str, int] = {
    "uketuke_qa":        30,
    "uketuke_default":   60,
    "gaiji_rag":         60,
    "sanbo_mcp":         60,
    "kengyo_vision":     60,
    "onmitsu_local":    120,
    "execute_step":     120,
    "shogun_plan":      120,
    "daigensui_audit":  200,
    "multi_role":       120,   # 複合タスク: 複数ロール並列実行
}

FALLBACK_MAP: dict[str, str] = {
    "uketuke_qa":       "uketuke_default",
    "parallel_uketuke": "uketuke_qa",
    "multi_role":       "sanbo_mcp",       # 複合タスク失敗 → 参謀に委ねる
    "gaiji_rag":        "uketuke_qa",
    "sanbo_mcp":        "uketuke_qa",
    "onmitsu_local":    "uketuke_default",
    "kengyo_vision":    "sanbo_mcp",
    "shogun_plan":      "sanbo_mcp",
    "execute_step":     "sanbo_mcp",
    "daigensui_audit":  "shogun_plan",
    "uketuke_default":  "uketuke_qa",
}

# fire-and-forget タスク管理
_bg_tasks: set = set()


def fire(coro, *, name: str = None) -> "asyncio.Task":
    """バックグラウンドタスクを起動し GC による中断を防ぐ"""
    t = asyncio.create_task(coro, name=name)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return t


async def refresh_notion_index_bg() -> None:
    """Notion インデックスをバックグラウンド更新"""
    try:
        from integrations.notion.index import refresh_index
        n = await refresh_index()
        logger.info("📋 Notionインデックス初期構築: %d件", n)
    except Exception as e:
        logger.warning("Notionインデックス初期構築スキップ: %s", e)


async def skill_observe(
    thread_id: str,
    message: str,
    handled_by: str,
    execution_time: float,
    success: bool = True,
    error: str = "",
    had_hitl: bool = False,
    used_fallback: bool = False,
) -> None:
    """スキルトラッカーにチャット結果を非同期記録"""
    try:
        from utils.skill_tracker import observe as _st_observe
        await _st_observe(
            thread_id, message, handled_by, execution_time,
            success=success, error=error,
            had_hitl=had_hitl, used_fallback=used_fallback,
        )
    except Exception as e:
        logger.error("skill_observe 失敗: %s", e)


def load_roles() -> dict:
    """roles/ パッケージから全ロールをロード（起動時1回のみ）"""
    from roles.uketuke import UketukeRole
    from roles.gaiji import GaijiRole
    from roles.sanbo import SanboRole
    from roles.shogun import ShogunRole
    from roles.daigensui import DaigensuiRole
    from roles.onmitsu import OnmitsuRole
    from roles.kengyo import KengyoRole
    return {
        "uketuke":   UketukeRole(),
        "gaiji":     GaijiRole(),
        "sanbo":     SanboRole(),
        "shogun":    ShogunRole(),
        "daigensui": DaigensuiRole(),
        "onmitsu":   OnmitsuRole(),
        "kengyo":    KengyoRole(),
    }
