"""
core/router/constants.py — LangGraph Router 定数・モジュールレベル関数
"""
import asyncio
import os
from utils.logger import get_logger

logger = get_logger(__name__)

POSTGRES_URL: str = os.environ.get("POSTGRES_URL", "")

NODE_TIMEOUTS: dict[str, int] = {
    "groq_qa":           30,
    "uketuke_default":   60,
    "gaiji_rag":         60,
    "sanbo_mcp":         60,
    "kengyo_vision":     60,
    "gunshi_haiku":      60,
    "metsuke_proc":      45,
    "yuhitsu_jp":        90,
    "onmitsu_local":    120,
    "execute_step":     120,
    "shogun_plan":      120,
    "daigensui_audit":  200,
}

FALLBACK_MAP: dict[str, str] = {
    "groq_qa":          "uketuke_default",
    "parallel_groq":    "groq_qa",
    "gunshi_haiku":     "metsuke_proc",
    "metsuke_proc":     "groq_qa",
    "gaiji_rag":        "groq_qa",
    "sanbo_mcp":        "gunshi_haiku",
    "yuhitsu_jp":       "metsuke_proc",
    "onmitsu_local":    "uketuke_default",
    "kengyo_vision":    "gunshi_haiku",
    "shogun_plan":      "gunshi_haiku",
    "execute_step":     "gunshi_haiku",
    "daigensui_audit":  "shogun_plan",
    "uketuke_default":  "groq_qa",
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
    from roles.gunshi import GunshiRole
    from roles.metsuke import MetsukeRole
    from roles.sanbo import SanboRole
    from roles.shogun import ShogunRole
    from roles.daigensui import DaigensuiRole
    from roles.yuhitsu import YuhitsuRole
    from roles.onmitsu import OnmitsuRole
    from roles.kengyo import KengyoRole
    return {
        "uketuke":   UketukeRole(),
        "gaiji":     GaijiRole(),
        "gunshi":    GunshiRole(),
        "metsuke":   MetsukeRole(),
        "sanbo":     SanboRole(),
        "shogun":    ShogunRole(),
        "daigensui": DaigensuiRole(),
        "yuhitsu":   YuhitsuRole(),
        "onmitsu":   OnmitsuRole(),
        "kengyo":    KengyoRole(),
    }
