"""
core/langgraph_router.py — 後方互換シム

実装は core/router/ パッケージへ移動済み。
既存の `from core.langgraph_router import LangGraphRouter` は引き続き動作する。
"""
from core.router.router import LangGraphRouter  # noqa: F401

__all__ = ["LangGraphRouter"]
