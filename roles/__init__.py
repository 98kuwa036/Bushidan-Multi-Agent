"""roles/ — 各LLMロールの定義パッケージ

各ロールは BaseRole を継承し、execute(state) → RoleResult を実装する。
LangGraph の各実行ノードがこれを呼び出す。
"""
from roles.base import BaseRole, RoleResult

__all__ = ["BaseRole", "RoleResult"]
