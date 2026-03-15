"""
core/mcp_permissions.py — MCP権限マトリクス管理 v14

system_orchestrator.py から抽出。
役職ごとのMCPアクセス権限を管理する。
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


class MCPPermissionLevel(Enum):
    """MCPアクセスレベル"""
    EXCLUSIVE = "exclusive"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    READONLY = "readonly"
    DELEGATED = "delegated"
    FORBIDDEN = "forbidden"


class MCPPermissionManager:
    """役職ごとのMCPアクセス権限を管理"""

    ROLE_PRIORITY = [
        "uketuke", "gaiji", "kengyo", "shogun", "gunshi",
        "sanbo", "yuhitsu", "seppou", "onmitsu", "daigensui",
    ]

    def __init__(self, config_path: Optional[str] = None):
        self.permissions: Dict[str, Dict[str, Dict]] = {}
        self.mcp_registry: Dict[str, Dict] = {}
        self.config_path = config_path or "config/mcp_permissions.yaml"
        self._loaded = False

    def load_permissions(self) -> bool:
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self._load_default_permissions()
                return False
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.mcp_registry = config.get("mcp_registry", {})
            for role, role_config in config.get("role_permissions", {}).items():
                self.permissions[role] = role_config.get("permissions", {})
            self._loaded = True
            return True
        except Exception as e:
            logger.error("MCP権限設定読み込み失敗: %s", e)
            self._load_default_permissions()
            return False

    def _load_default_permissions(self) -> None:
        self.permissions = {
            "daigensui": {
                "graph_memory": {"level": "exclusive", "priority": 1},
                "notion": {"level": "exclusive", "priority": 1},
                "sequential_thinking": {"level": "primary"},
                "filesystem": {"level": "readonly"},
                "git": {"level": "readonly"},
            },
            "shogun": {
                "graph_memory": {"level": "primary"},
                "notion": {"level": "primary"},
                "filesystem": {"level": "primary"},
                "git": {"level": "primary"},
                "prisma": {"level": "secondary"},
            },
            "gunshi": {
                "sequential_thinking": {"level": "exclusive", "priority": 1},
                "filesystem": {"level": "primary"},
                "git": {"level": "primary"},
                "graph_memory": {"level": "primary"},
            },
            "sanbo": {
                "filesystem": {"level": "primary"},
                "git": {"level": "primary"},
                "prisma": {"level": "primary"},
            },
            "kengyo": {
                "playwright": {"level": "exclusive", "priority": 1},
                "filesystem": {"level": "primary"},
            },
            "onmitsu": {
                "filesystem": {"level": "delegated"},
                "git": {"level": "delegated"},
            },
        }
        self._loaded = True
        logger.info("デフォルトMCP権限設定を使用")

    def check_permission(self, role: str, mcp_name: str) -> MCPPermissionLevel:
        if not self._loaded:
            self.load_permissions()
        role_perms = self.permissions.get(role, {})
        level_str = role_perms.get(mcp_name, {}).get("level", "forbidden")
        try:
            return MCPPermissionLevel(level_str)
        except ValueError:
            return MCPPermissionLevel.FORBIDDEN

    def can_access(self, role: str, mcp_name: str) -> bool:
        return self.check_permission(role, mcp_name) != MCPPermissionLevel.FORBIDDEN

    def can_write(self, role: str, mcp_name: str) -> bool:
        level = self.check_permission(role, mcp_name)
        return level in (
            MCPPermissionLevel.EXCLUSIVE, MCPPermissionLevel.PRIMARY,
            MCPPermissionLevel.SECONDARY, MCPPermissionLevel.DELEGATED,
        )

    def get_exclusive_owner(self, mcp_name: str) -> Optional[str]:
        for role in self.ROLE_PRIORITY:
            if self.check_permission(role, mcp_name) == MCPPermissionLevel.EXCLUSIVE:
                return role
        return None

    def resolve_conflict(self, mcp_name: str, requesting_roles: list) -> str:
        for role in requesting_roles:
            if self.check_permission(role, mcp_name) == MCPPermissionLevel.EXCLUSIVE:
                return role
        role_priorities = []
        for role in requesting_roles:
            prio = self.permissions.get(role, {}).get(mcp_name, {}).get("priority", 99)
            role_priorities.append((role, prio))
        role_priorities.sort(key=lambda x: x[1])
        best = role_priorities[0][1]
        candidates = [r for r, p in role_priorities if p == best]
        if len(candidates) == 1:
            return candidates[0]
        for role in self.ROLE_PRIORITY:
            if role in candidates:
                return role
        return requesting_roles[0]

    def get_role_permissions(self, role: str) -> Dict[str, str]:
        if not self._loaded:
            self.load_permissions()
        return {
            mcp: perm.get("level", "forbidden")
            for mcp, perm in self.permissions.get(role, {}).items()
        }
