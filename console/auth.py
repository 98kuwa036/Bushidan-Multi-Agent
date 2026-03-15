"""
console/auth.py — セッションベース認証

CONSOLE_PASSWORD 環境変数でパスワードを設定。
未設定の場合は認証をスキップ。
"""

import hashlib
import os
import secrets
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_SESSIONS: dict[str, float] = {}  # token → expiry timestamp
_SESSION_TTL = 86400  # 24 hours


def get_password() -> Optional[str]:
    return os.environ.get("CONSOLE_PASSWORD")


def create_session() -> str:
    """新しいセッショントークンを生成"""
    import time
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = time.time() + _SESSION_TTL
    return token


def validate_session(token: str) -> bool:
    """セッショントークンを検証"""
    import time
    if not get_password():
        return True  # パスワード未設定 → 認証スキップ
    expiry = _SESSIONS.get(token)
    if not expiry:
        return False
    if time.time() > expiry:
        _SESSIONS.pop(token, None)
        return False
    return True


def check_password(password: str) -> bool:
    """パスワードを検証"""
    expected = get_password()
    if not expected:
        return True
    return secrets.compare_digest(password, expected)
