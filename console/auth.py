"""
console/auth.py — セッションベース認証

.env の設定:
  CONSOLE_PASSWORD_HASH=<bcrypt ハッシュ>   ← 推奨
  CONSOLE_PASSWORD=<平文>                   ← 後方互換 (非推奨)

ハッシュ生成コマンド:
  python3 -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_PASSWORD', bcrypt.gensalt()).decode())"
"""

import os
import secrets
import time
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_SESSIONS: dict[str, float] = {}  # token → expiry timestamp
_SESSION_TTL = 86400  # 24 hours


def _get_hash() -> Optional[str]:
    return os.environ.get("CONSOLE_PASSWORD_HASH")


def _get_plaintext() -> Optional[str]:
    """後方互換: CONSOLE_PASSWORD_HASH が未設定の場合のみ使用"""
    return os.environ.get("CONSOLE_PASSWORD")


def check_password(password: str) -> bool:
    """パスワード検証。ハッシュ優先、なければ平文フォールバック"""
    pw_hash = _get_hash()
    if pw_hash:
        try:
            import bcrypt
            return bcrypt.checkpw(password.encode(), pw_hash.encode())
        except Exception as e:
            logger.error("bcrypt 検証エラー: %s", e)
            return False

    # 後方互換: 平文比較
    plain = _get_plaintext()
    if plain:
        return secrets.compare_digest(password, plain)

    # どちらも未設定 → 認証スキップ（開発環境用: 本番では必ず CONSOLE_PASSWORD_HASH を設定すること）
    logger.error("🚨 CONSOLE_PASSWORD_HASH 未設定: 認証をスキップしています！本番環境では危険です")
    return True


def create_session() -> str:
    """新しいセッショントークンを生成"""
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = time.time() + _SESSION_TTL
    return token


def validate_session(token: str) -> bool:
    """セッショントークンを検証"""
    # 認証が設定されていない場合はスキップ
    if not _get_hash() and not _get_plaintext():
        return True
    expiry = _SESSIONS.get(token)
    if not expiry:
        return False
    if time.time() > expiry:
        _SESSIONS.pop(token, None)
        return False
    return True
