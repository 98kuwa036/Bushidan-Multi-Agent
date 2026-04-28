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
import threading
import time
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_SESSIONS: dict[str, float] = {}  # token → expiry timestamp
_SESSION_TTL = 86400  # 24 hours
_LAST_CLEANUP: float = 0.0
_sessions_lock = threading.Lock()


def _cleanup_expired() -> None:
    """期限切れセッションを定期削除してメモリリークを防ぐ（スレッドセーフ）"""
    global _LAST_CLEANUP
    now = time.time()
    if now - _LAST_CLEANUP < 3600:  # 1時間に1回
        return
    with _sessions_lock:
        expired = [t for t, exp in _SESSIONS.items() if now > exp]
        for t in expired:
            _SESSIONS.pop(t, None)
        _LAST_CLEANUP = now
    if expired:
        logger.debug("🧹 期限切れセッション %d件を削除", len(expired))


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
            # bcrypt は最大72バイトまで。エラー回避のため事前に切り捨てる。
            encoded_pw = password.encode()[:72]
            return bcrypt.checkpw(encoded_pw, pw_hash.encode())
        except Exception as e:
            logger.error("bcrypt 検証エラー: %s", e)
            return False

    # 後方互換: 平文比較
    plain = _get_plaintext()
    if plain:
        return secrets.compare_digest(password, plain)

    # どちらも未設定 → CONSOLE_AUTH_BYPASS=true の明示が必要
    if os.environ.get("CONSOLE_AUTH_BYPASS", "").lower() == "true":
        logger.warning("🔓 CONSOLE_AUTH_BYPASS=true: 認証スキップ中 (開発環境専用)")
        return True
    logger.error("🚨 CONSOLE_PASSWORD_HASH 未設定: ログインを拒否します。開発環境では CONSOLE_AUTH_BYPASS=true を設定してください")
    return False


def create_session() -> str:
    """新しいセッショントークンを生成"""
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        _SESSIONS[token] = time.time() + _SESSION_TTL
    return token


def validate_session(token: str) -> bool:
    """セッショントークンを検証"""
    _cleanup_expired()
    # 認証が設定されていない場合は明示バイパスフラグを確認
    if not _get_hash() and not _get_plaintext():
        return os.environ.get("CONSOLE_AUTH_BYPASS", "").lower() == "true"
    with _sessions_lock:
        expiry = _SESSIONS.get(token)
        if not expiry:
            return False
        if time.time() > expiry:
            _SESSIONS.pop(token, None)
            return False
        return True
