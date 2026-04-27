"""
tests/unit/test_auth.py — console/auth.py のユニットテスト
"""

import importlib
import os
import sys
import time



def _load_auth(env: dict):
    """指定した環境変数で auth モジュールを新規ロード"""
    for k, v in env.items():
        os.environ[k] = v
    for k in list(sys.modules):
        if "console.auth" in k:
            del sys.modules[k]
    return importlib.import_module("console.auth")


class TestCheckPassword:
    def test_bcrypt_correct(self):
        import bcrypt
        pw = "test_password_123"
        h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
        auth = _load_auth({"CONSOLE_PASSWORD_HASH": h, "CONSOLE_PASSWORD": ""})
        assert auth.check_password(pw) is True

    def test_bcrypt_wrong(self):
        import bcrypt
        h = bcrypt.hashpw(b"correct", bcrypt.gensalt()).decode()
        auth = _load_auth({"CONSOLE_PASSWORD_HASH": h, "CONSOLE_PASSWORD": ""})
        assert auth.check_password("wrong") is False

    def test_plaintext_fallback_correct(self):
        auth = _load_auth({"CONSOLE_PASSWORD_HASH": "", "CONSOLE_PASSWORD": "secret"})
        assert auth.check_password("secret") is True

    def test_plaintext_fallback_wrong(self):
        auth = _load_auth({"CONSOLE_PASSWORD_HASH": "", "CONSOLE_PASSWORD": "secret"})
        assert auth.check_password("bad") is False

    def test_no_password_set_returns_false(self):
        auth = _load_auth({"CONSOLE_PASSWORD_HASH": "", "CONSOLE_PASSWORD": ""})
        assert auth.check_password("anything") is False

    def test_auth_bypass_flag_allows_access(self):
        auth = _load_auth({"CONSOLE_PASSWORD_HASH": "", "CONSOLE_PASSWORD": "",
                           "CONSOLE_AUTH_BYPASS": "true"})
        assert auth.check_password("anything") is True


class TestSession:
    def test_create_and_validate(self):
        import bcrypt
        h = bcrypt.hashpw(b"pw", bcrypt.gensalt()).decode()
        auth = _load_auth({"CONSOLE_PASSWORD_HASH": h, "CONSOLE_PASSWORD": ""})
        token = auth.create_session()
        assert auth.validate_session(token) is True

    def test_invalid_token(self):
        import bcrypt
        h = bcrypt.hashpw(b"pw", bcrypt.gensalt()).decode()
        auth = _load_auth({"CONSOLE_PASSWORD_HASH": h, "CONSOLE_PASSWORD": ""})
        assert auth.validate_session("not_a_real_token") is False

    def test_expired_session(self):
        import bcrypt
        h = bcrypt.hashpw(b"pw", bcrypt.gensalt()).decode()
        auth = _load_auth({"CONSOLE_PASSWORD_HASH": h, "CONSOLE_PASSWORD": ""})
        token = auth.create_session()
        # 有効期限を過去に書き換える
        auth._SESSIONS[token] = time.time() - 1
        assert auth.validate_session(token) is False
        assert token not in auth._SESSIONS  # クリーンアップも確認
