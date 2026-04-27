"""
utils/code_sandbox.py — コードサンドボックス v18

Docker 不使用のサブプロセスベース Python 実行環境。
pct100 は LXC コンテナのため Docker 未利用。
subprocess + リソース制限 + タイムアウトで安全に実行。

制限:
  - 実行時間: 10秒
  - メモリ: 256MB
  - ファイルアクセス: /tmp 配下のみ
  - ネットワーク: 禁止（import socket/requests/urllib 拒否）
  - 禁止モジュール: os.system, subprocess, socket, requests, urllib,
                    open (絶対パス), __import__

使用例:
    result = await run_code("print(1 + 2)")
    # => {"stdout": "3\\n", "stderr": "", "exit_code": 0, "elapsed_ms": 42}
"""
from __future__ import annotations

import asyncio
import logging
import os
import resource
import sys
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── 制限値 ──────────────────────────────────────────────────────────────────
TIMEOUT_SEC   = 10        # 最大実行時間 (秒)
MAX_MEM_BYTES = 256 * 1024 * 1024   # 256 MB
MAX_OUTPUT    = 16 * 1024           # 出力上限 16KB

# ── 禁止パターン（静的チェック）────────────────────────────────────────────
_BLOCKED_PATTERNS = [
    "import subprocess",
    "import socket",
    "import requests",
    "from requests",
    "import urllib",
    "from urllib",
    "import httpx",
    "import aiohttp",
    "__import__",
    "os.system",
    "os.popen",
    "os.execv",
    "os.execl",
    "os.fork",
    "os.spawn",
    "eval(",
    "exec(",
    "compile(",
    "open('/'",
    'open("/',
    "ctypes",
    "cffi",
    "importlib",
]

# ── プリアンブル（サンドボックス強化）────────────────────────────────────
_PREAMBLE = """\
import sys
import os

# ネットワーク無効化
import socket as _sock
_sock.socket = lambda *a, **k: (_ for _ in ()).throw(PermissionError("Network access disabled"))

# os.system / popen 無効化
os.system = lambda *a, **k: (_ for _ in ()).throw(PermissionError("os.system disabled"))
os.popen   = lambda *a, **k: (_ for _ in ()).throw(PermissionError("os.popen disabled"))

# /tmp 以外のファイル open を禁止
_builtin_open = open
def _safe_open(path, *args, **kwargs):
    spath = str(path)
    if spath.startswith('/') and not spath.startswith('/tmp'):
        raise PermissionError(f"File access outside /tmp is disabled: {path}")
    return _builtin_open(path, *args, **kwargs)

import builtins
builtins.open = _safe_open

# subprocess 無効化
import subprocess as _sp
_sp.Popen    = lambda *a, **k: (_ for _ in ()).throw(PermissionError("subprocess disabled"))
_sp.run      = lambda *a, **k: (_ for _ in ()).throw(PermissionError("subprocess disabled"))
_sp.call     = lambda *a, **k: (_ for _ in ()).throw(PermissionError("subprocess disabled"))
_sp.check_output = lambda *a, **k: (_ for _ in ()).throw(PermissionError("subprocess disabled"))

"""


def _static_check(code: str) -> Optional[str]:
    """禁止パターンの静的チェック。問題があればエラーメッセージを返す。"""
    code_lower = code.lower()
    for pat in _BLOCKED_PATTERNS:
        if pat.lower() in code_lower:
            return f"禁止されたコードパターン: '{pat}'"
    return None


def _preexec_limits():
    """サブプロセス起動前にリソース制限を設定（Unix専用）。"""
    try:
        # メモリ制限
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEM_BYTES, MAX_MEM_BYTES))
        # CPU時間制限（TIMEOUT_SEC + 2 秒マージン）
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_SEC + 2, TIMEOUT_SEC + 2))
        # プロセス数制限（fork 防止）
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    except Exception:
        pass   # 権限不足でも継続（タイムアウトで補完）


async def run_code(
    code: str,
    timeout: float = TIMEOUT_SEC,
    language: str = "python",
) -> dict:
    """
    コードを安全なサブプロセスで実行する。

    Args:
        code:     実行するソースコード
        timeout:  最大実行時間（秒）
        language: 現在は "python" のみ対応

    Returns:
        {
          "stdout":     str,   # 標準出力
          "stderr":     str,   # 標準エラー
          "exit_code":  int,   # 終了コード
          "elapsed_ms": int,   # 実行時間(ms)
          "error":      str,   # サンドボックスエラー (ある場合)
        }
    """
    if language != "python":
        return {
            "stdout": "", "stderr": "", "exit_code": 1,
            "elapsed_ms": 0, "error": f"Unsupported language: {language}",
        }

    # 静的チェック
    static_err = _static_check(code)
    if static_err:
        return {
            "stdout": "", "stderr": static_err, "exit_code": 1,
            "elapsed_ms": 0, "error": static_err,
        }

    # 一時ファイルにコードを書き出す
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir="/tmp", delete=False, prefix="sandbox_"
    ) as f:
        tmpfile = f.name
        f.write(_PREAMBLE)
        f.write(code)

    try:
        t0 = time.time()
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmpfile,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_preexec_limits,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "PYTHONPATH": "",
                "HOME": "/tmp",
                "TMPDIR": "/tmp",
            },
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            exit_code = proc.returncode
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.warning("Sandbox timeout after %.1fs", timeout)
            return {
                "stdout": "", "stderr": f"実行時間制限 ({timeout}秒) を超えました",
                "exit_code": -1, "elapsed_ms": elapsed_ms,
                "error": "timeout",
            }

        elapsed_ms = int((time.time() - t0) * 1000)

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT]

        logger.info("Sandbox: exit=%d, %.0fms, stdout=%d chars", exit_code, elapsed_ms, len(stdout))
        return {
            "stdout":     stdout,
            "stderr":     stderr,
            "exit_code":  exit_code,
            "elapsed_ms": elapsed_ms,
            "error":      "",
        }
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


async def run_code_with_packages(code: str, packages: list[str] | None = None) -> dict:
    """
    必要なパッケージのインポート可否を確認してからコードを実行。
    packages: 事前チェックするモジュール名リスト
    """
    if packages:
        check_code = "\n".join(f"import {p}" for p in packages)
        check = await run_code(check_code, timeout=5)
        if check["exit_code"] != 0:
            missing = [p for p in packages if p in check["stderr"]]
            if missing:
                return {
                    "stdout": "", "stderr": f"必要なパッケージが見つかりません: {missing}",
                    "exit_code": 1, "elapsed_ms": 0,
                    "error": f"missing_packages: {missing}",
                }
    return await run_code(code)
