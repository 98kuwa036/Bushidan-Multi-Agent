"""
console/maintenance.py — メンテナンス機能バックエンド v1

機能:
  - ログビューア (console.log / audit YAML / maintenance / journalctl)
  - システム情報 (CPU・メモリ・ディスク・サービス状態)
  - アップデート管理 (git fetch → sandbox venv → テスト → 本番適用)
  - パッケージ管理 (pip list / outdated / upgrade)
  - サービス管理 (systemctl --user start/stop/restart)
"""

import asyncio
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

# fire-and-forget タスク参照保持（GC 対策）
_bg_tasks: set = set()


def _fire(coro, *, name: str = None) -> "asyncio.Task":
    t = asyncio.create_task(coro, name=name)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return t

import psutil  # psutil は requirements に追加済み (v7.x)

_APP_DIR   = Path("/mnt/Bushidan-Multi-Agent")
_VENV      = Path("/home/claude/Bushidan-Multi-Agent/venv")
_SANDBOX   = Path("/home/claude/.venv_sandbox")
_PIP       = str(_VENV / "bin" / "pip")
_PYTHON    = str(_VENV / "bin" / "python")
_PIP_SB    = str(_SANDBOX / "bin" / "pip")
_PYTHON_SB = str(_SANDBOX / "bin" / "python")

# ────────────────────────────────────────────────────────────────
# ログ読み取り
# ────────────────────────────────────────────────────────────────

def _tail(path: str, lines: int = 300) -> str:
    """ファイルの末尾 N 行を文字列で返す"""
    p = Path(path)
    if not p.exists():
        return f"[ファイルが見つかりません: {path}]"
    try:
        result = subprocess.run(
            ["tail", f"-{lines}", str(p)],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout or "(空)"
    except Exception as e:
        return f"[読み取りエラー: {e}]"


def get_log_console(lines: int = 300) -> str:
    return _tail("/tmp/console.log", lines)


def get_log_maintenance(lines: int = 300) -> str:
    return _tail(str(_APP_DIR / "logs" / "maintenance.log"), lines)


def get_log_journal(service: str = "bushidan-console", lines: int = 200) -> str:
    try:
        result = subprocess.run(
            ["journalctl", "--user", "-u", service, "-n", str(lines), "--no-pager"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout or "(ログなし)"
    except Exception as e:
        return f"[journalctl エラー: {e}]"


def get_audit_dates() -> List[str]:
    """監査ログが存在する日付リストを返す (新しい順)"""
    audit_dir = _APP_DIR / "audit"
    if not audit_dir.exists():
        return []
    dates = []
    for d in sorted(audit_dir.iterdir(), reverse=True):
        if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name):
            dates.append(d.name)
    return dates[:30]


def get_audit_log(date: Optional[str] = None) -> str:
    """指定日付の監査ログを結合して返す"""
    if not date:
        dates = get_audit_dates()
        date = dates[0] if dates else datetime.date.today().isoformat()
    audit_dir = _APP_DIR / "audit" / date
    if not audit_dir.exists():
        # v18 形式も確認
        audit_dir = _APP_DIR / "audit" / "v18" / date
    if not audit_dir.exists():
        return f"[{date} の監査ログなし]"
    parts = []
    for f in sorted(audit_dir.iterdir()):
        if f.suffix in (".yaml", ".yml", ".log"):
            try:
                parts.append(f"--- {f.name} ---\n{f.read_text(encoding='utf-8', errors='replace')}")
            except Exception:
                pass
    return "\n\n".join(parts) if parts else "(ログなし)"


# ────────────────────────────────────────────────────────────────
# システム情報
# ────────────────────────────────────────────────────────────────

def get_system_info() -> Dict:
    """CPU・メモリ・ディスク・プロセス情報を返す"""
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # サービス状態
    services = {}
    for svc in ["bushidan-console", "jupyterlab"]:
        try:
            r = subprocess.run(
                ["systemctl", "--user", "is-active", svc],
                capture_output=True, text=True, timeout=3,
            )
            services[svc] = r.stdout.strip()
        except Exception:
            services[svc] = "unknown"

    # Git 情報
    try:
        git_hash = subprocess.run(
            ["git", "-C", str(_APP_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "-C", str(_APP_DIR), "branch", "--show-current"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
    except Exception:
        git_hash = git_branch = "unknown"

    return {
        "cpu_percent":   cpu,
        "mem_total_gb":  round(mem.total / 1e9, 1),
        "mem_used_gb":   round(mem.used  / 1e9, 1),
        "mem_percent":   mem.percent,
        "disk_total_gb": round(disk.total / 1e9, 1),
        "disk_used_gb":  round(disk.used  / 1e9, 1),
        "disk_percent":  disk.percent,
        "services":      services,
        "git_hash":      git_hash,
        "git_branch":    git_branch,
        "python_version": sys.version.split()[0],
        "uptime_seconds": int(time.time() - psutil.boot_time()),
    }


# ────────────────────────────────────────────────────────────────
# パッケージ管理
# ────────────────────────────────────────────────────────────────

def get_packages() -> List[Dict]:
    """インストール済みパッケージ一覧 (主要AIパッケージのみ)"""
    try:
        result = subprocess.run(
            [_PIP, "list", "--format=json"],
            capture_output=True, text=True, timeout=15,
        )
        pkgs = json.loads(result.stdout)
        # 重要パッケージを先頭に
        _PRIORITY = {
            "anthropic", "openai", "google-genai", "groq",
            "cohere", "mistralai", "langgraph", "langchain-core",
            "fastapi", "uvicorn", "notion-client", "aiohttp",
            "psycopg", "psutil",
        }
        priority = [p for p in pkgs if p["name"].lower() in _PRIORITY]
        rest = [p for p in pkgs if p["name"].lower() not in _PRIORITY]
        return priority + rest
    except Exception as e:
        return [{"name": "error", "version": str(e)}]


async def get_outdated_packages() -> List[Dict]:
    """更新可能なパッケージを非同期で取得"""
    try:
        proc = await asyncio.create_subprocess_exec(
            _PIP, "list", "--outdated", "--format=json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        return json.loads(stdout.decode())
    except Exception as e:
        return [{"name": "error", "version": str(e), "latest_version": "?"}]


async def upgrade_package_list(packages: list) -> Dict:
    """複数パッケージを一括インストール (apply-fix 用)"""
    try:
        proc = await asyncio.create_subprocess_exec(
            _PIP, "install", *packages,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_APP_DIR),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
        out = stdout.decode(errors="replace")
        if proc.returncode != 0:
            return {"success": False, "output": out}
        # 再チェック
        chk = await asyncio.create_subprocess_exec(
            _PIP, "check",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        chk_out, _ = await asyncio.wait_for(chk.communicate(), timeout=30)
        conflicts = chk_out.decode(errors="replace").strip()
        return {
            "success": True,
            "output": out,
            "conflicts": conflicts if conflicts and conflicts != "No broken requirements found." else "",
        }
    except Exception as e:
        return {"success": False, "output": str(e)}


async def upgrade_package(package: str) -> Dict:
    """単一パッケージをアップグレード (依存関係チェック付き)"""
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', package):
        return {"success": False, "output": "無効なパッケージ名"}
    try:
        # ① インストール
        proc = await asyncio.create_subprocess_exec(
            _PIP, "install", "--upgrade", package,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_APP_DIR),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        out = stdout.decode(errors="replace")

        if proc.returncode != 0:
            return {"success": False, "output": out}

        # ② pip check で依存競合を確認
        chk = await asyncio.create_subprocess_exec(
            _PIP, "check",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(_APP_DIR),
        )
        chk_out, _ = await asyncio.wait_for(chk.communicate(), timeout=30)
        conflicts = chk_out.decode(errors="replace").strip()

        return {
            "success": True,
            "output": out,
            "conflicts": conflicts if conflicts and conflicts != "No broken requirements found." else "",
        }
    except Exception as e:
        return {"success": False, "output": str(e)}


# ────────────────────────────────────────────────────────────────
# アップデート管理 (3段階)
# ────────────────────────────────────────────────────────────────

async def stream_update(stage: str) -> AsyncIterator[str]:
    """
    アップデートをストリーミング実行。SSE text/event-stream 形式で yield する。

    stage:
      "check"   — git fetch + 差分表示 (非破壊)
      "sandbox" — sandbox venv でのドライラン + インポートテスト
      "apply"   — 本番に git pull + pip install + 再起動
    """

    def _msg(text: str, level: str = "info") -> str:
        payload = json.dumps({"text": text, "level": level, "ts": time.time()})
        return f"data: {payload}\n\n"

    async def _run(cmd: List[str], cwd: str = str(_APP_DIR),
                   timeout: int = 120, env: Optional[Dict] = None) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return -1, f"タイムアウト ({timeout}s)"
        return proc.returncode, stdout.decode(errors="replace").strip()

    # ─── Stage 1: チェック ────────────────────────────────────────
    if stage == "check":
        yield _msg("🔍 git fetch origin...")
        rc_fetch, out_fetch = await _run(["git", "fetch", "origin"])
        if rc_fetch != 0:
            yield _msg(
                f"⚠️ git fetch が制限されています (bind-mount 権限制限):\n{out_fetch}\n"
                "→ 既存のリモート追跡情報を使用します",
                "warn",
            )
        else:
            yield _msg("  └ fetch 完了", "success")

        yield _msg("\n📋 ローカル vs origin/main:")
        _, ahead = await _run(
            ["git", "log", "origin/main..HEAD", "--oneline", "--no-merges"]
        )
        yield _msg(f"  ↑ ローカルのみ ({len(ahead.splitlines())}件):\n{ahead or '（なし）'}")

        _, behind = await _run(
            ["git", "log", "HEAD..origin/main", "--oneline", "--no-merges"]
        )
        yield _msg(f"  ↓ リモートのみ ({len(behind.splitlines())}件):\n{behind or '（なし）'}")

        yield _msg("\n📊 最新コミット (直近5件):")
        _, recent = await _run(
            ["git", "log", "--oneline", "-5", "--format=%h %s (%ar)"]
        )
        yield _msg(recent or "（なし）")

        yield _msg("\n📦 パッケージ変更確認 (dry-run):")
        rc, dry = await _run([
            _PIP, "install", "-r", "requirements.txt", "--dry-run",
        ], timeout=60)
        # "Would install X-1.0" 行だけ抽出（Requirement already satisfied は除外）
        would_install = [line for line in dry.splitlines() if "Would install" in line or "would install" in line.lower()]
        if would_install:
            yield _msg("  インストール予定:\n" + "\n".join(would_install), "warn")
        else:
            yield _msg("  └ すべてのパッケージが最新です", "success")
        yield _msg("\n✅ チェック完了。'サンドボックス検証' で安全性を確認してください。", "success")

    # ─── Stage 2: サンドボックス検証 ─────────────────────────────
    elif stage == "sandbox":
        yield _msg("🏗️ サンドボックス venv を準備中...")

        # sandbox venv が存在しなければコピー
        if not _SANDBOX.exists():
            yield _msg("  └ ~/.venv_sandbox を作成中 (初回のみ時間がかかります)...")
            rc, out = await _run(
                [sys.executable, "-m", "venv", str(_SANDBOX)],
                cwd="/tmp", timeout=60,
            )
            if rc != 0:
                yield _msg(f"❌ sandbox venv 作成失敗:\n{out}", "error")
                return
            yield _msg("  └ venv 作成完了", "success")

        yield _msg("📦 sandbox にパッケージをインストール中...")
        rc, out = await _run(
            [_PIP_SB, "install", "-r", "requirements.txt", "-q"],
            timeout=180,
        )
        if rc != 0:
            yield _msg(f"⚠️ 一部インストール失敗:\n{out[:1000]}", "warn")
        else:
            yield _msg("  └ パッケージインストール完了", "success")

        yield _msg("\n🧪 インポートテスト実行中...")
        test_imports = [
            "from console.app import app",
            "from core.langgraph_router import LangGraphRouter",
            "from roles.base import BaseRole",
            "from utils.client_registry import ClientRegistry",
            "from integrations.notion.client import get_notion_client",
        ]
        rc, out = await _run(
            [_PYTHON_SB, "-c", "; ".join(test_imports)],
            timeout=30,
        )
        if rc == 0:
            yield _msg("  └ インポートテスト: 全て正常", "success")
        else:
            yield _msg(f"❌ インポートテスト失敗:\n{out}", "error")
            return

        yield _msg("\n🔬 基本動作テスト実行中...")
        rc, out = await _run(
            [_PYTHON_SB, "-m", "pytest", "tests/", "-q", "--tb=short",
             "-x", "--ignore=tests/integration",
             "-W", "ignore::pytest.PytestCacheWarning"],
            timeout=120,
        )
        # rc=0: passed / rc=5: no tests collected (どちらも正常)
        # -W ignore::pytest.PytestCacheWarning で pytest.ini の filterwarnings=error を上書き
        if rc in (0, 5):
            msg = "テストなし (tests/ ディレクトリが空)" if rc == 5 else (out.splitlines()[-1] if out else "passed")
            yield _msg(f"  └ {msg}", "success")
        else:
            yield _msg(f"⚠️ テスト失敗 (内容を確認して続行してください):\n{out[:2000]}", "warn")

        yield _msg("\n✅ サンドボックス検証完了。'本番適用' で反映できます。", "success")

    # ─── Stage 3: 本番適用 ───────────────────────────────────────
    elif stage == "apply":
        yield _msg("🚀 本番環境にアップデートを適用します...")

        yield _msg("\n  ① git pull origin main...")
        rc, out = await _run(["git", "pull", "origin", "main"], timeout=60)
        if rc != 0:
            yield _msg(f"⚠️ git pull 失敗 (ローカル変更がある場合は手動マージが必要):\n{out[:500]}", "warn")
        else:
            yield _msg(f"  └ コード更新完了\n{out[:300]}", "success")

        yield _msg("\n  ② pip install -r requirements.txt...")
        rc, out = await _run([_PIP, "install", "-r", "requirements.txt", "-q"], timeout=180)
        if rc != 0:
            yield _msg(f"⚠️ パッケージインストール警告:\n{out[:500]}", "warn")
        else:
            yield _msg("  └ パッケージ更新完了", "success")

        # 再起動はバックグラウンドで遅延実行し、完了メッセージを先に送信する
        # (即時 systemctl restart だと SSE 接続が切れてエラー扱いになるため)
        yield _msg("\n✅ アップデート適用完了！3秒後にサービスを再起動します...", "success")
        _fire(_delayed_systemd_restart(), name="delayed_systemd_restart")


async def _delayed_systemd_restart():
    """3秒後に systemd 経由でサービスを再起動"""
    await asyncio.sleep(3)
    _uid = os.getuid()
    _env = {
        **os.environ,
        "XDG_RUNTIME_DIR": f"/run/user/{_uid}",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{_uid}/bus",
    }
    proc = await asyncio.create_subprocess_exec(
        "systemctl", "--user", "restart", "bushidan-console",
        cwd="/tmp", env=_env,
    )
    rc = await proc.wait()
    if rc != 0:
        # フォールバック: uvicorn を直接再起動
        await asyncio.create_subprocess_exec(
            "pkill", "-f", "uvicorn console.app", cwd="/tmp"
        )
        await asyncio.sleep(1)
        subprocess.Popen(
            [_PYTHON, "-m", "uvicorn", "console.app:app",
             "--host", "0.0.0.0", "--port", "8067"],
            cwd=str(_APP_DIR),
            stdout=open("/tmp/console.log", "a"),
            stderr=subprocess.STDOUT,
        )


async def _delayed_restart():
    """3秒後に uvicorn を再起動 (サービスファイルなし環境用)"""
    await asyncio.sleep(3)
    subprocess.Popen(
        [_PYTHON, "-m", "uvicorn", "console.app:app",
         "--host", "0.0.0.0", "--port", "8067"],
        cwd=str(_APP_DIR),
        stdout=open("/tmp/console.log", "a"),
        stderr=subprocess.STDOUT,
    )


# ────────────────────────────────────────────────────────────────
# サービス管理
# ────────────────────────────────────────────────────────────────

_ALLOWED_SERVICES = frozenset([
    "bushidan-console", "jupyterlab",
])


async def service_action(service: str, action: str) -> Dict:
    """サービスの起動/停止/再起動 (許可リストのみ)"""
    if service not in _ALLOWED_SERVICES:
        return {"success": False, "output": f"不許可サービス: {service}"}
    if action not in ("start", "stop", "restart", "status"):
        return {"success": False, "output": f"不正なアクション: {action}"}
    try:
        _uid = os.getuid()
        _env = {
            **os.environ,
            "XDG_RUNTIME_DIR": f"/run/user/{_uid}",
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{_uid}/bus",
        }
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "--user", action, service,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=_env,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        return {
            "success": proc.returncode == 0,
            "output": stdout.decode(errors="replace"),
        }
    except Exception as e:
        return {"success": False, "output": str(e)}
