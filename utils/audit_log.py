"""utils/audit_log.py — ロードマップ実行の YAML 監査ログ

Shogun の「ファイルベース透明性」を Bushidan に統合。
全ロードマップ実行を audit/YYYY-MM-DD/ 以下に YAML で書き出す。

構造:
  audit/
    2026-04-11/
      rm-thread_id-HHmmss.yaml   ← ロードマップ開始時
      rm-thread_id-HHmmss-done.yaml  ← 完了・監査後に更新
    2026-04-12/
      ...

単発チャット (ロードマップなし) は simple-chat として記録するかどうかは
AUDIT_SIMPLE_CHAT 環境変数で制御 (デフォルト: False)。
"""

import datetime
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 監査ログの出力先 (環境変数で上書き可)
_AUDIT_DIR = Path(
    os.environ.get("AUDIT_DIR", "/mnt/Bushidan-Multi-Agent/audit")
)
_AUDIT_SIMPLE_CHAT = os.environ.get("AUDIT_SIMPLE_CHAT", "").lower() in ("1", "true", "yes")


def _safe_str(v: Any, max_len: int = 400) -> str:
    """確実に str 化してトランケート"""
    s = str(v) if v is not None else ""
    return s[:max_len] + ("…" if len(s) > max_len else "")


def _yaml_block(value: str, indent: int = 2) -> str:
    """複数行文字列を YAML リテラルブロックスタイル (|) で返す"""
    if not value:
        return '""'
    if "\n" in value or len(value) > 80:
        pad = " " * indent
        lines = value.replace("\r\n", "\n").split("\n")
        body = "\n".join(f"{pad}{line}" for line in lines)
        return f"|\n{body}"
    # シングルライン — ダブルクォートでエスケープ
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_yaml(data: dict, indent: int = 0) -> str:
    """最小限の YAML シリアライザ (pyyaml 不要・日本語対応)"""
    lines = []
    pad = " " * indent
    for key, val in data.items():
        if isinstance(val, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_render_yaml(val, indent + 2))
        elif isinstance(val, list):
            if not val:
                lines.append(f"{pad}{key}: []")
            else:
                lines.append(f"{pad}{key}:")
                for item in val:
                    if isinstance(item, dict):
                        first = True
                        for k2, v2 in item.items():
                            prefix = f"{pad}  - " if first else f"{pad}    "
                            first = False
                            if isinstance(v2, (dict, list)):
                                lines.append(f"{prefix}{k2}:")
                                lines.append(_render_yaml({k2: v2}, indent + 6).lstrip())
                            else:
                                v2_str = _yaml_block(str(v2) if v2 is not None else "", indent + 4)
                                lines.append(f"{prefix}{k2}: {v2_str}")
                    else:
                        lines.append(f"{pad}  - {_yaml_block(str(item), indent + 4)}")
        elif isinstance(val, bool):
            lines.append(f"{pad}{key}: {'true' if val else 'false'}")
        elif isinstance(val, (int, float)):
            lines.append(f"{pad}{key}: {val}")
        elif val is None:
            lines.append(f"{pad}{key}: null")
        else:
            v_str = _yaml_block(str(val), indent + 2)
            lines.append(f"{pad}{key}: {v_str}")
    return "\n".join(lines)


class AuditLog:
    """
    ロードマップ実行の YAML 監査ログ。

    使い方:
        log = AuditLog.start(thread_id, message, roadmap)
        # ... 実行中 ...
        log.add_step_result(step_id, role, capability, result, elapsed)
        # ... 完了後 ...
        log.finish(final_response, audit_verdict=None)
    """

    def __init__(self, thread_id: str, message: str, roadmap: dict):
        now = datetime.datetime.now()
        self.thread_id = thread_id
        self.started_at = now
        self.date_str = now.strftime("%Y-%m-%d")
        self.time_str = now.strftime("%H%M%S")

        # スレッドIDをファイル名に使える形に正規化
        safe_tid = re.sub(r"[^a-zA-Z0-9_\-]", "_", thread_id[:20])
        self._filename_base = f"rm-{safe_tid}-{self.time_str}"
        self._dir = _AUDIT_DIR / self.date_str
        self._dir.mkdir(parents=True, exist_ok=True)

        self._data: dict = {
            "id":           f"{safe_tid}-{self.time_str}",
            "thread_id":    thread_id,
            "started_at":   now.isoformat(),
            "user_request": _safe_str(message, 500),
            "goal":         _safe_str(roadmap.get("goal", ""), 200),
            "complexity":   "",
            "steps":        [],
            "step_results": [],
            "audit":        None,
            "final_response": "",
            "total_execution_time": 0.0,
            "status":       "running",
        }

        # ステップ定義を記録
        for s in roadmap.get("steps", []):
            self._data["steps"].append({
                "id":           s.get("id", 0),
                "task":         _safe_str(s.get("task", ""), 200),
                "capability":   s.get("capability", ""),
                "assigned_role": s.get("assigned_role", ""),
                "can_parallel": bool(s.get("can_parallel", False)),
            })

        self._write(suffix="")
        logger.info("📋 監査ログ開始: %s/%s.yaml", self.date_str, self._filename_base)

    # ── クラスメソッド ────────────────────────────────────────────────

    @classmethod
    def start(cls, thread_id: str, message: str, roadmap: dict) -> "AuditLog":
        """ロードマップ開始時に呼び出す"""
        try:
            return cls(thread_id, message, roadmap)
        except Exception as e:
            logger.warning("AuditLog.start 失敗 (スキップ): %s", e)
            return _NullAuditLog()  # type: ignore

    # ── インスタンスメソッド ───────────────────────────────────────────

    def set_complexity(self, complexity: str) -> None:
        self._data["complexity"] = complexity

    def add_step_result(
        self,
        step_id: int,
        role: str,
        capability: str,
        task: str,
        result: str,
        execution_time: float,
        error: str = "",
    ) -> None:
        """ステップ完了時に呼び出す"""
        try:
            self._data["step_results"].append({
                "step_id":        step_id,
                "role":           role,
                "capability":     capability,
                "task":           _safe_str(task, 200),
                "result_summary": _safe_str(result, 300),
                "execution_time": round(execution_time, 2),
                "error":          _safe_str(error, 200) if error else "",
            })
            self._write(suffix="")
        except Exception as e:
            logger.debug("add_step_result 失敗: %s", e)

    def add_audit(self, verdict: str, comments: str, execution_time: float) -> None:
        """大元帥監査完了時に呼び出す"""
        try:
            self._data["audit"] = {
                "by":             "daigensui",
                "verdict":        _safe_str(verdict, 20),
                "comments":       _safe_str(comments, 500),
                "execution_time": round(execution_time, 2),
                "audited_at":     datetime.datetime.now().isoformat(),
            }
            self._write(suffix="")
        except Exception as e:
            logger.debug("add_audit 失敗: %s", e)

    def finish(self, final_response: str) -> None:
        """実行完了時に呼び出す。-done.yaml として書き出す。"""
        try:
            elapsed = (datetime.datetime.now() - self.started_at).total_seconds()
            self._data["final_response"] = _safe_str(final_response, 600)
            self._data["total_execution_time"] = round(elapsed, 2)
            self._data["status"] = "completed"
            self._data["completed_at"] = datetime.datetime.now().isoformat()
            self._write(suffix="-done")
            logger.info(
                "✅ 監査ログ完了: %s/%s-done.yaml (%.1fs)",
                self.date_str, self._filename_base, elapsed,
            )
        except Exception as e:
            logger.debug("AuditLog.finish 失敗: %s", e)

    def _write(self, suffix: str) -> None:
        path = self._dir / f"{self._filename_base}{suffix}.yaml"
        header = (
            f"# 武士団 Bushidan v18 — ロードマップ実行監査ログ\n"
            f"# 生成日時: {datetime.datetime.now().isoformat()}\n"
            f"# thread_id: {self.thread_id}\n"
            f"---\n"
        )
        content = header + _render_yaml(self._data)
        path.write_text(content, encoding="utf-8")


def write_simple_chat(
    thread_id: str,
    message: str,
    response: str,
    agent_role: str,
    handled_by: str,
    execution_time: float,
) -> None:
    """
    単発チャット (ロードマップなし) の監査ログ。
    AUDIT_SIMPLE_CHAT=true の場合のみ書き出す。
    """
    if not _AUDIT_SIMPLE_CHAT:
        return
    try:
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        safe_tid = re.sub(r"[^a-zA-Z0-9_\-]", "_", thread_id[:20])
        path = _AUDIT_DIR / date_str / f"chat-{safe_tid}-{time_str}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "id":             f"{safe_tid}-{time_str}",
            "type":           "simple_chat",
            "thread_id":      thread_id,
            "created_at":     now.isoformat(),
            "user_request":   _safe_str(message, 500),
            "agent_role":     agent_role,
            "handled_by":     handled_by,
            "response_summary": _safe_str(response, 300),
            "execution_time": round(execution_time, 2),
        }
        header = f"# 武士団 Bushidan v18 — 単発チャットログ\n# {now.isoformat()}\n---\n"
        path.write_text(header + _render_yaml(data), encoding="utf-8")
    except Exception as e:
        logger.debug("write_simple_chat 失敗: %s", e)


def list_audit_logs(days: int = 7) -> list[dict]:
    """
    直近 N 日分の監査ログ一覧を返す。
    各エントリ: {date, filename, path, size, is_done}
    """
    result = []
    try:
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        if not _AUDIT_DIR.exists():
            return []
        for day_dir in sorted(_AUDIT_DIR.iterdir(), reverse=True):
            if not day_dir.is_dir():
                continue
            try:
                day = datetime.date.fromisoformat(day_dir.name)
            except ValueError:
                continue
            if day < cutoff:
                break
            for f in sorted(day_dir.iterdir(), reverse=True):
                if not f.name.endswith(".yaml"):
                    continue
                result.append({
                    "date":      day_dir.name,
                    "filename":  f.name,
                    "path":      str(f),
                    "size":      f.stat().st_size,
                    "is_done":   f.name.endswith("-done.yaml"),
                    "type":      "roadmap" if f.name.startswith("rm-") else "simple",
                })
    except Exception as e:
        logger.warning("list_audit_logs 失敗: %s", e)
    return result


# ── Null オブジェクト (エラー時の安全な代替) ─────────────────────────────

class _NullAuditLog:
    """AuditLog の生成に失敗した場合の何もしない代替"""
    def set_complexity(self, *a, **kw): pass
    def add_step_result(self, *a, **kw): pass
    def add_audit(self, *a, **kw): pass
    def finish(self, *a, **kw): pass
