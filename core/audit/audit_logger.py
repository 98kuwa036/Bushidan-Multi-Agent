"""
武士団 v18 — Phase 3 時刻ベース YAML 監査ロガー

ファイル構造:
  audit/v18/
    YYYY-MM-DD/
      bushidan-hour-00.yaml   # 00:xx の全エントリ
      bushidan-hour-01.yaml
      ...
      bushidan-hour-23.yaml

各 YAML ファイルはリスト形式:
  - entries:
      - id: ...
        timestamp: ...
        ...

90日以上前のファイルは自動削除（cleanup_old_logs で実行）

設計方針:
  - asyncio.Lock で同一ファイルへの同時書き込みを防止
  - pyyaml 不要（カスタムシリアライザ使用）
  - 例外は全てサイレントに無視（監査がビジネスロジックをブロックしない）
"""
from __future__ import annotations

import asyncio
import datetime
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# 監査ログのルートディレクトリ
_AUDIT_ROOT = Path(
    os.environ.get("AUDIT_DIR", "/mnt/Bushidan-Multi-Agent/audit")
) / "v18"

# 保持日数（デフォルト: 90日）
_RETENTION_DAYS = int(os.environ.get("AUDIT_RETENTION_DAYS", "90"))

# ファイルごとのロック
_file_locks: Dict[str, asyncio.Lock] = {}
_lock_map_lock = asyncio.Lock()


async def _get_file_lock(path: str) -> asyncio.Lock:
    async with _lock_map_lock:
        if path not in _file_locks:
            _file_locks[path] = asyncio.Lock()
        return _file_locks[path]


# ── YAML シリアライザ ─────────────────────────────────────────────────────

def _yaml_scalar(val: Any, indent: int = 0) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    if not s:
        return '""'
    if "\n" in s or len(s) > 80:
        pad = " " * (indent + 2)
        lines = s.replace("\r\n", "\n").split("\n")
        body = "\n".join(f"{pad}{line}" for line in lines)
        return f"|\n{body}"
    # シングルライン
    if any(c in s for c in ('"', "'", ':', '{', '}', '[', ']', '#', '&', '*', '!', '|', '>')):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _render_yaml(obj: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                if isinstance(v, dict) and not v:
                    lines.append(f"{pad}{k}: {{}}")
                elif isinstance(v, list) and not v:
                    lines.append(f"{pad}{k}: []")
                else:
                    lines.append(f"{pad}{k}:")
                    lines.append(_render_yaml(v, indent + 2))
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v, indent)}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, dict):
                items = list(item.items())
                if items:
                    first_k, first_v = items[0]
                    if isinstance(first_v, (dict, list)):
                        lines.append(f"{pad}- {first_k}:")
                        lines.append(_render_yaml(first_v, indent + 4))
                    else:
                        lines.append(f"{pad}- {first_k}: {_yaml_scalar(first_v, indent + 2)}")
                    for k, v in items[1:]:
                        if isinstance(v, (dict, list)):
                            lines.append(f"{pad}  {k}:")
                            lines.append(_render_yaml(v, indent + 4))
                        else:
                            lines.append(f"{pad}  {k}: {_yaml_scalar(v, indent + 2)}")
                else:
                    lines.append(f"{pad}- {{}}")
            else:
                lines.append(f"{pad}- {_yaml_scalar(item, indent + 2)}")
        return "\n".join(lines)
    else:
        return f"{pad}{_yaml_scalar(obj, indent)}"


# ── データモデル ──────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """1回の処理に対する監査エントリ"""
    id: str                          # thread_id + timestamp
    timestamp: str                   # ISO 8601
    thread_id: str
    user_input_summary: str          # 最大 200 文字
    agent_role: str
    complexity: str
    intent_type: str
    selected_role: str
    notion_score: float
    karasu_results: int
    response_summary: str            # 最大 300 文字
    execution_time_ms: float
    pipeline_ms: float
    stage: str                       # e.g. "groq_draft+haiku_polish"
    success: bool = True
    error: str = ""
    # Phase 1 詳細
    uchu_confidence: float = 0.0
    routing_confidence: float = 0.0
    routing_path: List[str] = field(default_factory=list)
    routing_reasoning: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # リストを短縮
        d["routing_path"] = d["routing_path"][:5]
        return d


# ── メインロガークラス ────────────────────────────────────────────────────

class AuditLogger:
    """時刻ベース YAML 監査ロガー（シングルトン推奨）"""

    _instance: Optional[AuditLogger] = None

    @classmethod
    def get(cls) -> AuditLogger:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._root = _AUDIT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)

    def _get_log_path(self, dt: Optional[datetime.datetime] = None) -> Path:
        """時刻に対応する YAML ファイルパスを返す"""
        now = dt or datetime.datetime.now()
        date_dir = self._root / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        hour = now.strftime("%H")
        return date_dir / f"bushidan-hour-{hour}.yaml"

    async def write(self, entry: AuditEntry) -> None:
        """エントリを非同期で追記"""
        try:
            path = self._get_log_path()
            file_lock = await _get_file_lock(str(path))

            async with file_lock:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._append_entry, path, entry
                )
        except Exception as e:
            logger.debug("AuditLogger.write 失敗 (サイレント): %s", e)

    def _append_entry(self, path: Path, entry: AuditEntry) -> None:
        """同期ファイル追記（executor から実行）"""
        header = (
            f"# 武士団 Bushidan v18 — 時刻別監査ログ\n"
            f"# {path.parent.name} {path.stem.replace('bushidan-', '')}\n"
        )

        entry_yaml = _render_yaml(entry.to_dict(), indent=2)
        new_block = f"\n- {entry_yaml.lstrip()}\n"

        if path.exists():
            path.write_text(
                path.read_text(encoding="utf-8") + new_block,
                encoding="utf-8",
            )
        else:
            path.write_text(header + "entries:\n" + new_block, encoding="utf-8")

    def cleanup_old_logs(self, retention_days: int = _RETENTION_DAYS) -> int:
        """保持期限を超えたログディレクトリを削除。削除ファイル数を返す。"""
        count = 0
        try:
            cutoff = datetime.date.today() - datetime.timedelta(days=retention_days)
            if not self._root.exists():
                return 0
            for day_dir in sorted(self._root.iterdir()):
                if not day_dir.is_dir():
                    continue
                try:
                    day = datetime.date.fromisoformat(day_dir.name)
                except ValueError:
                    continue
                if day < cutoff:
                    for f in day_dir.iterdir():
                        f.unlink(missing_ok=True)
                        count += 1
                    day_dir.rmdir()
                    logger.info("監査ログ削除: %s (%d files)", day_dir.name, count)
        except Exception as e:
            logger.warning("cleanup_old_logs 失敗: %s", e)
        return count

    def list_logs(self, days: int = 7) -> List[dict]:
        """直近 N 日分のログファイル一覧を返す"""
        result = []
        try:
            cutoff = datetime.date.today() - datetime.timedelta(days=days)
            if not self._root.exists():
                return []
            for day_dir in sorted(self._root.iterdir(), reverse=True):
                if not day_dir.is_dir():
                    continue
                try:
                    day = datetime.date.fromisoformat(day_dir.name)
                except ValueError:
                    continue
                if day < cutoff:
                    break
                for f in sorted(day_dir.iterdir(), reverse=True):
                    if f.name.endswith(".yaml"):
                        result.append({
                            "date": day_dir.name,
                            "hour": re.search(r"hour-(\d+)", f.name).group(1) if re.search(r"hour-(\d+)", f.name) else "",
                            "filename": f.name,
                            "path": str(f),
                            "size_bytes": f.stat().st_size,
                        })
        except Exception as e:
            logger.warning("list_logs 失敗: %s", e)
        return result


# ── ヘルパー関数 ─────────────────────────────────────────────────────────

async def write_pipeline_audit(
    thread_id: str,
    user_input: str,
    agent_role: str,
    response: str,
    execution_time_ms: float,
    pipeline_ms: float = 0.0,
    stage: str = "",
    complexity: str = "medium",
    intent_type: str = "qa",
    selected_role: str = "auto",
    notion_score: float = 0.0,
    karasu_results: int = 0,
    uchu_confidence: float = 0.0,
    routing_confidence: float = 0.0,
    routing_path: Optional[List[str]] = None,
    routing_reasoning: str = "",
    success: bool = True,
    error: str = "",
) -> None:
    """パイプライン実行後に監査エントリを書き込むヘルパー"""
    now = datetime.datetime.now()
    entry = AuditEntry(
        id=f"{thread_id[:16]}-{now.strftime('%H%M%S%f')[:12]}",
        timestamp=now.isoformat(),
        thread_id=thread_id,
        user_input_summary=user_input[:200],
        agent_role=agent_role,
        complexity=complexity,
        intent_type=intent_type,
        selected_role=selected_role,
        notion_score=round(notion_score, 4),
        karasu_results=karasu_results,
        response_summary=response[:300],
        execution_time_ms=round(execution_time_ms, 1),
        pipeline_ms=round(pipeline_ms, 1),
        stage=stage,
        success=success,
        error=error[:200] if error else "",
        uchu_confidence=round(uchu_confidence, 3),
        routing_confidence=round(routing_confidence, 3),
        routing_path=routing_path or [],
        routing_reasoning=routing_reasoning[:300],
    )
    await AuditLogger.get().write(entry)
