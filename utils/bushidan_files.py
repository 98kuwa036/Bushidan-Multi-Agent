"""
utils/bushidan_files.py — /home/claude/Bushidan ファイルアクセスユーティリティ

全ロールが安全にアクセスできるファイル共有領域。
パスは /home/claude/Bushidan 以下に制限される。

■ 読み込み
    from utils.bushidan_files import extract_file_refs, read_file, get_directory_listing

■ 書き込み (LLMレスポンスから自動保存)
    LLMが以下の形式でレスポンスに含めると自動的にファイルが保存される:

    [FILE:ファイル名.txt]
    ここにファイルの内容
    [/FILE]
"""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BUSHIDAN_DIR = Path("/home/claude/Bushidan")

# メッセージ内のファイル参照パターン
_PATH_PATTERNS = [
    re.compile(r'/home/claude/Bushidan/[\w\-./]+'),
    re.compile(r'~/Bushidan/[\w\-./]+'),
    re.compile(r'\bBushidan/[\w\-./]+'),
    re.compile(r'(?:ファイル|file|参照|read)[:\s]+([^\s,\n「」【】]+)', re.IGNORECASE),
]

# LLMレスポンス内のファイル書き込み指示パターン
# [FILE:path/to/file.txt] ... [/FILE]
_WRITE_PATTERN = re.compile(
    r'\[FILE:([^\]]+)\]\s*\n(.*?)\n\[/FILE\]',
    re.DOTALL,
)


def _safe_resolve(path_str: str) -> Optional[Path]:
    """パスを安全に解決。Bushidan外はNoneを返す。"""
    path_str = path_str.strip().strip("「」【】")
    path_str = path_str.replace("~/Bushidan", str(BUSHIDAN_DIR))
    path_str = path_str.replace("~", str(Path.home()))

    # "Bushidan/xxx" 形式 → BUSHIDAN_DIR/xxx として解釈
    if path_str.startswith("Bushidan/"):
        path_str = str(BUSHIDAN_DIR / path_str[len("Bushidan/"):])

    candidates = [Path(path_str), BUSHIDAN_DIR / path_str]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if str(resolved).startswith(str(BUSHIDAN_DIR) + "/") or resolved == BUSHIDAN_DIR:
                return resolved
        except Exception:
            continue
    return None


# ─── 読み込み ──────────────────────────────────────────────────────────────


def extract_file_refs(message: str) -> list[str]:
    """メッセージからBushidanファイルパス参照を抽出して実パスのリストを返す。"""
    refs = []
    for pattern in _PATH_PATTERNS:
        for m in pattern.finditer(message):
            refs.append(m.group(1) if m.lastindex else m.group(0))

    valid: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        path = _safe_resolve(ref)
        if path and path.is_file() and str(path) not in seen:
            seen.add(str(path))
            valid.append(str(path))
    return valid


def read_file(path_str: str, max_chars: int = 8000) -> str:
    """ファイルを読み込む。Bushidan外のパスはValueErrorを送出。"""
    path = _safe_resolve(path_str)
    if not path:
        raise ValueError(f"Bushidan外のパスは参照できません: {path_str}")
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path_str}")
    if not path.is_file():
        raise ValueError(f"ファイルではありません: {path_str}")

    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n... (省略: {len(content) - max_chars}文字以降)"
    return content


def get_directory_listing(max_depth: int = 3, max_files: int = 60) -> str:
    """Bushidanディレクトリの構造を返す。ファイルが存在しない場合は空文字列。"""
    if not BUSHIDAN_DIR.exists():
        return ""

    lines: list[str] = []
    counter = [0]

    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth or counter[0] >= max_files:
            return
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError:
            return
        for item in items:
            if counter[0] >= max_files:
                lines.append(f"{prefix}... (省略)")
                return
            if item.is_dir():
                lines.append(f"{prefix}📁 {item.name}/")
                _walk(item, prefix + "  ", depth + 1)
            else:
                lines.append(f"{prefix}📄 {item.name}")
                counter[0] += 1

    _walk(BUSHIDAN_DIR, "  ", 0)

    if not lines:
        return ""

    return "📁 /home/claude/Bushidan/\n" + "\n".join(lines)


# ─── 書き込み ─────────────────────────────────────────────────────────────


def write_file(path_str: str, content: str) -> str:
    """
    ファイルを書き込む。Bushidan外のパスはValueErrorを送出。
    親ディレクトリが存在しない場合は自動作成。

    Returns:
        書き込んだ実パス文字列
    """
    path = _safe_resolve(path_str)
    if not path:
        raise ValueError(f"Bushidan外のパスには書き込めません: {path_str}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("📝 Bushidan書き込み: %s (%d文字)", path, len(content))
    return str(path)


def extract_and_save_files(response: str) -> list[str]:
    """
    LLMレスポンスから [FILE:xxx]...[/FILE] ブロックを検出して保存する。

    Returns:
        保存したファイルパスのリスト
    """
    saved: list[str] = []
    for m in _WRITE_PATTERN.finditer(response):
        path_str = m.group(1).strip()
        content = m.group(2)
        try:
            saved_path = write_file(path_str, content)
            saved.append(saved_path)
        except Exception as e:
            logger.warning("Bushidanファイル保存失敗 (%s): %s", path_str, e)
    return saved


# ─── システムプロンプト構築 ───────────────────────────────────────────────


_WRITE_INSTRUCTION = """\
ファイルをBushidan領域に保存する場合は以下の形式を使用してください:
[FILE:ファイル名.txt]
ここにファイルの内容を記述
[/FILE]
サブディレクトリも使用可能です (例: [FILE:notes/memo.md])"""


def build_file_context(message: str) -> str:
    """
    メッセージを解析し、参照ファイルの内容・ディレクトリ構造・書き込み手順を
    システムプロンプト注入用のテキストとして返す。
    """
    parts: list[str] = []

    # ディレクトリ構造 (Bushidanにファイルがある場合のみ)
    listing = get_directory_listing()
    if listing:
        parts.append(f"【Bushidan共有ファイル領域】\n{listing}\n\n{_WRITE_INSTRUCTION}")
    else:
        # ファイルがなくても書き込み方法は案内する
        parts.append(f"【Bushidan共有ファイル領域】\n📁 /home/claude/Bushidan/ (現在空)\n\n{_WRITE_INSTRUCTION}")

    # 明示的に参照されたファイルの内容を注入
    refs = extract_file_refs(message)
    if refs:
        file_parts: list[str] = []
        for path_str in refs:
            try:
                content = read_file(path_str)
                rel = str(Path(path_str).relative_to(BUSHIDAN_DIR))
                file_parts.append(f"--- {rel} ---\n{content}")
            except Exception:
                pass
        if file_parts:
            parts.append("【参照ファイル内容】\n" + "\n\n".join(file_parts))

    return "\n\n".join(parts)
