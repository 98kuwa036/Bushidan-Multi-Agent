"""
core/code_verifier.py — コード実証主義ノード v1

LangGraph の sandbox_verify ノードから呼び出す。
shogun / gunshi が生成したコードブロックを抽出し、
ローカルサブプロセスで安全に実行して結果をレスポンスに付記する。

設計:
  - Python / Bash のコードブロックのみ対象
  - asyncio.create_subprocess_exec + timeout で隔離実行
  - 標準出力 200 文字 + 終了コードを response に付記
  - 失敗時はスキップ (エラーで止めない)
"""

import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

# 実行タイムアウト (秒)
_EXEC_TIMEOUT = 10

# 実行許可する言語
_ALLOWED_LANGS = {"python", "python3", "py", "bash", "sh"}

# 実行を拒否するキーワード (危険なコマンド)
_DENY_PATTERNS = [
    r"rm\s+-rf",
    r"sudo\s+",
    r"os\.system",
    r"subprocess\.call.*shell=True",
    r"__import__.*os",
    r"shutil\.rmtree",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
    r"format\s+[A-Za-z]:",  # Windows format
]
_DENY_RE = re.compile("|".join(_DENY_PATTERNS), re.IGNORECASE)


@dataclass
class VerifyResult:
    lang: str
    code: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self) -> str:
        if self.timed_out:
            return f"⏱️ タイムアウト ({_EXEC_TIMEOUT}秒)"
        if self.ok:
            out = self.stdout[:200].strip()
            return f"✅ 実行成功\n```\n{out}\n```" if out else "✅ 実行成功 (出力なし)"
        err = (self.stderr or self.stdout)[:200].strip()
        return f"❌ 実行失敗 (exit={self.exit_code})\n```\n{err}\n```"


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Markdown コードブロックを抽出する。

    Returns:
        List of (lang, code) tuples
    """
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    blocks = []
    for m in pattern.finditer(text):
        lang = m.group(1).lower().strip() or "python"
        code = m.group(2).strip()
        if lang in _ALLOWED_LANGS and code:
            blocks.append((lang, code))
    return blocks


async def _run_code(lang: str, code: str) -> VerifyResult:
    """コードを subprocess で実行する"""
    if lang in ("bash", "sh"):
        cmd = ["bash", "-c", code]
    else:
        cmd = [sys.executable, "-c", code]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 64,  # 64KB 上限
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=_EXEC_TIMEOUT
            )
            return VerifyResult(
                lang=lang,
                code=code,
                exit_code=proc.returncode or 0,
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return VerifyResult(
                lang=lang, code=code,
                exit_code=-1, stdout="", stderr="", timed_out=True,
            )
    except Exception as e:
        return VerifyResult(
            lang=lang, code=code,
            exit_code=-1, stdout="", stderr=str(e),
        )


async def verify_response(response: str, max_blocks: int = 2) -> str:
    """
    レスポンス内のコードブロックを実行検証する。

    Args:
        response:   LLM の応答テキスト
        max_blocks: 最大検証ブロック数

    Returns:
        検証結果サマリ文字列 ("ok" | "error: ..." | "skipped")
    """
    blocks = extract_code_blocks(response)
    if not blocks:
        return "skipped"

    # 危険パターンを含むブロックは全体をスキップ
    for lang, code in blocks:
        if _DENY_RE.search(code):
            logger.warning("⚠️ コード検証スキップ (危険パターン検出)")
            return "skipped"

    results: List[VerifyResult] = []
    tasks = [_run_code(lang, code) for lang, code in blocks[:max_blocks]]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    for r in raw:
        if isinstance(r, VerifyResult):
            results.append(r)
        elif isinstance(r, BaseException):
            logger.warning("コード検証タスク例外: %s", r)

    if not results:
        return "skipped"

    all_ok = all(r.ok for r in results)
    summaries = "\n".join(r.summary() for r in results)

    if all_ok:
        logger.info("✅ コード検証 OK (%d ブロック)", len(results))
        return f"ok: {summaries}"
    else:
        logger.info("❌ コード検証 失敗 (%d ブロック)", len(results))
        return f"error: {summaries}"


def append_verify_note(response: str, verify_result: str) -> str:
    """
    検証結果をレスポンスに付記する。

    skipped や短すぎる結果は付記しない。
    """
    if not verify_result or verify_result == "skipped":
        return response

    status, _, detail = verify_result.partition(": ")
    if status == "ok":
        note = f"\n\n---\n**🔬 実証検証**: {detail}"
    else:
        note = f"\n\n---\n**⚠️ 実証検証 (要確認)**: {detail}"

    return response + note
