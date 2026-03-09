"""
Claude CLI クライアント — Proプラン優先 → API フォールバック

Claude Code CLI (claude -p) を使ってProプラン枠でリクエストし、
失敗した場合のみ Anthropic API (有料) にフォールバックする。

使い方:
    from utils.claude_cli_client import call_claude_with_fallback

    result = await call_claude_with_fallback(
        prompt="コードをレビューして",
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY,
        system="あなたは将軍です",
    )
"""

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# CLI出力でエラーと判断するキーワード
_CLI_ERROR_PATTERNS = [
    "usage limit reached",
    "rate limit",
    "authentication failed",
    "invalid api key",
    "error:",
    "claude: command not found",
    "permission denied",
]

# CLIのデフォルトタイムアウト (秒)
_CLI_TIMEOUT = 120


async def _try_claude_cli(prompt: str, model: str, system: Optional[str] = None) -> Optional[str]:
    """
    claude CLI でリクエストを試みる。
    成功時は出力文字列を返す。失敗・エラー時は None を返す。
    """
    cmd = ["claude", "--model", model, "-p", prompt]
    if system:
        cmd += ["--append-system-prompt", system]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_CLI_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.warning("⏱️ claude CLI タイムアウト → APIフォールバック")
            return None

        output = stdout.decode("utf-8", errors="replace").strip()
        err_out = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            logger.warning("⚠️ claude CLI exit=%d → APIフォールバック (%s)",
                           proc.returncode, err_out[:100])
            return None

        # stdout にエラーメッセージが含まれていないか確認
        lower = output.lower()
        for pattern in _CLI_ERROR_PATTERNS:
            if pattern in lower:
                logger.warning("⚠️ claude CLI エラー応答検出 ('%s') → APIフォールバック", pattern)
                return None

        if not output:
            logger.warning("⚠️ claude CLI 空応答 → APIフォールバック")
            return None

        logger.info("✅ claude CLI 成功 (Proプラン使用, model=%s)", model)
        return output

    except FileNotFoundError:
        logger.debug("claude CLI 未インストール → APIフォールバック")
        return None
    except Exception as e:
        logger.warning("⚠️ claude CLI 例外: %s → APIフォールバック", e)
        return None


async def _call_anthropic_api(
    prompt: str,
    model: str,
    api_key: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """Anthropic API を直接呼び出す（有料フォールバック）"""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = await client.messages.create(**kwargs)
    logger.info("✅ Anthropic API 使用 (有料, model=%s)", model)
    return response.content[0].text


async def call_claude_with_fallback(
    prompt: str,
    model: str,
    api_key: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    skip_cli: bool = False,
) -> str:
    """
    Proプラン優先 → API フォールバック でClaudeを呼び出す。

    Args:
        prompt:     ユーザーメッセージ
        model:      "claude-sonnet-4-6" または "claude-opus-4-6"
        api_key:    Anthropic APIキー (フォールバック用)
        system:     システムプロンプト (省略可)
        max_tokens: 最大トークン数 (API呼び出し時のみ適用)
        skip_cli:   True にするとCLIをスキップしてAPI直接呼び出し

    Returns:
        モデルの応答テキスト
    """
    if not skip_cli:
        cli_result = await _try_claude_cli(prompt, model, system)
        if cli_result is not None:
            return cli_result

    # CLIが失敗 or スキップ → API
    return await _call_anthropic_api(prompt, model, api_key, system, max_tokens)
