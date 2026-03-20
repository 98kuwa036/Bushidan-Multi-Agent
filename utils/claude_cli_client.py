"""
Claude CLI クライアント — リモート API優先 → ローカル CLI → API フォールバック

Proプラン優先 → API フォールバック でClaudeを呼び出す。
リモート Claude API Server が利用可能な場合はそれを優先利用。

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
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

# リモート Claude API Server の設定（オプション）
REMOTE_CLAUDE_API_URL = os.environ.get("CLAUDE_API_SERVER_URL")

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
# LangGraphノードタイムアウト (shogun:200s, daigensui:280s) より短く設定し、
# CLIタイムアウト後にAPIフォールバックが発動できる余裕を確保する
_CLI_TIMEOUT = 60

# ネイティブインストールの claude バイナリパス
_CLAUDE_BIN = shutil.which("claude") or "claude"


# ── リモート Claude API Server 関数 ────────────────────────────────

async def _try_remote_claude_api(
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    max_tokens: int = 4096,
) -> Optional[str]:
    """
    リモート Claude API Server (claude-dedicated LXC) で実行を試みる。
    成功時は応答文字列を返す。失敗時は None を返す。
    """
    if not REMOTE_CLAUDE_API_URL:
        return None

    try:
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{REMOTE_CLAUDE_API_URL}/api/claude",
                json={
                    "prompt": prompt,
                    "system": system,
                    "model": model,
                    "max_tokens": max_tokens,
                },
            )

            if response.status_code == 200:
                data = response.json()
                if not data.get("error"):
                    logger.info(
                        "✅ リモート Claude API 成功 "
                        "(source=%s, model=%s)",
                        data.get("source"),
                        data.get("model"),
                    )
                    return data["content"]
                else:
                    logger.warning(
                        "⚠️ リモート Claude API エラー: %s → フォールバック",
                        data["error"][:100],
                    )
                    return None
            else:
                logger.warning(
                    "⚠️ リモート Claude API HTTP %d → フォールバック",
                    response.status_code,
                )
                return None

    except Exception as e:
        logger.warning("⚠️ リモート Claude API 接続失敗: %s → フォールバック", e)
        return None


def _format_messages_to_prompt(messages: list) -> str:
    """会話履歴を単一プロンプトに変換する（CLI用）"""
    if not messages:
        return ""

    lines = []
    for msg in messages[:-1]:  # 最後のメッセージ以外は履歴として含める
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = "[ユーザー]" if role == "user" else "[アシスタント]"
        lines.append(f"{prefix}\n{content}")

    # 最後のメッセージをメインプロンプトとして
    last_msg = messages[-1]
    lines.append(f"\n[現在の質問]\n{last_msg.get('content', '')}")

    return "\n".join(lines)


async def _try_claude_cli(prompt: str, model: str, system: Optional[str] = None) -> Optional[str]:
    """
    claude CLI でリクエストを試みる。
    成功時は出力文字列を返す。失敗・エラー時は None を返す。
    ネイティブインストール対応: --no-session-persistence でセッションファイル生成を抑制
    """
    cmd = [_CLAUDE_BIN, "--model", model, "--no-session-persistence", "-p", prompt]
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


async def _call_anthropic_api_with_messages(
    messages: list,
    model: str,
    api_key: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """Anthropic API を messages 配列で呼び出す（会話履歴対応）"""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    response = await client.messages.create(**kwargs)
    logger.info("✅ Anthropic API 使用 (会話履歴 %d件, model=%s)", len(messages), model)
    return response.content[0].text


async def call_claude_with_history(
    messages: list,
    model: str,
    api_key: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    prefer_cli: bool = True,
) -> str:
    """
    会話履歴付きで Claude を呼び出す。
    リモート API優先 → CLI → API フォールバック

    Args:
        messages: [{"role": "user"/"assistant", "content": "..."}]
        model:    "claude-sonnet-4-6" など
        api_key:  Anthropic APIキー
        system:   システムプロンプト (省略可)
        max_tokens: 最大トークン数
        prefer_cli: True (デフォルト) で CLI 優先、False で API 直接
    """
    # 会話履歴をプロンプトに変換
    prompt = _format_messages_to_prompt(messages)

    # 1. リモート Claude API Server を試す
    remote_result = await _try_remote_claude_api(prompt, model, system, max_tokens)
    if remote_result is not None:
        return remote_result

    # 2. CLI優先の場合: ローカル CLI を試す
    if prefer_cli:
        cli_result = await _try_claude_cli(prompt, model, system)
        if cli_result is not None:
            return cli_result
        logger.info("📌 リモートAPI・CLI失敗 → API フォールバック (会話履歴 %d件)", len(messages))

    # 3. API フォールバック
    return await _call_anthropic_api_with_messages(messages, model, api_key, system, max_tokens)


async def call_claude_with_fallback(
    prompt: str,
    model: str,
    api_key: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    skip_cli: bool = False,
) -> str:
    """
    リモート API優先 → ローカル CLI → Anthropic API フォールバック

    Args:
        prompt:     ユーザーメッセージ
        model:      "claude-sonnet-4-6" または "claude-opus-4-6"
        api_key:    Anthropic APIキー (フォールバック用)
        system:     システムプロンプト (省略可)
        max_tokens: 最大トークン数 (API呼び出し時のみ適用)
        skip_cli:   True にするとローカルCLIをスキップ

    Returns:
        モデルの応答テキスト

    優先順位:
        1. リモート Claude API Server (利用可能な場合)
        2. ローカル Claude Pro CLI (skip_cli=False かつ利用可能な場合)
        3. Anthropic API (有料フォールバック)
    """
    # 1. リモート Claude API Server を試す
    remote_result = await _try_remote_claude_api(prompt, model, system, max_tokens)
    if remote_result is not None:
        return remote_result

    # 2. ローカル Claude CLI を試す
    if not skip_cli:
        cli_result = await _try_claude_cli(prompt, model, system)
        if cli_result is not None:
            return cli_result

    # 3. Anthropic API フォールバック
    logger.info("📌 リモートAPI・CLI失敗 → Anthropic API フォールバック")
    return await _call_anthropic_api(prompt, model, api_key, system, max_tokens)
