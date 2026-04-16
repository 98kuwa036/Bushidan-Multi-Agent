"""
Claude CLI クライアント — リモート API優先 → ローカル CLI → Anthropic API → Gemini/Bedrock フォールバック

フォールバック優先順位:
  1. リモート Claude API Server (pct237:8070)
  2. ローカル Claude CLI
  3. Anthropic API 直接
  4. Anthropic インシデント時 → Gemini 3.1 Pro (デフォルト) または AWS Bedrock (オプション)
  5. 最終フォールバック Bedrock (Gemini 失敗時)

インシデント検知エラーコード: 529 (overloaded), 503 (unavailable), 500 (internal), 401/403 (auth)

フォールバックモード切り替え:
    from utils.claude_cli_client import set_incident_fallback_mode
    set_incident_fallback_mode("gemini")   # デフォルト: Gemini 3.1 Pro
    set_incident_fallback_mode("bedrock")  # オプション: AWS Bedrock
"""

import asyncio
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

# リモート Claude API Server の設定（オプション）
REMOTE_CLAUDE_API_URL = os.environ.get("CLAUDE_API_SERVER_URL")

# AWS Bedrock 設定
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# モデル ID マッピング（Claude API モデル名 → Bedrock モデル ID）
_BEDROCK_MODEL_MAP = {
    "claude-opus-4-6": "anthropic.claude-opus-4-1-20250805-v1:0",
    "claude-opus-4-5": "anthropic.claude-opus-4-1-20250805-v1:0",
    "claude-sonnet-4-6": "anthropic.claude-sonnet-4-6-20250514-v1:0",
    "claude-sonnet-4-5": "anthropic.claude-sonnet-4-6-20250514-v1:0",
    "claude-haiku-4-5-20251001": "anthropic.claude-3-5-haiku-20241022-v2:0",
}

# ── Anthropic インシデント系エラーコード ────────────────────────────────────
# これらのコードが返ったとき Gemini / Bedrock へ動的切り替えする
_ANTHROPIC_INCIDENT_CODES = {529, 503, 500}   # overloaded / unavailable / internal
_ANTHROPIC_AUTH_CODES     = {401, 403}         # auth / permission issues


class AnthropicIncidentError(Exception):
    """Anthropic API がインシデント状態のとき発生 (529/503/500/401/403)"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Anthropic API incident (HTTP {status_code}): {message}")


# ── フォールバックモード (実行時に変更可能) ──────────────────────────────────
# "gemini"  : Gemini 3.1 Pro へ切り替え（デフォルト）
# "bedrock" : AWS Bedrock Claude へ切り替え
_incident_fallback_mode: str = os.environ.get("CLAUDE_INCIDENT_FALLBACK", "gemini").lower()


def set_incident_fallback_mode(mode: str) -> None:
    """インシデント時フォールバックモードを実行時に変更する。
    mode: "gemini" (デフォルト) または "bedrock"
    """
    global _incident_fallback_mode
    _incident_fallback_mode = mode.lower()
    logger.info("🔧 Claudeインシデントフォールバックモード変更: %s", mode)


def get_incident_fallback_mode() -> str:
    """現在のフォールバックモードを返す。"""
    return _incident_fallback_mode


# ── CLI出力でエラーと判断するキーワード
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

        async with httpx.AsyncClient(timeout=180.0) as client:
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
    if not api_key:
        raise RuntimeError("Anthropic APIキー未設定 — Claude Pro CLIをご確認ください")
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    # Prompt Caching: システムプロンプトを ephemeral キャッシュ対象に設定
    # プレフィックスが同一であれば入力トークンコストが最大 90% 削減される
    if system:
        kwargs["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

    try:
        response = await client.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        cache_hit = getattr(usage, "cache_read_input_tokens", 0) if usage else 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) if usage else 0
        if cache_hit:
            logger.info("✅ Anthropic API (model=%s) キャッシュヒット %d tokens", model, cache_hit)
        elif cache_write:
            logger.info("✅ Anthropic API (model=%s) キャッシュ書き込み %d tokens", model, cache_write)
        else:
            logger.info("✅ Anthropic API 使用 (model=%s)", model)
        return response.content[0].text
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower() or "too low" in str(e).lower():
            raise RuntimeError("Claude APIクレジット不足 — Pro CLIが利用できない場合は応答できません") from e
        raise
    except anthropic.APIStatusError as e:
        if e.status_code in _ANTHROPIC_INCIDENT_CODES or e.status_code in _ANTHROPIC_AUTH_CODES:
            raise AnthropicIncidentError(e.status_code, str(e)) from e
        raise


async def _call_anthropic_api_with_messages(
    messages: list,
    model: str,
    api_key: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """Anthropic API を messages 配列で呼び出す（会話履歴対応）"""
    if not api_key:
        raise RuntimeError("Anthropic APIキー未設定 — Claude Pro CLIをご確認ください")
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    # Prompt Caching: システムプロンプトを ephemeral キャッシュ対象に設定
    if system:
        kwargs["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

    try:
        response = await client.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        cache_hit = getattr(usage, "cache_read_input_tokens", 0) if usage else 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) if usage else 0
        if cache_hit:
            logger.info("✅ Anthropic API (model=%s, %d件) キャッシュヒット %d tokens",
                        model, len(messages), cache_hit)
        else:
            logger.info("✅ Anthropic API 使用 (会話履歴 %d件, model=%s, cache_write=%d)",
                        len(messages), model, cache_write)
        return response.content[0].text
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower() or "too low" in str(e).lower():
            raise RuntimeError("Claude APIクレジット不足 — AWS Bedrock へフォールバック") from e
        raise
    except anthropic.APIStatusError as e:
        if e.status_code in _ANTHROPIC_INCIDENT_CODES or e.status_code in _ANTHROPIC_AUTH_CODES:
            raise AnthropicIncidentError(e.status_code, str(e)) from e
        raise


async def _call_gemini_fallback(
    messages: list,
    system: Optional[str] = None,
    max_tokens: int = 4096,
) -> Optional[str]:
    """Gemini 3.1 Pro でClaudeインシデント時のフォールバック（Sonnet同等スペック想定）"""
    api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    if not api_key or len(api_key) < 5:
        logger.warning("⚠️ GEMINI_API_KEY 未設定 — Gemini フォールバック不可")
        return None
    try:
        from utils.gemini3_client import Gemini3Client
        client = Gemini3Client(api_key=api_key, model="gemini-3.1-pro-preview")
        all_msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        result = await client.generate(messages=all_msgs, max_output_tokens=max_tokens)
        logger.info("✅ Gemini 3.1 Pro フォールバック成功 (Claudeインシデント対応)")
        return result
    except Exception as e:
        logger.warning("⚠️ Gemini 3.1 Pro フォールバック失敗: %s", str(e)[:100])
        return None


async def _call_bedrock_api(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
) -> Optional[str]:
    """AWS Bedrock API で Claude を呼び出す（フォールバック用）"""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        logger.warning("⚠️ AWS Bedrock 認証情報未設定 — Bedrock フォールバック不可")
        return None

    try:
        import boto3

        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        # モデル ID 変換
        bedrock_model_id = _BEDROCK_MODEL_MAP.get(model)
        if not bedrock_model_id:
            logger.warning("⚠️ Bedrock モデル ID マッピングなし: %s", model)
            return None

        messages = [{"role": "user", "content": prompt}]
        system_blocks = [{"text": system}] if system else None

        # Bedrock Converse API 呼び出し (v16: thinking mode disabled by default)
        response = await asyncio.to_thread(
            bedrock_client.converse,
            modelId=bedrock_model_id,
            messages=messages,
            system=system_blocks,
            inferenceConfig={"maxTokens": max_tokens},
        )

        if response and "output" in response:
            output_text = response["output"]["message"]["content"][0]["text"]
            logger.info("✅ AWS Bedrock 使用 (フォールバック, model=%s)", model)
            return output_text
        return None

    except Exception as e:
        logger.warning("⚠️ AWS Bedrock エラー → 全フォールバック失敗: %s", str(e)[:100])
        return None


async def _call_bedrock_api_with_messages(
    messages: list,
    model: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
) -> Optional[str]:
    """AWS Bedrock API で Claude を呼び出す（会話履歴対応、フォールバック用）"""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        logger.warning("⚠️ AWS Bedrock 認証情報未設定 — Bedrock フォールバック不可")
        return None

    try:
        import boto3

        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        # モデル ID 変換
        bedrock_model_id = _BEDROCK_MODEL_MAP.get(model)
        if not bedrock_model_id:
            logger.warning("⚠️ Bedrock モデル ID マッピングなし: %s", model)
            return None

        system_blocks = [{"text": system}] if system else None

        # Bedrock Converse API 呼び出し (v16: thinking mode disabled by default)
        response = await asyncio.to_thread(
            bedrock_client.converse,
            modelId=bedrock_model_id,
            messages=messages,
            system=system_blocks,
            inferenceConfig={"maxTokens": max_tokens},
        )

        if response and "output" in response:
            output_text = response["output"]["message"]["content"][0]["text"]
            logger.info("✅ AWS Bedrock 使用 (会話履歴 %d件, フォールバック, model=%s)", len(messages), model)
            return output_text
        return None

    except Exception as e:
        logger.warning("⚠️ AWS Bedrock エラー → 全フォールバック失敗: %s", str(e)[:100])
        return None


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

    # 3. Anthropic API フォールバック
    try:
        return await _call_anthropic_api_with_messages(messages, model, api_key, system, max_tokens)
    except AnthropicIncidentError as e:
        logger.warning("🚨 Anthropic インシデント検知 (HTTP %d) → %s フォールバック",
                       e.status_code, _incident_fallback_mode)
        if _incident_fallback_mode == "bedrock":
            bedrock_result = await _call_bedrock_api_with_messages(messages, model, system, max_tokens)
            if bedrock_result is not None:
                return bedrock_result
        else:
            gemini_result = await _call_gemini_fallback(messages, system, max_tokens)
            if gemini_result is not None:
                return gemini_result
            # Gemini 失敗時は Bedrock へ
            logger.info("📌 Gemini フォールバック失敗 → AWS Bedrock 最終フォールバック")
            bedrock_result = await _call_bedrock_api_with_messages(messages, model, system, max_tokens)
            if bedrock_result is not None:
                return bedrock_result
    except RuntimeError as e:
        if "credit" in str(e).lower():
            logger.warning("⚠️ Anthropic API クレジット不足 → AWS Bedrock フォールバック")
        else:
            logger.warning("⚠️ Anthropic API エラー: %s → AWS Bedrock フォールバック", str(e)[:100])
        bedrock_result = await _call_bedrock_api_with_messages(messages, model, system, max_tokens)
        if bedrock_result is not None:
            return bedrock_result

    # 全フォールバック失敗
    raise RuntimeError("すべてのバックエンド失敗 (API・CLI・Anthropic・Gemini・Bedrock)")


async def call_claude_with_fallback(
    prompt: str,
    model: str,
    api_key: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    skip_cli: bool = False,
) -> str:
    """
    リモート API優先 → ローカル CLI → Anthropic API → AWS Bedrock フォールバック

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
        4. AWS Bedrock (最終フォールバック)
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
    messages_for_fallback = [{"role": "user", "content": prompt}]
    try:
        return await _call_anthropic_api(prompt, model, api_key, system, max_tokens)
    except AnthropicIncidentError as e:
        logger.warning("🚨 Anthropic インシデント検知 (HTTP %d) → %s フォールバック",
                       e.status_code, _incident_fallback_mode)
        if _incident_fallback_mode == "bedrock":
            bedrock_result = await _call_bedrock_api(prompt, model, system, max_tokens)
            if bedrock_result is not None:
                return bedrock_result
        else:
            gemini_result = await _call_gemini_fallback(messages_for_fallback, system, max_tokens)
            if gemini_result is not None:
                return gemini_result
            # Gemini 失敗時は Bedrock へ
            logger.info("📌 Gemini フォールバック失敗 → AWS Bedrock 最終フォールバック")
            bedrock_result = await _call_bedrock_api(prompt, model, system, max_tokens)
            if bedrock_result is not None:
                return bedrock_result
    except RuntimeError as e:
        if "credit" in str(e).lower():
            logger.warning("⚠️ Anthropic API クレジット不足 → AWS Bedrock フォールバック")
        else:
            logger.warning("⚠️ Anthropic API エラー: %s → AWS Bedrock フォールバック", str(e)[:100])
        bedrock_result = await _call_bedrock_api(prompt, model, system, max_tokens)
        if bedrock_result is not None:
            return bedrock_result

    # 全フォールバック失敗
    raise RuntimeError("すべてのバックエンド失敗 (API・CLI・Anthropic・Gemini・Bedrock)")
