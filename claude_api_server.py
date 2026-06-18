#!/usr/bin/env python3
"""
Claude API Server - Runs on claude-dedicated LXC (192.168.11.237)

Claude Pro CLI を優先実行し、失敗時に Anthropic API にフォールバック。
bushidan-honjin LXC からは HTTP API 経由でアクセス。

起動:
  python claude_api_server.py

環境変数:
  CLAUDE_API_PORT: API サーバーポート (デフォルト: 8070)
  ANTHROPIC_API_KEY: フォールバック用 API キー
"""

import asyncio
import logging
import os
import re
import subprocess
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

# ── ロギング ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("claude_api_server")

# ── Flask App ────────────────────────────────────────────────────────

app = Flask(__name__)
API_PORT = int(os.environ.get("CLAUDE_API_PORT", "8070"))
API_HOST = os.environ.get("CLAUDE_API_HOST", "127.0.0.1")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")


def _verify_api_key() -> bool:
    """X-API-Key ヘッダーを検証。CLAUDE_API_KEY 未設定時は拒否。"""
    import secrets as _secrets

    if not CLAUDE_API_KEY:
        logger.error(
            "🚨 CLAUDE_API_KEY unset — rejecting request. Set CLAUDE_API_KEY in .env to enable the API.",
        )
        return False
    key = request.headers.get("X-API-Key", "")
    return bool(key) and _secrets.compare_digest(key, CLAUDE_API_KEY)


class ClaudeClient:
    """Claude Pro CLI を優先、失敗時 Anthropic API にフォールバック"""

    def __init__(self):
        self.cli_path = "/home/claude/.local/bin/claude"  # Claude CLI パス
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    async def call(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> dict:
        """
        Claude Pro CLI を優先実行。失敗時 Anthropic API にフォールバック。

        Args:
            prompt: ユーザープロンプト
            model: モデル名 (unused for CLI, used for API)
            system: システムプロンプト
            max_tokens: 最大トークン数

        Returns:
            {
                "content": "応答テキスト",
                "model": "使用したモデル",
                "source": "cli" or "api",
                "error": null or エラーメッセージ
            }
        """
        # CLI で試行
        cli_result = await self._call_cli(prompt, system, max_tokens)
        if cli_result["success"]:
            logger.info("✅ Claude Pro CLI で応答")
            return {
                "content": cli_result["content"],
                "model": cli_result.get("model", "claude-pro-cli"),
                "source": "cli",
                "error": None,
            }

        logger.warning(
            f"⚠️  Claude CLI 失敗: {cli_result['error']} → API フォールバック"
        )

        # API でフォールバック
        api_result = await self._call_api(prompt, system, max_tokens, model=model)
        if api_result["success"]:
            logger.info("✅ Anthropic API で応答")
            return {
                "content": api_result["content"],
                "model": api_result.get("model", "claude-opus-4-6"),
                "source": "api",
                "error": None,
            }

        logger.error(
            f"❌ 両方失敗: CLI={cli_result['error']}, API={api_result['error']}"
        )
        return {
            "content": "",
            "model": None,
            "source": None,
            "error": f"CLI失敗: {cli_result['error']}, API失敗: {api_result['error']}",
        }

    async def _call_cli(
        self, prompt: str, system: Optional[str], max_tokens: int
    ) -> dict:
        """バインドマウントされたディレクトリからClaude Pro CLIで実行"""
        try:
            if not os.path.exists(self.cli_path):
                return {"success": False, "error": f"CLI not found at {self.cli_path}"}

            # バインドマウントされたディレクトリを指定
            mounted_dir = "/mnt/Bushidan-Multi-Agent"
            if not os.path.exists(mounted_dir):
                return {
                    "success": False,
                    "error": f"Mounted dir not found: {mounted_dir}",
                }

            # CLI コマンド構築 (-p の直後にプロンプト、--add-dir は後置)
            if system is not None and not isinstance(system, str):
                raise ValueError(
                    f"system must be str or None, got {type(system).__name__}"
                )
            cmd = [self.cli_path, "-p", prompt, "--add-dir", mounted_dir]
            if system:
                cmd.extend(["--append-system-prompt", system])

            logger.info(f"🔄 Claude CLI 実行 (バインドマウント {mounted_dir})")

            # 実行 (タイムアウト 60秒)
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # CLI warning lines (account/credit notices) を除去するが、
                # 正当なレスポンスを誤って除去しないよう具体的なパターンのみ対象とする
                _WARN_RE = re.compile(
                    r"^(balance remaining[:\s]|credit remaining[:\s]|"
                    r"usage limit (reached|reset)|your (account|claude) (balance|credit|plan|usage))",
                    re.IGNORECASE,
                )
                lines = [
                    line
                    for line in result.stdout.splitlines()
                    if not _WARN_RE.match(line.strip())
                ]
                content = "\n".join(lines).strip()
                if not content:
                    logger.warning("CLI returned empty content after filtering warnings.")
                    return {
                        "success": False,
                        "error": "Empty response from CLI after filtering",
                    }
                return {
                    "success": True,
                    "content": content,
                    "model": "claude-pro-cli",
                }
            else:
                error_msg = result.stderr[:200] or result.stdout[:200]
                logger.error(f"CLI error: {error_msg}")
                return {
                    "success": False,
                    "error": f"CLI returned {result.returncode}: {error_msg}",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "CLI timeout (60s)"}
        except Exception as e:
            logger.error(f"CLI exception: {e}")
            return {"success": False, "error": str(e)[:100]}

    async def _call_api(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        model: Optional[str] = None,
    ) -> dict:
        """AsyncAnthropic API でフォールバック (リクエストごとに新規クライアント生成)"""
        try:
            if not self.api_key:
                return {"success": False, "error": "ANTHROPIC_API_KEY not set"}

            import anthropic

            # Flask threaded=True + async view = リクエストごとに新しい event loop。
            # httpx.AsyncClient は生成 loop にバインドされるためシングルトン再利用は unsafe。
            async with anthropic.AsyncAnthropic(api_key=self.api_key) as client:
                response = await client.messages.create(
                    model=model or "claude-opus-4-6",
                    max_tokens=max_tokens,
                    system=system or "",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60.0,
                )
                return {
                    "success": True,
                    "content": response.content[0].text,
                    "model": response.model,
                }

        except Exception as e:
            return {"success": False, "error": str(e)[:100]}


claude = ClaudeClient()


# ── API エンドポイント ────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    """ヘルスチェック"""
    return jsonify({"status": "ok", "service": "claude-api-server"})


@app.route("/api/claude", methods=["POST"])
async def call_claude():
    """
    Claude を呼び出す

    Request:
    {
      "prompt": "ユーザープロンプト",
      "system": "システムプロンプト (optional)",
      "model": "モデル名 (optional, API用)",
      "max_tokens": 2000 (optional)
    }

    Response:
    {
      "content": "応答テキスト",
      "model": "使用したモデル",
      "source": "cli" or "api",
      "error": null or エラーメッセージ
    }
    """
    if not _verify_api_key():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400

        prompt = data["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"error": "'prompt' must be a non-empty string"}), 400
        system = data.get("system")
        model = data.get("model")
        max_tokens = data.get("max_tokens", 2000)
        if not isinstance(max_tokens, int) or not (1 <= max_tokens <= 8192):
            return (
                jsonify(
                    {"error": "'max_tokens' must be an integer between 1 and 8192"}
                ),
                400,
            )
        if system is not None:
            if not isinstance(system, str):
                return jsonify({"error": "'system' must be a string or null"}), 400
            if len(system) == 0 or len(system) > 8192:
                return jsonify({"error": "'system' must be between 1 and 8192 characters"}), 400

        result = await claude.call(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
        )

        if result["error"]:
            return jsonify(result), 503

        return jsonify(result), 200

    except Exception as e:
        logger.exception("API エラー: %s", e)
        return (
            jsonify({"error": "Internal server error", "content": "", "model": None, "source": None}),
            500,
        )


@app.route("/api/status", methods=["GET"])
def get_status():
    """サーバーステータス"""
    if not _verify_api_key():
        return jsonify({"error": "Unauthorized"}), 401
    cli_available = os.path.exists(claude.cli_path)
    api_available = bool(claude.api_key)

    return jsonify(
        {
            "status": "ok",
            "claude_cli_available": cli_available,
            "anthropic_api_available": api_available,
        }
    )


# ── メイン ────────────────────────────────────────────────────────────


def main():
    """サーバー起動"""
    logger.info("🚀 Claude API Server 起動")
    logger.info(f"   ポート: {API_PORT}")
    logger.info(f"   Claude CLI: {claude.cli_path}")
    logger.info(f"   Anthropic API: {'✅' if claude.api_key else '❌'}")

    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
