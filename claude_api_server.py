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
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request

# ── ロギング ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("claude_api_server")

# ── Flask App ────────────────────────────────────────────────────────

app = Flask(__name__)
API_PORT = int(os.environ.get("CLAUDE_API_PORT", "8070"))


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

        logger.warning(f"⚠️  Claude CLI 失敗: {cli_result['error']} → API フォールバック")

        # API でフォールバック
        api_result = await self._call_api(prompt, system, max_tokens)
        if api_result["success"]:
            logger.info("✅ Anthropic API で応答")
            return {
                "content": api_result["content"],
                "model": api_result.get("model", "claude-opus-4-6"),
                "source": "api",
                "error": None,
            }

        logger.error(f"❌ 両方失敗: CLI={cli_result['error']}, API={api_result['error']}")
        return {
            "content": "",
            "model": None,
            "source": None,
            "error": f"CLI失敗: {cli_result['error']}, API失敗: {api_result['error']}",
        }

    async def _call_cli(
        self, prompt: str, system: Optional[str], max_tokens: int
    ) -> dict:
        """Claude Pro CLI で実行"""
        try:
            if not os.path.exists(self.cli_path):
                return {"success": False, "error": f"CLI not found at {self.cli_path}"}

            # CLI コマンド構築
            cmd = [self.cli_path]
            if system:
                cmd.extend(["-s", system])
            cmd.append(prompt)

            # 実行 (タイムアウト 30秒)
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "content": result.stdout.strip(),
                    "model": "claude-pro-cli",
                }
            else:
                return {
                    "success": False,
                    "error": f"CLI returned {result.returncode}: {result.stderr[:100]}",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "CLI timeout (30s)"}
        except Exception as e:
            return {"success": False, "error": str(e)[:100]}

    async def _call_api(
        self, prompt: str, system: Optional[str], max_tokens: int
    ) -> dict:
        """Anthropic API でフォールバック"""
        try:
            if not self.api_key:
                return {"success": False, "error": "ANTHROPIC_API_KEY not set"}

            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
                timeout=30.0,
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
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400

        prompt = data["prompt"]
        system = data.get("system")
        model = data.get("model")
        max_tokens = data.get("max_tokens", 2000)

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
        logger.exception("API エラー")
        return jsonify({"error": str(e), "content": "", "model": None, "source": None}), 500


@app.route("/api/status", methods=["GET"])
def get_status():
    """サーバーステータス"""
    cli_available = os.path.exists(claude.cli_path)
    api_available = bool(claude.api_key)

    return jsonify(
        {
            "status": "ok",
            "claude_cli_available": cli_available,
            "anthropic_api_available": api_available,
            "cli_path": claude.cli_path,
        }
    )


# ── メイン ────────────────────────────────────────────────────────────

def main():
    """サーバー起動"""
    logger.info(
        "🚀 Claude API Server 起動"
    )
    logger.info(f"   ポート: {API_PORT}")
    logger.info(f"   Claude CLI: {claude.cli_path}")
    logger.info(f"   Anthropic API: {'✅' if claude.api_key else '❌'}")

    app.run(
        host="0.0.0.0",
        port=API_PORT,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
