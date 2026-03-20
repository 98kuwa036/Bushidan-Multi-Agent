"""
Claude API クライアント - claude-dedicated LXC へのリモート呼び出し

bushidan-honjin LXC から claude-dedicated LXC (192.168.11.237) の
Claude API Server に HTTP 経由でアクセス。

利用例:
    from utils.claude_api_client import call_claude_api
    response = await call_claude_api(
        prompt="Hello, Claude!",
        system="You are a helpful assistant."
    )
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("claude_api_client")

# Claude API Server のエンドポイント
CLAUDE_API_SERVER_URL = os.environ.get(
    "CLAUDE_API_SERVER_URL",
    "http://192.168.11.237:8070"
)
CLAUDE_API_TIMEOUT = float(os.environ.get("CLAUDE_API_TIMEOUT", "60"))


async def call_claude_api(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 2000,
) -> str:
    """
    Claude API Server 経由で Claude を呼び出す。

    Args:
        prompt: ユーザープロンプト
        system: システムプロンプト
        model: モデル名 (API用、CLI優先の場合は無視される可能性)
        max_tokens: 最大トークン数

    Returns:
        Claude からの応答テキスト

    Raises:
        Exception: API 呼び出し失敗時
    """
    try:
        async with httpx.AsyncClient(timeout=CLAUDE_API_TIMEOUT) as client:
            response = await client.post(
                f"{CLAUDE_API_SERVER_URL}/api/claude",
                json={
                    "prompt": prompt,
                    "system": system,
                    "model": model,
                    "max_tokens": max_tokens,
                },
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(
                    f"✅ Claude API 応答 (source={data['source']}, "
                    f"model={data['model']})"
                )
                return data["content"]
            else:
                error_data = response.json()
                error_msg = error_data.get("error", f"HTTP {response.status_code}")
                logger.error(f"❌ Claude API エラー: {error_msg}")
                raise Exception(f"Claude API error: {error_msg}")

    except httpx.TimeoutException:
        logger.error(f"⏱️ Claude API タイムアウト ({CLAUDE_API_TIMEOUT}s)")
        raise
    except httpx.ConnectError as e:
        logger.error(f"❌ Claude API サーバーに接続できません ({CLAUDE_API_SERVER_URL}): {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Claude API 呼び出し失敗: {e}")
        raise


async def get_claude_status() -> dict:
    """Claude API Server のステータスを取得"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{CLAUDE_API_SERVER_URL}/api/status")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
