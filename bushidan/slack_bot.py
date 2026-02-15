"""武士団 Slack Bot - Socket Mode (単一Bot Token対応)

Socket Mode を使用するため公開URLは不要です。
必要な環境変数:
  SLACK_BOT_TOKEN    - xoxb-... (Bot User OAuth Token)
  SLACK_APP_TOKEN    - xapp-... (App-Level Token, Socket Mode用)

起動方法:
  python -m bushidan.slack_bot
"""

import asyncio
import logging
import os
import re
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bushidan.slack")

try:
    from slack_sdk import WebClient
    from slack_sdk.socket_mode import SocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
    from slack_sdk.socket_mode.response import SocketModeResponse
    HAS_SLACK = True
except ImportError:
    HAS_SLACK = False


def _clean_mention(text: str) -> str:
    """@メンションタグを除去してタスク文字列を返す."""
    return re.sub(r"<@[A-Z0-9]+>", "", text).strip()


class BushidanSlackBot:
    """武士団 Slack Bot - Socket Mode."""

    def __init__(self) -> None:
        if not HAS_SLACK:
            raise RuntimeError(
                "slack-sdk が必要です: pip install slack-sdk"
            )

        bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
        app_token = os.environ.get("SLACK_APP_TOKEN", "")

        if not bot_token:
            raise RuntimeError(
                "SLACK_BOT_TOKEN が設定されていません (.env を確認してください)"
            )
        if not app_token:
            raise RuntimeError(
                "SLACK_APP_TOKEN が設定されていません。\n"
                "Slack App 設定 → Socket Mode → Enable → App-Level Token を生成してください。\n"
                "トークンは xapp-1-... で始まります。"
            )

        self.web_client = WebClient(token=bot_token)
        self.socket_client = SocketModeClient(
            app_token=app_token,
            web_client=self.web_client,
        )
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Slack Bot を起動する."""
        self._loop = asyncio.new_event_loop()

        self.socket_client.socket_mode_request_listeners.append(
            self._on_event
        )
        self.socket_client.connect()
        logger.info("🏯 武士団 Slack Bot 起動完了 (Socket Mode)")
        logger.info("チャンネルに @Bushidan と入力して呼び出してください")

    def _on_event(self, client: Any, req: SocketModeRequest) -> None:
        """Slack イベントを受信."""
        # 即座にACK応答
        client.send_socket_mode_response(
            SocketModeResponse(envelope_id=req.envelope_id)
        )

        if req.type != "events_api":
            return

        event = req.payload.get("event", {})
        event_type = event.get("type", "")

        if event_type == "app_mention":
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._handle_mention(event), self._loop
                )

    async def _handle_mention(self, event: dict) -> None:
        """@メンションを処理する."""
        text = event.get("text", "")
        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts", event.get("ts", ""))

        task = _clean_mention(text)
        if not task:
            self._post(channel, "はい、何かご用でしょうか？", thread_ts)
            return

        logger.info("タスク受信: %s", task)
        self._post(channel, f"📋 任務受領: {task}\n処理中...", thread_ts)

        try:
            result = await self._process(task)
            self._post(channel, f"✅ 完了\n\n{result}", thread_ts)
        except Exception as e:
            logger.exception("タスク処理エラー")
            self._post(channel, f"❌ エラーが発生しました: {e}", thread_ts)

    async def _process(self, task: str) -> str:
        """タスクを処理する (武士団ルーティング)."""
        # TODO: core.system_orchestrator と統合
        # 現時点では簡易応答
        return (
            f"武士団システムに接続しました。\n"
            f"タスク: {task}\n\n"
            f"（注: orchestratorとの統合が必要です。"
            f"core/system_orchestrator.py を確認してください）"
        )

    def _post(
        self, channel: str, text: str, thread_ts: str | None = None
    ) -> None:
        """Slack にメッセージを投稿."""
        try:
            kwargs: dict = {"channel": channel, "text": text}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            self.web_client.chat_postMessage(**kwargs)
        except Exception as e:
            logger.error("Slack 投稿エラー: %s", e)


def main() -> None:
    """エントリーポイント."""
    if not HAS_SLACK:
        print(
            "エラー: slack-sdk が必要です\n"
            "インストール: pip install slack-sdk",
            file=sys.stderr,
        )
        sys.exit(1)

    bot = BushidanSlackBot()
    bot.start()

    # シグナル待機
    import signal
    stop_event = asyncio.Event()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bot._loop = loop

    def _stop(*_: Any) -> None:
        logger.info("終了シグナル受信...")
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        loop.run_until_complete(stop_event.wait())
    finally:
        logger.info("武士団 Slack Bot を停止しました")


if __name__ == "__main__":
    main()
