"""Slack Integration - 10ボットによる会話劇

ボット一覧:
  1. shogun-bot  (本陣) - タスク受付・最終報告
  2. taisho-bot  (侍大将) - 推論・分析報告
  3-10. ashigaru-{1-8}-bot (足軽) - ツール実行報告

チャンネル構成:
  話題別: #合戦-{topic} (動的作成)
  家訓:   #家訓-{rule}  (永続的ルール)

起動方法:
  @shogun-bot       → 大隊モード
  @shogun-bot-light → 中隊モード (¥0)
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("shogun.slack")

try:
    from slack_sdk import WebClient
    from slack_sdk.socket_mode import SocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
    from slack_sdk.socket_mode.response import SocketModeResponse
    HAS_SLACK = True
except ImportError:
    HAS_SLACK = False


class SlackShogun:
    """Slack統合将軍システム - 10ボット会話劇."""

    # Bot name → env var for token
    BOT_TOKEN_VARS = {
        "shogun": "SLACK_TOKEN_SHOGUN",
        "taisho": "SLACK_TOKEN_TAISHO",
        "ashigaru-1": "SLACK_TOKEN_ASHIGARU_1",
        "ashigaru-2": "SLACK_TOKEN_ASHIGARU_2",
        "ashigaru-3": "SLACK_TOKEN_ASHIGARU_3",
        "ashigaru-4": "SLACK_TOKEN_ASHIGARU_4",
        "ashigaru-5": "SLACK_TOKEN_ASHIGARU_5",
        "ashigaru-6": "SLACK_TOKEN_ASHIGARU_6",
        "ashigaru-7": "SLACK_TOKEN_ASHIGARU_7",
        "ashigaru-8": "SLACK_TOKEN_ASHIGARU_8",
    }

    # MCP server name → ashigaru id (for drama)
    MCP_ASHIGARU = {
        "filesystem": 1,
        "github": 2,
        "fetch": 3,
        "memory": 4,
        "postgres": 5,
        "puppeteer": 6,
        "brave-search": 7,
        "slack": 8,
    }

    # Ashigaru specialty descriptions
    ASHIGARU_ROLE = {
        1: "ファイル操作",
        2: "Git/GitHub",
        3: "Web情報取得",
        4: "長期記憶",
        5: "データベース",
        6: "ブラウザ自動化",
        7: "Web検索",
        8: "チーム連携",
    }

    def __init__(self, controller: Any = None):
        if not HAS_SLACK:
            raise RuntimeError("pip install slack-sdk required")

        self.controller = controller
        self.clients: dict[str, WebClient] = {}

        # Initialize bot clients
        for name, env_var in self.BOT_TOKEN_VARS.items():
            token = os.environ.get(env_var, "")
            if token:
                self.clients[name] = WebClient(token=token)
            else:
                logger.warning("Slack token not set: %s", env_var)

        # Socket mode client (uses shogun bot)
        app_token = os.environ.get("SLACK_APP_TOKEN", "")
        if app_token and "shogun" in self.clients:
            self.socket_client = SocketModeClient(
                app_token=app_token,
                web_client=self.clients["shogun"],
            )
        else:
            self.socket_client = None

    def start(self) -> None:
        """Start Slack bot listener."""
        if not self.socket_client:
            logger.error("Socket client not available")
            return

        self.socket_client.socket_mode_request_listeners.append(
            self._process_event
        )
        self.socket_client.connect()
        logger.info("[本陣] Slack将軍システム起動完了")

    def _process_event(
        self, client: Any, req: Any
    ) -> None:
        """Process Slack event."""
        if req.type == "events_api":
            event = req.payload.get("event", {})
            if event.get("type") == "app_mention":
                asyncio.get_event_loop().create_task(
                    self._handle_mention(event)
                )

        response = SocketModeResponse(envelope_id=req.envelope_id)
        client.send_socket_mode_response(response)

    async def _handle_mention(self, event: dict) -> None:
        """Handle @mention event."""
        text = event.get("text", "")
        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts", event.get("ts", ""))

        # Determine mode: @shogun-bot-light → 中隊, @shogun-bot → 大隊
        if "shogun-bot-light" in text.lower() or "light" in text.lower():
            mode = "company"
            task = self._clean_mention(text)
            self._post_as("shogun", channel,
                "📋 任務受領\n編成: 中隊モード（¥0）", thread_ts)
        else:
            mode = "battalion"
            task = self._clean_mention(text)
            self._post_as("shogun", channel,
                "📋 任務受領\n編成: 大隊モード", thread_ts)

        # Process with drama
        await self._process_with_drama(task, channel, thread_ts, mode)

    async def _process_with_drama(
        self, task: str, channel: str, thread_ts: str, mode: str,
    ) -> None:
        """Process task with Slack conversation drama."""

        if mode == "company":
            # 中隊モード: 侍大将 + 足軽
            self._post_as("taisho", channel,
                "⚔️ 侍大将、出陣！\n足軽たち、情報を集めよ！", thread_ts)

            # Simulate ashigaru reports
            self._post_as("ashigaru-1", channel,
                "📁 足軽1番（ファイル操作）、報告\nファイルシステム確認完了でござる", thread_ts)
            self._post_as("ashigaru-2", channel,
                "📝 足軽2番（Git/GitHub）、報告\nリポジトリ状況確認完了でござる", thread_ts)

            self._post_as("taisho", channel,
                "🧠 <think>で思考中...", thread_ts)

            # Actual processing
            if self.controller:
                result = await self.controller.process_task(task, mode="company")
            else:
                result = "(Controller未接続)"

            self._post_as("taisho", channel,
                f"⚔️ 侍大将の判断\n\n{result}", thread_ts)

            self._post_as("shogun", channel,
                "✅ 中隊任務完了（¥0）", thread_ts)

        else:
            # 大隊モード
            if self.controller:
                from shogun.core.complexity import estimate_complexity
                complexity = estimate_complexity(task)
            else:
                complexity = "unknown"

            self._post_as("shogun", channel,
                f"🎯 複雑度: {complexity}", thread_ts)

            if self.controller:
                result = await self.controller.process_task(task, mode="battalion")
            else:
                result = "(Controller未接続)"

            self._post_as("shogun", channel,
                f"✅ 大隊任務完了\n\n{result}", thread_ts)

    def _post_as(
        self, bot_name: str, channel: str, text: str,
        thread_ts: str | None = None,
    ) -> None:
        """Post message as specified bot."""
        client = self.clients.get(bot_name)
        if not client:
            logger.warning("Bot not available: %s", bot_name)
            # Fallback to shogun
            client = self.clients.get("shogun")
            if not client:
                return
            text = f"[{bot_name}] {text}"

        try:
            client.chat_postMessage(
                channel=channel,
                text=text,
                thread_ts=thread_ts,
            )
        except Exception as e:
            logger.error("Slack post error (%s): %s", bot_name, e)

    @staticmethod
    def _clean_mention(text: str) -> str:
        """Remove @mention tags from text."""
        import re
        return re.sub(r"<@[A-Z0-9]+>", "", text).strip()


def run_slack_bot(controller: Any = None) -> None:
    """Entry point for Slack bot."""
    if not HAS_SLACK:
        print("slack-sdk is required: pip install slack-sdk", file=sys.stderr)
        sys.exit(1)

    bot = SlackShogun(controller=controller)
    bot.start()

    # Keep alive
    import signal
    event = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_: event.set())
    signal.signal(signal.SIGTERM, lambda *_: event.set())
    asyncio.get_event_loop().run_until_complete(event.wait())
