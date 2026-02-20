"""武士団 Discord Bot

公開URL不要。Bot Token 1つで動作します。

必要な環境変数:
  DISCORD_BOT_TOKEN  - Discord Developer Portal で取得

起動方法:
  python -m bushidan.discord_bot

セットアップ手順:
  1. https://discord.com/developers/applications → New Application
  2. Bot → Add Bot → Reset Token → コピー
  3. Bot → Privileged Gateway Intents → MESSAGE CONTENT INTENT を ON
  4. OAuth2 → URL Generator → scopes: bot → permissions: Send Messages
  5. 生成URLでサーバーに招待
  6. .env に DISCORD_BOT_TOKEN=... を追加
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# .env ファイルを読み込む
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bushidan.discord")

try:
    import discord
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False


def _clean_mention(text: str) -> str:
    """@メンションタグを除去してタスク文字列を返す."""
    return re.sub(r"<@!?\d+>", "", text).strip()


class BushidanDiscordBot(discord.Client):
    """武士団 Discord Bot."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # MESSAGE CONTENT INTENT 必須
        super().__init__(intents=intents)

    async def on_ready(self) -> None:
        """Bot 起動完了."""
        logger.info("🏯 武士団 Discord Bot 起動完了: %s (ID: %s)", self.user, self.user.id)
        logger.info("サーバー内で @%s とメンションして呼び出してください", self.user.name)
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="命令をお待ちしています"
            )
        )

    async def on_message(self, message: discord.Message) -> None:
        """メッセージ受信."""
        # 自分自身のメッセージは無視
        if message.author == self.user:
            return

        # メンションされた場合のみ応答
        if self.user not in message.mentions:
            return

        task = _clean_mention(message.content)
        if not task:
            await message.reply("はい、何かご用でしょうか？")
            return

        logger.info("タスク受信 from %s: %s", message.author.display_name, task)

        # 処理中メッセージ
        processing_msg = await message.reply(f"📋 任務受領: {task}\n処理中...")

        try:
            result = await self._process(task)
            await processing_msg.edit(content=f"✅ 完了\n\n{result}")
        except Exception as e:
            logger.exception("タスク処理エラー")
            await processing_msg.edit(content=f"❌ エラーが発生しました: {e}")

    async def _process(self, task: str) -> str:
        """タスクを処理する (武士団ルーティング).

        TODO: core.system_orchestrator と統合する
        """
        return (
            f"武士団システムに接続しました。\n"
            f"タスク: {task}\n\n"
            f"（core/system_orchestrator.py との統合が必要です）"
        )


def main() -> None:
    """エントリーポイント."""
    if not HAS_DISCORD:
        print(
            "エラー: discord.py が必要です\n"
            "インストール: pip install discord.py",
            file=sys.stderr,
        )
        sys.exit(1)

    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        print(
            "エラー: DISCORD_BOT_TOKEN が設定されていません\n"
            ".env に DISCORD_BOT_TOKEN=... を追加してください\n\n"
            "取得方法:\n"
            "  1. https://discord.com/developers/applications\n"
            "  2. New Application → Bot → Reset Token",
            file=sys.stderr,
        )
        sys.exit(1)

    bot = BushidanDiscordBot()
    logger.info("武士団 Discord Bot を起動します...")
    bot.run(token, log_handler=None)  # log_handler=None で独自ロガーを使用


if __name__ == "__main__":
    main()
