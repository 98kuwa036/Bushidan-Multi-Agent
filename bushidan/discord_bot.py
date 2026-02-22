"""武士団 Discord Bot - 完全統合版

core/system_orchestrator.py を経由して5層階層に委譲します。

  Discord メッセージ
       ↓
  BushidanDiscordBot.on_message()
       ↓
  SystemOrchestrator (setup_hook で初期化済み)
       ↓
  Shogun.process_task()  ← BDI + インテリジェントルーター
       ↓
  SIMPLE  → Groq (即応)
  MEDIUM  → 家老 → 大将 (4層フォールバック)
  COMPLEX → 軍師 → 家老 → 大将
  STRATEGIC → 将軍自ら処理
       ↓
  Discord に返答

必要な環境変数:
  DISCORD_BOT_TOKEN  - Discord Developer Portal で取得
  CLAUDE_API_KEY     - Anthropic (将軍層 必須)
  GEMINI_API_KEY     - Google    (最終防衛線 必須)

起動方法:
  python -m bushidan.discord_bot
"""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

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

DISCORD_MAX_LENGTH = 1900  # Discord 2000文字制限に余裕を持たせる


def _clean_mention(text: str) -> str:
    """@メンションタグを除去してタスク文字列を返す."""
    return re.sub(r"<@!?\d+>", "", text).strip()


def _split_message(text: str, limit: int = DISCORD_MAX_LENGTH) -> list[str]:
    """長いメッセージを Discord の文字数制限に合わせて分割."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunk = text[:limit]
        # 改行位置で切ると読みやすい
        last_newline = chunk.rfind("\n")
        if last_newline > limit // 2:
            chunk = text[:last_newline]
        chunks.append(chunk)
        text = text[len(chunk):].lstrip("\n")
    return chunks


class BushidanDiscordBot(discord.Client):
    """武士団 Discord Bot - 完全統合版."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # MESSAGE CONTENT INTENT 必須
        super().__init__(intents=intents)
        self._orchestrator = None
        self._init_error: Optional[str] = None

    async def setup_hook(self) -> None:
        """Bot 起動直後 (on_ready の前) にオーケストレーターを初期化."""
        try:
            from utils.config import load_config
            from core.system_orchestrator import SystemOrchestrator

            logger.info("🔧 武士団システムを初期化中...")
            config = load_config()
            self._orchestrator = SystemOrchestrator(config)
            await self._orchestrator.initialize()
            logger.info("✅ 武士団システム初期化完了")
        except Exception as e:
            self._init_error = str(e)
            logger.error("❌ 武士団システム初期化失敗: %s", e)

    async def on_ready(self) -> None:
        """Bot 起動完了."""
        logger.info("🏯 武士団 Discord Bot 起動完了: %s (ID: %s)", self.user, self.user.id)
        logger.info("サーバー内で @%s とメンションして呼び出してください", self.user.name)
        if self._init_error:
            logger.warning("⚠️ システム初期化エラーあり: %s", self._init_error)
        status = "命令をお待ちしています" if not self._init_error else "初期化エラー - ログを確認"
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=status,
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
        logger.info("📩 任務受信 from %s: %s", message.author.display_name, task[:100])

        if not task:
            await message.reply("はい、何かご用でしょうか？")
            return

        # 処理中メッセージを先に送る
        processing_msg = await message.reply(f"📋 任務受領: **{task[:80]}**\n⏳ 処理中...")

        try:
            result = await self._process(task, message)
            chunks = _split_message(result)
            await processing_msg.edit(content=chunks[0])
            for chunk in chunks[1:]:
                await message.channel.send(chunk)
        except Exception as e:
            logger.exception("タスク処理エラー")
            await processing_msg.edit(content=f"❌ エラーが発生しました: {e}")

    async def _process(self, task: str, message: discord.Message) -> str:
        """タスクを武士団システムに委譲する."""

        # オーケストレーター未初期化の場合
        if self._orchestrator is None:
            if self._init_error:
                return (
                    f"⚠️ 武士団システムが初期化されていません。\n"
                    f"エラー: `{self._init_error}`\n\n"
                    f".env の CLAUDE_API_KEY と GEMINI_API_KEY を確認してください。"
                )
            return "⚠️ 武士団システムを初期化中です。しばらくお待ちください。"

        # 将軍が使えない場合
        if self._orchestrator.shogun is None:
            return "❌ 将軍（Shogun）が初期化されていません。ログを確認してください。"

        # Task オブジェクトを生成して将軍に委譲
        from core.shogun import Task, TaskComplexity
        shogun_task = Task(
            content=task,
            complexity=TaskComplexity.MEDIUM,  # 将軍の _assess_complexity が自動評価・上書き
            context={
                "source": "discord",
                "author": str(message.author),
                "author_id": str(message.author.id),
                "channel": str(message.channel),
                "guild": str(message.guild) if message.guild else "DM",
            },
            priority=1,
            source="discord",
        )

        result = await self._orchestrator.shogun.process_task(shogun_task)

        # エラー処理
        if result.get("status") == "failed" or "error" in result:
            error_msg = result.get("error", "不明なエラー")
            return f"❌ 処理失敗: {error_msg}"

        content = result.get("result", "")
        elapsed = result.get("elapsed_time", 0)

        if not content:
            return "⚠️ 返答が空でした。ログを確認してください。"

        return f"{content}\n\n*⏱️ 処理時間: {elapsed:.1f}秒*"


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
