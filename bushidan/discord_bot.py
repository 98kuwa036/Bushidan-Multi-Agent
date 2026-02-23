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

コマンド:
  /mode          - 現在のモードを表示
  /mode battalion - 大隊モード（全機能）
  /mode company   - 中隊モード（軽量）
  /mode platoon   - 小隊モード（最軽量）
  /status        - システム状態を表示
  /help          - ヘルプを表示

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

# モード説明
MODE_DESCRIPTIONS = {
    "battalion": "🏯 **大隊モード** (battalion)\n将軍→軍師→家老→大将→足軽 全5層\n複雑なエンジニアリングタスク向け",
    "company": "🏠 **中隊モード** (company)\n家老→大将→足軽\n軽量・高速、Discord Bot 向け",
    "platoon": "⚔️ **小隊モード** (platoon)\n大将→足軽のみ\n最軽量・オフライン処理向け",
}


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
        self._current_mode: str = "battalion"  # デフォルトは大隊モード

    async def setup_hook(self) -> None:
        """Bot 起動直後 (on_ready の前) にオーケストレーターを初期化."""
        await self._initialize_orchestrator()

    async def _initialize_orchestrator(self, mode: str = None) -> bool:
        """オーケストレーターを初期化または再初期化."""
        try:
            from utils.config import load_config
            from core.system_orchestrator import SystemOrchestrator, SystemMode

            # モード指定があれば環境変数を一時的に設定
            if mode:
                os.environ["SYSTEM_MODE"] = mode

            logger.info("🔧 武士団システムを初期化中... (モード: %s)", mode or "default")
            config = load_config()
            self._orchestrator = SystemOrchestrator(config)
            await self._orchestrator.initialize()
            self._current_mode = config.mode.value
            self._init_error = None
            logger.info("✅ 武士団システム初期化完了 (モード: %s)", self._current_mode)
            return True
        except Exception as e:
            self._init_error = str(e)
            logger.error("❌ 武士団システム初期化失敗: %s", e)
            return False

    async def on_ready(self) -> None:
        """Bot 起動完了."""
        logger.info("🏯 武士団 Discord Bot 起動完了: %s (ID: %s)", self.user, self.user.id)
        logger.info("サーバー内で @%s とメンションして呼び出してください", self.user.name)
        if self._init_error:
            logger.warning("⚠️ システム初期化エラーあり: %s", self._init_error)
        await self._update_presence()

    async def _update_presence(self) -> None:
        """Discord ステータスを更新."""
        if self._init_error:
            status = "初期化エラー"
        else:
            mode_emoji = {"battalion": "🏯", "company": "🏠", "platoon": "⚔️"}
            status = f"{mode_emoji.get(self._current_mode, '')} {self._current_mode}"
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

        # コマンド処理
        if task.startswith("/"):
            await self._handle_command(task, message)
            return

        # 通常タスク処理
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

    async def _handle_command(self, command: str, message: discord.Message) -> None:
        """スラッシュコマンドを処理."""
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if cmd == "/mode":
            await self._cmd_mode(args, message)
        elif cmd == "/status":
            await self._cmd_status(message)
        elif cmd == "/help":
            await self._cmd_help(message)
        else:
            await message.reply(f"❓ 不明なコマンド: `{cmd}`\n`/help` でコマンド一覧を確認できます。")

    async def _cmd_mode(self, args: list, message: discord.Message) -> None:
        """モード表示・切り替えコマンド."""
        valid_modes = ["battalion", "company", "platoon"]

        # 引数なし: 現在のモードを表示
        if not args:
            current_desc = MODE_DESCRIPTIONS.get(self._current_mode, "不明")
            response = (
                f"**現在のモード:** `{self._current_mode}`\n\n"
                f"{current_desc}\n\n"
                f"---\n"
                f"**切り替え方法:**\n"
                f"`/mode battalion` - 大隊モード（全機能）\n"
                f"`/mode company` - 中隊モード（軽量）\n"
                f"`/mode platoon` - 小隊モード（最軽量）"
            )
            await message.reply(response)
            return

        # モード切り替え
        new_mode = args[0].lower()
        if new_mode not in valid_modes:
            await message.reply(
                f"❌ 無効なモード: `{new_mode}`\n"
                f"有効なモード: `battalion`, `company`, `platoon`"
            )
            return

        if new_mode == self._current_mode:
            await message.reply(f"ℹ️ 既に `{new_mode}` モードです。")
            return

        # モード切り替え実行
        status_msg = await message.reply(
            f"🔄 モードを `{self._current_mode}` → `{new_mode}` に切り替え中...\n"
            f"⏳ システムを再初期化しています..."
        )

        success = await self._initialize_orchestrator(new_mode)

        if success:
            await self._update_presence()
            new_desc = MODE_DESCRIPTIONS.get(new_mode, "")
            await status_msg.edit(
                content=(
                    f"✅ モード切り替え完了!\n\n"
                    f"{new_desc}"
                )
            )
        else:
            await status_msg.edit(
                content=(
                    f"❌ モード切り替え失敗\n"
                    f"エラー: `{self._init_error}`"
                )
            )

    async def _cmd_status(self, message: discord.Message) -> None:
        """システム状態を表示."""
        if self._orchestrator is None:
            await message.reply(
                f"⚠️ システム未初期化\n"
                f"エラー: `{self._init_error or '不明'}`"
            )
            return

        # 各層の状態を取得
        shogun = "✅" if self._orchestrator.shogun else "❌"
        gunshi = "✅" if self._orchestrator.gunshi else "❌"
        karo = "✅" if self._orchestrator.karo else "❌"
        taisho = "✅" if self._orchestrator.taisho else "❌"
        kengyo = "✅" if self._orchestrator.kengyo and self._orchestrator.kengyo.is_available() else "❌"

        # llama.cpp 状態
        llamacpp = "✅" if self._orchestrator.health_status.get("llamacpp") else "⚠️"

        response = (
            f"**🏯 武士団システム v{self._orchestrator.VERSION}**\n"
            f"**モード:** `{self._current_mode}`\n\n"
            f"**【5層階層】**\n"
            f"🎌 将軍: {shogun}\n"
            f"🧠 軍師: {gunshi}\n"
            f"👔 家老: {karo}\n"
            f"⚔️ 大将: {taisho}\n"
            f"👁️ 検校: {kengyo}\n\n"
            f"**【外部サービス】**\n"
            f"🔗 llama.cpp (CT 101): {llamacpp}\n"
        )
        await message.reply(response)

    async def _cmd_help(self, message: discord.Message) -> None:
        """ヘルプを表示."""
        response = (
            "**🏯 武士団 Discord Bot コマンド**\n\n"
            "**コマンド:**\n"
            "`/mode` - 現在のモードを表示\n"
            "`/mode <battalion|company|platoon>` - モード切り替え\n"
            "`/status` - システム状態を表示\n"
            "`/help` - このヘルプを表示\n\n"
            "**モード:**\n"
            "• `battalion` - 大隊（全5層、フル機能）\n"
            "• `company` - 中隊（家老→大将、軽量）\n"
            "• `platoon` - 小隊（大将のみ、最軽量）\n\n"
            "**使い方:**\n"
            "`@Bushidan <タスク>` - タスクを処理\n"
            "`@Bushidan /mode company` - 中隊モードに切り替え"
        )
        await message.reply(response)

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
                "mode": self._current_mode,
            },
            priority=1,
            source="discord",
        )

        result = await self._orchestrator.shogun.process_task(shogun_task)

        # エラー処理
        if result.get("status") == "failed" or "error" in result:
            error_msg = result.get("error", "不明なエラー")
            tb = result.get("traceback", "")
            if tb:
                # トレースバックの最後の数行だけ表示（Discord の文字数制限対策）
                tb_lines = tb.strip().splitlines()
                tb_short = "\n".join(tb_lines[-10:])
                logger.error("スタックトレース:\n%s", tb)
                return f"❌ 処理失敗: {error_msg}\n```\n{tb_short}\n```"
            return f"❌ 処理失敗: {error_msg}"

        content = result.get("result", "")
        elapsed = result.get("elapsed_time", 0)

        if not content:
            return "⚠️ 返答が空でした。ログを確認してください。"

        return f"{content}\n\n*⏱️ 処理時間: {elapsed:.1f}秒 | モード: {self._current_mode}*"


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
