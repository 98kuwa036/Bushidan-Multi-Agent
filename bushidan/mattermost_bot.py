"""武士団 Mattermost Bot - 完全統合版

core/system_orchestrator.py を経由して9層階層に委譲します。

  Mattermost メッセージ (@メンション)
       ↓
  BushidanMattermostBot.handle_event()
       ↓
  SystemOrchestrator.process_task()  ← BDI + インテリジェントルーター
       ↓
  SIMPLE    → 家老-B Groq (即応)
  MEDIUM    → 参謀-B Grok (並列実装)
  COMPLEX   → 軍師 → 参謀A/B → 家老
  STRATEGIC → 将軍自ら処理 / 大元帥最終裁可
       ↓
  Mattermost スレッドに返答

コマンド (メッセージで `!` プレフィックス):
  !mode              - 現在のモードを表示
  !mode battalion    - 大隊モード（全9層）
  !mode company      - 中隊モード（軽量）
  !mode platoon      - 小隊モード（最軽量）
  !status            - システム状態を表示（9層全員）
  !help              - ヘルプを表示

必要な環境変数:
  MATTERMOST_URL            - サーバー URL (例: chat.example.com)
  MATTERMOST_TOKEN          - Bot アクセストークン
  MATTERMOST_TEAM_NAME      - チーム名 (省略可)
  MATTERMOST_PORT           - ポート番号 (デフォルト: 443)
  MATTERMOST_SCHEME         - http/https (デフォルト: https)
  MATTERMOST_COMMAND_PREFIX - コマンドプレフィックス (デフォルト: !)

起動方法:
  python -m bushidan.mattermost_bot
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "mattermost_bot.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("bushidan.mattermost")

try:
    from mattermostdriver import AsyncDriver
    HAS_MATTERMOST = True
except ImportError:
    HAS_MATTERMOST = False

# Mattermost の文字数制限に余裕を持たせる (上限 16383)
MM_MAX_LENGTH = 16000

MODE_DESCRIPTIONS = {
    "battalion": (
        "🏯 **大隊モード** (battalion)\n"
        "大元帥→将軍→軍師→参謀A/B→家老A/B→検校→隠密 全9層\n"
        "複雑なエンジニアリングタスク・戦略的意思決定向け"
    ),
    "company": (
        "🏠 **中隊モード** (company)\n"
        "家老→大将→足軽\n"
        "軽量・高速、チャット Bot 向け"
    ),
    "platoon": (
        "⚔️ **小隊モード** (platoon)\n"
        "大将→足軽のみ\n"
        "最軽量・オフライン処理向け"
    ),
}


def _split_message(text: str, limit: int = MM_MAX_LENGTH) -> list[str]:
    """長いメッセージを Mattermost の文字数制限に合わせて分割."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunk = text[:limit]
        last_newline = chunk.rfind("\n")
        if last_newline > limit // 2:
            chunk = text[:last_newline]
        chunks.append(chunk)
        text = text[len(chunk):].lstrip("\n")
    return chunks


class BushidanMattermostBot:
    """武士団 Mattermost Bot - 完全統合版."""

    def __init__(self) -> None:
        self._driver: Optional[AsyncDriver] = None
        self._bot_user_id: Optional[str] = None
        self._bot_username: Optional[str] = None
        self._orchestrator = None
        self._init_error: Optional[str] = None
        self._current_mode: str = "battalion"
        self._cmd_prefix: str = os.environ.get("MATTERMOST_COMMAND_PREFIX", "!")

    async def start(self) -> None:
        """Bot を起動して WebSocket リスニングを開始."""
        url = os.environ.get("MATTERMOST_URL", "")
        token = os.environ.get("MATTERMOST_TOKEN", "")
        port = int(os.environ.get("MATTERMOST_PORT", "443"))
        scheme = os.environ.get("MATTERMOST_SCHEME", "https")

        if not url or not token:
            logger.error("MATTERMOST_URL と MATTERMOST_TOKEN を .env に設定してください。")
            sys.exit(1)

        self._driver = AsyncDriver({
            "url": url,
            "token": token,
            "port": port,
            "scheme": scheme,
            "debug": False,
        })

        await self._driver.login()

        me = await self._driver.users.get_user("me")
        self._bot_user_id = me["id"]
        self._bot_username = me["username"]
        logger.info(
            "🏯 Mattermost ログイン完了: @%s (ID: %s)",
            self._bot_username,
            self._bot_user_id,
        )

        # 武士団システムを初期化
        await self._initialize_orchestrator()

        logger.info("⚡ WebSocket 待機開始... @%s にメンションして呼び出せます", self._bot_username)
        await self._driver.init_websocket(self._handle_event)

    async def _initialize_orchestrator(self, mode: Optional[str] = None) -> bool:
        """オーケストレーターを初期化または再初期化."""
        try:
            from utils.config import load_config
            from core.system_orchestrator import SystemOrchestrator

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

    async def _handle_event(self, event_raw: str) -> None:
        """WebSocket から受信したイベントを処理."""
        try:
            event = json.loads(event_raw) if isinstance(event_raw, str) else event_raw

            if event.get("event") != "posted":
                return

            data = event.get("data", {})
            post_raw = data.get("post")
            if not post_raw:
                return

            post = json.loads(post_raw) if isinstance(post_raw, str) else post_raw

            # 自分自身のメッセージは無視
            if post.get("user_id") == self._bot_user_id:
                return

            message: str = post.get("message", "")
            channel_id: str = post.get("channel_id", "")
            post_id: str = post.get("id", "")
            # スレッドへの返信は root_id を維持、新規投稿は投稿自体を root に
            root_id: str = post.get("root_id") or post_id

            # @メンション検出
            is_mentioned = False
            if self._bot_username and f"@{self._bot_username}" in message:
                is_mentioned = True
            elif self._bot_user_id:
                mentions_raw = data.get("mentions", "[]")
                mentions = (
                    json.loads(mentions_raw)
                    if isinstance(mentions_raw, str)
                    else (mentions_raw or [])
                )
                if self._bot_user_id in mentions:
                    is_mentioned = True

            if not is_mentioned:
                return

            # @mention タグを除去してタスク文字列を取り出す
            task = re.sub(r"@\S+", "", message).strip()
            sender = data.get("sender_name", "unknown")
            logger.info("📩 任務受信 from %s: %s", sender, task[:100])

            await self._handle_message(task, channel_id, root_id, post_id)

        except Exception as e:
            logger.exception("イベント処理エラー: %s", e)

    async def _handle_message(
        self,
        task: str,
        channel_id: str,
        root_id: str,
        post_id: str,
    ) -> None:
        """メッセージを処理してレスポンスを投稿."""
        if not task:
            await self._post(channel_id, "はい、何かご用でしょうか？", root_id)
            return

        # コマンド処理
        if task.startswith(self._cmd_prefix):
            await self._handle_command(task, channel_id, root_id)
            return

        # 受領通知を先に投稿
        ack_id = await self._post(
            channel_id,
            f"📋 任務受領: **{task[:80]}**\n⏳ 処理中...",
            root_id,
        )

        try:
            result = await self._process_task(task, channel_id, root_id)
            chunks = _split_message(result)
            await self._update_post(ack_id, chunks[0])
            for chunk in chunks[1:]:
                await self._post(channel_id, chunk, root_id)
        except Exception as e:
            logger.exception("タスク処理エラー")
            await self._update_post(ack_id, f"❌ エラーが発生しました: {e}")

    async def _handle_command(self, command: str, channel_id: str, root_id: str) -> None:
        """コマンドを処理."""
        parts = command.split()
        cmd = parts[0].lstrip(self._cmd_prefix).lower()
        args = parts[1:] if len(parts) > 1 else []

        if cmd == "mode":
            await self._cmd_mode(args, channel_id, root_id)
        elif cmd == "status":
            await self._cmd_status(channel_id, root_id)
        elif cmd == "help":
            await self._cmd_help(channel_id, root_id)
        else:
            await self._post(
                channel_id,
                (
                    f"❓ 不明なコマンド: `{self._cmd_prefix}{cmd}`\n"
                    f"`{self._cmd_prefix}help` でコマンド一覧を確認できます。"
                ),
                root_id,
            )

    async def _cmd_mode(self, args: list[str], channel_id: str, root_id: str) -> None:
        """モード表示・切り替えコマンド."""
        valid_modes = ["battalion", "company", "platoon"]

        if not args:
            current_desc = MODE_DESCRIPTIONS.get(self._current_mode, "不明")
            p = self._cmd_prefix
            response = (
                f"**現在のモード:** `{self._current_mode}`\n\n"
                f"{current_desc}\n\n"
                f"---\n**切り替え方法:**\n"
                f"`{p}mode battalion` - 大隊モード（全9層）\n"
                f"`{p}mode company` - 中隊モード（軽量）\n"
                f"`{p}mode platoon` - 小隊モード（最軽量）"
            )
            await self._post(channel_id, response, root_id)
            return

        new_mode = args[0].lower()
        if new_mode not in valid_modes:
            await self._post(
                channel_id,
                f"❌ 無効なモード: `{new_mode}`\n有効なモード: `battalion`, `company`, `platoon`",
                root_id,
            )
            return

        if new_mode == self._current_mode:
            await self._post(channel_id, f"ℹ️ 既に `{new_mode}` モードです。", root_id)
            return

        msg_id = await self._post(
            channel_id,
            (
                f"🔄 モードを `{self._current_mode}` → `{new_mode}` に切り替え中...\n"
                f"⏳ システムを再初期化しています..."
            ),
            root_id,
        )

        success = await self._initialize_orchestrator(new_mode)

        if success:
            new_desc = MODE_DESCRIPTIONS.get(new_mode, "")
            await self._update_post(msg_id, f"✅ モード切り替え完了!\n\n{new_desc}")
        else:
            await self._update_post(
                msg_id,
                f"❌ モード切り替え失敗\nエラー: `{self._init_error}`",
            )

    async def _cmd_status(self, channel_id: str, root_id: str) -> None:
        """システム状態を表示 (v11.4 9層構成)."""
        if self._orchestrator is None:
            await self._post(
                channel_id,
                f"⚠️ システム未初期化\nエラー: `{self._init_error or '不明'}`",
                root_id,
            )
            return

        def _chk(attr: str) -> str:
            return "✅" if getattr(self._orchestrator, attr, None) else "❌"

        kengyo_obj = getattr(self._orchestrator, "kengyo", None)
        kengyo_ok = bool(
            kengyo_obj
            and (not hasattr(kengyo_obj, "is_available") or kengyo_obj.is_available())
        )
        llamacpp_ok = self._orchestrator.health_status.get("llamacpp", False)

        response = (
            f"**🏯 武士団システム v{self._orchestrator.VERSION}**\n"
            f"**モード:** `{self._current_mode}`\n\n"
            f"**【9層アーキテクチャ】**\n"
            f"👑 大元帥 (Claude Opus 4.5):  {_chk('daigensui')}\n"
            f"🎌 将軍   (Claude Sonnet 4.6): {_chk('shogun')}\n"
            f"🧠 軍師   (o3-mini high):      {_chk('gunshi')}\n"
            f"⚔️  参謀   (GPT-5 / Grok):     {_chk('sanbo')}\n"
            f"👔 家老-A (Gemini Flash):      {_chk('karo')}\n"
            f"🦙 家老-B (Llama 3.3 70B):    {_chk('karo')}\n"
            f"👁️  検校   (Gemini Vision):     {'✅' if kengyo_ok else '❌'}\n"
            f"🥷 隠密   (Nemotron Local):    {'✅' if llamacpp_ok else '⚠️ オフライン'}\n\n"
            f"**【外部サービス】**\n"
            f"🔗 llama.cpp (ProDesk 600): {'✅' if llamacpp_ok else '⚠️'}\n"
        )
        await self._post(channel_id, response, root_id)

    async def _cmd_help(self, channel_id: str, root_id: str) -> None:
        """ヘルプを表示."""
        p = self._cmd_prefix
        response = (
            "**🏯 武士団 Mattermost Bot コマンド**\n\n"
            "**コマンド:**\n"
            f"`{p}mode` - 現在のモードを表示\n"
            f"`{p}mode <battalion|company|platoon>` - モード切り替え\n"
            f"`{p}status` - システム状態を表示（9層全員）\n"
            f"`{p}help` - このヘルプを表示\n\n"
            "**モード:**\n"
            "- `battalion` - 大隊（全9層、戦略タスク向け）\n"
            "- `company`   - 中隊（家老→大将、軽量・高速）\n"
            "- `platoon`   - 小隊（大将のみ、最軽量）\n\n"
            "**使い方:**\n"
            "`@Bushidan <タスク>` - タスクを9層システムに投入\n"
            f"`@Bushidan {p}mode battalion` - 大隊モードに切り替え\n\n"
            "**Mattermost MCP ツール** (エージェント向け):\n"
            "`python -m mcp.mattermost_mcp_server` でMCPサーバーを起動"
        )
        await self._post(channel_id, response, root_id)

    async def _process_task(self, task: str, channel_id: str, root_id: str) -> str:
        """タスクを武士団システムに委譲."""
        if self._orchestrator is None:
            if self._init_error:
                return (
                    f"⚠️ 武士団システムが初期化されていません。\n"
                    f"エラー: `{self._init_error}`\n\n"
                    f"`.env` の API キー設定を確認してください。"
                )
            return "⚠️ 武士団システムを初期化中です。しばらくお待ちください。"

        context = {
            "source": "mattermost",
            "channel_id": channel_id,
            "root_id": root_id,
            "mode": self._current_mode,
        }

        try:
            result = await self._orchestrator.process_task(task, context)
        except Exception as e:
            logger.exception("タスク処理エラー: %s", e)
            return f"❌ 処理失敗: {e}"

        if result.get("status") == "failed" or "error" in result:
            error_msg = result.get("error", "不明なエラー")
            tb = result.get("traceback", "")
            if tb:
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

    async def _post(self, channel_id: str, message: str, root_id: str = "") -> str:
        """チャンネルにメッセージを投稿して post_id を返す."""
        options: dict = {"channel_id": channel_id, "message": message}
        if root_id:
            options["root_id"] = root_id
        post = await self._driver.posts.create_post(options=options)
        return post.get("id", "")

    async def _update_post(self, post_id: str, message: str) -> None:
        """既存の投稿内容を更新 (受領通知→結果に上書き)."""
        if not post_id:
            return
        try:
            await self._driver.posts.patch_post(post_id, options={"message": message})
        except Exception as e:
            logger.warning("投稿の更新に失敗: %s", e)


def main() -> None:
    """エントリーポイント."""
    if not HAS_MATTERMOST:
        print(
            "エラー: mattermostdriver が必要です\n"
            "インストール: pip install mattermostdriver",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("MATTERMOST_URL") or not os.environ.get("MATTERMOST_TOKEN"):
        print(
            "エラー: MATTERMOST_URL と MATTERMOST_TOKEN が設定されていません\n"
            ".env に以下を追加してください:\n"
            "  MATTERMOST_URL=chat.example.com\n"
            "  MATTERMOST_TOKEN=<bot-access-token>\n\n"
            "トークン取得方法:\n"
            "  Mattermost → メインメニュー → 統合機能 → ボットアカウント",
            file=sys.stderr,
        )
        sys.exit(1)

    bot = BushidanMattermostBot()
    logger.info("武士団 Mattermost Bot を起動します...")
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
