"""武士団 Mattermost Bot - インタラクティブボタン統合版

Discord でできなかったことを Mattermost で実現:
  ✅ [承認] [却下] [修正指示] ボタンによる承認フロー
  ✅ スラッシュコマンド /bushidan <タスク> で直接投入
  ✅ 16000文字の長文レスポンス
  ✅ 投稿の上書き更新 (受領通知 → 結果)

起動:  python -m bushidan.mattermost_bot

環境変数:
  MATTERMOST_URL           - ホスト名のみ (例: 192.168.11.234)
  MATTERMOST_PORT          - ポート番号 (デフォルト: 8065)
  MATTERMOST_SCHEME        - http / https (デフォルト: http)
  MATTERMOST_TOKEN         - Bot アクセストークン
  MATTERMOST_CHANNEL       - デフォルトチャンネルID
  MATTERMOST_CALLBACK_HOST - このサーバーのホスト (例: 192.168.11.230)
  MATTERMOST_CALLBACK_PORT - コールバック受信ポート (デフォルト: 8066)
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "mattermost_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("bushidan.mattermost")

# ── 依存チェック ──────────────────────────────────────────────────────

try:
    from mattermostdriver.driver import Driver
    from mattermostdriver.websocket import Websocket
    HAS_MATTERMOST = True
except ImportError:
    HAS_MATTERMOST = False

try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

MM_MAX_LENGTH = 16000

# ── エージェント別メンション設定 v12 (10役職) ────────────────────────────────
# Mattermost ユーザー名 → エージェントキー
AGENT_USERNAMES: dict[str, str] = {
    "daigensui-bot": "daigensui",
    "shogun-bot":    "shogun",
    "gunshi-bot":    "gunshi",
    "sanbo-a-bot":   "sanbo",      # 旧アカウント流用
    "kengyo-bot":    "kengyo",
    "onmitsu-bot":   "onmitsu",
    "uketuke-bot":   "uketuke",
    "gaiji-bot":     "gaiji",
    "yuhitsu-bot":   "yuhitsu",
    "seppou-bot":    "seppou",
    # 後方互換 (旧アカウント名)
    "karo-b-bot":    "seppou",
}

# Mattermost ユーザーID → エージェントキー
AGENT_USER_IDS: dict[str, str] = {
    "qdgwz7m43i8sxqqyz8sioopcme": "daigensui",
    "shydug161tdxunfjhtg6ksha1o": "shogun",
    "s5tqp9iqajydzg1ta3aysuofuw": "gunshi",
    "b1oyouc777rd3rp73hiog3hpir": "sanbo",
    "nzrkeikfebyjj8oo5fnn4jdfer": "kengyo",
    "ys4w69oc3fyxmbcq1hhm7cywar": "onmitsu",
    "t3pk8bgx9j8d5e8anr449b6ixh": "uketuke",
    "ipzbzaxfdifk7xudbe33k59pwy": "gaiji",
    "xsac94u8zin5xf355s9ca93pqe": "yuhitsu",
    "igpx8m7fainfmpr16gh6a84dzo": "seppou",
    # 後方互換 (旧アカウント)
    "36zqhqgnh3brbp6ng87wj7fjda": "seppou",   # 旧 karo_b
    "b5rdyq38pbfrxfneeecx9shdxe": "sanbo",    # 旧 sanbo_b
    "pebjz98tnpy6mk51sjh884py5a": "seppou",   # 旧 karo_a
}

# エージェントキー → LangGraph 強制ルート
# daigensui / shogun は Anthropic API を直接呼び出し (AGENT_CLAUDE_MODELS 参照)
AGENT_FORCED_ROUTES: dict[str, str] = {
    "gunshi":   "gunshi_pdca",    # 軍師 → o3-mini PDCA
    "sanbo":    "taisho_mcp",     # 参謀 → MCP ツール連携
    "kengyo":   "gaiji_rag",      # 検校 → 外事 Command R+ (視覚解析含む)
    "gaiji":    "gaiji_rag",      # 外事 → Command R+
    "uketuke":  "karo_default",   # 受付 → Command R フォールバック
    "yuhitsu":  "yuhitsu_jp",     # 右筆 → ELYZA 日本語清書
    "seppou":   "groq_qa",        # 斥候 → Llama 3.3 Groq
    "onmitsu":  "onmitsu_local",  # 隠密 → Nemotron ローカル
    # 後方互換
    "sanbo_a":  "taisho_mcp",
    "sanbo_b":  "taisho_mcp",
    "karo_a":   "gaiji_rag",
    "karo_b":   "groq_qa",
}

# エージェントペルソナ (Claude 直接呼び出し用)
AGENT_PERSONAS: dict[str, str] = {
    "daigensui": (
        "あなたは大元帥（Claude Opus 4.6）、武士団マルチエージェントシステムの総司令官です。"
        "最高難度の判断を下す最高意思決定者として、深く洞察に富んだ回答を日本語でしてください。"
    ),
    "shogun": (
        "あなたは将軍（Claude Sonnet 4.6）、武士団のメインワーカーです。"
        "高難度コーディングと実装を担当します。的確かつ実践的に日本語で回答してください。"
    ),
}

# エージェントキー → Claude モデルID (Anthropic 直接呼び出し用)
AGENT_CLAUDE_MODELS: dict[str, str] = {
    "daigensui": "claude-opus-4-6",
    "shogun":    "claude-sonnet-4-6",
}

MODE_DESCRIPTIONS = {
    "battalion": (
        "🏯 **大隊モード** (battalion) v12\n"
        "受付→外事→検校→将軍→軍師→参謀→右筆→斥候→隠密→大元帥 全10役職"
    ),
    "company":  "🏠 **中隊モード** (company)\n参謀→斥候→右筆 軽量・高速",
    "platoon":  "⚔️ **小隊モード** (platoon)\n斥候のみ 最軽量",
}


def _split_message(text: str, limit: int = MM_MAX_LENGTH) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunk = text[:limit]
        last_nl = chunk.rfind("\n")
        if last_nl > limit // 2:
            chunk = text[:last_nl]
        chunks.append(chunk)
        text = text[len(chunk):].lstrip("\n")
    return chunks


class MattermostAPI:
    """mattermostdriver (同期) を asyncio.to_thread でラップした薄いラッパー."""

    def __init__(self, driver: "Driver") -> None:
        self._d = driver

    async def get_user(self, user_id: str) -> Dict:
        return await asyncio.to_thread(self._d.users.get_user, user_id)

    async def create_post(self, options: Dict) -> Dict:
        return await asyncio.to_thread(self._d.posts.create_post, options=options)

    async def patch_post(self, post_id: str, options: Dict) -> None:
        await asyncio.to_thread(self._d.posts.patch_post, post_id, options=options)

    async def make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        def _call():
            return self._d.client.make_request(method, endpoint, **kwargs)
        return await asyncio.to_thread(_call)


class BushidanMattermostBot:
    """武士団 Mattermost Bot."""

    def __init__(self) -> None:
        self._driver: Optional[Driver]  = None
        self._api:    Optional[MattermostAPI] = None
        self._ws:     Optional[Websocket] = None
        self._bot_user_id:  Optional[str] = None
        self._bot_username: Optional[str] = None
        self._orchestrator  = None
        self._init_error:   Optional[str] = None
        self._current_mode: str = "battalion"
        self._cmd_prefix: str = os.environ.get("MATTERMOST_COMMAND_PREFIX", "!")
        self._approval_mgr  = None
        self._reporter      = None   # MattermostReporter (per-agent posting)
        self._callback_base: str = ""

    # ── 起動 ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        url    = os.environ.get("MATTERMOST_URL", "")
        token  = os.environ.get("MATTERMOST_TOKEN", "")
        port   = int(os.environ.get("MATTERMOST_PORT", "8065"))
        scheme = os.environ.get("MATTERMOST_SCHEME", "http")
        cb_host = os.environ.get("MATTERMOST_CALLBACK_HOST", "192.168.11.230")
        cb_port = int(os.environ.get("MATTERMOST_CALLBACK_PORT", "8066"))
        self._callback_base = f"http://{cb_host}:{cb_port}"

        if not url or not token:
            logger.error("MATTERMOST_URL と MATTERMOST_TOKEN を設定してください。")
            sys.exit(1)

        self._driver = Driver({
            "url": url, "token": token, "port": port, "scheme": scheme, "debug": False,
        })
        await asyncio.to_thread(self._driver.login)

        self._api = MattermostAPI(self._driver)
        me = await self._api.get_user("me")
        self._bot_user_id  = me["id"]
        self._bot_username = me["username"]
        logger.info("🏯 Mattermost ログイン: @%s", self._bot_username)

        # 承認マネージャー初期化
        from bushidan.mattermost_approval import MattermostApprovalManager
        self._approval_mgr = MattermostApprovalManager()
        self._approval_mgr.set_api(self._api)

        # エージェント別レポーター初期化
        from bushidan.mattermost_reporter import MattermostReporter
        channel_id = os.environ.get("MATTERMOST_CHANNEL", "")
        self._reporter = MattermostReporter(
            url=url, port=port, channel_id=channel_id, scheme=scheme
        )
        logger.info("🤖 エージェント別レポーター初期化完了 (%d エージェント)",
                    len(self._reporter.available_agents))

        # 武士団システム初期化
        await self._initialize_orchestrator()

        # HTTP コールバックサーバー (ボタン / スラッシュコマンド)
        if HAS_AIOHTTP:
            asyncio.create_task(self._run_callback_server(cb_port))
            logger.info("🌐 コールバックサーバー起動: %s", self._callback_base)

        # WebSocket 接続 (直接 await)
        self._ws = Websocket(self._driver.options, self._driver.client.token)
        logger.info("⚡ WebSocket 接続中... @%s をメンションして呼び出せます", self._bot_username)
        await self._ws.connect(self._handle_event)

    # ── HTTP コールバックサーバー ────────────────────────────────────

    async def _run_callback_server(self, port: int) -> None:
        app = web.Application()
        app.router.add_post("/api/actions", self._http_action_handler)
        app.router.add_post("/api/slash",   self._http_slash_handler)
        app.router.add_get("/health",       lambda r: web.json_response({"status": "ok"}))
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port).start()
        logger.info("✅ コールバックサーバー listening ::%d", port)

    async def _http_action_handler(self, request: "web.Request") -> "web.Response":
        try:
            body       = await request.json()
            ctx        = body.get("context", {})
            request_id = ctx.get("request_id", "")
            action     = ctx.get("action", "")
            user_name  = body.get("user_name", "unknown")
            user_text  = body.get("text", "")
            logger.info("🔔 ボタンクリック: [%s] %s by %s", request_id, action, user_name)

            if self._approval_mgr and request_id:
                msg = await self._approval_mgr.handle_action(
                    request_id, action, user_name, user_text)
            else:
                msg = "⚠️ 承認マネージャー未初期化"

            return web.json_response({"ephemeral_text": msg, "skip_slack_parsing": True})
        except Exception as e:
            logger.exception("アクションハンドラーエラー")
            return web.json_response({"ephemeral_text": f"❌ {e}"}, status=500)

    async def _http_slash_handler(self, request: "web.Request") -> "web.Response":
        try:
            data       = await request.post()
            text       = data.get("text", "").strip()
            channel_id = data.get("channel_id", "")
            user_name  = data.get("user_name", "unknown")
            post_id    = data.get("post_id", "")
            logger.info("⚡ /bushidan: user=%s text=%s", user_name, text[:80])

            if not text:
                return web.json_response({
                    "response_type": "ephemeral",
                    "text": "使い方: `/bushidan <タスク内容>`",
                })

            asyncio.create_task(
                self._process_slash_task(text, channel_id, post_id, user_name))

            return web.json_response({
                "response_type": "in_channel",
                "text": f"📋 **@{user_name}** の任務受領: `{text[:80]}`\n⏳ 処理中...",
            })
        except Exception as e:
            logger.exception("スラッシュハンドラーエラー")
            return web.json_response({"text": f"❌ {e}"}, status=500)

    async def _process_slash_task(
        self, task: str, channel_id: str, root_id: str, user_name: str
    ) -> None:
        try:
            result_dict = await self._process_task_dict(task, channel_id, root_id)
            handled_by = result_dict.get("handled_by", "unknown")
            result_text = result_dict.get("_formatted", "")
            chunks = _split_message(result_text)
            if self._reporter:
                for chunk in chunks:
                    await self._reporter.post_from_node(
                        handled_by, chunk, root_id=root_id, channel_id=channel_id)
            else:
                for chunk in chunks:
                    await self._post(channel_id, chunk, root_id)
        except Exception as e:
            logger.exception("スラッシュタスク処理エラー")
            await self._post(channel_id, f"❌ エラー: {e}", root_id)

    # ── WebSocket イベント処理 ────────────────────────────────────────

    async def _handle_event(self, event_raw: str) -> None:
        try:
            event = json.loads(event_raw) if isinstance(event_raw, str) else event_raw
            if event.get("event") != "posted":
                return

            data     = event.get("data", {})
            post_raw = data.get("post")
            if not post_raw:
                return

            post = json.loads(post_raw) if isinstance(post_raw, str) else post_raw
            if post.get("user_id") == self._bot_user_id:
                return

            message    = post.get("message", "")
            channel_id = post.get("channel_id", "")
            post_id    = post.get("id", "")
            root_id    = post.get("root_id") or post_id

            mentions_raw = data.get("mentions", "[]") or "[]"
            mentions = json.loads(mentions_raw) if isinstance(mentions_raw, str) else mentions_raw

            is_mentioned_main = (
                (self._bot_username and f"@{self._bot_username}" in message)
                or self._bot_user_id in mentions
            )
            # 特定エージェントへの直接メンション検出
            mentioned_agent = self._find_mentioned_agent(message, mentions)

            if not is_mentioned_main and not mentioned_agent:
                return

            sender = data.get("sender_name", "unknown")
            task   = re.sub(r"@\S+", "", message).strip() or message.strip()

            if mentioned_agent and not is_mentioned_main:
                # 特定エージェントへの直接呼び出し
                logger.info("🎯 エージェント指名 [%s] from %s: %s", mentioned_agent, sender, task[:80])
                asyncio.create_task(
                    self._call_agent_direct(mentioned_agent, task, channel_id, root_id))
                return

            logger.info("📩 任務受信 from %s: %s", sender, task[:100])
            await self._handle_message(task, channel_id, root_id, post_id)

        except Exception as e:
            logger.exception("イベント処理エラー: %s", e)

    async def _handle_message(
        self, task: str, channel_id: str, root_id: str, post_id: str
    ) -> None:
        if not task:
            await self._post(channel_id, "はい、何かご用でしょうか？", root_id)
            return

        if task.startswith(self._cmd_prefix):
            await self._handle_command(task, channel_id, root_id)
            return

        ack_id = await self._post(
            channel_id, f"📋 任務受領: **{task[:80]}**\n⏳ 処理中...", root_id)
        try:
            result_dict = await self._process_task_dict(task, channel_id, root_id)
            handled_by = result_dict.get("handled_by", "unknown")
            agent_role = result_dict.get("agent_role", "")
            result_text = result_dict.get("_formatted", result_dict.get("result", ""))

            # 受領メッセージを「ルーティング先」表示に更新
            await self._update_post(
                ack_id,
                f"📋 任務受領: **{task[:80]}**\n"
                f"🔀 ルーティング → **{agent_role or handled_by}**"
            )

            # エージェント専用アカウントから返答を投稿
            chunks = _split_message(result_text)
            if self._reporter:
                await self._reporter.post_from_node(
                    handled_by, chunks[0], root_id=root_id, channel_id=channel_id)
                for chunk in chunks[1:]:
                    await self._reporter.post_from_node(
                        handled_by, chunk, root_id=root_id, channel_id=channel_id)
            else:
                await self._post(channel_id, chunks[0], root_id)
                for chunk in chunks[1:]:
                    await self._post(channel_id, chunk, root_id)
        except Exception as e:
            logger.exception("タスク処理エラー")
            await self._update_post(ack_id, f"❌ エラー: {e}")

    # ── コマンド ─────────────────────────────────────────────────────

    async def _handle_command(self, cmd_str: str, channel_id: str, root_id: str) -> None:
        parts = cmd_str.split()
        cmd   = parts[0].lstrip(self._cmd_prefix).lower()
        args  = parts[1:]
        p     = self._cmd_prefix

        if cmd == "mode":
            await self._cmd_mode(args, channel_id, root_id)
        elif cmd == "status":
            await self._cmd_status(channel_id, root_id)
        elif cmd == "help":
            await self._post(channel_id, (
                "**🏯 武士団 Mattermost Bot コマンド**\n\n"
                "`@bushidan-bot <タスク>` — タスク投入\n"
                "`/bushidan <タスク>` — スラッシュコマンド\n\n"
                f"`{p}mode [battalion|company|platoon]` — モード確認・切り替え\n"
                f"`{p}status` — 9層システム状態\n"
                f"`{p}help` — このヘルプ\n\n"
                "**承認フロー:**\n"
                "重要操作時に [✅ 承認] [❌ 却下] [✏️ 修正指示] ボタンが表示されます。"
            ), root_id)
        else:
            await self._post(channel_id,
                f"❓ 不明なコマンド: `{p}{cmd}`\n`{p}help` でコマンド一覧", root_id)

    async def _cmd_mode(self, args: list, channel_id: str, root_id: str) -> None:
        p = self._cmd_prefix
        if not args:
            desc = MODE_DESCRIPTIONS.get(self._current_mode, "不明")
            await self._post(channel_id,
                f"**現在のモード:** `{self._current_mode}`\n\n{desc}\n\n"
                f"`{p}mode battalion|company|platoon` で切り替え", root_id)
            return
        new_mode = args[0].lower()
        if new_mode not in MODE_DESCRIPTIONS:
            await self._post(channel_id,
                f"❌ 無効: `{new_mode}` | battalion / company / platoon", root_id)
            return
        mid = await self._post(channel_id,
            f"🔄 `{self._current_mode}` → `{new_mode}` 切り替え中...", root_id)
        ok = await self._initialize_orchestrator(new_mode)
        if ok:
            await self._update_post(mid,
                f"✅ 完了!\n\n{MODE_DESCRIPTIONS[new_mode]}")
        else:
            await self._update_post(mid, f"❌ 失敗: `{self._init_error}`")

    async def _cmd_status(self, channel_id: str, root_id: str) -> None:
        if not self._orchestrator:
            await self._post(channel_id,
                f"⚠️ システム未初期化\n`{self._init_error or '不明'}`", root_id)
            return
        def chk(a): return "✅" if getattr(self._orchestrator, a, None) else "❌"
        llamacpp = self._orchestrator.health_status.get("llamacpp", False)
        kengyo   = getattr(self._orchestrator, "kengyo", None)
        kengyo_ok = bool(kengyo and (
            not hasattr(kengyo, "is_available") or kengyo.is_available()))
        pending = self._approval_mgr.get_pending_count() if self._approval_mgr else 0

        await self._post(channel_id, (
            f"**🏯 武士団 v{self._orchestrator.VERSION}** | "
            f"モード: `{self._current_mode}` | 承認待ち: {pending}件\n\n"
            f"👑 大元帥 {chk('daigensui')} | 🎌 将軍 {chk('shogun')} | "
            f"🧠 軍師 {chk('gunshi')}\n"
            f"⚔️ 参謀 {chk('sanbo')} | 👔 家老 {chk('karo')} | "
            f"👁️ 検校 {'✅' if kengyo_ok else '❌'} | "
            f"🥷 隠密 {'✅' if llamacpp else '⚠️'}\n\n"
            f"🌐 コールバック: `{self._callback_base}`"
        ), root_id)

    # ── オーケストレーター ────────────────────────────────────────────

    async def _initialize_orchestrator(self, mode: Optional[str] = None) -> bool:
        try:
            from utils.config import load_config
            from core.system_orchestrator import SystemOrchestrator
            if mode:
                os.environ["SYSTEM_MODE"] = mode
            config = load_config()
            self._orchestrator = SystemOrchestrator(config)
            await self._orchestrator.initialize()
            self._current_mode = config.mode.value
            self._init_error   = None
            logger.info("✅ 武士団システム初期化完了 (モード: %s)", self._current_mode)
            return True
        except Exception as e:
            self._init_error = str(e)
            logger.error("❌ 初期化失敗: %s", e)
            return False

    async def _process_task_dict(self, task: str, channel_id: str, root_id: str) -> dict:
        """LangGraphの生の結果辞書を返す。handled_by / agent_role を含む。"""
        if not self._orchestrator:
            return {"status": "failed", "error": self._init_error or "未初期化",
                    "_formatted": f"⚠️ 未初期化\nエラー: `{self._init_error}`",
                    "handled_by": "unknown", "agent_role": ""}
        context = {
            "source": "mattermost", "channel_id": channel_id, "root_id": root_id,
            "mode": self._current_mode,
            "approval_manager": self._approval_mgr,
            "callback_base": self._callback_base,
        }
        try:
            result = await self._orchestrator.process_task(task, context)
        except Exception as e:
            logger.exception("タスク処理エラー")
            return {"status": "failed", "error": str(e),
                    "_formatted": f"❌ 処理失敗: {e}",
                    "handled_by": "unknown", "agent_role": ""}

        if result.get("status") == "failed" or result.get("error"):
            err = result.get("error", "不明")
            tb  = result.get("traceback", "")
            tb_short = "\n".join(tb.strip().splitlines()[-10:]) if tb else ""
            fmt = f"❌ 処理失敗: {err}\n```\n{tb_short}\n```" if tb_short else f"❌ 処理失敗: {err}"
            result["_formatted"] = fmt
            return result

        content = result.get("result", "")
        elapsed = result.get("elapsed_time", result.get("execution_time", 0))
        fmt = (f"{content}\n\n*⏱️ {elapsed:.1f}秒 | {self._current_mode}*"
               if content else "⚠️ 返答が空でした。")
        result["_formatted"] = fmt
        return result

    async def _process_task(self, task: str, channel_id: str, root_id: str) -> str:
        """後方互換: 結果を文字列で返す。"""
        d = await self._process_task_dict(task, channel_id, root_id)
        return d.get("_formatted", d.get("result", "❌ 不明なエラー"))

    # ── 投稿ヘルパー ─────────────────────────────────────────────────

    async def _post(self, channel_id: str, message: str, root_id: str = "") -> str:
        opts: Dict[str, Any] = {"channel_id": channel_id, "message": message}
        if root_id:
            opts["root_id"] = root_id
        post = await self._api.create_post(opts)
        return post.get("id", "")

    async def _update_post(self, post_id: str, message: str) -> None:
        if not post_id:
            return
        try:
            await self._api.patch_post(post_id, {"message": message})
        except Exception as e:
            logger.warning("投稿更新失敗: %s", e)

    # ── エージェント直接メンション処理 ───────────────────────────────────

    def _find_mentioned_agent(self, message: str, mentions: list) -> Optional[str]:
        """メッセージ内で特定エージェントが @メンションされているか確認"""
        for username, agent_key in AGENT_USERNAMES.items():
            if f"@{username}" in message:
                return agent_key
        for user_id in mentions:
            if user_id in AGENT_USER_IDS:
                return AGENT_USER_IDS[user_id]
        return None

    async def _call_claude_direct(self, agent_key: str, message: str) -> str:
        """大元帥・将軍への呼び出し (Proプラン CLI 優先 → API フォールバック)"""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "❌ ANTHROPIC_API_KEY が設定されていません"

        model   = AGENT_CLAUDE_MODELS.get(agent_key, "claude-sonnet-4-6")
        persona = AGENT_PERSONAS.get(agent_key, "")
        logger.info("🎌 [%s] Claude呼び出し開始 (CLI優先): %s (%s...)", agent_key, model, message[:60])

        try:
            from utils.claude_cli_client import call_claude_with_fallback
            return await call_claude_with_fallback(
                prompt=message,
                model=model,
                api_key=api_key,
                system=persona or None,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error("[%s] Claude呼び出しエラー: %s", agent_key, e)
            return f"❌ {agent_key} 応答エラー: {e}"

    async def _call_agent_direct(
        self, agent_key: str, message: str, channel_id: str, root_id: str
    ) -> None:
        """特定エージェントに直接タスクを投げてそのアカウントから返信"""
        from bushidan.mattermost_reporter import AGENT_CONFIG
        cfg = AGENT_CONFIG.get(agent_key, {})
        emoji  = cfg.get("emoji", "🤖")
        model  = cfg.get("model", agent_key)
        logger.info("🎯 エージェント直接呼び出し: %s (%s)", agent_key, model)

        try:
            if agent_key in AGENT_CLAUDE_MODELS:
                # 大元帥・将軍: Anthropic API 直接
                result = await self._call_claude_direct(agent_key, message)
            else:
                # 他エージェント: LangGraph forced_route 経由
                forced_route = AGENT_FORCED_ROUTES.get(agent_key, "karo_default")
                context = {
                    "source":       "mattermost_direct",
                    "channel_id":   channel_id,
                    "root_id":      root_id,
                    "mode":         self._current_mode,
                    "forced_route": forced_route,
                    "forced_agent": agent_key,
                }
                result_dict = await self._process_task_dict_with_context(
                    message, channel_id, root_id, context)
                result = result_dict.get("_formatted", "❌ 処理失敗")

            # エージェントアカウントから投稿
            full_message = f"{emoji} **[{model}]**\n\n{result}"
            if self._reporter:
                for chunk in _split_message(full_message):
                    await self._reporter.post_as(
                        agent_key, chunk, channel_id=channel_id, root_id=root_id)
            else:
                await self._post(channel_id, full_message, root_id)

        except Exception as e:
            logger.exception("エージェント直接呼び出しエラー [%s]: %s", agent_key, e)
            await self._post(channel_id, f"❌ {agent_key} エラー: {e}", root_id)

    async def _process_task_dict_with_context(
        self, task: str, channel_id: str, root_id: str, extra_context: dict
    ) -> dict:
        """forced_route などの追加コンテキストを注入してタスク処理"""
        if not self._orchestrator:
            return {"_formatted": f"⚠️ 未初期化: {self._init_error}", "handled_by": "unknown"}
        context = {
            "source": "mattermost", "channel_id": channel_id, "root_id": root_id,
            "mode": self._current_mode,
            "approval_manager": self._approval_mgr,
            "callback_base": self._callback_base,
            **extra_context,
        }
        try:
            result = await self._orchestrator.process_task(task, context)
        except Exception as e:
            logger.exception("タスク処理エラー")
            return {"_formatted": f"❌ 処理失敗: {e}", "handled_by": "unknown"}

        if result.get("status") == "failed" or result.get("error"):
            err = result.get("error", "不明")
            result["_formatted"] = f"❌ 処理失敗: {err}"
            return result

        content = result.get("result", "")
        elapsed = result.get("elapsed_time", result.get("execution_time", 0))
        result["_formatted"] = (
            f"{content}\n\n*⏱️ {elapsed:.1f}秒*" if content else "⚠️ 返答が空でした。"
        )
        return result


# ── エントリーポイント ────────────────────────────────────────────

def main() -> None:
    missing = []
    if not HAS_MATTERMOST:
        missing.append("mattermostdriver")
    if not HAS_AIOHTTP:
        missing.append("aiohttp")
    if missing:
        print(f"pip install {' '.join(missing)} が必要です", file=sys.stderr)
        sys.exit(1)

    for k in ("MATTERMOST_URL", "MATTERMOST_TOKEN"):
        if not os.environ.get(k):
            print(f"環境変数 {k} が未設定です", file=sys.stderr)
            sys.exit(1)

    bot = BushidanMattermostBot()
    logger.info("🏯 武士団 Mattermost Bot を起動します...")
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
