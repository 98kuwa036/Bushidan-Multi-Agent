"""
interfaces/matrix_bot.py — Matrix 双方向ボット v1

Matrix の #general ルームを監視し、人間からのメッセージを
LangGraph Router に渡して応答する。

フロー:
  人間 → #general に投稿
    ↓
  MatrixBot が /sync で検知
    ↓
  LangGraph Router で処理 (10役職ルーティング)
    ↓
  担当役職の Matrix アカウントで応答投稿

起動:
  python -m interfaces.matrix_bot
  または main.py から asyncio.create_task() で起動
"""

import asyncio
import json
import logging
import os
import time
import urllib.parse
from typing import Optional

import aiohttp
from dotenv import load_dotenv
load_dotenv()  # .env を読み込む

logger = logging.getLogger(__name__)

# ── Prometheus メトリクス (オプション) ──────────────────────────────────
try:
    from prometheus_client import Counter, Histogram
    _matrix_recv = Counter(
        "bushidan_matrix_messages_received_total",
        "Matrix messages received from humans",
        ["channel"],
    )
    _matrix_sent = Counter(
        "bushidan_matrix_messages_sent_total",
        "Matrix messages sent by roles",
        ["channel", "role"],
    )
    _matrix_duration = Histogram(
        "bushidan_matrix_processing_seconds",
        "Matrix message processing time",
        ["channel"],
        buckets=[1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

MATRIX_HOMESERVER = "http://192.168.11.234:8008"
MATRIX_DOMAIN = "matrix.bushidan.local"
BOT_USER = "admin"
BOT_PASSWORD = "kuwa1998"

# 監視するルーム (alias → room_id)
WATCH_ROOMS = {
    "general":  "!roNaOIGVfsEZlDYEIz:matrix.bushidan.local",
    "strategy": "!vDMJzLMUaeAcSVYNTw:matrix.bushidan.local",
    "ops":      "!qvUHXosmxnHvxHMUpw:matrix.bushidan.local",
}

# ボット自身と各役職アカウントのメッセージは無視する
BOT_USERS = {
    f"@admin:{MATRIX_DOMAIN}",
    f"@daigensui:{MATRIX_DOMAIN}", f"@shogun:{MATRIX_DOMAIN}",
    f"@gunshi:{MATRIX_DOMAIN}",   f"@sanbo:{MATRIX_DOMAIN}",
    f"@gaiji:{MATRIX_DOMAIN}",    f"@uketuke:{MATRIX_DOMAIN}",
    f"@seppou:{MATRIX_DOMAIN}",   f"@kengyo:{MATRIX_DOMAIN}",
    f"@yuhitsu:{MATRIX_DOMAIN}",  f"@onmitsu:{MATRIX_DOMAIN}",
}

# ルーム別にデフォルトの forced_role を設定 (None = 自動ルーティング)
ROOM_FORCED_ROLE = {
    "strategy": "shogun",   # strategy は将軍に
    "ops":      "sanbo",    # ops は参謀に
    "general":  None,       # general は自動ルーティング
}


class MatrixBot:
    """
    Matrix ボット — LangGraph と Matrix を繋ぐブリッジ

    /sync API でリアルタイムにメッセージを受信し、
    LangGraph Router で処理後、担当役職名で返答する。
    """

    def __init__(self):
        self._token: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._since: Optional[str] = None  # sync トークン
        self._running = False
        self._router = None
        self._txn_id = 0
        # HITL 保留中: room_id → {"thread_id": str, "question": str}
        self._pending_hitl: dict = {}

    async def start(self):
        """ボットを起動して Matrix sync ループを開始"""
        connector = aiohttp.TCPConnector(family=2)  # IPv4 only
        self._session = aiohttp.ClientSession(
            base_url=MATRIX_HOMESERVER,
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=35),
        )

        if not await self._login():
            logger.error("❌ Matrix bot ログイン失敗")
            return

        # LangGraph Router 初期化
        from core.langgraph_router import LangGraphRouter
        self._router = LangGraphRouter()
        await self._router.initialize()

        self._running = True
        logger.info("🤖 Matrix bot 起動 — 監視ルーム: %s", list(WATCH_ROOMS.keys()))

        # 初回 sync で since トークンだけ取得 (過去メッセージは処理しない)
        await self._sync(timeout_ms=0)

        # メインループ
        while self._running:
            try:
                await self._sync(timeout_ms=30000)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Matrix sync エラー: %s — 5秒後リトライ", e)
                await asyncio.sleep(5)

    async def stop(self):
        self._running = False
        if self._session:
            await self._session.close()

    async def _login(self) -> bool:
        try:
            async with self._session.post(
                "/_matrix/client/v3/login",
                json={"type": "m.login.password", "user": BOT_USER, "password": BOT_PASSWORD},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                self._token = data["access_token"]
                logger.info("✅ Matrix bot ログイン: @%s:%s", BOT_USER, MATRIX_DOMAIN)
                return True
        except Exception as e:
            logger.error("Matrix login failed: %s", e)
            return False

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    async def _sync(self, timeout_ms: int = 30000):
        """Matrix /sync を呼んで新着メッセージを処理"""
        params = {"timeout": timeout_ms, "full_state": "false"}
        if self._since:
            params["since"] = self._since

        async with self._session.get(
            "/_matrix/client/v3/sync",
            headers=self._auth(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=timeout_ms / 1000 + 5),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        self._since = data.get("next_batch")

        # 各ルームのイベントを処理
        rooms = data.get("rooms", {}).get("join", {})
        for room_id, room_data in rooms.items():
            timeline = room_data.get("timeline", {}).get("events", [])
            for event in timeline:
                if event.get("type") == "m.room.message":
                    await self._handle_message(room_id, event)

    async def _handle_message(self, room_id: str, event: dict):
        """受信メッセージを LangGraph に渡して応答する"""
        sender = event.get("sender", "")
        content = event.get("content", {})
        body = content.get("body", "")
        msgtype = content.get("msgtype", "")

        # ボット自身・役職アカウントのメッセージは無視
        if sender in BOT_USERS:
            return
        if msgtype != "m.text":
            return
        if not body.strip():
            return

        # ルーム名を特定
        channel = next((k for k, v in WATCH_ROOMS.items() if v == room_id), "general")
        forced_role = ROOM_FORCED_ROLE.get(channel)

        logger.info("📩 Matrix[%s] %s: %s", channel, sender, body[:60])
        if _HAS_PROMETHEUS:
            _matrix_recv.labels(channel=channel).inc()

        t0 = time.time()

        # ── HITL 保留中チェック ───────────────────────────────────────
        if room_id in self._pending_hitl:
            pending = self._pending_hitl.pop(room_id)
            await self._handle_hitl_resume(room_id, channel, pending["thread_id"], body)
            return

        # 「考え中...」インジケーター送信
        await self._send_to_room(room_id, "⏳ 処理中...")

        try:
            # ルームごとの固定 thread_id → WebUI と会話履歴を共有
            thread_id = f"room_{channel}"
            result = await self._router.process_message(
                message=body,
                thread_id=thread_id,
                source=f"matrix_{channel}",
                forced_role=forced_role,
                user_id=sender,
            )

            response = result.get("response", "")
            agent_role = result.get("agent_role", "")
            handled_by = result.get("handled_by", "")

            if not response:
                response = "（応答なし）"

            # route (node名) から role_key に変換して担当役職アカウントで返信
            route = result.get("route", "")
            _node_to_role = {
                "groq_qa": "seppou", "gunshi_haiku": "gunshi", "gaiji_rag": "gaiji",
                "sanbo_mcp": "sanbo", "yuhitsu_jp": "yuhitsu", "uketuke_default": "uketuke",
                "onmitsu_local": "onmitsu", "metsuke_proc": "metsuke",
                "shogun_plan": "shogun", "daigensui_audit": "daigensui",
                "kengyo_vision": "kengyo",
            }
            # ── HITL 検出 ──────────────────────────────────────────────
            dialog_status  = result.get("dialog_status", "")
            human_question = result.get("human_question", "")
            if dialog_status == "waiting_for_human" or human_question:
                self._pending_hitl[room_id] = {
                    "thread_id": thread_id,
                    "question":  human_question,
                }
                question_text = human_question or "確認が必要です。回答してください。"
                await self._send_to_room(
                    room_id,
                    f"🙋 **確認が必要です**\n{question_text}\n\n_次のメッセージで回答してください_",
                )
                return

            role_key = _node_to_role.get(route, agent_role or "uketuke")
            await self._send_as_role(room_id, role_key, response)

            elapsed = time.time() - t0
            if _HAS_PROMETHEUS:
                _matrix_sent.labels(channel=channel, role=role_key).inc()
                _matrix_duration.labels(channel=channel).observe(elapsed)

            logger.info(
                "✅ Matrix 応答完了: channel=%s role=%s time=%.1fs",
                channel, agent_role, elapsed
            )

        except Exception as e:
            logger.error("Matrix 応答エラー: %s", e)
            await self._send_to_room(room_id, f"❌ 処理エラー: {e}")

    async def _handle_hitl_resume(
        self, room_id: str, channel: str, thread_id: str, human_response: str
    ):
        """HITL 保留中スレッドを人間の応答で再開する"""
        await self._send_to_room(room_id, "⏳ 再開中...")
        try:
            result = await self._router.resume(
                thread_id=thread_id,
                human_response=human_response,
            )
            response = result.get("response", "（応答なし）")
            route = result.get("route", "")
            _node_to_role = {
                "groq_qa": "seppou", "gunshi_haiku": "gunshi", "gaiji_rag": "gaiji",
                "sanbo_mcp": "sanbo", "yuhitsu_jp": "yuhitsu", "uketuke_default": "uketuke",
                "onmitsu_local": "onmitsu", "metsuke_proc": "metsuke",
                "shogun_plan": "shogun", "daigensui_audit": "daigensui",
                "kengyo_vision": "kengyo",
            }
            agent_role = result.get("agent_role", "")
            role_key = _node_to_role.get(route, agent_role or "uketuke")
            await self._send_as_role(room_id, role_key, response)
            logger.info("✅ HITL resume 完了: thread=%s channel=%s", thread_id[:8], channel)
        except Exception as e:
            logger.error("HITL resume 失敗: %s", e)
            await self._send_to_room(room_id, f"❌ 再開エラー: {e}")

    async def _send_to_room(self, room_id: str, text: str) -> bool:
        """admin アカウントでメッセージ送信"""
        self._txn_id += 1
        room_enc = urllib.parse.quote(room_id, safe="")
        try:
            async with self._session.put(
                f"/_matrix/client/v3/rooms/{room_enc}/send/m.room.message/{self._txn_id}",
                headers=self._auth(),
                json={"msgtype": "m.text", "body": text},
            ) as resp:
                resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("Matrix send failed: %s", e)
            return False

    async def _send_as_role(self, room_id: str, role_key: str, text: str) -> bool:
        """担当役職アカウントで送信 (MatrixClient を使用)"""
        try:
            from utils.matrix_client import MatrixClient
            client = MatrixClient(role_key)
            if not await client.connect():
                # フォールバック: admin で送信
                return await self._send_to_room(room_id, f"[{role_key}] {text}")
            ok = await client.send_message(room_id, text)
            await client.close()
            return ok
        except Exception as e:
            logger.error("send_as_role failed (%s): %s", role_key, e)
            return await self._send_to_room(room_id, f"[{role_key}] {text}")


# ── スタンドアロン起動 ─────────────────────────────────────────────────────

async def run_bot():
    bot = MatrixBot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_bot())
