"""
utils/matrix_client.py — Matrix (Synapse) クライアント v2 (REST API 直接実装)

サーバー: 192.168.11.234:8008 (matrix.bushidan.local)
matrix-nio を使わず httpx で直接 Matrix Client-Server API を呼ぶ。

使用方法:
    client = MatrixClient("shogun")
    await client.connect()
    await client.send_message("!roomid:matrix.bushidan.local", "こんにちは")
    await client.close()
"""

import asyncio
import logging
import urllib.parse
from typing import Optional, List, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)

MATRIX_HOMESERVER = "http://192.168.11.234:8008"
MATRIX_DOMAIN = "matrix.bushidan.local"
DEFAULT_PASSWORD = "kuwa1998"

# 各役職のルーム ID (setup_matrix_rooms.py で作成済み)
ROOM_IDS: Dict[str, str] = {
    "general":  "!roNaOIGVfsEZlDYEIz:matrix.bushidan.local",
    "strategy": "!vDMJzLMUaeAcSVYNTw:matrix.bushidan.local",
    "ops":      "!qvUHXosmxnHvxHMUpw:matrix.bushidan.local",
    "intel":    "!ZdGGsoXaMInSONFTew:matrix.bushidan.local",
    "logs":     "!nEBWgDTIuzudaVskXA:matrix.bushidan.local",
}


class MatrixClient:
    """
    Matrix クライアント — シングルロール用

    role_key: "shogun", "daigensui" など
    """

    def __init__(self, role_key: str, password: str = DEFAULT_PASSWORD):
        self.role_key = role_key
        self.user_id = f"@{role_key}:{MATRIX_DOMAIN}"
        self.password = password
        self._token: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._txn_id = 0

    async def connect(self) -> bool:
        connector = aiohttp.TCPConnector(family=2)  # AF_INET (IPv4 only)
        self._session = aiohttp.ClientSession(
            base_url=MATRIX_HOMESERVER,
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        try:
            async with self._session.post(
                "/_matrix/client/v3/login",
                json={"type": "m.login.password", "user": self.role_key, "password": self.password},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                self._token = data["access_token"]
            logger.info("✅ Matrix connected: %s", self.user_id)
            return True
        except Exception as e:
            logger.error("Matrix connect error for %s: %s", self.role_key, e)
            return False

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
        self._token = None

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    async def send_message(self, room_id: str, text: str) -> bool:
        if not self._token or not self._session:
            return False
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
            logger.error("Matrix send error (%s): %s", self.role_key, e)
            return False

    async def send_to_channel(self, channel: str, text: str) -> bool:
        """チャンネル名 (general/strategy/ops/intel/logs) で送信"""
        room_id = ROOM_IDS.get(channel)
        if not room_id:
            logger.warning("Unknown channel: %s", channel)
            return False
        return await self.send_message(room_id, text)

    async def join_room(self, room_id: str) -> bool:
        if not self._token or not self._session:
            return False
        room_enc = urllib.parse.quote(room_id, safe="")
        try:
            async with self._session.post(
                f"/_matrix/client/v3/join/{room_enc}",
                headers=self._auth(),
                json={},
            ) as resp:
                resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("Matrix join error (%s): %s", self.role_key, e)
            return False

    async def get_messages(self, channel: str, limit: int = 20) -> List[Dict[str, Any]]:
        room_id = ROOM_IDS.get(channel)
        if not room_id or not self._token or not self._session:
            return []
        room_enc = urllib.parse.quote(room_id, safe="")
        try:
            async with self._session.get(
                f"/_matrix/client/v3/rooms/{room_enc}/messages",
                headers=self._auth(),
                params={"dir": "b", "limit": limit},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            events = data.get("chunk", [])
            messages = []
            for ev in events:
                if ev.get("type") == "m.room.message":
                    messages.append({
                        "sender": ev.get("sender"),
                        "body": ev.get("content", {}).get("body", ""),
                        "event_id": ev.get("event_id"),
                        "timestamp": ev.get("origin_server_ts"),
                    })
            return messages
        except Exception as e:
            logger.error("Matrix get_messages error: %s", e)
            return []


class MatrixBridge:
    """
    全役職分のクライアントをプールして管理するシングルトン。
    各役職が send_as(role_key, channel, text) で投稿できる。
    """

    ROLES = [
        "daigensui", "shogun", "gunshi", "sanbo", "gaiji",
        "uketuke", "seppou", "kengyo", "yuhitsu", "onmitsu",
    ]

    _instance: Optional["MatrixBridge"] = None

    def __init__(self):
        self._clients: Dict[str, MatrixClient] = {}
        self._connected = False

    @classmethod
    def get(cls) -> "MatrixBridge":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def connect_all(self):
        tasks = [MatrixClient(r) for r in self.ROLES]
        self._clients = {r: c for r, c in zip(self.ROLES, tasks)}
        results = await asyncio.gather(
            *[c.connect() for c in self._clients.values()],
            return_exceptions=True,
        )
        ok = sum(1 for r in results if r is True)
        logger.info("Matrix bridge: %d/%d roles connected", ok, len(self.ROLES))
        self._connected = True

    async def send_as(self, role_key: str, channel: str, text: str) -> bool:
        client = self._clients.get(role_key)
        if not client:
            # 遅延初期化
            client = MatrixClient(role_key)
            if not await client.connect():
                return False
            self._clients[role_key] = client
        return await client.send_to_channel(channel, text)

    async def close_all(self):
        await asyncio.gather(*[c.close() for c in self._clients.values()], return_exceptions=True)
        self._clients.clear()
        self._connected = False

    def get_client(self, role_key: str) -> Optional[MatrixClient]:
        return self._clients.get(role_key)
