"""
utils/rabbitmq_client.py — RabbitMQ 役職間メッセージングクライアント v1

武士団の役職間非同期通信を担う。
ブローカー: 192.168.11.231:5672 (pct100 localhost)

Exchange 設計:
  bushidan.direct   — 特定役職への直接メッセージ (routing_key = role_key)
  bushidan.fanout   — 全役職へのブロードキャスト
  bushidan.topic    — トピック別メッセージ (routing_key = "category.role")

Queue:
  役職ごとに専用キュー: bushidan.q.<role_key>

使用例:
    bus = RabbitMQBus.get()
    await bus.connect()

    # 特定役職へ送信
    await bus.send_to_role("shogun", {"type": "task", "content": "分析してください"})

    # 全役職へブロードキャスト
    await bus.broadcast({"type": "alert", "content": "システム通知"})

    # メッセージ購読
    await bus.subscribe("shogun", my_callback)
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional

import aio_pika

logger = logging.getLogger(__name__)

RABBITMQ_URL = "amqp://bushidan:kuwa1998@192.168.11.231:5672/"

EXCHANGE_DIRECT  = "bushidan.direct"
EXCHANGE_FANOUT  = "bushidan.fanout"
EXCHANGE_TOPIC   = "bushidan.topic"

ROLES = [
    "daigensui", "shogun", "gunshi", "sanbo", "gaiji",
    "uketuke", "seppou", "kengyo", "yuhitsu", "onmitsu",
]


def _queue_name(role_key: str) -> str:
    return f"bushidan.q.{role_key}"


class RabbitMQBus:
    """
    武士団 メッセージバス — シングルトン

    役職間の非同期メッセージングを管理する。
    """

    _instance: Optional["RabbitMQBus"] = None

    def __init__(self):
        self._connection: Optional[aio_pika.abc.AbstractRobustConnection] = None
        self._channel: Optional[aio_pika.abc.AbstractChannel] = None
        self._exchanges: Dict[str, aio_pika.abc.AbstractExchange] = {}
        self._queues: Dict[str, aio_pika.abc.AbstractQueue] = {}
        self._consumer_tags: Dict[str, str] = {}
        self._connected = False

    @classmethod
    def get(cls) -> "RabbitMQBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def connect(self) -> bool:
        """RabbitMQ に接続してエクスチェンジ・キューをセットアップ"""
        try:
            self._connection = await aio_pika.connect_robust(RABBITMQ_URL)
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=10)

            # Exchange 宣言
            self._exchanges[EXCHANGE_DIRECT] = await self._channel.declare_exchange(
                EXCHANGE_DIRECT, aio_pika.ExchangeType.DIRECT, durable=True
            )
            self._exchanges[EXCHANGE_FANOUT] = await self._channel.declare_exchange(
                EXCHANGE_FANOUT, aio_pika.ExchangeType.FANOUT, durable=True
            )
            self._exchanges[EXCHANGE_TOPIC] = await self._channel.declare_exchange(
                EXCHANGE_TOPIC, aio_pika.ExchangeType.TOPIC, durable=True
            )

            # 全役職のキューを宣言・バインド
            for role in ROLES:
                q = await self._channel.declare_queue(
                    _queue_name(role), durable=True
                )
                self._queues[role] = q
                # direct exchange にバインド (routing_key = role_key)
                await q.bind(self._exchanges[EXCHANGE_DIRECT], routing_key=role)
                # fanout exchange にバインド (全員受信)
                await q.bind(self._exchanges[EXCHANGE_FANOUT])
                # topic exchange にバインド (role.# パターン)
                await q.bind(self._exchanges[EXCHANGE_TOPIC], routing_key=f"{role}.#")

            self._connected = True
            logger.info("✅ RabbitMQ 接続完了: %s", RABBITMQ_URL.split("@")[-1])
            return True
        except Exception as e:
            logger.error("❌ RabbitMQ 接続失敗: %s", e)
            return False

    async def close(self):
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        self._connected = False

    def _encode(self, payload: Any) -> bytes:
        return json.dumps(payload, ensure_ascii=False).encode()

    async def send_to_role(
        self,
        role_key: str,
        payload: Any,
        priority: int = 0,
    ) -> bool:
        """特定役職へ直接メッセージ送信"""
        if not self._connected:
            return False
        try:
            exchange = self._exchanges[EXCHANGE_DIRECT]
            msg = aio_pika.Message(
                body=self._encode(payload),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                priority=priority,
            )
            await exchange.publish(msg, routing_key=role_key)
            logger.debug("→ %s: %s", role_key, str(payload)[:80])
            return True
        except Exception as e:
            logger.error("RabbitMQ send_to_role failed: %s", e)
            return False

    async def broadcast(self, payload: Any) -> bool:
        """全役職へブロードキャスト"""
        if not self._connected:
            return False
        try:
            exchange = self._exchanges[EXCHANGE_FANOUT]
            msg = aio_pika.Message(
                body=self._encode(payload),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )
            await exchange.publish(msg, routing_key="")
            logger.debug("broadcast: %s", str(payload)[:80])
            return True
        except Exception as e:
            logger.error("RabbitMQ broadcast failed: %s", e)
            return False

    async def publish_topic(self, routing_key: str, payload: Any) -> bool:
        """トピック送信 (例: 'shogun.task', 'all.alert')"""
        if not self._connected:
            return False
        try:
            exchange = self._exchanges[EXCHANGE_TOPIC]
            msg = aio_pika.Message(
                body=self._encode(payload),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )
            await exchange.publish(msg, routing_key=routing_key)
            return True
        except Exception as e:
            logger.error("RabbitMQ publish_topic failed: %s", e)
            return False

    async def subscribe(
        self,
        role_key: str,
        callback: Callable[[Dict], Any],
        no_ack: bool = False,
    ) -> bool:
        """役職キューを購読してメッセージを callback に渡す"""
        queue = self._queues.get(role_key)
        if not queue:
            logger.warning("キューが見つかりません: %s", role_key)
            return False

        async def _on_message(message: aio_pika.abc.AbstractIncomingMessage):
            async with message.process(ignore_processed=True):
                try:
                    payload = json.loads(message.body)
                    await callback(payload)
                except Exception as e:
                    logger.error("RabbitMQ callback error (%s): %s", role_key, e)

        await queue.consume(_on_message, no_ack=no_ack)
        logger.info("✅ 購読開始: %s", role_key)
        return True

    async def get_queue_stats(self) -> Dict[str, int]:
        """各キューのメッセージ数を返す"""
        stats = {}
        for role in ROLES:
            try:
                q = await self._channel.declare_queue(
                    _queue_name(role), durable=True, passive=True
                )
                stats[role] = q.declaration_result.message_count
            except Exception:
                stats[role] = -1
        return stats
