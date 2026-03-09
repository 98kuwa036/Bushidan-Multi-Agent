#!/usr/bin/env python3
"""
Discord LLM Messenger

各エージェントが個別のDiscordアカウントとしてメッセージを送信

Usage:
    from bushidan.discord_llm_messenger import DiscordLLMMessenger

    messenger = DiscordLLMMessenger()
    await messenger.send_as_agent("shogun", "将軍より：任務を受領しました")
"""
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import aiohttp
from discord import Webhook

from utils.logger import get_logger

logger = get_logger(__name__)


class DiscordLLMMessenger:
    """Discord LLMメッセンジャー - エージェント別メッセージ送信"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルパス（デフォルト: config/discord_llm_accounts.json）
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "discord_llm_accounts.json"

        self.config_path = config_path
        self.accounts: Dict[str, Any] = {}
        self.session: Optional[aiohttp.ClientSession] = None

        self._load_config()

    def _load_config(self):
        """設定ファイルを読み込み"""
        if not self.config_path.exists():
            logger.warning(f"Discord LLMアカウント設定が見つかりません: {self.config_path}")
            logger.info("maintenance/setup_discord_llm_accounts.py を実行してセットアップしてください")
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.accounts = json.load(f)
            logger.info(f"Discord LLMアカウント読み込み完了: {len(self.accounts)}アカウント")
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")

    async def _ensure_session(self):
        """HTTPセッションを確保"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def send_as_agent(
        self,
        agent_id: str,
        message: str,
        username_override: Optional[str] = None,
        avatar_override: Optional[str] = None,
        embed: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        エージェントとしてメッセージを送信

        Args:
            agent_id: エージェントID（shogun, gunshi, karo等）
            message: 送信メッセージ
            username_override: ユーザー名上書き（オプション）
            avatar_override: アバターURL上書き（オプション）
            embed: Embed情報（オプション）

        Returns:
            bool: 送信成功ならTrue
        """
        account = self.accounts.get(agent_id)
        if not account:
            logger.warning(f"エージェントID '{agent_id}' が見つかりません")
            return False

        webhook_url = account.get("webhook_url")
        if not webhook_url:
            logger.error(f"ウェブフックURLがありません: {agent_id}")
            return False

        try:
            await self._ensure_session()

            webhook = Webhook.from_url(webhook_url, session=self.session)

            # デフォルトのアバターURLを生成
            default_avatar = self._generate_avatar_url(account)

            # メッセージ送信
            await webhook.send(
                content=message,
                username=username_override or account['name'],
                avatar_url=avatar_override or default_avatar,
                embed=embed
            )

            logger.debug(f"✓ {account['name']}: メッセージ送信成功")
            return True

        except Exception as e:
            logger.error(f"❌ {account.get('name', agent_id)}: メッセージ送信失敗 - {e}")
            return False

    async def send_with_color(
        self,
        agent_id: str,
        title: str,
        description: str,
        fields: Optional[list] = None
    ) -> bool:
        """
        色付きEmbed付きメッセージを送信

        Args:
            agent_id: エージェントID
            title: タイトル
            description: 説明
            fields: フィールドリスト（オプション）

        Returns:
            bool: 送信成功ならTrue
        """
        account = self.accounts.get(agent_id)
        if not account:
            return False

        from discord import Embed

        embed = Embed(
            title=title,
            description=description,
            color=account['color']
        )

        if fields:
            for field in fields:
                embed.add_field(
                    name=field.get('name', ''),
                    value=field.get('value', ''),
                    inline=field.get('inline', False)
                )

        return await self.send_as_agent(agent_id, "", embed=embed)

    async def send_status_update(
        self,
        agent_id: str,
        status: str,
        details: Optional[str] = None
    ) -> bool:
        """
        ステータス更新を送信

        Args:
            agent_id: エージェントID
            status: ステータス（"開始", "進行中", "完了", "エラー"等）
            details: 詳細情報（オプション）

        Returns:
            bool: 送信成功ならTrue
        """
        account = self.accounts.get(agent_id)
        if not account:
            return False

        status_emoji = {
            "開始": "🚀",
            "進行中": "⏳",
            "完了": "✅",
            "エラー": "❌",
            "警告": "⚠️",
            "情報": "ℹ️"
        }

        emoji = status_emoji.get(status, "📢")
        message = f"{emoji} **{status}**"

        if details:
            message += f"\n{details}"

        return await self.send_as_agent(agent_id, message)

    async def broadcast_to_all(self, message: str) -> Dict[str, bool]:
        """
        全エージェントから同じメッセージを送信

        Args:
            message: 送信メッセージ

        Returns:
            Dict[str, bool]: エージェントIDごとの送信結果
        """
        results = {}

        for agent_id in self.accounts.keys():
            results[agent_id] = await self.send_as_agent(agent_id, message)
            await asyncio.sleep(0.5)  # レート制限対策

        return results

    def _generate_avatar_url(self, account: Dict[str, Any]) -> str:
        """
        アバターURLを生成

        Args:
            account: アカウント情報

        Returns:
            str: アバターURL
        """
        # 絵文字をURLエンコードして背景色付き画像を生成
        emoji = account.get('emoji', '🤖')
        color_hex = f"{account.get('color', 0):06x}"

        # placeholder.comを使用して絵文字アバターを生成
        return f"https://via.placeholder.com/150/{color_hex}/ffffff?text={emoji}"

    async def close(self):
        """セッションをクローズ"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("Discord LLM Messengerセッションをクローズしました")

    async def __aenter__(self):
        """非同期コンテキストマネージャー (enter)"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャー (exit)"""
        await self.close()


# 使用例
async def example_usage():
    """使用例"""
    async with DiscordLLMMessenger() as messenger:
        # 将軍からメッセージ
        await messenger.send_as_agent(
            "shogun",
            "🎌 将軍より：新たな任務を受領しました。各員は持ち場につけ。"
        )

        # 軍師からステータス更新
        await messenger.send_status_update(
            "gunshi",
            "進行中",
            "タスク分解を実施中..."
        )

        # 家老から色付きEmbed
        await messenger.send_with_color(
            "karo",
            "タスク分解完了",
            "以下のサブタスクに分解しました",
            fields=[
                {"name": "サブタスク1", "value": "データ収集", "inline": True},
                {"name": "サブタスク2", "value": "分析処理", "inline": True},
                {"name": "サブタスク3", "value": "レポート作成", "inline": True}
            ]
        )


if __name__ == "__main__":
    # テスト実行
    asyncio.run(example_usage())
