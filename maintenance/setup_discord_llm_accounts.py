#!/usr/bin/env python3
"""
Discord LLMアカウント作成スクリプト

各LLMエージェント用のDiscordウェブフック/スレッドアカウントを設定

Usage:
    python maintenance/setup_discord_llm_accounts.py --create-webhooks
    python maintenance/setup_discord_llm_accounts.py --list
    python maintenance/setup_discord_llm_accounts.py --test
"""
import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# プロジェクトルート
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env読み込み
load_dotenv(project_root / ".env")


class DiscordLLMSetup:
    """Discord LLMアカウントセットアップ"""

    def __init__(self):
        self.token = os.getenv("DISCORD_BOT_TOKEN")
        if not self.token:
            raise ValueError("DISCORD_BOT_TOKEN が .env に設定されていません")

        self.config_path = project_root / "config" / "discord_llm_accounts.json"

        # 武士団エージェント定義
        self.agents = {
            "shogun": {
                "name": "将軍 (Shogun)",
                "description": "戦略的意思決定・品質保証",
                "model": "Claude Sonnet 4.6",
                "avatar_emoji": "🎌",
                "color": 0x9B59B6  # Purple
            },
            "gunshi": {
                "name": "軍師 (Gunshi)",
                "description": "作戦立案・PDCA Engine",
                "model": "Qwen3-Coder-Next",
                "avatar_emoji": "📋",
                "color": 0x3498DB  # Blue
            },
            "karo": {
                "name": "家老 (Karo)",
                "description": "タスク分解・調整",
                "model": "Gemini 3 Flash",
                "avatar_emoji": "👔",
                "color": 0x2ECC71  # Green
            },
            "taisho": {
                "name": "大将 (Taisho)",
                "description": "実装層・MCP駆使",
                "model": "Local Qwen3",
                "avatar_emoji": "⚔️",
                "color": 0xE74C3C  # Red
            },
            "yohei": {
                "name": "傭兵 (Yohei/Kimi)",
                "description": "並列サブタスク実行",
                "model": "Kimi K2.5",
                "avatar_emoji": "🗡️",
                "color": 0xF39C12  # Orange
            },
            "kengyo": {
                "name": "検校 (Kengyo)",
                "description": "ビジュアル検証",
                "model": "Kimi Vision",
                "avatar_emoji": "👁️",
                "color": 0x1ABC9C  # Teal
            },
            "groq": {
                "name": "足軽-Groq (Ashigaru)",
                "description": "Simple タスク高速処理",
                "model": "Llama 3.3 70B",
                "avatar_emoji": "⚡",
                "color": 0x95A5A6  # Gray
            }
        }

        self.accounts = {}
        self._load_existing_config()

    def _load_existing_config(self):
        """既存の設定を読み込み"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.accounts = json.load(f)
                print(f"既存の設定を読み込みました: {len(self.accounts)}アカウント")
            except Exception as e:
                print(f"設定読み込みエラー: {e}")

    def _save_config(self):
        """設定を保存"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.accounts, f, ensure_ascii=False, indent=2)
            print(f"✓ 設定を保存しました: {self.config_path}")
        except Exception as e:
            print(f"❌ 設定保存エラー: {e}")

    async def create_webhooks(self, channel_id: int):
        """ウェブフックを作成"""
        try:
            import discord

            intents = discord.Intents.default()
            client = discord.Client(intents=intents)

            @client.event
            async def on_ready():
                print(f"✓ Botログイン成功: {client.user}")

                try:
                    channel = client.get_channel(channel_id)
                    if not channel:
                        print(f"❌ チャンネルが見つかりません: {channel_id}")
                        await client.close()
                        return

                    print(f"チャンネル: #{channel.name}")
                    print()

                    # 各エージェント用のウェブフックを作成
                    for agent_id, agent_info in self.agents.items():
                        print(f"作成中: {agent_info['name']}...")

                        try:
                            webhook = await channel.create_webhook(
                                name=agent_info['name'],
                                reason=f"{agent_info['name']}専用ウェブフック"
                            )

                            self.accounts[agent_id] = {
                                "webhook_url": webhook.url,
                                "webhook_id": webhook.id,
                                "name": agent_info['name'],
                                "description": agent_info['description'],
                                "model": agent_info['model'],
                                "emoji": agent_info['avatar_emoji'],
                                "color": agent_info['color'],
                                "created_at": discord.utils.utcnow().isoformat()
                            }

                            print(f"  ✓ ウェブフック作成完了: {webhook.id}")

                        except discord.HTTPException as e:
                            print(f"  ❌ ウェブフック作成失敗: {e}")

                        # レート制限対策
                        await asyncio.sleep(1)

                    self._save_config()
                    print(f"\n✓ {len(self.accounts)}個のウェブフックを作成しました")

                except Exception as e:
                    print(f"❌ ウェブフック作成エラー: {e}")
                finally:
                    await client.close()

            await client.start(self.token)

        except ImportError:
            print("❌ discord.py がインストールされていません")
            print("   pip install discord.py")
        except Exception as e:
            print(f"❌ エラー: {e}")

    async def test_webhooks(self):
        """ウェブフックをテスト"""
        if not self.accounts:
            print("❌ 設定されたアカウントがありません")
            return

        print("ウェブフックテスト中...")
        print()

        for agent_id, account in self.accounts.items():
            webhook_url = account.get("webhook_url")
            if not webhook_url:
                print(f"⚠️  {account['name']}: ウェブフックURLなし")
                continue

            try:
                import discord
                from discord import Webhook
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    webhook = Webhook.from_url(webhook_url, session=session)

                    await webhook.send(
                        content=f"テストメッセージ: {account['name']} ({account['model']})",
                        username=account['name'],
                        avatar_url=f"https://via.placeholder.com/150/{account['color']:06x}/ffffff?text={account['emoji']}"
                    )

                    print(f"✓ {account['name']}: テスト送信成功")

            except Exception as e:
                print(f"❌ {account['name']}: テスト失敗 - {e}")

            await asyncio.sleep(1)

    def list_accounts(self):
        """アカウント一覧を表示"""
        if not self.accounts:
            print("設定されたアカウントがありません")
            return

        print("=" * 80)
        print("Discord LLMアカウント一覧")
        print("=" * 80)
        print()

        for agent_id, account in self.accounts.items():
            print(f"{account['emoji']} {account['name']}")
            print(f"  ID: {agent_id}")
            print(f"  説明: {account['description']}")
            print(f"  モデル: {account['model']}")
            print(f"  ウェブフックID: {account.get('webhook_id', 'N/A')}")
            print(f"  作成日: {account.get('created_at', 'N/A')}")
            print()

        print(f"総アカウント数: {len(self.accounts)}")

    def generate_integration_code(self):
        """統合コードを生成"""
        print("=" * 80)
        print("統合コード例")
        print("=" * 80)
        print()
        print("# bushidan/discord_reporter.py に以下を追加:")
        print()
        print("```python")
        print("import json")
        print("from pathlib import Path")
        print()
        print("class DiscordLLMReporter:")
        print("    def __init__(self):")
        print("        config_path = Path('config/discord_llm_accounts.json')")
        print("        with open(config_path, 'r') as f:")
        print("            self.accounts = json.load(f)")
        print()
        print("    async def send_as_agent(self, agent_id: str, message: str):")
        print("        account = self.accounts.get(agent_id)")
        print("        if not account:")
        print("            return")
        print()
        print("        webhook = Webhook.from_url(account['webhook_url'], session=self.session)")
        print("        await webhook.send(")
        print("            content=message,")
        print("            username=account['name'],")
        print("            avatar_url=f\"https://via.placeholder.com/150/{account['color']:06x}/ffffff?text={account['emoji']}\"")
        print("        )")
        print("```")
        print()


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Discord LLMアカウントセットアップ")
    parser.add_argument("--create-webhooks", type=int, metavar="CHANNEL_ID",
                        help="ウェブフックを作成（チャンネルIDを指定）")
    parser.add_argument("--test", action="store_true", help="ウェブフックをテスト")
    parser.add_argument("--list", action="store_true", help="アカウント一覧を表示")
    parser.add_argument("--code", action="store_true", help="統合コード例を表示")
    args = parser.parse_args()

    try:
        setup = DiscordLLMSetup()

        if args.create_webhooks:
            print("=" * 80)
            print("Discord LLMアカウント作成")
            print("=" * 80)
            print()
            await setup.create_webhooks(args.create_webhooks)

        elif args.test:
            await setup.test_webhooks()

        elif args.list:
            setup.list_accounts()

        elif args.code:
            setup.generate_integration_code()

        else:
            parser.print_help()
            print()
            print("使用例:")
            print("  # ウェブフック作成（チャンネルID: 1234567890）")
            print("  python maintenance/setup_discord_llm_accounts.py --create-webhooks 1234567890")
            print()
            print("  # アカウント一覧表示")
            print("  python maintenance/setup_discord_llm_accounts.py --list")
            print()
            print("  # ウェブフックテスト")
            print("  python maintenance/setup_discord_llm_accounts.py --test")
            print()
            print("  # 統合コード例表示")
            print("  python maintenance/setup_discord_llm_accounts.py --code")

    except ValueError as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
