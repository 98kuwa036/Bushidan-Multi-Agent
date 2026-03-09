"""
Bushidan Multi-Agent Discord Reporter

Enables multi-agent presence in Discord through webhook-based reporting.
Each agent (将軍, 軍師, 家老, 大将, 検校, 足軽) can report progress to
Discord threads, creating a "war council" (合戦会議) experience.

Architecture:
- 将軍 (Shogun): Real bot account (BushidanDiscordBot)
- Other agents: Webhooks with custom names/emojis
- Thread-based: Each task gets dedicated thread
- Rich visualization: Embeds, reactions, delegation chains
"""

import discord
import asyncio
import time
from typing import Optional, Dict, Any, TYPE_CHECKING, List
from datetime import datetime
from dataclasses import dataclass, field

from utils.logger import get_logger

if TYPE_CHECKING:
    from bushidan.discord_bot import BushidanDiscordBot

logger = get_logger(__name__)


# =============================================================================
# Agent Configuration
# =============================================================================

AGENT_CONFIG = {
    "shogun": {
        "display_name": "将軍 (Shogun)",
        "emoji": "🎌",
        "color": 0x8B0000,  # Dark red
        "role_description": "戦略的意思決定",
    },
    "gunshi": {
        "display_name": "軍師 (Gunshi)",
        "emoji": "🧠",
        "color": 0x4B0082,  # Indigo
        "role_description": "PDCA作戦立案",
    },
    "karo": {
        "display_name": "家老 (Karo)",
        "emoji": "👔",
        "color": 0x006400,  # Dark green
        "role_description": "戦術調整",
    },
    "taisho": {
        "display_name": "大将 (Taisho)",
        "emoji": "⚔️",
        "color": 0xFF4500,  # Orange red
        "role_description": "実装実行",
    },
    "kengyo": {
        "display_name": "検校 (Kengyo)",
        "emoji": "👁️",
        "color": 0x9400D3,  # Dark violet
        "role_description": "ビジュアル検証",
    },
    "ashigaru": {
        "display_name": "足軽 (Ashigaru)",
        "emoji": "👣",
        "color": 0x808080,  # Gray
        "role_description": "MCP実行",
    },
    "langgraph": {
        "display_name": "LangGraph Router",
        "emoji": "🔗",
        "color": 0x1E90FF,  # Dodger blue
        "role_description": "タスク分析・ルーティング",
    },
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TaskThread:
    """
    Tracks Discord thread state for a task.

    Attributes:
        task_id: Unique task identifier
        thread: Discord thread object
        original_message: User's original message
        created_at: Thread creation timestamp
        status_message: Initial status message with reactions
        delegation_chain: List of agents involved (ordered)
        active_agent: Currently active agent
        completed: bool = False
        update_lock: Async lock for thread updates
        total_reports: Number of progress reports
        agents_involved: Set of all agents that participated
        paused: Whether task is paused awaiting approval
        pending_approval: Current pending approval request
        user_messages: List of user messages in this thread
    """
    task_id: str
    thread: discord.Thread
    original_message: discord.Message
    created_at: datetime
    status_message: discord.Message
    delegation_chain: list[str] = field(default_factory=list)
    active_agent: Optional[str] = None
    completed: bool = False
    update_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    total_reports: int = 0
    agents_involved: set[str] = field(default_factory=set)
    paused: bool = False
    pending_approval: Optional[Any] = None  # ApprovalRequest (avoid circular import)
    user_messages: list[Any] = field(default_factory=list)  # List[discord.Message]


# =============================================================================
# Webhook Pool Manager
# =============================================================================

class WebhookPoolManager:
    """
    Manages webhook pool for agent reporting.

    Creates and caches webhooks per channel to avoid rate limits.
    Handles webhook deletion and recreation gracefully.

    Supports three modes:
    1. LLM Accounts mode: Use dedicated webhooks per agent from discord_llm_accounts.json
    2. Auto-create mode: Bot creates webhooks automatically (requires Manage Webhooks permission)
    3. Manual mode: Use pre-existing webhook URLs from environment variables
    """

    def __init__(self):
        self.webhooks: Dict[int, discord.Webhook] = {}
        self.webhook_lock = asyncio.Lock()
        self.manual_webhook_url: Optional[str] = None
        self.llm_accounts: Dict[str, Any] = {}
        self.llm_webhooks: Dict[str, discord.Webhook] = {}
        self._aiohttp_session: Optional[Any] = None
        self._load_manual_webhook()
        self._load_llm_accounts()

    def _load_manual_webhook(self) -> None:
        """Load manual webhook URL from environment variable if available."""
        import os
        self.manual_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if self.manual_webhook_url:
            logger.info("✅ Manual webhook URL loaded from DISCORD_WEBHOOK_URL")
            logger.info("ℹ️ Using manual webhook mode (no auto-creation)")

    def _load_llm_accounts(self) -> None:
        """Load LLM accounts configuration from discord_llm_accounts.json"""
        import json
        import os
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "config" / "discord_llm_accounts.json"

        if not config_path.exists():
            logger.debug("ℹ️ LLM accounts config not found (optional)")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.llm_accounts = json.load(f)
            logger.info(f"✅ LLM accounts loaded: {len(self.llm_accounts)} agents")
            logger.info("ℹ️ Using LLM accounts mode for agent-specific webhooks")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load LLM accounts: {e}")

    async def get_webhook(
        self,
        channel: discord.TextChannel
    ) -> discord.Webhook:
        """
        Get or create webhook for channel.

        Args:
            channel: Discord text channel

        Returns:
            Discord webhook instance

        Raises:
            discord.Forbidden: If bot lacks webhook permissions and no manual webhook configured
        """
        async with self.webhook_lock:
            # Manual webhook mode
            if self.manual_webhook_url:
                if channel.id not in self.webhooks:
                    webhook = discord.Webhook.from_url(
                        self.manual_webhook_url,
                        session=channel._state.http._HTTPClient__session
                    )
                    self.webhooks[channel.id] = webhook
                    logger.info(f"✅ Using manual webhook for channel #{channel.name}")
                return self.webhooks[channel.id]

            # Auto-create mode
            if channel.id in self.webhooks:
                webhook = self.webhooks[channel.id]
                # Verify webhook still exists
                try:
                    await webhook.fetch()
                    return webhook
                except discord.NotFound:
                    # Webhook deleted, remove from cache
                    del self.webhooks[channel.id]
                    logger.warning(f"⚠️ Webhook for channel {channel.name} was deleted, recreating...")

            # Create new webhook
            try:
                webhook = await channel.create_webhook(
                    name="Bushidan Multi-Agent",
                    reason="Multi-agent task reporting system"
                )
                self.webhooks[channel.id] = webhook
                logger.info(f"✅ Created webhook for channel #{channel.name}")
                return webhook
            except discord.Forbidden:
                logger.error(f"❌ No permission to create webhook in #{channel.name}")
                logger.error(f"💡 解決策:")
                logger.error(f"   1. Bot に 'Manage Webhooks' 権限を追加")
                logger.error(f"   2. または手動でWebhookを作成して DISCORD_WEBHOOK_URL に設定")
                logger.error(f"      Discordチャンネル設定 → 連携サービス → ウェブフック → 新しいウェブフック")
                raise

    async def send_as_agent(
        self,
        channel: discord.TextChannel,
        agent_name: str,
        message: str = "",
        thread: Optional[discord.Thread] = None,
        embed: Optional[discord.Embed] = None
    ) -> None:
        """
        Send message as specific agent via webhook.

        Prioritizes LLM account webhooks if available, falls back to standard webhook.

        Args:
            channel: Parent channel (where webhook lives)
            agent_name: Agent identifier (e.g., "shogun", "gunshi")
            message: Message content (optional if embed provided)
            thread: Thread to post in (optional)
            embed: Rich embed (optional)
        """
        try:
            logger.info(f"📤 send_as_agent called: agent={agent_name}, has_llm_accounts={len(self.llm_accounts)}")

            # Priority 1: Use dedicated LLM account webhook if available
            llm_account = self.llm_accounts.get(agent_name)
            logger.info(f"🔍 LLM account lookup for {agent_name}: {'Found' if llm_account else 'Not found'}")
            if llm_account:
                webhook_url = llm_account.get("webhook_url")
                logger.info(f"🔗 webhook_url for {agent_name}: {'exists' if webhook_url else 'missing'}")
                if webhook_url:
                    # Get or create webhook instance
                    if agent_name not in self.llm_webhooks:
                        logger.info(f"🔨 Creating webhook instance for {agent_name}")
                        import aiohttp

                        # Create aiohttp session if not exists
                        if self._aiohttp_session is None:
                            self._aiohttp_session = aiohttp.ClientSession()
                            logger.info("✅ Created aiohttp session for webhooks")

                        # discord.py v2.0+ uses session parameter directly
                        self.llm_webhooks[agent_name] = discord.Webhook.from_url(
                            webhook_url,
                            session=self._aiohttp_session
                        )
                        logger.info(f"✅ Webhook instance created for {agent_name}")

                    webhook = self.llm_webhooks[agent_name]
                    display_name = llm_account.get("name", agent_name)
                    emoji = llm_account.get('emoji', '🤖')

                    # Add emoji to display name for visual identification
                    # Note: avatar_url removed because placeholder.com is unreachable
                    display_name_with_emoji = f"{emoji} {display_name}"

                    logger.info(f"🎯 Using LLM account webhook for {agent_name}: {display_name_with_emoji}")
                    await webhook.send(
                        content=message if message else None,
                        username=display_name_with_emoji,
                        embed=embed,
                        thread=thread,
                        wait=False
                    )
                    logger.info(f"✓ Sent via LLM account webhook: {agent_name}")
                    return

            # Priority 2: Fall back to standard webhook
            webhook = await self.get_webhook(channel)

            config = AGENT_CONFIG.get(agent_name, {})
            display_name = config.get("display_name", agent_name)

            # Note: Webhook avatars require hosted image URLs
            # For now, we use emoji in the username

            logger.info(f"⚠️ Fallback to standard webhook for {agent_name}")
            await webhook.send(
                content=message if message else None,
                username=display_name,
                embed=embed,
                thread=thread,
                wait=False
            )
            logger.info(f"✓ Sent via standard webhook: {agent_name}")
        except Exception as e:
            logger.error(f"❌ Failed to send webhook message as {agent_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")


# =============================================================================
# Discord Agent Reporter
# =============================================================================

class DiscordAgentReporter:
    """
    Enables agents to report progress to Discord threads.

    Provides a unified interface for all agents to report:
    - Task start
    - Progress updates
    - Task delegation
    - Completion/errors

    Uses webhooks to simulate multiple users in Discord.
    """

    def __init__(self, bot: "BushidanDiscordBot"):
        """
        Initialize reporter.

        Args:
            bot: BushidanDiscordBot instance
        """
        self.bot = bot
        self.webhook_manager = bot.webhook_manager
        self.agent_config = AGENT_CONFIG
        self._load_speaking_styles()

    def _load_speaking_styles(self) -> None:
        """Load speaking styles from config"""
        try:
            import yaml
            import os
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'interactive_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.speaking_styles = config.get('interactive_mode', {}).get('speaking_style', {}).get('agent_styles', {})
                    self.speaking_enabled = config.get('interactive_mode', {}).get('speaking_style', {}).get('enabled', False)
            else:
                self.speaking_styles = {}
                self.speaking_enabled = False
        except Exception as e:
            logger.warning(f"⚠️ Failed to load speaking styles: {e}")
            self.speaking_styles = {}
            self.speaking_enabled = False

    def _get_speaking_style(self, agent_name: str, style_key: str) -> str:
        """
        Get speaking style for agent.

        Args:
            agent_name: Agent identifier
            style_key: Style key (greeting, approval_request, approved, rejected)

        Returns:
            Speaking style text or empty string
        """
        if not self.speaking_enabled:
            return ""

        agent_styles = self.speaking_styles.get(agent_name, {})
        return agent_styles.get(style_key, "")

    async def report_start(
        self,
        task_id: str,
        agent_name: str,
        message: str
    ) -> None:
        """
        Report that agent has started working on task.

        Args:
            task_id: Task identifier
            agent_name: Agent identifier (e.g., "shogun")
            message: Start message to display
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            logger.warning(f"⚠️ Task thread not found: {task_id}")
            return

        async with thread_data.update_lock:
            # Update delegation chain
            if agent_name not in thread_data.delegation_chain:
                thread_data.delegation_chain.append(agent_name)
            thread_data.active_agent = agent_name
            thread_data.agents_involved.add(agent_name)
            thread_data.total_reports += 1

            # Get agent config
            config = self.agent_config.get(agent_name, {})
            emoji = config.get("emoji", "🤖")
            display_name = config.get("display_name", agent_name)
            role_desc = config.get("role_description", "")
            color = config.get("color", 0x808080)

            # 戦国武将風の挨拶を追加
            greeting = self._get_speaking_style(agent_name, "greeting")

            # Create embed with detailed agent info
            embed = discord.Embed(
                title=f"{emoji} {display_name}",
                description=f"**役割:** {role_desc}\n{greeting}\n\n{message}",
                color=color,
                timestamp=datetime.now()
            )
            # Set author to make agent name prominent
            embed.set_author(name=f"{emoji} {display_name}")

            # Send via webhook
            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=agent_name,
                message="",
                thread=thread_data.thread,
                embed=embed
            )

    async def report_progress(
        self,
        task_id: str,
        message: str,
        progress: Optional[float] = None
    ) -> None:
        """
        Report progress update.

        Args:
            task_id: Task identifier
            message: Progress message
            progress: Optional progress value (0.0 to 1.0)
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            return

        async with thread_data.update_lock:
            agent_name = thread_data.active_agent
            if not agent_name:
                logger.warning(f"⚠️ No active agent for task {task_id}")
                return

            thread_data.total_reports += 1

            config = self.agent_config.get(agent_name, {})
            emoji = config.get("emoji", "🤖")
            display_name = config.get("display_name", agent_name)

            # 絵文字 + エージェント名 + メッセージ
            content = f"{emoji} **{display_name}**\n{message}"
            if progress is not None:
                progress_bar = self._create_progress_bar(progress)
                content += f"\n{progress_bar}"

            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=agent_name,
                message=content,
                thread=thread_data.thread
            )

    async def report_complete(
        self,
        task_id: str,
        result: str
    ) -> None:
        """
        Report task completion.

        Args:
            task_id: Task identifier
            result: Completion result summary
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            return

        async with thread_data.update_lock:
            agent_name = thread_data.active_agent
            if not agent_name:
                return

            thread_data.total_reports += 1

            config = self.agent_config.get(agent_name, {})
            emoji = config.get("emoji", "🤖")
            display_name = config.get("display_name", agent_name)

            # 戦国武将風の完了報告
            completed_style = self._get_speaking_style(agent_name, "approved")
            description_text = f"{completed_style}\n\n{result[:1000]}"  # Discord limit

            embed = discord.Embed(
                title=f"{emoji} {display_name} - 任務完了",
                description=description_text,
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            # Set author to show completing agent
            embed.set_author(name=f"{emoji} {display_name}")

            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=agent_name,
                message="",
                thread=thread_data.thread,
                embed=embed
            )

    async def report_error(
        self,
        task_id: str,
        error: str
    ) -> None:
        """
        Report error.

        Args:
            task_id: Task identifier
            error: Error message
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            return

        async with thread_data.update_lock:
            agent_name = thread_data.active_agent
            if not agent_name:
                return

            thread_data.total_reports += 1

            config = self.agent_config.get(agent_name, {})
            emoji = config.get("emoji", "🤖")
            display_name = config.get("display_name", agent_name)

            # Truncate error for Discord
            error_text = error[:500] if len(error) > 500 else error

            # 戦国武将風のエラー報告
            embed = discord.Embed(
                title=f"{emoji} {display_name} - 不調が生じ申した",
                description=f"不具合の詳細:\n```\n{error_text}\n```",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )

            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=agent_name,
                message="",
                thread=thread_data.thread,
                embed=embed
            )

    async def report_delegation(
        self,
        task_id: str,
        from_agent: str,
        to_agent: str,
        reason: Optional[str] = None
    ) -> None:
        """
        Report task delegation.

        Args:
            task_id: Task identifier
            from_agent: Agent delegating the task
            to_agent: Agent receiving the task
            reason: Optional delegation reason
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            return

        async with thread_data.update_lock:
            # Update delegation chain
            if to_agent not in thread_data.delegation_chain:
                thread_data.delegation_chain.append(to_agent)
            thread_data.agents_involved.add(to_agent)
            thread_data.total_reports += 1

            from_config = self.agent_config.get(from_agent, {})
            to_config = self.agent_config.get(to_agent, {})

            from_emoji = from_config.get("emoji", "🤖")
            from_name = from_config.get("display_name", from_agent)
            to_emoji = to_config.get("emoji", "🤖")
            to_name = to_config.get("display_name", to_agent)

            # 戦国武将風の委譲表現
            description = f"**委譲元:** {from_emoji} {from_name}\n**委譲先:** {to_emoji} {to_name}"
            if reason:
                description += f"\n\n💭 **申し送り:** {reason}"

            embed = discord.Embed(
                title="🔄 任務の申し送り",
                description=description,
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            # Set author to show delegating agent
            embed.set_author(name=f"{from_emoji} {from_name}")

            # Show delegation chain with names
            chain_str = " → ".join([
                f"{self.agent_config.get(a, {}).get('emoji', '🤖')} {self.agent_config.get(a, {}).get('display_name', a)}"
                for a in thread_data.delegation_chain
            ])
            embed.add_field(name="📜 采配の系譜", value=chain_str, inline=False)

            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=from_agent,
                message="",
                thread=thread_data.thread,
                embed=embed
            )

    async def report_battle_start(
        self,
        task_id: str,
        task_content: str,
        strategy: Optional[str] = None
    ) -> None:
        """
        Report battle start with detailed strategy.

        Args:
            task_id: Task identifier
            task_content: Task content
            strategy: Battle strategy description
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            return

        async with thread_data.update_lock:
            # 将軍が合戦を開始
            embed = discord.Embed(
                title="⚔️ 合戦開始",
                description=f"**任務内容:**\n{task_content[:500]}",
                color=discord.Color.dark_red(),
                timestamp=datetime.now()
            )

            if strategy:
                embed.add_field(name="🎯 戦略", value=strategy, inline=False)

            embed.add_field(
                name="🏯 武士団配置",
                value=(
                    "🎌 将軍 (Shogun) - 戦略的意思決定\n"
                    "🧠 軍師 (Gunshi) - PDCA作戦立案\n"
                    "👔 家老 (Karo) - 戦術調整\n"
                    "⚔️ 大将 (Taisho) - 実装実行\n"
                    "👁️ 検校 (Kengyo) - ビジュアル検証\n"
                    "👣 足軽 (Ashigaru) - MCP実行"
                ),
                inline=False
            )

            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name="shogun",
                message="",
                thread=thread_data.thread,
                embed=embed
            )

    async def report_routing_analysis(
        self,
        task_id: str,
        analysis_result: Dict[str, Any]
    ) -> None:
        """
        Report LangGraph routing analysis results.

        Args:
            task_id: Task identifier
            analysis_result: Analysis result dict
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            return

        async with thread_data.update_lock:
            embed = discord.Embed(
                title="🔗 LangGraph タスク分析",
                description="タスクの複雑度とルーティングを分析しました",
                color=0x1E90FF,  # Dodger blue
                timestamp=datetime.now()
            )

            # 分析結果を詳細表示
            complexity = analysis_result.get("complexity", "unknown")
            is_multi_step = analysis_result.get("is_multi_step", False)
            is_action_task = analysis_result.get("is_action_task", False)
            is_simple_qa = analysis_result.get("is_simple_qa", False)
            confidence = analysis_result.get("confidence", 0.0)

            embed.add_field(
                name="📊 複雑度",
                value=f"**{complexity.upper()}**",
                inline=True
            )
            embed.add_field(
                name="🎯 信頼度",
                value=f"{confidence * 100:.0f}%",
                inline=True
            )
            embed.add_field(
                name="📋 タスク種別",
                value=(
                    f"{'✅' if is_multi_step else '❌'} 複数ステップ\n"
                    f"{'✅' if is_action_task else '❌'} アクション\n"
                    f"{'✅' if is_simple_qa else '❌'} Q&A"
                ),
                inline=False
            )

            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name="langgraph",
                message="",
                thread=thread_data.thread,
                embed=embed
            )

    async def report_mcp_usage(
        self,
        task_id: str,
        mcp_name: str,
        operation: str,
        params: Optional[Dict[str, Any]] = None,
        result: Optional[str] = None
    ) -> None:
        """
        Report MCP tool usage.

        Args:
            task_id: Task identifier
            mcp_name: MCP tool name (e.g., "filesystem", "git", "sequential_thinking")
            operation: Operation name (e.g., "read_file", "git_commit")
            params: Optional operation parameters
            result: Optional result summary
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            return

        async with thread_data.update_lock:
            agent_name = thread_data.active_agent
            if not agent_name:
                return

            config = self.agent_config.get(agent_name, {})
            emoji = config.get("emoji", "🤖")
            role_name = config.get("display_name", agent_name)

            # MCP名の日本語マッピング
            mcp_names_ja = {
                "filesystem": "ファイルシステム",
                "git": "Git管理",
                "sequential_thinking": "段階的思考",
                "brave_search": "Brave検索",
                "fetch": "Web取得",
                "memory": "記憶管理",
                "postgres": "PostgreSQL",
                "sqlite": "SQLite",
                "puppeteer": "Puppeteer（ブラウザ操作）",
                "github": "GitHub",
                "gitlab": "GitLab",
                "google_maps": "Google Maps",
                "slack": "Slack",
                "time": "時刻管理",
                "cloudflare-mcp": "Cloudflare",
                "aws-kb-retrieval-mcp": "AWS知識ベース",
            }
            mcp_display = mcp_names_ja.get(mcp_name, mcp_name)

            # Format message with MCP name clearly shown
            message = f"{emoji} **{role_name}**\n\n🔧 **使用MCP:** `{mcp_name}` ({mcp_display})\n⚙️ **実行操作:** `{operation}`"

            if params:
                # Show key parameters (limit to avoid spam)
                param_items = list(params.items())[:3]
                param_lines = [f"  • `{k}`: {str(v)[:50]}" for k, v in param_items]
                if len(params) > 3:
                    param_lines.append(f"  （他 {len(params) - 3}個のパラメータ）")
                message += f"\n📝 **パラメータ:**\n" + "\n".join(param_lines)

            if result:
                message += f"\n✅ **実行結果:** {result[:150]}"

            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=agent_name,
                message=message,
                thread=thread_data.thread
            )

    async def report_artifact_created(
        self,
        task_id: str,
        artifact_type: str,
        path: str,
        description: Optional[str] = None
    ) -> None:
        """
        Report artifact creation (file, directory, etc.).

        Args:
            task_id: Task identifier
            artifact_type: Type (file, directory, commit, etc.)
            path: File/directory path
            description: Optional description
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            return

        async with thread_data.update_lock:
            agent_name = thread_data.active_agent
            if not agent_name:
                return

            config = self.agent_config.get(agent_name, {})
            emoji = config.get("emoji", "🤖")
            role_name = config.get("display_name", agent_name)

            # Artifact type emoji and name mapping
            artifact_mapping = {
                "file": ("📄", "新規書状"),
                "directory": ("📁", "書庫"),
                "commit": ("💾", "記録"),
                "image": ("🖼️", "絵図"),
                "code": ("💻", "機構"),
                "config": ("⚙️", "設定書"),
            }
            art_emoji, art_name = artifact_mapping.get(artifact_type.lower(), ("📦", "成果物"))

            # 戦国武将風の説明
            samurai_desc = description or f"{art_name}を作成いたしました"

            embed = discord.Embed(
                title=f"{art_emoji} {art_name}作成",
                description=samurai_desc,
                color=discord.Color.green(),
                timestamp=datetime.now()
            )

            embed.add_field(name="👤 担当武将", value=f"{emoji} {role_name}", inline=True)
            embed.add_field(name="📂 所在", value=f"`{path}`", inline=False)

            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=agent_name,
                message="",
                thread=thread_data.thread,
                embed=embed
            )

    async def report_detailed_progress(
        self,
        task_id: str,
        step_name: str,
        details: str,
        progress: Optional[float] = None
    ) -> None:
        """
        Report detailed progress with step name.

        Args:
            task_id: Task identifier
            step_name: Current step name
            details: Detailed description
            progress: Optional progress value (0.0 to 1.0)
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            return

        async with thread_data.update_lock:
            agent_name = thread_data.active_agent
            if not agent_name:
                return

            config = self.agent_config.get(agent_name, {})
            emoji = config.get("emoji", "🤖")
            role_name = config.get("display_name", agent_name)

            # ステップ名を明確に表示
            message = f"{emoji} **{role_name}**\n\n📍 **現在の作業:** {step_name}\n💬 {details}"

            if progress is not None:
                progress_bar = self._create_progress_bar(progress)
                message += f"\n{progress_bar}"

            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=agent_name,
                message=message,
                thread=thread_data.thread
            )

    def _create_progress_bar(self, progress: float) -> str:
        """
        Create ASCII progress bar.

        Args:
            progress: Progress value (0.0 to 1.0)

        Returns:
            Progress bar string
        """
        filled = int(progress * 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"[{bar}] {progress * 100:.0f}%"

    async def request_approval(
        self,
        task_id: str,
        action_type: str,
        action_details: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Any:  # Returns ApprovalResult
        """
        Request approval from user before executing an action.

        Args:
            task_id: Task identifier
            action_type: Type of action (file_create, git_commit, delegation, etc.)
            action_details: Details about the action to approve
            timeout: Optional timeout in seconds (uses config default if not specified)

        Returns:
            ApprovalResult with status and optional user_instruction
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            logger.warning(f"⚠️ Task thread not found for approval: {task_id}")
            # Return auto-approved for missing thread
            from bushidan.approval_manager import ApprovalResult, ApprovalStatus
            return ApprovalResult(
                status=ApprovalStatus.APPROVED,
                user_instruction=None,
                reason="Task thread not found"
            )

        async with thread_data.update_lock:
            agent_name = thread_data.active_agent or "unknown"
            thread_data.paused = True

            logger.info(f"📝 Requesting approval for {action_type} in task {task_id}")

            # Delegate to approval manager
            result = await self.bot.approval_manager.request_approval(
                task_id=task_id,
                agent_name=agent_name,
                action_type=action_type,
                action_details=action_details,
                thread=thread_data.thread,
                timeout=timeout
            )

            thread_data.paused = False
            thread_data.pending_approval = None

            logger.info(f"✅ Approval result: {result.status.value}")
            return result

    async def wait_for_user_response(
        self,
        task_id: str,
        prompt_message: str,
        timeout: int = 300
    ) -> Optional[str]:
        """
        Wait for user response in thread (for agent-user conversation).

        Args:
            task_id: Task identifier
            prompt_message: Message to prompt user with
            timeout: Timeout in seconds

        Returns:
            User's message content, or None if timeout
        """
        thread_data = self.bot.task_threads.get(task_id)
        if not thread_data:
            logger.warning(f"⚠️ Task thread not found for conversation: {task_id}")
            return None

        async with thread_data.update_lock:
            agent_name = thread_data.active_agent or "unknown"
            config = self.agent_config.get(agent_name, {})
            emoji = config.get("emoji", "🤖")

            # Send prompt message
            full_prompt = f"{emoji} {prompt_message}\n\n💬 返信をお待ちしています..."
            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=agent_name,
                message=full_prompt,
                thread=thread_data.thread
            )

            thread_data.paused = True

        # Wait for user message using bot's approval manager
        try:
            # Create an event to wait on
            response_event = asyncio.Event()
            response_content = {"content": None}

            # Define check function for user messages
            def check(message: discord.Message) -> bool:
                return (
                    message.channel.id == thread_data.thread.id and
                    message.author != self.bot.user and
                    not message.webhook_id
                )

            # Wait for message
            user_message = await self.bot.wait_for(
                'message',
                check=check,
                timeout=timeout
            )

            async with thread_data.update_lock:
                thread_data.paused = False
                thread_data.user_messages.append(user_message)

            logger.info(f"💬 Received user response: {user_message.content[:100]}")
            return user_message.content

        except asyncio.TimeoutError:
            async with thread_data.update_lock:
                thread_data.paused = False

            # Notify timeout
            await self.webhook_manager.send_as_agent(
                channel=thread_data.thread.parent,
                agent_name=agent_name,
                message=f"{emoji} ⏰ タイムアウトしました（{timeout}秒）\n続行します...",
                thread=thread_data.thread
            )

            logger.warning(f"⏰ User response timeout for task {task_id}")
            return None
