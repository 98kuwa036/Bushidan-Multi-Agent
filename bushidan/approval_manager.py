"""
Bushidan Approval Manager

インタラクティブモードの承認システムを管理します。
ユーザーのリアクションやコメントを監視し、承認/却下/修正指示を処理します。
"""

import asyncio
import discord
import yaml
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


class ApprovalStatus(Enum):
    """承認ステータス"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ApprovalRequest:
    """
    承認リクエストの状態管理

    Attributes:
        request_id: リクエストID
        task_id: タスクID
        agent_name: エージェント名
        action_type: 作業種別（file_create, git_commit, delegation など）
        action_details: 作業の詳細情報
        approval_message: 承認待ちメッセージ
        created_at: 作成時刻
        status: 承認ステータス
        user_instruction: ユーザーの指示（修正時）
        response_user: 応答したユーザー
        response_time: 応答時刻
    """
    request_id: str
    task_id: str
    agent_name: str
    action_type: str
    action_details: Dict[str, Any]
    approval_message: discord.Message
    created_at: datetime = field(default_factory=datetime.now)
    status: ApprovalStatus = ApprovalStatus.PENDING
    user_instruction: Optional[Dict[str, Any]] = None
    response_user: Optional[discord.User] = None
    response_time: Optional[datetime] = None


@dataclass
class ApprovalResult:
    """
    承認結果

    Attributes:
        status: 承認ステータス
        user_instruction: ユーザーの指示（修正時）
        message: 結果メッセージ
        response_user: 応答したユーザー
    """
    status: ApprovalStatus
    user_instruction: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    response_user: Optional[discord.User] = None


class InteractiveConfig:
    """インタラクティブモード設定管理"""

    def __init__(self, config_path: str = "config/interactive_config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """設定ファイルを読み込む"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"⚠️ 設定ファイルが見つかりません: {self.config_path}")
                self._load_default_config()
                return

            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            logger.info(f"✅ インタラクティブ設定読み込み完了: {self.config_path}")
        except Exception as e:
            logger.error(f"❌ 設定ファイル読み込みエラー: {e}")
            self._load_default_config()

    def _load_default_config(self) -> None:
        """デフォルト設定をロード"""
        self.config = {
            "interactive_mode": {
                "enabled": False,
                "approval_required": {
                    "delegation": True,
                    "file_create": True,
                    "file_edit": True,
                    "file_delete": True,
                    "git_commit": True,
                    "git_push": True,
                },
                "approval_timeout": 300,
                "timeout_actions": {
                    "file_create": "auto_approve",
                    "file_edit": "auto_approve",
                    "git_commit": "auto_approve",
                    "file_delete": "auto_reject",
                    "git_push": "auto_reject",
                },
                "reaction_emojis": {
                    "approve": "👍",
                    "reject": "👎",
                    "retry": "🔄",
                    "modify": "✏️",
                },
                "keywords": {
                    "approve": ["承認", "ok", "approve", "yes"],
                    "reject": ["却下", "no", "reject", "やめて"],
                    "retry": ["やり直し", "retry"],
                },
            }
        }
        logger.info("ℹ️ デフォルトのインタラクティブ設定を使用")

    def is_enabled(self) -> bool:
        """インタラクティブモードが有効か"""
        return self.config.get("interactive_mode", {}).get("enabled", False)

    def is_approval_required(self, action_type: str) -> bool:
        """指定された作業種別で承認が必要か"""
        required = self.config.get("interactive_mode", {}).get("approval_required", {})
        return required.get(action_type, False)

    def get_timeout(self) -> int:
        """タイムアウト時間（秒）を取得"""
        return self.config.get("interactive_mode", {}).get("approval_timeout", 300)

    def get_timeout_action(self, action_type: str) -> str:
        """タイムアウト時の動作を取得"""
        actions = self.config.get("interactive_mode", {}).get("timeout_actions", {})
        return actions.get(action_type, "auto_reject")

    def get_reaction_emojis(self) -> Dict[str, str]:
        """リアクション絵文字を取得"""
        return self.config.get("interactive_mode", {}).get("reaction_emojis", {
            "approve": "👍",
            "reject": "👎",
            "retry": "🔄",
            "modify": "✏️",
        })

    def get_keywords(self) -> Dict[str, List[str]]:
        """キーワードを取得"""
        return self.config.get("interactive_mode", {}).get("keywords", {})


class ApprovalManager:
    """
    承認管理システム

    ユーザーのリアクションとコメントを監視し、承認/却下/修正を処理します。
    """

    def __init__(self, bot: "BushidanDiscordBot"):
        """
        初期化

        Args:
            bot: BushidanDiscordBot インスタンス
        """
        self.bot = bot
        self.config = InteractiveConfig()
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.approval_events: Dict[str, asyncio.Event] = {}
        self.message_id_to_request_id: Dict[int, str] = {}

    async def request_approval(
        self,
        task_id: str,
        agent_name: str,
        action_type: str,
        action_details: Dict[str, Any],
        thread: discord.Thread,
        timeout: Optional[int] = None
    ) -> ApprovalResult:
        """
        承認をリクエストし、結果を待つ

        Args:
            task_id: タスクID
            agent_name: エージェント名
            action_type: 作業種別
            action_details: 作業詳細
            thread: Discordスレッド
            timeout: タイムアウト時間（秒、Noneの場合は設定から取得）

        Returns:
            ApprovalResult: 承認結果
        """
        # インタラクティブモードが無効またはこの作業で承認不要の場合は自動承認
        if not self.config.is_enabled() or not self.config.is_approval_required(action_type):
            logger.debug(f"ℹ️ {action_type} は自動承認（設定により承認不要）")
            return ApprovalResult(
                status=ApprovalStatus.APPROVED,
                message="自動承認（設定により承認不要）"
            )

        # タイムアウト取得
        if timeout is None:
            timeout = self.config.get_timeout()

        # リクエストIDを生成
        request_id = f"{task_id}_{agent_name}_{action_type}_{int(datetime.now().timestamp())}"

        # 承認メッセージを投稿
        approval_message = await self._post_approval_message(
            thread, agent_name, action_type, action_details
        )

        # リアクションを追加
        await self._add_reaction_buttons(approval_message)

        # ApprovalRequestを作成
        approval_request = ApprovalRequest(
            request_id=request_id,
            task_id=task_id,
            agent_name=agent_name,
            action_type=action_type,
            action_details=action_details,
            approval_message=approval_message
        )

        # 保存
        self.pending_approvals[request_id] = approval_request
        self.approval_events[request_id] = asyncio.Event()
        self.message_id_to_request_id[approval_message.id] = request_id

        logger.info(f"📋 承認リクエスト: {agent_name} - {action_type}")

        # ユーザーの応答を待つ
        try:
            await asyncio.wait_for(
                self.approval_events[request_id].wait(),
                timeout=timeout
            )

            # 応答あり
            result = self._create_result_from_request(approval_request)
            logger.info(f"✅ 承認応答受領: {result.status.value}")

        except asyncio.TimeoutError:
            # タイムアウト
            logger.warning(f"⏰ 承認タイムアウト: {agent_name} - {action_type}")
            result = await self._handle_timeout(approval_request)

        finally:
            # クリーンアップ
            self.pending_approvals.pop(request_id, None)
            self.approval_events.pop(request_id, None)
            self.message_id_to_request_id.pop(approval_message.id, None)

        return result

    async def handle_reaction(
        self,
        reaction: discord.Reaction,
        user: discord.User
    ) -> None:
        """
        リアクションを処理

        Args:
            reaction: リアクション
            user: リアクションしたユーザー
        """
        # 承認待ちメッセージへのリアクションか確認
        request_id = self.message_id_to_request_id.get(reaction.message.id)
        if not request_id:
            return

        approval_request = self.pending_approvals.get(request_id)
        if not approval_request:
            return

        # リアクション解釈
        emojis = self.config.get_reaction_emojis()
        emoji_str = str(reaction.emoji)

        if emoji_str == emojis.get("approve"):
            # 承認
            approval_request.status = ApprovalStatus.APPROVED
            approval_request.response_user = user
            approval_request.response_time = datetime.now()
            logger.info(f"👍 承認: {user.name} - {approval_request.action_type}")

        elif emoji_str == emojis.get("reject"):
            # 却下
            approval_request.status = ApprovalStatus.REJECTED
            approval_request.response_user = user
            approval_request.response_time = datetime.now()
            logger.info(f"👎 却下: {user.name} - {approval_request.action_type}")

        elif emoji_str == emojis.get("retry"):
            # やり直し
            approval_request.status = ApprovalStatus.MODIFIED
            approval_request.user_instruction = {"action": "retry"}
            approval_request.response_user = user
            approval_request.response_time = datetime.now()
            logger.info(f"🔄 やり直し: {user.name} - {approval_request.action_type}")

        else:
            # 不明なリアクション
            return

        # イベントをセット
        event = self.approval_events.get(request_id)
        if event:
            event.set()

    async def handle_comment(
        self,
        message: discord.Message,
        task_id: str
    ) -> None:
        """
        コメントを処理

        Args:
            message: ユーザーのメッセージ
            task_id: タスクID
        """
        # このタスクの承認待ちを検索
        approval_request = None
        request_id = None

        for rid, req in self.pending_approvals.items():
            if req.task_id == task_id:
                approval_request = req
                request_id = rid
                break

        if not approval_request:
            return

        # メッセージ内容を解釈
        content = message.content.lower().strip()
        keywords = self.config.get_keywords()

        # 承認キーワードチェック
        if any(kw in content for kw in keywords.get("approve", [])):
            approval_request.status = ApprovalStatus.APPROVED
            approval_request.response_user = message.author
            approval_request.response_time = datetime.now()
            logger.info(f"✅ 承認（コメント）: {message.author.name}")

        # 却下キーワードチェック
        elif any(kw in content for kw in keywords.get("reject", [])):
            approval_request.status = ApprovalStatus.REJECTED
            approval_request.response_user = message.author
            approval_request.response_time = datetime.now()
            logger.info(f"❌ 却下（コメント）: {message.author.name}")

        # やり直しキーワードチェック
        elif any(kw in content for kw in keywords.get("retry", [])):
            approval_request.status = ApprovalStatus.MODIFIED
            approval_request.user_instruction = {"action": "retry"}
            approval_request.response_user = message.author
            approval_request.response_time = datetime.now()
            logger.info(f"🔄 やり直し（コメント）: {message.author.name}")

        # 修正指示（自然言語）
        else:
            # InstructionParserで解釈（後で実装）
            approval_request.status = ApprovalStatus.MODIFIED
            approval_request.user_instruction = {
                "action": "modify",
                "raw_message": message.content,
                "parsed": None  # InstructionParserで解析予定
            }
            approval_request.response_user = message.author
            approval_request.response_time = datetime.now()
            logger.info(f"✏️ 修正指示: {message.author.name} - {content[:50]}")

        # イベントをセット
        event = self.approval_events.get(request_id)
        if event:
            event.set()

    async def _post_approval_message(
        self,
        thread: discord.Thread,
        agent_name: str,
        action_type: str,
        action_details: Dict[str, Any]
    ) -> discord.Message:
        """承認メッセージを投稿"""
        # エージェント情報取得
        from bushidan.discord_reporter import AGENT_CONFIG
        config = AGENT_CONFIG.get(agent_name, {})
        emoji = config.get("emoji", "🤖")
        display_name = config.get("display_name", agent_name)
        color = config.get("color", 0x808080)

        # 戦国武将風の口調を取得
        approval_request_style = self._get_speaking_style(agent_name, "approval_request")

        # 作業内容の説明を生成
        description = self._format_action_description(action_type, action_details)

        # 詳細情報のフォーマット
        details_text = self._format_action_details(action_type, action_details)

        # Embed作成（戦国武将風）
        embed = discord.Embed(
            title=f"{emoji} {display_name}",
            description=f"{approval_request_style}\n\n**作業内容**: {description}\n\n{details_text}",
            color=color,
            timestamp=datetime.now()
        )

        embed.add_field(
            name="御下命を",
            value="👍 承認 | 👎 却下 | 🔄 やり直し | ✏️ 修正指示\n\nまたは、コメントで指示を入力してください\n（例: 「test2.pyに変更して」「承認」「やめて」）",
            inline=False
        )

        # Webhookで投稿
        message = await self.bot.webhook_manager.send_as_agent(
            channel=thread.parent,
            agent_name=agent_name,
            message="",
            thread=thread,
            embed=embed
        )

        # Webhook送信はメッセージオブジェクトを返さないため、
        # 最新メッセージを取得
        async for msg in thread.history(limit=1):
            return msg

        raise Exception("承認メッセージの投稿に失敗しました")

    def _format_action_description(
        self,
        action_type: str,
        action_details: Dict[str, Any]
    ) -> str:
        """作業内容の説明を生成（戦国武将風）"""
        descriptions = {
            "file_create": f"📄 **新規書状作成:** `{action_details.get('filename', '?')}`",
            "file_edit": f"✏️ **書状修正:** `{action_details.get('filename', '?')}`",
            "file_delete": f"🔥 **書状破棄（危険）:** `{action_details.get('filename', '?')}`",
            "git_commit": f"📜 **記録を後世に:** `{action_details.get('message', '?')[:50]}`",
            "git_push": f"🚀 **記録を遠方へ送付（危険）:** `{action_details.get('branch', 'main')}`",
            "delegation": f"🤝 **任務委譲:** {action_details.get('from_agent', '?')} → {action_details.get('to_agent', '?')}",
            "command_execution": f"⚔️ **命令実行（危険）:** `{action_details.get('command', '?')}`",
        }

        return descriptions.get(action_type, f"{action_type}: {action_details}")

    def _format_action_details(
        self,
        action_type: str,
        action_details: Dict[str, Any]
    ) -> str:
        """作業詳細のフォーマット"""
        if action_type == "file_create":
            size = action_details.get('size', 0)
            preview = action_details.get('preview', '')
            details = f"📊 **文字数:** {size}文字\n"
            if preview:
                details += f"\n```\n{preview}\n```"
            return details

        elif action_type == "git_commit":
            files = action_details.get('files', 0)
            file_list = action_details.get('file_list', [])
            details = f"📁 **対象ファイル数:** {files}件\n"
            if file_list:
                details += "📝 **対象:**\n" + "\n".join([f"  • `{f}`" for f in file_list[:5]])
                if len(file_list) > 5:
                    details += f"\n  （他 {len(file_list) - 5}件）"
            return details

        elif action_type == "delegation":
            reason = action_details.get('reason', '')
            complexity = action_details.get('complexity', '')
            details = ""
            if reason:
                details += f"💭 **理由:** {reason}\n"
            if complexity:
                details += f"📊 **複雑度:** {complexity}"
            return details

        return action_details.get('preview', '')

    def _get_speaking_style(self, agent_name: str, style_key: str) -> str:
        """
        戦国武将風の口調を取得

        Args:
            agent_name: エージェント名
            style_key: スタイルキー（approval_request, approved, rejected）

        Returns:
            口調テキスト
        """
        try:
            speaking_config = self.config.config.get('interactive_mode', {}).get('speaking_style', {})
            if not speaking_config.get('enabled', False):
                return ""

            agent_styles = speaking_config.get('agent_styles', {}).get(agent_name, {})
            return agent_styles.get(style_key, "")
        except Exception:
            return ""

    async def _add_reaction_buttons(self, message: discord.Message) -> None:
        """リアクションボタンを追加"""
        emojis = self.config.get_reaction_emojis()

        try:
            await message.add_reaction(emojis.get("approve", "👍"))
            await asyncio.sleep(0.5)  # レート制限対策
            await message.add_reaction(emojis.get("reject", "👎"))
            await asyncio.sleep(0.5)
            await message.add_reaction(emojis.get("retry", "🔄"))
            await asyncio.sleep(0.5)
            await message.add_reaction(emojis.get("modify", "✏️"))
        except Exception as e:
            logger.warning(f"⚠️ リアクション追加エラー: {e}")

    def _create_result_from_request(
        self,
        approval_request: ApprovalRequest
    ) -> ApprovalResult:
        """ApprovalRequestからApprovalResultを生成"""
        return ApprovalResult(
            status=approval_request.status,
            user_instruction=approval_request.user_instruction,
            message=self._get_status_message(approval_request.status),
            response_user=approval_request.response_user
        )

    async def _handle_timeout(
        self,
        approval_request: ApprovalRequest
    ) -> ApprovalResult:
        """タイムアウト処理"""
        timeout_action = self.config.get_timeout_action(approval_request.action_type)

        if timeout_action == "auto_approve":
            # 自動承認（戦国武将風）
            status = ApprovalStatus.APPROVED
            message = f"⏰ 時が過ぎ申した。安全な作業ゆえ、自動にて承認し続行いたしまする"
            logger.info(f"⏰ タイムアウト → 自動承認: {approval_request.action_type}")

        else:  # auto_reject
            # 自動却下（戦国武将風）
            status = ApprovalStatus.REJECTED
            message = f"⏰ 時が過ぎ申した。危険な作業ゆえ、自動にて却下し中止いたしまする"
            logger.info(f"⏰ タイムアウト → 自動却下: {approval_request.action_type}")

        # タイムアウトメッセージを投稿
        try:
            await approval_request.approval_message.reply(message)
        except Exception as e:
            logger.warning(f"⚠️ タイムアウトメッセージ投稿エラー: {e}")

        return ApprovalResult(
            status=status,
            message=message
        )

    def _get_status_message(self, status: ApprovalStatus) -> str:
        """ステータスメッセージを取得（戦国武将風）"""
        messages = {
            ApprovalStatus.APPROVED: "✅ かたじけない。直ちに取り掛かる所存にて候",
            ApprovalStatus.REJECTED: "❌ 承知つかまつった。直ちに中止いたす",
            ApprovalStatus.MODIFIED: "✏️ 御意。御指示の通りに進めまする",
            ApprovalStatus.TIMEOUT: "⏰ 時が過ぎ申した",
            ApprovalStatus.ERROR: "❌ 不調が生じ申した",
        }
        return messages.get(status, "不明なる事態")
