"""
武士団 Mattermost 承認マネージャー

Discordのリアクション絵文字による承認フローを、
Mattermostのインタラクティブボタンで置き換えます。

Discord でできなかったこと:
  - リアクションが不安定・遅延する
  - ボタン UI がない（絵文字で代用していた）
  - タイムアウト管理が複雑

Mattermost で実現すること:
  - [✅ 承認] [❌ 却下] [✏️ 修正指示] ボタン
  - ボタンクリックで即座に確定
  - 修正指示はモーダル入力（テキスト追記）
  - asyncio Event でスレッドセーフな待機
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("bushidan.mattermost_approval")


class ApprovalStatus(Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    REJECTED  = "rejected"
    MODIFIED  = "modified"   # 修正指示あり承認
    TIMEOUT   = "timeout"
    ERROR     = "error"


@dataclass
class ApprovalResult:
    status: ApprovalStatus
    instruction: Optional[str] = None   # 修正指示テキスト
    responder: Optional[str] = None     # 応答したユーザー名
    responded_at: Optional[datetime] = None


@dataclass
class PendingApproval:
    request_id: str
    post_id: str                        # Mattermostの投稿ID
    channel_id: str
    agent_name: str
    action_type: str
    action_details: Dict[str, Any]
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Optional[ApprovalResult] = None
    created_at: datetime = field(default_factory=datetime.now)


# action_type の日本語ラベル
ACTION_LABELS = {
    "file_create":  "📄 ファイル作成",
    "file_edit":    "✏️ ファイル編集",
    "file_delete":  "🗑️ ファイル削除",
    "git_commit":   "📦 Gitコミット",
    "git_push":     "🚀 Gitプッシュ",
    "delegation":   "🤝 エージェント委譲",
    "shell_exec":   "💻 シェル実行",
    "api_call":     "🌐 API呼び出し",
}

# デフォルト承認タイムアウト (秒)
DEFAULT_TIMEOUT = 300

# タイムアウト時のデフォルト動作
TIMEOUT_POLICY = {
    "file_create":  "auto_approve",
    "file_edit":    "auto_approve",
    "git_commit":   "auto_approve",
    "file_delete":  "auto_reject",
    "git_push":     "auto_reject",
    "delegation":   "auto_approve",
    "shell_exec":   "auto_reject",
    "api_call":     "auto_approve",
}


class MattermostApprovalManager:
    """Mattermostボタンによる承認フロー管理."""

    def __init__(self) -> None:
        self._pending: Dict[str, PendingApproval] = {}
        self._api = None   # MattermostAPI インスタンスを注入

    def set_api(self, api) -> None:
        self._api = api

    # ── 承認リクエスト投稿 ────────────────────────────────────────────

    async def request_approval(
        self,
        agent_name: str,
        action_type: str,
        action_details: Dict[str, Any],
        channel_id: str,
        root_id: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        callback_url_base: str = "",
    ) -> ApprovalResult:
        """
        承認ボタン付きメッセージを投稿して応答を待つ.

        Args:
            agent_name:         エージェント名 (例: "軍師")
            action_type:        作業種別 (例: "git_push")
            action_details:     作業内容の辞書
            channel_id:         投稿先チャンネルID
            root_id:            スレッドのルートID (空なら新規スレッド)
            timeout:            待機秒数
            callback_url_base:  ボタン受信サーバーのベースURL
        Returns:
            ApprovalResult
        """
        request_id = str(uuid.uuid4())[:8]
        action_label = ACTION_LABELS.get(action_type, f"🔧 {action_type}")

        # 詳細テキストの構築
        detail_lines = []
        for k, v in action_details.items():
            val = str(v)
            if len(val) > 200:
                val = val[:200] + "..."
            detail_lines.append(f"- **{k}**: `{val}`")
        detail_text = "\n".join(detail_lines) if detail_lines else "(詳細なし)"

        message = (
            f"### 🔔 承認リクエスト `[{request_id}]`\n"
            f"**エージェント:** {agent_name}\n"
            f"**作業種別:** {action_label}\n\n"
            f"**作業内容:**\n{detail_text}\n\n"
            f"⏳ タイムアウト: {timeout}秒"
        )

        # インタラクティブボタンの構築
        base = callback_url_base.rstrip("/")
        attachments = [
            {
                "text": "",
                "actions": [
                    {
                        "id":           f"approve_{request_id}",
                        "name":         "✅ 承認",
                        "integration": {
                            "url": f"{base}/api/actions",
                            "context": {
                                "request_id": request_id,
                                "action":     "approve",
                            },
                        },
                    },
                    {
                        "id":           f"reject_{request_id}",
                        "name":         "❌ 却下",
                        "integration": {
                            "url": f"{base}/api/actions",
                            "context": {
                                "request_id": request_id,
                                "action":     "reject",
                            },
                        },
                    },
                    {
                        "id":           f"modify_{request_id}",
                        "name":         "✏️ 修正指示",
                        "integration": {
                            "url": f"{base}/api/actions",
                            "context": {
                                "request_id": request_id,
                                "action":     "modify",
                            },
                        },
                    },
                ],
            }
        ]

        post_options: Dict[str, Any] = {
            "channel_id": channel_id,
            "message":    message,
            "props":      {"attachments": attachments},
        }
        if root_id:
            post_options["root_id"] = root_id

        try:
            post = await self._api.create_post(post_options)
            post_id = post.get("id", "")
        except Exception as e:
            logger.error("承認メッセージ投稿失敗: %s", e)
            return ApprovalResult(status=ApprovalStatus.ERROR)

        # Pending 登録
        pending = PendingApproval(
            request_id=request_id,
            post_id=post_id,
            channel_id=channel_id,
            agent_name=agent_name,
            action_type=action_type,
            action_details=action_details,
        )
        self._pending[request_id] = pending

        # ボタン応答を待機
        try:
            await asyncio.wait_for(pending.event.wait(), timeout=timeout)
            result = pending.result or ApprovalResult(status=ApprovalStatus.ERROR)
        except asyncio.TimeoutError:
            result = self._apply_timeout_policy(action_type)
            await self._update_approval_post(
                post_id, message, f"⏰ タイムアウト → {result.status.value}"
            )
        finally:
            self._pending.pop(request_id, None)

        return result

    # ── ボタンクリック受信 ────────────────────────────────────────────

    async def handle_action(
        self,
        request_id: str,
        action: str,
        user_name: str,
        user_text: str = "",
    ) -> str:
        """
        HTTP callback から呼ばれる。ボタンクリックを処理して待機中タスクを解放.

        Args:
            request_id: 承認リクエストID
            action:     "approve" / "reject" / "modify"
            user_name:  クリックしたユーザー名
            user_text:  修正指示テキスト (modify 時)
        Returns:
            Mattermostに返すレスポンスメッセージ
        """
        pending = self._pending.get(request_id)
        if not pending:
            return f"⚠️ リクエスト `{request_id}` は既に処理済みか存在しません。"

        now = datetime.now()

        if action == "approve":
            status = ApprovalStatus.APPROVED
            label = "✅ 承認されました"
        elif action == "reject":
            status = ApprovalStatus.REJECTED
            label = "❌ 却下されました"
        elif action == "modify":
            status = ApprovalStatus.MODIFIED
            label = f"✏️ 修正指示: {user_text or '(テキストなし)'}"
        else:
            return f"❓ 不明なアクション: {action}"

        pending.result = ApprovalResult(
            status=status,
            instruction=user_text if action == "modify" else None,
            responder=user_name,
            responded_at=now,
        )

        # 投稿を結果表示に更新
        original_msg = (
            f"### 🔔 承認リクエスト `[{request_id}]`\n"
            f"**エージェント:** {pending.agent_name}\n"
            f"**作業種別:** {ACTION_LABELS.get(pending.action_type, pending.action_type)}"
        )
        await self._update_approval_post(
            pending.post_id,
            original_msg,
            f"{label}\n👤 by @{user_name} ({now.strftime('%H:%M:%S')})",
        )

        # 待機中のコルーチンを解放
        pending.event.set()

        return f"✅ `{request_id}` への応答を受け付けました。"

    # ── ユーティリティ ────────────────────────────────────────────────

    def _apply_timeout_policy(self, action_type: str) -> ApprovalResult:
        policy = TIMEOUT_POLICY.get(action_type, "auto_reject")
        if policy == "auto_approve":
            return ApprovalResult(
                status=ApprovalStatus.APPROVED,
                instruction="タイムアウトのため自動承認",
            )
        return ApprovalResult(
            status=ApprovalStatus.REJECTED,
            instruction="タイムアウトのため自動却下",
        )

    async def _update_approval_post(
        self, post_id: str, original: str, result_line: str
    ) -> None:
        """承認ボタン投稿をボタンなしの結果表示に差し替え."""
        if not post_id or not self._api:
            return
        try:
            await self._api.patch_post(
                post_id,
                {"message": f"{original}\n\n---\n{result_line}", "props": {}},
            )
        except Exception as e:
            logger.warning("承認投稿の更新失敗: %s", e)

    def get_pending_count(self) -> int:
        return len(self._pending)
