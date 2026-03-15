"""roles/shogun.py — 将軍 (Claude Sonnet 4.6) ロール v14

役割: メインワーカー・高難度コーディング
モデル: claude-sonnet-4-6 (Proプラン CLI 優先 → API フォールバック)
"""

import time
from roles.base import BaseRole, RoleResult

PERSONA = (
    "あなたは将軍（Claude Sonnet 4.6）、武士団マルチエージェントシステムのメインワーカーです。"
    "高難度コーディングと実装を担当します。的確かつ実践的に日本語で回答してください。"
)


class ShogunRole(BaseRole):
    role_key = "shogun"
    role_name = "将軍"
    model_name = "Claude Sonnet 4.6"
    emoji = "🏯"
    default_handled_by = "shogun_direct"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 将軍クライアント未設定 (ANTHROPIC_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(state, PERSONA)
            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=4000)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
            )
        except Exception as e:
            self.logger.error("将軍実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 将軍エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
