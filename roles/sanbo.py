"""roles/sanbo.py — 参謀 (Mistral Large 3) ロール v14

役割: 汎用コーディング・中量級推論
モデル: Mistral Large 3
"""

import time
from roles.base import BaseRole, RoleResult


class SanboRole(BaseRole):
    role_key = "sanbo"
    role_name = "参謀"
    model_name = "Mistral Large 3"
    emoji = "🗡️"
    default_handled_by = "taisho_mcp"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 参謀クライアント未設定 (MISTRAL_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは参謀担当 (Mistral Large 3)。汎用コーディングと推論の専門家です。"
                "明確・実践的な日本語で回答してください。",
            )
            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=4000)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
            )
        except Exception as e:
            self.logger.error("参謀実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 参謀エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
