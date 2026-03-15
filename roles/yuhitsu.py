"""roles/yuhitsu.py — 右筆 (Llama ELYZA) ロール v14

役割: 日本語清書・翻訳・添削
モデル: Llama-3-ELYZA-JP-8B (Local port 8081) / Nemotron フォールバック
"""

import time
from roles.base import BaseRole, RoleResult


class YuhitsuRole(BaseRole):
    role_key = "yuhitsu"
    role_name = "右筆"
    model_name = "Llama ELYZA"
    emoji = "🖊️"
    default_handled_by = "yuhitsu_jp"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 右筆クライアント未設定",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは日本語テキスト清書アシスタントです。自然で流暢な日本語で回答してください。",
            )
            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=2048)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
            )
        except Exception as e:
            self.logger.error("右筆実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 右筆エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
