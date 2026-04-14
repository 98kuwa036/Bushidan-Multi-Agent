"""roles/yuhitsu.py — 右筆 (Gemma4 MoE Local) ロール v18

役割: 日本語会話・意図伝達・清書・翻訳・添削
モデル: Gemma4 MoE Local（Gemini Flash-Lite と同等、ローカルに統合）
"""

import time
from roles.base import BaseRole, RoleResult

_SYSTEM_PROMPT = (
    "あなたは日本語アシスタントの右筆です。"
    "ユーザーの意図を正確に汲み取り、自然で丁寧な日本語で回答してください。"
    "清書・翻訳・添削が必要な場合は適切に整えてください。"
)


class YuhitsuRole(BaseRole):
    role_key = "yuhitsu"
    role_name = "右筆"
    model_name = "Gemma4 MoE Local"
    emoji = "✍️"
    default_handled_by = "yuhitsu_jp"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 右筆クライアント未設定 (GEMINI_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(state, _SYSTEM_PROMPT)
            messages = self._format_messages(state)
            response = await client.generate(
                messages=messages, system=system, max_tokens=2048,
            )
            return RoleResult(
                response=response,
                agent_role=self.role_name,
                handled_by=self.default_handled_by,
                execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
            )
        except Exception as e:
            self.logger.error("右筆実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 右筆エラー: {e}",
                agent_role=self.role_name,
                handled_by=self.default_handled_by,
                execution_time=time.time() - start,
                error=str(e),
                status="failed",
            )
