"""roles/seppou.py — 斥候 (Llama 3.3 70B Groq) ロール v14

役割: 高速フィルタ・雑談・シンプルQ&A
モデル: Meta Llama 3.3 70B (Groq)
"""

import time
from roles.base import BaseRole, RoleResult


class SeppouRole(BaseRole):
    role_key = "seppou"
    role_name = "斥候"
    model_name = "Llama 3.3 70B (Groq)"
    emoji = "🏹"
    default_handled_by = "groq_qa"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 斥候クライアント未設定 (GROQ_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは斥候担当。簡潔で正確な回答を日本語で素早く返してください。",
            )
            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=512)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
            )
        except Exception as e:
            self.logger.error("斥候実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 斥候エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
