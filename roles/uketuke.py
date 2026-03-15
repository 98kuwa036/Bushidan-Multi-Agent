"""roles/uketuke.py — 受付 (Command R 7B) ロール v14.1

役割: ルーティング受付・軽量フォールバック応答
モデル: Cohere Command R 7B (軽量版、$0.0375/M tokens)

v14.1 更新: command-r → command-r-7b (Cohere Command A への移行対応)
"""

import time
from roles.base import BaseRole, RoleResult


class UketukeRole(BaseRole):
    role_key = "uketuke"
    role_name = "受付"
    model_name = "Command R"
    emoji = "🚪"
    default_handled_by = "karo_default"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 受付クライアント未設定 (COHERE_API_KEY を確認してください)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(state)
            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=2048)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
            )
        except Exception as e:
            self.logger.error("受付実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 受付エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
