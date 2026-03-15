"""roles/onmitsu.py — 隠密 (Nemotron Local) ロール v14

役割: 機密データ処理・完全ローカル (API送信不可)
モデル: Nemotron-3-Nano-30B-A3B (Local port 8080)
"""

import time
from roles.base import BaseRole, RoleResult


class OnmitsuRole(BaseRole):
    role_key = "onmitsu"
    role_name = "隠密"
    model_name = "Nemotron-3-Nano (Local)"
    emoji = "🥷"
    default_handled_by = "onmitsu_local"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 隠密クライアント未設定 (llama.cpp サーバー確認)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            messages = self._format_messages(state)
            response = await client.generate(messages=messages, max_tokens=4096, task_type="confidential")
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
            )
        except Exception as e:
            self.logger.error("隠密実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 隠密エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
