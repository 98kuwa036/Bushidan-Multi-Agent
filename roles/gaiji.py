"""roles/gaiji.py — 外事 (Command R+) ロール v14

役割: 外部情報収集・RAG・マルチステップ処理
モデル: Cohere Command R+ (RAG特化)
"""

import time
from roles.base import BaseRole, RoleResult


class GaijiRole(BaseRole):
    role_key = "gaiji"
    role_name = "外事"
    model_name = "Command R+"
    emoji = "🌐"
    default_handled_by = "gaiji_rag"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 外事クライアント未設定 (COHERE_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは外事担当 (Command R+)。外部情報収集・RAG・マルチステップ処理の専門家です。"
                "正確で包括的な回答を日本語で提供してください。",
            )
            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=2048)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
            )
        except Exception as e:
            self.logger.error("外事実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 外事エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
