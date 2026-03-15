"""roles/gaiji.py — 外事 (Command A 03-2025) ロール v14.1

役割: 外部情報収集・RAG・マルチステップ処理
モデル: Cohere Command A (03-2025) - 256K コンテキスト・RAG最適化

v14.1 更新: command-r-plus → command-a-03-2025 (Cohere最新推奨モデル)
"""

import time
from roles.base import BaseRole, RoleResult


class GaijiRole(BaseRole):
    role_key = "gaiji"
    role_name = "外事"
    model_name = "Command A (03-2025)"
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
                "あなたは外事担当 (Command A)。外部情報収集・RAG・マルチステップ処理・ツール統合の専門家です。"
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
