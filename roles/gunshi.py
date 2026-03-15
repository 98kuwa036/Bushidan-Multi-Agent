"""roles/gunshi.py — 軍師 (Mistral Large 3) ロール v14.1

役割: 深層推論・PDCA・複雑問題分析
モデル: Mistral AI Large 3 (統一版)

v14.1 変更: o3-mini → Mistral Large 3 に統一
"""

import time
from roles.base import BaseRole, RoleResult


class GunshiRole(BaseRole):
    role_key = "gunshi"
    role_name = "軍師"
    model_name = "Mistral Large 3"
    emoji = "📜"
    default_handled_by = "gunshi_pdca"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 軍師クライアント未設定 (MISTRAL_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは武士団の軍師（Mistral Large 3）。深層推論・PDCA分析の専門家として、"
                "複雑なタスクを日本語で詳細に分析・解決してください。"
                "常に論理的・段階的にアプローチしてください。",
            )
            messages = self._format_messages(state)
            response = await client.generate(
                messages=messages, system=system, max_tokens=4000,
            )
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
            )
        except Exception as e:
            self.logger.error("軍師実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 軍師エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
