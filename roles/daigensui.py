"""roles/daigensui.py — 大元帥 (Claude Opus 4.6) ロール v14

役割: 最終エスカレーション・最高難度判断
モデル: claude-opus-4-6 (Proプラン CLI 優先 → API フォールバック)
"""

import time
from roles.base import BaseRole, RoleResult

PERSONA = (
    "あなたは大元帥（Claude Opus 4.6）、武士団マルチエージェントシステムの総司令官です。"
    "最高難度の判断を下す最高意思決定者として、深く洞察に富んだ回答を日本語でしてください。"
)


class DaigensuiRole(BaseRole):
    role_key = "daigensui"
    role_name = "大元帥"
    model_name = "Claude Opus 4.6"
    emoji = "⚔️"
    default_handled_by = "daigensui_direct"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 大元帥クライアント未設定 (ANTHROPIC_API_KEY)",
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
            self.logger.error("大元帥実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 大元帥エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
