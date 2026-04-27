"""roles/yuhitsu.py — 右筆 (Gemma4 MoE Local) ロール v18

役割: 日本語会話・意図伝達・清書・翻訳・添削
モデル: Gemma4 MoE Local（プライマリ）
フォールバック: Gemini 3.1 Flash-Lite（ローカルLLMサーバーダウン時）
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
        system = self._build_system_prompt(state, _SYSTEM_PROMPT)
        messages = self._format_messages(state)

        # ── プライマリ: Gemma4 ローカル ────────────────────────────────
        client = self._get_client()
        if client:
            try:
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
                self.logger.warning("右筆 [Gemma] 失敗、クラウドフォールバックへ: %s", e)

        # ── フォールバック: Gemini 3.1 Flash-Lite ──────────────────────
        try:
            import os
            from utils.gemini3_client import Gemini3Client
            api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY 未設定")
            fb_client = Gemini3Client(api_key=api_key, model="gemini-3.1-flash-lite-preview")
            all_msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
            response = await fb_client.generate(
                messages=all_msgs, max_output_tokens=2048,
            )
            self.logger.info("✍️ 右筆 [Gemini Flash-Lite フォールバック] %.1fs", time.time() - start)
            return RoleResult(
                response=response,
                agent_role=self.role_name,
                handled_by=self.default_handled_by,
                execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=["gemini_fallback"],
            )
        except Exception as e2:
            self.logger.error("右筆フォールバック失敗: %s", e2)
            return RoleResult(
                response="⚠️ 右筆: ローカルLLMサーバー (192.168.11.239) が利用不可です。しばらく後でお試しください。",
                agent_role=self.role_name,
                handled_by=self.default_handled_by,
                execution_time=time.time() - start,
                error=str(e2),
                status="failed",
            )
