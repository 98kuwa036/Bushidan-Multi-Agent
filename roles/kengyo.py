"""roles/kengyo.py — 検校 (Gemini 3.1 Flash Image) ロール v18

役割: 画像解析・マルチモーダル処理
モデル: Gemini 3.1 Flash Image Preview
"""

import time
from roles.base import BaseRole, RoleResult


class KengyoRole(BaseRole):
    role_key = "kengyo"
    role_name = "検校"
    model_name = "Gemini 3.1 Flash Image"
    emoji = "👁️"
    default_handled_by = "kengyo_vision"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 検校クライアント未設定 (GEMINI_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは検校担当 (Gemini Vision)。画像解析・マルチモーダル処理の専門家です。"
                "URLが提供された場合はスクリーンショットを取得して視覚的に分析してください。",
            )
            mcp_used = []
            msg = state.get("message", "")
            attachments = list(state.get("attachments", []))

            # ── URL があれば playwright でスクリーンショット取得 ──────
            urls = self._extract_urls(msg)
            for url in urls[:2]:
                screenshot = await self._mcp_screenshot(url)
                if screenshot:
                    attachments.append({"type": "image_base64", "data": screenshot})
                    mcp_used.append("browser_take_screenshot")
                    self.logger.info("👁️ 検校: スクリーンショット取得 %s", url[:60])

            messages = self._format_messages(state)
            response = await client.generate(
                messages=messages, system=system, max_tokens=2048,
                attachments=attachments,
            )
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=mcp_used,
            )
        except Exception as e:
            self.logger.error("検校実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 検校エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
