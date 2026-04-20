"""roles/metsuke.py — 目付 (Mistral Small) ロール v18

役割: 低中難度タスク・要約・整形・軽量推論
モデル: Mistral Small (mistral-small-latest)
位置: 難易度ティア「low_medium」担当
      Groq(斥候)より推論力が必要、Haiku(軍師)ほどでない作業を担う
"""

import time
from roles.base import BaseRole, RoleResult


class MetsukeRole(BaseRole):
    role_key = "metsuke"
    role_name = "目付"
    model_name = "Mistral Small"
    emoji = "🔎"
    default_handled_by = "metsuke_proc"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 目付クライアント未設定 (MISTRAL_API_KEY を確認)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは武士団の目付（Mistral Small）。低中程度の難易度のタスクを"
                "簡潔・実用的な日本語で処理してください。"
                "要約・整形・軽量な推論・情報のとりまとめが得意です。",
            )
            mcp_used = []
            msg = state.get("message", "")

            # 短めのメモリ読み込み
            if len(msg) > 80:
                _results = await self._mcp_parallel([("read_graph", {})])
                mem_ctx = _results[0] if _results else None
                if mem_ctx:
                    system = self._append_mcp_context(system, "ナレッジベース", str(mem_ctx)[:800])
                    mcp_used.append("read_graph")

            messages = self._format_messages(state)
            response = await client.generate(
                messages=messages, system=system, max_tokens=1024,
            )
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=mcp_used,
            )
        except Exception as e:
            self.logger.error("目付実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 目付エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
