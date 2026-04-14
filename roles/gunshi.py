"""roles/gunshi.py — 軍師 (Command A) ロール v18

役割: 汎用処理・推論・分析・ステップ実行
モデル: Cohere Command A (command-a-03-2025)
位置: 難易度ティア「medium」担当。
      Command R(外事/RAG特化)と同一ベンダーで役割分担:
      外事=検索+取得、軍師=推論+分析+長文処理。
      Player役: 実際のタスク処理を担うメイン実行エージェント。
"""

import time
from roles.base import BaseRole, RoleResult


class GunshiRole(BaseRole):
    role_key = "gunshi"
    role_name = "軍師"
    model_name = "Command A"
    emoji = "🧠"
    default_handled_by = "gunshi_haiku"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 軍師クライアント未設定 (COHERE_API_KEY を確認)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは武士団の軍師（Command A）。汎用処理・推論・分析・段階的実行の専門家として、"
                "明快・実用的な日本語でタスクを処理してください。"
                "複雑な指示の分解・論理的分析・長文処理・実行ステップの整理が得意です。",
            )
            mcp_used = []
            msg = state.get("message", "")

            # ロードマップのステップ実行コンテキストがあれば注入
            step_task = state.get("_step_task", "")
            if step_task:
                system += f"\n\n【現在のタスク】\n{step_task}"

            # ロードマップ全体のコンテキスト
            roadmap = state.get("roadmap", {})
            if roadmap and roadmap.get("goal"):
                prev_results = state.get("roadmap_results", [])
                if prev_results:
                    prev_ctx = "\n".join(
                        f"Step {r.get('step_id', i+1)}: {r.get('summary', '')}"
                        for i, r in enumerate(prev_results[-3:])
                    )
                    system += f"\n\n【完了済みステップ】\n{prev_ctx}"

            # メモリ読み込み (必要な場合のみ)
            if len(msg) > 100 or step_task:
                _results = await self._mcp_parallel([("read_graph", {})])
                mem_ctx = _results[0] if _results else None
                if mem_ctx:
                    system = self._append_mcp_context(system, "ナレッジベース", str(mem_ctx)[:1000])
                    mcp_used.append("read_graph")

            messages = self._format_messages(state)
            response = await client.generate(
                messages=messages, system=system, max_tokens=2048,
            )
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=mcp_used,
            )
        except Exception as e:
            self.logger.error("軍師実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 軍師エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
