"""roles/onmitsu.py — 隠密 (Nemotron Local) ロール v18

役割: 機密データ処理・完全ローカル (外部API送信不可)
モデル: Nemotron Local（プライマリ）→ Gemma4 Local（第二）→ エラー
特徴: 外部APIへのデータ送信なし。機密情報の処理に特化。
      排他制御は LocalModelManager に一元化。
フォールバック: 同一サーバーの Gemma4 のみ（クラウドAPIは使用不可）
"""

import time
from roles.base import BaseRole, RoleResult

_SYSTEM = (
    "[機密処理モード] 以下の内容を完全にローカルで処理します。"
    "外部APIには送信されません。明確・簡潔な日本語で回答してください。"
)

_GEMMA_SYSTEM = (
    "[機密処理モード — Gemmaフォールバック] Nemotronが利用不可のため Gemma4 で処理します。"
    "外部APIには送信されません。明確・簡潔な日本語で回答してください。"
)


class OnmitsuRole(BaseRole):
    role_key = "onmitsu"
    role_name = "隠密"
    model_name = "Nemotron Local"
    emoji = "🥷"
    default_handled_by = "onmitsu_local"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        try:
            from utils.local_model_manager import LocalModelManager
            manager = LocalModelManager.get()

            message = state.get("message", "")
            system = self._build_system_prompt(state, _SYSTEM)
            mcp_used = []

            _FILE_KWS = ["ファイル", "読んで", "保存して", "書いて", "作成して",
                         "file", "read", "write", "save"]
            needs_file = any(kw in message for kw in _FILE_KWS)

            if needs_file:
                for ref in self._extract_file_refs(message)[:2]:
                    content = await self._mcp_read_file(ref)
                    if content:
                        system = self._append_mcp_context(system, f"ファイル: {ref}", content)
                        mcp_used.append("read_file")

            # ── プライマリ: Nemotron ───────────────────────────────────
            try:
                prompt = f"{system}\n\nユーザー: {message}\n\n隠密:"
                response = await manager.generate_nemotron(prompt, max_tokens=4096)
                self.logger.info("🥷 隠密 [Nemotron] MCP=%s %.1fs", mcp_used, time.time() - start)
                return RoleResult(
                    response=response,
                    agent_role=self.role_name,
                    handled_by=self.default_handled_by,
                    execution_time=time.time() - start,
                    requires_followup=self._needs_followup(response, state),
                    mcp_tools_used=mcp_used,
                )
            except Exception as e_nem:
                self.logger.warning("隠密 [Nemotron] 失敗、Gemma4フォールバックへ: %s", e_nem)

            # ── 第二フォールバック: Gemma4 (同一ローカルサーバー) ──────
            try:
                gemma_system = self._build_system_prompt(state, _GEMMA_SYSTEM)
                prompt = f"{gemma_system}\n\nユーザー: {message}\n\n隠密:"
                response = await manager.generate_gemma(prompt, max_tokens=2048)
                self.logger.info("🥷 隠密 [Gemma フォールバック] %.1fs", time.time() - start)
                return RoleResult(
                    response=response,
                    agent_role=self.role_name,
                    handled_by=self.default_handled_by,
                    execution_time=time.time() - start,
                    requires_followup=self._needs_followup(response, state),
                    mcp_tools_used=mcp_used + ["gemma_fallback"],
                )
            except Exception as e_gemma:
                self.logger.error("隠密 [Gemma フォールバック] 失敗: %s", e_gemma)

            # ── 完全ダウン: クラウドAPIは使用不可 ─────────────────────
            return RoleResult(
                response=(
                    "⚠️ **隠密オフライン**\n\n"
                    "ローカルLLMサーバー (192.168.11.239) が現在利用できません。\n"
                    "機密データは外部APIに送信できないため、処理を中断しました。\n\n"
                    "**復旧手順:**\n"
                    "```\nssh 192.168.11.239 'systemctl --user start local_llm_server'\n```\n"
                    "または管理者にサーバーの確認を依頼してください。"
                ),
                agent_role=self.role_name,
                handled_by=self.default_handled_by,
                execution_time=time.time() - start,
                error="local_llm_unavailable",
                status="failed",
            )

        except Exception as e:
            self.logger.error("隠密実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 隠密エラー: {e}",
                agent_role=self.role_name,
                handled_by=self.default_handled_by,
                execution_time=time.time() - start,
                error=str(e), status="failed",
            )
