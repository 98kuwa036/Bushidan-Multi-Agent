"""roles/onmitsu.py — 隠密 (Nemotron Local) ロール v18

役割: 機密データ処理・完全ローカル (外部API送信不可)
モデル: Nemotron Local (192.168.11.239 経由)
特徴: 外部APIへのデータ送信なし。機密情報の処理に特化。
      排他制御は LocalModelManager に一元化。
"""

import time
from roles.base import BaseRole, RoleResult

_SYSTEM = (
    "[機密処理モード] 以下の内容を完全にローカルで処理します。"
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

        except RuntimeError as e:
            self.logger.error("隠密: %s", e)
            return RoleResult(
                response=f"⚠️ 隠密: {e}",
                agent_role=self.role_name,
                handled_by=self.default_handled_by,
                execution_time=time.time() - start,
                error=str(e), status="failed",
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
