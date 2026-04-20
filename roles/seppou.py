"""roles/seppou.py — 斥候 (Llama 3.3 70B Groq) ロール v18

役割: 高速フィルタ・雑談・シンプルQ&A・簡易コード爆速書き上げ
モデル: Meta Llama 3.3 70B (Groq)
MCP:  tavily_search (検索トリガーキーワード時)
Post: Gemini Flash（参謀）によるコードレビューループ
      複雑なロジックエラー時は Command A（軍師）にエスカレーション
"""

import re
import time
from roles.base import BaseRole, RoleResult

_CODE_KWS = [
    "書いて", "作って", "実装", "関数", "スクリプト", "コード",
    "write", "create", "implement", "function", "script", "code",
    "snippet", "example", "サンプル",
]

_LANG_MAP = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "go": "go", "rust": "rust", "bash": "bash",
    "sh": "bash", "sql": "sql", "yaml": "yaml",
}


def _has_code_request(msg: str) -> bool:
    return any(kw in msg for kw in _CODE_KWS)


def _detect_language(msg: str) -> str:
    """メッセージから対象言語を推定"""
    lower = msg.lower()
    for key, lang in _LANG_MAP.items():
        if key in lower:
            return lang
    return "python"  # デフォルト


def _extract_first_code(text: str) -> tuple[str, str]:
    """最初のコードブロックを (language, code) で返す"""
    m = re.search(r'```(\w*)\n(.*?)```', text, re.DOTALL)
    if m:
        lang = _LANG_MAP.get(m.group(1).lower(), m.group(1).lower() or "python")
        return lang, m.group(2).strip()
    return "python", ""


class SeppouRole(BaseRole):
    role_key = "seppou"
    role_name = "斥候"
    model_name = "Llama 3.3 70B (Groq)"
    emoji = "🏹"
    default_handled_by = "groq_qa"

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 斥候クライアント未設定 (GROQ_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            msg = state.get("message", "")
            is_code = _has_code_request(msg)

            system = self._build_system_prompt(
                state,
                "あなたは斥候担当。簡潔で正確な回答を日本語で素早く返してください。"
                + ("コードを書く場合は実用的で動作するコードをコードブロックで提供してください。"
                   if is_code else ""),
            )
            mcp_used = []

            # ── Web 検索 ──────────────────────────────────────────────
            if self._needs_web_search(msg):
                web_ctx = await self._mcp_search(msg[:200], max_results=3)
                if web_ctx:
                    system = self._append_mcp_context(system, "最新情報", web_ctx)
                    mcp_used.append("tavily_search")

            max_tokens = 2048 if is_code else 512
            messages = self._format_messages(state)
            response = await client.generate(
                messages=messages, system=system, max_tokens=max_tokens,
            )

            # ── コードレビューループ ───────────────────────────────────
            if is_code:
                lang, code = _extract_first_code(response)
                if code:
                    try:
                        from utils.code_quality_loop import run_review_loop, format_review_summary
                        from utils.client_registry import ClientRegistry

                        review_client     = ClientRegistry.get().get_client("sanbo")    # Gemini Flash
                        escalation_client = ClientRegistry.get().get_client("gunshi")   # Command A（Noneでも可）

                        if review_client:
                            final_code, history, rounds = await run_review_loop(
                                code=code,
                                requirements=msg,
                                language=lang,
                                groq_client=client,
                                review_client=review_client,
                                escalation_client=escalation_client,
                            )

                            summary = format_review_summary(history)

                            # 最終コードでレスポンスを再構成
                            response = (
                                f"```{lang}\n{final_code}\n```\n\n"
                                f"---\n{summary}"
                            )
                            mcp_used.append(f"code_review_loop(x{rounds})")
                            self.logger.info("🏹 斥候: コードレビューループ完了 (%dラウンド)", rounds)

                    except Exception as e:
                        self.logger.debug("コードレビュースキップ: %s", e)

            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=mcp_used,
            )
        except Exception as e:
            self.logger.error("斥候実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 斥候エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
