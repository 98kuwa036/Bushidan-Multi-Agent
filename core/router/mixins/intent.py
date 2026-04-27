"""
core/router/mixins/intent.py — 意図分析ノード Mixin

analyze_intent: ショートカット → SemanticRouter → Gemini Flash-Lite → キーワードフォールバック
の順で意図を分析しルーティング情報を返す。
"""
from typing import TYPE_CHECKING, Optional
from utils.logger import get_logger
from core.router.batch.mode import ProcessingMode, BATCH_CONFIG

logger = get_logger(__name__)

if TYPE_CHECKING:
    from core.state import BushidanState


class IntentMixin:
    """analyze_intent ノードとショートカット判定定数を提供する"""

    _GREETING_SET = frozenset([
        "こんにちは", "おはよう", "こんばんは", "やあ", "ねえ", "おい",
        "hello", "hi", "hey", "yo", "morning",
    ])
    _CONF_KWS_FAST = frozenset(["機密", "秘密", "confidential", "社外秘", "オフライン", "外部送信禁止"])
    _JP_WRITING_KWS = frozenset([
        "日本語で書いて", "和訳", "添削", "翻訳して", "ビジネスメール", "敬語で",
        "メールを書", "手紙を書", "校正して",
    ])

    async def _analyze_intent(self, state: "BushidanState") -> dict:
        """受付 (Gemini Flash-Lite) でメッセージを分析しルーティング情報を返す。

        処理順:
        1. ショートカット判定（画像・強制ロール・機密・超短文・日本語文章）
        2. SemanticRouter（INTERACTIVE モードのみ）
        3. 長大コンテキスト圧縮
        4. Gemini Flash-Lite LLM 分析
        5. キーワードフォールバック
        """
        message = state.get("message", "")
        has_vision = bool(state.get("attachments"))
        forced = state.get("forced_role") or ("kengyo" if has_vision else None)
        mode = ProcessingMode(state.get("processing_mode", ProcessingMode.INTERACTIVE))

        # ── ① ショートカット: 画像添付 → kengyo 確定 ──────────────────
        if has_vision:
            logger.info("🚪 [受付] ショートカット: 画像添付 → kengyo")
            return {
                "complexity": "medium", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": False, "is_japanese_priority": False, "is_confidential": False,
                "attachments": state.get("attachments", []), "forced_role": "kengyo",
                "intent_structured": {"intent_type": "image", "domain": "general",
                                      "required_capabilities": ["image"], "user_goal": message[:80]},
            }

        # 強制ロール → ルーティングのみ確定
        if state.get("forced_role"):
            logger.info("🚪 [受付] ショートカット: forced_role=%s", state["forced_role"])
            return {
                "complexity": "medium", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": False, "is_japanese_priority": False, "is_confidential": False,
                "attachments": [], "forced_role": state["forced_role"],
                "intent_structured": {"intent_type": "task", "domain": "general",
                                      "required_capabilities": [], "user_goal": message[:80]},
            }

        msg_lower = message.lower().strip()
        msg_stripped = message.strip()

        # 機密キーワード → onmitsu 確定
        if any(kw in msg_stripped for kw in self._CONF_KWS_FAST):
            logger.info("🚪 [受付] ショートカット: 機密キーワード → onmitsu")
            return {
                "complexity": "medium", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": False, "is_japanese_priority": False, "is_confidential": True,
                "attachments": [], "forced_role": forced,
                "intent_structured": {"intent_type": "confidential", "domain": "security",
                                      "required_capabilities": [], "user_goal": message[:80]},
            }

        # 超短文 (15文字以下) → simple Q&A 確定
        if len(msg_stripped) <= 15:
            is_greeting = any(g in msg_lower for g in self._GREETING_SET)
            logger.info("🚪 [受付] ショートカット: 超短文(%d文字)", len(msg_stripped))
            return {
                "complexity": "simple", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": not is_greeting, "is_japanese_priority": False, "is_confidential": False,
                "attachments": [], "forced_role": forced,
                "intent_structured": {"intent_type": "qa", "domain": "general",
                                      "required_capabilities": [], "user_goal": msg_stripped},
            }

        # 日本語文章作成キーワード → japanese 確定
        if any(kw in msg_stripped for kw in self._JP_WRITING_KWS):
            logger.info("🚪 [受付] ショートカット: 日本語文章作成 → yuhitsu")
            return {
                "complexity": "medium", "is_multi_step": False, "is_action_task": False,
                "is_simple_qa": False, "is_japanese_priority": True, "is_confidential": False,
                "attachments": [], "forced_role": forced,
                "intent_structured": {"intent_type": "japanese", "domain": "general",
                                      "required_capabilities": ["japanese"], "user_goal": message[:80]},
            }

        # ── ② SemanticRouter（INTERACTIVE モードのみ）────────────────────
        use_sr = mode == ProcessingMode.INTERACTIVE or BATCH_CONFIG.get("semantic_router_shortcut", False)
        if use_sr:
            try:
                from utils.semantic_router import SemanticRouter, CONFIDENT_THRESHOLD
                sr = SemanticRouter.get()
                if not sr.is_ready:
                    sr.initialize()
                if sr.is_ready:
                    sem_route, sem_score = sr.route(message)
                    if sem_route and sem_score >= CONFIDENT_THRESHOLD:
                        _ROUTE_TO_ROLE = {
                            "groq_qa":       "seppou",
                            "yuhitsu_jp":    "yuhitsu",
                            "metsuke_proc":  "metsuke",
                            "gunshi_haiku":  "gunshi",
                            "gaiji_rag":     "gaiji",
                            "sanbo_mcp":     "sanbo",
                            "kengyo_vision": "kengyo",
                            "onmitsu_local": "onmitsu",
                            "shogun_plan":   "shogun",
                        }
                        sem_role = _ROUTE_TO_ROLE.get(sem_route)
                        if sem_role:
                            is_jp  = sem_role == "yuhitsu"
                            is_qa  = sem_role == "seppou"
                            is_conf = sem_role == "onmitsu"
                            logger.info("🧭 SemanticRouter ショートカット: %s (%.3f) → %s",
                                        sem_route, sem_score, sem_role)
                            return {
                                "complexity": "simple" if is_qa else "medium",
                                "is_multi_step": False, "is_action_task": False,
                                "is_simple_qa": is_qa, "is_japanese_priority": is_jp,
                                "is_confidential": is_conf,
                                "attachments": [], "forced_role": forced or sem_role,
                                "intent_structured": {
                                    "intent_type": "semantic", "domain": "general",
                                    "required_capabilities": [],
                                    "user_goal": message[:80],
                                    "sem_score": round(sem_score, 3),
                                },
                            }
            except Exception as _sr_err:
                logger.debug("SemanticRouter スキップ: %s", _sr_err)

        logger.info("🚪 [受付] 分析開始: '%s'...", message[:60])

        # ── ③ 長大コンテキスト圧縮 ──────────────────────────────────────
        _history_trimmed: Optional[list] = None
        _summary_text: Optional[str] = None
        history_now = state.get("conversation_history", [])
        if len(history_now) > 12 and not state.get("context_summary"):
            try:
                from utils.client_registry import ClientRegistry
                _sum_client = (
                    ClientRegistry.get().get_client("metsuke")
                    or ClientRegistry.get().get_client("seppou")
                )
                if _sum_client:
                    _old = history_now[:-6]
                    _joined = "\n".join(
                        f"{m['role']}: {str(m.get('content',''))[:300]}"
                        for m in _old
                    )
                    _summary_text = await _sum_client.generate(
                        messages=[{"role": "user", "content":
                            f"以下の会話を3〜5行の要点に圧縮してください:\n{_joined[:3000]}"}],
                        system="会話要約アシスタント。箇条書きで簡潔に。",
                        max_tokens=300,
                    )
                    _history_trimmed = history_now[-6:]
                    logger.info("📝 [受付] 事前コンテキスト圧縮: %d→%d turns", len(history_now), 6)
            except Exception as _ce:
                logger.debug("受付コンテキスト圧縮スキップ: %s", _ce)

        # ── ④ Gemini Flash-Lite による LLM 分析 ─────────────────────────
        try:
            from utils.client_registry import ClientRegistry
            client = ClientRegistry.get().get_client("uketuke")
            if client:
                system = (
                    "あなたは武士団マルチエージェントシステムの受付係です。"
                    "ユーザーのメッセージを分析し、以下の JSON **のみ** を返してください。余分なテキスト不要。\n\n"
                    "{\n"
                    '  "complexity": "simple"|"low_medium"|"medium"|"complex"|"strategic",\n'
                    '  "intent_type": "qa"|"task"|"analysis"|"research"|"code"|"creative"|"rag"|"image"|"japanese"|"confidential",\n'
                    '  "domain": "tech"|"business"|"creative"|"security"|"general",\n'
                    '  "required_capabilities": ["analysis","rag","web_search","code","tools","japanese","image","quick_task"],\n'
                    '  "user_goal": "ユーザーが達成したいことを1文で",\n'
                    '  "is_multi_step": true|false,\n'
                    '  "is_action_task": true|false,\n'
                    '  "is_simple_qa": true|false,\n'
                    '  "is_japanese_priority": true|false,\n'
                    '  "is_confidential": true|false,\n'
                    '  "is_correction": true|false\n'
                    "}\n\n"
                    "判断基準:\n"
                    "- simple: 短い質問・挨拶・事実確認\n"
                    "- low_medium: やや詳しい説明・要約・軽い比較や整理\n"
                    "- medium: 説明・分析・中程度の質問\n"
                    "- complex: コーディング・実装・複数ステップ作業\n"
                    "- strategic: アーキテクチャ設計・戦略立案\n"
                    "- required_capabilities は複数選択可 (空配列も可)\n"
                    "- rag: 既存ドキュメント検索が必要な場合\n"
                    "- web_search: 最新情報・外部情報が必要な場合\n"
                    "- tools: git/ファイル/shell操作が必要な場合\n"
                    "- is_confidential: 機密・社外秘・オフライン指定の場合 true\n"
                    "- is_correction: 前の回答の誤りをユーザーが指摘・訂正している場合 true\n"
                    "  (例: 違います/それは間違い/そうじゃなくて/正しくは〜/ではないです)\n"
                    "⚠️重要: is_japanese_priority は日本語で書かれているだけでは true にしない。"
                    "翻訳・添削・ビジネスメール・敬語変換など日本語文章処理が明示的に求められる場合のみ true。"
                )
                raw = await client.generate(
                    messages=[{"role": "user", "content": message[:1000]}],
                    system=system,
                    max_tokens=120,
                )
                import re
                import json
                m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
                    complexity         = parsed.get("complexity", "medium")
                    is_multi           = bool(parsed.get("is_multi_step", False))
                    is_action          = bool(parsed.get("is_action_task", False))
                    is_simple_qa       = bool(parsed.get("is_simple_qa", False))
                    is_japanese        = bool(parsed.get("is_japanese_priority", False))
                    is_confidential    = bool(parsed.get("is_confidential", False))
                    is_correction      = bool(parsed.get("is_correction", False))
                    intent_structured  = {
                        "intent_type":           parsed.get("intent_type", "general"),
                        "domain":                parsed.get("domain", "general"),
                        "required_capabilities": parsed.get("required_capabilities", []),
                        "user_goal":             parsed.get("user_goal", message[:120]),
                    }
                    logger.info("🚪 [受付] 分析完了: complexity=%s caps=%s",
                                complexity, intent_structured["required_capabilities"])
                    _ret = {
                        "complexity": complexity, "is_multi_step": is_multi,
                        "is_action_task": is_action, "is_simple_qa": is_simple_qa,
                        "is_japanese_priority": is_japanese, "is_confidential": is_confidential,
                        "is_correction": is_correction,
                        "attachments": state.get("attachments", []),
                        "forced_role": forced,
                        "intent_structured": intent_structured,
                    }
                    if _summary_text:
                        _ret["context_summary"] = _summary_text
                    if _history_trimmed is not None:
                        _ret["conversation_history"] = _history_trimmed
                    return _ret
        except Exception as e:
            logger.warning("🚪 [受付] LLM分析失敗 → キーワードフォールバック: %s", e)

        # ── ⑤ キーワードフォールバック ──────────────────────────────────
        content_lower = message.lower()
        strategic_kws = ["設計", "アーキテクチャ", "システム全体", "戦略", "design", "architecture"]
        complex_kws   = ["実装して", "作って", "修正して", "リファクタ", "implement", "build", "refactor"]
        if any(kw in message for kw in strategic_kws) or len(message) > 800:
            complexity = "strategic"
        elif any(kw in message for kw in complex_kws) or len(message) > 300:
            complexity = "complex"
        elif len(message) < 60:
            complexity = "simple"
        else:
            complexity = "medium"
        is_multi        = any(kw in content_lower for kw in ["そして", "次に", "さらに", "まず", "step", "then"]) and len(message) > 100
        is_action       = any(kw in content_lower for kw in ["clone", "push", "pull", "install", "実行して", "削除して", "作成して"])
        is_simple_qa    = any(kw in content_lower for kw in ["とは", "って何", "ですか", "what is", "explain"]) and not is_action
        is_japanese     = any(kw in message for kw in ["日本語で書いて", "和訳", "添削", "翻訳", "ビジネスメール", "敬語で"])
        is_confidential = any(kw in content_lower for kw in ["機密", "秘密", "confidential", "オフライン", "社外秘"])
        _CORRECTION_KWS = ["違います", "違う", "それは間違", "間違ってい", "そうじゃなくて",
                           "そうじゃない", "正しくは", "ではないです", "ではなく",
                           "誤りです", "誤っています", "not correct", "incorrect"]
        is_correction = any(kw in message for kw in _CORRECTION_KWS)
        logger.info("🚪 [受付] フォールバック分析: complexity=%s", complexity)
        _ret = {
            "complexity": complexity, "is_multi_step": is_multi,
            "is_action_task": is_action, "is_simple_qa": is_simple_qa,
            "is_japanese_priority": is_japanese, "is_confidential": is_confidential,
            "is_correction": is_correction,
            "attachments": state.get("attachments", []),
            "forced_role": forced,
            "intent_structured": {"intent_type": "general", "domain": "general",
                                  "required_capabilities": [], "user_goal": message[:120]},
        }
        if _summary_text:
            _ret["context_summary"] = _summary_text
        if _history_trimmed is not None:
            _ret["conversation_history"] = _history_trimmed
        return _ret
