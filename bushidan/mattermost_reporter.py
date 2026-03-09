"""
武士団 Mattermost レポーター v12

各エージェント（役職）が自分専用のMattermostアカウントから発言する。
双方向の軍議会話を実現する。

v12 メンバー構成 (10役職):
  受付 (uketuke)  - Command R       : ルーター・受付
  外事 (gaiji)    - Command R+      : 外部ツール・RAG
  検校 (kengyo)   - Gemini Vision   : 視覚解析
  将軍 (shogun)   - Claude Sonnet   : メインワーカー
  軍師 (gunshi)   - o3-mini         : 深層推論
  参謀 (sanbo)    - Mistral Large 3 : 中量級推論
  右筆 (yuhitsu)  - Llama ELYZA     : 日本語清書
  斥候 (seppou)   - Llama 3.3 Groq  : 高速フィルタ
  隠密 (onmitsu)  - Nemotron Local  : 完全ローカル
  大元帥 (daigensui) - Claude Opus  : 最終エスカレーション

使い方:
    reporter = MattermostReporter(base_url, channel_id)
    await reporter.post_as("gunshi", "軍師の分析結果はこちらです...")
    await reporter.post_reply_as("daigensui", root_post_id, "承認します")
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from mattermostdriver.driver import Driver

logger = logging.getLogger("bushidan.mattermost_reporter")

# ─── エージェント設定 v12 (10役職) ───────────────────────────────────────────
# 全トークン設定済み (2026-03-09)
AGENT_CONFIG: Dict[str, Dict[str, str]] = {
    # ── 受付 (uketuke) — Command R ──────────────────────────────────────────
    "uketuke": {
        "token":   "8zp818ckfj81zptpubwh6zqtse",
        "display": "受付",
        "model":   "Command R",
        "emoji":   "🚪",
    },
    # ── 外事 (gaiji) — Command R+ ───────────────────────────────────────────
    "gaiji": {
        "token":   "zfcekita5pdqfmjag1y3utw4ke",
        "display": "外事",
        "model":   "Command R+",
        "emoji":   "🌐",
    },
    # ── 検校 (kengyo) — Gemini Flash Vision ─────────────────────────────────
    "kengyo": {
        "token":   "m9x519cbqb8hff5e5jcbpmqz8a",
        "display": "検校",
        "model":   "Gemini 3 Flash Vision",
        "emoji":   "👁️",
    },
    # ── 将軍 (shogun) — Claude Sonnet 4.6 ───────────────────────────────────
    "shogun": {
        "token":   "ptfkjhgsdpgqxbfpnk36ybwy1r",
        "display": "将軍",
        "model":   "Claude Sonnet 4.6",
        "emoji":   "🏯",
    },
    # ── 軍師 (gunshi) — o3-mini ─────────────────────────────────────────────
    "gunshi": {
        "token":   "6n3ddwssh3njijtjjk7nhgxfew",
        "display": "軍師",
        "model":   "o3-mini",
        "emoji":   "📜",
    },
    # ── 参謀 (sanbo) — Mistral Large 3 (旧: sanbo-a-bot を流用) ─────────────
    "sanbo": {
        "token":   "xjc9at7uzbn49bpj8dusea5ypo",
        "display": "参謀",
        "model":   "Mistral Large 3",
        "emoji":   "🗡️",
    },
    # ── 右筆 (yuhitsu) — Llama ELYZA ────────────────────────────────────────
    "yuhitsu": {
        "token":   "9ktzxdari7dkfjw9np66aznfby",
        "display": "右筆",
        "model":   "Llama ELYZA",
        "emoji":   "🖊️",
    },
    # ── 斥候 (seppou) — Llama 3.3 70B Groq ──────────────────────────────────
    "seppou": {
        "token":   "iff7zm9mai86fcik6higjox4ua",
        "display": "斥候",
        "model":   "Llama 3.3 70B (Groq)",
        "emoji":   "🏹",
    },
    # ── 隠密 (onmitsu) — Nemotron Local ─────────────────────────────────────
    "onmitsu": {
        "token":   "ekff6cdkytdkdq7o5msb76s5ea",
        "display": "隠密",
        "model":   "Nemotron-3-Nano (Local)",
        "emoji":   "🥷",
    },
    # ── 大元帥 (daigensui) — Claude Opus 4.6 ────────────────────────────────
    "daigensui": {
        "token":   "81cag4cjdbd8jmsn8jmb5o4ohr",
        "display": "大元帥",
        "model":   "Claude Opus 4.6",
        "emoji":   "⚔️",
    },
}

# ─── 後方互換エイリアス (旧キー → 新キー) ────────────────────────────────────
_AGENT_ALIASES: Dict[str, str] = {
    "sanbo_a": "sanbo",
    "sanbo_b": "sanbo",
    "karo_a":  "seppou",   # Gemini Flash → 斥候 Groq (フォールバック)
    "karo_b":  "seppou",
}

# ─── LangGraph ノード名 → エージェントキー マッピング v12 ─────────────────────
NODE_TO_AGENT: Dict[str, str] = {
    # 直接指定
    "daigensui":     "daigensui",
    "shogun":        "shogun",
    "gunshi":        "gunshi",
    "sanbo":         "sanbo",
    "kengyo":        "kengyo",
    "onmitsu":       "onmitsu",
    "uketuke":       "uketuke",
    "gaiji":         "gaiji",
    "yuhitsu":       "yuhitsu",
    "seppou":        "seppou",
    # LangGraph ノード名
    "gunshi_pdca":   "gunshi",
    "gaiji_rag":     "gaiji",
    "yuhitsu_jp":    "yuhitsu",
    "taisho_mcp":    "shogun",       # 参謀が MCP 統括 (将軍投稿)
    "groq_qa":       "seppou",       # 斥候 Llama 3.3 Groq
    "karo_default":  "uketuke",      # 受付 Command R フォールバック
    "onmitsu_local": "onmitsu",
    "onmitsu_nemotron":         "onmitsu",
    "onmitsu_nemotron_fallback": "onmitsu",
    "yuhitsu_elyza":            "yuhitsu",
    "yuhitsu_nemotron":         "yuhitsu",
    # 後方互換 (旧ノード名)
    "gemini_autonomous": "gaiji",
    "sanbo_a":           "sanbo",
    "sanbo_b":           "sanbo",
    "karo_a":            "seppou",
    "karo_b":            "seppou",
    # フォールバック
    "router":       "uketuke",
    "orchestrator": "daigensui",
    "unknown":      "shogun",
}


class AgentDriver:
    """エージェント専用の軽量Mattermost Driverラッパー。"""

    def __init__(self, token: str, url: str, port: int, scheme: str) -> None:
        self._driver = Driver({
            "url": url,
            "token": token,
            "port": port,
            "scheme": scheme,
            "verify": False,
        })
        self._logged_in = False

    async def _ensure_login(self) -> None:
        if not self._logged_in:
            await asyncio.to_thread(self._driver.login)
            self._logged_in = True

    async def create_post(self, options: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_login()
        return await asyncio.to_thread(self._driver.posts.create_post, options=options)


class MattermostReporter:
    """
    エージェントごとに専用アカウントでMattermostに投稿する。

    Args:
        url:        Mattermostサーバーホスト (例: "192.168.11.234")
        port:       ポート番号 (例: 8065)
        channel_id: デフォルトチャンネルID
        scheme:     "http" または "https"
    """

    def __init__(
        self,
        url: str,
        port: int,
        channel_id: str,
        scheme: str = "http",
    ) -> None:
        self._url = url
        self._port = port
        self._channel_id = channel_id
        self._scheme = scheme
        self._drivers: Dict[str, AgentDriver] = {}

    def _resolve_agent_key(self, agent_key: str) -> str:
        """エイリアス (旧キー) を正規キーに解決する。"""
        return _AGENT_ALIASES.get(agent_key, agent_key)

    def _get_driver(self, agent_key: str) -> AgentDriver:
        agent_key = self._resolve_agent_key(agent_key)
        if agent_key not in self._drivers:
            cfg = AGENT_CONFIG.get(agent_key)
            if not cfg:
                raise ValueError(f"Unknown agent: {agent_key}")
            token = cfg["token"]
            # PLACEHOLDER トークンはスキップ (投稿はメインbotに委ねる)
            if token.startswith("PLACEHOLDER_"):
                logger.warning(
                    "[%s] Mattermostトークン未設定 (PLACEHOLDER)。投稿をスキップします。",
                    cfg["display"],
                )
                raise ValueError(f"Agent {agent_key} has no Mattermost account yet")
            self._drivers[agent_key] = AgentDriver(
                token=token,
                url=self._url,
                port=self._port,
                scheme=self._scheme,
            )
        return self._drivers[agent_key]

    async def post_as(
        self,
        agent_key: str,
        message: str,
        channel_id: Optional[str] = None,
        root_id: str = "",
        props: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        指定エージェントとして投稿する。

        Returns:
            投稿ID (エラー時は None)
        """
        agent_key = self._resolve_agent_key(agent_key)
        cfg = AGENT_CONFIG.get(agent_key)
        if not cfg:
            logger.warning("Unknown agent key: %s", agent_key)
            return None

        try:
            drv = self._get_driver(agent_key)
        except ValueError as e:
            logger.warning("投稿スキップ: %s", e)
            return None
        options: Dict[str, Any] = {
            "channel_id": channel_id or self._channel_id,
            "message": message,
        }
        if root_id:
            options["root_id"] = root_id
        if props:
            options["props"] = props

        try:
            post = await drv.create_post(options)
            post_id = post.get("id", "")
            logger.debug("[%s] posted: %s", cfg["display"], post_id)
            return post_id
        except Exception as e:
            logger.error("[%s] 投稿失敗: %s", cfg["display"], e)
            return None

    async def post_reply_as(
        self,
        agent_key: str,
        root_id: str,
        message: str,
        channel_id: Optional[str] = None,
    ) -> Optional[str]:
        """スレッド返信として投稿する。"""
        return await self.post_as(
            agent_key, message, channel_id=channel_id, root_id=root_id
        )

    async def post_from_node(
        self,
        node_name: str,
        message: str,
        root_id: str = "",
        channel_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        LangGraphのノード名からエージェントを解決して投稿する。

        Args:
            node_name: LangGraphのノード名 (例: "gunshi", "onmitsu_local")
            message:   投稿本文
            root_id:   スレッドルートID
        """
        agent_key = NODE_TO_AGENT.get(node_name, "shogun")
        cfg = AGENT_CONFIG.get(agent_key, {})
        emoji = cfg.get("emoji", "🤖")
        model = cfg.get("model", node_name)

        # ヘッダー付きメッセージ
        full_message = f"{emoji} **[{model}]**\n\n{message}"

        return await self.post_as(
            agent_key,
            full_message,
            root_id=root_id,
            channel_id=channel_id,
        )

    def resolve_agent(self, node_name: str) -> str:
        """ノード名からエージェントキーを返す。"""
        return NODE_TO_AGENT.get(node_name, "shogun")

    @property
    def available_agents(self):
        return list(AGENT_CONFIG.keys())
