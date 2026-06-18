"""roles/gaiji.py — 外事 (Cohere Command R 08-2024) ロール v18

役割: 外部情報収集・RAG・マルチステップ処理
モデル: Cohere Command R 08-2024 - RAG・ツール使用特化・128K context
MCP:  tavily_search (常時 — 外部情報収集が主務)
データソース: Web検索・気象庁API・為替API・中央銀行RSS
"""

import asyncio
import time
from roles.base import BaseRole, RoleResult


class GaijiRole(BaseRole):
    role_key = "gaiji"
    role_name = "外事"
    model_name = "Command R 08-2024"
    emoji = "🌐"
    default_handled_by = "gaiji_rag"

    # ── 専門データソース トリガーキーワード ────────────────────────────
    _WEATHER_KWS = frozenset(["天気", "気象", "降水", "気温", "台風", "予報", "weather"])
    _FX_KWS      = frozenset(["為替", "ドル", "円", "レート", "外貨", "fx", "usd", "jpy", "eur"])
    _BANK_KWS    = frozenset(["日銀", "fed", "ecb", "金利", "政策金利", "中央銀行", "boj", "利上げ", "利下げ"])

    async def _fetch_weather(self, msg: str) -> str:
        """気象庁API から天気予報を取得（東京: エリアコード 130000）"""
        try:
            import urllib.request, json, re
            # メッセージから都道府県名を検出してエリアコード選択
            area_map = {
                "北海道": "016000", "青森": "020000", "岩手": "030000", "宮城": "040000",
                "秋田": "050000", "山形": "060000", "福島": "070000", "茨城": "080000",
                "栃木": "090000", "群馬": "100000", "埼玉": "110000", "千葉": "120000",
                "東京": "130000", "神奈川": "140000", "新潟": "150000", "富山": "160000",
                "石川": "170000", "福井": "180000", "山梨": "190000", "長野": "200000",
                "岐阜": "210000", "静岡": "220000", "愛知": "230000", "三重": "240000",
                "滋賀": "250000", "京都": "260000", "大阪": "270000", "兵庫": "280000",
                "奈良": "290000", "和歌山": "300000", "鳥取": "310000", "島根": "320000",
                "岡山": "330000", "広島": "340000", "山口": "350000", "徳島": "360000",
                "香川": "370000", "愛媛": "380000", "高知": "390000", "福岡": "400000",
                "佐賀": "410000", "長崎": "420000", "熊本": "430000", "大分": "440000",
                "宮崎": "450000", "鹿児島": "460100", "沖縄": "471000",
            }
            area_code = "130000"
            for name, code in area_map.items():
                if name in msg:
                    area_code = code
                    break
            url = f"https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            area = data[0]["timeSeries"][0]["areas"][0]
            area_name = area.get("area", {}).get("name", "")
            times = data[0]["timeSeries"][0]["timeDefines"][:3]
            weathers = area.get("weathers", [])[:3]
            lines = [f"【気象庁予報 {area_name}】"]
            for t, w in zip(times, weathers):
                lines.append(f"{t[:10]}: {w.replace('　', ' ')[:30]}")
            return "\n".join(lines)
        except Exception as e:
            self.logger.debug("気象庁API 失敗 (スキップ): %s", e)
            return ""

    async def _fetch_fx(self) -> str:
        """frankfurter.app から主要通貨の対円レートを取得"""
        try:
            import urllib.request, json
            url = "https://api.frankfurter.app/latest?from=USD&to=JPY,EUR,GBP,CNY,KRW,AUD"
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            rates = data.get("rates", {})
            date  = data.get("date", "")
            lines = [f"【為替レート ({date}) 基準: USD】"]
            for cur, val in rates.items():
                lines.append(f"  USD/{cur}: {val:.2f}")
            return "\n".join(lines)
        except Exception as e:
            self.logger.debug("為替API 失敗 (スキップ): %s", e)
            return ""

    async def _fetch_boj_rss(self) -> str:
        """日本銀行 RSS から最新リリースを取得"""
        try:
            import urllib.request
            from xml.etree import ElementTree as ET
            url = "https://www.boj.or.jp/rss/news.rss"
            with urllib.request.urlopen(url, timeout=5) as r:
                root = ET.fromstring(r.read())
            ns = {"": ""}
            items = root.findall("./channel/item")[:5]
            lines = ["【日本銀行 最新リリース】"]
            for item in items:
                title = item.findtext("title", "").strip()
                date  = item.findtext("pubDate", "")[:16]
                if title:
                    lines.append(f"  {date} {title}")
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception as e:
            self.logger.debug("日銀RSS 失敗 (スキップ): %s", e)
            return ""

    async def execute(self, state: dict) -> RoleResult:
        start = time.time()
        client = self._get_client()
        if not client:
            return RoleResult(
                response="⚠️ 外事クライアント未設定 (COHERE_API_KEY)",
                agent_role=self.role_name, handled_by=self.default_handled_by,
                execution_time=time.time() - start, error="client not available", status="failed",
            )
        try:
            system = self._build_system_prompt(
                state,
                "あなたは外事担当 (Command R)。外部情報収集・RAG・マルチステップ処理の専門家です。"
                "最新の検索結果・公的データを活用して正確で包括的な回答を日本語で提供してください。",
            )
            mcp_used = []
            msg = state.get("message", "")

            # ── 専門データソース (並列取得) ──────────────────────────────
            fetch_tasks = []
            fetch_labels = []

            if any(kw in msg for kw in self._WEATHER_KWS):
                fetch_tasks.append(self._fetch_weather(msg))
                fetch_labels.append("気象庁予報")

            if any(kw in msg for kw in self._FX_KWS):
                fetch_tasks.append(self._fetch_fx())
                fetch_labels.append("為替レート")

            if any(kw in msg for kw in self._BANK_KWS):
                fetch_tasks.append(self._fetch_boj_rss())
                fetch_labels.append("日銀リリース")

            if fetch_tasks:
                results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                for label, result in zip(fetch_labels, results):
                    if isinstance(result, str) and result:
                        system = self._append_mcp_context(system, label, result)
                        mcp_used.append(label)

            # ── Web 検索 (常時実行 — 外部情報収集が主務) ──────────────────
            web_ctx = await self._mcp_search(msg[:300], max_results=5)
            if web_ctx:
                system = self._append_mcp_context(system, "Web検索結果", web_ctx)
                mcp_used.append("tavily_search")

            if mcp_used:
                self.logger.info("🌐 外事: データソース使用 %s", mcp_used)

            messages = self._format_messages(state)
            response = await client.generate(messages=messages, system=system, max_tokens=2048)
            return RoleResult(
                response=response, agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                requires_followup=self._needs_followup(response, state),
                mcp_tools_used=mcp_used,
            )
        except Exception as e:
            self.logger.error("外事実行失敗: %s", e)
            return RoleResult(
                response=f"❌ 外事エラー: {e}", agent_role=self.role_name,
                handled_by=self.default_handled_by, execution_time=time.time() - start,
                error=str(e), status="failed",
            )
