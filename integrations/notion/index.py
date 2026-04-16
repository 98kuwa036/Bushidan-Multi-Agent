"""integrations/notion/index.py — Notion ローカルインデックス v16

Notion全文検索の代替。
  - ページのタイトル + 2〜3文概要を data/notion_index.json にキャッシュ
  - 起動時 + 2時間ごとに自動更新
  - lookup(user_goal) はローカル検索のみ (ネットワーク不要)
  - ヒットしたページのみ Notion API でブロック取得 (最大2件)

返り値: List[NotionChunk] (retrieval.py 互換)
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional

# fire-and-forget タスク参照保持（GC 対策）
_bg_tasks: set = set()


def _fire(coro, *, name: str = None) -> "asyncio.Task":
    t = asyncio.create_task(coro, name=name)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return t

logger = logging.getLogger("integrations.notion.index")

# ローカルキャッシュファイルのパス
_INDEX_PATH = Path("/mnt/Bushidan-Multi-Agent/data/notion_index.json")
_REFRESH_INTERVAL = 7200  # 2時間

# インメモリキャッシュ
_index_cache: list = []
_last_refresh: float = 0.0
_refresh_lock = asyncio.Lock()


# ── インデックス構造 ──────────────────────────────────────────────────────
# [{
#   "page_id": str,
#   "title": str,
#   "summary": str,      # 2〜3文
#   "tags": [str],
#   "entry_type": str,   # "knowledge" | "task" | "page"
#   "updated_at": str,
# }]


async def refresh_index() -> int:
    """Notion からページ一覧を取得し、タイトル+概要でインデックスを再構築。
    保存件数を返す。"""
    global _index_cache, _last_refresh
    async with _refresh_lock:
        try:
            from integrations.notion.client import get_notion_client
            client = get_notion_client()

            entries = []

            # ナレッジベースDBを検索
            try:
                kb_db = os.environ.get("NOTION_KB_DATABASE_ID", "")
                if kb_db:
                    pages = await client.search_database(kb_db, query="", page_size=50)
                    for page in pages:
                        entry = await _page_to_entry(client, page, "knowledge")
                        if entry:
                            entries.append(entry)
            except Exception as e:
                logger.debug("KB DB インデックス取得スキップ: %s", e)

            # 全ページをテキスト検索 (追加分)
            try:
                results = await client.search_all_text("", page_size=30)
                existing_ids = {e["page_id"] for e in entries}
                for page in results:
                    pid = page.get("id", "")
                    if pid not in existing_ids:
                        entry = await _page_to_entry(client, page, "page")
                        if entry:
                            entries.append(entry)
            except Exception as e:
                logger.debug("全ページインデックス取得スキップ: %s", e)

            if entries:
                _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
                _INDEX_PATH.write_text(
                    json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _index_cache = entries
                _last_refresh = time.time()
                logger.info("✅ Notionインデックス更新: %d件 → %s", len(entries), _INDEX_PATH)
                return len(entries)
            else:
                logger.warning("⚠️ Notionインデックス: 0件取得 (スキップ)")
                return 0

        except Exception as e:
            logger.warning("Notionインデックス更新失敗: %s", e)
            return 0


async def _page_to_entry(client, page: dict, entry_type: str) -> Optional[dict]:
    """ページオブジェクトからインデックスエントリを生成。"""
    try:
        page_id = page.get("id", "")
        if not page_id:
            return None

        # タイトル取得
        title = await client.get_page_title(page)
        if not title or title == "Untitled":
            return None

        # 概要: プロパティの description/summary があれば使用、なければ本文冒頭
        summary = _extract_summary_from_props(page)
        if not summary:
            try:
                content = await client.get_page_content(page_id)
                # 最初の400文字から2〜3文を抽出
                text = content[:500].replace("\n", " ").strip()
                sentences = re.split(r"[。！？\.!?]", text)
                summary = "。".join(s.strip() for s in sentences[:3] if s.strip())
                if summary and not summary.endswith("。"):
                    summary += "。"
            except Exception:
                summary = ""

        if not summary:
            return None

        # タグ (properties から)
        tags = _extract_tags(page)

        updated = page.get("last_edited_time", "")[:10]

        return {
            "page_id":    page_id,
            "title":      title,
            "summary":    summary[:300],
            "tags":       tags,
            "entry_type": entry_type,
            "updated_at": updated,
        }
    except Exception as e:
        logger.debug("ページエントリ変換失敗: %s", e)
        return None


def _extract_summary_from_props(page: dict) -> str:
    props = page.get("properties", {})
    for key in ("description", "Description", "summary", "Summary", "概要", "説明"):
        p = props.get(key, {})
        if p.get("type") == "rich_text":
            texts = p.get("rich_text", [])
            if texts:
                return "".join(t.get("plain_text", "") for t in texts)[:300]
    return ""


def _extract_tags(page: dict) -> list:
    props = page.get("properties", {})
    for key in ("tags", "Tags", "タグ", "category", "Category"):
        p = props.get(key, {})
        if p.get("type") == "multi_select":
            return [opt.get("name", "") for opt in p.get("multi_select", [])]
    return []


async def _load_cache() -> list:
    """ローカルキャッシュファイルを読み込む。"""
    global _index_cache, _last_refresh
    if _index_cache:
        return _index_cache
    if _INDEX_PATH.exists():
        try:
            data = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
            _index_cache = data
            _last_refresh = _INDEX_PATH.stat().st_mtime
            logger.info("📂 Notionインデックスをキャッシュから読み込み: %d件", len(data))
            return data
        except Exception as e:
            logger.warning("キャッシュ読み込み失敗: %s", e)
    return []


async def get_index() -> list:
    """インデックスを返す。必要なら自動更新。"""
    cache = await _load_cache()
    now = time.time()
    if not cache or (now - _last_refresh) > _REFRESH_INTERVAL:
        _fire(refresh_index(), name="notion_index_refresh")
    return cache or []


def _score_entry(entry: dict, tokens: list[str]) -> float:
    """エントリとクエリトークンのスコアを計算。"""
    score = 0.0
    title_lower = entry.get("title", "").lower()
    summary_lower = entry.get("summary", "").lower()
    tags_lower = " ".join(entry.get("tags", [])).lower()

    for tok in tokens:
        if tok in title_lower:
            score += 2.0
        if tok in summary_lower:
            score += 0.8
        if tok in tags_lower:
            score += 1.2

    return score


def _tokenize(text: str) -> list[str]:
    """簡易トークナイズ: 英単語 + 日本語2文字gram。"""
    tokens = []
    # 英単語
    for w in re.findall(r"[a-zA-Z]{2,}", text.lower()):
        tokens.append(w)
    # 日本語文字 bigram
    ja = re.sub(r"[^\u3040-\u9fff]", "", text)
    for i in range(len(ja) - 1):
        tokens.append(ja[i:i+2])
    # 日本語単語 (助詞で区切る簡易版)
    for w in re.split(r"[\s、。！？「」『』【】\(\)\[\]]+", text):
        if len(w) >= 2:
            tokens.append(w.lower())
    return list(set(tokens))


async def lookup(user_goal: str, top_k: int = 3) -> list:
    """user_goal に関連するインデックスエントリを返す。
    ヒットしたページの詳細を Notion API で取得して NotionChunk リストを返す。
    ヒットなしなら空リスト。
    """
    if not user_goal or not user_goal.strip():
        return []

    index = await get_index()
    if not index:
        return []

    tokens = _tokenize(user_goal)
    if not tokens:
        return []

    scored = [(entry, _score_entry(entry, tokens)) for entry in index]
    scored.sort(key=lambda x: x[1], reverse=True)

    # スコア閾値 1.5 以上のみ採用
    hits = [(e, s) for e, s in scored[:top_k] if s >= 1.5]
    if not hits:
        return []

    logger.info("📋 Notionインデックスヒット: %d件 (goal='%s')", len(hits), user_goal[:40])

    # 詳細コンテンツを並列取得 (最大2件)
    chunks = []
    try:
        from integrations.notion.client import get_notion_client
        from integrations.notion.retrieval import NotionChunk
        client = get_notion_client()

        async def _fetch(entry: dict, score: float):
            try:
                content = await client.get_page_content(entry["page_id"])
                return NotionChunk(
                    title=entry["title"],
                    content=content[:800],
                    page_id=entry["page_id"],
                    url=f"https://notion.so/{entry['page_id'].replace('-','')}",
                    score=score,
                    entry_type=entry.get("entry_type", "page"),
                )
            except Exception as e:
                logger.debug("詳細取得失敗 %s: %s", entry["page_id"], e)
                # 概要のみで返す
                return NotionChunk(
                    title=entry["title"],
                    content=entry.get("summary", ""),
                    page_id=entry["page_id"],
                    url="",
                    score=score,
                    entry_type=entry.get("entry_type", "page"),
                )

        results = await asyncio.gather(*[_fetch(e, s) for e, s in hits[:2]], return_exceptions=True)
        chunks = [r for r in results if r is not None]
    except Exception as e:
        logger.warning("Notionチャンク取得失敗: %s", e)

    return chunks
