"""
utils/anthropic_batch.py — Anthropic Message Batches API ユーティリティ

リアルタイム不要な重い処理（スキル進化・監査ログ解析・大規模要約）を
Batch API 経由で実行し、コストを最大 50% 削減する。

使い方:
    from utils.anthropic_batch import submit_batch, poll_batch_result

    # バッチ送信
    batch_id = await submit_batch([
        {"custom_id": "audit-001", "system": "...", "prompt": "...", "max_tokens": 500},
        {"custom_id": "skill-001", "system": "...", "prompt": "...", "max_tokens": 800},
    ])

    # 結果ポーリング (完了まで待つ)
    results = await poll_batch_result(batch_id)
    # results: {"audit-001": "レスポンステキスト", "skill-001": "..."}

    # fire-and-forget (結果は別途取得)
    await submit_batch_bg([...])
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger("utils.anthropic_batch")

_API_KEY      = os.environ.get("ANTHROPIC_API_KEY", "")
_BASE_URL     = "https://api.anthropic.com/v1"
_HEADERS      = {
    "anthropic-version": "2023-06-01",
    "anthropic-beta":    "message-batches-2024-09-24",
    "content-type":      "application/json",
}
_DEFAULT_MODEL = "claude-opus-4-6"
_POLL_INTERVAL = 30   # 秒
_POLL_TIMEOUT  = 600  # 最大10分待機


def _make_headers() -> dict:
    return {**_HEADERS, "x-api-key": _API_KEY}


async def submit_batch(
    requests: List[Dict],
    model: str = _DEFAULT_MODEL,
) -> Optional[str]:
    """
    複数のメッセージリクエストをバッチ送信する。

    Args:
        requests: リクエストリスト。各要素:
            {
                "custom_id": str,        # 識別ID (結果マッピングに使用)
                "system":    str,        # システムプロンプト (省略可)
                "prompt":    str,        # ユーザープロンプト
                "max_tokens": int,       # 最大トークン (デフォルト 1000)
            }
        model: 使用するモデル (デフォルト claude-opus-4-6)

    Returns:
        バッチID (失敗時 None)
    """
    if not _API_KEY:
        logger.warning("[Batch] ANTHROPIC_API_KEY 未設定 — バッチをスキップ")
        return None

    batch_requests = []
    for req in requests:
        messages = [{"role": "user", "content": req["prompt"]}]
        body: Dict = {
            "model":      model,
            "max_tokens": req.get("max_tokens", 1000),
            "messages":   messages,
        }
        if req.get("system"):
            body["system"] = req["system"]

        batch_requests.append({
            "custom_id": req["custom_id"],
            "params":    body,
        })

    payload = {"requests": batch_requests}

    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{_BASE_URL}/messages/batches",
                headers=_make_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                if r.status not in (200, 201):
                    text = await r.text()
                    logger.error("[Batch] 送信失敗 status=%d: %s", r.status, text[:200])
                    return None
                data = await r.json()
                batch_id = data.get("id")
                logger.info("[Batch] 送信完了: id=%s requests=%d", batch_id, len(requests))
                return batch_id
    except Exception as e:
        logger.error("[Batch] submit_batch エラー: %s", e)
        return None


async def get_batch_status(batch_id: str) -> Optional[Dict]:
    """バッチのステータスを取得する"""
    if not _API_KEY or not batch_id:
        return None
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{_BASE_URL}/messages/batches/{batch_id}",
                headers=_make_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status != 200:
                    return None
                return await r.json()
    except Exception as e:
        logger.debug("[Batch] get_batch_status エラー: %s", e)
        return None


async def poll_batch_result(
    batch_id: str,
    timeout: float = _POLL_TIMEOUT,
    interval: float = _POLL_INTERVAL,
) -> Dict[str, str]:
    """
    バッチ完了まで待機し、結果を返す。

    Returns:
        {custom_id: response_text} のdict。
        エラー・タイムアウト時は空dict。
    """
    if not _API_KEY or not batch_id:
        return {}

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = await get_batch_status(batch_id)
        if not status:
            await asyncio.sleep(interval)
            continue

        processing_status = status.get("processing_status", "")
        if processing_status == "ended":
            return await _fetch_batch_results(batch_id)
        elif processing_status in ("canceling", "canceled"):
            logger.warning("[Batch] バッチキャンセル: %s", batch_id)
            return {}

        counts = status.get("request_counts", {})
        logger.debug(
            "[Batch] ポーリング中: %s status=%s counts=%s",
            batch_id[:12], processing_status, counts,
        )
        await asyncio.sleep(interval)

    logger.warning("[Batch] ポーリングタイムアウト: %s (%.0fs)", batch_id, timeout)
    return {}


async def _fetch_batch_results(batch_id: str) -> Dict[str, str]:
    """完了バッチの結果を取得してパースする"""
    try:
        import json as _json
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{_BASE_URL}/messages/batches/{batch_id}/results",
                headers=_make_headers(),
                timeout=aiohttp.ClientTimeout(total=60),
            ) as r:
                if r.status != 200:
                    logger.error("[Batch] 結果取得失敗 status=%d", r.status)
                    return {}
                text = await r.text()

        results: Dict[str, str] = {}
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            obj = _json.loads(line)
            cid = obj.get("custom_id", "")
            result = obj.get("result", {})
            if result.get("type") == "succeeded":
                content = result.get("message", {}).get("content", [])
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                results[cid] = "".join(text_parts)
            else:
                err = result.get("error", {})
                logger.warning("[Batch] リクエスト失敗 id=%s: %s", cid, err)
                results[cid] = ""

        logger.info("[Batch] 結果取得完了: %s → %d件", batch_id[:12], len(results))
        return results
    except Exception as e:
        logger.error("[Batch] _fetch_batch_results エラー: %s", e)
        return {}


async def submit_batch_bg(requests: List[Dict], model: str = _DEFAULT_MODEL) -> None:
    """
    fire-and-forget バッチ送信。結果は取得しない。
    バックグラウンドタスクとして呼び出す際に使用。
    """
    try:
        batch_id = await submit_batch(requests, model=model)
        if batch_id:
            logger.info("[Batch] バックグラウンド送信完了: %s", batch_id[:12])
    except Exception as e:
        logger.debug("[Batch] submit_batch_bg エラー (無視): %s", e)
