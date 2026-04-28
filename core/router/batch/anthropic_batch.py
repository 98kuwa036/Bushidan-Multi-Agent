"""
core/router/batch/anthropic_batch.py — Anthropic Messages Batch API プロセッサ

BATCH_CONFIG["use_anthropic_batch"] = True のとき、
Anthropic モデル (shogun / daigensui) への複数呼び出しを
一括送信して API コスト 50% 削減。

Anthropic Batch API の特性:
- 非同期処理: 送信後数秒〜24 時間で完了
- コスト: 通常 API の 50%
- 対象: claude-* モデルのみ (Groq / Gemini 不可)
- タイムアウト: 24 時間
"""
import asyncio
import os
from typing import Any
from utils.logger import get_logger

logger = get_logger(__name__)

# Anthropic モデルを使うロールキー（Batch API 送信対象）
ANTHROPIC_ROLES = frozenset({"shogun", "daigensui"})

# ロールキー → デフォルトモデル ID
_ROLE_MODEL: dict[str, str] = {
    "shogun":    "claude-sonnet-4-6",
    "daigensui": "claude-opus-4-6",
}


class AnthropicBatchProcessor:
    """
    Anthropic Messages Batch API を使ったバッチ一括処理。

    使用例:
        processor = AnthropicBatchProcessor.from_env()
        results = await processor.run([
            {"custom_id": "s1", "role_key": "shogun",
             "messages": [...], "system": "...", "max_tokens": 800},
        ])
        text = results["s1"]   # 応答テキスト
    """

    def __init__(
        self,
        api_key: str,
        poll_interval: float = 5.0,    # 開発時は短め; 本番は 60 以上推奨
        max_wait: float = 3600.0,      # 最大 1 時間待機
    ):
        self._api_key = api_key
        self._poll_interval = poll_interval
        self._max_wait = max_wait

    @classmethod
    def from_env(cls) -> "AnthropicBatchProcessor":
        """環境変数 ANTHROPIC_API_KEY からインスタンスを生成する。"""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY が未設定です")
        try:
            poll_interval = float(os.environ.get("ANTHROPIC_BATCH_POLL_INTERVAL", "5.0"))
        except ValueError as e:
            raise RuntimeError("ANTHROPIC_BATCH_POLL_INTERVAL は数値で指定してください") from e
        if poll_interval <= 0:
            raise RuntimeError("ANTHROPIC_BATCH_POLL_INTERVAL は 0 より大きい値を指定してください")
        try:
            max_wait = float(os.environ.get("ANTHROPIC_BATCH_MAX_WAIT", "3600.0"))
        except ValueError as e:
            raise RuntimeError("ANTHROPIC_BATCH_MAX_WAIT は数値で指定してください") from e
        if max_wait <= 0:
            raise RuntimeError("ANTHROPIC_BATCH_MAX_WAIT は 0 より大きい値を指定してください")
        return cls(api_key=api_key, poll_interval=poll_interval, max_wait=max_wait)

    # ── public API ───────────────────────────────────────────────────────────

    async def run(self, requests: list[dict]) -> dict[str, tuple[str, str | None]]:
        """
        複数リクエストを Batch API に送信し、全て完了したら結果を返す。

        Args:
            requests: 各要素は以下のキーを持つ dict
                - custom_id (str): リクエスト識別子
                - role_key  (str): "shogun" | "daigensui"
                - messages  (list): [{"role": ..., "content": ...}]
                - system    (str, optional): システムプロンプト
                - max_tokens (int, optional): デフォルト 2000

        Returns:
            {custom_id: (response_text, error_msg | None)}
            error_msg は失敗時のみ設定、成功時は None
        """
        if not requests:
            return {}

        batch_id = await self._submit(requests)
        logger.info("📦 Anthropic Batch 送信完了: id=%s requests=%d", batch_id, len(requests))

        await self._wait_for_completion(batch_id)

        results = await self._fetch_results(batch_id)
        success = sum(1 for _t, _e in results.values() if _e is None)
        logger.info("📦 Anthropic Batch 取得完了: id=%s success=%d/%d",
                    batch_id, success, len(results))
        return results

    # ── internal ─────────────────────────────────────────────────────────────

    async def _submit(self, requests: list[dict]) -> str:
        """バッチリクエストを送信してバッチ ID を返す。"""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        batch_requests: list[dict[str, Any]] = []

        for req in requests:
            role_key = req.get("role_key", "shogun")
            model    = _ROLE_MODEL.get(role_key, "claude-sonnet-4-6")
            params: dict[str, Any] = {
                "model":      model,
                "max_tokens": req.get("max_tokens", 2000),
                "messages":   req["messages"],
            }
            if req.get("system"):
                params["system"] = req["system"]

            batch_requests.append({
                "custom_id": req["custom_id"],
                "params":    params,
            })

        batch = await client.messages.batches.create(requests=batch_requests)
        return batch.id

    async def _wait_for_completion(self, batch_id: str) -> None:
        """バッチ処理が "ended" になるまでポーリングする。"""
        import anthropic

        client  = anthropic.AsyncAnthropic(api_key=self._api_key)
        elapsed = 0.0

        while elapsed < self._max_wait:
            batch  = await client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            if status == "ended":
                return
            counts = getattr(batch, "request_counts", None)
            logger.debug(
                "📦 Batch ポーリング: id=%s status=%s elapsed=%.0fs counts=%s",
                batch_id, status, elapsed,
                f"proc={getattr(counts,'processing','-')} done={getattr(counts,'succeeded','-')}"
                if counts else "-",
            )
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

        raise TimeoutError(
            f"Anthropic Batch タイムアウト: id={batch_id} elapsed={elapsed:.0f}s"
        )

    async def _fetch_results(self, batch_id: str) -> dict[str, tuple[str, str | None]]:
        """`results()` ストリームを消費して {custom_id: (text, error|None)} を返す。"""
        import anthropic

        client  = anthropic.AsyncAnthropic(api_key=self._api_key)
        results: dict[str, tuple[str, str | None]] = {}

        async for item in client.messages.batches.results(batch_id):
            cid    = item.custom_id
            result = item.result

            if result.type == "succeeded":
                content = result.message.content
                text = ""
                if isinstance(content, list) and content:
                    first = content[0]
                    text  = first.text if hasattr(first, "text") else str(first)
                elif isinstance(content, str):
                    text = content
                results[cid] = (text, None)

            elif result.type == "errored":
                err = getattr(result, "error", {})
                err_type = getattr(err, "type", "?") if hasattr(err, "type") else err.get("type", "?")
                err_msg  = getattr(err, "message", "") if hasattr(err, "message") else err.get("message", "")
                results[cid] = ("", f"バッチエラー [{err_type}]: {err_msg}")
                logger.warning("📦 Batch 個別エラー: cid=%s type=%s msg=%s", cid, err_type, err_msg)

            else:
                results[cid] = ("", f"{result.type}: custom_id={cid}")
                logger.warning("📦 Batch 非成功: cid=%s type=%s", cid, result.type)

        return results
