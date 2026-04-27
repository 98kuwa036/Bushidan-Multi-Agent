"""
core/router/batch/mode.py — 処理モード定義

INTERACTIVE: リアルタイム。ユーザーが待機中。低レイテンシ優先。
BATCH:       非同期。ユーザー待機なし。コスト効率・完全性優先。
             用途: スキル進化分析・Notionインデックス再構築・監査ログ生成
"""
from enum import Enum


class ProcessingMode(str, Enum):
    INTERACTIVE = "interactive"
    BATCH       = "batch"


# バッチモードで変わる動作の設定
BATCH_CONFIG = {
    "streaming_enabled":         False,   # ストリーミング無効
    "cache_enabled":             False,   # 重複チェック不要（毎回完全実行が目的）
    "semantic_router_shortcut":  False,   # LLM 完全分析を使う
    "autonomous_loop":           False,   # 1回完結
    "hitl_enabled":              False,   # ユーザー待機なし → human_interrupt をスキップ
    "notion_store_sync":         True,    # fire-and-forget ではなく await 同期保存
    "sandbox_verify_enabled":    False,   # コード実行不要
    "node_timeout_multiplier":   3.0,     # タイムアウト緩和（ユーザー待機なし）
    # ── Anthropic Batch API ─────────────────────────────────────────────────
    "use_anthropic_batch":       True,    # shogun/daigensui ステップを一括送信 (コスト 50% 削減)
    "anthropic_batch_poll_interval": 5.0, # ポーリング間隔 (秒); 本番は 60 以上推奨
    "anthropic_batch_max_wait":  3600.0,  # 最大待機 (秒)
    # ── 並列実行 ────────────────────────────────────────────────────────────
    "max_parallel_batch_steps":  8,       # 並列上限 (0 = 無制限は LLM バックエンドを飽和させるリスクあり)
    # ── 完了通知 ────────────────────────────────────────────────────────────
    "notify_on_completion":      True,    # バッチ完了時に通知 (BATCH_NOTIFY_WEBHOOK)
}
