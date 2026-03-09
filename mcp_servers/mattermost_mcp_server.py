"""武士団 Mattermost MCP サーバー

Mattermost API を MCP ツールとして公開します。
Bushidan AI エージェントが Mattermost チャンネルへの投稿・タスク報告・
進捗通知などを MCP プロトコル経由で行えるようにします。

参考: https://github.com/jagan-shanmugam/mattermost-mcp-host

ツール一覧:
  【Mattermost 操作】
  - post_message            : チャンネルに投稿
  - post_direct_message     : ユーザーに DM 送信
  - get_channel_messages    : チャンネルの最新メッセージ取得
  - search_messages         : メッセージ全文検索
  - add_reaction            : 絵文字リアクション追加
  - create_channel          : チャンネル作成
  - get_team_channels       : チームのチャンネル一覧取得

  【武士団システム連携】
  - submit_task             : 武士団 9 層システムにタスク投入
  - get_bushidan_status     : 全 9 層エージェントの状態確認
  - report_agent_progress   : エージェントが進捗をチャンネルに報告

必要な環境変数:
  MATTERMOST_URL    - サーバー URL (例: chat.example.com)
  MATTERMOST_TOKEN  - Bot アクセストークン
  MATTERMOST_PORT   - ポート番号 (デフォルト: 443)
  MATTERMOST_SCHEME - http/https (デフォルト: https)

起動方法 (stdio モード):
  python -m mcp.mattermost_mcp_server
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bushidan.mattermost_mcp")

# ── MCP SDK ─────────────────────────────────────────────────────────────────
try:
    from mcp.server.fastmcp import FastMCP
    _mcp = FastMCP("bushidan-mattermost")
    HAS_MCP = True
except ImportError:
    # パッケージ未インストール時のスタブ (ツール定義は通過させる)
    class _StubMCP:
        def tool(self, **kwargs):
            def decorator(fn):
                return fn
            return decorator
        def run(self):
            print("エラー: mcp パッケージが必要です\nインストール: pip install mcp", file=sys.stderr)
            sys.exit(1)
    _mcp = _StubMCP()
    HAS_MCP = False

# ── Mattermost ドライバー ────────────────────────────────────────────────────
try:
    from mattermostdriver import AsyncDriver
    HAS_MATTERMOST = True
except ImportError:
    HAS_MATTERMOST = False

_driver: Optional[AsyncDriver] = None
_driver_lock = asyncio.Lock()


async def _get_driver() -> AsyncDriver:
    """Mattermost ドライバーのシングルトンを返す."""
    global _driver
    async with _driver_lock:
        if _driver is None:
            url = os.environ.get("MATTERMOST_URL", "")
            token = os.environ.get("MATTERMOST_TOKEN", "")
            port = int(os.environ.get("MATTERMOST_PORT", "443"))
            scheme = os.environ.get("MATTERMOST_SCHEME", "https")

            if not url or not token:
                raise RuntimeError(
                    "MATTERMOST_URL と MATTERMOST_TOKEN を環境変数に設定してください"
                )

            _driver = AsyncDriver({
                "url": url,
                "token": token,
                "port": port,
                "scheme": scheme,
                "debug": False,
            })
            await _driver.login()
            logger.info("Mattermost ドライバー初期化完了: %s", url)
    return _driver


# ═══════════════════════════════════════════════════════════════════════════
# Mattermost 操作ツール
# ═══════════════════════════════════════════════════════════════════════════

@_mcp.tool()
async def post_message(channel_id: str, message: str, root_id: str = "") -> str:
    """Mattermost チャンネルにメッセージを投稿する。

    Args:
        channel_id: 投稿先チャンネル ID
        message: 投稿するメッセージ (Markdown 使用可)
        root_id: スレッド返信先の投稿 ID (省略時はスレッド外に投稿)

    Returns:
        投稿された post の ID
    """
    driver = await _get_driver()
    options: dict = {"channel_id": channel_id, "message": message}
    if root_id:
        options["root_id"] = root_id
    post = await driver.posts.create_post(options=options)
    post_id = post.get("id", "")
    logger.info("投稿完了: channel=%s, post_id=%s", channel_id, post_id)
    return f"投稿成功: post_id={post_id}"


@_mcp.tool()
async def post_direct_message(user_id: str, message: str) -> str:
    """指定ユーザーにダイレクトメッセージを送信する。

    Args:
        user_id: 送信先ユーザー ID
        message: 送信するメッセージ

    Returns:
        作成された DM チャンネル ID と投稿 ID
    """
    driver = await _get_driver()
    me = await driver.users.get_user("me")
    dm_channel = await driver.channels.create_direct_message_channel(
        [me["id"], user_id]
    )
    channel_id = dm_channel["id"]
    post = await driver.posts.create_post(
        options={"channel_id": channel_id, "message": message}
    )
    return f"DM 送信成功: channel_id={channel_id}, post_id={post.get('id', '')}"


@_mcp.tool()
async def get_channel_messages(channel_id: str, limit: int = 20) -> str:
    """チャンネルの最新メッセージを取得する。

    Args:
        channel_id: チャンネル ID
        limit: 取得するメッセージ数 (デフォルト 20、最大 200)

    Returns:
        メッセージ一覧 (JSON)
    """
    driver = await _get_driver()
    limit = max(1, min(limit, 200))
    data = await driver.posts.get_posts_for_channel(
        channel_id, params={"per_page": limit}
    )
    order = data.get("order", [])
    posts = data.get("posts", {})

    messages = [
        {
            "id": posts[pid].get("id", ""),
            "user_id": posts[pid].get("user_id", ""),
            "message": posts[pid].get("message", ""),
            "create_at": posts[pid].get("create_at", 0),
        }
        for pid in order[:limit]
        if pid in posts
    ]
    return json.dumps(messages, ensure_ascii=False, indent=2)


@_mcp.tool()
async def search_messages(query: str, team_id: str = "") -> str:
    """Mattermost 内のメッセージを全文検索する。

    Args:
        query: 検索クエリ (Mattermost 検索構文使用可、例: "from:@user in:channel keyword")
        team_id: 検索対象チーム ID (省略時は最初のチームを使用)

    Returns:
        検索結果 (JSON)
    """
    driver = await _get_driver()
    if not team_id:
        teams = await driver.teams.get_teams()
        team_id = teams[0]["id"] if teams else ""
    if not team_id:
        return json.dumps({"error": "チームが見つかりませんでした"}, ensure_ascii=False)

    results = await driver.posts.search_for_team_posts(
        team_id, options={"terms": query, "is_or_search": False}
    )
    order = results.get("order", [])
    posts = results.get("posts", {})

    matches = [
        {
            "id": posts[pid].get("id", ""),
            "channel_id": posts[pid].get("channel_id", ""),
            "user_id": posts[pid].get("user_id", ""),
            "message": posts[pid].get("message", ""),
            "create_at": posts[pid].get("create_at", 0),
        }
        for pid in order
        if pid in posts
    ]
    return json.dumps(
        {"query": query, "count": len(matches), "posts": matches},
        ensure_ascii=False,
        indent=2,
    )


@_mcp.tool()
async def add_reaction(post_id: str, emoji_name: str) -> str:
    """投稿に絵文字リアクションを追加する。

    Args:
        post_id: リアクション対象の投稿 ID
        emoji_name: 絵文字名 (例: "white_check_mark", "tada", "fire", "eyes")

    Returns:
        成功/失敗メッセージ
    """
    driver = await _get_driver()
    me = await driver.users.get_user("me")
    await driver.reactions.create_reaction({
        "user_id": me["id"],
        "post_id": post_id,
        "emoji_name": emoji_name,
    })
    return f"リアクション追加完了: post_id={post_id}, emoji=:{emoji_name}:"


@_mcp.tool()
async def create_channel(
    team_id: str,
    name: str,
    display_name: str,
    purpose: str = "",
    channel_type: str = "O",
) -> str:
    """Mattermost チャンネルを作成する。

    Args:
        team_id: チーム ID
        name: チャンネル名スラッグ (英数字・ハイフンのみ、例: "bushidan-reports")
        display_name: 表示名 (例: "武士団レポート")
        purpose: チャンネルの目的説明 (省略可)
        channel_type: "O"=公開チャンネル / "P"=プライベートチャンネル

    Returns:
        作成されたチャンネルの ID
    """
    driver = await _get_driver()
    options: dict = {
        "team_id": team_id,
        "name": name,
        "display_name": display_name,
        "type": channel_type,
    }
    if purpose:
        options["purpose"] = purpose
    channel = await driver.channels.create_channel(options=options)
    channel_id = channel.get("id", "")
    logger.info("チャンネル作成: name=%s, id=%s", name, channel_id)
    return f"チャンネル作成成功: id={channel_id}, display_name={display_name}"


@_mcp.tool()
async def get_team_channels(team_id: str) -> str:
    """チームの公開チャンネル一覧を取得する。

    Args:
        team_id: チーム ID

    Returns:
        チャンネル一覧 (JSON)
    """
    driver = await _get_driver()
    channels = await driver.channels.get_public_channels(team_id)
    result = [
        {
            "id": ch.get("id", ""),
            "name": ch.get("name", ""),
            "display_name": ch.get("display_name", ""),
            "purpose": ch.get("purpose", ""),
            "total_msg_count": ch.get("total_msg_count", 0),
        }
        for ch in channels
    ]
    return json.dumps(result, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# 武士団システム連携ツール
# ═══════════════════════════════════════════════════════════════════════════

@_mcp.tool()
async def submit_task(
    task: str,
    mode: str = "battalion",
    notify_channel_id: str = "",
) -> str:
    """武士団 9 層 AI システムにタスクを投入する。

    MCP クライアント (Claude Code など) から Bushidan の 9 層アーキテクチャに
    タスクを委譲し、インテリジェントルーターによる最適な処理を受けます。

    ルーティング:
        SIMPLE    → 家老-B Groq Llama 3.3 70B (即応・無料)
        MEDIUM    → 参謀-B Grok-code-fast-1 (並列実装)
        COMPLEX   → 軍師 o3-mini → 参謀A/B → 家老
        STRATEGIC → 将軍 Sonnet 4.6 → 大元帥 Opus 4.5 (最高品質)

    Args:
        task: 実行するタスクの説明 (自然言語)
        mode: 処理モード - "battalion"(全9層) / "company"(軽量) / "platoon"(最軽量)
        notify_channel_id: 完了通知を送る Mattermost チャンネル ID (省略時は通知なし)

    Returns:
        タスクの処理結果テキスト
    """
    if mode not in ("battalion", "company", "platoon"):
        return f"エラー: 無効なモード '{mode}'。battalion / company / platoon のいずれかを指定してください。"

    try:
        from utils.config import load_config
        from core.system_orchestrator import SystemOrchestrator

        os.environ["SYSTEM_MODE"] = mode
        config = load_config()
        orchestrator = SystemOrchestrator(config)
        await orchestrator.initialize()

        context = {"source": "mattermost_mcp", "mode": mode}
        result = await orchestrator.process_task(task, context)

        if result.get("status") == "failed":
            error = result.get("error", "不明なエラー")
            response_text = f"❌ タスク失敗: {error}"
        else:
            content = result.get("result", "")
            elapsed = result.get("elapsed_time", 0)
            response_text = f"{content}\n\n⏱️ 処理時間: {elapsed:.1f}秒 | モード: {mode}"

        # 完了通知 (チャンネル ID が指定されている場合)
        if notify_channel_id:
            try:
                driver = await _get_driver()
                await driver.posts.create_post(options={
                    "channel_id": notify_channel_id,
                    "message": (
                        f"🏯 **武士団タスク完了**\n\n"
                        f"**タスク:** {task[:150]}\n\n"
                        f"{response_text}"
                    ),
                })
            except Exception as e:
                logger.warning("完了通知の送信に失敗: %s", e)

        return response_text

    except Exception as e:
        logger.exception("submit_task エラー")
        return f"❌ エラー: {e}"


@_mcp.tool()
async def get_bushidan_status() -> str:
    """武士団 9 層エージェントシステムの現在の状態を取得する。

    Returns:
        バージョン情報・各エージェント層のオンライン状態・インフラ情報 (JSON)
    """
    try:
        from utils.config import load_config
        from core.system_orchestrator import SystemOrchestrator

        config = load_config()
        orchestrator = SystemOrchestrator(config)
        await orchestrator.initialize()

        health = getattr(orchestrator, "health_status", {})
        version = getattr(orchestrator, "VERSION", "unknown")

        kengyo = getattr(orchestrator, "kengyo", None)
        kengyo_ok = bool(
            kengyo
            and (not hasattr(kengyo, "is_available") or kengyo.is_available())
        )

        agents = {
            "layer1_daigensui": {
                "name": "大元帥 (Claude Opus 4.5)",
                "online": bool(getattr(orchestrator, "daigensui", None)),
            },
            "layer2_shogun": {
                "name": "将軍 (Claude Sonnet 4.6)",
                "online": bool(getattr(orchestrator, "shogun", None)),
            },
            "layer3_gunshi": {
                "name": "軍師 (o3-mini high)",
                "online": bool(getattr(orchestrator, "gunshi", None)),
            },
            "layer4_sanbo": {
                "name": "参謀 (GPT-5 / Grok-code-fast-1)",
                "online": bool(getattr(orchestrator, "sanbo", None)),
            },
            "layer5_karo": {
                "name": "家老 (Gemini Flash / Llama 3.3 70B)",
                "online": bool(getattr(orchestrator, "karo", None)),
            },
            "layer6_kengyo": {
                "name": "検校 (Gemini Flash Vision)",
                "online": kengyo_ok,
            },
            "layer7_onmitsu": {
                "name": "隠密 (Nemotron-3-Nano Local)",
                "online": health.get("llamacpp", False),
            },
        }

        return json.dumps(
            {
                "version": version,
                "agents": agents,
                "infrastructure": {
                    "llamacpp_endpoint": os.environ.get(
                        "LLAMACPP_ENDPOINT", "http://192.168.11.239:8080"
                    ),
                    "llamacpp_online": health.get("llamacpp", False),
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        logger.exception("get_bushidan_status エラー")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@_mcp.tool()
async def report_agent_progress(
    agent_role: str,
    task_id: str,
    channel_id: str,
    message: str,
    progress_percent: int = -1,
    root_id: str = "",
) -> str:
    """エージェントが処理中タスクの進捗を Mattermost チャンネルに報告する。

    各エージェント層が長時間タスクの途中経過をリアルタイムで報告するために使用。
    家老層が subtask 完了を報告したり、軍師が PDCA サイクルの状態を通知する際に
    このツールを呼び出します。

    Args:
        agent_role: エージェントの役職 (例: "将軍", "軍師", "参謀-A", "家老-A")
        task_id: タスクの識別子 (追跡・スレッド管理に使用)
        channel_id: 報告先 Mattermost チャンネル ID
        message: 報告メッセージ (何を完了したか・次のステップなど)
        progress_percent: 進捗率 0-100 (-1 = 不明/該当なし)
        root_id: スレッド返信先の投稿 ID (省略時は新規スレッド)

    Returns:
        投稿された post_id
    """
    emoji_map = {
        "大元帥": "👑", "将軍": "🎌", "軍師": "🧠",
        "参謀-A": "⚔️", "参謀-B": "⚡", "参謀": "⚔️",
        "家老-A": "👔", "家老-B": "🦙", "家老": "👔",
        "検校": "👁️", "隠密": "🥷",
    }
    emoji = emoji_map.get(agent_role, "🤖")

    progress_str = ""
    if 0 <= progress_percent <= 100:
        bar_len = 20
        filled = int(bar_len * progress_percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        progress_str = f"\n`[{bar}] {progress_percent}%`"

    formatted = (
        f"{emoji} **[{agent_role}]** `{task_id}`"
        f"{progress_str}\n{message}"
    )

    driver = await _get_driver()
    options: dict = {"channel_id": channel_id, "message": formatted}
    if root_id:
        options["root_id"] = root_id
    post = await driver.posts.create_post(options=options)
    return f"進捗報告完了: post_id={post.get('id', '')}"


# ═══════════════════════════════════════════════════════════════════════════
# エントリーポイント
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """MCP サーバーを stdio モードで起動."""
    if not HAS_MCP:
        print(
            "エラー: mcp パッケージが必要です\nインストール: pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)
    if not HAS_MATTERMOST:
        print(
            "エラー: mattermostdriver が必要です\nインストール: pip install mattermostdriver",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("🏯 武士団 Mattermost MCP サーバー起動 (stdio)")
    _mcp.run()


if __name__ == "__main__":
    main()
