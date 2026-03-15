#!/usr/bin/env python3
"""
Mattermostエージェントプロフィール一括更新スクリプト

各エージェントアカウントの表示名を [漢字名 | LLM名] 形式に更新します。
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("update_mattermost_profiles")

try:
    from mattermostdriver.driver import Driver
except ImportError:
    logger.error("❌ mattermostdriver がインストールされていません")
    sys.exit(1)


# ────────────────────────────────────────────────────────────────
# エージェント設定 (display_name は [漢字名 | LLM名])
# ────────────────────────────────────────────────────────────────
AGENT_PROFILES = {
    "uketuke-bot": {
        "token": os.environ.get("MM_TOKEN_UKETUKE", ""),
        "display_name": "受付 | Command R7B-12-2024",
        "position": "Reception / Fallback",
    },
    "gaiji-bot": {
        "token": os.environ.get("MM_TOKEN_GAIJI", ""),
        "display_name": "外事 | Command A 03-2025",
        "position": "External Affairs / RAG",
    },
    "kengyo-bot": {
        "token": os.environ.get("MM_TOKEN_KENGYO", ""),
        "display_name": "検校 | Gemini Vision",
        "position": "Inspector / Vision",
    },
    "shogun-bot": {
        "token": os.environ.get("MM_TOKEN_SHOGUN", ""),
        "display_name": "将軍 | Claude Sonnet 4.6",
        "position": "Shogun / Main Worker",
    },
    "gunshi-bot": {
        "token": os.environ.get("MM_TOKEN_GUNSHI", ""),
        "display_name": "軍師 | Mistral Large 3",
        "position": "Military Strategist / PDCA",
    },
    "sanbo-a-bot": {  # 旧アカウント名
        "token": os.environ.get("MM_TOKEN_SANBO", ""),
        "display_name": "参謀 | Mistral Large 3",
        "position": "Staff Officer / Tools",
    },
    "yuhitsu-bot": {
        "token": os.environ.get("MM_TOKEN_YUHITSU", ""),
        "display_name": "右筆 | ELYZA Local",
        "position": "Secretary / Japanese Cleanup",
    },
    "seppou-bot": {
        "token": os.environ.get("MM_TOKEN_SEPPOU", ""),
        "display_name": "斥候 | Llama 3.3 Groq",
        "position": "Scout / Fast Q&A",
    },
    "onmitsu-bot": {
        "token": os.environ.get("MM_TOKEN_ONMITSU", ""),
        "display_name": "隠密 | Nemotron Local",
        "position": "Secret Agent / Local",
    },
    "daigensui-bot": {
        "token": os.environ.get("MM_TOKEN_DAIGENSUI", ""),
        "display_name": "大元帥 | Claude Opus 4.6",
        "position": "Commander-in-Chief",
    },
}


async def update_profile_as_admin(username: str, user_id: str, profile_cfg: dict, admin_driver: Driver) -> bool:
    """管理者トークンでプロフィール更新"""
    display_name = profile_cfg["display_name"]
    position = profile_cfg["position"]

    try:
        # プロフィール更新
        update_data = {
            "first_name": display_name.split(" | ")[0],  # 漢字名
            "nickname": display_name,  # 表示名全体
            "position": position,
        }

        await asyncio.to_thread(admin_driver.users.patch_user, user_id, update_data)
        logger.info(f"✅ {username}: {display_name}")
        return True

    except Exception as e:
        logger.error(f"❌ {username}: {e}")
        return False


async def get_user_id_by_username(username: str, admin_driver: Driver) -> str:
    """ユーザー名からユーザーIDを取得"""
    try:
        user = await asyncio.to_thread(admin_driver.users.get_user_by_username, username)
        return user["id"]
    except Exception as e:
        logger.error(f"❌ {username} ID取得失敗: {e}")
        return ""


async def main():
    logger.info("🏯 Mattermostエージェントプロフィール一括更新開始...")

    url = os.environ.get("MATTERMOST_URL", "")
    port = int(os.environ.get("MATTERMOST_PORT", "8065"))
    scheme = os.environ.get("MATTERMOST_SCHEME", "http")
    admin_pat = os.environ.get("MATTERMOST_ADMIN_PAT", "")

    if not admin_pat:
        logger.error("❌ MATTERMOST_ADMIN_PAT 環境変数が設定されていません")
        return False

    # 管理者ドライバー作成
    admin_driver = Driver({
        "url": url,
        "token": admin_pat,
        "port": port,
        "scheme": scheme,
        "verify": False,
    })

    try:
        await asyncio.to_thread(admin_driver.login)
        logger.info("✅ 管理者トークンでログイン完了")
    except Exception as e:
        logger.error(f"❌ 管理者ログイン失敗: {e}")
        return False

    tasks = []
    for username, cfg in AGENT_PROFILES.items():
        user_id = await get_user_id_by_username(username, admin_driver)
        if user_id:
            tasks.append(update_profile_as_admin(username, user_id, cfg, admin_driver))
        else:
            tasks.append(asyncio.sleep(0))  # ダミータスク

    results = await asyncio.gather(*tasks)
    success = sum(1 for r in results if r)
    total = len(AGENT_PROFILES)

    logger.info(f"✅ 完了: {success}/{total} 件")
    return success == total


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
