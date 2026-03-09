#!/usr/bin/env python3
"""
APIキーの有効性をテストするスクリプト

Usage:
    python maintenance/check_api_keys.py
    python maintenance/check_api_keys.py --verbose
    python maintenance/check_api_keys.py --json
"""
import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import google.generativeai as genai
import requests
from groq import Groq

# プロジェクトルートの.envファイルを読み込み
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

def test_claude_api():
    """Claude APIキーをテスト"""
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        return False, "APIキーが設定されていません"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        # 最小限のリクエストでテスト
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        return True, "有効です"
    except anthropic.AuthenticationError:
        return False, "認証エラー: APIキーが無効です"
    except anthropic.PermissionDeniedError:
        return False, "権限エラー: このAPIキーには権限がありません"
    except anthropic.RateLimitError:
        return False, "レート制限エラー: 使用量制限に達しています"
    except Exception as e:
        return False, f"エラー: {str(e)}"

def test_gemini_api():
    """Gemini APIキーをテスト"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False, "APIキーが設定されていません"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content("Hi",
            generation_config=genai.GenerationConfig(max_output_tokens=10))
        return True, "有効です"
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            return False, "認証エラー: APIキーが無効です"
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return False, "クォータエラー: 使用量制限に達しています"
        return False, f"エラー: {error_msg}"

def test_openrouter_api():
    """OpenRouter APIキーをテスト"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return False, "APIキーが設定されていません"

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "qwen/qwen-2.5-coder-32b-instruct",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            return True, "有効です"
        elif response.status_code == 401:
            return False, "認証エラー: APIキーが無効です"
        elif response.status_code == 402:
            return False, "支払いエラー: クレジットが不足しています"
        elif response.status_code == 429:
            return False, "レート制限エラー: 使用量制限に達しています"
        else:
            return False, f"エラー: HTTP {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"エラー: {str(e)}"

def test_kimi_api():
    """Kimi (Moonshot) APIキーをテスト"""
    api_key = os.getenv("KIMI_API_KEY")
    provider = os.getenv("KIMI_PROVIDER", "moonshot")

    if not api_key:
        return False, "APIキーが設定されていません"

    try:
        if provider == "moonshot":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "moonshot-v1-8k",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                return True, "有効です"
            elif response.status_code == 401:
                return False, "認証エラー: APIキーが無効です"
            elif response.status_code == 429:
                return False, "レート制限エラー: 使用量制限に達しています"
            else:
                return False, f"エラー: HTTP {response.status_code} - {response.text}"
        else:
            return None, "OpenRouter経由の場合はOpenRouterのテスト結果を参照してください"
    except Exception as e:
        return False, f"エラー: {str(e)}"

def test_groq_api():
    """Groq APIキーをテスト"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False, "APIキーが設定されていません"

    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="llama-3.3-70b-versatile",
            max_tokens=10
        )
        return True, "有効です"
    except Exception as e:
        error_msg = str(e)
        if "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return False, "認証エラー: APIキーが無効です"
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            return False, "レート制限エラー: 使用量制限に達しています"
        return False, f"エラー: {error_msg}"

def test_tavily_api():
    """Tavily APIキーをテスト"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return False, "APIキーが設定されていません"

    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "api_key": api_key,
            "query": "test",
            "max_results": 1
        }
        response = requests.post(
            "https://api.tavily.com/search",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            return True, "有効です"
        elif response.status_code == 401 or response.status_code == 403:
            return False, "認証エラー: APIキーが無効です"
        elif response.status_code == 429:
            return False, "レート制限エラー: 使用量制限に達しています"
        else:
            return False, f"エラー: HTTP {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"エラー: {str(e)}"

def test_discord_token():
    """Discord Bot Tokenをテスト"""
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        return False, "トークンが設定されていません"

    try:
        headers = {"Authorization": f"Bot {token}"}
        response = requests.get(
            "https://discord.com/api/v10/users/@me",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            bot_info = response.json()
            return True, f"有効です (Bot: {bot_info.get('username', 'Unknown')})"
        elif response.status_code == 401:
            return False, "認証エラー: トークンが無効です"
        else:
            return False, f"エラー: HTTP {response.status_code}"
    except Exception as e:
        return False, f"エラー: {str(e)}"

def test_notion_api():
    """Notion APIキーをテスト"""
    api_key = os.getenv("NOTION_API_KEY")
    if not api_key:
        return False, "APIキーが設定されていません"

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        response = requests.get(
            "https://api.notion.com/v1/users/me",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            return True, "有効です"
        elif response.status_code == 401:
            return False, "認証エラー: APIキーが無効です"
        else:
            return False, f"エラー: HTTP {response.status_code}"
    except Exception as e:
        return False, f"エラー: {str(e)}"

def test_github_token():
    """GitHub Tokenをテスト"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return False, "トークンが設定されていません"

    try:
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        response = requests.get(
            "https://api.github.com/user",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            user_info = response.json()
            return True, f"有効です (User: {user_info.get('login', 'Unknown')})"
        elif response.status_code == 401:
            return False, "認証エラー: トークンが無効です"
        else:
            return False, f"エラー: HTTP {response.status_code}"
    except Exception as e:
        return False, f"エラー: {str(e)}"

def main():
    """すべてのAPIキーをテスト"""
    parser = argparse.ArgumentParser(description="APIキー有効性テスト")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細な出力")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    args = parser.parse_args()

    tests = [
        ("Claude API (Shogun)", test_claude_api),
        ("Gemini API (最終防衛線)", test_gemini_api),
        ("OpenRouter API (Qwen)", test_openrouter_api),
        ("Kimi API (Moonshot)", test_kimi_api),
        ("Groq API", test_groq_api),
        ("Tavily API (Web検索)", test_tavily_api),
        ("Discord Bot Token", test_discord_token),
        ("Notion API", test_notion_api),
        ("GitHub Token", test_github_token),
    ]

    results = []

    if not args.json:
        print("=" * 80)
        print("APIキー有効性テスト")
        print("=" * 80)
        print()

    for name, test_func in tests:
        if not args.json:
            print(f"テスト中: {name}...", end=" ", flush=True)

        try:
            is_valid, message = test_func()
            results.append({
                "name": name,
                "valid": is_valid,
                "message": message
            })

            if not args.json:
                if is_valid is None:
                    print(f"⚠️  {message}")
                elif is_valid:
                    print(f"✓ {message}")
                else:
                    print(f"✗ {message}")
                print()
        except Exception as e:
            error_msg = f"予期しないエラー: {str(e)}"
            results.append({
                "name": name,
                "valid": False,
                "message": error_msg
            })
            if not args.json:
                print(f"✗ {error_msg}")
                print()

    if args.json:
        # JSON出力
        output = {
            "results": results,
            "summary": {
                "valid": sum(1 for r in results if r["valid"] is True),
                "invalid": sum(1 for r in results if r["valid"] is False),
                "skipped": sum(1 for r in results if r["valid"] is None),
                "total": len(results)
            }
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # 通常出力
        print("=" * 80)
        print("サマリー")
        print("=" * 80)
        valid_count = sum(1 for r in results if r["valid"] is True)
        invalid_count = sum(1 for r in results if r["valid"] is False)
        skipped_count = sum(1 for r in results if r["valid"] is None)

        print(f"有効: {valid_count} / 無効: {invalid_count} / スキップ: {skipped_count}")
        print()

        if invalid_count > 0:
            print("無効なAPIキー:")
            for result in results:
                if result["valid"] is False:
                    print(f"  • {result['name']}: {result['message']}")
            print()

        # 終了コード設定
        sys.exit(0 if invalid_count == 0 else 1)

if __name__ == "__main__":
    main()
