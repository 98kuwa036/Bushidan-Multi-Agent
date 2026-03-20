"""
Bushidan Multi-Agent System v10.1 - LLM Availability Checker
各LLM/APIの可用性を確認するモジュール
"""

import asyncio
import os
import subprocess
from typing import Dict, Optional
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMStatus:
    """LLM可用性ステータス"""
    name: str
    available: bool
    reason: str = ""
    response_time_ms: Optional[float] = None


class LLMAvailabilityChecker:
    """各LLM/APIの可用性確認"""

    def __init__(self):
        self.statuses: Dict[str, LLMStatus] = {}

    async def check_all(self) -> Dict[str, LLMStatus]:
        """すべてのLLM可用性を確認（並列実行）"""
        logger.info("🔍 LLM可用性確認開始...")

        checks = [
            self.check_claude_pro_cli(),
            self.check_anthropic_api(),
            self.check_gemini_flash(),
            self.check_groq(),
            self.check_mistral(),
            self.check_cohere(),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        for status in results:
            if isinstance(status, Exception):
                logger.warning(f"❌ 可用性確認エラー: {status}")
            else:
                self.statuses[status.name] = status
                self._log_status(status)

        return self.statuses

    async def check_claude_pro_cli(self) -> LLMStatus:
        """Claude Pro CLI が使用可能か"""
        import time

        start = time.time()
        name = "Claude Pro CLI"

        try:
            # パス確認（複数の候補をチェック）
            possible_paths = [
                "/home/claude/.local/bin/claude",
                "/home/claude/.npm-global/bin/claude",
                os.path.expanduser("~/.local/bin/claude"),
            ]
            cli_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    cli_path = path
                    break

            if not cli_path:
                return LLMStatus(name, False, f"Claude CLI not found (tried: {', '.join(possible_paths)})")

            # バージョン確認コマンド
            result = subprocess.run(
                [cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            elapsed = (time.time() - start) * 1000

            if result.returncode == 0:
                version = result.stdout.strip()
                return LLMStatus(
                    name, True, f"Version: {version}", response_time_ms=elapsed
                )
            else:
                return LLMStatus(
                    name,
                    False,
                    f"claude --version failed: {result.stderr[:100]}",
                    response_time_ms=elapsed,
                )

        except subprocess.TimeoutExpired:
            return LLMStatus(name, False, "Command timeout (5s)")
        except Exception as e:
            return LLMStatus(name, False, str(e))

    async def check_anthropic_api(self) -> LLMStatus:
        """Anthropic API が使用可能か"""
        import time

        start = time.time()
        name = "Anthropic API"

        try:
            # 認証キー確認
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return LLMStatus(name, False, "ANTHROPIC_API_KEY not set")

            # 軽量リクエスト送信
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            # max_tokens=1 で最小限のリクエスト
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}],
                timeout=10,
            )

            elapsed = (time.time() - start) * 1000

            return LLMStatus(
                name,
                True,
                f"Model: {response.model}",
                response_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return LLMStatus(
                name,
                False,
                f"{str(e)[:100]}",
                response_time_ms=elapsed,
            )

    async def check_gemini_flash(self) -> LLMStatus:
        """Gemini 3.0 Flash が使用可能か"""
        import time

        start = time.time()
        name = "Gemini 3.0 Flash"

        try:
            # Google API キー確認（GOOGLE_API_KEY または GEMINI_API_KEY）
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return LLMStatus(name, False, "GOOGLE_API_KEY or GEMINI_API_KEY not set")

            # 軽量リクエスト送信
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3-flash-preview")

            response = model.generate_content(
                "test",
                generation_config={"max_output_tokens": 1},
                request_options={"timeout": 10},
            )

            elapsed = (time.time() - start) * 1000

            return LLMStatus(
                name,
                True,
                f"Model: gemini-3-flash-preview",
                response_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return LLMStatus(
                name,
                False,
                str(e)[:100],
                response_time_ms=elapsed,
            )

    async def check_groq(self) -> LLMStatus:
        """Groq が使用可能か"""
        import time

        start = time.time()
        name = "Groq"

        try:
            # Groq API キー確認
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                return LLMStatus(name, False, "GROQ_API_KEY not set")

            # 軽量リクエスト送信
            from groq import Groq

            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )

            elapsed = (time.time() - start) * 1000

            return LLMStatus(
                name,
                True,
                f"Model: {response.model}",
                response_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return LLMStatus(
                name,
                False,
                str(e)[:100],
                response_time_ms=elapsed,
            )

    async def check_mistral(self) -> LLMStatus:
        """Mistral AI が使用可能か"""
        import time

        start = time.time()
        name = "Mistral AI"

        try:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                return LLMStatus(name, False, "MISTRAL_API_KEY not set")

            # mistralai ライブラリでAPIテスト
            from mistralai import Mistral

            client = Mistral(api_key=api_key)
            response = client.chat(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )

            elapsed = (time.time() - start) * 1000

            return LLMStatus(
                name,
                True,
                f"Model: mistral-large-latest",
                response_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return LLMStatus(
                name,
                False,
                str(e)[:100],
                response_time_ms=elapsed,
            )

    async def check_cohere(self) -> LLMStatus:
        """Cohere が使用可能か"""
        import time

        start = time.time()
        name = "Cohere"

        try:
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                return LLMStatus(name, False, "COHERE_API_KEY not set")

            # Cohere API テスト（v2 API）
            import httpx

            url = "https://api.cohere.com/v2/chat"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "command-r",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                elapsed = (time.time() - start) * 1000

                if response.status_code == 200:
                    return LLMStatus(
                        name,
                        True,
                        "command-r responding",
                        response_time_ms=elapsed,
                    )
                else:
                    return LLMStatus(
                        name,
                        False,
                        f"HTTP {response.status_code}",
                        response_time_ms=elapsed,
                    )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return LLMStatus(
                name,
                False,
                str(e)[:100],
                response_time_ms=elapsed,
            )

    def _log_status(self, status: LLMStatus) -> None:
        """ステータスをログに出力"""
        if status.available:
            time_str = f" ({status.response_time_ms:.0f}ms)" if status.response_time_ms else ""
            logger.info(f"✅ {status.name} available{time_str}: {status.reason}")
        else:
            logger.warning(f"⚠️  {status.name} unavailable: {status.reason}")

    def get_available_llms(self) -> Dict[str, bool]:
        """利用可能なLLMを一覧返却"""
        return {name: status.available for name, status in self.statuses.items()}

    def print_summary(self) -> str:
        """可用性サマリーを出力"""
        available = [name for name, status in self.statuses.items() if status.available]
        unavailable = [name for name, status in self.statuses.items() if not status.available]

        summary = "🔍 LLM可用性サマリー\n"
        summary += f"✅ 利用可能 ({len(available)}): {', '.join(available) if available else 'None'}\n"
        summary += (
            f"⚠️  利用不可 ({len(unavailable)}): {', '.join(unavailable) if unavailable else 'None'}\n"
        )

        return summary
