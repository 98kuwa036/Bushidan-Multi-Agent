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
            self.check_local_qwen3(),
            self.check_gemini_flash(),
            self.check_groq(),
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
            # パス確認
            cli_path = "/home/claude/.npm-global/bin/claude"
            if not os.path.exists(cli_path):
                return LLMStatus(name, False, f"Claude CLI not found at {cli_path}")

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

    async def check_local_qwen3(self) -> LLMStatus:
        """ローカル Qwen3 (llama.cpp) が使用可能か"""
        import time

        start = time.time()
        name = "Local Qwen3 (llama.cpp)"

        try:
            # localhost:8000 接続確認
            import aiohttp

            url = "http://localhost:8000/v1/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "prompt": "test",
                "max_tokens": 1,
                "temperature": 0.7,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    elapsed = (time.time() - start) * 1000

                    if response.status == 200:
                        return LLMStatus(
                            name,
                            True,
                            "llama.cpp server responding",
                            response_time_ms=elapsed,
                        )
                    else:
                        return LLMStatus(
                            name,
                            False,
                            f"HTTP {response.status}",
                            response_time_ms=elapsed,
                        )

        except asyncio.TimeoutError:
            return LLMStatus(name, False, "Connection timeout (10s)")
        except aiohttp.ClientConnectorError:
            return LLMStatus(name, False, "Cannot connect to localhost:8000")
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return LLMStatus(
                name,
                False,
                str(e)[:100],
                response_time_ms=elapsed,
            )

    async def check_gemini_flash(self) -> LLMStatus:
        """Gemini 3.0 Flash が使用可能か"""
        import time

        start = time.time()
        name = "Gemini 3.0 Flash"

        try:
            # Google API キー確認
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return LLMStatus(name, False, "GOOGLE_API_KEY not set")

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
