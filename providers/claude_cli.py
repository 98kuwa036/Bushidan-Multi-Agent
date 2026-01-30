"""Claude CLI Provider - Pro版 claude-cli (npm) ラッパー

Primary execution path for Shogun (Opus) and Karo (Sonnet).
Pro版の制限がかかった場合、Anthropic APIにフォールバックする。

Usage:
    provider = ClaudeCLIProvider()
    result = await provider.generate("prompt", model="opus")
"""

import asyncio
import json
import logging
import os
import re
import shutil
from typing import AsyncIterator

logger = logging.getLogger("shogun.provider.claude_cli")

# Rate-limit detection patterns
RATE_LIMIT_PATTERNS = [
    "rate limit",
    "too many requests",
    "usage limit",
    "capacity",
    "throttl",
    "429",
    "overloaded",
]


class ClaudeCLIProvider:
    """Claude Code CLI (npm @anthropic-ai/claude-code) wrapper.

    Uses the `claude` command with `--print` for non-interactive mode.
    Detects Pro rate limiting and signals for API fallback.
    """

    def __init__(self, cli_path: str | None = None):
        self.cli_path = cli_path or shutil.which("claude") or "claude"
        self._available: bool | None = None

    async def check_available(self) -> bool:
        """Check if claude CLI is installed and accessible."""
        if self._available is not None:
            return self._available
        try:
            proc = await asyncio.create_subprocess_exec(
                self.cli_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            self._available = proc.returncode == 0
            if self._available:
                version = stdout.decode().strip()
                logger.info("Claude CLI available: %s", version)
            return self._available
        except (FileNotFoundError, asyncio.TimeoutError):
            self._available = False
            logger.warning("Claude CLI not found at: %s", self.cli_path)
            return False

    async def generate(
        self,
        prompt: str,
        model: str = "sonnet",
        system_prompt: str = "",
        max_turns: int = 1,
        cwd: str | None = None,
    ) -> "CLIResult":
        """Execute a prompt via claude --print.

        Args:
            prompt: The user prompt.
            model: Model name (opus, sonnet, haiku).
            system_prompt: System prompt to prepend.
            max_turns: Max agentic turns.
            cwd: Working directory for the CLI.

        Returns:
            CLIResult with text, success flag, and rate_limited flag.
        """
        if not await self.check_available():
            return CLIResult(
                text="",
                success=False,
                rate_limited=False,
                error="Claude CLI not available",
            )

        # Build command
        cmd = [self.cli_path, "--print"]

        # Model selection
        if model:
            cmd.extend(["--model", model])

        # Max turns
        if max_turns > 1:
            cmd.extend(["--max-turns", str(max_turns)])

        # System prompt via --system-prompt
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        # The prompt itself is passed via stdin
        full_prompt = prompt

        logger.info(
            "[CLI] model=%s, prompt_len=%d, max_turns=%d",
            model, len(prompt), max_turns,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or os.getcwd(),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=full_prompt.encode("utf-8")),
                timeout=600,  # 10 min max
            )

            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()

            # Check for rate limiting
            combined = (stdout_text + stderr_text).lower()
            rate_limited = any(p in combined for p in RATE_LIMIT_PATTERNS)

            if rate_limited:
                logger.warning("[CLI] Pro版制限検出。APIフォールバック推奨。")
                return CLIResult(
                    text=stdout_text,
                    success=False,
                    rate_limited=True,
                    error="Pro rate limit reached",
                )

            if proc.returncode != 0:
                logger.error("[CLI] Exit code %d: %s", proc.returncode, stderr_text[:200])
                return CLIResult(
                    text=stdout_text,
                    success=False,
                    rate_limited=False,
                    error=f"CLI error (exit {proc.returncode}): {stderr_text[:200]}",
                )

            return CLIResult(
                text=stdout_text,
                success=True,
                rate_limited=False,
                error="",
            )

        except asyncio.TimeoutError:
            logger.error("[CLI] Timeout after 600s")
            return CLIResult(
                text="",
                success=False,
                rate_limited=False,
                error="CLI timeout (600s)",
            )
        except Exception as e:
            logger.error("[CLI] Exception: %s", e)
            return CLIResult(
                text="",
                success=False,
                rate_limited=False,
                error=str(e),
            )


class CLIResult:
    """Result from Claude CLI execution."""

    __slots__ = ("text", "success", "rate_limited", "error")

    def __init__(self, text: str, success: bool, rate_limited: bool, error: str = ""):
        self.text = text
        self.success = success
        self.rate_limited = rate_limited
        self.error = error

    def __repr__(self) -> str:
        status = "ok" if self.success else ("rate_limited" if self.rate_limited else "error")
        return f"<CLIResult status={status} len={len(self.text)}>"
