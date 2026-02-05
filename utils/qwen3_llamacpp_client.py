"""
Bushidan Multi-Agent System v10.1 - llama.cpp Qwen3-Coder-30B Client

CPU推論に最適化されたllama.cppクライアント
HP ProDesk 600 (CPU only) での高速推論を実現

Model: Qwen3-Coder-30B-Q4_K_M.gguf (unsloth版)
Backend: llama.cpp (not Ollama)
Hardware: HP ProDesk 600 (Intel i5/i7, 16-32GB RAM, CPU only)

Key Optimizations:
- CPU-only inference with all physical cores
- Memory locking (mlock) to prevent swapping
- Optimized batch size for CPU
- 4096 context window for speed
- NUMA awareness for multi-socket CPUs

llama.cpp Server Settings:
- ./llama-server -m model.gguf -c 4096 -t 8 --mlock -b 512 --parallel 2 --host 0.0.0.0 --port 8080
"""

import asyncio
import logging
import subprocess
import os
import signal
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class LlamaCppConfig:
    """llama.cpp server configuration for CPU optimization"""

    # Model path
    model_path: str = "models/qwen3/Qwen3-Coder-30B-Q4_K_M.gguf"

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080

    # CPU Optimization
    threads: int = 6  # HP ProDesk 600: i5-8500 (6C/6T)
    context_size: int = 4096  # Reduced for speed
    batch_size: int = 512  # Optimal for CPU
    parallel_requests: int = 2  # Concurrent request handling

    # Memory Optimization
    mlock: bool = True  # Lock memory to prevent swapping
    mmap: bool = True  # Memory-mapped file access

    # Performance
    flash_attention: bool = False  # CPU doesn't benefit from flash attention
    numa: bool = False  # Enable if multi-socket CPU

    # Generation defaults
    default_temperature: float = 0.7
    default_top_p: float = 0.8
    default_top_k: int = 40
    repeat_penalty: float = 1.1


@dataclass
class Qwen3LlamaCppStats:
    """llama.cpp Qwen3-Coder-30B usage statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    context_overflow_count: int = 0
    average_tokens_per_second: float = 0.0
    total_inference_time_seconds: float = 0.0
    estimated_power_cost_yen: float = 0.0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0


class Qwen3LlamaCppClient:
    """
    llama.cpp based Local Qwen3-Coder-30B Client

    CPU最適化されたローカル推論クライアント
    HP ProDesk 600 (Intel i5/i7, CPU only) に最適化

    Model: Qwen3-Coder-30B-A3B-instruct-q4_k_m.gguf (Q4_K_M量子化)
    - 4096 context limit (速度最適化)
    - MoE architecture (3.3B active params)
    - Expected: 10-18 tok/s on CPU (i5-8500)
    - RAM usage: 16-20GB
    - Cost: ¥0 inference + ~¥3 electricity per task

    Role: Primary implementation layer (大将)
    - Handles all Medium/Complex tasks first
    - Falls back to Cloud Qwen3 (Kagemusha) if context > 4k
    """

    VERSION = "10.1"

    def __init__(
        self,
        config: Optional[LlamaCppConfig] = None,
        api_base: Optional[str] = None
    ):
        """
        Initialize llama.cpp client

        Args:
            config: LlamaCppConfig for server settings
            api_base: Override API base URL (default: http://127.0.0.1:8080)
        """
        self.config = config or LlamaCppConfig()
        self.api_base = api_base or f"http://{self.config.host}:{self.config.port}"

        # Statistics
        self.stats = Qwen3LlamaCppStats()

        # Server process handle (if we start it)
        self._server_process: Optional[subprocess.Popen] = None

        # Cost estimation (CPU is more efficient than GPU)
        self.power_cost_per_task_yen = 3.0  # Lower than GPU

        logger.info(
            f"🏯 llama.cpp Qwen3-Coder-30B client initialized\n"
            f"   📍 API: {self.api_base}\n"
            f"   🧵 Threads: {self.config.threads}\n"
            f"   📝 Context: {self.config.context_size}\n"
            f"   🔒 Memory Lock: {self.config.mlock}"
        )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        detect_overflow: bool = True
    ) -> str:
        """
        Generate completion using llama.cpp server

        Args:
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            detect_overflow: Whether to detect and report context overflow

        Returns:
            Generated text response

        Raises:
            ContextOverflowError: If context exceeds limit
            Exception: If API call fails
        """

        if temperature is None:
            temperature = self.config.default_temperature

        # Check for context overflow
        if detect_overflow:
            context_estimate = self._estimate_context_size(messages)
            if context_estimate > self.config.context_size:
                self.stats.context_overflow_count += 1
                logger.warning(
                    f"⚠️ Context overflow: {context_estimate} > {self.config.context_size}. "
                    f"Kagemusha (Cloud Qwen3) を推奨。"
                )
                raise ContextOverflowError(
                    f"Context size {context_estimate} exceeds local limit {self.config.context_size}"
                )

        start_time = asyncio.get_event_loop().time()

        try:
            # Make API request to llama.cpp server
            response_data = await self._make_request(messages, max_tokens, temperature)

            # Extract response text
            response_text = response_data.get("content", "")

            # Update statistics
            elapsed_time = asyncio.get_event_loop().time() - start_time
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.total_inference_time_seconds += elapsed_time
            self.stats.estimated_power_cost_yen += self.power_cost_per_task_yen

            # Update token counts if available
            if "tokens_predicted" in response_data:
                self.stats.completion_tokens_total += response_data["tokens_predicted"]
            if "tokens_evaluated" in response_data:
                self.stats.prompt_tokens_total += response_data["tokens_evaluated"]

            # Calculate tokens/second
            tokens_predicted = response_data.get("tokens_predicted", len(response_text.split()))
            if elapsed_time > 0:
                tok_per_sec = tokens_predicted / elapsed_time

                # Update running average
                n = self.stats.successful_requests
                self.stats.average_tokens_per_second = (
                    (self.stats.average_tokens_per_second * (n - 1) + tok_per_sec) / n
                )
            else:
                tok_per_sec = 0

            logger.info(
                f"🏯 llama.cpp生成完了: {tokens_predicted} tokens / {elapsed_time:.2f}s "
                f"({tok_per_sec:.1f} tok/s)"
            )

            return response_text

        except ContextOverflowError:
            raise
        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            logger.error(f"❌ llama.cpp生成失敗: {e}")
            raise

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Make API request to llama.cpp server

        llama.cpp server supports OpenAI-compatible API at /v1/chat/completions
        and native API at /completion

        Args:
            messages: Conversation messages
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Response dictionary with content and metadata
        """

        try:
            import httpx

            # Try OpenAI-compatible endpoint first
            url = f"{self.api_base}/v1/chat/completions"

            payload = {
                "model": "qwen3-coder-30b",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": self.config.default_top_p,
                "top_k": self.config.default_top_k,
                "repeat_penalty": self.config.repeat_penalty,
                "stream": False
            }

            async with httpx.AsyncClient(timeout=600.0) as client:  # 10min timeout for CPU
                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    result = response.json()

                    # OpenAI format response
                    if "choices" in result:
                        choice = result["choices"][0]
                        return {
                            "content": choice["message"]["content"],
                            "tokens_predicted": result.get("usage", {}).get("completion_tokens", 0),
                            "tokens_evaluated": result.get("usage", {}).get("prompt_tokens", 0)
                        }

                # Fallback to native llama.cpp endpoint
                return await self._make_native_request(messages, max_tokens, temperature)

        except httpx.ConnectError:
            logger.warning("⚠️ OpenAI互換エンドポイント接続失敗、ネイティブAPIを試行...")
            return await self._make_native_request(messages, max_tokens, temperature)
        except Exception as e:
            logger.error(f"❌ llama.cpp API request failed: {e}")
            raise

    async def _make_native_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Make request using llama.cpp native /completion endpoint

        Args:
            messages: Conversation messages
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Response dictionary
        """

        try:
            import httpx

            url = f"{self.api_base}/completion"

            # Convert messages to single prompt
            prompt = self._format_chat_prompt(messages)

            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_p": self.config.default_top_p,
                "top_k": self.config.default_top_k,
                "repeat_penalty": self.config.repeat_penalty,
                "stop": ["<|im_end|>", "<|endoftext|>", "</s>"],
                "stream": False
            }

            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(url, json=payload)

                if response.status_code != 200:
                    error_detail = response.text
                    raise Exception(f"llama.cpp API error {response.status_code}: {error_detail}")

                result = response.json()

                return {
                    "content": result.get("content", ""),
                    "tokens_predicted": result.get("tokens_predicted", 0),
                    "tokens_evaluated": result.get("tokens_evaluated", 0)
                }

        except Exception as e:
            logger.error(f"❌ llama.cpp native API request failed: {e}")
            raise

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into Qwen3 chat template

        Qwen3 ChatML format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant

        Args:
            messages: Conversation messages

        Returns:
            Formatted prompt string
        """

        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Add assistant start token for generation
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def _estimate_context_size(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate context size in tokens

        Rough estimation for Japanese/English mix: ~1.5 tokens per word
        Qwen tokenizer tends to be more efficient for CJK

        Args:
            messages: Conversation messages

        Returns:
            Estimated token count
        """

        total_text = " ".join([msg.get("content", "") for msg in messages])

        # Count characters (better for Japanese)
        char_count = len(total_text)

        # Qwen tokenizer: ~1.5 chars per token for mixed content
        token_estimate = int(char_count / 1.5)

        # Add overhead for chat template
        token_estimate += len(messages) * 10

        return token_estimate

    def compress_context(
        self,
        messages: List[Dict[str, str]],
        target_tokens: int
    ) -> List[Dict[str, str]]:
        """
        Compress context to fit within target token limit

        Strategy:
        1. Keep system message
        2. Keep most recent user message
        3. Summarize/truncate middle messages

        Args:
            messages: Original messages
            target_tokens: Target token count

        Returns:
            Compressed message list
        """

        if len(messages) <= 2:
            return messages

        compressed = []

        # Keep system message
        if messages[0].get("role") == "system":
            compressed.append(messages[0])
            remaining = messages[1:]
        else:
            remaining = messages

        # Keep last user message
        if remaining:
            compressed.append(remaining[-1])

        # Check size
        current_estimate = self._estimate_context_size(compressed)

        if current_estimate > target_tokens:
            # Truncate last message
            last_msg = compressed[-1]
            content = last_msg["content"]
            target_chars = int(target_tokens * 1.5)
            truncated = content[:target_chars] + "... [truncated]"
            compressed[-1] = {
                "role": last_msg["role"],
                "content": truncated
            }

        logger.info(f"📦 Context圧縮: {len(messages)} → {len(compressed)} messages")
        return compressed

    async def health_check(self) -> bool:
        """
        Check if llama.cpp server is available

        Returns:
            True if healthy, False otherwise
        """

        try:
            import httpx

            # Check /health endpoint
            url = f"{self.api_base}/health"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "unknown")

                    if status == "ok":
                        logger.info("✅ llama.cpp server健全性確認済み")
                        return True

            # Fallback: try a simple completion
            test_messages = [{"role": "user", "content": "Hi"}]
            await self.generate(test_messages, max_tokens=5, detect_overflow=False)
            logger.info("✅ llama.cpp server応答確認済み")
            return True

        except Exception as e:
            logger.warning(f"⚠️ llama.cpp server健全性確認失敗: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics

        Returns:
            Dictionary with usage metrics
        """

        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests

        overflow_rate = 0.0
        if self.stats.total_requests > 0:
            overflow_rate = self.stats.context_overflow_count / self.stats.total_requests

        return {
            "backend": "llama.cpp",
            "model": "Qwen3-Coder-30B-A3B-instruct-q4_k_m.gguf",
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": round(success_rate, 3),
            "context_overflow_count": self.stats.context_overflow_count,
            "overflow_rate": round(overflow_rate, 3),
            "average_tokens_per_second": round(self.stats.average_tokens_per_second, 1),
            "total_inference_time_seconds": round(self.stats.total_inference_time_seconds, 1),
            "prompt_tokens_total": self.stats.prompt_tokens_total,
            "completion_tokens_total": self.stats.completion_tokens_total,
            "estimated_power_cost_yen": round(self.stats.estimated_power_cost_yen, 2),
            "context_size": self.config.context_size,
            "cpu_threads": self.config.threads,
            "optimizations": {
                "backend": "llama.cpp (CPU optimized)",
                "quantization": "Q4_K_M",
                "memory_lock": self.config.mlock,
                "batch_size": self.config.batch_size,
                "expected_speed": "15-25 tok/s (i7 CPU)"
            }
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics"""
        self.stats = Qwen3LlamaCppStats()
        logger.info("📊 llama.cpp統計リセット完了")

    def get_server_command(self) -> str:
        """
        Get the llama.cpp server startup command

        Returns:
            Shell command to start llama.cpp server
        """

        cmd_parts = [
            "llama-server",  # or ./llama-server for local build
            f"-m {self.config.model_path}",
            f"-c {self.config.context_size}",
            f"-t {self.config.threads}",
            f"-b {self.config.batch_size}",
            f"--parallel {self.config.parallel_requests}",
            f"--host {self.config.host}",
            f"--port {self.config.port}"
        ]

        if self.config.mlock:
            cmd_parts.append("--mlock")

        if self.config.mmap:
            cmd_parts.append("--mmap")

        if self.config.numa:
            cmd_parts.append("--numa")

        return " ".join(cmd_parts)

    async def start_server(self, llama_cpp_path: str = "./llama.cpp") -> bool:
        """
        Start llama.cpp server process

        Args:
            llama_cpp_path: Path to llama.cpp directory

        Returns:
            True if server started successfully
        """

        if self._server_process is not None:
            logger.warning("⚠️ llama.cpp server already running")
            return True

        server_path = Path(llama_cpp_path) / "llama-server"

        if not server_path.exists():
            # Try build directory
            server_path = Path(llama_cpp_path) / "build" / "bin" / "llama-server"

        if not server_path.exists():
            logger.error(f"❌ llama-server not found at {server_path}")
            return False

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            logger.error(f"❌ Model not found: {model_path}")
            return False

        cmd = [
            str(server_path),
            "-m", str(model_path),
            "-c", str(self.config.context_size),
            "-t", str(self.config.threads),
            "-b", str(self.config.batch_size),
            "--parallel", str(self.config.parallel_requests),
            "--host", self.config.host,
            "--port", str(self.config.port)
        ]

        if self.config.mlock:
            cmd.append("--mlock")

        if self.config.mmap:
            cmd.append("--mmap")

        if self.config.numa:
            cmd.append("--numa")

        try:
            logger.info(f"🚀 llama.cpp server起動中...\n   Command: {' '.join(cmd)}")

            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            # Wait for server to be ready
            await asyncio.sleep(5)

            if await self.health_check():
                logger.info("✅ llama.cpp server起動完了")
                return True
            else:
                logger.error("❌ llama.cpp server起動後の健全性確認失敗")
                self.stop_server()
                return False

        except Exception as e:
            logger.error(f"❌ llama.cpp server起動失敗: {e}")
            return False

    def stop_server(self) -> None:
        """Stop llama.cpp server process"""

        if self._server_process is not None:
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                self._server_process.wait(timeout=10)
                logger.info("✅ llama.cpp server停止完了")
            except Exception as e:
                logger.error(f"⚠️ llama.cpp server停止エラー: {e}")
                try:
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
                except:
                    pass
            finally:
                self._server_process = None


class ContextOverflowError(Exception):
    """Raised when context exceeds llama.cpp's configured limit"""
    pass


# Convenience function for HP ProDesk 600 setup
def create_prodesck_600_client() -> Qwen3LlamaCppClient:
    """
    Create a pre-configured client for HP ProDesk 600

    HP ProDesk 600 typical specs:
    - Intel Core i5-8500 (6C/6T, no HT)
    - 16-32GB DDR4 RAM
    - No discrete GPU (Intel UHD Graphics)

    Returns:
        Optimized Qwen3LlamaCppClient
    """

    config = LlamaCppConfig(
        model_path="models/Qwen3-Coder-30B-A3B-instruct-q4_k_m.gguf",
        host="127.0.0.1",
        port=8080,
        threads=6,  # i5-8500 has 6 cores
        context_size=4096,  # Reduced for CPU speed
        batch_size=512,  # Good for CPU
        parallel_requests=1,  # Single request for CPU stability
        mlock=True,  # Lock memory
        mmap=True,  # Memory-mapped loading
        numa=False,  # ProDesk 600 is single socket
        default_temperature=0.7,
        default_top_p=0.8,
        default_top_k=40,
        repeat_penalty=1.1
    )

    return Qwen3LlamaCppClient(config=config)


# Alias for backward compatibility
Qwen3Client = Qwen3LlamaCppClient
