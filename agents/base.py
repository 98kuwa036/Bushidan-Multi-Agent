"""Base Agent - エージェント基底クラス"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from shogun.core.task_queue import Task

logger = logging.getLogger("shogun.agent")


class BaseAgent(ABC):
    """Base class for all Shogun agents."""

    codename: str = ""
    role: str = ""
    tier: str = ""  # "cloud" or "local"

    def __init__(self, codename: str, role: str, tier: str):
        self.codename = codename
        self.role = role
        self.tier = tier
        self.logger = logging.getLogger(f"shogun.agent.{codename.lower()}")

    @abstractmethod
    async def execute(self, task: Task) -> str:
        """Execute a task and return the result."""
        ...

    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> str:
        """Raw generation (prompt -> response)."""
        ...

    def format_system_prompt(self, extra: str = "") -> str:
        """Build the system prompt for this agent."""
        base = f"あなたは「{self.codename}」です。役割: {self.role}"
        if extra:
            return f"{base}\n\n{extra}"
        return base

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} codename={self.codename} tier={self.tier}>"


class LocalAgent(BaseAgent):
    """Agent backed by a local Ollama model."""

    def __init__(
        self,
        codename: str,
        role: str,
        model: str,
        ollama_client: Any,
        num_ctx: int = 4096,
        temperature: float = 0.7,
        system_prompt: str = "",
    ):
        super().__init__(codename=codename, role=role, tier="local")
        self.model = model
        self.ollama = ollama_client
        self.num_ctx = num_ctx
        self.temperature = temperature
        self._system_prompt = system_prompt

    async def execute(self, task: Task) -> str:
        system = self._system_prompt or self.format_system_prompt()
        prompt = task.prompt

        # Include context if available
        if task.context:
            context_str = "\n".join(
                f"[{k}]: {v}" for k, v in task.context.items()
            )
            prompt = f"## コンテキスト\n{context_str}\n\n## タスク\n{prompt}"

        self.logger.info("[%s] Executing task %s", self.codename, task.id)
        result = await self.generate(prompt, system)
        self.logger.info("[%s] Task %s completed (%d chars)", self.codename, task.id, len(result))
        return result

    async def generate(self, prompt: str, system: str = "") -> str:
        sys = system or self._system_prompt or self.format_system_prompt()
        return await self.ollama.generate(
            model=self.model,
            prompt=prompt,
            system=sys,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )


class CloudAgent(BaseAgent):
    """Agent backed by Anthropic API."""

    def __init__(
        self,
        codename: str,
        role: str,
        model: str,
        anthropic_client: Any,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        system_prompt: str = "",
    ):
        super().__init__(codename=codename, role=role, tier="cloud")
        self.model = model
        self.anthropic = anthropic_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._system_prompt = system_prompt

    async def execute(self, task: Task) -> str:
        system = self._system_prompt or self.format_system_prompt()
        prompt = task.prompt

        if task.context:
            context_str = "\n".join(
                f"[{k}]: {v}" for k, v in task.context.items()
            )
            prompt = f"## コンテキスト\n{context_str}\n\n## タスク\n{prompt}"

        self.logger.info("[%s] Executing task %s (API)", self.codename, task.id)
        result = await self.generate(prompt, system)
        self.logger.info("[%s] Task %s completed (%d chars)", self.codename, task.id, len(result))
        return result

    async def generate(self, prompt: str, system: str = "") -> str:
        sys = system or self._system_prompt or self.format_system_prompt()
        return await self.anthropic.generate(
            model=self.model,
            prompt=prompt,
            system=sys,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
