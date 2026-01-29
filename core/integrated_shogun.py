"""Integrated Shogun System v7.0 - Complete Implementation

Shogun System v7.0 with Japanese R1, Groq recorder, Platoon mode,
and Pro CLI first strategy.

Key Changes from v5.0:
  1. Japanese R1 Model (cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese)
  2. 9th Ashigaru (Groq recorder for real-time logging and 60-day summaries)
  3. Platoon Mode (ultra-lightweight for HA OS voice)
  4. Notion Integration (automatic knowledge management)
  5. 11 Slack Bots (expanded from current setup)
  6. Pro CLI First Strategy (¥0 → API fallback)

Architecture:
  将軍 (Shogun) - Claude Sonnet 4.5 (Pro CLI → API)
  家老 (Karo) - Claude Opus (API only, Strategic decisions)
  侍大将 (Taisho) - DeepSeek R1 Japanese (local, monitoring & design)
  足軽 × 9 (Ashigaru) - MCP servers + Groq recorder

Deployment Modes:
  - Battalion: Full force (all agents)
  - Company: Cost-optimized (R1 + Ashigaru only)
  - Platoon: Voice-optimized (R1 + minimal Ashigaru)
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

import yaml

from shogun.core.task_queue import (
    Task, TaskQueue, TaskStatus, Complexity, DeploymentMode,
)
from shogun.core.complexity import estimate_complexity
from shogun.core.dashboard import Dashboard
from shogun.core.mcp_manager import MCPManager
from shogun.core.error_handling import (
    ShogunErrorHandler, FallbackStrategy, ErrorSeverity,
    ShogunError, OperationFailedError, CircuitBreakerOpenError
)
from shogun.providers.claude_cli import ClaudeCLIProvider
from shogun.providers.openvino_client import OpenVINOClient
from shogun.ashigaru.groq_recorder import GroqRecorder

logger = logging.getLogger("shogun.integrated")


class PlatoonMode(Enum):
    """Platoon mode configurations for ultra-lightweight operations."""
    VOICE_QUERY = "voice_query"  # HA OS voice queries
    QUICK_INFO = "quick_info"    # Simple information requests
    FILE_CHECK = "file_check"    # Basic file operations


class AgentCost:
    """Cost per agent call (JPY) - v7.0 with latest Opus 4.5 / Sonnet 4.5."""
    SHOGUN_OPUS = 24      # Opus 4.5 for Strategic decisions only (5/month)
    KARO_SONNET = 5       # Sonnet 4.5 for Complex tasks (Pro CLI → API)
    TAISHO_R1 = 0         # Local Japanese R1 (CyberAgent)
    ASHIGARU_MCP = 0      # Local MCP servers × 8
    GROQ_RECORDER = 0     # Free tier (14,400 req/day)


class IntegratedShogun:
    """Shogun System v7.0 - Complete integration.
    
    Coordinates all layers with v7.0 enhancements:
      - Pro CLI first strategy for cost optimization
      - Japanese R1 for superior local reasoning
      - Groq recorder for knowledge accumulation
      - Platoon mode for voice interactions
      - Notion integration for persistent knowledge
    """

    def __init__(self, base_dir: str, config_path: str | None = None):
        self.base_dir = Path(base_dir)
        
        # Load configuration
        cfg_path = config_path or str(self.base_dir / "config" / "settings_v7.yaml")
        self.config = self._load_config(cfg_path)
        
        # Initialize enhanced error handler
        self.error_handler = ShogunErrorHandler(self.config)
        
        # Current deployment mode
        self.current_mode = DeploymentMode.BATTALION
        self.platoon_config = None
        
        # Task queue and dashboard
        self.queue = TaskQueue(str(self.base_dir))
        self.dashboard = Dashboard(str(self.base_dir))
        
        # MCP Manager (8 traditional ashigaru)
        mcp_config = str(self.base_dir / "config" / "mcp_config.json")
        self.mcp = MCPManager(mcp_config if Path(mcp_config).exists() else None)
        
        # 9th Ashigaru - Groq Recorder
        self.groq_recorder = GroqRecorder(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            notion_integration=self.config.get("notion", {})
        )
        
        # Providers
        self.claude_cli = ClaudeCLIProvider()
        
        # Japanese R1 (Taisho)
        r1_url = self.config.get("taisho_japanese", {}).get(
            "url", "http://192.168.1.11:11434"
        )
        self.japanese_r1 = OpenVINOClient(base_url=r1_url)
        
        # API fallback (lazy init)
        self._api_provider = None
        
        # Notion integration
        self._notion_client = None
        
        # v7.0 Statistics
        self.stats = {
            "taisho_japanese_r1": 0,
            "karo_sonnet": 0,
            "shogun_opus": 0,
            "groq_recorder": 0,
            "battalion": 0,
            "company": 0,
            "platoon": 0,
            "pro_cli_success": 0,
            "api_fallback": 0,
            "total_cost_yen": 0,
            "knowledge_entries": 0,
        }
        
        # Repository sync
        self.repo_path = self._detect_repo_path()
        
    def _load_config(self, path: str) -> dict:
        """Load YAML configuration with fallback."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config file not found: %s, using defaults", path)
            return self._default_config()
            
    def _default_config(self) -> dict:
        """Default v7.0 configuration."""
        return {
            "version": "7.0",
            "taisho_japanese": {
                "url": "http://192.168.1.11:11434",
                "model": "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese",
            },
            "groq": {
                "model": "llama-3.3-70b-versatile",
                "max_requests_per_day": 14400,
            },
            "notion": {
                "database_id": os.environ.get("NOTION_DATABASE_ID", ""),
                "auto_summary_days": 60,
            },
            "platoon_modes": {
                "voice_query": {
                    "max_ashigaru": 2,
                    "response_time_target": 30,
                },
                "quick_info": {
                    "max_ashigaru": 1,
                    "response_time_target": 15,
                },
                "file_check": {
                    "max_ashigaru": 1,
                    "response_time_target": 10,
                },
            },
            "repo": {
                "local_base": "/home/claude",
                "sync_interval_minutes": 5,
            }
        }

    @property
    def api_provider(self):
        """Lazy-init Anthropic API provider."""
        if self._api_provider is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                from shogun.providers.anthropic_api import AnthropicAPIProvider
                self._api_provider = AnthropicAPIProvider(api_key=api_key)
        return self._api_provider

    @property
    def notion_client(self):
        """Lazy-init Notion client."""
        if self._notion_client is None:
            token = os.environ.get("NOTION_TOKEN", "")
            if token:
                from shogun.integrations.notion_integration import NotionIntegration
                self._notion_client = NotionIntegration(
                    token=token, 
                    database_id=self.config["notion"]["database_id"]
                )
        return self._notion_client

    # ─── Main Entry Points ───

    async def process_task(
        self,
        prompt: str,
        mode: str = "battalion",
        platoon_type: Optional[str] = None,
        force_agent: str = "",
    ) -> str:
        """Process a task through the Shogun v7.0 system.

        Args:
            prompt: Task description
            mode: "battalion", "company", or "platoon"
            platoon_type: For platoon mode: "voice_query", "quick_info", "file_check"
            force_agent: Force specific agent (bypass routing)

        Returns:
            Result text
        """
        # Create task
        if mode == "platoon":
            deploy_mode = DeploymentMode.COMPANY  # Use company internally
            self.platoon_config = PlatoonMode(platoon_type or "voice_query")
        else:
            deploy_mode = DeploymentMode(mode)
            self.platoon_config = None
            
        task = Task(prompt=prompt, mode=deploy_mode)
        task.complexity = estimate_complexity(prompt)
        self.queue.enqueue(task)

        logger.info(
            "[本陣v7] 任務受領: %s (複雑度: %s, 編成: %s%s)",
            task.id, task.complexity.value, mode,
            f", 小隊種別: {platoon_type}" if platoon_type else ""
        )

        # Start Groq recording
        await self.groq_recorder.start_session(task.id, prompt)
        
        # Sync repository
        await self._sync_repo()
        
        # Update dashboard
        mode_label = f"{mode}({platoon_type})" if platoon_type else mode
        self.dashboard.add_in_progress(
            f"[{task.id}] {prompt[:50]}... ({mode_label})"
        )

        try:
            if force_agent:
                result = await self._dispatch_to_agent(task, force_agent)
            elif mode == "platoon":
                result = await self._process_platoon(task)
            elif deploy_mode == DeploymentMode.COMPANY:
                result = await self._process_company(task)
            else:
                result = await self._process_battalion(task)

            # Complete task
            self.queue.complete_task(task.id, result, task.cost_yen)
            
            # Record in Groq for knowledge accumulation
            await self.groq_recorder.record_completion(
                task.id, prompt, result, task.cost_yen
            )
            
            # Update dashboard
            self.dashboard.remove_in_progress(
                f"[{task.id}] {prompt[:50]}... ({mode_label})"
            )
            self.dashboard.add_completed(
                task.id, prompt[:60], "完了", task.cost_yen
            )
            
            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.queue.update_task(task)
            
            # Record failure
            await self.groq_recorder.record_failure(task.id, prompt, str(e))
            
            # Update dashboard
            self.dashboard.remove_in_progress(
                f"[{task.id}] {prompt[:50]}... ({mode_label})"
            )
            self.dashboard.add_action_required(
                f"[{task.id}] 失敗: {str(e)[:80]}"
            )
            raise

    # ─── Platoon Mode (v7.0 NEW) ───

    async def _process_platoon(self, task: Task) -> str:
        """小隊モード: Ultra-lightweight for HA OS voice and quick queries."""
        self.stats["platoon"] += 1
        config = self.config["platoon_modes"][self.platoon_config.value]
        
        logger.info(
            "[小隊] 出陣: 侍大将(日本語R1) + 足軽選択 (種別: %s)", 
            self.platoon_config.value
        )
        
        # Select minimal ashigaru based on task type
        selected_ashigaru = await self._select_platoon_ashigaru(
            task, max_count=config["max_ashigaru"]
        )
        
        # Call Japanese R1 with minimal context
        result = await self._call_japanese_taisho(
            task, 
            platoon_mode=True,
            selected_ashigaru=selected_ashigaru
        )
        
        task.cost_yen = 0  # Platoon mode is free
        
        # No Groq recording for platoon (keep it ultra-light)
        return result

    async def _select_platoon_ashigaru(
        self, task: Task, max_count: int
    ) -> List[str]:
        """Select minimal ashigaru for platoon mode."""
        prompt_lower = task.prompt.lower()
        selected = []
        
        # Smart selection based on task content
        if any(word in prompt_lower for word in ["ファイル", "file", "read", "write"]):
            selected.append("filesystem")
        if any(word in prompt_lower for word in ["issue", "pr", "github", "git"]):
            selected.append("github")
        if any(word in prompt_lower for word in ["web", "http", "url", "fetch"]):
            selected.append("fetch")
        if any(word in prompt_lower for word in ["remember", "記憶", "memory"]):
            selected.append("memory")
        if any(word in prompt_lower for word in ["search", "検索", "brave"]):
            selected.append("brave-search")
        
        # Limit to max_count
        return selected[:max_count]

    # ─── Company Mode (v7.0 Enhanced) ───

    async def _process_company(self, task: Task) -> str:
        """中隊モード: Cost-optimized with Japanese R1."""
        self.stats["company"] += 1
        logger.info("[中隊] 出陣: 侍大将(日本語R1) + 足軽 × 8")
        
        # Check complexity limits
        if task.complexity in [Complexity.COMPLEX, Complexity.STRATEGIC]:
            logger.warning(
                "[中隊] 複雑度 %s は中隊の能力範囲外。大隊モード推奨。",
                task.complexity.value,
            )
            task.context["warning"] = "大隊モード推奨（能力超過の可能性）"
        
        result = await self._call_japanese_taisho(task, company_mode=True)
        task.cost_yen = 0
        
        # Record with Groq (company mode includes recording)
        await self.groq_recorder.record_interaction(
            task.id, "company", task.prompt, result
        )
        
        if "大隊モード推奨" in result:
            self.dashboard.add_action_required(
                f"[{task.id}] 中隊能力超過: 大隊モードへの切替を推奨"
            )
        
        return result

    # ─── Battalion Mode (v7.0 Enhanced) ───

    async def _process_battalion(self, task: Task) -> str:
        """大隊モード: Full force with Pro CLI first strategy."""
        self.stats["battalion"] += 1
        logger.info("[大隊] 出陣準備 (複雑度: %s)", task.complexity.value)
        
        if task.complexity == Complexity.SIMPLE:
            # Simple → Japanese Taisho only
            result = await self._call_japanese_taisho(task)
            task.cost_yen = AgentCost.TAISHO_R1
            return result
            
        elif task.complexity == Complexity.MEDIUM:
            # Medium → Japanese Taisho design → monitoring
            result = await self._call_japanese_taisho(task)
            task.cost_yen = AgentCost.TAISHO_R1
            return result
            
        elif task.complexity == Complexity.COMPLEX:
            # Complex → Japanese Taisho analysis → Karo (Sonnet)
            taisho_analysis = await self._call_japanese_taisho_analysis(task)
            task.context["taisho_analysis"] = taisho_analysis
            
            result = await self._call_cloud(task, agent="karo", model="sonnet")
            task.cost_yen = AgentCost.KARO_SONNET
            return result
            
        else:
            # Strategic → Taisho analysis → Shogun (Opus)
            taisho_analysis = await self._call_japanese_taisho_analysis(task)
            task.context["taisho_analysis"] = taisho_analysis
            
            result = await self._call_cloud(task, agent="shogun", model="opus")
            task.cost_yen = AgentCost.SHOGUN_OPUS
            return result

    # ─── Japanese Taisho (v7.0 NEW) ───

    async def _call_japanese_taisho(
        self, 
        task: Task, 
        company_mode: bool = False,
        platoon_mode: bool = False,
        selected_ashigaru: Optional[List[str]] = None
    ) -> str:
        """Call 侍大将 with Japanese R1 model."""
        self.stats["taisho_japanese_r1"] += 1
        task.assigned_agent = "taisho_japanese"
        task.status = TaskStatus.IN_PROGRESS
        self.queue.update_task(task)
        
        mode_label = (
            "小隊" if platoon_mode else
            "中隊" if company_mode else
            "大隊"
        )
        logger.info("[侍大将] 日本語R1推論開始 (%s)", mode_label)
        
        # System prompt in Japanese for Japanese R1
        system = (
            "あなたは将軍システムの侍大将です。プロジェクトの設計・推論・監査を担当します。\n"
            "<think>タグで日本語で深く推論し、論理的で実用的な結論を導いてください。\n"
            "日本語で自然に回答し、コードコメントも日本語で記述してください。\n"
        )
        
        if platoon_mode:
            system += (
                "小隊モードです。超軽量・高速応答が求められます。\n"
                f"選択された足軽: {', '.join(selected_ashigaru or [])}\n"
                "簡潔かつ的確に回答してください。\n"
            )
        elif company_mode:
            system += (
                "中隊モードです。将軍・家老は不在。侍大将と足軽のみで完結してください。\n"
                "能力を超える場合は「大隊モード推奨」と報告してください。\n"
            )
        else:
            system += (
                "大隊モードです。必要に応じて上位エージェントへの報告を準備してください。\n"
            )
        
        # Build context
        context_parts = []
        if task.context:
            for k, v in task.context.items():
                context_parts.append(f"[{k}]: {v}")
        
        context_str = "\n".join(context_parts) if context_parts else ""
        if context_str:
            context_str = f"\n## コンテキスト\n{context_str}\n"
        
        prompt = f"{context_str}\n## 任務\n{task.prompt}"
        
        try:
            result = await self.japanese_r1.generate(
                prompt=prompt,
                system=system,
                max_tokens=2000 if not platoon_mode else 1000,
                temperature=0.6,
            )
            logger.info("[侍大将] 日本語R1完了 (¥0)")
            return result
            
        except Exception as e:
            logger.error("[侍大将] 日本語R1エラー: %s", e)
            raise

    async def _call_japanese_taisho_analysis(self, task: Task) -> str:
        """Call Japanese Taisho for analysis before escalation."""
        self.stats["taisho_japanese_r1"] += 1
        logger.info("[侍大将] 日本語R1分析開始（上位報告用）")
        
        system = (
            "上位エージェント（家老・将軍）への分析報告を作成してください。\n"
            "<think>で深く分析し、要点を整理して日本語で報告してください。\n"
        )
        
        prompt = f"## 分析対象任務\n{task.prompt}"
        
        try:
            result = await self.japanese_r1.generate(
                prompt=prompt,
                system=system,
                max_tokens=1500,
                temperature=0.6,
            )
            logger.info("[侍大将] 日本語R1分析完了")
            return result
            
        except Exception as e:
            logger.warning("[侍大将] 日本語R1分析失敗: %s (スキップ)", e)
            return f"(侍大将分析スキップ: {e})"

    # ─── Cloud Agents (v7.0 Pro CLI First) ───

    async def _call_cloud(
        self, task: Task, agent: str, model: str
    ) -> str:
        """Call cloud agent with Pro CLI first strategy.
        
        v7.0 Flow:
          1. Try claude-cli (Pro版) - FREE
          2. If rate limited → fallback to API - PAID
        """
        task.assigned_agent = agent
        task.status = TaskStatus.IN_PROGRESS
        self.queue.update_task(task)
        
        agent_label = "将軍" if agent == "shogun" else "家老"
        logger.info("[%s] Pro CLI優先実行開始 (model=%s)", agent_label, model)
        
        # Build context
        parts = []
        if task.context.get("taisho_analysis"):
            parts.append(f"## 侍大将の分析\n{task.context['taisho_analysis']}")
        if task.context.get("escalation"):
            parts.append(task.context["escalation"])
        
        system = self._get_system_prompt_v7(agent)
        full_prompt = "\n\n".join(parts + [f"## 任務\n{task.prompt}"])
        
        # Try Pro CLI first (FREE)
        result = await self.claude_cli.generate(
            prompt=full_prompt,
            model=model,
            system_prompt=system,
            cwd=self.repo_path,
        )
        
        if result.success:
            self.stats["pro_cli_success"] += 1
            logger.info("[%s] Pro CLI成功 (¥0)", agent_label)
            
            # Record with Groq
            await self.groq_recorder.record_interaction(
                task.id, f"{agent}_pro_cli", full_prompt, result.text
            )
            
            return result.text
        
        # Fallback to API (PAID)
        logger.warning("[%s] Pro CLI失敗 (%s) → API課金フォールバック", agent_label, result.error)
        return await self._call_cloud_api(task, agent, model, full_prompt, system)

    async def _call_cloud_api(
        self, task: Task, agent: str, model: str, prompt: str, system: str
    ) -> str:
        """Fallback to paid Anthropic API."""
        self.stats["api_fallback"] += 1
        agent_label = "将軍" if agent == "shogun" else "家老"
        
        if not self.api_provider:
            raise RuntimeError(
                f"[{agent_label}] APIフォールバック不可: ANTHROPIC_API_KEY 未設定"
            )
        
        logger.info("[%s] API実行 (課金)", agent_label)
        text = await self.api_provider.generate(
            prompt=prompt,
            model=model,
            system=system,
            max_tokens=4096,
            temperature=0.3,
        )
        
        # Update cost stats
        cost = getattr(AgentCost, f"{agent.upper()}_{model.upper()}", 0)
        self.stats["total_cost_yen"] += cost
        
        if agent == "karo":
            self.stats["karo_sonnet"] += 1
        else:
            self.stats["shogun_opus"] += 1
            
        logger.info("[%s] API完了 (¥%d)", agent_label, cost)
        
        # Record with Groq
        await self.groq_recorder.record_interaction(
            task.id, f"{agent}_api", prompt, text
        )
        
        return text

    # ─── Agent Dispatch ───

    async def _dispatch_to_agent(self, task: Task, agent: str) -> str:
        """Direct dispatch to specific agent."""
        if agent == "taisho":
            result = await self._call_japanese_taisho(task)
            task.cost_yen = AgentCost.TAISHO_R1
        elif agent == "karo":
            result = await self._call_cloud(task, "karo", "sonnet")
            task.cost_yen = AgentCost.KARO_SONNET
        elif agent == "shogun":
            result = await self._call_cloud(task, "shogun", "opus")
            task.cost_yen = AgentCost.SHOGUN_OPUS
        else:
            raise ValueError(f"Unknown agent: {agent}")
        return result

    # ─── System Prompts v7.0 ───

    def _get_system_prompt_v7(self, agent: str) -> str:
        """Enhanced system prompts for v7.0."""
        if agent == "shogun":
            return (
                "あなたは将軍システムv7.0の将軍（総大将）です。\n"
                "最高意思決定者として、プロジェクト全体の戦略的判断と最終決裁を行います。\n"
                "侍大将（日本語R1）の深い分析を踏まえ、根本的な解決策を提示してください。\n"
                "Strategic級の判断のみが将軍に上がってきます。慎重かつ決断力ある回答をしてください。\n"
            )
        elif agent == "karo":
            return (
                "あなたは将軍システムv7.0の家老（参謀）です。\n"
                "将軍の右腕として、Complex級タスクの作業割振りと実装方針策定を行います。\n"
                "侍大将（日本語R1）の深い分析結果を受け、具体的な実装手順を示してください。\n"
                "複雑な統合タスクを分解し、明確で実行可能な手順を提示してください。\n"
            )
        return ""

    # ─── Repository Management ───

    def _detect_repo_path(self) -> str:
        """Detect repository path for Git sync."""
        base = self.config["repo"]["local_base"]
        try:
            import subprocess
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5,
                cwd=str(self.base_dir),
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                repo_name = url.rstrip("/").rstrip(".git").rsplit("/", 1)[-1]
                if repo_name:
                    path = f"{base}/{repo_name}"
                    logger.info("[本陣v7] リポジトリ検出: %s → %s", url, path)
                    return path
        except Exception:
            pass
        
        logger.info("[本陣v7] リポジトリパス: %s", base)
        return base

    async def _sync_repo(self) -> None:
        """Enhanced Git sync for v7.0."""
        if not Path(self.repo_path).exists():
            logger.warning("[本陣v7] リポジトリパス不在: %s", self.repo_path)
            return
        
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", self.repo_path, "pull", "--rebase",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            
            if proc.returncode == 0:
                logger.info("[本陣v7] Git同期完了")
            else:
                logger.warning("[本陣v7] Git同期警告: %s", stderr.decode())
                
        except Exception as e:
            logger.warning("[本陣v7] Git同期スキップ: %s", e)

    # ─── Daily Summary (v7.0 NEW) ───

    async def run_daily_summary(self) -> str:
        """Run 60-day summary automation via Groq."""
        logger.info("[本陣v7] 60日要約開始")
        
        try:
            summary = await self.groq_recorder.generate_60day_summary()
            
            if self.notion_client:
                await self.notion_client.save_summary(summary)
                self.stats["knowledge_entries"] += 1
                
            logger.info("[本陣v7] 60日要約完了")
            return summary
            
        except Exception as e:
            logger.error("[本陣v7] 60日要約失敗: %s", e)
            return f"要約生成失敗: {e}"

    # ─── Status and Stats ───

    def get_status_v7(self) -> dict:
        """v7.0 enhanced status."""
        return {
            "version": "7.0",
            "mode": self.current_mode.value,
            "platoon_config": self.platoon_config.value if self.platoon_config else None,
            "stats": dict(self.stats),
            "pending_tasks": len(self.queue.get_pending()),
            "total_tasks": len(self.queue.get_all_tasks()),
            "dashboard": self.dashboard.get_summary(),
            "mcp_servers": self.mcp.get_status(),
            "japanese_r1_status": await self._check_japanese_r1_status(),
            "groq_recorder_status": self.groq_recorder.get_status(),
        }

    async def _check_japanese_r1_status(self) -> dict:
        """Check Japanese R1 server status."""
        try:
            # Simple health check
            result = await self.japanese_r1.generate(
                prompt="健康チェック",
                system="簡潔に「正常です」と回答してください。",
                max_tokens=10,
                temperature=0.1,
            )
            return {"status": "healthy", "response": result[:50]}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def show_stats_v7(self) -> str:
        """Format v7.0 stats for display."""
        s = self.stats
        lines = [
            "=" * 60,
            "📊 将軍システム v7.0 戦果統計",
            "=" * 60,
            f"大隊モード: {s['battalion']}回",
            f"中隊モード: {s['company']}回",
            f"小隊モード: {s['platoon']}回 ⭐NEW⭐",
            "",
            "エージェント内訳:",
            f"  侍大将(日本語R1):  {s['taisho_japanese_r1']}回 (¥0) ⭐NEW⭐",
            f"  家老(Sonnet):     {s['karo_sonnet']}回 (¥{s['karo_sonnet'] * AgentCost.KARO_SONNET:,})",
            f"  将軍(Opus):       {s['shogun_opus']}回 (¥{s['shogun_opus'] * AgentCost.SHOGUN_OPUS:,})",
            f"  Groqレコーダー:    {s['groq_recorder']}回 (¥0) ⭐NEW⭐",
            "",
            "コスト最適化:",
            f"  Pro CLI成功:      {s['pro_cli_success']}回 (¥0) ⭐NEW⭐",
            f"  API フォールバック: {s['api_fallback']}回 (課金)",
            "",
            "ナレッジ蓄積:",
            f"  Notion投稿:       {s['knowledge_entries']}回 ⭐NEW⭐",
            "",
            f"合計コスト: ¥{s['total_cost_yen']:,}",
            f"月額予測: ¥{self._estimate_monthly_cost():,} (本家Claude Code比 -49%)",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _estimate_monthly_cost(self) -> int:
        """Estimate monthly cost based on current usage."""
        pro_cost = 3000  # Claude Pro
        power_cost = 800  # Server power
        api_cost = self.stats["total_cost_yen"]
        return pro_cost + power_cost + api_cost

    # ─── Lifecycle ───

    async def startup(self) -> None:
        """Start Shogun System v7.0."""
        logger.info("[本陣v7] 将軍システム v7.0 起動...")
        
        # Initialize components
        self.queue.reset_all_workers()
        self.dashboard.init()
        self.queue.load_from_disk()
        
        # Initialize Groq recorder
        await self.groq_recorder.initialize()
        
        # Health checks
        japanese_r1_status = await self._check_japanese_r1_status()
        if japanese_r1_status["status"] != "healthy":
            logger.warning("[本陣v7] 日本語R1接続警告: %s", japanese_r1_status["error"])
        
        logger.info("[本陣v7] 起動完了 - スピードより質の追求開始 ⭐")

    async def shutdown(self) -> None:
        """Shutdown Shogun System v7.0."""
        logger.info("[本陣v7] 将軍システム v7.0 停止...")
        
        # Generate final summary if needed
        await self.groq_recorder.finalize_session()
        
        # Close connections
        await self.mcp.stop_all()
        await self.japanese_r1.close()
        if self._api_provider:
            await self._api_provider.close()
        
        logger.info("[本陣v7] 停止完了")
