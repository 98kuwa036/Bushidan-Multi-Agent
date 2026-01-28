"""Controller - 本陣 (統合制御)

大隊モード / 中隊モード の切替と、タスクルーティングを制御する。

処理フロー:
  1. タスク受信 → 複雑度判定
  2. モードに応じたルーティング
     - 中隊: 侍大将 + 足軽(MCP) のみ (¥0)
     - 大隊: 複雑度に応じて 侍大将 → 家老 → 将軍

クラウドエージェントの実行:
  Primary: claude-cli (Pro版, npm)
  Fallback: Anthropic API (console.anthropic.com, 課金)
"""

import asyncio
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from shogun.core.task_queue import (
    Task, TaskQueue, TaskStatus, Complexity, DeploymentMode,
)
from shogun.core.complexity import estimate_complexity, estimated_cost_yen
from shogun.core.escalation import (
    get_handler, get_next_escalation, should_escalate,
    can_handle_in_company_mode, build_escalation_context,
    build_taisho_analysis_prompt, AGENT_COST,
)
from shogun.core.dashboard import Dashboard
from shogun.core.mcp_manager import MCPManager
from shogun.providers.claude_cli import ClaudeCLIProvider, CLIResult
from shogun.providers.openvino_client import OpenVINOClient

logger = logging.getLogger("shogun.controller")


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Controller:
    """本陣: Central controller for the Shogun system.

    Coordinates all layers:
      - Cloud: claude-cli (Pro) → Anthropic API (fallback)
      - Local: OpenVINO R1 (侍大将)
      - Tools: MCP servers × 8 (足軽)
    """

    def __init__(self, base_dir: str, config_path: str | None = None):
        self.base_dir = Path(base_dir)

        # Load config
        cfg_path = config_path or str(self.base_dir / "config" / "settings.yaml")
        self.config = _load_config(cfg_path) if Path(cfg_path).exists() else {}

        # Current deployment mode
        self.current_mode = DeploymentMode.BATTALION

        # Task queue
        self.queue = TaskQueue(str(self.base_dir))

        # Dashboard
        self.dashboard = Dashboard(str(self.base_dir))

        # MCP Manager (足軽)
        mcp_config = str(self.base_dir / "config" / "mcp_config.json")
        self.mcp = MCPManager(mcp_config if Path(mcp_config).exists() else None)

        # Providers
        self.claude_cli = ClaudeCLIProvider()

        r1_url = self.config.get("taisho", {}).get(
            "url", "http://192.168.1.11:11434"
        )
        self.openvino = OpenVINOClient(base_url=r1_url)

        # API fallback (lazy init)
        self._api_provider = None

        # Stats
        self.stats = {
            "taisho_r1": 0,
            "karo_sonnet": 0,
            "shogun_opus": 0,
            "battalion": 0,
            "company": 0,
            "api_fallback": 0,
            "total_cost_yen": 0,
        }

        # Repo path for git sync
        # GitHub username/ 以降と /home/claude/ 以降を同期
        repo_cfg = self.config.get("repo", {})
        self.repo_local_base = repo_cfg.get("local_base", "/home/claude")
        self.repo_path = self._detect_repo_path()

    @property
    def api_provider(self):
        """Lazy-init Anthropic API provider."""
        if self._api_provider is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                from shogun.providers.anthropic_api import AnthropicAPIProvider
                self._api_provider = AnthropicAPIProvider(api_key=api_key)
        return self._api_provider

    # ─── Main Entry Point ───

    async def process_task(
        self,
        prompt: str,
        mode: str = "battalion",
        force_agent: str = "",
    ) -> str:
        """Process a task through the Shogun system.

        Args:
            prompt: Task description.
            mode: "battalion" or "company".
            force_agent: Force a specific agent (bypass routing).

        Returns:
            Result text.
        """
        # Create task
        deploy_mode = DeploymentMode(mode)
        task = Task(prompt=prompt, mode=deploy_mode)
        task.complexity = estimate_complexity(prompt)
        self.queue.enqueue(task)

        logger.info(
            "[本陣] 任務受領: %s (複雑度: %s, 編成: %s)",
            task.id, task.complexity.value, deploy_mode.value,
        )

        # Sync repo
        await self._sync_repo()

        # Dashboard update
        self.dashboard.add_in_progress(
            f"[{task.id}] {prompt[:50]}... ({deploy_mode.value})"
        )

        try:
            if force_agent:
                result = await self._dispatch_to_agent(task, force_agent)
            elif deploy_mode == DeploymentMode.COMPANY:
                result = await self._process_company(task)
            else:
                result = await self._process_battalion(task)

            # Complete
            self.queue.complete_task(task.id, result, task.cost_yen)
            self.dashboard.remove_in_progress(
                f"[{task.id}] {prompt[:50]}... ({deploy_mode.value})"
            )
            self.dashboard.add_completed(
                task.id, prompt[:60], "完了", task.cost_yen
            )
            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.queue.update_task(task)
            self.dashboard.remove_in_progress(
                f"[{task.id}] {prompt[:50]}... ({deploy_mode.value})"
            )
            self.dashboard.add_action_required(
                f"[{task.id}] 失敗: {str(e)[:80]}"
            )
            raise

    # ─── Company Mode (中隊) ───

    async def _process_company(self, task: Task) -> str:
        """中隊モード: 侍大将 + 足軽(MCP) のみ。API不使用 (¥0)。"""
        self.stats["company"] += 1
        logger.info("[中隊] 出陣: 侍大将 + 足軽 × 8")

        # Check if task is within company capabilities
        if not can_handle_in_company_mode(task.complexity):
            logger.warning(
                "[中隊] 複雑度 %s は中隊の能力範囲外。大隊モード推奨。",
                task.complexity.value,
            )
            # Still try with Taisho, but add warning
            task.context["warning"] = "大隊モード推奨（能力超過の可能性）"

        result = await self._call_taisho(task, company_mode=True)
        task.cost_yen = 0

        # Check if Taisho recommends battalion escalation
        if "大隊モード推奨" in result:
            self.dashboard.add_action_required(
                f"[{task.id}] 中隊能力超過: 大隊モードへの切替を推奨"
            )

        return result

    # ─── Battalion Mode (大隊) ───

    async def _process_battalion(self, task: Task) -> str:
        """大隊モード: 複雑度に応じてエージェントをルーティング。"""
        self.stats["battalion"] += 1
        logger.info(
            "[大隊] 出陣準備 (複雑度: %s)", task.complexity.value
        )

        handler = get_handler(task.complexity)

        if handler == "taisho":
            # Simple/Medium → 侍大将のみ (¥0)
            result = await self._call_taisho(task)
            task.cost_yen = 0
            return result

        elif handler == "karo":
            # Complex → 侍大将分析 → 家老(Sonnet)が作業割振り
            taisho_analysis = await self._call_taisho_analysis(task)
            task.context["taisho_analysis"] = taisho_analysis
            result = await self._call_cloud(
                task, agent="karo", model="sonnet"
            )
            task.cost_yen = AGENT_COST["karo"]
            return result

        else:
            # Strategic → 将軍(Opus)が最終決裁
            taisho_analysis = await self._call_taisho_analysis(task)
            task.context["taisho_analysis"] = taisho_analysis
            result = await self._call_cloud(
                task, agent="shogun", model="opus"
            )
            task.cost_yen = AGENT_COST["shogun"]
            return result

    # ─── Agent Dispatch ───

    async def _dispatch_to_agent(self, task: Task, agent: str) -> str:
        """Dispatch to a specific agent (force mode)."""
        if agent == "taisho":
            result = await self._call_taisho(task)
            task.cost_yen = 0
        elif agent == "karo":
            result = await self._call_cloud(task, "karo", "sonnet")
            task.cost_yen = AGENT_COST["karo"]
        elif agent == "shogun":
            result = await self._call_cloud(task, "shogun", "opus")
            task.cost_yen = AGENT_COST["shogun"]
        else:
            raise ValueError(f"Unknown agent: {agent}")
        return result

    # ─── Taisho (侍大将) ───

    async def _call_taisho(self, task: Task, company_mode: bool = False) -> str:
        """Call 侍大将 R1 (OpenVINO)."""
        self.stats["taisho_r1"] += 1
        task.assigned_agent = "taisho"
        task.status = TaskStatus.IN_PROGRESS
        self.queue.update_task(task)

        mode_label = "中隊" if company_mode else "大隊"
        logger.info("[侍大将] 推論開始 (%s)", mode_label)

        system = (
            "あなたは侍大将です。プロジェクトの設計・推論担当。\n"
            "<think>タグで日本語で深く推論し、論理的な結論を導いてください。\n"
        )
        if company_mode:
            system += (
                "中隊モードです。将軍・家老は不在。侍大将と足軽のみで完結してください。\n"
                "能力を超える場合は「大隊モード推奨」と報告してください。\n"
            )

        context_str = ""
        if task.context:
            context_str = "\n".join(
                f"[{k}]: {v}" for k, v in task.context.items()
            )
            context_str = f"\n## コンテキスト\n{context_str}\n"

        prompt = f"{context_str}\n## 任務\n{task.prompt}"

        try:
            result = await self.openvino.generate(
                prompt=prompt,
                system=system,
                max_tokens=2000,
                temperature=0.6,
            )
            logger.info("[侍大将] 完了 (¥0)")
            return result
        except Exception as e:
            logger.error("[侍大将] エラー: %s", e)
            raise

    async def _call_taisho_analysis(self, task: Task) -> str:
        """Call Taisho for analysis only (before escalation)."""
        self.stats["taisho_r1"] += 1
        logger.info("[侍大将] 分析開始（上位への報告用）")

        prompt = build_taisho_analysis_prompt(task)
        try:
            result = await self.openvino.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.6,
            )
            logger.info("[侍大将] 分析完了")
            return result
        except Exception as e:
            logger.warning("[侍大将] 分析失敗: %s (スキップ)", e)
            return f"(侍大将分析スキップ: {e})"

    # ─── Cloud Agents (家老/将軍) ───

    async def _call_cloud(
        self, task: Task, agent: str, model: str,
    ) -> str:
        """Call cloud agent (claude-cli → API fallback).

        Flow:
          1. Try claude-cli (Pro版)
          2. If rate limited → fallback to Anthropic API
        """
        task.assigned_agent = agent
        task.status = TaskStatus.IN_PROGRESS
        self.queue.update_task(task)

        agent_label = "将軍" if agent == "shogun" else "家老"
        logger.info("[%s] %s 実行開始 (model=%s)", agent_label, agent, model)

        # Build prompt with context
        parts = []
        if task.context.get("taisho_analysis"):
            parts.append(f"## 侍大将の分析\n{task.context['taisho_analysis']}")
        if task.context.get("escalation"):
            parts.append(task.context["escalation"])

        system = self._get_system_prompt(agent)
        full_prompt = "\n\n".join(parts + [f"## 任務\n{task.prompt}"])

        # Try claude-cli first (Pro版)
        result = await self.claude_cli.generate(
            prompt=full_prompt,
            model=model,
            system_prompt=system,
            cwd=self.repo_path,
        )

        if result.success:
            cost = AGENT_COST.get(agent, 0)
            self.stats["total_cost_yen"] += cost
            if agent == "karo":
                self.stats["karo_sonnet"] += 1
            else:
                self.stats["shogun_opus"] += 1
            logger.info(
                "[%s] 完了 (claude-cli, ¥%d)", agent_label, cost
            )
            return result.text

        # Rate limited → API fallback
        if result.rate_limited:
            logger.warning(
                "[%s] Pro版制限。APIフォールバック。", agent_label
            )
            return await self._call_cloud_api(task, agent, model, full_prompt, system)

        # Other error → try API fallback anyway
        logger.warning(
            "[%s] CLI error: %s → APIフォールバック", agent_label, result.error
        )
        return await self._call_cloud_api(task, agent, model, full_prompt, system)

    async def _call_cloud_api(
        self, task: Task, agent: str, model: str,
        prompt: str, system: str,
    ) -> str:
        """Fallback to Anthropic API (console.anthropic.com)."""
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

        cost = AGENT_COST.get(agent, 0)
        self.stats["total_cost_yen"] += cost
        if agent == "karo":
            self.stats["karo_sonnet"] += 1
        else:
            self.stats["shogun_opus"] += 1
        logger.info("[%s] API完了 (¥%d)", agent_label, cost)
        return text

    # ─── System Prompts ───

    def _get_system_prompt(self, agent: str) -> str:
        if agent == "shogun":
            return (
                "あなたはプロジェクトの将軍（Shogun）です。\n"
                "最高意思決定者として、プロジェクト全体の戦略的判断と最終決裁を行います。\n"
                "配下の家老・侍大将が解決できなかった難問がエスカレーションとして届きます。\n"
                "前任者の分析結果を踏まえつつ、根本的な解決策を提示してください。\n"
            )
        elif agent == "karo":
            return (
                "あなたはプロジェクトの家老（Karo）です。\n"
                "将軍の右腕として、作業の割振りと高度な実装方針の策定を行います。\n"
                "侍大将からの分析結果を受け、具体的なコード変更提案を行ってください。\n"
                "複雑な統合タスクを分解し、明確な実装手順を示してください。\n"
            )
        return ""

    # ─── Git Sync ───

    def _detect_repo_path(self) -> str:
        """Detect local repo path from git remote URL.

        Maps: github.com/{user}/{repo} → {local_base}/{repo}
        """
        base = self.repo_local_base
        # Try to read git remote from current directory
        try:
            import subprocess
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5,
                cwd=str(self.base_dir),
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Extract repo name from URL
                # https://github.com/user/repo.git → repo
                # git@github.com:user/repo.git → repo
                repo_name = url.rstrip("/").rstrip(".git").rsplit("/", 1)[-1]
                if repo_name:
                    path = f"{base}/{repo_name}"
                    logger.info("[本陣] リポジトリ検出: %s → %s", url, path)
                    return path
        except Exception:
            pass

        # Fallback: use base dir directly
        logger.info("[本陣] リポジトリパス: %s", base)
        return base

    async def _sync_repo(self) -> None:
        """Git sync (リポジトリ同期)."""
        sync_script = str(self.base_dir / "setup" / "auto_sync.sh")
        if not Path(sync_script).exists():
            # Fallback: direct git pull
            if Path(self.repo_path).exists():
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "git", "-C", self.repo_path, "pull", "--rebase",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(proc.communicate(), timeout=30)
                    logger.info("[本陣] Git同期完了")
                except Exception as e:
                    logger.warning("[本陣] Git同期スキップ: %s", e)
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", sync_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)
            logger.info("[本陣] Git同期完了")
        except Exception as e:
            logger.warning("[本陣] Git同期スキップ: %s", e)

    # ─── Status ───

    def get_status(self) -> dict:
        return {
            "mode": self.current_mode.value,
            "stats": dict(self.stats),
            "pending_tasks": len(self.queue.get_pending()),
            "total_tasks": len(self.queue.get_all_tasks()),
            "dashboard": self.dashboard.get_summary(),
            "mcp_servers": self.mcp.get_status(),
        }

    def show_stats(self) -> str:
        """Format stats for display."""
        s = self.stats
        lines = [
            "=" * 50,
            "📊 戦果統計",
            "=" * 50,
            f"大隊モード: {s['battalion']}回",
            f"中隊モード: {s['company']}回",
            "",
            "内訳:",
            f"  侍大将R1:  {s['taisho_r1']}回 (¥0)",
            f"  家老Sonnet: {s['karo_sonnet']}回 (¥{s['karo_sonnet'] * 280:,})",
            f"  将軍Opus:  {s['shogun_opus']}回 (¥{s['shogun_opus'] * 1350:,})",
            f"  APIフォールバック: {s['api_fallback']}回",
            "",
            f"合計コスト: ¥{s['total_cost_yen']:,}",
            "=" * 50,
        ]
        return "\n".join(lines)

    # ─── Lifecycle ───

    async def startup(self) -> None:
        """System startup."""
        logger.info("[本陣] 将軍システム起動...")
        self.queue.reset_all_workers()
        self.dashboard.init()
        self.queue.load_from_disk()
        logger.info("[本陣] 起動完了")

    async def shutdown(self) -> None:
        """System shutdown."""
        logger.info("[本陣] 将軍システム停止...")
        await self.mcp.stop_all()
        await self.openvino.close()
        if self._api_provider:
            await self._api_provider.close()
        logger.info("[本陣] 停止完了")
