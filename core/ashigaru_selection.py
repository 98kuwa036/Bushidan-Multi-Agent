"""Dynamic Ashigaru Selection System

Intelligent selection of the most suitable 足軽 (ashigaru) agents for each task.

Features:
  - Dynamic selection based on task requirements
  - Maximum 4 concurrent ashigaru limit
  - Load balancing and performance monitoring
  - Priority-based assignment
  - Automatic failover and replacement

This system ensures optimal resource utilization while maintaining the traditional
shogun hierarchy structure.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json


logger = logging.getLogger("shogun.ashigaru_selection")


class TaskComplexity(Enum):
    """Task complexity levels for ashigaru selection."""
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"
    CRITICAL = "critical"


class AshigaruStatus(Enum):
    """Ashigaru operational status."""
    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AshigaruAgent:
    """Represents a single ashigaru agent."""
    id: int
    name: str
    role: str
    capabilities: List[str]
    current_load: float = 0.0
    status: AshigaruStatus = AshigaruStatus.AVAILABLE
    last_used: Optional[datetime] = None
    success_rate: float = 1.0
    avg_response_time: float = 0.0
    memory_usage_mb: int = 0
    current_task: Optional[str] = None


@dataclass
class TaskRequirement:
    """Task requirements for ashigaru selection."""
    task_id: str
    complexity: TaskComplexity
    required_capabilities: List[str]
    preferred_agents: List[str] = None
    max_duration: Optional[timedelta] = None
    priority: int = 5  # 1-10, higher is more urgent


class AshigaruSelectionManager:
    """Manages dynamic selection of ashigaru agents."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_active = config.get("max_active", 4)
        self.total_count = config.get("count", 8)
        
        # Initialize ashigaru agents
        self.agents: Dict[int, AshigaruAgent] = {}
        self.active_tasks: Dict[str, Set[int]] = {}  # task_id -> agent_ids
        self.task_queue: List[TaskRequirement] = []
        
        # Statistics
        self.stats = {
            "selections_made": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_selection_time": 0.0,
            "load_balanced_count": 0,
            "failover_count": 0,
        }
        
        self._initialize_agents()
        
    def _initialize_agents(self) -> None:
        """Initialize ashigaru agents from configuration."""
        servers = self.config.get("servers", [])
        
        for server_config in servers:
            agent_id = server_config.get("id")
            if agent_id is None:
                continue
                
            # Map server roles to capabilities
            capabilities = self._map_role_to_capabilities(server_config.get("role", ""))
            
            agent = AshigaruAgent(
                id=agent_id,
                name=server_config.get("name", f"ashigaru_{agent_id}"),
                role=server_config.get("role", "汎用"),
                capabilities=capabilities,
            )
            
            self.agents[agent_id] = agent
            
        logger.info("[足軽選抜] 初期化完了: %d名の足軽", len(self.agents))
    
    def _map_role_to_capabilities(self, role: str) -> List[str]:
        """Map server role to specific capabilities."""
        capability_map = {
            "ファイル操作": ["file_read", "file_write", "file_management"],
            "Git/GitHub操作": ["git", "github", "version_control"],
            "Web情報取得": ["web_fetch", "http_requests", "api_calls"],
            "長期記憶": ["memory", "storage", "persistence"],
            "データベース": ["database", "sql", "data_management"],
            "ブラウザ自動化": ["browser", "automation", "web_scraping"],
            "Web検索": ["search", "information_retrieval"],
            "チーム連携": ["communication", "notifications", "slack"],
        }
        
        return capability_map.get(role, ["general"])
    
    async def select_ashigaru(
        self, 
        task_req: TaskRequirement,
        exclude_agents: Optional[Set[int]] = None
    ) -> List[int]:
        """Select optimal ashigaru agents for a task."""
        start_time = datetime.now()
        
        try:
            # Check if we're at capacity
            current_active = self._count_active_agents()
            if current_active >= self.max_active:
                logger.warning("[足軽選抜] 最大同時実行数に到達 (%d/%d)", current_active, self.max_active)
                # Queue the task if at capacity
                self.task_queue.append(task_req)
                return []
            
            # Filter available agents
            available_agents = self._get_available_agents(exclude_agents)
            if not available_agents:
                logger.warning("[足軽選抜] 利用可能な足軽なし")
                return []
            
            # Score and rank agents
            scored_agents = self._score_agents(available_agents, task_req)
            
            # Select optimal number based on task complexity
            optimal_count = self._get_optimal_agent_count(task_req.complexity, current_active)
            selected_count = min(optimal_count, len(scored_agents))
            
            # Select top agents
            selected_agents = [agent_id for agent_id, score in scored_agents[:selected_count]]
            
            # Mark agents as busy
            for agent_id in selected_agents:
                self.agents[agent_id].status = AshigaruStatus.BUSY
                self.agents[agent_id].current_task = task_req.task_id
                self.agents[agent_id].last_used = datetime.now()
            
            # Track task assignment
            self.active_tasks[task_req.task_id] = set(selected_agents)
            
            # Update statistics
            selection_time = (datetime.now() - start_time).total_seconds()
            self.stats["selections_made"] += 1
            self.stats["avg_selection_time"] = (
                self.stats["avg_selection_time"] * (self.stats["selections_made"] - 1) + selection_time
            ) / self.stats["selections_made"]
            
            logger.info("[足軽選抜] 選抜完了: %s -> %s (%.2fs)", 
                       task_req.task_id, [self.agents[aid].name for aid in selected_agents], selection_time)
            
            return selected_agents
            
        except Exception as e:
            logger.error("[足軽選抜] 選抜エラー: %s", e)
            return []
    
    def _get_available_agents(self, exclude_agents: Optional[Set[int]] = None) -> List[int]:
        """Get list of available agent IDs."""
        available = []
        exclude = exclude_agents or set()
        
        for agent_id, agent in self.agents.items():
            if (agent_id not in exclude and 
                agent.status == AshigaruStatus.AVAILABLE and
                agent.current_load < 0.9):  # Don't overload agents
                available.append(agent_id)
        
        return available
    
    def _score_agents(
        self, 
        agent_ids: List[int], 
        task_req: TaskRequirement
    ) -> List[Tuple[int, float]]:
        """Score agents based on task requirements."""
        scored = []
        
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            score = 0.0
            
            # Capability match score (40%)
            capability_score = self._calculate_capability_score(agent, task_req.required_capabilities)
            score += capability_score * 0.4
            
            # Performance score (30%)
            performance_score = agent.success_rate * (1.0 - agent.current_load)
            score += performance_score * 0.3
            
            # Availability score (20%)
            # Prefer agents that haven't been used recently
            recency_score = 1.0
            if agent.last_used:
                hours_since = (datetime.now() - agent.last_used).total_seconds() / 3600
                recency_score = min(1.0, hours_since / 24.0)  # Full score after 24h
            score += recency_score * 0.2
            
            # Priority boost (10%)
            if task_req.preferred_agents and agent.name in task_req.preferred_agents:
                score += 0.1
            
            scored.append((agent_id, score))
        
        # Sort by score (descending)
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def _calculate_capability_score(self, agent: AshigaruAgent, required_caps: List[str]) -> float:
        """Calculate how well agent capabilities match requirements."""
        if not required_caps:
            return 1.0  # No specific requirements
        
        agent_caps = set(agent.capabilities)
        required_caps_set = set(required_caps)
        
        # Calculate intersection over union
        intersection = len(agent_caps.intersection(required_caps_set))
        union = len(agent_caps.union(required_caps_set))
        
        return intersection / max(union, 1)
    
    def _get_optimal_agent_count(self, complexity: TaskComplexity, current_active: int) -> int:
        """Determine optimal number of agents for task complexity."""
        remaining_slots = self.max_active - current_active
        
        optimal_counts = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 2,
            TaskComplexity.COMPLEX: 3,
            TaskComplexity.CRITICAL: 4,
        }
        
        return min(optimal_counts.get(complexity, 1), remaining_slots)
    
    def _count_active_agents(self) -> int:
        """Count currently active (busy) agents."""
        return sum(1 for agent in self.agents.values() if agent.status == AshigaruStatus.BUSY)
    
    async def complete_task(
        self, 
        task_id: str, 
        success: bool = True,
        performance_metrics: Optional[Dict] = None
    ) -> None:
        """Mark task as complete and free up agents."""
        if task_id not in self.active_tasks:
            logger.warning("[足軽選抜] 不明なタスク完了: %s", task_id)
            return
        
        agent_ids = self.active_tasks[task_id]
        
        for agent_id in agent_ids:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.status = AshigaruStatus.AVAILABLE
                agent.current_task = None
                agent.current_load = 0.0
                
                # Update performance metrics
                if performance_metrics and agent_id in performance_metrics:
                    metrics = performance_metrics[agent_id]
                    
                    # Update success rate (exponential moving average)
                    task_success = 1.0 if success else 0.0
                    agent.success_rate = agent.success_rate * 0.9 + task_success * 0.1
                    
                    # Update response time
                    if "response_time" in metrics:
                        agent.avg_response_time = (
                            agent.avg_response_time * 0.8 + metrics["response_time"] * 0.2
                        )
                    
                    # Update memory usage
                    if "memory_usage" in metrics:
                        agent.memory_usage_mb = metrics["memory_usage"]
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        # Update statistics
        if success:
            self.stats["tasks_completed"] += 1
        else:
            self.stats["tasks_failed"] += 1
        
        logger.info("[足軽選抜] タスク完了: %s (%s) - 足軽解放: %d名", 
                   task_id, "成功" if success else "失敗", len(agent_ids))
        
        # Process queued tasks
        await self._process_task_queue()
    
    async def handle_agent_failure(self, agent_id: int, task_id: str) -> List[int]:
        """Handle agent failure and select replacement if needed."""
        if agent_id not in self.agents:
            return []
        
        agent = self.agents[agent_id]
        agent.status = AshigaruStatus.ERROR
        agent.current_task = None
        agent.current_load = 0.0
        agent.success_rate *= 0.8  # Reduce success rate
        
        logger.warning("[足軽選抜] 足軽障害: %s (タスク: %s)", agent.name, task_id)
        
        # Update active task assignment
        if task_id in self.active_tasks:
            self.active_tasks[task_id].discard(agent_id)
            
            # Select replacement if task is still active
            if self.active_tasks[task_id]:  # Other agents still working
                # Find task requirement (would need to be stored)
                # For now, just return empty list
                self.stats["failover_count"] += 1
                
        return []
    
    async def _process_task_queue(self) -> None:
        """Process queued tasks if agents become available."""
        if not self.task_queue:
            return
        
        current_active = self._count_active_agents()
        if current_active >= self.max_active:
            return
        
        # Try to process queued tasks
        processed = []
        for i, task_req in enumerate(self.task_queue):
            selected = await self.select_ashigaru(task_req)
            if selected:
                processed.append(i)
                logger.info("[足軽選抜] キューからタスク実行: %s", task_req.task_id)
        
        # Remove processed tasks from queue (reverse order to maintain indices)
        for i in reversed(processed):
            del self.task_queue[i]
    
    def rebalance_load(self) -> None:
        """Rebalance load across available agents."""
        # Simple rebalancing: reset overloaded agents to maintenance
        for agent in self.agents.values():
            if agent.current_load > 0.95 and agent.status == AshigaruStatus.AVAILABLE:
                agent.status = AshigaruStatus.MAINTENANCE
                logger.info("[足軽選抜] 負荷軽減のため一時休止: %s", agent.name)
            elif agent.current_load < 0.1 and agent.status == AshigaruStatus.MAINTENANCE:
                agent.status = AshigaruStatus.AVAILABLE
                logger.info("[足軽選抜] 休止解除: %s", agent.name)
        
        self.stats["load_balanced_count"] += 1
    
    def get_agent_status(self, agent_id: int) -> Optional[Dict]:
        """Get detailed status of specific agent."""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            "id": agent.id,
            "name": agent.name,
            "role": agent.role,
            "status": agent.status.value,
            "current_load": agent.current_load,
            "success_rate": agent.success_rate,
            "avg_response_time": agent.avg_response_time,
            "memory_usage_mb": agent.memory_usage_mb,
            "current_task": agent.current_task,
            "capabilities": agent.capabilities,
            "last_used": agent.last_used.isoformat() if agent.last_used else None,
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        status_counts = {}
        total_load = 0.0
        
        for agent in self.agents.values():
            status = agent.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_load += agent.current_load
        
        return {
            "total_agents": len(self.agents),
            "max_active": self.max_active,
            "currently_active": self._count_active_agents(),
            "queued_tasks": len(self.task_queue),
            "status_breakdown": status_counts,
            "average_load": total_load / len(self.agents) if self.agents else 0.0,
            "stats": dict(self.stats),
        }
    
    def show_status(self) -> str:
        """Format system status for display."""
        status = self.get_system_status()
        
        lines = [
            "=" * 60,
            "⚔️  足軽選抜システム 状況報告",
            "=" * 60,
            f"総足軽数: {status['total_agents']}名",
            f"最大同時実行: {status['max_active']}名",
            f"現在活動中: {status['currently_active']}名",
            f"待機中タスク: {status['queued_tasks']}件",
            f"平均負荷: {status['average_load']:.1%}",
            "",
            "状態別内訳:",
        ]
        
        for status_name, count in status["status_breakdown"].items():
            emoji = {"available": "✅", "busy": "🔥", "maintenance": "🔧", "error": "❌", "offline": "⚫"}.get(status_name, "❓")
            lines.append(f"  {emoji} {status_name}: {count}名")
        
        lines.extend([
            "",
            "統計情報:",
            f"  選抜実行: {self.stats['selections_made']}回",
            f"  タスク完了: {self.stats['tasks_completed']}件",
            f"  タスク失敗: {self.stats['tasks_failed']}件",
            f"  平均選抜時間: {self.stats['avg_selection_time']:.3f}秒",
            f"  負荷分散実行: {self.stats['load_balanced_count']}回",
            f"  障害時切り替え: {self.stats['failover_count']}回",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# Utility functions
def create_task_requirement(
    task_id: str,
    task_description: str,
    complexity: TaskComplexity = TaskComplexity.MODERATE,
    required_capabilities: List[str] = None
) -> TaskRequirement:
    """Create a task requirement from description."""
    # Simple capability inference from description
    capabilities = required_capabilities or []
    
    if not capabilities:
        # Infer capabilities from task description
        description_lower = task_description.lower()
        
        if any(word in description_lower for word in ["ファイル", "file", "読み", "書き"]):
            capabilities.append("file_management")
        if any(word in description_lower for word in ["git", "github", "commit"]):
            capabilities.append("git")
        if any(word in description_lower for word in ["web", "http", "api", "fetch"]):
            capabilities.append("web_fetch")
        if any(word in description_lower for word in ["検索", "search"]):
            capabilities.append("search")
        if any(word in description_lower for word in ["データベース", "database", "sql"]):
            capabilities.append("database")
        
        # Default to general if nothing specific found
        if not capabilities:
            capabilities = ["general"]
    
    return TaskRequirement(
        task_id=task_id,
        complexity=complexity,
        required_capabilities=capabilities,
    )