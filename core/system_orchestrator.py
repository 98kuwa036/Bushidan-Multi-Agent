"""
将軍システム v8.0 - System Orchestrator
システム統合: 全コンポーネントの統合・調整・制御

Features:
- Complete v8.0 system integration
- Agent hierarchy coordination (Shogun/Karo/Taisho)
- Resource management and optimization
- Health monitoring and auto-recovery
- Performance analytics and reporting
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from pathlib import Path

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.knowledge_base import KnowledgeBase, create_knowledge_base
from core.activity_memory import ActivityMemory, create_activity_memory, TaskComplexity
from core.sandbox_executor import SandboxExecutor, create_sandbox_executor
from ashigaru.ollama_web_search import OllamaWebSearch, create_ollama_web_search
from agents.taisho import TaishoAgent, TaishoTask, TaskPriority, create_taisho_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System deployment modes"""
    BATTALION = "battalion"  # Full system (将軍 + 家老 + 侍大将 + 足軽 × 10)
    COMPANY = "company"      # Economic mode (侍大将 + 足軽 × 10, no API)
    PLATOON = "platoon"      # Minimal mode (侍大将 + selected 足軽)


class SystemStatus(Enum):
    """Overall system status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_processing_time_seconds: float = 0.0
    knowledge_base_entries: int = 0
    activity_memory_records: int = 0
    sandbox_executions: int = 0
    web_searches: int = 0
    system_uptime_hours: float = 0.0
    memory_usage_percent: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return (self.successful_tasks / self.total_tasks_processed * 100) if self.total_tasks_processed > 0 else 0.0


class SystemOrchestrator:
    """
    将軍システム v8.0 System Orchestrator
    
    Provides centralized coordination and management of all v8.0 components:
    - System initialization and configuration
    - Agent lifecycle management
    - Resource optimization and monitoring
    - Health checks and auto-recovery
    - Performance analytics and reporting
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.start_time = datetime.utcnow()
        
        # Core v8.0 systems
        self.knowledge_base: Optional[KnowledgeBase] = None
        self.activity_memory: Optional[ActivityMemory] = None
        self.sandbox_executor: Optional[SandboxExecutor] = None
        self.web_search: Optional[OllamaWebSearch] = None
        self.taisho_agent: Optional[TaishoAgent] = None
        
        # System state
        self.current_mode = SystemMode.BATTALION
        self.status = SystemStatus.HEALTHY
        self.metrics = SystemMetrics()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        logger.info(f"SystemOrchestrator v8.0 initialized with config: {config_path}")
    
    async def initialize(self) -> bool:
        """Initialize the complete v8.0 system"""
        try:
            logger.info("🏯 Initializing Shogun System v8.0...")
            
            # Load configuration
            if not await self._load_configuration():
                logger.error("Failed to load configuration")
                return False
            
            # Initialize core systems in dependency order
            success_steps = []
            
            # 1. Knowledge Base (RAG)
            if self.config.get("knowledge_base", {}).get("enabled", True):
                if await self._initialize_knowledge_base():
                    success_steps.append("knowledge_base")
                    logger.info("✅ Knowledge Base initialized")
                else:
                    logger.error("❌ Knowledge Base initialization failed")
            
            # 2. Activity Memory (陣中日記)
            if self.config.get("activity_memory", {}).get("enabled", True):
                if await self._initialize_activity_memory():
                    success_steps.append("activity_memory")
                    logger.info("✅ Activity Memory initialized")
                else:
                    logger.error("❌ Activity Memory initialization failed")
            
            # 3. Sandbox Executor (演習場)
            if self.config.get("sandbox", {}).get("enabled", True):
                if await self._initialize_sandbox_executor():
                    success_steps.append("sandbox_executor")
                    logger.info("✅ Sandbox Executor initialized")
                else:
                    logger.warning("⚠️  Sandbox Executor initialization failed (non-critical)")
            
            # 4. Web Search (10番足軽)
            if self.config.get("ollama_web_search", {}).get("enabled", True):
                if await self._initialize_web_search():
                    success_steps.append("web_search")
                    logger.info("✅ Web Search initialized")
                else:
                    logger.warning("⚠️  Web Search initialization failed (non-critical)")
            
            # 5. Taisho Agent (侍大将)
            if await self._initialize_taisho_agent():
                success_steps.append("taisho_agent")
                logger.info("✅ Taisho Agent initialized")
            else:
                logger.error("❌ Taisho Agent initialization failed")
                return False
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Determine system status
            critical_systems = ["taisho_agent"]  # Minimum required
            if all(system in success_steps for system in critical_systems):
                self.status = SystemStatus.HEALTHY
                logger.info("🎌 Shogun System v8.0 initialization complete - Status: HEALTHY")
                return True
            else:
                self.status = SystemStatus.DEGRADED
                logger.warning("⚠️  Shogun System v8.0 partially initialized - Status: DEGRADED")
                return True
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.status = SystemStatus.CRITICAL
            return False
    
    async def _load_configuration(self) -> bool:
        """Load system configuration"""
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded: v{self.config.get('system', {}).get('version', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    async def _initialize_knowledge_base(self) -> bool:
        """Initialize Knowledge Base (RAG)"""
        try:
            kb_config = self.config.get("knowledge_base", {})
            self.knowledge_base = create_knowledge_base(kb_config)
            
            # Test connectivity
            health = await self.knowledge_base.health_check()
            return health.get("status") in ["healthy", "ok"]
            
        except Exception as e:
            logger.error(f"Knowledge Base initialization failed: {e}")
            return False
    
    async def _initialize_activity_memory(self) -> bool:
        """Initialize Activity Memory (陣中日記)"""
        try:
            am_config = self.config.get("activity_memory", {})
            self.activity_memory = create_activity_memory(am_config)
            
            # Test database
            stats = await self.activity_memory.get_statistics()
            return isinstance(stats, dict) and "error" not in stats
            
        except Exception as e:
            logger.error(f"Activity Memory initialization failed: {e}")
            return False
    
    async def _initialize_sandbox_executor(self) -> bool:
        """Initialize Sandbox Executor (演習場)"""
        try:
            sandbox_config = self.config.get("sandbox", {})
            self.sandbox_executor = create_sandbox_executor(sandbox_config)
            
            # Optional: Test if container management works
            # For now, assume success if creation worked
            return True
            
        except Exception as e:
            logger.error(f"Sandbox Executor initialization failed: {e}")
            return False
    
    async def _initialize_web_search(self) -> bool:
        """Initialize Web Search (10番足軽)"""
        try:
            search_config = self.config.get("ollama_web_search", {})
            self.web_search = create_ollama_web_search(search_config, self.knowledge_base)
            
            # Test with simple health check
            health = await self.web_search.health_check()
            return health.get("status") in ["healthy", "partially_healthy"]
            
        except Exception as e:
            logger.error(f"Web Search initialization failed: {e}")
            return False
    
    async def _initialize_taisho_agent(self) -> bool:
        """Initialize Taisho Agent (侍大将)"""
        try:
            taisho_config = self.config.get("taisho", {})
            self.taisho_agent = create_taisho_agent(
                config=taisho_config,
                knowledge_base=self.knowledge_base,
                activity_memory=self.activity_memory,
                sandbox_executor=self.sandbox_executor,
                web_search=self.web_search
            )
            
            # Test agent responsiveness
            health = await self.taisho_agent.health_check()
            return health.get("status") == "healthy" or health.get("r1_accessible", False)
            
        except Exception as e:
            logger.error(f"Taisho Agent initialization failed: {e}")
            return False
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        try:
            # Cleanup task (runs every 6 hours)
            cleanup_task = asyncio.create_task(self._cleanup_task())
            self.background_tasks.append(cleanup_task)
            
            # Health monitoring (runs every 5 minutes)
            health_task = asyncio.create_task(self._health_monitoring_task())
            self.background_tasks.append(health_task)
            
            # Metrics collection (runs every hour)
            metrics_task = asyncio.create_task(self._metrics_collection_task())
            self.background_tasks.append(metrics_task)
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
    
    async def process_task(
        self,
        description: str,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        priority: TaskPriority = TaskPriority.MEDIUM,
        context: Optional[str] = None,
        requires_execution: bool = False,
        requires_latest_info: bool = False,
        requested_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Process a task through the v8.0 system
        
        Returns:
            Dict with task results and metadata
        """
        try:
            if not self.taisho_agent:
                return {
                    "success": False,
                    "error": "Taisho agent not available",
                    "status": self.status.value
                }
            
            # Create task
            task = TaishoTask(
                id="",  # Will be auto-generated
                description=description,
                complexity=complexity,
                priority=priority,
                context=context,
                requires_execution=requires_execution,
                requires_latest_info=requires_latest_info,
                requested_by=requested_by
            )
            
            # Process through Taisho
            response = await self.taisho_agent.process_task(task)
            
            # Update metrics
            self.metrics.total_tasks_processed += 1
            if response.success:
                self.metrics.successful_tasks += 1
            else:
                self.metrics.failed_tasks += 1
            
            # Update average processing time
            total_time = (self.metrics.avg_processing_time_seconds * (self.metrics.total_tasks_processed - 1) + 
                         response.processing_time_seconds)
            self.metrics.avg_processing_time_seconds = total_time / self.metrics.total_tasks_processed
            
            # Return results
            return {
                "success": response.success,
                "response": response.response,
                "reasoning": response.reasoning,
                "confidence": response.confidence.value,
                "processing_time_seconds": response.processing_time_seconds,
                "tools_used": response.tools_used,
                "similar_tasks_found": response.similar_tasks_found,
                "knowledge_retrieved": response.knowledge_retrieved,
                "code_executed": response.code_executed,
                "execution_successful": response.execution_successful,
                "task_id": response.task_id,
                "system_status": self.status.value
            }
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            self.metrics.total_tasks_processed += 1
            self.metrics.failed_tasks += 1
            
            return {
                "success": False,
                "error": str(e),
                "system_status": self.status.value
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Component health checks
            component_health = {}
            
            if self.knowledge_base:
                kb_health = await self.knowledge_base.health_check()
                component_health["knowledge_base"] = kb_health.get("status", "unknown")
            
            if self.activity_memory:
                am_stats = await self.activity_memory.get_statistics()
                component_health["activity_memory"] = "healthy" if am_stats else "unhealthy"
            
            if self.sandbox_executor:
                sb_health = await self.sandbox_executor.health_check()
                component_health["sandbox_executor"] = sb_health.get("status", "unknown")
            
            if self.web_search:
                ws_health = await self.web_search.health_check()
                component_health["web_search"] = ws_health.get("status", "unknown")
            
            if self.taisho_agent:
                ta_health = await self.taisho_agent.health_check()
                component_health["taisho_agent"] = ta_health.get("status", "unknown")
            
            # System uptime
            uptime = datetime.utcnow() - self.start_time
            self.metrics.system_uptime_hours = uptime.total_seconds() / 3600
            
            # Collect latest statistics
            await self._update_component_metrics()
            
            return {
                "system_status": self.status.value,
                "system_mode": self.current_mode.value,
                "version": self.config.get("system", {}).get("version", "8.0"),
                "uptime_hours": round(self.metrics.system_uptime_hours, 2),
                "component_health": component_health,
                "performance_metrics": asdict(self.metrics),
                "background_tasks_active": len([t for t in self.background_tasks if not t.done()]),
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "system_status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    async def _update_component_metrics(self):
        """Update metrics from components"""
        try:
            if self.knowledge_base:
                kb_stats = await self.knowledge_base.get_stats()
                self.metrics.knowledge_base_entries = kb_stats.get("total_entries", 0)
            
            if self.activity_memory:
                am_stats = await self.activity_memory.get_statistics()
                self.metrics.activity_memory_records = am_stats.get("total_records", 0)
            
            if self.sandbox_executor:
                sb_stats = await self.sandbox_executor.get_statistics()
                self.metrics.sandbox_executions = sb_stats.get("total_executions", 0)
            
            if self.web_search:
                ws_stats = self.web_search.get_usage_statistics()
                self.metrics.web_searches = ws_stats.get("total_searches", 0)
                
        except Exception as e:
            logger.warning(f"Failed to update component metrics: {e}")
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while not self.shutdown_event.is_set():
            try:
                logger.info("Running system cleanup...")
                
                # Knowledge base cleanup (30 days)
                if self.knowledge_base:
                    deleted = await self.knowledge_base.cleanup_old_entries()
                    logger.info(f"Cleaned up {deleted} old knowledge entries")
                
                # Activity memory cleanup (90 days)
                if self.activity_memory:
                    deleted = await self.activity_memory.cleanup_old_records()
                    logger.info(f"Cleaned up {deleted} old activity records")
                
                logger.info("System cleanup completed")
                
                # Wait 6 hours before next cleanup
                await asyncio.sleep(6 * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour on error
    
    async def _health_monitoring_task(self):
        """Background health monitoring task"""
        consecutive_failures = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Check critical components
                critical_healthy = True
                
                if self.taisho_agent:
                    health = await self.taisho_agent.health_check()
                    if health.get("status") != "healthy":
                        critical_healthy = False
                
                # Update system status
                if critical_healthy:
                    if consecutive_failures > 0:
                        logger.info("System recovered - Status: HEALTHY")
                    self.status = SystemStatus.HEALTHY
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        self.status = SystemStatus.CRITICAL
                        logger.error("System critical - multiple health check failures")
                    else:
                        self.status = SystemStatus.DEGRADED
                        logger.warning(f"System degraded - health check failure {consecutive_failures}/3")
                
                # Wait 5 minutes before next check
                await asyncio.sleep(5 * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute on error
    
    async def _metrics_collection_task(self):
        """Background metrics collection task"""
        while not self.shutdown_event.is_set():
            try:
                await self._update_component_metrics()
                logger.debug("Metrics updated")
                
                # Wait 1 hour before next collection
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes on error
    
    async def shutdown(self):
        """Graceful system shutdown"""
        try:
            logger.info("🏯 Shutting down Shogun System v8.0...")
            
            # Signal background tasks to stop
            self.shutdown_event.set()
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Shutdown components
            if self.sandbox_executor:
                await self.sandbox_executor.shutdown()
            
            self.status = SystemStatus.CRITICAL
            logger.info("🎌 Shogun System v8.0 shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# Factory function
async def create_system_orchestrator(config_path: str = "config/settings.yaml") -> SystemOrchestrator:
    """Create and initialize SystemOrchestrator"""
    orchestrator = SystemOrchestrator(config_path)
    
    success = await orchestrator.initialize()
    if not success:
        logger.error("Failed to initialize system orchestrator")
        return None
    
    return orchestrator


# Example usage
if __name__ == "__main__":
    async def test_system_orchestrator():
        """Test the system orchestrator"""
        # Initialize system
        orchestrator = SystemOrchestrator(config_path="../config/settings.yaml")
        
        if await orchestrator.initialize():
            logger.info("System initialized successfully")
            
            # Test task processing
            result = await orchestrator.process_task(
                description="I2S設定の最適化について教えてください",
                complexity=TaskComplexity.MEDIUM,
                requires_latest_info=True,
                requested_by="test"
            )
            
            print(f"Task result: {result}")
            
            # Get system status
            status = await orchestrator.get_system_status()
            print(f"System status: {status}")
            
            # Shutdown
            await orchestrator.shutdown()
        else:
            logger.error("System initialization failed")
    
    # Run test
    asyncio.run(test_system_orchestrator())