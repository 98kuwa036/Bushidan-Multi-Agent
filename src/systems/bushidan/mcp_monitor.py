```python
"""
MCP Monitoring Service Implementation.
Handles continuous monitoring of MCP components with heavy mode optimizations.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

# Internal dependencies
from mcp_core import McpComponent
from logger_service import get_logger
from config_manager import ConfigManager
from .mcp_status_checker import McpStatusChecker


@dataclass
class MonitoringConfig:
    """Configuration for monitoring service."""
    check_interval: float = 5.0      # seconds between checks
    max_history_size: int = 100     # maximum number of history entries
    alert_threshold: float = 0.8    # percentage threshold to trigger alerts


class MspMonitor:
    """
    Continuous monitoring service for MCP components.
    
    This class provides heavy mode optimized monitoring with:
    - Parallel checking using thread pools
    - History tracking of status changes
    - Alerting when system health drops below threshold
    """

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.get_config('bushidan')
        self.logger = get_logger(__name__)
        self.status_checker = McpStatusChecker(config_manager)
        
        # Configuration from config file
        monitor_config = self.config.get('mcp_monitoring', {})
        self.check_interval = float(monitor_config.get('check_interval', 5.0))
        self.max_history_size = int(monitor_config.get('max_history_size', 100))
        self.alert_threshold = float(monitor_config.get('alert_threshold', 0.8))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.mcp_components: List[McpComponent] = []
        self.status_history = deque(maxlen=self.max_history_size)
        self.last_overview: Dict[str, any] = {}
        
    def start_monitoring(self, mcp_list: List[McpComponent]):
        """
        Start continuous monitoring of MCP components.
        
        Args:
            mcp_list (List[McpComponent]): List of MCPs to monitor
        """
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
            
        self.mcp_components = mcp_list
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info(f"Started MCP monitoring with {len(mcp_list)} components")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
            
        self.logger.info("Stopped MCP monitoring")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get system overview
                overview = self.status_checker.get_system_overview(self.mcp_components)
                
                # Store in history
                self.status_history.append(overview)
                self.last_overview = overview
                
                # Check for alerts
                health_percentage = overview['health_percentage']
                if health_percentage < (self.alert_threshold * 100):
                    self._trigger_alert(health_percentage, overview)
                    
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                # Continue monitoring even if one cycle fails
                time.sleep(self.check_interval)
    
    def _trigger_alert(self, health_percentage: float, overview: Dict[str, any]):
        """
        Trigger alert when system health drops below threshold.
        
        Args:
            health_percentage (float): Current health percentage
            overview (Dict[str, any]): System overview data
        """
        self.logger.warning(
            f"System health critical: {health_percentage:.1f}% "
            f"(threshold: {self.alert_threshold*100}%)"
        )
        
        # Log unhealthy components
        unhealthy = [
            name for name, details in overview['component_details'].items()
            if not details['is_healthy']
        ]
        
        self.logger.warning(f"Unhealthy components: {unhealthy}")
    
    def get_current_status(self) -> Optional[Dict[str, any]]:
        """
        Get the most recent status overview.
        
        Returns:
            Dict[str, any]: Current system status or None if not monitoring
        """
        return self.last_overview
    
    def get_history(self, limit: int = 10) -> List[Dict[str, any]]:
        """
        Get historical status records.
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            List[Dict[str, any]]: Recent status history
        """
        # Convert deque to list in reverse order for most recent first
        history_list = list(self.status_history)
        return history_list[-limit:] if len(history_list) > limit else history_list

    def add_component(self, mcp: McpComponent):
        """Add a new MCP component to monitoring."""
        self.mcp_components.append(mcp)
        
    def remove_component(self, mcp_name: str):
        """Remove an MCP component from monitoring."""
        self.mcp_components = [mcp for mcp in self.mcp_components if mcp.name != mcp_name]
```