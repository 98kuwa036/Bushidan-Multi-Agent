```python
"""
MCP Status Checker for Bushidan System
Handles heavy mode monitoring of MCP components.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Internal dependencies
from mcp_core import McpComponent
from system_health import HealthChecker
from logger_service import get_logger
from config_manager import ConfigManager


@dataclass
class MspStatus:
    """Represents the status of a single MCP component."""
    name: str
    is_healthy: bool
    last_updated: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None


class McpStatusChecker:
    """
    Main class for checking MCP statuses in heavy mode.
    
    This implementation optimizes performance by using parallel processing
    and batch operations to handle multiple MCP components efficiently.
    """

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.get_config('bushidan')
        self.logger = get_logger(__name__)
        self.health_checker = HealthChecker()
        self.max_workers = self.config.get('mcp_monitoring', {}).get('max_workers', 4)
        
    def check_all_mcp_components(self, mcp_list: List[McpComponent]) -> Dict[str, MspStatus]:
        """
        Check status of all MCP components in parallel.
        
        Args:
            mcp_list (List[McpComponent]): List of MCP components to check
            
        Returns:
            Dict[str, MspStatus]: Status information for each component
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_mcp = {
                executor.submit(self._check_single_component, mcp): mcp 
                for mcp in mcp_list
            }
            
            # Collect results
            for future in as_completed(future_to_mcp):
                try:
                    result = future.result(timeout=10)
                    if result:
                        results[result.name] = result
                except Exception as e:
                    mcp = future_to_mcp[future]
                    self.logger.error(
                        f"Error checking MCP component {mcp.name}: {str(e)}"
                    )
                    
        return results
    
    def _check_single_component(self, mcp: McpComponent) -> Optional[MspStatus]:
        """
        Check status of a single MCP component.
        
        Args:
            mcp (McpComponent): The MCP to check
            
        Returns:
            MspStatus: Status information for the component
        """
        try:
            # Perform health check
            is_healthy, metrics = self.health_checker.check_component_health(mcp)
            
            return MspStatus(
                name=mcp.name,
                is_healthy=is_healthy,
                last_updated=time.time(),
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check MCP {mcp.name}: {str(e)}")
            return MspStatus(
                name=mcp.name,
                is_healthy=False,
                last_updated=time.time(),
                metrics={},
                error_message=str(e)
            )

    def get_system_overview(self, mcp_list: List[McpComponent]) -> Dict[str, Any]:
        """
        Get a system-wide overview of MCP status.
        
        Args:
            mcp_list (List[McpComponent]): List of all MCP components
            
        Returns:
            Dict[str, Any]: System health summary
        """
        statuses = self.check_all_mcp_components(mcp_list)
        
        total_count = len(statuses)
        healthy_count = sum(1 for status in statuses.values() if status.is_healthy)
        
        return {
            "timestamp": time.time(),
            "total_components": total_count,
            "healthy_components": healthy_count,
            "unhealthy_components": total_count - healthy_count,
            "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0,
            "component_details": {
                name: {
                    "is_healthy": status.is_healthy,
                    "metrics": status.metrics
                } for name, status in statuses.items()
            }
        }
```