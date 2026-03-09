```python
"""
MCP Status Checker Module for Bushidan System v10.1

This module provides functionality to check the status of MCP services
within the Bushidan system, including network connectivity,
resource usage, and process health.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import psutil
import requests
import yaml

# Local imports
from .health_monitor import HealthMonitor

logger = logging.getLogger(__name__)

@dataclass
class MCPStatus:
    """Data class to represent the status of an MCP service"""
    service_name: str
    is_healthy: bool
    network_status: Dict[str, Any]
    resource_usage: Dict[str, Any]
    process_info: Dict[str, Any]
    last_updated: datetime

class MCPStatusChecker:
    """
    Main class for checking the status of MCP services in Bushidan system.
    
    Provides methods to monitor network connectivity,
    system resources, and process health of MCP components.
    """

    def __init__(self, config_path: str = "config/mcp_status_config.yaml"):
        """Initialize the MCP Status Checker with configuration."""
        self.config = self._load_config(config_path)
        self.health_monitor = HealthMonitor()
        logger.info("MCPStatusChecker initialized with config from %s", config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load config from %s: %s", config_path, str(e))
            raise

    async def check_mcp_status(self) -> Dict[str, MCPStatus]:
        """
        Check status of all configured MCP services.
        
        Returns:
            Dictionary mapping service names to their status objects
        """
        tasks = [
            self._check_single_service(service_name)
            for service_name in self.config.get('mcp_services', [])
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert results to MCPStatus objects
        status_dict = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Error checking service %s: %s", 
                           self.config['mcp_services'][i], str(result))
                continue
            
            status_dict[result.service_name] = result
        
        return status_dict

    async def _check_single_service(self, service_name: str) -> MCPStatus:
        """Check status of a single MCP service."""
        try:
            # Get service configuration
            service_config = self.config.get('services', {}).get(service_name, {})
            
            # Check network connectivity
            network_status = await self._check_network_connectivity(
                service_config.get('endpoint')
            )
            
            # Check resource usage
            resource_usage = self._check_resource_usage()
            
            # Check process health
            process_info = self._check_process_health(service_name)
            
            is_healthy = all([
                network_status['reachable'],
                process_info['running']
            ])
            
            return MCPStatus(
                service_name=service_name,
                is_healthy=is_healthy,
                network_status=network_status,
                resource_usage=resource_usage,
                process_info=process_info,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error("Error checking status for %s: %s", service_name, str(e))
            raise

    async def _check_network_connectivity(self, endpoint: str) -> Dict[str, Any]:
        """Check network connectivity to MCP endpoint."""
        try:
            response = requests.get(
                endpoint,
                timeout=self.config.get('network_timeout', 5),
                verify=False
            )
            
            return {
                'reachable': True,
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
        except requests.exceptions.RequestException as e:
            logger.warning("Network connectivity check failed for %s: %s", endpoint, str(e))
            return {
                'reachable': False,
                'error': str(e),
                'status_code': None,
                'response_time_ms': None
            }

    def _check_resource_usage(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_total_gb': round(memory_info.total / (1024**3), 2),
                'memory_used_gb': round(memory_info.used / (1024**3), 2),
                'memory_percent': memory_info.percent
            }
        except Exception as e:
            logger.error("Error checking resource usage: %s", str(e))
            return {}

    def _check_process_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of MCP process."""
        try:
            # Simple process check - could be extended with more sophisticated logic
            processes = []
            
            for proc in psutil.process_iter(['pid', 'name']):
                if service_name.lower() in proc.info['name'].lower():
                    processes.append(proc)
            
            return {
                'running': len(processes) > 0,
                'process_count': len(processes),
                'process_list': [p.info for p in processes]
            }
        except Exception as e:
            logger.error("Error checking process health for %s: %s", service_name, str(e))
            return {'running': False, 'error': str(e)}

    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of system status."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            return {
                'timestamp': datetime.now(),
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory_info.percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.error("Error getting system summary: %s", str(e))
            return {}
```