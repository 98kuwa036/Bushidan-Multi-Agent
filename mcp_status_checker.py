```python
import yaml
import json
from typing import List, Dict, Any
from datetime import datetime
from interfaces.mcp_api_client import MCPAPIClient
from models.mcp_health import MCPHealthStatus, MCPMonitorResult
from utils.bushidan_logger import get_logger


class MCPStatusChecker:
    """Class responsible for checking the status of MCP endpoints."""
    
    def __init__(self):
        self.logger = get_logger("MCPStatusChecker")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open('bushidan_config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error("Configuration file not found")
            raise
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise
    
    def check_single_mcp(self, mcp_config: Dict[str, Any]) -> MCPHealthStatus:
        """Check the status of a single MCP endpoint."""
        name = mcp_config.get('name', 'Unknown')
        url = mcp_config.get('url', '')
        timeout = mcp_config.get('timeout', 30)
        
        self.logger.info(f"Checking MCP: {name} at {url}")
        
        # Validate URL
        if not url:
            error_msg = "Missing URL in configuration"
            self.logger.error(error_msg)
            return MCPHealthStatus(
                name=name,
                url=url,
                is_healthy=False,
                response_time_ms=0,
                timestamp=datetime.now().isoformat(),
                error_message=error_msg
            )
        
        # Create API client for this endpoint
        api_client = MCPAPIClient(url, timeout)
        
        try:
            # Check connectivity first
            if not api_client.test_connectivity():
                error_msg = "Failed to connect to MCP endpoint"
                self.logger.error(error_msg)
                return MCPHealthStatus(
                    name=name,
                    url=url,
                    is_healthy=False,
                    response_time_ms=0,
                    timestamp=datetime.now().isoformat(),
                    error_message=error_msg
                )
            
            # Perform health check
            result = api_client.health_check()
            
            if not result['success']:
                self.logger.error(f"Health check failed for {name}: {result.get('error', 'Unknown error')}")
                return MCPHealthStatus(
                    name=name,
                    url=url,
                    is_healthy=False,
                    response_time_ms=result['response_time_ms'],
                    timestamp=datetime.now().isoformat(),
                    error_message=result.get('error')
                )
            
            # Parse health status from JSON data
            json_data = result['json_data']
            is_healthy = True  # Default assumption
            
            details = {}
            if isinstance(json_data, dict):
                details.update(json_data)
                
                # Check for specific health indicators in the response
                if 'status' in json_data:
                    status_value = str(json_data['status']).lower()
                    is_healthy = status_value == 'healthy'
                elif 'health' in json_data:
                    health_value = json_data['health']
                    is_healthy = health_value.lower() == 'ok' or health_value is True
                
            self.logger.info(f"MCP {name} is {'healthy' if is_healthy else 'unhealthy'}")
            
            return MCPHealthStatus(
                name=name,
                url=url,
                is_healthy=is_healthy,
                response_time_ms=result['response_time_ms'],
                timestamp=datetime.now().isoformat(),
                details=details
            )
            
        except Exception as e:
            error_msg = f"Unexpected error checking {name}: {str(e)}"
            self.logger.error(error_msg)
            return MCPHealthStatus(
                name=name,
                url=url,
                is_healthy=False,
                response_time_ms=0,
                timestamp=datetime.now().isoformat(),
                error_message=error_msg
            )
    
    def check_all_mcp_endpoints(self) -> MCPMonitorResult:
        """Check the status of all configured MCP endpoints."""
        self.logger.info("Starting monitoring of all MCP endpoints")
        
        mcp_statuses = []
        system_name = self.config.get('system', {}).get('name', 'Bushidan System')
        version = self.config.get('system', {}).get('version', 'Unknown')
        
        # Get MCP endpoint configurations
        mcp_endpoints = self.config.get('mcp', {}).get('endpoints', [])
        
        if not mcp_endpoints:
            error_msg = "No MCP endpoints configured"
            self.logger.error(error_msg)
            
            return MCPMonitorResult(
                system_name=system_name,
                version