```python
"""
Bushidan MCP Status Checker Module
This module provides functionality to check the status of MCP components 
within the Bushidan system.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Configure logger
logger = logging.getLogger('bushidan.mcp.status')

class MCPPingStatus:
    """Represents the status of an MCP ping operation"""
    
    def __init__(self, success: bool, message: str, response_time: float = 0.0):
        self.success = success
        self.message = message
        self.response_time = response_time

class MCPStatusChecker:
    """
    Main class for checking MCP status in Bushidan system.
    Handles communication with MCP services and provides status reporting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp_endpoints = config.get('mcp_endpoints', {})
        self.timeout = config.get('http_timeout', 10)
        self.max_retries = config.get('max_retries', 3)
        
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    timeout=self.timeout
                )
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed to reach {url}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def check_mcp_health(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Check health status of a specific MCP endpoint
        
        Args:
            endpoint_name (str): Name of the MCP endpoint to check
            
        Returns:
            dict: Status information including success flag and details
        """
        try:
            endpoint_config = self.mcp_endpoints.get(endpoint_name)
            if not endpoint_config:
                return {
                    'endpoint': endpoint_name,
                    'status': 'ERROR',
                    'message': f'No configuration found for {endpoint_name}',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            url = endpoint_config.get('health_check_url')
            if not url:
                return {
                    'endpoint': endpoint_name,
                    'status': 'ERROR', 
                    'message': f'Health check URL missing for {endpoint_name}',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            start_time = time.time()
            response = self._make_request(url)
            end_time = time.time()
            
            if not response:
                return {
                    'endpoint': endpoint_name,
                    'status': 'FAILED',
                    'message': f'Failed to reach {endpoint_name} after {self.max_retries} attempts',
                    'response_time_ms': round((end_time - start_time) * 1000, 2),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            # Check response status
            if response.status_code == 200:
                return {
                    'endpoint': endpoint_name,
                    'status': 'HEALTHY',
                    'message': f'{endpoint_name} is healthy',
                    'response_time_ms': round((end_time - start_time) * 1000, 2),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'endpoint': endpoint_name,
                    'status': 'UNHEALTHY',
                    'message': f'{endpoint_name} returned status code {response.status_code}',
                    'response_time_ms': round((end_time - start_time) * 1000, 2),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Unexpected error checking {endpoint_name}: {str(e)}")
            return {
                'endpoint': endpoint_name,
                'status': 'ERROR',
                'message': f'Unexpected error: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def check_all_mcp_endpoints(self) -> Dict[str, Any]:
        """
        Check status of all configured MCP endpoints
        
        Returns:
            dict: Overall system status with individual endpoint details
        """
        overall_status = {
            'system_status': 'HEALTHY',
            'timestamp': datetime.utcnow().isoformat(),
            'endpoints': {}
        }
        
        for endpoint_name in self.mcp_endpoints.keys():
            endpoint_status = self.check_mcp_health(endpoint_name)
            overall_status['endpoints'][endpoint_name] = endpoint_status
            
            # Update overall status if any endpoint is unhealthy
            if endpoint_status['status'] != 'HEALTHY':
                overall_status['system_status'] = 'UNHEALTHY'
                
        return overall_status

# Example usage function (can be removed or used for testing)
def main():
    """Example of how to use the MCP status checker"""
    # Load config
    config = {
        "mcp_endpoints": {
            "core_mcp": {
                "health_check_url": "http://localhost:8080/health"
            },
            "data_mcp": {
                "health_check_url": "http://localhost:8081/health"
            }
        },
        "http_timeout": 5,
        "max_retries": 3
    }
    
    checker = MCPStatusChecker(config)
    status_report = checker.check_all_mcp_endpoints()
    
    print(json.dumps(status_report, indent=2))

if __name__ == "__main__":
    main()
```