```python
import requests
from typing import Dict, Any, Optional
import time
from urllib.parse import urlparse


class MCPAPIClient:
    """API client for communicating with MCP endpoints."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the MCP endpoint."""
        try:
            start_time = time.time()
            
            # Construct URL for health endpoint
            url = f"{self.base_url}/health"
            
            response = self.session.get(
                url,
                timeout=self.timeout
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return {
                'status_code': response.status_code,
                'response_time_ms': response_time_ms,
                'json_data': response.json() if response.content else None,
                'headers': dict(response.headers),
                'success': True
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'error': str(e),
                'status_code': None,
                'response_time_ms': 0,
                'json_data': None,
                'headers': {},
                'success': False
            }
    
    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get information about the endpoint."""
        try:
            url = f"{self.base_url}/info"
            
            response = self.session.get(
                url,
                timeout=self.timeout
            )
            
            return {
                'status_code': response.status_code,
                'json_data': response.json() if response.content else None,
                'success': True
            }
        except requests.exceptions.RequestException as e:
            return {
                'error': str(e),
                'status_code': None,
                'json_data': None,
                'success': False
            }
    
    def test_connectivity(self) -> bool:
        """Test basic connectivity to the endpoint."""
        try:
            # Simple HEAD request to check if service is reachable
            response = self.session.head(
                self.base_url,
                timeout=self.timeout
            )
            return response.status_code < 400
        except requests.exceptions.RequestException:
            return False
    
    def validate_url(self) -> bool:
        """Validate that the URL is properly formatted."""
        try:
            result = urlparse(self.base_url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
```