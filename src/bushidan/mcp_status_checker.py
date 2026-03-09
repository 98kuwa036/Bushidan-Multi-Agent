```python
"""
MCP Status Checker Module for Bushidan System
Handles monitoring and verification of MCP (Management Control Point) status.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .mcp_monitor import MCPMonitor
from .config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class MCPStatusChecker:
    """
    Main class for checking MCP system status in Bushidan v10.1.
    
    This class coordinates monitoring activities and provides 
    comprehensive status reporting for the MCP components.
    """

    def __init__(self, config_path: str = "config/mcp_config.yaml"):
        """
        Initialize the MCP Status Checker with configuration.
        
        Args:
            config_path (str): Path to the MCP configuration file
        """
        self.config_loader = ConfigLoader(config_path)
        self.monitor = MCPMonitor(self.config_loader.get_config())
        self.status_cache = {}

    def check_all_components(self) -> Dict[str, Any]:
        """
        Perform comprehensive status check of all MCP components.
        
        Returns:
            Dict containing the complete status report
        """
        logger.info("Starting comprehensive MCP status check")
        
        # Collect all component statuses
        try:
            mcp_status = self.monitor.get_mcp_status()
            db_status = self.monitor.check_database_health()
            system_status = self.monitor.check_system_resources()
            
            # Combine all status information
            full_status = {
                "timestamp": datetime.now().isoformat(),
                "mcp": mcp_status,
                "database": db_status,
                "system": system_status,
                "overall_status": self._calculate_overall_status(
                    mcp_status, 
                    db_status, 
                    system_status
                )
            }
            
            logger.info("MCP status check completed successfully")
            return full_status
            
        except Exception as e:
            logger.error(f"Error during MCP status check: {str(e)}")
            raise

    def _calculate_overall_status(self, mcp_status: Dict, db_status: Dict, 
                                  system_status: Dict) -> str:
        """
        Calculate overall system status based on component statuses.
        
        Args:
            mcp_status (Dict): Status of MCP components
            db_status (Dict): Database health status
            system_status (Dict): System resource status
            
        Returns:
            str: Overall system status ('healthy', 'warning', or 'critical')
        """
        # Check for critical issues first
        if not mcp_status.get('connected', False) or \
           db_status.get('status') == 'critical':
            return 'critical'
            
        # Check for warning conditions
        if (mcp_status.get('connection_errors', 0) > 0 or 
            system_status.get('cpu_usage_percent', 0) > 85 or
            system_status.get('memory_usage_percent', 0) > 90):
            return 'warning'
            
        # Default to healthy status
        return 'healthy'

    def get_status_report(self, format: str = "json") -> Any:
        """
        Get formatted status report.
        
        Args:
            format (str): Output format ('json' or 'dict')
            
        Returns:
            Formatted status report based on requested format
        """
        status_data = self.check_all_components()
        
        if format == "json":
            import json
            return json.dumps(status_data, indent=2)
        else:
            return status_data

    def is_healthy(self) -> bool:
        """
        Check if the overall system is healthy.
        
        Returns:
            bool: True if system is healthy, False otherwise
        """
        status = self.check_all_components()
        return status.get('overall_status') == 'healthy'


# Example usage function (for testing purposes)
def main():
    """Main function to demonstrate MCP status checking."""
    try:
        checker = MCPStatusChecker()
        report = checker.get_status_report("json")
        print(report)
        
        if checker.is_healthy():
            logger.info("MCP system is healthy")
        else:
            logger.warning("MCP system has issues")
            
    except Exception as e:
        logger.error(f"Failed to check MCP status: {str(e)}")


if __name__ == "__main__":
    main()
```