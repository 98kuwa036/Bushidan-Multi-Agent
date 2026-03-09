```python
"""
MCP Monitoring Service for Bushidan System
Provides monitoring capabilities for MCP components.
"""

import logging
from typing import Dict, Any
import time
import psutil
import pymongo
import requests

logger = logging.getLogger(__name__)


class MCPMonitor:
    """
    Monitor service for checking MCP system status and health.
    
    This class handles the actual monitoring tasks including database 
    connectivity checks, resource usage monitoring, and configuration validation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MCP monitor with configuration.
        
        Args:
            config (Dict): Configuration dictionary for MCP components
        """
        self.config = config
        # Cache for database connection to avoid reconnection overhead
        self._db_client_cache = None

    def get_mcp_status(self) -> Dict[str, Any]:
        """
        Get status of the main MCP service.
        
        Returns:
            Dictionary containing MCP connectivity and health info
        """
        try:
            mcp_config = self.config.get('mcp', {})
            
            # Check connection to MCP endpoint
            connected = False
            connection_errors = 0
            
            if 'endpoint' in mcp_config:
                try:
                    response = requests.get(
                        mcp_config['endpoint'], 
                        timeout=mcp_config.get('timeout', 5)
                    )
                    connected = response.status_code == 200
                    logger.debug(f"MCP endpoint responded with status: {response.status_code}")
                except Exception as e:
                    connection_errors += 1
                    logger.warning(f"Failed to connect to MCP endpoint: {str(e)}")
            
            return {
                "connected": connected,
                "connection_errors": connection_errors,
                "endpoint": mcp_config.get('endpoint', 'N/A'),
                "version": mcp_config.get('version', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error checking MCP status: {str(e)}")
            return {
                "connected": False,
                "connection_errors": 1,
                "endpoint": self.config.get('mcp', {}).get('endpoint', 'N/A'),
                "version": 'unknown'
            }

    def check_database_health(self) -> Dict[str, Any]:
        """
        Check MongoDB instance health and replica set status.
        
        Returns:
            Dictionary containing database health information
        """
        try:
            db_config = self.config.get('database', {})
            
            # Use cached client if available to avoid repeated connections
            if not self._db_client_cache:
                connection_string = db_config.get('connection_string')
                if connection_string:
                    self._db_client_cache = pymongo.MongoClient(
                        connection_string,
                        serverSelectionTimeoutMS=db_config.get('timeout', 5000)
                    )
            
            # Check database connectivity
            if not self._db_client_cache:
                return {
                    "status": "critical",
                    "error": "No database connection string configured"
                }
                
            # Test server selection
            try:
                # This will trigger a server selection timeout if there's an issue
                self._db_client_cache.server_info()
                db_status = "healthy"
                
                # Check replica set status if applicable
                try:
                    repl_set_status = self._db_client_cache.admin.command("replSetGetStatus")
                    primary = None
                    
                    for member in repl_set_status.get('members', []):
                        if member['stateStr'] == 'PRIMARY':
                            primary = member['name']
                            break
                            
                    return {
                        "status": db_status,
                        "replica_set": True,
                        "primary": primary,
                        "nodes": len(repl_set_status.get('members', []))
                    }
                except:
                    # Not a replica set or command not available, proceed with basic status
                    return {
                        "status": db_status,
                        "replica_set": False,
                        "primary": None,
                        "nodes": 1
                    }
                    
            except Exception as e:
                logger.error(f"Database connection failed: {str(e)}")
                return {
                    "status": "critical",
                    "error": str(e)
                }
                
        except Exception as e:
            logger.error(f"Error checking database health: {str(e)}")
            return {
                "status": "unknown",
                "error": str(e)
            }

    def check_system_resources(self) -> Dict[str, Any]:
        """
        Monitor system resource usage affecting MCP performance.
        
        Returns:
            Dictionary containing system resource information
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            return {
                "cpu_usage_percent": cpu_percent,
                "memory_total_gb": round(memory_info.total / (1024**3), 2),
                "memory_used_gb": round(memory_info.used / (1024**3), 2),
                "memory_usage_percent": memory_info.percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
            
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return {
                "cpu_usage_percent": -1,
                "memory_total_gb": 0,
                "memory_used_gb": 0,
                "memory_usage_percent": -1,
                "disk_usage_percent": -1
            }

    def validate_config(self) -> Dict[str, Any]:
        """
        Validate MCP configuration against expected parameters.
        
        Returns:
            Dictionary containing validation results
        """
        try:
            required_fields = ['endpoint', 'version']
            missing_fields = []
            
            mcp_config = self.config.get('mcp', {})
            
            for field in required_fields:
                if not mcp_config.get(field):
                    missing_fields.append(field)
                    
            return {
                "valid": len(missing_fields) == 0,
                "missing_fields": missing_fields
            }
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "missing_fields": []
            }

    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system health report.
        
        Returns:
            Dictionary with complete system health information
        """
        try:
            mcp_status = self.get_mcp_status()
            db_status = self.check_database_health()
            resource_status = self.check_system_resources()
            
            return {
                "timestamp": time.time(),
                "mcp": mcp_status,
                "database": db_status,
                "resources": resource_status
            }
        except Exception as e:
            logger.error(f"Error generating system health report: {str(e)}")
            raise
```