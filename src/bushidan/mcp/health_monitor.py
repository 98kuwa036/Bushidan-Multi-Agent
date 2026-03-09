```python
"""
Health Monitoring Service for Bushidan System MCP Components.

This module provides a service to monitor the health of MCP services
and report on system status.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio

logger = logging.getLogger(__name__)

class HealthMonitor:
    """
    Health monitoring service for MCP components in Bushidan system.
    
    Tracks service health over time and provides reporting capabilities.
    """

    def __init__(self):
        """Initialize the health monitor."""
        self.health_history: Dict[str, List[Dict]] = {}
        self.alert_thresholds = {
            'cpu_high': 80.0,
            'memory_high': 85.0
        }
        logger.info("HealthMonitor initialized")

    def record_service_health(self, service_name: str, status: Dict) -> None:
        """
        Record health status for a service.
        
        Args:
            service_name (str): Name of the service
            status (Dict): Health status dictionary
        """
        if service_name not in self.health_history:
            self.health_history[service_name] = []
            
        # Add current status to history
        record = {
            'timestamp': datetime.now(),
            'status': status.copy()
        }
        
        self.health_history[service_name].append(record)
        
        # Keep only last 100 records
        if len(self.health_history[service_name]) > 100:
            self.health_history[service_name] = \
                self.health_history[service_name][-100:]

    def get_service_health_report(self, service_name: str) -> Optional[Dict]:
        """
        Generate a health report for a specific service.
        
        Args:
            service_name (str): Name of the service
            
        Returns:
            Dict containing health report or None if no data
        """
        if service_name not in self.health_history:
            return None
            
        history = self.health_history[service_name]
        if not history:
            return None
            
        # Get latest status
        latest_status = history[-1]['status']
        
        # Calculate average resource usage over recent history (last 10 entries)
        recent_data = history[-10:] if len(history) >= 10 else history
        
        avg_cpu = sum(record['status']['resource_usage'].get('cpu_percent', 0) 
                     for record in recent_data) / len(recent_data)
        
        return {
            'service': service_name,
            'latest_status': latest_status,
            'average_cpu_percent': round(avg_cpu, 2),
            'total_records': len(history),
            'last_updated': history[-1]['timestamp']
        }

    def check_alerts(self) -> List[Dict]:
        """
        Check for any alerts based on system thresholds.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Check overall system health (could be extended to check specific services)
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            if cpu_percent > self.alert_thresholds['cpu_high']:
                alerts.append({
                    'type': 'HIGH_CPU_USAGE',
                    'message': f'CPU usage is high: {cpu_percent}%',
                    'severity': 'WARNING'
                })
                
            if memory_info.percent > self.alert_thresholds['memory_high']:
                alerts.append({
                    'type': 'HIGH_MEMORY_USAGE', 
                    'message': f'Memory usage is high: {memory_info.percent}%',
                    'severity': 'WARNING'
                })
                
        except Exception as e:
            logger.error("Error checking system alerts: %s", str(e))
            
        return alerts

    def get_overall_system_health(self) -> Dict[str, Any]:
        """
        Get overall health of the system based on recent history.
        
        Returns:
            Dictionary with system health metrics
        """
        healthy_services = 0
        total_services = len(self.health_history)
        
        for service_name, records in self.health_history.items():
            if records and records[-1]['status'].get('is_healthy', False):
                healthy_services += 1
                
        return {
            'timestamp': datetime.now(),
            'healthy_services': healthy_services,
            'total_services': total_services,
            'health_percentage': (healthy_services / total_services * 100) if total_services > 0 else 0
        }
```