```python
"""
Unit Tests for MCP Status Checker in Bushidan System
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

# Add the src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.bushidan.mcp_status_checker import MCPStatusChecker
from src.bushidan.mcp_monitor import MCPMonitor


class TestMCPStatusChecker(unittest.TestCase):
    """Test cases for MCP Status Checker functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock config with minimal required fields
        self.mock_config = {
            'mcp': {
                'endpoint': 'http://localhost:8080/api/health',
                'version': 'v10.1',
                'timeout': 5
            },
            'database': {
                'connection_string': 'mongodb://localhost:27017/test_db',
                'timeout': 5000
            }
        }

    def test_init(self):
        """Test MCPStatusChecker initialization."""
        checker = MCPStatusChecker()
        
        self.assertIsNotNone(checker)
        self.assertIsNotNone(checker.config_loader)
        self.assertIsNotNone(checker.monitor)

    @patch('requests.get')
    def test_check_mcp_status_connected(self, mock_get):
        """Test checking MCP status when connected."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        monitor = MCPMonitor(self.mock_config)
        result = monitor.get_mcp_status()
        
        self.assertTrue(result['connected'])
        self.assertEqual(result['connection_errors'], 0)

    @patch('requests.get')
    def test_check_mcp_status_disconnected(self, mock_get):
        """Test checking MCP status when disconnected."""