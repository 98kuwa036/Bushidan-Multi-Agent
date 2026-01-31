"""
Bushidan Multi-Agent System - Test Configuration

Shared pytest fixtures and configuration for all tests.
Provides mock objects, test data, and common utilities.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock system configuration"""
    return {
        "claude_api_key": "test-api-key-123",
        "gemini_api_key": "test-gemini-key",
        "groq_api_key": "test-groq-key",
        "alibaba_api_key": "test-alibaba-key",
        "system": {
            "version": "9.3.2",
            "log_level": "INFO"
        },
        "shogun": {
            "cli_model": "sonnet",
            "api_model": "claude-sonnet-4-5-20250929",
            "pro_limit": 2000,
            "opus_model": "claude-opus-4-20250514"
        },
        "karo": {
            "primary_model": "gemini-3-flash",
            "fallback_model": "groq"
        },
        "taisho": {
            "local_model": "qwen3-coder-30b",
            "cloud_model": "qwen3-coder-plus",
            "context_length": 4096
        },
        "intelligent_routing": {
            "enabled": True,
            "power_saving": True
        }
    }


@pytest.fixture
def mock_orchestrator(mock_config):
    """Mock SystemOrchestrator"""
    orchestrator = Mock()
    orchestrator.config = Mock()
    
    # Configure mock attributes
    for key, value in mock_config.items():
        setattr(orchestrator.config, key, value)
    
    # Mock MCP methods
    orchestrator.get_mcp = Mock(return_value=None)
    orchestrator.initialize = AsyncMock()
    
    return orchestrator


@pytest.fixture
def mock_claude_client():
    """Mock ClaudeClient"""
    client = Mock()
    client.generate = AsyncMock(return_value="Test response from Claude")
    client.pro_calls_used = 0
    client.pro_limit = 2000
    return client


@pytest.fixture
def mock_opus_client():
    """Mock OpusClient"""
    from utils.opus_client import OpusReview
    
    client = Mock()
    mock_review = OpusReview(
        score=95.0,
        decision="approved",
        critical_issues=[],
        recommendations=["Consider adding more tests"],
        review_text="Excellent implementation",
        cost_yen=10.5,
        review_time_seconds=15.2
    )
    client.conduct_premium_review = AsyncMock(return_value=mock_review)
    client.get_statistics = Mock(return_value={
        "total_reviews": 10,
        "total_cost_yen": 105.0
    })
    return client


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini3Client"""
    client = Mock()
    client.generate = AsyncMock(return_value="Test response from Gemini")
    client.generate_with_context = AsyncMock(return_value="Contextual response")
    return client


@pytest.fixture
def mock_groq_client():
    """Mock GroqClient"""
    client = Mock()
    client.generate = AsyncMock(return_value="Fast response from Groq")
    client.check_rate_limit = Mock(return_value=True)
    return client


@pytest.fixture
def mock_qwen3_client():
    """Mock Qwen3Client (local)"""
    client = Mock()
    client.generate = AsyncMock(return_value="def hello(): return 'world'")
    client.check_context_overflow = Mock(return_value=False)
    return client


@pytest.fixture
def mock_intelligent_router():
    """Mock IntelligentRouter"""
    from core.intelligent_router import TaskComplexity, RouteTarget, RoutingDecision
    
    router = Mock()
    
    # Mock judge_complexity method
    router.judge_complexity = Mock(return_value=TaskComplexity.MEDIUM)
    
    # Mock determine_route method
    mock_decision = RoutingDecision(
        target=RouteTarget.LOCAL_QWEN3,
        complexity=TaskComplexity.MEDIUM,
        fallback_chain=[RouteTarget.LOCAL_QWEN3, RouteTarget.CLOUD_QWEN3, RouteTarget.GEMINI3],
        reasoning="Medium task routed to local Qwen3",
        estimated_time_seconds=12,
        estimated_cost_yen=0.0,
        power_saving=False
    )
    router.determine_route = Mock(return_value=mock_decision)
    
    # Mock statistics
    router.get_routing_stats = Mock(return_value={
        "total_tasks": 100,
        "routes_by_target": {"groq": 40, "local_qwen3": 50, "cloud_qwen3": 8, "gemini3": 2},
        "fallbacks_triggered": 10,
        "average_time_seconds": 8.5,
        "total_cost_yen": 24.5,
        "power_savings_yen": 80.0
    })
    
    return router


@pytest.fixture
def mock_quality_metrics():
    """Mock QualityMetricsCollector"""
    from utils.quality_metrics import (
        QualityReport, ComplexityMetrics, SecurityFindings, RiskLevel
    )
    
    collector = Mock()
    
    # Create mock report
    mock_complexity = ComplexityMetrics(
        lines_of_code=150,
        cyclomatic_complexity=8,
        cognitive_complexity=12,
        max_nesting_depth=3,
        average_function_length=15.5,
        complexity_score=45.2,
        risk_level=RiskLevel.MEDIUM
    )
    
    mock_security = SecurityFindings(
        vulnerabilities=[],
        warnings=["Consider input validation"],
        security_score=85.0,
        risk_level=RiskLevel.LOW
    )
    
    mock_report = QualityReport(
        task_id="test-123",
        complexity_metrics=mock_complexity,
        security_findings=mock_security,
        overall_quality_score=88.5,
        recommendations=["Add unit tests", "Consider error handling"]
    )
    
    collector.collect_metrics = Mock(return_value=mock_report)
    collector.get_aggregate_stats = Mock(return_value={
        "total_analyses": 50,
        "average_quality_score": 87.3
    })
    
    return collector


@pytest.fixture
def sample_task_simple():
    """Sample simple task"""
    from core.shogun import Task, TaskComplexity
    return Task(
        content="What is the current date?",
        complexity=TaskComplexity.SIMPLE,
        context=None,
        priority=1,
        source="test"
    )


@pytest.fixture
def sample_task_medium():
    """Sample medium complexity task"""
    from core.shogun import Task, TaskComplexity
    return Task(
        content="Implement a function to calculate fibonacci numbers up to n",
        complexity=TaskComplexity.MEDIUM,
        context={"language": "python"},
        priority=2,
        source="test"
    )


@pytest.fixture
def sample_task_complex():
    """Sample complex task"""
    from core.shogun import Task, TaskComplexity
    return Task(
        content="Refactor the authentication system to use JWT tokens with refresh token rotation",
        complexity=TaskComplexity.COMPLEX,
        context={"files": ["auth.py", "middleware.py", "models.py"]},
        priority=3,
        source="test"
    )


@pytest.fixture
def sample_task_strategic():
    """Sample strategic task"""
    from core.shogun import Task, TaskComplexity
    return Task(
        content="Design the architecture for a new microservices-based payment system",
        complexity=TaskComplexity.STRATEGIC,
        context={"requirements": ["PCI compliance", "99.99% uptime", "multi-currency"]},
        priority=5,
        source="test"
    )


@pytest.fixture
def mock_memory_mcp():
    """Mock Memory MCP"""
    mcp = Mock()
    mcp.store = AsyncMock()
    mcp.retrieve = AsyncMock(return_value=[])
    mcp.search = AsyncMock(return_value=[])
    return mcp


@pytest.fixture
def mock_filesystem_mcp():
    """Mock Filesystem MCP"""
    mcp = Mock()
    mcp.read_file = AsyncMock(return_value="File content")
    mcp.write_file = AsyncMock()
    mcp.list_directory = AsyncMock(return_value=["file1.py", "file2.py"])
    return mcp


@pytest.fixture
def mock_git_mcp():
    """Mock Git MCP"""
    mcp = Mock()
    mcp.git_status = AsyncMock(return_value="On branch main")
    mcp.git_commit = AsyncMock()
    mcp.git_diff = AsyncMock(return_value="")
    return mcp


# Test data fixtures

@pytest.fixture
def sample_code_implementation():
    """Sample code implementation for testing"""
    return '''
def fibonacci(n):
    """Calculate fibonacci numbers up to n"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

# Test
print(fibonacci(10))
'''


@pytest.fixture
def sample_code_with_issues():
    """Sample code with security and quality issues"""
    return '''
import os

def execute_command(user_input):
    # Security issue: Command injection vulnerability
    os.system(user_input)

def get_password():
    # Security issue: Hardcoded credentials
    password = "admin123"
    return password

def complex_nested_function(x):
    # Quality issue: High complexity
    if x > 0:
        if x > 10:
            if x > 20:
                if x > 30:
                    return "very high"
                return "high"
            return "medium"
        return "low"
    return "zero or negative"
'''


# Pytest configuration

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for system flows"
    )
    config.addinivalue_line(
        "markers", "benchmark: Performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (>1 second)"
    )


@pytest.fixture(autouse=True)
def reset_mocks(monkeypatch):
    """Automatically reset mocks between tests"""
    yield
    # Cleanup happens automatically with pytest
