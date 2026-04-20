# Bushidan Multi-Agent System - Testing Guide

Comprehensive guide to testing the Bushidan Multi-Agent System, including unit tests, integration tests, and best practices.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Fixtures](#test-fixtures)
- [Coverage Goals](#coverage-goals)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Bushidan testing infrastructure provides comprehensive test coverage across all system components:

- **Unit Tests**: Individual component testing (target: 70%+ coverage)
- **Integration Tests**: End-to-end workflow testing
- **Benchmark Tests**: Performance and quality measurement
- **Mock Objects**: Isolated testing without external dependencies

**Current Status** (v9.3.2):
- Test files: 4
- Test cases: 60+
- Categories: unit, integration, benchmark
- Estimated coverage: 60-70%

## Test Structure

```
Bushidan-Multi-Agent/
├── tests/
│   ├── conftest.py              # Shared fixtures and configuration
│   ├── unit/                    # Unit tests
│   │   ├── test_shogun.py      # Shogun (Strategic Layer) tests
│   │   ├── test_karo.py        # Karo (Tactical Layer) tests
│   │   ├── test_taisho.py      # Taisho (Implementation Layer) tests
│   │   ├── test_intelligent_router.py  # Router logic tests
│   │   └── test_*_client.py    # LLM client tests
│   ├── integration/             # Integration tests
│   │   └── test_task_flows.py  # End-to-end workflow tests
│   └── fixtures/                # Test data files
├── benchmarks/                  # Performance benchmarks
│   ├── benchmark_framework.py
│   └── standard_tasks.py
└── pytest.ini                   # Pytest configuration
```

## Running Tests

### All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=core --cov=utils --cov-report=html
```

### Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_shogun.py -v

# Specific test class or function
pytest tests/unit/test_shogun.py::TestComplexityAssessment -v
pytest tests/unit/test_shogun.py::TestComplexityAssessment::test_assess_simple_task -v
```

### Using Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Run only slow tests
pytest -m slow

# Exclude slow tests
pytest -m "not slow"

# Run tests requiring API keys (with keys set)
pytest -m requires_api
```

### Coverage Analysis

```bash
# Generate coverage report
pytest --cov=core --cov=utils --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=core --cov=utils --cov-report=html
# Open htmlcov/index.html in browser

# Check coverage threshold
pytest --cov=core --cov=utils --cov-fail-under=70
```

## Writing Tests

### Unit Test Example

```python
"""tests/unit/test_my_component.py"""
import pytest
from unittest.mock import Mock, AsyncMock

from core.my_component import MyComponent


@pytest.mark.unit
class TestMyComponent:
    """Test MyComponent functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_functionality(self, mock_orchestrator):
        """Test basic component operation"""
        # Arrange
        component = MyComponent(mock_orchestrator)
        component.dependency = AsyncMock(return_value="expected")
        
        # Act
        result = await component.process("input")
        
        # Assert
        assert result == "expected"
        assert component.dependency.called
    
    def test_error_handling(self, mock_orchestrator):
        """Test error scenarios"""
        component = MyComponent(mock_orchestrator)
        component.dependency = Mock(side_effect=Exception("Error"))
        
        with pytest.raises(Exception) as exc_info:
            component.sync_process("input")
        
        assert "Error" in str(exc_info.value)
```

### Integration Test Example

```python
"""tests/integration/test_my_flow.py"""
import pytest

from core.shogun import Shogun, Task, TaskComplexity


@pytest.mark.integration
class TestMyWorkflow:
    """Test complete workflow"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_end_to_end_flow(self, mock_orchestrator):
        """Test complete task processing"""
        # Setup
        shogun = Shogun(mock_orchestrator)
        await shogun.initialize()
        
        # Create task
        task = Task(
            content="Test task content",
            complexity=TaskComplexity.MEDIUM
        )
        
        # Execute
        result = await shogun.process_task(task)
        
        # Verify
        assert result["status"] == "completed"
        assert "result" in result
```

### Async Test Example

```python
@pytest.mark.asyncio
async def test_async_operation(self, mock_orchestrator):
    """Test asynchronous operation"""
    component = AsyncComponent(mock_orchestrator)
    
    result = await component.async_method("input")
    
    assert result is not None
```

### Parametrized Test Example

```python
@pytest.mark.parametrize("input,expected", [
    ("simple", TaskComplexity.SIMPLE),
    ("medium implementation", TaskComplexity.MEDIUM),
    ("complex architecture", TaskComplexity.COMPLEX),
    ("strategic decision", TaskComplexity.STRATEGIC),
])
def test_complexity_judgment(self, input, expected):
    """Test various complexity judgments"""
    router = IntelligentRouter(config)
    
    result = router.judge_complexity(input)
    
    assert result == expected
```

## Test Fixtures

### Available Fixtures

Fixtures are defined in `tests/conftest.py`:

#### Configuration Fixtures

```python
def test_with_mock_config(mock_config):
    """Use mock configuration"""
    assert mock_config["system"]["version"] == "9.3.2"
```

#### Component Fixtures

```python
def test_with_orchestrator(mock_orchestrator):
    """Use mock system orchestrator"""
    assert mock_orchestrator.config is not None
```

#### LLM Client Fixtures

```python
async def test_with_claude(mock_claude_client):
    """Use mock Claude client"""
    response = await mock_claude_client.generate(
        messages=[{"role": "user", "content": "test"}]
    )
    assert response == "Test response from Claude"
```

Available client fixtures:
- `mock_claude_client`
- `mock_opus_client`
- `mock_gemini_client`
- `mock_groq_client`
- `mock_qwen3_client`

#### Task Fixtures

```python
def test_with_sample_task(sample_task_medium):
    """Use pre-defined sample task"""
    assert sample_task_medium.complexity == TaskComplexity.MEDIUM
```

Available task fixtures:
- `sample_task_simple`
- `sample_task_medium`
- `sample_task_complex`
- `sample_task_strategic`

#### MCP Fixtures

```python
async def test_with_memory_mcp(mock_memory_mcp):
    """Use mock Memory MCP"""
    await mock_memory_mcp.store({"data": "test"})
    assert mock_memory_mcp.store.called
```

Available MCP fixtures:
- `mock_memory_mcp`
- `mock_filesystem_mcp`
- `mock_git_mcp`

### Creating Custom Fixtures

```python
# In conftest.py or test file

@pytest.fixture
def my_custom_fixture():
    """Custom fixture for specific tests"""
    # Setup
    data = {"key": "value"}
    
    yield data  # Provide to test
    
    # Teardown (optional)
    data.clear()


def test_with_custom_fixture(my_custom_fixture):
    """Use custom fixture"""
    assert my_custom_fixture["key"] == "value"
```

## Coverage Goals

### Target Coverage

- **Overall**: 70%+
- **Core Components**: 80%+
  - `core/shogun.py`: 85%+
  - `core/karo.py`: 85%+
  - `core/taisho.py`: 80%+
  - `core/intelligent_router.py`: 90%+
- **Utilities**: 70%+
- **Integration**: Key workflows covered

### Checking Coverage

```bash
# Generate coverage report
pytest --cov=core --cov=utils --cov-report=term-missing

# View detailed coverage
pytest --cov=core --cov=utils --cov-report=html
open htmlcov/index.html
```

### Coverage Report Example

```
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
core/__init__.py                     10      0   100%
core/shogun.py                      527     79    85%   89-92, 245-248
core/karo.py                        312     47    85%   123-125, 289-295
core/intelligent_router.py          368     22    94%   301-305
utils/claude_client.py              145     29    80%   67-72, 134-138
---------------------------------------------------------------
TOTAL                              2847    312    89%
```

## Best Practices

### 1. Test Organization

✅ **Good**: Organized by component
```
tests/unit/test_shogun.py
tests/unit/test_karo.py
```

❌ **Bad**: Mixed or unclear organization
```
tests/test_everything.py
tests/random_tests.py
```

### 2. Test Naming

✅ **Good**: Clear, descriptive names
```python
def test_simple_task_routed_to_groq():
    """Test that simple tasks are routed to Groq"""
```

❌ **Bad**: Unclear names
```python
def test_1():
    """Test something"""
```

### 3. Test Structure (Arrange-Act-Assert)

✅ **Good**: Clear AAA pattern
```python
def test_complexity_assessment():
    # Arrange
    router = IntelligentRouter(config)
    task = "implement function"
    
    # Act
    complexity = router.judge_complexity(task)
    
    # Assert
    assert complexity == TaskComplexity.MEDIUM
```

### 4. Mocking

✅ **Good**: Mock external dependencies
```python
async def test_with_mocked_api(mock_claude_client):
    mock_claude_client.generate = AsyncMock(return_value="response")
    result = await shogun.process_task(task)
    assert mock_claude_client.generate.called
```

❌ **Bad**: Real API calls in tests
```python
async def test_with_real_api():
    # Calls real Claude API - slow, expensive, unreliable
    result = await real_client.generate(...)
```

### 5. Test Independence

✅ **Good**: Each test is independent
```python
def test_one():
    component = Component()
    assert component.method() == "result"

def test_two():
    component = Component()  # Fresh instance
    assert component.other_method() == "other"
```

❌ **Bad**: Tests depend on each other
```python
component = None

def test_one():
    global component
    component = Component()  # Shared state

def test_two():
    # Depends on test_one running first
    assert component.method() == "result"
```

### 6. Assertions

✅ **Good**: Specific, meaningful assertions
```python
assert result["status"] == "completed"
assert result["quality_score"] >= 90.0
assert "error" not in result
```

❌ **Bad**: Weak or unclear assertions
```python
assert result  # Too vague
assert True  # Meaningless
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'core'`

**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/Bushidan-Multi-Agent

# Run pytest from root
pytest

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

#### 2. Async Test Failures

**Problem**: `RuntimeError: asyncio.run() cannot be called from a running event loop`

**Solution**: Use `@pytest.mark.asyncio` decorator
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is not None
```

#### 3. Fixture Not Found

**Problem**: `fixture 'mock_orchestrator' not found`

**Solution**: Ensure `conftest.py` is in correct location
```bash
tests/
├── conftest.py  # Must be here
├── unit/
│   └── test_shogun.py
```

#### 4. API Key Errors in Tests

**Problem**: Tests fail with API key errors

**Solution**: Set mock API keys
```bash
export ANTHROPIC_API_KEY=test-key-123
export GOOGLE_API_KEY=test-key-123
pytest
```

Or in pytest:
```python
@pytest.fixture(autouse=True)
def set_mock_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
```

#### 5. Slow Tests

**Problem**: Tests take too long

**Solution**: 
1. Mark slow tests:
```python
@pytest.mark.slow
def test_long_operation():
    # Takes >1 second
    pass
```

2. Skip slow tests:
```bash
pytest -m "not slow"
```

3. Use mocks instead of real operations

### Getting Help

1. **Check test output**: `pytest -v --tb=short`
2. **Run specific test**: `pytest path/to/test.py::test_name -v`
3. **Enable debug logging**: `pytest --log-cli-level=DEBUG`
4. **Check fixtures**: `pytest --fixtures`

## Continuous Integration

Tests are automatically run on:
- Push to `main`, `develop`, `claude/*` branches
- Pull requests to `main`, `develop`
- Manual workflow dispatch

See `.github/workflows/ci.yml` for CI configuration.

## Next Steps

1. **Increase Coverage**: Aim for 80%+ overall coverage
2. **Add More Integration Tests**: Cover all major workflows
3. **Performance Tests**: Add timing benchmarks
4. **Property-Based Testing**: Consider using Hypothesis
5. **Mutation Testing**: Verify test quality with mutmut

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py](https://coverage.readthedocs.io/)
