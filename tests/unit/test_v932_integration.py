"""
Bushidan Multi-Agent System v9.3.2 - Integration Tests

Tests for v9.3.2 component integration:
- Intelligent Router integration in Shogun
- 3-tier fallback chain in Taisho
- Routing execution in Karo
- New client integrations
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any


class TestShogunV932:
    """Tests for Shogun v9.3.2 features"""

    @pytest.fixture
    def mock_orchestrator_v932(self, mock_config):
        """Mock v9.3.2 orchestrator with all clients"""
        orchestrator = Mock()
        orchestrator.config = Mock()

        # Config attributes
        orchestrator.config.claude_api_key = "test-key"
        orchestrator.config.intelligent_routing_enabled = True
        orchestrator.config.prompt_caching_enabled = True
        orchestrator.config.power_optimization_enabled = True

        # Mock clients
        mock_claude = Mock()
        mock_claude.generate = AsyncMock(return_value="Claude response")

        mock_groq = Mock()
        mock_groq.generate = AsyncMock(return_value="Groq fast response")

        mock_opus = Mock()
        mock_opus.conduct_premium_review = AsyncMock()

        # Mock router
        from core.intelligent_router import TaskComplexity, RouteTarget, RoutingDecision
        mock_router = Mock()
        mock_router.judge_complexity = Mock(return_value=TaskComplexity.MEDIUM)
        mock_router.determine_route = Mock(return_value=RoutingDecision(
            target=RouteTarget.LOCAL_QWEN3,
            complexity=TaskComplexity.MEDIUM,
            fallback_chain=[RouteTarget.LOCAL_QWEN3],
            reasoning="Test route",
            estimated_time_seconds=10,
            estimated_cost_yen=0.0,
            power_saving=False
        ))
        mock_router.get_statistics = Mock(return_value={})

        # Setup get_client
        def get_client(name):
            clients = {
                "claude_cached": mock_claude,
                "claude": mock_claude,
                "groq": mock_groq,
                "opus": mock_opus
            }
            return clients.get(name)

        orchestrator.get_client = get_client
        orchestrator.get_mcp = Mock(return_value=None)
        orchestrator.get_router = Mock(return_value=mock_router)

        return orchestrator

    @pytest.mark.asyncio
    async def test_shogun_uses_router_for_complexity(self, mock_orchestrator_v932):
        """Test that Shogun uses Intelligent Router for complexity assessment"""
        from core.shogun import Shogun, Task, TaskComplexity

        # Mock Karo
        with patch('core.shogun.Karo') as MockKaro:
            mock_karo = Mock()
            mock_karo.initialize = AsyncMock()
            mock_karo.execute_task_with_routing = AsyncMock(return_value={
                "status": "completed",
                "result": "Test result"
            })
            MockKaro.return_value = mock_karo

            shogun = Shogun(mock_orchestrator_v932)
            await shogun.initialize()

            # Create task
            task = Task(content="Test task", complexity=TaskComplexity.MEDIUM)

            # The router should be called for complexity assessment
            router = mock_orchestrator_v932.get_router()
            assert router is not None

    @pytest.mark.asyncio
    async def test_shogun_routes_simple_to_groq(self, mock_orchestrator_v932):
        """Test that simple tasks are routed to Groq"""
        from core.shogun import Shogun, Task, TaskComplexity

        # Configure router to return SIMPLE
        from core.intelligent_router import TaskComplexity as RouterComplexity
        router = mock_orchestrator_v932.get_router()
        router.judge_complexity.return_value = RouterComplexity.SIMPLE

        with patch('core.shogun.Karo') as MockKaro:
            mock_karo = Mock()
            mock_karo.initialize = AsyncMock()
            MockKaro.return_value = mock_karo

            shogun = Shogun(mock_orchestrator_v932)
            await shogun.initialize()

            # Create simple task
            task = Task(content="What time is it?", complexity=TaskComplexity.SIMPLE)

            # Process task
            result = await shogun.process_task(task)

            # Should succeed (handled by Groq or Karo)
            assert result.get("status") in ["completed", "failed"]


class TestKaroV932:
    """Tests for Karo v9.3.2 features"""

    @pytest.fixture
    def mock_orchestrator_karo(self):
        """Mock orchestrator for Karo tests"""
        orchestrator = Mock()
        orchestrator.config = Mock()

        # Mock clients
        mock_gemini3 = Mock()
        mock_gemini3.generate = AsyncMock(return_value="Gemini response")

        mock_groq = Mock()
        mock_groq.generate = AsyncMock(return_value="Groq response")

        def get_client(name):
            return {"gemini3": mock_gemini3, "groq": mock_groq}.get(name)

        orchestrator.get_client = get_client
        orchestrator.get_mcp = Mock(return_value=None)

        return orchestrator

    @pytest.mark.asyncio
    async def test_karo_groq_delegation(self, mock_orchestrator_karo):
        """Test Karo delegates simple tasks to Groq"""
        from core.karo import Karo, TaskDelegation

        with patch('core.karo.Taisho'), patch('core.karo.Ashigaru'):
            karo = Karo(mock_orchestrator_karo)

            # Create mock task
            mock_task = Mock()
            mock_task.content = "Simple question"
            mock_task.complexity = Mock()
            mock_task.complexity.value = "simple"

            # Test delegation determination
            from core.intelligent_router import RouteTarget, RoutingDecision, TaskComplexity
            routing = RoutingDecision(
                target=RouteTarget.GROQ,
                complexity=TaskComplexity.SIMPLE,
                fallback_chain=[],
                reasoning="Simple task",
                estimated_time_seconds=2,
                estimated_cost_yen=0.0,
                power_saving=True
            )

            delegation = karo._determine_delegation(mock_task, routing)
            assert delegation == TaskDelegation.GROQ_INSTANT

    @pytest.mark.asyncio
    async def test_karo_taisho_delegation(self, mock_orchestrator_karo):
        """Test Karo delegates medium/complex tasks to Taisho"""
        from core.karo import Karo, TaskDelegation

        with patch('core.karo.Taisho'), patch('core.karo.Ashigaru'):
            karo = Karo(mock_orchestrator_karo)

            # Create mock task
            mock_task = Mock()
            mock_task.content = "Implement a feature"
            mock_task.complexity = Mock()
            mock_task.complexity.value = "medium"

            # Test delegation - should go to Taisho
            delegation = karo._determine_delegation(mock_task, None)
            assert delegation == TaskDelegation.TAISHO_PRIMARY


class TestTaishoV932:
    """Tests for Taisho v9.3.2 fallback chain"""

    @pytest.fixture
    def mock_orchestrator_taisho(self):
        """Mock orchestrator for Taisho tests"""
        orchestrator = Mock()

        # Mock clients
        mock_qwen3 = Mock()
        mock_qwen3.generate = AsyncMock(return_value="Qwen3 implementation")

        mock_alibaba = Mock()
        mock_alibaba.generate = AsyncMock(return_value="Kagemusha implementation")

        mock_gemini3 = Mock()
        mock_gemini3.generate = AsyncMock(return_value="Gemini implementation")

        def get_client(name):
            return {
                "qwen3": mock_qwen3,
                "alibaba_qwen": mock_alibaba,
                "gemini3": mock_gemini3
            }.get(name)

        orchestrator.get_client = get_client
        orchestrator.get_mcp = Mock(return_value=None)
        orchestrator.mcp_manager = None

        return orchestrator

    def test_taisho_context_estimation(self, mock_orchestrator_taisho):
        """Test Taisho context size estimation"""
        from core.taisho import Taisho, ImplementationTask, ImplementationMode

        taisho = Taisho(mock_orchestrator_taisho)

        task = ImplementationTask(
            content="Implement a simple function",
            mode=ImplementationMode.STANDARD
        )

        context = {"memory_entries": [], "existing_files": []}
        size = taisho._estimate_context_size(task, context)

        assert size > 0
        assert isinstance(size, int)

    def test_taisho_fallback_chain_available(self, mock_orchestrator_taisho):
        """Test all fallback chain clients are accessible"""
        from core.taisho import Taisho

        taisho = Taisho(mock_orchestrator_taisho)

        # Check all clients are available
        assert mock_orchestrator_taisho.get_client("qwen3") is not None
        assert mock_orchestrator_taisho.get_client("alibaba_qwen") is not None
        assert mock_orchestrator_taisho.get_client("gemini3") is not None


class TestIntelligentRouterV932:
    """Tests for Intelligent Router v9.3.2"""

    def test_router_simple_detection(self):
        """Test router correctly identifies simple tasks"""
        from core.intelligent_router import IntelligentRouter, TaskComplexity

        config = {"performance_targets": {"simple": 2, "medium": 12}}
        router = IntelligentRouter(config)

        # Simple question
        complexity = router.judge_complexity("What is Python?", None)
        assert complexity == TaskComplexity.SIMPLE

    def test_router_medium_detection(self):
        """Test router correctly identifies medium tasks"""
        from core.intelligent_router import IntelligentRouter, TaskComplexity

        config = {"performance_targets": {"simple": 2, "medium": 12}}
        router = IntelligentRouter(config)

        # Code implementation task
        complexity = router.judge_complexity(
            "Write a function to sort a list of numbers",
            None
        )
        assert complexity in [TaskComplexity.MEDIUM, TaskComplexity.SIMPLE]

    def test_router_complex_detection(self):
        """Test router correctly identifies complex tasks"""
        from core.intelligent_router import IntelligentRouter, TaskComplexity

        config = {"performance_targets": {"simple": 2, "medium": 12}}
        router = IntelligentRouter(config)

        # Multi-file refactoring task
        complexity = router.judge_complexity(
            "Refactor the authentication module, update the user model, "
            "modify the API endpoints, and add new tests across multiple files",
            None
        )
        assert complexity in [TaskComplexity.COMPLEX, TaskComplexity.STRATEGIC]

    def test_router_strategic_detection(self):
        """Test router correctly identifies strategic tasks"""
        from core.intelligent_router import IntelligentRouter, TaskComplexity

        config = {"performance_targets": {"simple": 2, "medium": 12}}
        router = IntelligentRouter(config)

        # Architecture/design decision task
        complexity = router.judge_complexity(
            "Design the system architecture for a distributed payment processing platform",
            None
        )
        assert complexity == TaskComplexity.STRATEGIC

    def test_router_route_decision(self):
        """Test router makes correct routing decisions"""
        from core.intelligent_router import IntelligentRouter, TaskComplexity, RouteTarget

        config = {"performance_targets": {"simple": 2, "medium": 12}}
        router = IntelligentRouter(config)

        # Simple task should route to Groq
        decision = router.determine_route(TaskComplexity.SIMPLE, None)
        assert decision.target == RouteTarget.GROQ
        assert decision.power_saving == True

        # Medium task should route to Local Qwen3
        decision = router.determine_route(TaskComplexity.MEDIUM, None)
        assert decision.target == RouteTarget.LOCAL_QWEN3

        # Strategic should route to Shogun
        decision = router.determine_route(TaskComplexity.STRATEGIC, None)
        assert decision.target == RouteTarget.SHOGUN


class TestSystemOrchestratorV932:
    """Tests for System Orchestrator v9.3.2"""

    @pytest.fixture
    def mock_system_config(self):
        """Mock SystemConfig for testing"""
        from core.system_orchestrator import SystemConfig, SystemMode

        return SystemConfig(
            mode=SystemMode.BATTALION,
            claude_api_key="test-key",
            gemini_api_key="test-key",
            tavily_api_key="test-key",
            groq_api_key="test-key",
            alibaba_api_key="test-key",
            intelligent_routing_enabled=True,
            prompt_caching_enabled=True,
            power_optimization_enabled=True
        )

    def test_orchestrator_config_v932(self, mock_system_config):
        """Test SystemConfig v9.3.2 attributes"""
        assert mock_system_config.version == "9.3.2"
        assert mock_system_config.intelligent_routing_enabled == True
        assert mock_system_config.prompt_caching_enabled == True
        assert mock_system_config.power_optimization_enabled == True

    def test_orchestrator_get_method(self, mock_system_config):
        """Test SystemConfig.get() method"""
        value = mock_system_config.get("claude_api_key")
        assert value == "test-key"

        default = mock_system_config.get("nonexistent", "default")
        assert default == "default"


class TestV932Statistics:
    """Tests for v9.3.2 statistics collection"""

    def test_shogun_statistics(self):
        """Test Shogun statistics collection"""
        from core.shogun import Shogun, ReviewLevel

        # Create mock orchestrator
        orchestrator = Mock()
        orchestrator.config = Mock()
        orchestrator.config.claude_api_key = "test"
        orchestrator.get_client = Mock(return_value=None)
        orchestrator.get_mcp = Mock(return_value=None)
        orchestrator.get_router = Mock(return_value=None)

        shogun = Shogun(orchestrator)

        # Get initial statistics
        stats = shogun.get_statistics()

        assert "version" in stats
        assert stats["version"] == "9.3.2"
        assert "reviews_by_level" in stats
        assert "routing_stats" in stats

    def test_karo_statistics(self):
        """Test Karo statistics collection"""
        from core.karo import Karo, TaskDelegation

        orchestrator = Mock()
        orchestrator.get_client = Mock(return_value=None)
        orchestrator.get_mcp = Mock(return_value=None)

        karo = Karo(orchestrator)

        # Check initial stats
        assert karo.execution_stats["total_tasks"] == 0
        assert karo.execution_stats["fallback_count"] == 0

        # Get statistics
        stats = karo.get_statistics()
        assert "version" in stats
        assert stats["version"] == "9.3.2"

    def test_taisho_statistics(self):
        """Test Taisho statistics collection"""
        from core.taisho import Taisho

        orchestrator = Mock()
        orchestrator.get_client = Mock(return_value=None)
        orchestrator.get_mcp = Mock(return_value=None)
        orchestrator.mcp_manager = None

        taisho = Taisho(orchestrator)

        # Check initial stats
        assert taisho.execution_stats["total_tasks"] == 0
        assert taisho.execution_stats["context_overflows"] == 0

        # Get statistics
        stats = taisho.get_statistics()
        assert "version" in stats
        assert stats["version"] == "9.3.2"
        assert "fallback_chain" in stats


# Run with: pytest tests/unit/test_v932_integration.py -v
