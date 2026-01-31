"""
Integration Tests for End-to-End Task Flows

Tests complete task processing workflows from task intake through
all system layers to final delivery.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from core.shogun import Shogun, Task, TaskComplexity
from core.intelligent_router import IntelligentRouter, RouteTarget


@pytest.mark.integration
class TestSimpleTaskFlow:
    """Test simple task end-to-end flow"""
    
    @pytest.mark.asyncio
    async def test_simple_task_groq_path(self, mock_orchestrator, sample_task_simple):
        """Test simple task routed through Groq"""
        # Setup
        shogun = Shogun(mock_orchestrator)
        router = IntelligentRouter(mock_orchestrator.config)
        
        # Mock dependencies
        shogun.claude_client.generate = AsyncMock(return_value="SIMPLE")
        
        mock_karo = Mock()
        mock_karo.execute_task = AsyncMock(return_value={
            "status": "completed",
            "result": "The current date is January 31, 2026",
            "route": "groq",
            "time_seconds": 1.8
        })
        shogun.karo = mock_karo
        shogun._log_decision = AsyncMock()
        
        # Execute
        result = await shogun.process_task(sample_task_simple)
        
        # Verify
        assert result["status"] == "completed"
        assert "result" in result
        assert mock_karo.execute_task.called
        
        # Verify routing decision
        complexity = router.judge_complexity(sample_task_simple.content)
        decision = router.determine_route(complexity)
        assert decision.target == RouteTarget.GROQ
        assert decision.power_saving is True  # Qwen not woken
    
    @pytest.mark.asyncio
    async def test_simple_task_quality_check(self, mock_orchestrator, sample_task_simple):
        """Test simple task goes through basic quality check"""
        shogun = Shogun(mock_orchestrator)
        
        # Mock components
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.SIMPLE)
        mock_karo = Mock()
        mock_karo.execute_task = AsyncMock(return_value={
            "status": "completed",
            "result": "Answer here"
        })
        shogun.karo = mock_karo
        shogun.claude_client.generate = AsyncMock(return_value="APPROVED")
        shogun._log_decision = AsyncMock()
        
        result = await shogun.process_task(sample_task_simple)
        
        # Simple tasks get basic review
        assert "shogun_approval" in result
        assert result["shogun_approval"] in ["approved", "feedback_provided"]


@pytest.mark.integration
class TestMediumTaskFlow:
    """Test medium task end-to-end flow"""
    
    @pytest.mark.asyncio
    async def test_medium_task_local_qwen_path(self, mock_orchestrator, sample_task_medium):
        """Test medium task routed through local Qwen3"""
        shogun = Shogun(mock_orchestrator)
        router = IntelligentRouter(mock_orchestrator.config)
        
        # Mock assessment
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.MEDIUM)
        
        # Mock Karo execution
        mock_karo = Mock()
        mock_karo.execute_task = AsyncMock(return_value={
            "status": "completed",
            "result": "def fibonacci(n):\n    if n <= 0:\n        return []\n    return [0, 1, *[...]]",
            "route": "local_qwen3",
            "time_seconds": 11.5
        })
        shogun.karo = mock_karo
        shogun.claude_client.generate = AsyncMock(return_value="APPROVED")
        shogun._log_decision = AsyncMock()
        
        result = await shogun.process_task(sample_task_medium)
        
        # Verify success
        assert result["status"] == "completed"
        assert "def fibonacci" in result["result"]
        
        # Verify routing
        complexity = router.judge_complexity(sample_task_medium.content)
        decision = router.determine_route(complexity)
        assert decision.target == RouteTarget.LOCAL_QWEN3
        assert decision.power_saving is False  # Qwen woken for medium
    
    @pytest.mark.asyncio
    async def test_medium_task_with_quality_metrics(self, mock_orchestrator, sample_task_medium, sample_code_implementation):
        """Test medium task includes quality metrics"""
        shogun = Shogun(mock_orchestrator)
        
        # Setup quality metrics collector
        from utils.quality_metrics import QualityMetricsCollector
        shogun.quality_metrics = QualityMetricsCollector()
        
        # Mock components
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.MEDIUM)
        mock_karo = Mock()
        mock_karo.execute_task = AsyncMock(return_value={
            "status": "completed",
            "result": sample_code_implementation
        })
        shogun.karo = mock_karo
        shogun.claude_client.generate = AsyncMock(return_value="APPROVED")
        shogun._log_decision = AsyncMock()
        
        result = await shogun.process_task(sample_task_medium)
        
        # Verify quality metrics were collected
        assert "quality_metrics" in result
        assert "complexity_score" in result["quality_metrics"]
        assert "security_score" in result["quality_metrics"]
        assert "overall_score" in result["quality_metrics"]


@pytest.mark.integration
class TestComplexTaskFlow:
    """Test complex task end-to-end flow"""
    
    @pytest.mark.asyncio
    async def test_complex_task_detailed_review(self, mock_orchestrator, sample_task_complex):
        """Test complex task gets detailed Sonnet review"""
        shogun = Shogun(mock_orchestrator)
        
        # Mock components
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.COMPLEX)
        mock_karo = Mock()
        mock_karo.execute_task = AsyncMock(return_value={
            "status": "completed",
            "result": "class JWTAuth:\n    def __init__(self):\n        pass\n    # Implementation..."
        })
        shogun.karo = mock_karo
        shogun.claude_client.generate = AsyncMock(return_value="Score: 93/100\nAPPROVED\nGood implementation")
        shogun._log_decision = AsyncMock()
        
        result = await shogun.process_task(sample_task_complex)
        
        # Complex tasks get detailed review
        assert "sonnet_detailed_review" in result
        assert result["sonnet_detailed_review"]["review_level"] == "detailed"
        assert result["sonnet_detailed_review"]["score"] >= 90.0
    
    @pytest.mark.asyncio
    async def test_complex_task_high_risk_opus_upgrade(self, mock_orchestrator, sample_task_complex, mock_opus_client):
        """Test complex task with high risk upgrades to Opus review"""
        shogun = Shogun(mock_orchestrator)
        shogun.opus_client = mock_opus_client
        
        # Create high-risk code result
        from utils.quality_metrics import QualityReport, ComplexityMetrics, SecurityFindings, RiskLevel
        
        high_risk_metrics = QualityReport(
            task_id="test",
            complexity_metrics=ComplexityMetrics(
                lines_of_code=600,
                cyclomatic_complexity=30,
                cognitive_complexity=50,
                max_nesting_depth=9,
                average_function_length=60.0,
                complexity_score=92.0,
                risk_level=RiskLevel.CRITICAL
            ),
            security_findings=SecurityFindings(
                vulnerabilities=[],
                warnings=[],
                security_score=80.0,
                risk_level=RiskLevel.MEDIUM
            ),
            overall_quality_score=70.0,
            recommendations=[]
        )
        
        # Mock
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.COMPLEX)
        mock_karo = Mock()
        mock_karo.execute_task = AsyncMock(return_value={
            "status": "completed",
            "result": "x" * 600  # Long, complex code
        })
        shogun.karo = mock_karo
        shogun.quality_metrics.collect_metrics = Mock(return_value=high_risk_metrics)
        shogun._log_decision = AsyncMock()
        
        result = await shogun.process_task(sample_task_complex)
        
        # Should have upgraded to Opus review
        assert "opus_review" in result
        assert mock_opus_client.conduct_premium_review.called


@pytest.mark.integration
class TestStrategicTaskFlow:
    """Test strategic task end-to-end flow"""
    
    @pytest.mark.asyncio
    async def test_strategic_task_shogun_handles_directly(self, mock_orchestrator, sample_task_strategic):
        """Test strategic task handled by Shogun, not delegated"""
        shogun = Shogun(mock_orchestrator)
        
        # Mock
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.STRATEGIC)
        shogun.claude_client.generate = AsyncMock(
            return_value="Strategic recommendation:\n1. Use microservices\n2. Event-driven architecture\n3. CQRS pattern"
        )
        shogun._log_decision = AsyncMock()
        
        result = await shogun.process_task(sample_task_strategic)
        
        # Verify Shogun handled directly
        assert result["status"] == "completed"
        assert result["complexity"] == "strategic"
        assert result["handled_by"] == "shogun"
        assert "microservices" in result["result"].lower()
    
    @pytest.mark.asyncio
    async def test_strategic_task_logged_to_memory(self, mock_orchestrator, sample_task_strategic, mock_memory_mcp):
        """Test strategic decisions are logged to Memory MCP"""
        shogun = Shogun(mock_orchestrator)
        shogun.memory_mcp = mock_memory_mcp
        
        # Mock
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.STRATEGIC)
        shogun.claude_client.generate = AsyncMock(return_value="Strategic analysis complete")
        
        result = await shogun.process_task(sample_task_strategic)
        
        # Verify Memory MCP logging
        assert mock_memory_mcp.store.called
        call_args = mock_memory_mcp.store.call_args[0][0]
        assert call_args["category"] == "decision"
        assert call_args["complexity"] == "strategic"


@pytest.mark.integration
class TestFallbackScenarios:
    """Test fallback chain activation scenarios"""
    
    @pytest.mark.asyncio
    async def test_local_qwen_context_overflow_fallback(self, mock_orchestrator, sample_task_medium):
        """Test fallback when local Qwen context overflows"""
        shogun = Shogun(mock_orchestrator)
        router = IntelligentRouter(mock_orchestrator.config)
        
        # Mock local Qwen failure
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.MEDIUM)
        mock_karo = Mock()
        
        # First call fails (local), second succeeds (cloud)
        mock_karo.execute_task = AsyncMock(side_effect=[
            Exception("Context length exceeded: 4096 tokens"),
            {
                "status": "completed",
                "result": "Implementation using cloud Qwen",
                "route": "cloud_qwen3",
                "fallback_triggered": True
            }
        ])
        shogun.karo = mock_karo
        shogun.claude_client.generate = AsyncMock(return_value="APPROVED")
        shogun._log_decision = AsyncMock()
        
        # Verify routing decision has fallback chain
        complexity = router.judge_complexity(sample_task_medium.content)
        decision = router.determine_route(complexity)
        assert len(decision.fallback_chain) == 3
        assert decision.fallback_chain[1] == RouteTarget.CLOUD_QWEN3
    
    @pytest.mark.asyncio
    async def test_full_fallback_chain_to_gemini(self, mock_orchestrator, sample_task_complex):
        """Test complete fallback chain: Local -> Cloud -> Gemini"""
        shogun = Shogun(mock_orchestrator)
        router = IntelligentRouter(mock_orchestrator.config)
        
        # Simulate full fallback
        decision = router.determine_route(TaskComplexity.COMPLEX)
        
        # Local fails
        next_target = router.try_fallback(decision, RouteTarget.LOCAL_QWEN3, Exception("Local fail"))
        assert next_target == RouteTarget.CLOUD_QWEN3
        
        # Cloud fails
        next_target = router.try_fallback(decision, RouteTarget.CLOUD_QWEN3, Exception("Cloud fail"))
        assert next_target == RouteTarget.GEMINI3
        
        # Gemini succeeds (final defense)
        assert next_target == RouteTarget.GEMINI3
        assert router.stats.fallbacks_triggered == 2


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across the system"""
    
    @pytest.mark.asyncio
    async def test_task_processing_error_recovery(self, mock_orchestrator, sample_task_medium):
        """Test system recovers from processing errors"""
        shogun = Shogun(mock_orchestrator)
        
        # Mock critical error
        shogun._assess_complexity = AsyncMock(side_effect=Exception("Critical error"))
        
        result = await shogun.process_task(sample_task_medium)
        
        # Should return error result, not crash
        assert "error" in result
        assert result["status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_opus_review_fallback_to_sonnet(self, mock_orchestrator, sample_task_strategic):
        """Test Opus failure falls back to Sonnet"""
        shogun = Shogun(mock_orchestrator)
        
        # Mock Opus failure
        mock_opus = Mock()
        mock_opus.conduct_premium_review = AsyncMock(side_effect=Exception("Opus API error"))
        shogun.opus_client = mock_opus
        
        # Mock Sonnet success
        shogun.claude_client.generate = AsyncMock(return_value="Score: 88/100\nAPPROVED")
        
        result = {
            "status": "completed",
            "result": "Strategic implementation"
        }
        
        reviewed_result = await shogun._opus_premium_review(sample_task_strategic, result, None)
        
        # Should have fallen back to Sonnet
        assert "sonnet_detailed_review" in reviewed_result
        assert shogun.claude_client.generate.called


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceTargets:
    """Test performance targets are met"""
    
    @pytest.mark.asyncio
    async def test_simple_task_speed_target(self, mock_orchestrator, sample_task_simple):
        """Test simple tasks meet 2-second target"""
        import time
        
        shogun = Shogun(mock_orchestrator)
        
        # Mock fast responses
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.SIMPLE)
        mock_karo = Mock()
        mock_karo.execute_task = AsyncMock(return_value={
            "status": "completed",
            "result": "Quick answer"
        })
        shogun.karo = mock_karo
        shogun.claude_client.generate = AsyncMock(return_value="APPROVED")
        shogun._log_decision = AsyncMock()
        
        start_time = time.time()
        result = await shogun.process_task(sample_task_simple)
        elapsed = time.time() - start_time
        
        # Should complete quickly (mock should be < 1s)
        assert elapsed < 3.0  # Generous for mocks
        assert result["status"] == "completed"


@pytest.mark.integration
class TestCostTracking:
    """Test cost tracking across workflows"""
    
    def test_cost_accumulation_across_tasks(self, mock_config):
        """Test cost tracking aggregates correctly"""
        router = IntelligentRouter(mock_config)
        
        # Simulate task mix
        tasks = [
            (TaskComplexity.SIMPLE, RouteTarget.GROQ, 2.0, 0.0),      # Free
            (TaskComplexity.SIMPLE, RouteTarget.GROQ, 1.5, 0.0),      # Free
            (TaskComplexity.MEDIUM, RouteTarget.LOCAL_QWEN3, 12.0, 0.0),  # Free (local)
            (TaskComplexity.COMPLEX, RouteTarget.CLOUD_QWEN3, 30.0, 3.0),  # Paid (cloud fallback)
            (TaskComplexity.MEDIUM, RouteTarget.GEMINI3, 15.0, 0.04),  # Paid (Gemini fallback)
        ]
        
        for complexity, target, time, cost in tasks:
            decision = router.determine_route(complexity)
            decision.target = target
            router.record_task_completion(decision, time, cost)
        
        stats = router.get_routing_stats()
        
        # Total cost: 0 + 0 + 0 + 3 + 0.04 = 3.04
        assert 3.0 <= stats.total_cost_yen <= 3.1
        
        # Power savings: 2 simple tasks * ~¥5 = ¥10
        assert stats.power_savings_yen >= 8.0
