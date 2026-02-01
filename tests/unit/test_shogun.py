"""
Unit Tests for Shogun (Strategic Layer)

Tests the core functionality of the Shogun class including:
- Task complexity assessment
- Strategic task handling
- Adaptive review system
- Quality assurance workflow
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from core.shogun import Shogun, Task, TaskComplexity, ReviewLevel


@pytest.mark.unit
class TestShogunInitialization:
    """Test Shogun initialization"""
    
    @pytest.mark.asyncio
    async def test_shogun_init(self, mock_orchestrator):
        """Test Shogun initializes correctly"""
        shogun = Shogun(mock_orchestrator)
        
        assert shogun.orchestrator == mock_orchestrator
        assert shogun.claude_client is not None
        assert shogun.quality_metrics is not None
        assert shogun.karo is None  # Not initialized yet
        assert shogun.opus_client is None  # Not initialized yet
    
    @pytest.mark.asyncio
    async def test_shogun_initialize(self, mock_orchestrator, mock_opus_client):
        """Test Shogun full initialization"""
        shogun = Shogun(mock_orchestrator)
        
        # Mock the Opus client creation
        with patch('core.shogun.OpusClient', return_value=mock_opus_client):
            # Mock Karo initialization
            with patch('core.shogun.Karo') as MockKaro:
                mock_karo_instance = Mock()
                mock_karo_instance.initialize = AsyncMock()
                MockKaro.return_value = mock_karo_instance
                
                await shogun.initialize()
                
                assert shogun.opus_client == mock_opus_client
                assert shogun.karo == mock_karo_instance
                assert mock_karo_instance.initialize.called


@pytest.mark.unit
class TestComplexityAssessment:
    """Test task complexity assessment logic"""
    
    @pytest.mark.asyncio
    async def test_assess_simple_task(self, mock_orchestrator, sample_task_simple):
        """Test assessment of simple task"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(return_value="SIMPLE")
        
        complexity = await shogun._assess_complexity(sample_task_simple)
        
        assert complexity == TaskComplexity.SIMPLE
        assert shogun.claude_client.generate.called
    
    @pytest.mark.asyncio
    async def test_assess_medium_task(self, mock_orchestrator, sample_task_medium):
        """Test assessment of medium task"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(return_value="MEDIUM")
        
        complexity = await shogun._assess_complexity(sample_task_medium)
        
        assert complexity == TaskComplexity.MEDIUM
    
    @pytest.mark.asyncio
    async def test_assess_complex_task(self, mock_orchestrator, sample_task_complex):
        """Test assessment of complex task"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(return_value="COMPLEX")
        
        complexity = await shogun._assess_complexity(sample_task_complex)
        
        assert complexity == TaskComplexity.COMPLEX
    
    @pytest.mark.asyncio
    async def test_assess_strategic_task(self, mock_orchestrator, sample_task_strategic):
        """Test assessment of strategic task"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(return_value="STRATEGIC")
        
        complexity = await shogun._assess_complexity(sample_task_strategic)
        
        assert complexity == TaskComplexity.STRATEGIC
    
    @pytest.mark.asyncio
    async def test_assess_complexity_failure_defaults_to_medium(self, mock_orchestrator, sample_task_simple):
        """Test that assessment failure defaults to MEDIUM"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(side_effect=Exception("API Error"))
        
        complexity = await shogun._assess_complexity(sample_task_simple)
        
        assert complexity == TaskComplexity.MEDIUM


@pytest.mark.unit
class TestStrategicTaskHandling:
    """Test strategic task handling"""
    
    @pytest.mark.asyncio
    async def test_handle_strategic_task(self, mock_orchestrator, sample_task_strategic):
        """Test strategic task is handled by Shogun directly"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(
            return_value="Strategic recommendation: Use microservices with event-driven architecture"
        )
        
        result = await shogun._handle_strategic_task(sample_task_strategic)
        
        assert result["status"] == "completed"
        assert result["complexity"] == "strategic"
        assert result["handled_by"] == "shogun"
        assert "Strategic recommendation" in result["result"]
        assert shogun.claude_client.generate.called
    
    @pytest.mark.asyncio
    async def test_strategic_task_considers_context(self, mock_orchestrator, sample_task_strategic):
        """Test strategic task uses provided context"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(return_value="Strategic analysis complete")
        
        result = await shogun._handle_strategic_task(sample_task_strategic)
        
        # Check that the prompt included context
        call_args = shogun.claude_client.generate.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "Context:" in prompt


@pytest.mark.unit
class TestAdaptiveReview:
    """Test adaptive review system"""
    
    def test_determine_review_level_strategic(self, mock_orchestrator, sample_task_strategic):
        """Test strategic tasks get premium review"""
        shogun = Shogun(mock_orchestrator)
        
        level = shogun._determine_review_level(sample_task_strategic, None)
        
        assert level == ReviewLevel.PREMIUM
    
    def test_determine_review_level_complex(self, mock_orchestrator, sample_task_complex):
        """Test complex tasks get detailed review"""
        shogun = Shogun(mock_orchestrator)
        
        level = shogun._determine_review_level(sample_task_complex, None)
        
        assert level == ReviewLevel.DETAILED
    
    def test_determine_review_level_medium(self, mock_orchestrator, sample_task_medium):
        """Test medium tasks get basic review"""
        shogun = Shogun(mock_orchestrator)
        
        level = shogun._determine_review_level(sample_task_medium, None)
        
        assert level == ReviewLevel.BASIC
    
    def test_determine_review_level_high_risk_upgrades(self, mock_orchestrator, sample_task_medium, mock_quality_metrics):
        """Test high risk code upgrades to premium review"""
        shogun = Shogun(mock_orchestrator)
        
        # Create quality report with HIGH risk
        from utils.quality_metrics import QualityReport, ComplexityMetrics, SecurityFindings, RiskLevel
        
        high_risk_report = QualityReport(
            task_id="test",
            complexity_metrics=ComplexityMetrics(
                lines_of_code=500,
                cyclomatic_complexity=25,
                cognitive_complexity=40,
                max_nesting_depth=8,
                average_function_length=50.0,
                complexity_score=85.0,
                risk_level=RiskLevel.HIGH
            ),
            security_findings=SecurityFindings(
                vulnerabilities=[],
                warnings=[],
                security_score=90.0,
                risk_level=RiskLevel.LOW
            ),
            overall_quality_score=75.0,
            recommendations=[]
        )
        
        level = shogun._determine_review_level(sample_task_medium, high_risk_report)
        
        assert level == ReviewLevel.PREMIUM
    
    def test_determine_review_level_security_vulnerabilities_upgrade(self, mock_orchestrator, sample_task_medium):
        """Test security vulnerabilities upgrade to premium review"""
        shogun = Shogun(mock_orchestrator)
        
        # Create quality report with vulnerabilities
        from utils.quality_metrics import QualityReport, ComplexityMetrics, SecurityFindings, RiskLevel
        
        vuln_report = QualityReport(
            task_id="test",
            complexity_metrics=ComplexityMetrics(
                lines_of_code=100,
                cyclomatic_complexity=5,
                cognitive_complexity=8,
                max_nesting_depth=2,
                average_function_length=15.0,
                complexity_score=30.0,
                risk_level=RiskLevel.LOW
            ),
            security_findings=SecurityFindings(
                vulnerabilities=["SQL injection risk", "Command injection"],
                warnings=[],
                security_score=40.0,
                risk_level=RiskLevel.HIGH
            ),
            overall_quality_score=50.0,
            recommendations=[]
        )
        
        level = shogun._determine_review_level(sample_task_medium, vuln_report)
        
        assert level == ReviewLevel.PREMIUM


@pytest.mark.unit
class TestReviewMethods:
    """Test individual review methods"""
    
    @pytest.mark.asyncio
    async def test_opus_premium_review(self, mock_orchestrator, sample_task_strategic, mock_opus_client):
        """Test Opus premium review execution"""
        shogun = Shogun(mock_orchestrator)
        shogun.opus_client = mock_opus_client
        
        result = {
            "status": "completed",
            "result": "Implementation details here"
        }
        
        reviewed_result = await shogun._opus_premium_review(sample_task_strategic, result, None)
        
        assert "opus_review" in reviewed_result
        assert reviewed_result["opus_review"]["score"] == 95.0
        assert reviewed_result["opus_review"]["decision"] == "approved"
        assert reviewed_result["shogun_approval"] == "approved"
        assert mock_opus_client.conduct_premium_review.called
    
    @pytest.mark.asyncio
    async def test_opus_review_fallback_on_error(self, mock_orchestrator, sample_task_strategic):
        """Test fallback to Sonnet when Opus fails"""
        shogun = Shogun(mock_orchestrator)
        shogun.opus_client = Mock()
        shogun.opus_client.conduct_premium_review = AsyncMock(
            side_effect=Exception("Opus API error")
        )
        shogun.claude_client.generate = AsyncMock(return_value="Score: 90/100\nAPPROVED")
        
        result = {
            "status": "completed",
            "result": "Implementation"
        }
        
        reviewed_result = await shogun._opus_premium_review(sample_task_strategic, result, None)
        
        # Should have fallen back to Sonnet detailed review
        assert "sonnet_detailed_review" in reviewed_result
        assert shogun.claude_client.generate.called
    
    @pytest.mark.asyncio
    async def test_sonnet_detailed_review(self, mock_orchestrator, sample_task_complex):
        """Test Sonnet detailed review"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(
            return_value="Score: 92/100\nAPPROVED\nExcellent implementation with good error handling"
        )
        
        result = {
            "status": "completed",
            "result": "def complex_function(): pass"
        }
        
        reviewed_result = await shogun._sonnet_detailed_review(sample_task_complex, result, None)
        
        assert "sonnet_detailed_review" in reviewed_result
        assert reviewed_result["sonnet_detailed_review"]["score"] == 92.0
        assert reviewed_result["shogun_approval"] == "approved"
    
    @pytest.mark.asyncio
    async def test_sonnet_basic_review_approved(self, mock_orchestrator, sample_task_simple):
        """Test Sonnet basic review - approved"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(return_value="APPROVED - looks good")
        
        result = {
            "status": "completed",
            "result": "The date is 2025-01-31"
        }
        
        reviewed_result = await shogun._sonnet_basic_review(sample_task_simple, result)
        
        assert reviewed_result["shogun_approval"] == "approved"
    
    @pytest.mark.asyncio
    async def test_sonnet_basic_review_with_feedback(self, mock_orchestrator, sample_task_simple):
        """Test Sonnet basic review - feedback provided"""
        shogun = Shogun(mock_orchestrator)
        shogun.claude_client.generate = AsyncMock(
            return_value="Consider adding timezone information"
        )
        
        result = {
            "status": "completed",
            "result": "The date is 2025-01-31"
        }
        
        reviewed_result = await shogun._sonnet_basic_review(sample_task_simple, result)
        
        assert reviewed_result["shogun_approval"] == "feedback_provided"
        assert "shogun_feedback" in reviewed_result


@pytest.mark.unit
class TestProcessTask:
    """Test main task processing pipeline"""
    
    @pytest.mark.asyncio
    async def test_process_simple_task_end_to_end(self, mock_orchestrator, sample_task_simple):
        """Test complete processing of simple task"""
        shogun = Shogun(mock_orchestrator)
        
        # Mock all required components
        shogun.claude_client.generate = AsyncMock(return_value="SIMPLE")
        mock_karo = Mock()
        mock_karo.execute_task = AsyncMock(return_value={
            "status": "completed",
            "result": "The current date is 2025-01-31"
        })
        shogun.karo = mock_karo
        shogun._adaptive_review = AsyncMock(return_value={
            "status": "completed",
            "result": "The current date is 2025-01-31",
            "shogun_approval": "approved"
        })
        shogun._log_decision = AsyncMock()
        
        result = await shogun.process_task(sample_task_simple)
        
        assert result["status"] == "completed"
        assert result["shogun_approval"] == "approved"
        assert mock_karo.execute_task.called
    
    @pytest.mark.asyncio
    async def test_process_strategic_task_handled_by_shogun(self, mock_orchestrator, sample_task_strategic):
        """Test strategic task is handled by Shogun, not delegated"""
        shogun = Shogun(mock_orchestrator)
        
        # Mock assessment to return STRATEGIC
        shogun._assess_complexity = AsyncMock(return_value=TaskComplexity.STRATEGIC)
        shogun._handle_strategic_task = AsyncMock(return_value={
            "status": "completed",
            "result": "Strategic analysis complete",
            "complexity": "strategic",
            "handled_by": "shogun"
        })
        shogun._log_decision = AsyncMock()
        
        result = await shogun.process_task(sample_task_strategic)
        
        assert result["handled_by"] == "shogun"
        assert shogun._handle_strategic_task.called
    
    @pytest.mark.asyncio
    async def test_process_task_error_handling(self, mock_orchestrator, sample_task_medium):
        """Test error handling in task processing"""
        shogun = Shogun(mock_orchestrator)
        shogun._assess_complexity = AsyncMock(side_effect=Exception("Critical error"))
        
        result = await shogun.process_task(sample_task_medium)
        
        assert "error" in result
        assert result["status"] == "failed"


@pytest.mark.unit
class TestStatistics:
    """Test statistics and monitoring"""
    
    def test_get_review_statistics(self, mock_orchestrator, mock_opus_client):
        """Test retrieval of review statistics"""
        shogun = Shogun(mock_orchestrator)
        shogun.opus_client = mock_opus_client
        
        # Simulate some reviews
        shogun.reviews_by_level[ReviewLevel.BASIC] = 10
        shogun.reviews_by_level[ReviewLevel.DETAILED] = 5
        shogun.reviews_by_level[ReviewLevel.PREMIUM] = 2
        
        stats = shogun.get_review_statistics()
        
        assert stats["total_reviews"] == 17
        assert stats["reviews_by_level"]["basic"] == 10
        assert stats["reviews_by_level"]["detailed"] == 5
        assert stats["reviews_by_level"]["premium"] == 2
        assert "opus_statistics" in stats


@pytest.mark.integration
class TestShogunIntegration:
    """Integration tests for Shogun with real components"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_workflow_with_quality_metrics(self, mock_orchestrator, sample_task_medium, sample_code_implementation):
        """Test full workflow including quality metrics collection"""
        shogun = Shogun(mock_orchestrator)
        
        # Setup with quality metrics
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
        
        # Quality metrics should be collected
        assert "quality_metrics" in result
        assert "complexity_score" in result["quality_metrics"]
        assert "security_score" in result["quality_metrics"]
