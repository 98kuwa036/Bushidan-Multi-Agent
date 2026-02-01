"""
Unit Tests for IntelligentRouter

Tests the intelligent routing logic for v9.3.2 including:
- Complexity judgment heuristics
- Route determination
- Fallback chain management
- Power-saving optimization
- Performance tracking
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from core.intelligent_router import (
    IntelligentRouter,
    TaskComplexity,
    RouteTarget,
    RoutingDecision,
    RoutingStats
)


@pytest.mark.unit
class TestComplexityJudgment:
    """Test complexity judgment heuristics"""
    
    def test_judge_simple_short_query(self, mock_config):
        """Test simple task: short query"""
        router = IntelligentRouter(mock_config)
        
        task = "What is the time?"
        complexity = router.judge_complexity(task)
        
        assert complexity == TaskComplexity.SIMPLE
    
    def test_judge_simple_question(self, mock_config):
        """Test simple task: question"""
        router = IntelligentRouter(mock_config)
        
        task = "How do I install Python?"
        complexity = router.judge_complexity(task)
        
        assert complexity == TaskComplexity.SIMPLE
    
    def test_judge_medium_implementation(self, mock_config):
        """Test medium task: implementation with code keywords"""
        router = IntelligentRouter(mock_config)
        
        task = "Implement a function to calculate the sum of an array of numbers in Python"
        complexity = router.judge_complexity(task)
        
        assert complexity == TaskComplexity.MEDIUM
    
    def test_judge_medium_short_code_task(self, mock_config):
        """Test medium/simple boundary: short code task"""
        router = IntelligentRouter(mock_config)
        
        task = "Create a hello world function"
        complexity = router.judge_complexity(task)
        
        # Should be SIMPLE due to short length (< 50 chars after lowercase)
        assert complexity == TaskComplexity.SIMPLE
    
    def test_judge_complex_multi_component(self, mock_config):
        """Test complex task: multi-component system"""
        router = IntelligentRouter(mock_config)
        
        task = "Refactor the authentication system across multiple files including middleware, models, and controllers"
        complexity = router.judge_complexity(task)
        
        assert complexity == TaskComplexity.COMPLEX
    
    def test_judge_complex_long_task(self, mock_config):
        """Test complex task: very long description"""
        router = IntelligentRouter(mock_config)
        
        task = "x" * 350  # Very long task
        complexity = router.judge_complexity(task)
        
        assert complexity == TaskComplexity.COMPLEX
    
    def test_judge_strategic_architecture(self, mock_config):
        """Test strategic task: architecture decision"""
        router = IntelligentRouter(mock_config)
        
        task = "Design the overall architecture for a new microservices platform"
        complexity = router.judge_complexity(task)
        
        assert complexity == TaskComplexity.STRATEGIC
    
    def test_judge_strategic_technology_choice(self, mock_config):
        """Test strategic task: technology selection"""
        router = IntelligentRouter(mock_config)
        
        task = "Recommend the best technology stack for our long-term product strategy"
        complexity = router.judge_complexity(task)
        
        assert complexity == TaskComplexity.STRATEGIC
    
    def test_judge_strategic_japanese(self, mock_config):
        """Test strategic task: Japanese keywords"""
        router = IntelligentRouter(mock_config)
        
        task = "新しいシステム全体の設計を考えてください"
        complexity = router.judge_complexity(task)
        
        assert complexity == TaskComplexity.STRATEGIC


@pytest.mark.unit
class TestRouteDetermination:
    """Test route determination logic"""
    
    def test_route_simple_to_groq(self, mock_config):
        """Test simple tasks route to Groq"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.SIMPLE)
        
        assert decision.target == RouteTarget.GROQ
        assert decision.complexity == TaskComplexity.SIMPLE
        assert decision.power_saving is True  # Don't wake Qwen
        assert decision.estimated_cost_yen == 0.0
        assert decision.estimated_time_seconds == 2
    
    def test_route_simple_groq_no_fallback(self, mock_config):
        """Test Groq has no fallback chain (it's the fallback)"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.SIMPLE)
        
        assert len(decision.fallback_chain) == 1
        assert decision.fallback_chain[0] == RouteTarget.GROQ
    
    def test_route_medium_to_local_qwen(self, mock_config):
        """Test medium tasks route to local Qwen3"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.MEDIUM)
        
        assert decision.target == RouteTarget.LOCAL_QWEN3
        assert decision.complexity == TaskComplexity.MEDIUM
        assert decision.power_saving is False  # Wake Qwen
        assert decision.estimated_cost_yen == 0.0  # Local is free
        assert decision.estimated_time_seconds == 12
    
    def test_route_medium_has_fallback_chain(self, mock_config):
        """Test medium tasks have 3-tier fallback"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.MEDIUM)
        
        assert len(decision.fallback_chain) == 3
        assert decision.fallback_chain[0] == RouteTarget.LOCAL_QWEN3
        assert decision.fallback_chain[1] == RouteTarget.CLOUD_QWEN3
        assert decision.fallback_chain[2] == RouteTarget.GEMINI3
    
    def test_route_complex_to_local_qwen(self, mock_config):
        """Test complex tasks also route to local Qwen3"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.COMPLEX)
        
        assert decision.target == RouteTarget.LOCAL_QWEN3
        assert decision.complexity == TaskComplexity.COMPLEX
        assert decision.estimated_time_seconds == 28
    
    def test_route_complex_has_same_fallback(self, mock_config):
        """Test complex tasks have same 3-tier fallback"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.COMPLEX)
        
        assert len(decision.fallback_chain) == 3
        assert decision.fallback_chain == [
            RouteTarget.LOCAL_QWEN3,
            RouteTarget.CLOUD_QWEN3,
            RouteTarget.GEMINI3
        ]
    
    def test_route_strategic_to_shogun(self, mock_config):
        """Test strategic tasks route to Shogun"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.STRATEGIC)
        
        assert decision.target == RouteTarget.SHOGUN
        assert decision.complexity == TaskComplexity.STRATEGIC
        assert decision.estimated_time_seconds == 45
        assert len(decision.fallback_chain) == 1  # No fallback for strategic
        assert decision.fallback_chain[0] == RouteTarget.SHOGUN


@pytest.mark.unit
class TestFallbackLogic:
    """Test fallback chain activation"""
    
    def test_try_fallback_success(self, mock_config):
        """Test successful fallback to next target"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.MEDIUM)
        current_target = RouteTarget.LOCAL_QWEN3
        error = Exception("Local Qwen3 failed")
        
        next_target = router.try_fallback(decision, current_target, error)
        
        assert next_target == RouteTarget.CLOUD_QWEN3
        assert router.stats.fallbacks_triggered == 1
    
    def test_try_fallback_chain(self, mock_config):
        """Test full fallback chain"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.MEDIUM)
        
        # First fallback: Local -> Cloud
        next1 = router.try_fallback(decision, RouteTarget.LOCAL_QWEN3, Exception())
        assert next1 == RouteTarget.CLOUD_QWEN3
        
        # Second fallback: Cloud -> Gemini
        next2 = router.try_fallback(decision, RouteTarget.CLOUD_QWEN3, Exception())
        assert next2 == RouteTarget.GEMINI3
        
        # No more fallbacks
        next3 = router.try_fallback(decision, RouteTarget.GEMINI3, Exception())
        assert next3 is None
        
        assert router.stats.fallbacks_triggered == 3
    
    def test_fallback_updates_cost(self, mock_config):
        """Test fallback updates cost estimates"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.MEDIUM)
        
        # Fallback to cloud (costs money)
        next_target = router.try_fallback(decision, RouteTarget.LOCAL_QWEN3, Exception())
        
        assert next_target == RouteTarget.CLOUD_QWEN3
        # Cost should be updated in stats when task completes


@pytest.mark.unit
class TestPowerSavingLogic:
    """Test power-saving optimization"""
    
    def test_simple_tasks_dont_wake_qwen(self, mock_config):
        """Test simple tasks don't wake local Qwen (power saving)"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.SIMPLE)
        
        assert decision.power_saving is True
        assert decision.target != RouteTarget.LOCAL_QWEN3
    
    def test_medium_tasks_wake_qwen(self, mock_config):
        """Test medium tasks wake local Qwen"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.MEDIUM)
        
        assert decision.power_saving is False
        assert decision.target == RouteTarget.LOCAL_QWEN3
    
    def test_power_saving_cost_calculation(self, mock_config):
        """Test power savings are tracked"""
        router = IntelligentRouter(mock_config)
        
        # Process 10 simple tasks (Groq, power-saving)
        for _ in range(10):
            decision = router.determine_route(TaskComplexity.SIMPLE)
            router.record_task_completion(decision, 2.0, 0.0)
        
        stats = router.get_routing_stats()
        
        # Each simple task saves ~¥5 of local power
        assert stats.power_savings_yen >= 40.0  # At least ¥4/task * 10


@pytest.mark.unit
class TestStatistics:
    """Test statistics tracking"""
    
    def test_record_task_completion(self, mock_config):
        """Test task completion recording"""
        router = IntelligentRouter(mock_config)
        
        decision = router.determine_route(TaskComplexity.MEDIUM)
        router.record_task_completion(decision, actual_time=15.0, actual_cost=0.0)
        
        stats = router.get_routing_stats()
        
        assert stats.total_tasks == 1
        assert stats.routes_by_target[RouteTarget.LOCAL_QWEN3.value] == 1
        assert stats.average_time_seconds == 15.0
        assert stats.total_cost_yen == 0.0
    
    def test_statistics_aggregation(self, mock_config):
        """Test statistics aggregate correctly"""
        router = IntelligentRouter(mock_config)
        
        # Simulate various tasks
        tasks = [
            (TaskComplexity.SIMPLE, 2.0, 0.0),      # Groq
            (TaskComplexity.SIMPLE, 1.5, 0.0),      # Groq
            (TaskComplexity.MEDIUM, 12.0, 0.0),     # Local Qwen
            (TaskComplexity.COMPLEX, 30.0, 3.0),    # Cloud Qwen (fallback)
            (TaskComplexity.STRATEGIC, 45.0, 0.0),  # Shogun
        ]
        
        for complexity, time, cost in tasks:
            decision = router.determine_route(complexity)
            # Simulate fallback for complex task
            if complexity == TaskComplexity.COMPLEX:
                decision.target = RouteTarget.CLOUD_QWEN3
            router.record_task_completion(decision, time, cost)
        
        stats = router.get_routing_stats()
        
        assert stats.total_tasks == 5
        assert stats.routes_by_target[RouteTarget.GROQ.value] == 2
        assert stats.routes_by_target[RouteTarget.LOCAL_QWEN3.value] == 1
        assert stats.routes_by_target[RouteTarget.CLOUD_QWEN3.value] == 1
        assert stats.routes_by_target[RouteTarget.SHOGUN.value] == 1
        assert stats.total_cost_yen == 3.0
        
        # Average time: (2 + 1.5 + 12 + 30 + 45) / 5 = 18.1
        assert 18.0 <= stats.average_time_seconds <= 18.2
    
    def test_routing_history_tracking(self, mock_config):
        """Test routing history is maintained"""
        router = IntelligentRouter(mock_config)
        
        decision1 = router.determine_route(TaskComplexity.SIMPLE)
        router.record_task_completion(decision1, 2.0, 0.0)
        
        decision2 = router.determine_route(TaskComplexity.MEDIUM)
        router.record_task_completion(decision2, 12.0, 0.0)
        
        assert len(router.routing_history) == 2
        assert all(isinstance(entry[0], datetime) for entry in router.routing_history)
        assert all(isinstance(entry[1], RoutingDecision) for entry in router.routing_history)


@pytest.mark.unit
class TestCostEstimates:
    """Test cost estimation logic"""
    
    def test_cost_estimates_by_target(self, mock_config):
        """Test cost estimates match expectations"""
        router = IntelligentRouter(mock_config)
        
        assert router.cost_estimates[RouteTarget.GROQ] == 0.0
        assert router.cost_estimates[RouteTarget.LOCAL_QWEN3] == 0.0
        assert router.cost_estimates[RouteTarget.CLOUD_QWEN3] == 3.0
        assert router.cost_estimates[RouteTarget.GEMINI3] == 0.04
        assert router.cost_estimates[RouteTarget.SHOGUN] == 0.0
    
    def test_cost_accumulation(self, mock_config):
        """Test cost accumulation over multiple tasks"""
        router = IntelligentRouter(mock_config)
        
        # 10 Groq tasks: ¥0
        for _ in range(10):
            decision = router.determine_route(TaskComplexity.SIMPLE)
            router.record_task_completion(decision, 2.0, 0.0)
        
        # 5 Cloud Qwen tasks: ¥15
        for _ in range(5):
            decision = router.determine_route(TaskComplexity.MEDIUM)
            decision.target = RouteTarget.CLOUD_QWEN3
            router.record_task_completion(decision, 12.0, 3.0)
        
        # 2 Gemini tasks: ¥0.08
        for _ in range(2):
            decision = router.determine_route(TaskComplexity.COMPLEX)
            decision.target = RouteTarget.GEMINI3
            router.record_task_completion(decision, 28.0, 0.04)
        
        stats = router.get_routing_stats()
        
        # Total cost: 0 + 15 + 0.08 = 15.08
        assert 15.0 <= stats.total_cost_yen <= 15.1


@pytest.mark.unit
class TestPerformanceTargets:
    """Test performance target adherence"""
    
    def test_time_targets_by_complexity(self, mock_config):
        """Test time targets match specifications"""
        router = IntelligentRouter(mock_config)
        
        assert router.target_times[TaskComplexity.SIMPLE] == 2
        assert router.target_times[TaskComplexity.MEDIUM] == 12
        assert router.target_times[TaskComplexity.COMPLEX] == 28
        assert router.target_times[TaskComplexity.STRATEGIC] == 45
    
    def test_estimated_time_in_decisions(self, mock_config):
        """Test decisions include estimated times"""
        router = IntelligentRouter(mock_config)
        
        simple_decision = router.determine_route(TaskComplexity.SIMPLE)
        assert simple_decision.estimated_time_seconds == 2
        
        medium_decision = router.determine_route(TaskComplexity.MEDIUM)
        assert medium_decision.estimated_time_seconds == 12
        
        complex_decision = router.determine_route(TaskComplexity.COMPLEX)
        assert complex_decision.estimated_time_seconds == 28
        
        strategic_decision = router.determine_route(TaskComplexity.STRATEGIC)
        assert strategic_decision.estimated_time_seconds == 45


@pytest.mark.integration
class TestRouterIntegration:
    """Integration tests for router with real scenarios"""
    
    def test_full_routing_workflow(self, mock_config):
        """Test complete routing workflow"""
        router = IntelligentRouter(mock_config)
        
        # User submits task
        task_content = "Implement a REST API for user management with authentication"
        
        # Judge complexity
        complexity = router.judge_complexity(task_content)
        assert complexity == TaskComplexity.MEDIUM  # Has "implement" keyword
        
        # Determine route
        decision = router.determine_route(complexity)
        assert decision.target == RouteTarget.LOCAL_QWEN3
        assert decision.fallback_chain == [
            RouteTarget.LOCAL_QWEN3,
            RouteTarget.CLOUD_QWEN3,
            RouteTarget.GEMINI3
        ]
        
        # Record completion
        router.record_task_completion(decision, 15.0, 0.0)
        
        # Check stats
        stats = router.get_routing_stats()
        assert stats.total_tasks == 1
        assert stats.average_time_seconds == 15.0
    
    def test_fallback_scenario(self, mock_config):
        """Test realistic fallback scenario"""
        router = IntelligentRouter(mock_config)
        
        # Medium task routes to local Qwen
        decision = router.determine_route(TaskComplexity.MEDIUM)
        assert decision.target == RouteTarget.LOCAL_QWEN3
        
        # Local Qwen fails (e.g., context overflow)
        next_target = router.try_fallback(
            decision,
            RouteTarget.LOCAL_QWEN3,
            Exception("Context length exceeded")
        )
        assert next_target == RouteTarget.CLOUD_QWEN3
        
        # Cloud Qwen succeeds
        decision.target = next_target
        router.record_task_completion(decision, 18.0, 3.0)
        
        # Verify stats
        stats = router.get_routing_stats()
        assert stats.fallbacks_triggered == 1
        assert stats.total_cost_yen == 3.0
        assert stats.routes_by_target[RouteTarget.CLOUD_QWEN3.value] == 1
