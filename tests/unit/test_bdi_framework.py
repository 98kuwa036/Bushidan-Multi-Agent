"""
Unit tests for BDI Framework

Tests the core BDI (Belief-Desire-Intention) framework components including:
- Belief, Desire, Intention data structures
- BeliefBase, DesireSet, IntentionStack management
- BDIAgent abstract base class
"""

import pytest
import asyncio
from datetime import datetime

from core.bdi_framework import (
    Belief, Desire, Intention,
    BeliefType, DesireType,
    BeliefBase, DesireSet, IntentionStack,
    BDIAgent
)


# Test Belief data structure
def test_belief_creation():
    """Test basic Belief creation"""
    belief = Belief(
        id="test_belief",
        type=BeliefType.FACTUAL,
        content={"fact": "value"},
        confidence=0.9,
        source="test"
    )
    
    assert belief.id == "test_belief"
    assert belief.type == BeliefType.FACTUAL
    assert belief.content == {"fact": "value"}
    assert belief.confidence == 0.9
    assert belief.source == "test"


def test_belief_hashable():
    """Test that beliefs are hashable (for set operations)"""
    belief1 = Belief(id="b1", type=BeliefType.FACTUAL, content="test")
    belief2 = Belief(id="b1", type=BeliefType.FACTUAL, content="test")
    belief3 = Belief(id="b2", type=BeliefType.FACTUAL, content="test")
    
    # Same ID should have same hash
    assert hash(belief1) == hash(belief2)
    # Different ID should have different hash
    assert hash(belief1) != hash(belief3)


# Test Desire data structure
def test_desire_creation():
    """Test basic Desire creation"""
    desire = Desire(
        id="test_desire",
        type=DesireType.ACHIEVEMENT,
        description="Achieve goal X",
        priority=0.9,
        feasibility=0.8,
        conditions=["belief1", "belief2"]
    )
    
    assert desire.id == "test_desire"
    assert desire.type == DesireType.ACHIEVEMENT
    assert desire.description == "Achieve goal X"
    assert desire.priority == 0.9
    assert desire.feasibility == 0.8
    assert len(desire.conditions) == 2


# Test Intention data structure
def test_intention_creation():
    """Test basic Intention creation"""
    intention = Intention(
        id="test_intention",
        desire_id="desire1",
        plan=[
            {"action": "step1", "agent": "self"},
            {"action": "step2", "agent": "other"}
        ],
        status="pending"
    )
    
    assert intention.id == "test_intention"
    assert intention.desire_id == "desire1"
    assert len(intention.plan) == 2
    assert intention.status == "pending"


# Test BeliefBase
def test_belief_base_add_belief():
    """Test adding beliefs to belief base"""
    bb = BeliefBase()
    
    belief = Belief(
        id="b1",
        type=BeliefType.FACTUAL,
        content="test content"
    )
    
    bb.add_belief(belief)
    
    assert len(bb.beliefs) == 1
    assert bb.get_belief("b1") == belief


def test_belief_base_remove_belief():
    """Test removing beliefs from belief base"""
    bb = BeliefBase()
    
    belief = Belief(id="b1", type=BeliefType.FACTUAL, content="test")
    bb.add_belief(belief)
    
    assert len(bb.beliefs) == 1
    
    bb.remove_belief("b1")
    
    assert len(bb.beliefs) == 0
    assert bb.get_belief("b1") is None


def test_belief_base_query_by_type():
    """Test querying beliefs by type"""
    bb = BeliefBase()
    
    bb.add_belief(Belief(id="b1", type=BeliefType.FACTUAL, content="fact"))
    bb.add_belief(Belief(id="b2", type=BeliefType.CONTEXTUAL, content="context"))
    bb.add_belief(Belief(id="b3", type=BeliefType.FACTUAL, content="fact2"))
    
    factual_beliefs = bb.query_beliefs(type=BeliefType.FACTUAL)
    
    assert len(factual_beliefs) == 2
    assert all(b.type == BeliefType.FACTUAL for b in factual_beliefs)


def test_belief_base_query_by_confidence():
    """Test querying beliefs by minimum confidence"""
    bb = BeliefBase()
    
    bb.add_belief(Belief(id="b1", type=BeliefType.FACTUAL, content="1", confidence=0.9))
    bb.add_belief(Belief(id="b2", type=BeliefType.FACTUAL, content="2", confidence=0.5))
    bb.add_belief(Belief(id="b3", type=BeliefType.FACTUAL, content="3", confidence=0.7))
    
    high_confidence = bb.query_beliefs(min_confidence=0.8)
    
    assert len(high_confidence) == 1
    assert high_confidence[0].id == "b1"


def test_belief_base_consistency_check():
    """Test consistency checking"""
    bb = BeliefBase()
    
    # Add consistent beliefs
    bb.add_belief(Belief(id="b1", type=BeliefType.FACTUAL, content="A", confidence=0.9))
    bb.add_belief(Belief(id="b2", type=BeliefType.FACTUAL, content="B", confidence=0.8))
    
    inconsistencies = bb.check_consistency()
    
    # Should find no inconsistencies (or implement your own logic)
    assert isinstance(inconsistencies, list)


def test_belief_base_statistics():
    """Test belief base statistics"""
    bb = BeliefBase()
    
    bb.add_belief(Belief(id="b1", type=BeliefType.FACTUAL, content="1", confidence=0.9))
    bb.add_belief(Belief(id="b2", type=BeliefType.CONTEXTUAL, content="2", confidence=0.7))
    
    stats = bb.get_statistics()
    
    assert stats["total_beliefs"] == 2
    assert stats["by_type"][BeliefType.FACTUAL.value] == 1
    assert stats["by_type"][BeliefType.CONTEXTUAL.value] == 1
    assert 0.7 < stats["average_confidence"] < 0.9


# Test DesireSet
def test_desire_set_add_desire():
    """Test adding desires to desire set"""
    ds = DesireSet()
    
    desire = Desire(
        id="d1",
        type=DesireType.ACHIEVEMENT,
        description="Test desire",
        priority=0.8
    )
    
    ds.add_desire(desire)
    
    assert len(ds.desires) == 1
    assert ds.get_desire("d1") == desire


def test_desire_set_remove_desire():
    """Test removing desires from desire set"""
    ds = DesireSet()
    
    desire = Desire(id="d1", type=DesireType.ACHIEVEMENT, description="Test")
    ds.add_desire(desire)
    
    assert len(ds.desires) == 1
    
    ds.remove_desire("d1")
    
    assert len(ds.desires) == 0


def test_desire_set_get_top_desires():
    """Test getting top desires by priority"""
    ds = DesireSet()
    
    ds.add_desire(Desire(id="d1", type=DesireType.ACHIEVEMENT, description="Low", priority=0.3))
    ds.add_desire(Desire(id="d2", type=DesireType.ACHIEVEMENT, description="High", priority=0.9))
    ds.add_desire(Desire(id="d3", type=DesireType.ACHIEVEMENT, description="Medium", priority=0.6))
    
    top_2 = ds.get_top_desires(n=2)
    
    assert len(top_2) == 2
    assert top_2[0].id == "d2"  # Highest priority
    assert top_2[1].id == "d3"  # Second highest


def test_desire_set_filter_feasible():
    """Test filtering feasible desires based on beliefs"""
    ds = DesireSet()
    bb = BeliefBase()
    
    # Add beliefs
    bb.add_belief(Belief(id="b1", type=BeliefType.FACTUAL, content="available"))
    
    # Add desires with conditions
    ds.add_desire(Desire(
        id="d1",
        type=DesireType.ACHIEVEMENT,
        description="Feasible",
        conditions=["b1"]  # Required belief exists
    ))
    
    ds.add_desire(Desire(
        id="d2",
        type=DesireType.ACHIEVEMENT,
        description="Not feasible",
        conditions=["b2"]  # Required belief doesn't exist
    ))
    
    ds.add_desire(Desire(
        id="d3",
        type=DesireType.ACHIEVEMENT,
        description="No conditions"
    ))
    
    feasible = ds.filter_feasible(bb)
    
    assert len(feasible) == 2  # d1 and d3 are feasible
    assert any(d.id == "d1" for d in feasible)
    assert any(d.id == "d3" for d in feasible)
    assert not any(d.id == "d2" for d in feasible)


# Test IntentionStack
def test_intention_stack_adopt_intention():
    """Test adopting intentions"""
    stack = IntentionStack()
    
    intention = Intention(
        id="i1",
        desire_id="d1",
        plan=[{"action": "test"}]
    )
    
    stack.adopt_intention(intention)
    
    assert len(stack.intentions) == 1
    assert stack.active_intention == "i1"


def test_intention_stack_drop_intention():
    """Test dropping intentions"""
    stack = IntentionStack()
    
    intention = Intention(id="i1", desire_id="d1", plan=[])
    stack.adopt_intention(intention)
    
    assert len(stack.intentions) == 1
    
    stack.drop_intention("i1")
    
    assert len(stack.intentions) == 0
    assert stack.active_intention is None


def test_intention_stack_update_status():
    """Test updating intention status"""
    stack = IntentionStack()
    
    intention = Intention(id="i1", desire_id="d1", plan=[], status="pending")
    stack.adopt_intention(intention)
    
    stack.update_status("i1", "executing")
    
    updated_intention = stack.get_intention("i1")
    assert updated_intention.status == "executing"
    assert updated_intention.started_at is not None
    
    stack.update_status("i1", "completed")
    
    assert updated_intention.status == "completed"
    assert updated_intention.completed_at is not None


# Test BDIAgent (using a concrete implementation for testing)
class TestBDIAgent(BDIAgent):
    """Concrete BDI agent for testing"""
    
    def __init__(self):
        super().__init__("TestAgent")
        self.perceive_called = False
        self.deliberate_called = False
        self.plan_called = False
        self.execute_called = False
    
    async def perceive(self, observations):
        self.perceive_called = True
        # Add a test belief
        self.belief_base.add_belief(Belief(
            id="test_observation",
            type=BeliefType.FACTUAL,
            content=observations
        ))
    
    async def deliberate(self):
        self.deliberate_called = True
        # Add and return a test desire
        desire = Desire(
            id="test_desire",
            type=DesireType.ACHIEVEMENT,
            description="Test goal"
        )
        self.desire_set.add_desire(desire)
        return desire
    
    async def plan(self, desire):
        self.plan_called = True
        # Create a simple test plan
        return Intention(
            id="test_intention",
            desire_id=desire.id,
            plan=[{"action": "test_action"}]
        )
    
    async def execute(self, intention):
        self.execute_called = True
        return {"status": "completed", "result": "test_result"}


@pytest.mark.asyncio
async def test_bdi_agent_cycle():
    """Test complete BDI reasoning cycle"""
    agent = TestBDIAgent()
    
    # Create a test task
    task = {"content": "test task"}
    
    # Run BDI cycle
    result = await agent.bdi_cycle(task)
    
    # Verify all steps were called
    assert agent.perceive_called
    assert agent.deliberate_called
    assert agent.plan_called
    assert agent.execute_called
    
    # Verify result
    assert result["status"] == "completed"
    assert result["result"] == "test_result"
    
    # Verify beliefs, desires, and intentions were updated
    assert len(agent.belief_base.beliefs) > 0
    assert len(agent.desire_set.desires) > 0
    assert len(agent.intention_stack.intentions) > 0


@pytest.mark.asyncio
async def test_bdi_agent_no_feasible_desires():
    """Test BDI cycle when no feasible desires exist"""
    
    class NoDesireAgent(BDIAgent):
        def __init__(self):
            super().__init__("NoDesireAgent")
        
        async def perceive(self, observations):
            pass
        
        async def deliberate(self):
            return None  # No feasible desires
        
        async def plan(self, desire):
            pass
        
        async def execute(self, intention):
            pass
    
    agent = NoDesireAgent()
    result = await agent.bdi_cycle({})
    
    assert result["status"] == "no_action"
    assert "error" in result


def test_bdi_agent_get_state():
    """Test getting agent state"""
    agent = TestBDIAgent()
    
    # Add some test data
    agent.belief_base.add_belief(Belief(id="b1", type=BeliefType.FACTUAL, content="test"))
    agent.desire_set.add_desire(Desire(id="d1", type=DesireType.ACHIEVEMENT, description="test"))
    agent.intention_stack.adopt_intention(Intention(id="i1", desire_id="d1", plan=[]))
    
    state = agent.get_agent_state()
    
    assert state["agent_name"] == "TestAgent"
    assert "beliefs" in state
    assert "desires" in state
    assert "intentions" in state
    assert "consistency_issues" in state
    
    assert state["beliefs"]["total_beliefs"] == 1
    assert state["desires"]["total_desires"] == 1
    assert state["intentions"]["total_intentions"] == 1
