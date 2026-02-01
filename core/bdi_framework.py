"""
Bushidan Multi-Agent System - BDI Framework

Belief-Desire-Intention (BDI) framework for formal multi-agent system theory.
Adds academic rigor and theoretical foundation to the Bushidan system.

BDI Model:
- Beliefs: Agent's knowledge about the world
- Desires: Goals the agent wants to achieve  
- Intentions: Committed plans of action

References:
- Rao & Georgeff (1995): "BDI Agents: From Theory to Practice"
- Bratman (1987): "Intention, Plans, and Practical Reason"
- Wooldridge (2009): "An Introduction to MultiAgent Systems"
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

from utils.logger import get_logger


logger = get_logger(__name__)


class BeliefType(Enum):
    """Types of beliefs an agent can hold"""
    FACTUAL = "factual"           # Facts about the world
    CONTEXTUAL = "contextual"     # Context from Memory MCP
    OPERATIONAL = "operational"   # System state and capabilities
    HISTORICAL = "historical"     # Past decisions and outcomes


class DesireType(Enum):
    """Types of desires/goals"""
    ACHIEVEMENT = "achievement"   # Achieve a specific state
    MAINTENANCE = "maintenance"   # Maintain a condition
    OPTIMIZATION = "optimization" # Optimize a metric
    EXPLORATION = "exploration"   # Gather information


@dataclass
class Belief:
    """
    Representation of an agent's belief
    
    In BDI theory, beliefs represent the agent's knowledge about the world.
    They can be uncertain, incomplete, or incorrect.
    """
    id: str
    type: BeliefType
    content: Any
    confidence: float = 1.0  # 0.0 to 1.0
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Desire:
    """
    Representation of an agent's desire/goal
    
    In BDI theory, desires represent states the agent wants to achieve.
    Not all desires become intentions.
    """
    id: str
    type: DesireType
    description: str
    priority: float = 1.0  # Higher = more important
    feasibility: float = 1.0  # Estimated feasibility (0.0 to 1.0)
    conditions: List[str] = field(default_factory=list)  # Required beliefs
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Intention:
    """
    Representation of an agent's intention
    
    In BDI theory, intentions are committed plans of action.
    Once adopted, intentions guide the agent's behavior.
    """
    id: str
    desire_id: str  # Which desire this satisfies
    plan: List[Dict[str, Any]]  # Ordered steps
    status: str = "pending"  # pending, executing, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


class BeliefBase:
    """
    Agent's belief base - stores and manages beliefs
    
    Provides operations for:
    - Adding/removing beliefs
    - Querying beliefs
    - Belief revision and consistency checking
    """
    
    def __init__(self):
        self.beliefs: Dict[str, Belief] = {}
        self._belief_types_index: Dict[BeliefType, Set[str]] = {
            bt: set() for bt in BeliefType
        }
    
    def add_belief(self, belief: Belief) -> None:
        """Add or update a belief"""
        self.beliefs[belief.id] = belief
        self._belief_types_index[belief.type].add(belief.id)
        logger.debug(f"💭 Added belief: {belief.id} (type: {belief.type.value})")
    
    def remove_belief(self, belief_id: str) -> None:
        """Remove a belief"""
        if belief_id in self.beliefs:
            belief = self.beliefs[belief_id]
            self._belief_types_index[belief.type].discard(belief_id)
            del self.beliefs[belief_id]
            logger.debug(f"🗑️ Removed belief: {belief_id}")
    
    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """Retrieve a specific belief"""
        return self.beliefs.get(belief_id)
    
    def query_beliefs(self, 
                     type: Optional[BeliefType] = None,
                     min_confidence: float = 0.0) -> List[Belief]:
        """Query beliefs with filters"""
        beliefs = self.beliefs.values()
        
        if type:
            belief_ids = self._belief_types_index[type]
            beliefs = [self.beliefs[bid] for bid in belief_ids]
        
        if min_confidence > 0.0:
            beliefs = [b for b in beliefs if b.confidence >= min_confidence]
        
        return list(beliefs)
    
    def check_consistency(self) -> List[str]:
        """
        Check for contradictory beliefs
        
        Returns list of inconsistencies found
        """
        inconsistencies = []
        
        # Simple consistency check: look for beliefs with same id but different content
        # In a full BDI implementation, this would use formal logic
        
        factual_beliefs = self.query_beliefs(type=BeliefType.FACTUAL)
        for i, b1 in enumerate(factual_beliefs):
            for b2 in factual_beliefs[i+1:]:
                if b1.content == b2.content and b1.confidence != b2.confidence:
                    inconsistencies.append(
                        f"Conflicting confidences for belief: {b1.id}"
                    )
        
        return inconsistencies
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get belief base statistics"""
        return {
            "total_beliefs": len(self.beliefs),
            "by_type": {
                bt.value: len(self._belief_types_index[bt])
                for bt in BeliefType
            },
            "average_confidence": sum(b.confidence for b in self.beliefs.values()) / len(self.beliefs) if self.beliefs else 0.0
        }


class DesireSet:
    """
    Agent's desire set - stores and manages desires/goals
    
    Provides operations for:
    - Adding/removing desires
    - Prioritizing desires
    - Checking feasibility
    """
    
    def __init__(self):
        self.desires: Dict[str, Desire] = {}
        self._priority_sorted: List[str] = []
    
    def add_desire(self, desire: Desire) -> None:
        """Add a desire"""
        self.desires[desire.id] = desire
        self._resort_priorities()
        logger.debug(f"🎯 Added desire: {desire.id} (priority: {desire.priority})")
    
    def remove_desire(self, desire_id: str) -> None:
        """Remove a desire"""
        if desire_id in self.desires:
            del self.desires[desire_id]
            self._resort_priorities()
            logger.debug(f"🗑️ Removed desire: {desire_id}")
    
    def get_desire(self, desire_id: str) -> Optional[Desire]:
        """Retrieve a specific desire"""
        return self.desires.get(desire_id)
    
    def get_top_desires(self, n: int = 5) -> List[Desire]:
        """Get top N desires by priority"""
        return [self.desires[did] for did in self._priority_sorted[:n]]
    
    def _resort_priorities(self) -> None:
        """Re-sort desires by priority"""
        self._priority_sorted = sorted(
            self.desires.keys(),
            key=lambda did: self.desires[did].priority * self.desires[did].feasibility,
            reverse=True
        )
    
    def filter_feasible(self, belief_base: BeliefBase) -> List[Desire]:
        """
        Filter desires to those that are feasible given current beliefs
        
        A desire is feasible if all its required beliefs exist
        """
        feasible = []
        
        for desire in self.desires.values():
            if not desire.conditions:
                feasible.append(desire)
                continue
            
            # Check if all required beliefs are satisfied
            all_satisfied = True
            for condition in desire.conditions:
                belief = belief_base.get_belief(condition)
                if not belief or belief.confidence < 0.5:
                    all_satisfied = False
                    break
            
            if all_satisfied:
                feasible.append(desire)
        
        return feasible
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get desire set statistics"""
        return {
            "total_desires": len(self.desires),
            "by_type": {
                dt.value: len([d for d in self.desires.values() if d.type == dt])
                for dt in DesireType
            },
            "average_priority": sum(d.priority for d in self.desires.values()) / len(self.desires) if self.desires else 0.0
        }


class IntentionStack:
    """
    Agent's intention stack - stores and manages intentions
    
    Provides operations for:
    - Adopting/dropping intentions
    - Tracking execution status
    - Managing intention priorities
    """
    
    def __init__(self):
        self.intentions: Dict[str, Intention] = {}
        self.active_intention: Optional[str] = None
    
    def adopt_intention(self, intention: Intention) -> None:
        """Adopt a new intention"""
        self.intentions[intention.id] = intention
        if not self.active_intention:
            self.active_intention = intention.id
        logger.debug(f"✨ Adopted intention: {intention.id}")
    
    def drop_intention(self, intention_id: str) -> None:
        """Drop an intention"""
        if intention_id in self.intentions:
            del self.intentions[intention_id]
            if self.active_intention == intention_id:
                self.active_intention = None
            logger.debug(f"🗑️ Dropped intention: {intention_id}")
    
    def get_intention(self, intention_id: str) -> Optional[Intention]:
        """Retrieve a specific intention"""
        return self.intentions.get(intention_id)
    
    def get_active_intention(self) -> Optional[Intention]:
        """Get the currently active intention"""
        if self.active_intention:
            return self.intentions.get(self.active_intention)
        return None
    
    def update_status(self, intention_id: str, status: str) -> None:
        """Update intention status"""
        if intention_id in self.intentions:
            intention = self.intentions[intention_id]
            intention.status = status
            
            if status == "executing" and intention.started_at is None:
                intention.started_at = datetime.now()
            elif status in ["completed", "failed"] and intention.completed_at is None:
                intention.completed_at = datetime.now()
            
            logger.debug(f"📊 Updated intention {intention_id} status: {status}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get intention stack statistics"""
        status_counts = {}
        for intention in self.intentions.values():
            status_counts[intention.status] = status_counts.get(intention.status, 0) + 1
        
        return {
            "total_intentions": len(self.intentions),
            "active_intention": self.active_intention,
            "by_status": status_counts
        }


class BDIAgent(ABC):
    """
    Abstract base class for BDI agents
    
    Implements the classic BDI reasoning cycle:
    1. Perceive (update beliefs based on observations)
    2. Deliberate (decide which desires to pursue)
    3. Plan (create intentions to achieve desires)
    4. Execute (carry out intentions)
    5. Reconsider (update beliefs, desires, intentions)
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.belief_base = BeliefBase()
        self.desire_set = DesireSet()
        self.intention_stack = IntentionStack()
        
        logger.info(f"🧠 Initialized BDI agent: {agent_name}")
    
    @abstractmethod
    async def perceive(self, observations: Dict[str, Any]) -> None:
        """
        Update beliefs based on observations
        
        Args:
            observations: New information from environment
        """
        pass
    
    @abstractmethod
    async def deliberate(self) -> Optional[Desire]:
        """
        Select which desire to pursue based on current beliefs
        
        Returns:
            The selected desire, or None if no feasible desires
        """
        pass
    
    @abstractmethod
    async def plan(self, desire: Desire) -> Optional[Intention]:
        """
        Create a plan (intention) to achieve the selected desire
        
        Args:
            desire: The desire to plan for
            
        Returns:
            An intention with a plan, or None if no plan possible
        """
        pass
    
    @abstractmethod
    async def execute(self, intention: Intention) -> Dict[str, Any]:
        """
        Execute the current intention
        
        Args:
            intention: The intention to execute
            
        Returns:
            Execution results
        """
        pass
    
    async def bdi_cycle(self, task: Any) -> Dict[str, Any]:
        """
        Run one complete BDI reasoning cycle
        
        This is the main control loop for BDI agents:
        Perceive → Deliberate → Plan → Execute → Reconsider
        """
        
        logger.info(f"🔄 Starting BDI cycle for {self.agent_name}")
        
        try:
            # Step 1: Perceive - update beliefs from observations
            observations = await self._gather_observations(task)
            await self.perceive(observations)
            
            # Step 2: Deliberate - select desire to pursue
            desire = await self.deliberate()
            if not desire:
                logger.warning(f"⚠️ No feasible desires for {self.agent_name}")
                return {"error": "No feasible desires", "status": "no_action"}
            
            # Step 3: Plan - create intention for desire
            intention = await self.plan(desire)
            if not intention:
                logger.warning(f"⚠️ Could not create plan for desire: {desire.id}")
                return {"error": "Planning failed", "status": "failed"}
            
            # Adopt the intention
            self.intention_stack.adopt_intention(intention)
            self.intention_stack.update_status(intention.id, "executing")
            
            # Step 4: Execute - carry out the intention
            result = await self.execute(intention)
            
            # Step 5: Reconsider - update based on execution results
            await self._reconsider(intention, result)
            
            logger.info(f"✅ BDI cycle complete for {self.agent_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ BDI cycle failed for {self.agent_name}: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _gather_observations(self, task: Any) -> Dict[str, Any]:
        """
        Gather observations from the environment
        
        Override in subclasses to customize observation gathering
        """
        return {
            "task": task,
            "timestamp": datetime.now()
        }
    
    async def _reconsider(self, intention: Intention, result: Dict[str, Any]) -> None:
        """
        Reconsider beliefs, desires, and intentions based on execution results
        
        Override in subclasses to customize reconsideration logic
        """
        if result.get("status") == "completed":
            self.intention_stack.update_status(intention.id, "completed")
            # Remove satisfied desire
            self.desire_set.remove_desire(intention.desire_id)
        else:
            self.intention_stack.update_status(intention.id, "failed")
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get complete agent state for inspection"""
        return {
            "agent_name": self.agent_name,
            "beliefs": self.belief_base.get_statistics(),
            "desires": self.desire_set.get_statistics(),
            "intentions": self.intention_stack.get_statistics(),
            "consistency_issues": self.belief_base.check_consistency()
        }
