# Theoretical Foundation: BDI Framework in Bushidan Multi-Agent System

## Executive Summary

This document establishes the formal theoretical foundation for the Bushidan Multi-Agent System's implementation of the Belief-Desire-Intention (BDI) framework. By integrating BDI theory with the existing hierarchical architecture, Bushidan achieves academic rigor while maintaining practical effectiveness.

---

## 1. Introduction to BDI Theory

### 1.1 Historical Context

The BDI (Belief-Desire-Intention) model originated from philosophical work on practical reasoning (Bratman, 1987) and was formalized for multi-agent systems by Rao & Georgeff (1991, 1995).

**Key Publications:**
- Bratman, M. E. (1987). *Intention, Plans, and Practical Reason*. Harvard University Press.
- Rao, A. S., & Georgeff, M. P. (1991). "Modeling Rational Agents within a BDI-Architecture." *KR*, 91, 473-484.
- Rao, A. S., & Georgeff, M. P. (1995). "BDI Agents: From Theory to Practice." *ICMAS*, 95, 312-319.
- Wooldridge, M. (2009). *An Introduction to MultiAgent Systems* (2nd ed.). Wiley.

### 1.2 Core Concepts

**Beliefs** represent the agent's knowledge about the world. Beliefs can be:
- Factual (known facts)
- Contextual (situational knowledge)
- Operational (system capabilities)
- Historical (past experiences)

**Desires** represent goals or states the agent wants to achieve. Not all desires become intentions due to:
- Resource constraints
- Conflicting goals
- Feasibility limitations

**Intentions** are committed plans of action. Once adopted, intentions guide agent behavior and represent a commitment to achieve a desire.

### 1.3 BDI Reasoning Cycle

```
┌──────────────────────────────────────────────┐
│ BDI Reasoning Cycle                         │
└──────────────────────────────────────────────┘

1. Perceive
   ↓ (Update beliefs based on observations)
   
2. Deliberate
   ↓ (Select which desire to pursue)
   
3. Plan
   ↓ (Create intention to achieve desire)
   
4. Execute
   ↓ (Carry out intention)
   
5. Reconsider
   ↓ (Update beliefs, desires, intentions)
   
   → Back to 1. Perceive
```

---

## 2. BDI Integration in Bushidan Architecture

### 2.1 Hierarchical Mapping

Bushidan's hierarchical structure naturally aligns with BDI theory:

| Bushidan Layer | BDI Mapping | Reasoning Focus |
|----------------|-------------|-----------------|
| **Shogun (将軍)** | Strategic BDI | High-level goal selection, quality vs cost trade-offs |
| **Karo (家老)** | Tactical BDI | Task decomposition, resource allocation |
| **Taisho (大将)** | Implementation BDI | Execution planning, tool selection |
| **Ashigaru (足軽)** | Reactive | Simple stimulus-response (no full BDI) |

### 2.2 Design Philosophy

**Non-Breaking Integration:**
- BDI is opt-in via `task.context['use_bdi'] = True`
- Original agents continue to function without BDI
- BDI adds formal semantics without disrupting workflows

**Backward Compatibility:**
- All existing code paths remain functional
- BDI-enhanced agents wrap original implementations
- No changes to existing interfaces

---

## 3. Shogun BDI Implementation

### 3.1 Beliefs (Strategic Knowledge)

**Operational Beliefs:**
```python
Belief(
    id="has_karo",
    type=BeliefType.OPERATIONAL,
    content={"capability": "tactical_coordination", "available": True},
    confidence=1.0
)
```

**Task Assessment Beliefs:**
```python
Belief(
    id=f"task_complexity_{task_id}",
    type=BeliefType.FACTUAL,
    content={"task": task.content, "complexity": "strategic"},
    confidence=0.9
)
```

**Historical Context Beliefs:**
```python
Belief(
    id=f"historical_context_{task_id}",
    type=BeliefType.HISTORICAL,
    content={"entries": memory_mcp_results},
    confidence=0.8,
    source="memory_mcp"
)
```

### 3.2 Desires (Strategic Goals)

**Quality Maintenance:**
```python
Desire(
    id="maintain_quality",
    type=DesireType.MAINTENANCE,
    description="Maintain high quality standards (95+ points)",
    priority=0.9,
    feasibility=1.0
)
```

**Cost Optimization:**
```python
Desire(
    id="optimize_cost",
    type=DesireType.OPTIMIZATION,
    description="Optimize cost while maintaining quality",
    priority=0.7,
    feasibility=1.0
)
```

**Continuous Learning:**
```python
Desire(
    id="learn_and_improve",
    type=DesireType.EXPLORATION,
    description="Learn from past decisions",
    priority=0.6,
    feasibility=0.8,
    conditions=["has_memory_mcp"]
)
```

### 3.3 Intentions (Strategic Plans)

**Quality-Focused Plan:**
```python
plan = [
    {"action": "handle_strategic_directly", "agent": "self"},
    {"action": "opus_premium_review", "agent": "opus"}
]
```

**Cost-Optimized Plan:**
```python
plan = [
    {"action": "delegate_to_karo", "agent": "karo"},
    {"action": "basic_review", "agent": "self"}
]
```

### 3.4 Deliberation Process

Shogun deliberation considers:
1. Task complexity (from beliefs)
2. Available resources (operational beliefs)
3. Strategic priorities (desire priorities)
4. Historical patterns (historical beliefs)

**Deliberation Algorithm:**
```
feasible_desires = filter_by_conditions(all_desires, beliefs)
top_desire = max(feasible_desires, key=lambda d: d.priority * d.feasibility)
```

---

## 4. Karo BDI Implementation

### 4.1 Beliefs (Tactical Knowledge)

**Resource Availability:**
```python
Belief(
    id="available_ashigaru",
    type=BeliefType.OPERATIONAL,
    content={"ashigaru_types": ["filesystem", "git", "memory"], "count": 3},
    confidence=1.0
)
```

**Task Decomposability:**
```python
Belief(
    id=f"task_info_{task_id}",
    type=BeliefType.FACTUAL,
    content={"estimated_subtasks": 3, "parallelizable": True},
    confidence=0.9
)
```

### 4.2 Desires (Tactical Goals)

**Efficient Decomposition:**
```python
Desire(
    id="efficient_decomposition",
    type=DesireType.OPTIMIZATION,
    description="Decompose tasks efficiently",
    priority=0.9,
    conditions=["has_decomposition"]
)
```

**Maximize Parallelization:**
```python
Desire(
    id="maximize_parallelization",
    type=DesireType.OPTIMIZATION,
    description="Maximize parallel execution",
    priority=0.8,
    conditions=["has_parallel_execution", "available_ashigaru"]
)
```

### 4.3 Planning Strategy

Karo planning includes:
- Decomposition strategy (sequential vs parallel)
- Ashigaru allocation (which tools for which subtasks)
- Integration approach (how to combine results)

---

## 5. Taisho BDI Implementation

### 5.1 Beliefs (Implementation Knowledge)

**Tool Availability:**
```python
Belief(
    id="available_mcp_tools",
    type=BeliefType.OPERATIONAL,
    content={"tools": ["filesystem", "git", "memory"], "count": 3}
)
```

**Code Quality Metrics:**
```python
Belief(
    id="has_self_healing",
    type=BeliefType.OPERATIONAL,
    content={"capability": "error_correction", "max_attempts": 3}
)
```

### 5.2 Desires (Implementation Goals)

**Correctness:**
```python
Desire(
    id="correct_implementation",
    type=DesireType.ACHIEVEMENT,
    description="Generate correct code",
    priority=1.0,
    conditions=["has_qwen3_coder"]
)
```

**Efficiency:**
```python
Desire(
    id="efficient_execution",
    type=DesireType.OPTIMIZATION,
    description="Minimize resource usage",
    priority=0.7
)
```

---

## 6. Formal Semantics

### 6.1 Belief Revision

**Update Rule:**
```
If new_belief.confidence > old_belief.confidence:
    BeliefBase.update(belief_id, new_belief)
```

**Consistency Checking:**
```python
def check_consistency(self) -> List[str]:
    for b1, b2 in all_pairs(factual_beliefs):
        if b1.content == b2.content and b1.confidence != b2.confidence:
            flag_inconsistency(b1, b2)
```

### 6.2 Desire Selection

**Feasibility Filter:**
```
feasible = {d ∈ Desires | ∀c ∈ d.conditions: c ∈ Beliefs}
```

**Priority Function:**
```
score(d) = d.priority × d.feasibility
selected_desire = argmax(feasible, score)
```

### 6.3 Plan Execution

**Sequential Execution:**
```
for step in intention.plan:
    result = execute_action(step)
    if result.failed:
        reconsider()
        break
```

**Status Tracking:**
```
pending → executing → {completed | failed}
```

---

## 7. Comparison with Classical MAS Frameworks

### 7.1 vs Pure Reactive Agents

| Aspect | Reactive Agents | BDI Agents (Bushidan) |
|--------|----------------|----------------------|
| Reasoning | Stimulus-response | Goal-directed |
| Planning | None | Explicit plans |
| Flexibility | Low | High |
| Complexity | Low | Medium |

### 7.2 vs Planning Agents

| Aspect | Planning Agents | BDI Agents (Bushidan) |
|--------|----------------|----------------------|
| Real-time | Poor | Excellent |
| Partial info | Poor | Good |
| Belief tracking | None | Explicit |
| Implementation | Complex | Practical |

### 7.3 vs Hybrid Architectures

Bushidan implements a **hybrid BDI-hierarchical architecture**:
- Top layers (Shogun, Karo): Full BDI reasoning
- Bottom layer (Ashigaru): Reactive (efficient for simple tasks)

**Advantages:**
- Best of both worlds: reasoning + efficiency
- Scalable to complex multi-agent coordination
- Maintains theoretical soundness

---

## 8. Academic Contributions

### 8.1 Novel Aspects

**1. Hierarchical BDI with Heterogeneous Models:**
- Different LLM models per layer (Claude, Gemini, Qwen)
- Each layer has appropriate BDI complexity
- Novel integration of cloud + local models in BDI

**2. Opt-in BDI Design:**
- Backward compatibility preserved
- Gradual adoption path
- Practical for production systems

**3. Cost-Aware Deliberation:**
- Desires include cost considerations
- Planning balances quality vs efficiency
- Novel economic dimension in BDI

### 8.2 Theoretical Extensions

**Memory-Enhanced BDI:**
- Historical beliefs from Memory MCP
- Learning from past deliberations
- Not standard in classical BDI

**Quality Metrics in Beliefs:**
- Code quality scores as beliefs
- Security findings as beliefs
- Enables formal reasoning about quality

---

## 9. Validation and Benchmarking

### 9.1 Formal Properties

**Completeness:**
- All tasks are processed (if feasible desires exist)
- BDI cycle always terminates

**Soundness:**
- Plans are consistent with desires
- Beliefs are updated correctly

**Optimality:**
- Highest priority feasible desire is selected
- Plans maximize expected utility

### 9.2 Empirical Validation

**Metrics to Track:**
1. **Belief Accuracy**: How often beliefs match reality
2. **Desire Satisfaction Rate**: % of desires successfully achieved
3. **Plan Success Rate**: % of intentions completed without failure
4. **Deliberation Time**: Overhead of BDI reasoning
5. **Quality Improvement**: BDI vs non-BDI quality scores

**Benchmark Suite:**
- Standard tasks (SWE-bench, HumanEval)
- Complex strategic decisions
- Resource-constrained scenarios

---

## 10. Future Research Directions

### 10.1 Short-term (3-6 months)

1. **Quantitative Evaluation:**
   - Run benchmark suite comparing BDI vs non-BDI modes
   - Measure quality, cost, time trade-offs
   - Publish results

2. **Belief Revision Algorithms:**
   - Implement more sophisticated consistency checking
   - Add probabilistic belief updates
   - Integrate with Memory MCP for learning

### 10.2 Long-term (6-12 months)

1. **Multi-Agent Coordination:**
   - Inter-agent belief sharing (Shogun ↔ Karo ↔ Taisho)
   - Distributed deliberation
   - Conflict resolution mechanisms

2. **Adaptive Learning:**
   - Learn desire priorities from outcomes
   - Adjust feasibility estimates over time
   - Meta-level reasoning about BDI parameters

3. **Formal Verification:**
   - Prove safety properties (no deadlock, progress)
   - Model checking for critical scenarios
   - Temporal logic specifications

---

## 11. References

### Core BDI Theory

1. Bratman, M. E. (1987). *Intention, Plans, and Practical Reason*. Harvard University Press.

2. Rao, A. S., & Georgeff, M. P. (1991). "Modeling Rational Agents within a BDI-Architecture." *Proceedings of KR-91*, 473-484.

3. Rao, A. S., & Georgeff, M. P. (1995). "BDI Agents: From Theory to Practice." *Proceedings of ICMAS-95*, 312-319.

4. Wooldridge, M. (2009). *An Introduction to MultiAgent Systems* (2nd edition). Wiley.

### Implementation and Applications

5. Bordini, R. H., Hübner, J. F., & Wooldridge, M. (2007). *Programming Multi-Agent Systems in AgentSpeak using Jason*. Wiley.

6. Pokahr, A., Braubach, L., & Lamersdorf, W. (2005). "Jadex: A BDI reasoning engine." *Multi-Agent Programming*, 149-174.

### Related Multi-Agent Research

7. Wooldridge, M., & Jennings, N. R. (1995). "Intelligent agents: Theory and practice." *The Knowledge Engineering Review*, 10(2), 115-152.

8. Ferber, J. (1999). *Multi-Agent Systems: An Introduction to Distributed Artificial Intelligence*. Addison Wesley.

---

## 12. Appendix: Implementation Details

### A. Code Structure

```
core/
├── bdi_framework.py        # Core BDI abstractions
├── bdi_shogun.py          # Strategic BDI agent
├── bdi_karo.py            # Tactical BDI agent
└── bdi_taisho.py          # Implementation BDI agent

tests/unit/
└── test_bdi_framework.py  # Unit tests

docs/research/
└── theoretical_foundation.md  # This document
```

### B. Usage Example

```python
from core.bdi_shogun import BDIShogun
from core.shogun import Task, TaskComplexity

# Initialize BDI-enhanced Shogun
shogun = BDIShogun(orchestrator)
await shogun.initialize()

# Process task with BDI reasoning
task = Task(
    content="Implement user authentication system",
    complexity=TaskComplexity.COMPLEX,
    context={"use_bdi": True}  # Enable BDI mode
)

result = await shogun.process_task(task)

# Inspect BDI state
state = shogun.get_agent_state()
print(f"Beliefs: {state['beliefs']}")
print(f"Desires: {state['desires']}")
print(f"Intentions: {state['intentions']}")
```

### C. Configuration

Add to `config/settings.yaml`:

```yaml
bdi_framework:
  enabled: true
  default_mode: false  # Opt-in per task
  
  shogun:
    strategic_desires:
      - maintain_quality
      - optimize_cost
      - learn_and_improve
  
  karo:
    tactical_desires:
      - efficient_decomposition
      - maximize_parallelization
  
  taisho:
    implementation_desires:
      - correct_implementation
      - quality_validation
```

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-01  
**Authors:** Bushidan Multi-Agent System Development Team  
**Status:** Active Research
