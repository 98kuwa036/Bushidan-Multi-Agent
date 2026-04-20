# Bushidan Multi-Agent System - Benchmarking Guide

Comprehensive guide to benchmarking the Bushidan Multi-Agent System for performance, quality, and cost analysis.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Benchmark Framework](#benchmark-framework)
- [Standard Task Sets](#standard-task-sets)
- [Running Benchmarks](#running-benchmarks)
- [Analyzing Results](#analyzing-results)
- [Baseline Comparisons](#baseline-comparisons)
- [Custom Benchmarks](#custom-benchmarks)
- [Best Practices](#best-practices)

## Overview

The Bushidan benchmark framework provides:

- **Standardized Tasks**: 60+ tasks across 9 categories
- **Performance Metrics**: Processing time, quality scores, cost per task
- **Baseline Comparisons**: Compare against Claude Code, AutoGPT, Copilot
- **Automated Reporting**: Comprehensive JSON and console reports
- **Category Analysis**: Performance breakdown by task type and difficulty

**Benchmark Categories**:
1. Code Generation (10 tasks)
2. Question Answering (8 tasks)
3. Refactoring (3 tasks)
4. Debugging (3 tasks)
5. Architecture (4 tasks)
6. Optimization (2 tasks)
7. Testing (3 tasks)
8. Documentation (4 tasks)
9. Japanese Language (4 tasks)

## Quick Start

### Running Quick Benchmark (10 tasks, ~2 minutes)

```python
import asyncio
from benchmarks.benchmark_framework import BenchmarkRunner
from benchmarks.standard_tasks import get_quick_benchmark_tasks
from core.shogun import Shogun
from core.system_orchestrator import SystemOrchestrator

async def main():
    # Initialize system
    orchestrator = SystemOrchestrator()
    await orchestrator.initialize()
    
    shogun = Shogun(orchestrator)
    await shogun.initialize()
    
    # Run quick benchmark
    runner = BenchmarkRunner(system_version="9.3.2")
    tasks = get_quick_benchmark_tasks()
    
    report = await runner.run_benchmark_suite(
        tasks=tasks,
        system_processor=shogun,
        max_concurrent=3
    )
    
    # Print results
    runner.print_summary(report)
    
    # Save report
    from pathlib import Path
    runner.save_report(report, Path("benchmarks/results/quick_benchmark.json"))

if __name__ == "__main__":
    asyncio.run(main())
```

### Expected Output

```
🏯 Starting Bushidan Benchmark Suite
Version: 9.3.2
Tasks: 10
Max Concurrent: 3

  Running: quick_001 (simple)
  Running: quick_002 (simple)
  Running: quick_003 (medium)
  ...

✅ Benchmark Complete
Total Time: 127.45s
Success Rate: 90.0%
Average Quality: 87.3/100

================================================================================
🏯 BUSHIDAN BENCHMARK REPORT - 9.3.2
================================================================================

📊 OVERVIEW:
  Total Tasks:      10
  Successful:       9 (90.0%)
  Failed:           1
  Errors:           0

⚡ PERFORMANCE:
  Average Time:     12.75s
  Median Time:      11.20s
  P95 Time:         28.50s

💰 COST:
  Total Cost:       ¥10.50
  Cost per Task:    ¥1.05

🎯 QUALITY:
  Average:          87.3/100
  Median:           88.0/100
  Range:            75.0 - 95.0

🔀 ROUTING:
  groq           :   2 ( 20.0%)
  local_qwen3    :   5 ( 50.0%)
  cloud_qwen3    :   2 ( 20.0%)
  gemini3        :   1 ( 10.0%)

📁 BY DIFFICULTY:
  simple    : 2/2 success, avg 2.3s, quality 85.5
  medium    : 4/4 success, avg 12.1s, quality 86.8
  complex   : 2/3 success, avg 29.5s, quality 90.0
  strategic : 1/1 success, avg 45.2s, quality 92.0

================================================================================
```

## Benchmark Framework

### BenchmarkRunner

Main class for running benchmarks:

```python
from benchmarks.benchmark_framework import BenchmarkRunner

runner = BenchmarkRunner(system_version="9.3.2")

# Run benchmark suite
report = await runner.run_benchmark_suite(
    tasks=task_list,
    system_processor=shogun_instance,
    max_concurrent=5  # Parallel execution
)

# Generate outputs
runner.print_summary(report)
runner.save_report(report, output_path)
```

### BenchmarkTask

Define individual benchmark tasks:

```python
from benchmarks.benchmark_framework import BenchmarkTask

task = BenchmarkTask(
    id="custom_001",
    category="code_generation",
    difficulty="medium",
    content="Implement a binary search tree in Python",
    expected_output="class BinarySearchTree:...",  # Optional
    evaluation_criteria={"has_insert": True, "has_search": True}  # Optional
)
```

### BenchmarkReport

Comprehensive report structure:

```python
@dataclass
class BenchmarkReport:
    timestamp: str
    system_version: str
    total_tasks: int
    successful_tasks: int
    
    # Performance
    average_time_seconds: float
    median_time_seconds: float
    p95_time_seconds: float
    
    # Cost
    total_cost_yen: float
    average_cost_per_task: float
    
    # Quality
    average_quality_score: float
    median_quality_score: float
    
    # Reliability
    success_rate: float
    fallback_rate: float
    
    # Detailed results
    route_distribution: Dict[str, int]
    results_by_category: Dict[str, Dict]
    results_by_difficulty: Dict[str, Dict]
    task_results: List[TaskResult]
```

## Standard Task Sets

### Full Benchmark Suite (60+ tasks)

```python
from benchmarks.standard_tasks import get_all_standard_tasks

tasks = get_all_standard_tasks()
# 60+ tasks across all categories
```

### Category-Specific Benchmarks

```python
from benchmarks.standard_tasks import (
    get_code_generation_tasks,
    get_qa_tasks,
    get_refactoring_tasks,
    get_debugging_tasks,
    get_architecture_tasks,
    get_optimization_tasks,
    get_testing_tasks,
    get_documentation_tasks,
    get_japanese_tasks
)

# Run only code generation benchmarks
code_tasks = get_code_generation_tasks()  # 10 tasks
```

### Quick Benchmark (10 tasks)

```python
from benchmarks.standard_tasks import get_quick_benchmark_tasks

quick_tasks = get_quick_benchmark_tasks()
# 2 simple, 4 medium, 3 complex, 1 strategic
```

### Stress Test (3 large tasks)

```python
from benchmarks.standard_tasks import get_stress_test_tasks

stress_tasks = get_stress_test_tasks()
# Large, complex tasks to test system limits
```

## Running Benchmarks

### Full Benchmark Script

Create `benchmarks/run_full_benchmark.py`:

```python
#!/usr/bin/env python3
"""Run full Bushidan benchmark suite"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_framework import BenchmarkRunner, compare_with_baseline
from benchmarks.standard_tasks import get_all_standard_tasks, BASELINE_CLAUDE_CODE
from core.shogun import Shogun
from core.system_orchestrator import SystemOrchestrator


async def main():
    print("🏯 Bushidan Full Benchmark Suite")
    print("=" * 80)
    
    # Initialize system
    print("\n📦 Initializing Bushidan system...")
    orchestrator = SystemOrchestrator()
    await orchestrator.initialize()
    
    shogun = Shogun(orchestrator)
    await shogun.initialize()
    
    # Load tasks
    print("📋 Loading standard tasks...")
    tasks = get_all_standard_tasks()
    print(f"   Loaded {len(tasks)} tasks")
    
    # Run benchmark
    runner = BenchmarkRunner(system_version="9.3.2")
    report = await runner.run_benchmark_suite(
        tasks=tasks,
        system_processor=shogun,
        max_concurrent=5
    )
    
    # Print summary
    runner.print_summary(report)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"full_benchmark_{timestamp}.json"
    runner.save_report(report, output_path)
    
    # Compare with baseline
    print("\n📊 Baseline Comparison (vs Claude Code):")
    comparison = compare_with_baseline(report, BASELINE_CLAUDE_CODE)
    
    for metric, data in comparison["metrics"].items():
        improvement = data["improvement"]
        symbol = "📈" if improvement > 0 else "📉"
        print(f"  {symbol} {metric:20s}: {improvement:+6.1f}%")
    
    print(f"\n✅ Complete! Report saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python benchmarks/run_full_benchmark.py
```

### Continuous Benchmarking

For tracking performance over time:

```python
# benchmarks/continuous_benchmark.py

import asyncio
from pathlib import Path
from datetime import datetime

async def run_daily_benchmark():
    """Run benchmark and track trends"""
    
    # Run benchmark
    runner = BenchmarkRunner()
    tasks = get_quick_benchmark_tasks()  # Quick daily check
    report = await runner.run_benchmark_suite(tasks, shogun, max_concurrent=3)
    
    # Save with timestamp
    date = datetime.now().strftime("%Y-%m-%d")
    output = Path(f"benchmarks/results/daily/{date}.json")
    runner.save_report(report, output)
    
    # Check for regressions
    if report.success_rate < 0.85:
        print("⚠️  WARNING: Success rate below 85%!")
    
    if report.average_time_seconds > 15.0:
        print("⚠️  WARNING: Average time above target!")
    
    return report

# Schedule daily
asyncio.run(run_daily_benchmark())
```

## Analyzing Results

### Loading Saved Reports

```python
import json
from pathlib import Path

# Load report
with open("benchmarks/results/full_benchmark_20260131.json") as f:
    report_data = json.load(f)

# Analyze
print(f"Success Rate: {report_data['success_rate']:.1%}")
print(f"Average Cost: ¥{report_data['average_cost_per_task']:.2f}")
```

### Comparing Multiple Reports

```python
def compare_reports(report1_path, report2_path):
    """Compare two benchmark reports"""
    
    with open(report1_path) as f:
        report1 = json.load(f)
    
    with open(report2_path) as f:
        report2 = json.load(f)
    
    print("Comparison:")
    print(f"  Success Rate: {report1['success_rate']:.1%} → {report2['success_rate']:.1%}")
    print(f"  Avg Time: {report1['average_time_seconds']:.1f}s → {report2['average_time_seconds']:.1f}s")
    print(f"  Avg Cost: ¥{report1['average_cost_per_task']:.2f} → ¥{report2['average_cost_per_task']:.2f}")
    print(f"  Avg Quality: {report1['average_quality_score']:.1f} → {report2['average_quality_score']:.1f}")
```

### Trend Analysis

```python
from pathlib import Path
import json
import matplotlib.pyplot as plt

def plot_benchmark_trends(results_dir="benchmarks/results/daily"):
    """Plot benchmark metrics over time"""
    
    reports = []
    for path in sorted(Path(results_dir).glob("*.json")):
        with open(path) as f:
            reports.append(json.load(f))
    
    dates = [r["timestamp"][:10] for r in reports]
    success_rates = [r["success_rate"] for r in reports]
    avg_times = [r["average_time_seconds"] for r in reports]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(dates, success_rates, marker='o')
    ax1.set_title("Success Rate Over Time")
    ax1.set_ylabel("Success Rate")
    ax1.grid(True)
    
    ax2.plot(dates, avg_times, marker='o', color='orange')
    ax2.set_title("Average Time Over Time")
    ax2.set_ylabel("Seconds")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("benchmarks/results/trends.png")
    plt.show()
```

## Baseline Comparisons

### Available Baselines

```python
from benchmarks.standard_tasks import (
    BASELINE_CLAUDE_CODE,
    BASELINE_AUTOGPT,
    BASELINE_COPILOT
)

# Claude Code (Sonnet 4.5)
# - Success Rate: 98%
# - Avg Time: 8.0s
# - Avg Cost: ¥5.0
# - Avg Quality: 98.0

# AutoGPT
# - Success Rate: 75%
# - Avg Time: 45.0s
# - Avg Cost: ¥8.0
# - Avg Quality: 80.0

# GitHub Copilot
# - Success Rate: 85%
# - Avg Time: 3.0s
# - Avg Cost: ¥1.0
# - Avg Quality: 85.0
```

### Running Comparison

```python
from benchmarks.benchmark_framework import compare_with_baseline

# Run benchmark
report = await runner.run_benchmark_suite(tasks, shogun, max_concurrent=5)

# Compare with Claude Code
comparison = compare_with_baseline(report, BASELINE_CLAUDE_CODE)

print("\nComparison with Claude Code:")
for metric, data in comparison["metrics"].items():
    bushidan = data["bushidan"]
    baseline = data["baseline"]
    improvement = data["improvement"]
    
    print(f"\n{metric}:")
    print(f"  Bushidan:    {bushidan}")
    print(f"  Claude Code: {baseline}")
    print(f"  Difference:  {improvement:+.1f}%")
```

## Custom Benchmarks

### Creating Custom Task Sets

```python
from benchmarks.benchmark_framework import BenchmarkTask

custom_tasks = [
    BenchmarkTask(
        id="custom_001",
        category="domain_specific",
        difficulty="medium",
        content="Implement domain-specific logic for my use case",
        evaluation_criteria={
            "correctness": True,
            "performance": True,
            "maintainability": True
        }
    ),
    # Add more tasks...
]

# Run custom benchmark
report = await runner.run_benchmark_suite(
    tasks=custom_tasks,
    system_processor=shogun,
    max_concurrent=3
)
```

### Custom Evaluation

```python
class CustomBenchmarkRunner(BenchmarkRunner):
    """Custom benchmark runner with domain-specific evaluation"""
    
    def _evaluate_quality(self, task, result):
        """Custom quality evaluation"""
        
        # Use parent evaluation
        base_score = super()._evaluate_quality(task, result)
        
        # Add custom checks
        if task.category == "domain_specific":
            # Check domain-specific criteria
            if self._meets_domain_requirements(result):
                base_score += 5.0
        
        return min(base_score, 100.0)
    
    def _meets_domain_requirements(self, result):
        """Check domain-specific requirements"""
        # Custom logic here
        return True
```

## Best Practices

### 1. Regular Benchmarking

✅ **Do**: Run benchmarks regularly
```bash
# Daily quick benchmark
python benchmarks/run_quick_benchmark.py

# Weekly full benchmark
python benchmarks/run_full_benchmark.py
```

### 2. Track Trends

✅ **Do**: Save all benchmark results
```python
# Save with timestamps
output = Path(f"results/{datetime.now().isoformat()}.json")
runner.save_report(report, output)
```

### 3. Compare with Baselines

✅ **Do**: Always compare with baseline systems
```python
comparison = compare_with_baseline(report, BASELINE_CLAUDE_CODE)
```

### 4. Test Representative Workloads

✅ **Do**: Use tasks similar to production workload
```python
# Add custom tasks that match your use case
my_tasks = [
    BenchmarkTask(id="prod_001", category="my_domain", ...),
    # ...
]
```

### 5. Monitor Key Metrics

**Focus on**:
- Success rate (target: 90%+)
- Average time vs targets (Simple: 2s, Medium: 12s, Complex: 28s)
- Cost per task (target: ¥1-2)
- Quality score (target: 85%+)

### 6. Investigate Regressions

❗ **When metrics degrade**:
1. Compare with previous reports
2. Check specific failing tasks
3. Review route distribution changes
4. Analyze error messages

## Resources

- [Benchmark Framework Code](../../benchmarks/benchmark_framework.py)
- [Standard Tasks](../../benchmarks/standard_tasks.py)
- [Results Directory](../../benchmarks/results/)
- [CI Benchmark Workflow](../ci-workflow.yml)
