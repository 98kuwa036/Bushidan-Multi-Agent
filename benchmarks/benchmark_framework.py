"""
Bushidan Multi-Agent System - Benchmark Framework

Comprehensive benchmarking system for measuring and comparing performance:
- Task success rates
- Processing times
- Cost per task
- Quality scores
- Comparison with baseline systems
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import statistics


@dataclass
class BenchmarkTask:
    """Individual benchmark task"""
    id: str
    category: str  # "code_generation", "qa", "refactoring", etc.
    difficulty: str  # "simple", "medium", "complex", "strategic"
    content: str
    expected_output: Optional[str] = None
    evaluation_criteria: Optional[Dict[str, Any]] = None


@dataclass
class TaskResult:
    """Result of running a single benchmark task"""
    task_id: str
    status: str  # "success", "failure", "error"
    processing_time_seconds: float
    cost_yen: float
    quality_score: float  # 0-100
    route_used: str  # "groq", "local_qwen3", "cloud_qwen3", "gemini3", "shogun"
    fallbacks_triggered: int
    error_message: Optional[str] = None
    output: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    timestamp: str
    system_version: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    error_tasks: int
    
    # Performance metrics
    average_time_seconds: float
    median_time_seconds: float
    p95_time_seconds: float
    
    # Cost metrics
    total_cost_yen: float
    average_cost_per_task: float
    
    # Quality metrics
    average_quality_score: float
    median_quality_score: float
    min_quality_score: float
    max_quality_score: float
    
    # Reliability metrics
    success_rate: float
    fallback_rate: float
    
    # Route distribution
    route_distribution: Dict[str, int]
    
    # Detailed results
    results_by_category: Dict[str, Dict[str, Any]]
    results_by_difficulty: Dict[str, Dict[str, Any]]
    task_results: List[TaskResult]


class BenchmarkRunner:
    """
    Main benchmark runner for Bushidan Multi-Agent System
    
    Runs standardized benchmark suites and generates comprehensive reports
    comparing performance against targets and baselines.
    """
    
    def __init__(self, system_version: str = "9.3.2"):
        self.system_version = system_version
        self.results: List[TaskResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    async def run_benchmark_suite(
        self,
        tasks: List[BenchmarkTask],
        system_processor,
        max_concurrent: int = 5
    ) -> BenchmarkReport:
        """
        Run a complete benchmark suite
        
        Args:
            tasks: List of benchmark tasks to run
            system_processor: The Bushidan system to benchmark (e.g., Shogun instance)
            max_concurrent: Maximum concurrent tasks
        
        Returns:
            Comprehensive benchmark report
        """
        
        print(f"\n🏯 Starting Bushidan Benchmark Suite")
        print(f"Version: {self.system_version}")
        print(f"Tasks: {len(tasks)}")
        print(f"Max Concurrent: {max_concurrent}\n")
        
        self.start_time = time.time()
        self.results = []
        
        # Run tasks with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_task_with_semaphore(task: BenchmarkTask):
            async with semaphore:
                return await self._run_single_task(task, system_processor)
        
        # Execute all tasks
        task_results = await asyncio.gather(*[
            run_task_with_semaphore(task) for task in tasks
        ], return_exceptions=True)
        
        # Process results
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                # Handle exceptions
                self.results.append(TaskResult(
                    task_id=tasks[i].id,
                    status="error",
                    processing_time_seconds=0.0,
                    cost_yen=0.0,
                    quality_score=0.0,
                    route_used="none",
                    fallbacks_triggered=0,
                    error_message=str(result)
                ))
            else:
                self.results.append(result)
        
        self.end_time = time.time()
        
        # Generate report
        report = self._generate_report(tasks)
        
        print(f"\n✅ Benchmark Complete")
        print(f"Total Time: {self.end_time - self.start_time:.2f}s")
        print(f"Success Rate: {report.success_rate:.1%}")
        print(f"Average Quality: {report.average_quality_score:.1f}/100\n")
        
        return report
    
    async def _run_single_task(
        self,
        task: BenchmarkTask,
        system_processor
    ) -> TaskResult:
        """Run a single benchmark task"""
        
        print(f"  Running: {task.id} ({task.difficulty})")
        
        start_time = time.time()
        
        try:
            # Convert to system task format
            from core.shogun import Task, TaskComplexity
            
            complexity_map = {
                "simple": TaskComplexity.SIMPLE,
                "medium": TaskComplexity.MEDIUM,
                "complex": TaskComplexity.COMPLEX,
                "strategic": TaskComplexity.STRATEGIC
            }
            
            system_task = Task(
                content=task.content,
                complexity=complexity_map.get(task.difficulty, TaskComplexity.MEDIUM),
                context={"benchmark": True, "task_id": task.id}
            )
            
            # Process task
            result = await system_processor.process_task(system_task)
            
            processing_time = time.time() - start_time
            
            # Extract metrics
            status = "success" if result.get("status") == "completed" else "failure"
            cost = self._estimate_cost(result)
            quality = self._evaluate_quality(task, result)
            route = result.get("route", "unknown")
            fallbacks = result.get("fallbacks_triggered", 0)
            
            return TaskResult(
                task_id=task.id,
                status=status,
                processing_time_seconds=processing_time,
                cost_yen=cost,
                quality_score=quality,
                route_used=route,
                fallbacks_triggered=fallbacks,
                output=result.get("result")
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.id,
                status="error",
                processing_time_seconds=processing_time,
                cost_yen=0.0,
                quality_score=0.0,
                route_used="none",
                fallbacks_triggered=0,
                error_message=str(e)
            )
    
    def _estimate_cost(self, result: Dict[str, Any]) -> float:
        """Estimate cost of a task execution"""
        
        cost = 0.0
        
        # Opus review cost
        if "opus_review" in result:
            cost += result["opus_review"].get("cost_yen", 10.0)
        
        # Route-specific costs
        route = result.get("route", "")
        if "cloud_qwen" in route:
            cost += 3.0
        elif "gemini" in route:
            cost += 0.04
        
        # API costs (if any)
        if result.get("api_calls", 0) > 0:
            cost += result["api_calls"] * 0.5  # Estimate
        
        return cost
    
    def _evaluate_quality(self, task: BenchmarkTask, result: Dict[str, Any]) -> float:
        """Evaluate quality of task result"""
        
        # If Opus reviewed, use its score
        if "opus_review" in result:
            return result["opus_review"].get("score", 85.0)
        
        # If Sonnet reviewed, use its score
        if "sonnet_detailed_review" in result:
            return result["sonnet_detailed_review"].get("score", 85.0)
        
        # If quality metrics available
        if "quality_metrics" in result:
            return result["quality_metrics"].get("overall_score", 85.0)
        
        # Default based on status
        if result.get("status") == "completed":
            return 85.0  # Assume good quality if completed
        else:
            return 50.0  # Lower quality for failures
    
    def _generate_report(self, tasks: List[BenchmarkTask]) -> BenchmarkReport:
        """Generate comprehensive benchmark report"""
        
        # Basic counts
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r.status == "success")
        failed_tasks = sum(1 for r in self.results if r.status == "failure")
        error_tasks = sum(1 for r in self.results if r.status == "error")
        
        # Time metrics
        times = [r.processing_time_seconds for r in self.results]
        avg_time = statistics.mean(times) if times else 0.0
        median_time = statistics.median(times) if times else 0.0
        p95_time = self._percentile(times, 95) if times else 0.0
        
        # Cost metrics
        costs = [r.cost_yen for r in self.results]
        total_cost = sum(costs)
        avg_cost = statistics.mean(costs) if costs else 0.0
        
        # Quality metrics
        qualities = [r.quality_score for r in self.results if r.status == "success"]
        avg_quality = statistics.mean(qualities) if qualities else 0.0
        median_quality = statistics.median(qualities) if qualities else 0.0
        min_quality = min(qualities) if qualities else 0.0
        max_quality = max(qualities) if qualities else 0.0
        
        # Reliability metrics
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        fallback_count = sum(r.fallbacks_triggered for r in self.results)
        fallback_rate = fallback_count / total_tasks if total_tasks > 0 else 0.0
        
        # Route distribution
        route_dist = {}
        for result in self.results:
            route = result.route_used
            route_dist[route] = route_dist.get(route, 0) + 1
        
        # Category analysis
        results_by_category = self._analyze_by_category(tasks)
        results_by_difficulty = self._analyze_by_difficulty(tasks)
        
        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            system_version=self.system_version,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            error_tasks=error_tasks,
            average_time_seconds=avg_time,
            median_time_seconds=median_time,
            p95_time_seconds=p95_time,
            total_cost_yen=total_cost,
            average_cost_per_task=avg_cost,
            average_quality_score=avg_quality,
            median_quality_score=median_quality,
            min_quality_score=min_quality,
            max_quality_score=max_quality,
            success_rate=success_rate,
            fallback_rate=fallback_rate,
            route_distribution=route_dist,
            results_by_category=results_by_category,
            results_by_difficulty=results_by_difficulty,
            task_results=self.results
        )
    
    def _analyze_by_category(self, tasks: List[BenchmarkTask]) -> Dict[str, Dict[str, Any]]:
        """Analyze results by category"""
        
        categories = {}
        task_map = {t.id: t for t in tasks}
        
        for result in self.results:
            task = task_map.get(result.task_id)
            if not task:
                continue
            
            category = task.category
            if category not in categories:
                categories[category] = {
                    "count": 0,
                    "successful": 0,
                    "avg_time": 0.0,
                    "avg_quality": 0.0
                }
            
            cat_data = categories[category]
            cat_data["count"] += 1
            if result.status == "success":
                cat_data["successful"] += 1
        
        # Calculate averages
        for category, data in categories.items():
            cat_results = [r for r in self.results if task_map.get(r.task_id) and task_map[r.task_id].category == category]
            times = [r.processing_time_seconds for r in cat_results]
            qualities = [r.quality_score for r in cat_results if r.status == "success"]
            
            data["avg_time"] = statistics.mean(times) if times else 0.0
            data["avg_quality"] = statistics.mean(qualities) if qualities else 0.0
            data["success_rate"] = data["successful"] / data["count"] if data["count"] > 0 else 0.0
        
        return categories
    
    def _analyze_by_difficulty(self, tasks: List[BenchmarkTask]) -> Dict[str, Dict[str, Any]]:
        """Analyze results by difficulty"""
        
        difficulties = {}
        task_map = {t.id: t for t in tasks}
        
        for result in self.results:
            task = task_map.get(result.task_id)
            if not task:
                continue
            
            difficulty = task.difficulty
            if difficulty not in difficulties:
                difficulties[difficulty] = {
                    "count": 0,
                    "successful": 0,
                    "avg_time": 0.0,
                    "avg_cost": 0.0,
                    "avg_quality": 0.0
                }
            
            diff_data = difficulties[difficulty]
            diff_data["count"] += 1
            if result.status == "success":
                diff_data["successful"] += 1
        
        # Calculate averages
        for difficulty, data in difficulties.items():
            diff_results = [r for r in self.results if task_map.get(r.task_id) and task_map[r.task_id].difficulty == difficulty]
            times = [r.processing_time_seconds for r in diff_results]
            costs = [r.cost_yen for r in diff_results]
            qualities = [r.quality_score for r in diff_results if r.status == "success"]
            
            data["avg_time"] = statistics.mean(times) if times else 0.0
            data["avg_cost"] = statistics.mean(costs) if costs else 0.0
            data["avg_quality"] = statistics.mean(qualities) if qualities else 0.0
            data["success_rate"] = data["successful"] / data["count"] if data["count"] > 0 else 0.0
        
        return difficulties
    
    def save_report(self, report: BenchmarkReport, output_path: Path) -> None:
        """Save benchmark report to file"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        report_dict = asdict(report)
        report_dict["task_results"] = [asdict(r) for r in report.task_results]
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"📊 Report saved to: {output_path}")
    
    def print_summary(self, report: BenchmarkReport) -> None:
        """Print benchmark summary to console"""
        
        print("\n" + "="*80)
        print(f"🏯 BUSHIDAN BENCHMARK REPORT - {self.system_version}")
        print("="*80)
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Total Tasks:      {report.total_tasks}")
        print(f"  Successful:       {report.successful_tasks} ({report.success_rate:.1%})")
        print(f"  Failed:           {report.failed_tasks}")
        print(f"  Errors:           {report.error_tasks}")
        
        print(f"\n⚡ PERFORMANCE:")
        print(f"  Average Time:     {report.average_time_seconds:.2f}s")
        print(f"  Median Time:      {report.median_time_seconds:.2f}s")
        print(f"  P95 Time:         {report.p95_time_seconds:.2f}s")
        
        print(f"\n💰 COST:")
        print(f"  Total Cost:       ¥{report.total_cost_yen:.2f}")
        print(f"  Cost per Task:    ¥{report.average_cost_per_task:.2f}")
        
        print(f"\n🎯 QUALITY:")
        print(f"  Average:          {report.average_quality_score:.1f}/100")
        print(f"  Median:           {report.median_quality_score:.1f}/100")
        print(f"  Range:            {report.min_quality_score:.1f} - {report.max_quality_score:.1f}")
        
        print(f"\n🔀 ROUTING:")
        for route, count in sorted(report.route_distribution.items(), key=lambda x: -x[1]):
            percentage = count / report.total_tasks * 100
            print(f"  {route:15s}: {count:3d} ({percentage:5.1f}%)")
        
        print(f"\n📁 BY DIFFICULTY:")
        for diff, data in report.results_by_difficulty.items():
            print(f"  {diff:10s}: {data['successful']}/{data['count']} success, "
                  f"avg {data['avg_time']:.1f}s, "
                  f"quality {data['avg_quality']:.1f}")
        
        print("\n" + "="*80 + "\n")
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]


def compare_with_baseline(
    bushidan_report: BenchmarkReport,
    baseline_report: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare Bushidan performance with baseline system
    
    Args:
        bushidan_report: Bushidan benchmark report
        baseline_report: Baseline system report (e.g., Claude Code, AutoGPT)
    
    Returns:
        Comparison analysis
    """
    
    comparison = {
        "system": "Bushidan vs " + baseline_report.get("system_name", "Baseline"),
        "metrics": {}
    }
    
    # Compare success rate
    baseline_success = baseline_report.get("success_rate", 0.95)
    comparison["metrics"]["success_rate"] = {
        "bushidan": bushidan_report.success_rate,
        "baseline": baseline_success,
        "difference": bushidan_report.success_rate - baseline_success,
        "improvement": ((bushidan_report.success_rate - baseline_success) / baseline_success * 100)
            if baseline_success > 0 else 0
    }
    
    # Compare processing time
    baseline_time = baseline_report.get("average_time_seconds", 10.0)
    comparison["metrics"]["processing_time"] = {
        "bushidan": bushidan_report.average_time_seconds,
        "baseline": baseline_time,
        "difference": bushidan_report.average_time_seconds - baseline_time,
        "improvement": ((baseline_time - bushidan_report.average_time_seconds) / baseline_time * 100)
            if baseline_time > 0 else 0
    }
    
    # Compare cost
    baseline_cost = baseline_report.get("average_cost_per_task", 5.0)
    comparison["metrics"]["cost_per_task"] = {
        "bushidan": bushidan_report.average_cost_per_task,
        "baseline": baseline_cost,
        "difference": bushidan_report.average_cost_per_task - baseline_cost,
        "improvement": ((baseline_cost - bushidan_report.average_cost_per_task) / baseline_cost * 100)
            if baseline_cost > 0 else 0
    }
    
    # Compare quality
    baseline_quality = baseline_report.get("average_quality_score", 90.0)
    comparison["metrics"]["quality_score"] = {
        "bushidan": bushidan_report.average_quality_score,
        "baseline": baseline_quality,
        "difference": bushidan_report.average_quality_score - baseline_quality,
        "improvement": ((bushidan_report.average_quality_score - baseline_quality) / baseline_quality * 100)
            if baseline_quality > 0 else 0
    }
    
    return comparison
