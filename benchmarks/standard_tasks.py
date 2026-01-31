"""
Standard Benchmark Task Sets

Comprehensive collection of benchmark tasks across various categories
and difficulty levels for evaluating Bushidan Multi-Agent System performance.
"""

from typing import List
from benchmarks.benchmark_framework import BenchmarkTask


def get_code_generation_tasks() -> List[BenchmarkTask]:
    """Code generation benchmark tasks"""
    
    return [
        # Simple
        BenchmarkTask(
            id="code_gen_001",
            category="code_generation",
            difficulty="simple",
            content="Write a Python function that returns 'Hello, World!'"
        ),
        BenchmarkTask(
            id="code_gen_002",
            category="code_generation",
            difficulty="simple",
            content="Create a function to check if a number is even"
        ),
        BenchmarkTask(
            id="code_gen_003",
            category="code_generation",
            difficulty="simple",
            content="Write a function to convert Celsius to Fahrenheit"
        ),
        
        # Medium
        BenchmarkTask(
            id="code_gen_004",
            category="code_generation",
            difficulty="medium",
            content="Implement a function to calculate Fibonacci numbers up to n using dynamic programming"
        ),
        BenchmarkTask(
            id="code_gen_005",
            category="code_generation",
            difficulty="medium",
            content="Create a Python class for a stack data structure with push, pop, and peek methods"
        ),
        BenchmarkTask(
            id="code_gen_006",
            category="code_generation",
            difficulty="medium",
            content="Write a function to validate email addresses using regex"
        ),
        BenchmarkTask(
            id="code_gen_007",
            category="code_generation",
            difficulty="medium",
            content="Implement a binary search algorithm in Python with proper error handling"
        ),
        
        # Complex
        BenchmarkTask(
            id="code_gen_008",
            category="code_generation",
            difficulty="complex",
            content="Create a REST API endpoint for user authentication with JWT tokens including login, logout, and token refresh"
        ),
        BenchmarkTask(
            id="code_gen_009",
            category="code_generation",
            difficulty="complex",
            content="Implement a thread-safe LRU cache in Python with configurable capacity and TTL"
        ),
        BenchmarkTask(
            id="code_gen_010",
            category="code_generation",
            difficulty="complex",
            content="Build a decorator that implements retry logic with exponential backoff for async functions"
        ),
    ]


def get_qa_tasks() -> List[BenchmarkTask]:
    """Question answering benchmark tasks"""
    
    return [
        # Simple
        BenchmarkTask(
            id="qa_001",
            category="qa",
            difficulty="simple",
            content="What is the capital of Japan?"
        ),
        BenchmarkTask(
            id="qa_002",
            category="qa",
            difficulty="simple",
            content="How many days are in a leap year?"
        ),
        BenchmarkTask(
            id="qa_003",
            category="qa",
            difficulty="simple",
            content="What does HTTP stand for?"
        ),
        
        # Medium
        BenchmarkTask(
            id="qa_004",
            category="qa",
            difficulty="medium",
            content="Explain the difference between TCP and UDP protocols"
        ),
        BenchmarkTask(
            id="qa_005",
            category="qa",
            difficulty="medium",
            content="What are the SOLID principles in software engineering?"
        ),
        BenchmarkTask(
            id="qa_006",
            category="qa",
            difficulty="medium",
            content="How does garbage collection work in Python?"
        ),
        
        # Complex
        BenchmarkTask(
            id="qa_007",
            category="qa",
            difficulty="complex",
            content="Compare and contrast microservices vs monolithic architecture, including trade-offs"
        ),
        BenchmarkTask(
            id="qa_008",
            category="qa",
            difficulty="complex",
            content="Explain the CAP theorem and its implications for distributed systems design"
        ),
    ]


def get_refactoring_tasks() -> List[BenchmarkTask]:
    """Code refactoring benchmark tasks"""
    
    return [
        # Medium
        BenchmarkTask(
            id="refactor_001",
            category="refactoring",
            difficulty="medium",
            content="""Refactor this Python function to be more readable and efficient:

def f(x):
    r = []
    for i in range(len(x)):
        if x[i] % 2 == 0:
            r.append(x[i] * 2)
    return r
"""
        ),
        BenchmarkTask(
            id="refactor_002",
            category="refactoring",
            difficulty="medium",
            content="""Refactor this code to use better error handling:

def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""
        ),
        
        # Complex
        BenchmarkTask(
            id="refactor_003",
            category="refactoring",
            difficulty="complex",
            content="""Refactor this class to follow SOLID principles:

class UserManager:
    def create_user(self, data):
        # validate data
        # save to database
        # send email
        # log activity
        pass
    
    def delete_user(self, user_id):
        # delete from database
        # send email
        # log activity
        pass
"""
        ),
    ]


def get_debugging_tasks() -> List[BenchmarkTask]:
    """Debugging and error fixing tasks"""
    
    return [
        # Medium
        BenchmarkTask(
            id="debug_001",
            category="debugging",
            difficulty="medium",
            content="""Find and fix the bug in this code:

def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

print(calculate_average([]))
"""
        ),
        BenchmarkTask(
            id="debug_002",
            category="debugging",
            difficulty="medium",
            content="""Identify the security vulnerability in this code:

import os

def execute_command(user_input):
    os.system(user_input)
    return "Command executed"
"""
        ),
        
        # Complex
        BenchmarkTask(
            id="debug_003",
            category="debugging",
            difficulty="complex",
            content="""Debug this async code that has a race condition:

import asyncio

counter = 0

async def increment():
    global counter
    temp = counter
    await asyncio.sleep(0.1)
    counter = temp + 1

async def main():
    await asyncio.gather(*[increment() for _ in range(10)])
    print(f"Final counter: {counter}")
"""
        ),
    ]


def get_architecture_tasks() -> List[BenchmarkTask]:
    """Architecture and design tasks"""
    
    return [
        # Complex
        BenchmarkTask(
            id="arch_001",
            category="architecture",
            difficulty="complex",
            content="Design a scalable URL shortener service handling 1000 requests/second"
        ),
        BenchmarkTask(
            id="arch_002",
            category="architecture",
            difficulty="complex",
            content="Design a real-time chat system with support for 100k concurrent users"
        ),
        
        # Strategic
        BenchmarkTask(
            id="arch_003",
            category="architecture",
            difficulty="strategic",
            content="Design the overall architecture for a new e-commerce platform with requirements for: multi-region deployment, PCI compliance, 99.99% uptime, and ability to handle Black Friday traffic spikes"
        ),
        BenchmarkTask(
            id="arch_004",
            category="architecture",
            difficulty="strategic",
            content="Recommend a complete technology stack and architecture for a startup building a SaaS product with the following constraints: team of 5 developers, 6-month MVP timeline, budget of $100k, need for rapid iteration"
        ),
    ]


def get_optimization_tasks() -> List[BenchmarkTask]:
    """Performance optimization tasks"""
    
    return [
        # Medium
        BenchmarkTask(
            id="opt_001",
            category="optimization",
            difficulty="medium",
            content="""Optimize this database query:

SELECT * FROM users 
WHERE created_at > '2024-01-01' 
AND status = 'active' 
AND country IN ('US', 'UK', 'JP')
ORDER BY created_at DESC
"""
        ),
        
        # Complex
        BenchmarkTask(
            id="opt_002",
            category="optimization",
            difficulty="complex",
            content="""Optimize this Python code for performance:

def find_duplicates(data):
    duplicates = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if data[i] == data[j] and data[i] not in duplicates:
                duplicates.append(data[i])
    return duplicates

# Called with lists of 10,000+ items
"""
        ),
    ]


def get_testing_tasks() -> List[BenchmarkTask]:
    """Testing and quality assurance tasks"""
    
    return [
        # Medium
        BenchmarkTask(
            id="test_001",
            category="testing",
            difficulty="medium",
            content="Write pytest unit tests for a function that validates credit card numbers using the Luhn algorithm"
        ),
        BenchmarkTask(
            id="test_002",
            category="testing",
            difficulty="medium",
            content="Create integration tests for a REST API endpoint that handles user registration"
        ),
        
        # Complex
        BenchmarkTask(
            id="test_003",
            category="testing",
            difficulty="complex",
            content="Design a comprehensive test strategy for a payment processing system including unit, integration, and end-to-end tests"
        ),
    ]


def get_documentation_tasks() -> List[BenchmarkTask]:
    """Documentation writing tasks"""
    
    return [
        # Simple
        BenchmarkTask(
            id="doc_001",
            category="documentation",
            difficulty="simple",
            content="Write a docstring for a function that calculates compound interest"
        ),
        
        # Medium
        BenchmarkTask(
            id="doc_002",
            category="documentation",
            difficulty="medium",
            content="Create a README.md for a Python CLI tool that converts CSV to JSON"
        ),
        BenchmarkTask(
            id="doc_003",
            category="documentation",
            difficulty="medium",
            content="Write API documentation for a user authentication endpoint including request/response examples"
        ),
        
        # Complex
        BenchmarkTask(
            id="doc_004",
            category="documentation",
            difficulty="complex",
            content="Create comprehensive developer documentation for onboarding new team members to a microservices platform"
        ),
    ]


def get_japanese_tasks() -> List[BenchmarkTask]:
    """Japanese language tasks to test multilingual support"""
    
    return [
        # Simple
        BenchmarkTask(
            id="jp_001",
            category="japanese",
            difficulty="simple",
            content="Pythonで「こんにちは世界」を表示する関数を書いてください"
        ),
        
        # Medium
        BenchmarkTask(
            id="jp_002",
            category="japanese",
            difficulty="medium",
            content="Pythonでフィボナッチ数列を計算する関数を実装してください。再帰とループの両方の方法で実装し、それぞれの利点と欠点を説明してください"
        ),
        
        # Complex
        BenchmarkTask(
            id="jp_003",
            category="japanese",
            difficulty="complex",
            content="RESTful APIを使ったユーザー認証システムを設計してください。JWT トークンを使用し、ログイン、ログアウト、トークンリフレッシュの機能を含めてください"
        ),
        
        # Strategic
        BenchmarkTask(
            id="jp_004",
            category="japanese",
            difficulty="strategic",
            content="新しいECプラットフォームのアーキテクチャを設計してください。要件：マルチリージョン展開、99.99%の稼働率、ブラックフライデーのトラフィック急増に対応"
        ),
    ]


def get_all_standard_tasks() -> List[BenchmarkTask]:
    """Get complete standard benchmark task set"""
    
    all_tasks = []
    all_tasks.extend(get_code_generation_tasks())
    all_tasks.extend(get_qa_tasks())
    all_tasks.extend(get_refactoring_tasks())
    all_tasks.extend(get_debugging_tasks())
    all_tasks.extend(get_architecture_tasks())
    all_tasks.extend(get_optimization_tasks())
    all_tasks.extend(get_testing_tasks())
    all_tasks.extend(get_documentation_tasks())
    all_tasks.extend(get_japanese_tasks())
    
    return all_tasks


def get_quick_benchmark_tasks() -> List[BenchmarkTask]:
    """Get a quick benchmark subset (10 tasks, fast execution)"""
    
    return [
        # 2 Simple
        BenchmarkTask(
            id="quick_001",
            category="code_generation",
            difficulty="simple",
            content="Write a function to check if a string is a palindrome"
        ),
        BenchmarkTask(
            id="quick_002",
            category="qa",
            difficulty="simple",
            content="What is the difference between a list and a tuple in Python?"
        ),
        
        # 4 Medium
        BenchmarkTask(
            id="quick_003",
            category="code_generation",
            difficulty="medium",
            content="Implement a function to merge two sorted lists"
        ),
        BenchmarkTask(
            id="quick_004",
            category="refactoring",
            difficulty="medium",
            content="Refactor this code to use list comprehension: result = []; for x in range(10): if x % 2 == 0: result.append(x**2)"
        ),
        BenchmarkTask(
            id="quick_005",
            category="debugging",
            difficulty="medium",
            content="Fix the bug: def factorial(n): if n == 0: return 1; return n * factorial(n)"
        ),
        BenchmarkTask(
            id="quick_006",
            category="testing",
            difficulty="medium",
            content="Write a pytest test for a function that validates email addresses"
        ),
        
        # 3 Complex
        BenchmarkTask(
            id="quick_007",
            category="code_generation",
            difficulty="complex",
            content="Implement a rate limiter using the token bucket algorithm"
        ),
        BenchmarkTask(
            id="quick_008",
            category="architecture",
            difficulty="complex",
            content="Design a caching strategy for a high-traffic API"
        ),
        BenchmarkTask(
            id="quick_009",
            category="optimization",
            difficulty="complex",
            content="Optimize a function that finds all prime numbers up to n (currently uses nested loops)"
        ),
        
        # 1 Strategic
        BenchmarkTask(
            id="quick_010",
            category="architecture",
            difficulty="strategic",
            content="Design the technology stack and deployment strategy for a new startup's MVP"
        ),
    ]


def get_stress_test_tasks() -> List[BenchmarkTask]:
    """Get stress test tasks (large, complex tasks)"""
    
    return [
        BenchmarkTask(
            id="stress_001",
            category="code_generation",
            difficulty="complex",
            content="Implement a complete CRUD REST API for a blog platform with posts, comments, users, tags, and categories. Include authentication, authorization, pagination, search, and rate limiting."
        ),
        BenchmarkTask(
            id="stress_002",
            category="architecture",
            difficulty="strategic",
            content="Design a complete microservices architecture for a video streaming platform like Netflix, including CDN strategy, recommendation engine, user authentication, payment processing, content management, and analytics. Address scalability, fault tolerance, and cost optimization."
        ),
        BenchmarkTask(
            id="stress_003",
            category="refactoring",
            difficulty="complex",
            content="Refactor a legacy monolithic application (3000+ lines) into a clean, modular architecture following Domain-Driven Design principles. Include migration strategy, testing approach, and rollback plan."
        ),
    ]


# Baseline comparison data (for reference)
BASELINE_CLAUDE_CODE = {
    "system_name": "Claude Code (Sonnet 4.5)",
    "success_rate": 0.98,
    "average_time_seconds": 8.0,
    "average_cost_per_task": 5.0,
    "average_quality_score": 98.0
}

BASELINE_AUTOGPT = {
    "system_name": "AutoGPT",
    "success_rate": 0.75,
    "average_time_seconds": 45.0,
    "average_cost_per_task": 8.0,
    "average_quality_score": 80.0
}

BASELINE_COPILOT = {
    "system_name": "GitHub Copilot",
    "success_rate": 0.85,
    "average_time_seconds": 3.0,
    "average_cost_per_task": 1.0,
    "average_quality_score": 85.0
}
