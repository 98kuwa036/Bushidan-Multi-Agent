#!/usr/bin/env python3
"""
Bushidan Multi-Agent System v9.1 - CLI Interface

Simplified command-line interface for the Universal Multi-LLM Framework.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from utils.config import load_config
from utils.logger import setup_logger
from core.shogun import Shogun, Task, TaskComplexity
from core.system_orchestrator import SystemOrchestrator


async def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="🏯 Bushidan Multi-Agent System v9.1 CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Interactive mode
    parser_interactive = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Task execution
    parser_task = subparsers.add_parser("task", help="Execute single task")
    parser_task.add_argument("content", help="Task description")
    parser_task.add_argument("--complexity", choices=["simple", "medium", "complex", "strategic"], 
                           default="medium", help="Task complexity")
    parser_task.add_argument("--source", default="cli", help="Task source")
    
    # System status
    parser_status = subparsers.add_parser("status", help="Show system status")
    
    # Health check
    parser_health = subparsers.add_parser("health", help="Run health check")
    
    # Memory stats
    parser_memory = subparsers.add_parser("memory", help="Show memory statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logger = setup_logger("bushidan_cli")
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize system
        orchestrator = SystemOrchestrator(config)
        await orchestrator.initialize()
        
        shogun = Shogun(orchestrator)
        await shogun.initialize()
        
        # Execute command
        if args.command == "interactive":
            await interactive_mode(shogun)
        elif args.command == "task":
            await execute_task(shogun, args)
        elif args.command == "status":
            await show_status(orchestrator)
        elif args.command == "health":
            await health_check(orchestrator)
        elif args.command == "memory":
            await memory_stats(orchestrator)
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ CLI error: {e}")
        sys.exit(1)


async def interactive_mode(shogun: Shogun):
    """Interactive command mode"""
    
    print("🏯 Bushidan v9.1 Interactive Mode")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("🎌 Shogun> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Sayonara!")
                break
            
            if user_input.lower() == "help":
                print_help()
                continue
            
            # Create and execute task
            task = Task(
                content=user_input,
                complexity=TaskComplexity.MEDIUM,
                source="interactive"
            )
            
            print("🔄 Processing...")
            result = await shogun.process_task(task)
            
            print(f"✅ Result:")
            print(result.get("result", "No result available"))
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Sayonara!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def execute_task(shogun: Shogun, args):
    """Execute a single task"""
    
    complexity = TaskComplexity(args.complexity)
    
    task = Task(
        content=args.content,
        complexity=complexity,
        source=args.source
    )
    
    print(f"🎌 Executing task: {task.content[:50]}...")
    print(f"📊 Complexity: {complexity.value}")
    
    result = await shogun.process_task(task)
    
    print("✅ Task completed!")
    print("📋 Result:")
    print(result.get("result", "No result available"))
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)


async def show_status(orchestrator: SystemOrchestrator):
    """Show system status"""
    
    print("🏯 Bushidan v9.1 System Status")
    print("=" * 40)
    
    print(f"📊 Mode: {orchestrator.config.mode.value}")
    print(f"🔧 Initialized: {orchestrator.initialized}")
    print(f"📝 MCPs: {len(orchestrator.mcps)}")
    
    for name, mcp in orchestrator.mcps.items():
        print(f"  • {name.title()} MCP: ✅")
    
    print("=" * 40)


async def health_check(orchestrator: SystemOrchestrator):
    """Run system health check"""
    
    print("🏥 Running Health Check...")
    print("-" * 30)
    
    # Check MCP health
    all_healthy = True
    
    for name, mcp in orchestrator.mcps.items():
        try:
            # Basic health check (if MCP supports it)
            if hasattr(mcp, "health_check"):
                healthy = await mcp.health_check()
                status = "✅" if healthy else "❌"
            else:
                status = "✅"  # Assume healthy if no check method
            
            print(f"{status} {name.title()} MCP")
            
            if status == "❌":
                all_healthy = False
                
        except Exception as e:
            print(f"❌ {name.title()} MCP: {e}")
            all_healthy = False
    
    print("-" * 30)
    if all_healthy:
        print("🎉 All systems healthy!")
    else:
        print("⚠️ Some systems need attention")
        sys.exit(1)


async def memory_stats(orchestrator: SystemOrchestrator):
    """Show memory statistics"""
    
    memory_mcp = orchestrator.get_mcp("memory")
    
    if not memory_mcp:
        print("❌ Memory MCP not available")
        return
    
    print("📚 Memory Statistics")
    print("=" * 25)
    
    try:
        stats = await memory_mcp.get_stats()
        
        print(f"📝 Total entries: {stats.get('total_entries', 0)}")
        print(f"💾 File size: {stats.get('file_size', 0)} bytes")
        print(f"📁 File path: {stats.get('file_path', 'Unknown')}")
        
        categories = stats.get("categories", {})
        if categories:
            print("\n📊 Categories:")
            for category, count in categories.items():
                print(f"  • {category}: {count}")
                
    except Exception as e:
        print(f"❌ Error getting memory stats: {e}")


def print_help():
    """Print interactive help"""
    
    print("""
🏯 Bushidan v9.1 Interactive Commands:

Basic Commands:
  help              - Show this help
  quit, exit, q     - Exit interactive mode
  
Task Examples:
  Create a Python function to calculate fibonacci
  Explain quantum computing in simple terms  
  Design a REST API for user management
  
Complexity Levels:
  • Simple (10s): Direct answers, info queries
  • Medium (25s): Standard implementation  
  • Complex (40s): Multi-component systems
  • Strategic (60s): High-level decisions
    """)


if __name__ == "__main__":
    asyncio.run(main())