"""
将軍システム v8.0 - Main Application Entry Point
メインアプリケーション: v8.0統合システムの起動・制御

Features:
- Complete v8.0 system orchestration
- Multi-mode deployment (Battalion/Company/Platoon)
- REST API endpoints for external access
- Slack Bot integration
- Health monitoring and metrics
- Graceful shutdown handling
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Local imports
sys.path.append(str(Path(__file__).parent))

from core.system_orchestrator import SystemOrchestrator, create_system_orchestrator
from core.activity_memory import TaskComplexity
from agents.taisho_v8 import TaskPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global system orchestrator
system_orchestrator: Optional[SystemOrchestrator] = None

# FastAPI app
app = FastAPI(
    title="将軍システム v8.0 API",
    description="Complete AI Development Environment - Shogun System v8.0",
    version="8.0.0"
)


# Request/Response Models
class TaskRequest(BaseModel):
    """Task processing request"""
    description: str
    complexity: str = "medium"  # simple, medium, complex, strategic
    priority: str = "medium"    # low, medium, high, critical
    context: Optional[str] = None
    requires_execution: bool = False
    requires_latest_info: bool = False
    requested_by: str = "api"


class TaskResponse(BaseModel):
    """Task processing response"""
    success: bool
    response: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    tools_used: Optional[list] = None
    similar_tasks_found: Optional[int] = None
    knowledge_retrieved: Optional[int] = None
    code_executed: Optional[bool] = None
    execution_successful: Optional[bool] = None
    task_id: Optional[str] = None
    system_status: Optional[str] = None
    error: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """System status response"""
    system_status: str
    system_mode: str
    version: str
    uptime_hours: float
    component_health: Dict[str, str]
    performance_metrics: Dict[str, Any]
    background_tasks_active: int
    checked_at: str


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "将軍システム v8.0",
        "description": "Complete AI Development Environment",
        "version": "8.0.0",
        "status": system_orchestrator.status.value if system_orchestrator else "not_initialized",
        "endpoints": {
            "/health": "System health check",
            "/status": "Detailed system status",
            "/task": "Process task (POST)",
            "/metrics": "System metrics",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    if not system_orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    status = await system_orchestrator.get_system_status()
    
    if status["system_status"] in ["healthy", "degraded"]:
        return {"status": "ok", "system_status": status["system_status"]}
    else:
        raise HTTPException(
            status_code=503, 
            detail=f"System unhealthy: {status['system_status']}"
        )


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get detailed system status and metrics"""
    if not system_orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = await system_orchestrator.get_system_status()
        return SystemStatusResponse(**status)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/task", response_model=TaskResponse)
async def process_task(request: TaskRequest):
    """Process a task through the v8.0 system"""
    if not system_orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Convert string enums
        complexity_map = {
            "simple": TaskComplexity.SIMPLE,
            "medium": TaskComplexity.MEDIUM,
            "complex": TaskComplexity.COMPLEX,
            "strategic": TaskComplexity.STRATEGIC
        }
        
        priority_map = {
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL
        }
        
        complexity = complexity_map.get(request.complexity.lower(), TaskComplexity.MEDIUM)
        priority = priority_map.get(request.priority.lower(), TaskPriority.MEDIUM)
        
        # Process task
        result = await system_orchestrator.process_task(
            description=request.description,
            complexity=complexity,
            priority=priority,
            context=request.context,
            requires_execution=request.requires_execution,
            requires_latest_info=request.requires_latest_info,
            requested_by=request.requested_by
        )
        
        return TaskResponse(**result)
        
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        return TaskResponse(
            success=False,
            error=str(e),
            system_status=system_orchestrator.status.value if system_orchestrator else "error"
        )


@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics"""
    if not system_orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = await system_orchestrator.get_system_status()
        return {
            "performance_metrics": status["performance_metrics"],
            "component_health": status["component_health"],
            "uptime_hours": status["uptime_hours"],
            "system_mode": status["system_mode"],
            "collected_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/shutdown")
async def shutdown_system(background_tasks: BackgroundTasks):
    """Graceful system shutdown (admin only)"""
    if not system_orchestrator:
        return {"message": "System not running"}
    
    background_tasks.add_task(perform_shutdown)
    return {"message": "Shutdown initiated"}


async def perform_shutdown():
    """Perform graceful shutdown"""
    global system_orchestrator
    if system_orchestrator:
        await system_orchestrator.shutdown()
        system_orchestrator = None


# System Lifecycle Management

async def startup_system():
    """Initialize the v8.0 system on startup"""
    global system_orchestrator
    
    try:
        logger.info("🏯 Starting Shogun System v8.0...")
        
        # Initialize system orchestrator
        config_path = Path(__file__).parent / "config" / "settings_v8.yaml"
        system_orchestrator = await create_system_orchestrator(str(config_path))
        
        if system_orchestrator:
            logger.info("🎌 Shogun System v8.0 startup complete")
            return True
        else:
            logger.error("❌ Failed to initialize system orchestrator")
            return False
            
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        return False


async def shutdown_handler():
    """Handle graceful shutdown"""
    global system_orchestrator
    
    logger.info("🏯 Shutting down Shogun System v8.0...")
    
    if system_orchestrator:
        await system_orchestrator.shutdown()
        system_orchestrator = None
    
    logger.info("🎌 Shogun System v8.0 shutdown complete")


# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    asyncio.create_task(shutdown_handler())
    sys.exit(0)


# CLI Interface Functions

async def cli_interactive_mode():
    """Interactive CLI mode for direct system interaction"""
    if not system_orchestrator:
        print("❌ System not initialized")
        return
    
    print("🏯 将軍システム v8.0 - Interactive Mode")
    print("Type 'exit' to quit, 'help' for commands")
    
    while True:
        try:
            user_input = input("\n侍大将> ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'help':
                print("""
Available commands:
  help              - Show this help
  status            - Show system status
  task <description> - Process a task
  metrics           - Show performance metrics
  exit              - Exit interactive mode
                """)
                continue
            elif user_input.lower() == 'status':
                status = await system_orchestrator.get_system_status()
                print(f"System Status: {status['system_status']}")
                print(f"Uptime: {status['uptime_hours']:.1f} hours")
                print(f"Tasks Processed: {status['performance_metrics']['total_tasks_processed']}")
                continue
            elif user_input.lower() == 'metrics':
                status = await system_orchestrator.get_system_status()
                metrics = status['performance_metrics']
                print(f"📊 System Metrics:")
                print(f"  Total Tasks: {metrics['total_tasks_processed']}")
                print(f"  Success Rate: {metrics.get('success_rate', 0):.1f}%")
                print(f"  Avg Processing Time: {metrics['avg_processing_time_seconds']:.2f}s")
                print(f"  Knowledge Entries: {metrics['knowledge_base_entries']}")
                print(f"  Memory Records: {metrics['activity_memory_records']}")
                continue
            elif user_input.startswith('task '):
                task_description = user_input[5:].strip()
                if task_description:
                    print("🤔 Processing task...")
                    result = await system_orchestrator.process_task(
                        description=task_description,
                        requested_by="cli"
                    )
                    
                    if result['success']:
                        print(f"✅ {result['response']}")
                        print(f"📈 Confidence: {result['confidence']}")
                        print(f"⏱️  Time: {result['processing_time_seconds']:.2f}s")
                        if result['tools_used']:
                            print(f"🛠️  Tools: {', '.join(result['tools_used'])}")
                    else:
                        print(f"❌ Error: {result.get('error', 'Unknown error')}")
                else:
                    print("Please provide a task description")
                continue
            elif user_input:
                # Treat as task
                print("🤔 Processing task...")
                result = await system_orchestrator.process_task(
                    description=user_input,
                    requested_by="cli"
                )
                
                if result['success']:
                    print(f"✅ {result['response']}")
                    print(f"📈 Confidence: {result['confidence']}")
                    print(f"⏱️  Time: {result['processing_time_seconds']:.2f}s")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Exiting interactive mode")


async def main():
    """Main application entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="将軍システム v8.0")
    parser.add_argument("--mode", choices=["server", "cli"], default="server",
                       help="Run mode: server (API) or cli (interactive)")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--config", default="config/settings_v8.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize system
    if not await startup_system():
        logger.error("Failed to start system")
        sys.exit(1)
    
    try:
        if args.mode == "server":
            # Run API server
            logger.info(f"Starting API server on {args.host}:{args.port}")
            
            config = uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        elif args.mode == "cli":
            # Run interactive CLI
            await cli_interactive_mode()
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await shutdown_handler()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)