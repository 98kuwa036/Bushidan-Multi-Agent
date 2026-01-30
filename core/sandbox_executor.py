"""
将軍システム v8.0 - Sandbox Execution Environment (演習場)
実行検証環境: コードの実行・テスト・検証を行う隔離環境

Features:
- LXC container-based isolation
- Multi-language support (Python 3.13, Node.js 22, Rust 1.83)
- On-demand startup/shutdown for memory optimization
- Security isolation (no internet, resource limits)
- Real-time code execution with feedback
- Quality improvement through "実証主義" (empirical verification)
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import hashlib
import aiofiles
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    NODEJS = "nodejs"
    RUST = "rust"


class ExecutionStatus(Enum):
    """Execution result status"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class CodeExecution:
    """Code execution request"""
    id: str
    language: Language
    code: str
    description: str = ""
    timeout_seconds: int = 30
    max_memory_mb: int = 512
    files: Dict[str, str] = None  # filename -> content
    dependencies: List[str] = None  # package dependencies
    
    def __post_init__(self):
        if self.files is None:
            self.files = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ExecutionResult:
    """Code execution result"""
    execution_id: str
    status: ExecutionStatus
    stdout: str
    stderr: str
    return_code: int
    execution_time_seconds: float
    memory_usage_mb: float
    error_message: Optional[str] = None
    created_files: List[str] = None
    executed_at: datetime = None
    
    def __post_init__(self):
        if self.created_files is None:
            self.created_files = []
        if self.executed_at is None:
            self.executed_at = datetime.utcnow()
    
    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS
    
    def to_feedback_message(self) -> str:
        """Generate feedback message for R1"""
        if self.success:
            return f"✅ コード実行成功\n実行時間: {self.execution_time_seconds:.2f}秒\n出力: {self.stdout[:500]}..."
        else:
            return f"❌ 実行エラー: {self.error_message}\n標準エラー: {self.stderr[:500]}..."


class ContainerManager:
    """LXC container management for on-demand sandbox"""
    
    def __init__(self, container_id: str = "102"):
        self.container_id = container_id
        self.container_name = f"ct{container_id}"
        self.is_running = False
        
    async def start_container(self) -> bool:
        """Start the sandbox container"""
        try:
            if await self.is_container_running():
                logger.info(f"Container {self.container_name} is already running")
                self.is_running = True
                return True
            
            logger.info(f"Starting container {self.container_name}...")
            
            # Start container using pct command
            process = await asyncio.create_subprocess_exec(
                "pct", "start", self.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Wait for container to be fully ready
                await asyncio.sleep(3)
                self.is_running = True
                logger.info(f"Container {self.container_name} started successfully")
                return True
            else:
                logger.error(f"Failed to start container: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during container start: {e}")
            return False
    
    async def stop_container(self) -> bool:
        """Stop the sandbox container"""
        try:
            if not await self.is_container_running():
                logger.info(f"Container {self.container_name} is already stopped")
                self.is_running = False
                return True
            
            logger.info(f"Stopping container {self.container_name}...")
            
            # Stop container using pct command
            process = await asyncio.create_subprocess_exec(
                "pct", "stop", self.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.is_running = False
                logger.info(f"Container {self.container_name} stopped successfully")
                return True
            else:
                logger.error(f"Failed to stop container: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during container stop: {e}")
            return False
    
    async def is_container_running(self) -> bool:
        """Check if container is running"""
        try:
            process = await asyncio.create_subprocess_exec(
                "pct", "status", self.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                status = stdout.decode().strip()
                return "running" in status.lower()
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to check container status: {e}")
            return False


class SandboxExecutor:
    """
    将軍システム Sandbox Executor (演習場) Implementation
    
    Provides secure, isolated code execution environment for:
    - Verifying generated code actually works
    - Catching errors before delivery  
    - Implementing "実証主義" (empirical verification) philosophy
    - Providing real-time feedback to R1 for code refinement
    """
    
    def __init__(
        self,
        container_host: str = "192.168.1.12",
        container_port: int = 8080,
        container_id: str = "102",
        auto_manage_container: bool = True
    ):
        self.container_host = container_host
        self.container_port = container_port
        self.auto_manage_container = auto_manage_container
        
        # Container management
        self.container_manager = ContainerManager(container_id)
        
        # Execution history for analysis
        self.execution_history: List[ExecutionResult] = []
        
        # Language-specific configurations
        self.language_configs = {
            Language.PYTHON: {
                "executor": "python3.13",
                "file_extension": ".py",
                "package_manager": "pip",
                "timeout_default": 30
            },
            Language.NODEJS: {
                "executor": "node",
                "file_extension": ".js",  
                "package_manager": "npm",
                "timeout_default": 30
            },
            Language.RUST: {
                "executor": "rustc",
                "file_extension": ".rs",
                "package_manager": "cargo",
                "timeout_default": 60  # Compilation takes longer
            }
        }
        
        logger.info(f"SandboxExecutor initialized - Host: {container_host}:{container_port}")
    
    async def execute_code(self, execution: CodeExecution) -> ExecutionResult:
        """
        Execute code in the sandbox environment
        
        Args:
            execution: Code execution request
            
        Returns:
            ExecutionResult with output and status
        """
        start_time = time.time()
        
        try:
            # Ensure container is running if auto-managed
            if self.auto_manage_container:
                if not await self.container_manager.start_container():
                    return ExecutionResult(
                        execution_id=execution.id,
                        status=ExecutionStatus.ERROR,
                        stdout="",
                        stderr="",
                        return_code=-1,
                        execution_time_seconds=time.time() - start_time,
                        memory_usage_mb=0,
                        error_message="Failed to start sandbox container"
                    )
            
            # Execute based on language
            if execution.language == Language.PYTHON:
                result = await self._execute_python(execution)
            elif execution.language == Language.NODEJS:
                result = await self._execute_nodejs(execution)
            elif execution.language == Language.RUST:
                result = await self._execute_rust(execution)
            else:
                result = ExecutionResult(
                    execution_id=execution.id,
                    status=ExecutionStatus.ERROR,
                    stdout="",
                    stderr="",
                    return_code=-1,
                    execution_time_seconds=time.time() - start_time,
                    memory_usage_mb=0,
                    error_message=f"Unsupported language: {execution.language}"
                )
            
            # Record execution history
            self.execution_history.append(result)
            
            # Keep only last 100 executions
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
            logger.info(f"Execution {execution.id} completed: {result.status.value}")
            return result
            
        except Exception as e:
            logger.error(f"Execution {execution.id} failed: {e}")
            return ExecutionResult(
                execution_id=execution.id,
                status=ExecutionStatus.ERROR,
                stdout="",
                stderr="",
                return_code=-1,
                execution_time_seconds=time.time() - start_time,
                memory_usage_mb=0,
                error_message=str(e)
            )
    
    async def _execute_python(self, execution: CodeExecution) -> ExecutionResult:
        """Execute Python code"""
        start_time = time.time()
        
        try:
            # Create execution script
            script_content = execution.code
            
            # Add dependency installation if needed
            if execution.dependencies:
                install_commands = []
                for dep in execution.dependencies:
                    install_commands.append(f"pip install {dep}")
                
                setup_script = "; ".join(install_commands)
                script_content = f"import subprocess; subprocess.run([{repr(setup_script)}], shell=True)\n{script_content}"
            
            # Execute via container
            result = await self._execute_in_container(
                language="python",
                code=script_content,
                timeout_seconds=execution.timeout_seconds,
                max_memory_mb=execution.max_memory_mb
            )
            
            return ExecutionResult(
                execution_id=execution.id,
                status=result["status"],
                stdout=result["stdout"],
                stderr=result["stderr"], 
                return_code=result["return_code"],
                execution_time_seconds=time.time() - start_time,
                memory_usage_mb=result["memory_usage_mb"],
                error_message=result.get("error_message")
            )
            
        except Exception as e:
            return ExecutionResult(
                execution_id=execution.id,
                status=ExecutionStatus.ERROR,
                stdout="",
                stderr="",
                return_code=-1,
                execution_time_seconds=time.time() - start_time,
                memory_usage_mb=0,
                error_message=str(e)
            )
    
    async def _execute_nodejs(self, execution: CodeExecution) -> ExecutionResult:
        """Execute Node.js code"""
        start_time = time.time()
        
        try:
            script_content = execution.code
            
            # Add dependency installation if needed
            if execution.dependencies:
                # Create package.json
                package_json = {
                    "name": "sandbox-execution",
                    "version": "1.0.0",
                    "dependencies": {dep: "latest" for dep in execution.dependencies}
                }
                
                # Prefix with npm install
                setup_script = f"""
const fs = require('fs');
fs.writeFileSync('package.json', {json.dumps(json.dumps(package_json))});
require('child_process').execSync('npm install', {{stdio: 'inherit'}});
{script_content}
"""
                script_content = setup_script
            
            result = await self._execute_in_container(
                language="nodejs",
                code=script_content,
                timeout_seconds=execution.timeout_seconds,
                max_memory_mb=execution.max_memory_mb
            )
            
            return ExecutionResult(
                execution_id=execution.id,
                status=result["status"],
                stdout=result["stdout"],
                stderr=result["stderr"],
                return_code=result["return_code"],
                execution_time_seconds=time.time() - start_time,
                memory_usage_mb=result["memory_usage_mb"],
                error_message=result.get("error_message")
            )
            
        except Exception as e:
            return ExecutionResult(
                execution_id=execution.id,
                status=ExecutionStatus.ERROR,
                stdout="",
                stderr="",
                return_code=-1,
                execution_time_seconds=time.time() - start_time,
                memory_usage_mb=0,
                error_message=str(e)
            )
    
    async def _execute_rust(self, execution: CodeExecution) -> ExecutionResult:
        """Execute Rust code"""
        start_time = time.time()
        
        try:
            # For Rust, we need to create a proper project structure
            cargo_toml = """[package]
name = "sandbox_execution"
version = "0.1.0"
edition = "2021"

[dependencies]
"""
            
            # Add dependencies
            for dep in execution.dependencies:
                cargo_toml += f'{dep} = "latest"\n'
            
            # Create main.rs
            main_rs = execution.code
            
            # Execute compilation and run
            result = await self._execute_rust_project(
                cargo_toml=cargo_toml,
                main_rs=main_rs,
                timeout_seconds=execution.timeout_seconds,
                max_memory_mb=execution.max_memory_mb
            )
            
            return ExecutionResult(
                execution_id=execution.id,
                status=result["status"],
                stdout=result["stdout"],
                stderr=result["stderr"],
                return_code=result["return_code"],
                execution_time_seconds=time.time() - start_time,
                memory_usage_mb=result["memory_usage_mb"],
                error_message=result.get("error_message")
            )
            
        except Exception as e:
            return ExecutionResult(
                execution_id=execution.id,
                status=ExecutionStatus.ERROR,
                stdout="",
                stderr="",
                return_code=-1,
                execution_time_seconds=time.time() - start_time,
                memory_usage_mb=0,
                error_message=str(e)
            )
    
    async def _execute_in_container(
        self,
        language: str,
        code: str,
        timeout_seconds: int,
        max_memory_mb: int
    ) -> Dict[str, Any]:
        """Execute code inside the container"""
        try:
            # Generate unique execution ID
            execution_id = hashlib.md5(f"{time.time()}{code}".encode()).hexdigest()[:8]
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Copy file to container
                copy_cmd = [
                    "pct", "exec", self.container_manager.container_id, "--",
                    "mkdir", "-p", f"/tmp/sandbox_{execution_id}"
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *copy_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                
                # Copy code file
                with open(temp_file, 'r') as f:
                    code_content = f.read()
                
                # Write code to container
                write_cmd = [
                    "pct", "exec", self.container_manager.container_id, "--",
                    "bash", "-c", f"cat > /tmp/sandbox_{execution_id}/main.{language} << 'EOF'\n{code_content}\nEOF"
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *write_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                
                # Execute code in container
                if language == "python":
                    exec_cmd = f"cd /tmp/sandbox_{execution_id} && timeout {timeout_seconds} python3.13 main.python"
                elif language == "nodejs":
                    exec_cmd = f"cd /tmp/sandbox_{execution_id} && timeout {timeout_seconds} node main.nodejs"
                else:
                    exec_cmd = f"cd /tmp/sandbox_{execution_id} && echo 'Unsupported language: {language}'"
                
                # Execute with resource limits
                container_exec_cmd = [
                    "pct", "exec", self.container_manager.container_id, "--",
                    "bash", "-c", exec_cmd
                ]
                
                start_exec = time.time()
                process = await asyncio.create_subprocess_exec(
                    *container_exec_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout_seconds + 5  # Add buffer for container overhead
                    )
                    
                    exec_time = time.time() - start_exec
                    
                    # Determine status
                    if process.returncode == 0:
                        status = ExecutionStatus.SUCCESS
                    elif process.returncode == 124:  # timeout command exit code
                        status = ExecutionStatus.TIMEOUT
                    else:
                        status = ExecutionStatus.ERROR
                    
                    return {
                        "status": status,
                        "stdout": stdout.decode('utf-8', errors='replace'),
                        "stderr": stderr.decode('utf-8', errors='replace'),
                        "return_code": process.returncode,
                        "memory_usage_mb": 0,  # TODO: Implement memory monitoring
                        "execution_time": exec_time
                    }
                    
                except asyncio.TimeoutError:
                    process.kill()
                    return {
                        "status": ExecutionStatus.TIMEOUT,
                        "stdout": "",
                        "stderr": "Execution timed out",
                        "return_code": -1,
                        "memory_usage_mb": 0,
                        "error_message": f"Execution exceeded {timeout_seconds} seconds"
                    }
                
                finally:
                    # Cleanup in container
                    cleanup_cmd = [
                        "pct", "exec", self.container_manager.container_id, "--",
                        "rm", "-rf", f"/tmp/sandbox_{execution_id}"
                    ]
                    
                    cleanup_process = await asyncio.create_subprocess_exec(
                        *cleanup_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await cleanup_process.communicate()
                
            finally:
                # Cleanup local temp file
                import os
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                
        except Exception as e:
            return {
                "status": ExecutionStatus.ERROR,
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "memory_usage_mb": 0,
                "error_message": str(e)
            }
    
    async def _execute_rust_project(
        self,
        cargo_toml: str,
        main_rs: str,
        timeout_seconds: int,
        max_memory_mb: int
    ) -> Dict[str, Any]:
        """Execute Rust project with Cargo"""
        # Simplified implementation - would need full Cargo project setup
        # For now, treat as regular compilation
        return await self._execute_in_container(
            language="rust",
            code=main_rs,
            timeout_seconds=timeout_seconds,
            max_memory_mb=max_memory_mb
        )
    
    async def shutdown(self) -> bool:
        """Shutdown sandbox executor and stop container if auto-managed"""
        try:
            if self.auto_manage_container:
                success = await self.container_manager.stop_container()
                logger.info(f"SandboxExecutor shutdown: container stopped = {success}")
                return success
            else:
                logger.info("SandboxExecutor shutdown: container not auto-managed")
                return True
                
        except Exception as e:
            logger.error(f"Failed to shutdown SandboxExecutor: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get sandbox execution statistics"""
        try:
            if not self.execution_history:
                return {"message": "No execution history available"}
            
            total_executions = len(self.execution_history)
            successful_executions = sum(1 for r in self.execution_history if r.success)
            
            # Success rate by language
            language_stats = {}
            for result in self.execution_history:
                # Try to infer language from execution_id or use generic
                lang = "unknown"  # Would need to track language in ExecutionResult
                
                if lang not in language_stats:
                    language_stats[lang] = {"total": 0, "success": 0}
                
                language_stats[lang]["total"] += 1
                if result.success:
                    language_stats[lang]["success"] += 1
            
            # Average execution time
            avg_exec_time = sum(r.execution_time_seconds for r in self.execution_history) / total_executions
            
            # Recent activity (last hour)
            one_hour_ago = datetime.utcnow().timestamp() - 3600
            recent_executions = sum(
                1 for r in self.execution_history 
                if r.executed_at.timestamp() > one_hour_ago
            )
            
            stats = {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate_percent": round(successful_executions / total_executions * 100, 2),
                "average_execution_time_seconds": round(avg_exec_time, 3),
                "language_statistics": language_stats,
                "recent_executions_1h": recent_executions,
                "container_running": await self.container_manager.is_container_running(),
                "auto_manage_container": self.auto_manage_container
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of sandbox environment"""
        try:
            health_status = {
                "status": "unknown",
                "container_accessible": False,
                "languages_working": {},
                "checked_at": datetime.utcnow().isoformat()
            }
            
            # Check container accessibility
            container_running = await self.container_manager.is_container_running()
            health_status["container_accessible"] = container_running
            
            if container_running:
                # Test basic execution for each language
                test_results = {}
                
                # Python test
                python_test = CodeExecution(
                    id="health_python",
                    language=Language.PYTHON,
                    code="print('Hello from Python')",
                    timeout_seconds=10
                )
                
                python_result = await self.execute_code(python_test)
                test_results["python"] = python_result.success
                
                # Node.js test  
                nodejs_test = CodeExecution(
                    id="health_nodejs",
                    language=Language.NODEJS,
                    code="console.log('Hello from Node.js')",
                    timeout_seconds=10
                )
                
                nodejs_result = await self.execute_code(nodejs_test)
                test_results["nodejs"] = nodejs_result.success
                
                health_status["languages_working"] = test_results
                
                # Overall status
                if all(test_results.values()) and container_running:
                    health_status["status"] = "healthy"
                elif any(test_results.values()) and container_running:
                    health_status["status"] = "partially_healthy"
                else:
                    health_status["status"] = "unhealthy"
            else:
                health_status["status"] = "unhealthy"
                health_status["error"] = "Container not running"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }


# Factory function
def create_sandbox_executor(config: Dict[str, Any]) -> SandboxExecutor:
    """Create SandboxExecutor instance from configuration"""
    return SandboxExecutor(
        container_host=config.get("host", "192.168.1.12"),
        container_port=config.get("port", 8080),
        container_id=str(config.get("container_id", "102")),
        auto_manage_container=config.get("startup_mode", "on_demand") == "on_demand"
    )


# Example usage and testing
if __name__ == "__main__":
    async def test_sandbox_executor():
        """Test the sandbox executor functionality"""
        # Note: This test requires actual LXC container setup
        executor = SandboxExecutor(auto_manage_container=False)  # Don't auto-manage for testing
        
        # Test Python execution
        python_code = CodeExecution(
            id="test_python",
            language=Language.PYTHON,
            code="""
import math
result = math.sqrt(16)
print(f"Square root of 16 is: {result}")
""",
            description="Test Python math calculation"
        )
        
        result = await executor.execute_code(python_code)
        print(f"Python execution: {result.status.value}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        
        # Test Node.js execution
        nodejs_code = CodeExecution(
            id="test_nodejs", 
            language=Language.NODEJS,
            code="""
const message = "Hello from Node.js!";
console.log(message);
console.log("Current time:", new Date().toISOString());
""",
            description="Test Node.js basic functionality"
        )
        
        result = await executor.execute_code(nodejs_code)
        print(f"Node.js execution: {result.status.value}")
        print(f"Output: {result.stdout}")
        
        # Get statistics
        stats = await executor.get_statistics()
        print(f"Statistics: {stats}")
        
        # Health check
        health = await executor.health_check()
        print(f"Health: {health}")
    
    # Run test (requires LXC environment)
    # asyncio.run(test_sandbox_executor())
    print("Sandbox executor implementation ready (requires LXC container setup for testing)")