"""
Bushidan Multi-Agent System v9.3.2 - Taisho (Implementation Layer)

The Taisho serves as the primary implementation layer with 3-tier fallback chain.
Handles actual code generation, file operations, and heavy computational tasks.

v9.3.2 Enhancements:
- 3-tier fallback chain: Local Qwen3 → Cloud Qwen3 (Kagemusha) → Gemini 3 Flash
- Qwen3-Coder-30B optimization with 4k context (1.5x speed)
- Cloud Qwen3-plus (Kagemusha) for context overflow (32k capacity)
- Gemini 3.0 Flash as final defense line
- Self-healing execution (Layer 2)
- DSPy validation (Layer 3)
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from core.system_orchestrator import SystemOrchestrator


logger = get_logger(__name__)


class ImplementationMode(Enum):
    """Implementation modes for different task types"""
    LIGHTWEIGHT = "lightweight"    # Single file, simple tasks
    STANDARD = "standard"          # Multi-file, standard complexity
    HEAVY = "heavy"               # Complex architecture, multiple components
    PARALLEL = "parallel"         # Multiple parallel implementations


@dataclass
class ImplementationTask:
    """Task representation for implementation"""
    content: str
    mode: ImplementationMode
    context: Optional[Dict[str, Any]] = None
    files_needed: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None


class FallbackStatus(Enum):
    """Status of fallback chain execution"""
    LOCAL_SUCCESS = "local_qwen3_success"
    CLOUD_SUCCESS = "cloud_qwen3_success"
    GEMINI_SUCCESS = "gemini3_success"
    ALL_FAILED = "all_failed"


class Taisho:
    """
    大将 (Taisho) - Implementation Layer v9.3.2

    Primary responsibilities:
    1. Heavy implementation tasks with 3-tier fallback chain
    2. Local Qwen3 (4k context, ¥0) as primary
    3. Cloud Qwen3-plus (Kagemusha, 32k context) for overflow
    4. Gemini 3.0 Flash as final defense
    5. MCP-driven file operations
    6. Self-healing code execution (Layer 2)
    7. DSPy validation (Layer 3)

    v9.3.2 Fallback Chain:
    Primary: Local Qwen3-Coder-30B (4k ctx, ¥0, fast)
    Fallback 1: Cloud Qwen3-plus (32k ctx, ¥3, Kagemusha)
    Fallback 2: Gemini 3.0 Flash (defense, ¥0.04)
    """

    VERSION = "9.3.2"
    LOCAL_CONTEXT_LIMIT = 4096  # Local Qwen3 optimized context

    def __init__(self, orchestrator: SystemOrchestrator):
        self.orchestrator = orchestrator

        # AI clients (initialized from orchestrator)
        self.qwen3_client = None
        self.alibaba_qwen_client = None
        self.gemini3_client = None

        # MCP connections
        self.mcp_manager = None
        self.memory_mcp = None
        self.filesystem_mcp = None
        self.git_mcp = None

        # Error handling layers
        self.self_healing = None
        self.validator = None

        # Statistics
        self.execution_stats = {
            "total_tasks": 0,
            "local_success": 0,
            "cloud_fallback": 0,
            "gemini_fallback": 0,
            "total_failures": 0,
            "total_time_seconds": 0.0,
            "context_overflows": 0
        }

    async def initialize(self) -> None:
        """Initialize Taisho with 3-tier fallback chain"""
        logger.info(f"🏯 Initializing Taisho v{self.VERSION} (Implementation Layer)...")

        # Get AI clients from orchestrator
        self.qwen3_client = self.orchestrator.get_client("qwen3")
        self.alibaba_qwen_client = self.orchestrator.get_client("alibaba_qwen")
        self.gemini3_client = self.orchestrator.get_client("gemini3")

        # Log available clients
        if self.qwen3_client:
            logger.info("✅ Local Qwen3 client available (primary)")
        else:
            logger.warning("⚠️ Local Qwen3 not available")

        if self.alibaba_qwen_client:
            logger.info("✅ Cloud Qwen3-plus (Kagemusha) available")
        else:
            logger.warning("⚠️ Kagemusha not available")

        if self.gemini3_client:
            logger.info("✅ Gemini 3.0 Flash (final defense) available")
        else:
            # Try fallback to standard Gemini
            self.gemini3_client = self.orchestrator.get_client("gemini")
            if self.gemini3_client:
                logger.info("✅ Gemini client (fallback) available")
            else:
                logger.warning("⚠️ No Gemini client available")

        # Get MCP connections
        self.mcp_manager = self.orchestrator.mcp_manager if hasattr(self.orchestrator, 'mcp_manager') else None
        self.memory_mcp = self.orchestrator.get_mcp("memory")
        self.filesystem_mcp = self.orchestrator.get_mcp("filesystem")
        self.git_mcp = self.orchestrator.get_mcp("git")

        # Initialize error handling layers
        await self._initialize_error_handling()

        logger.info(f"✅ Taisho v{self.VERSION} initialization complete")
        self._log_fallback_chain_status()

    async def _initialize_error_handling(self) -> None:
        """Initialize Layer 2 (Self-healing) and Layer 3 (Validation)"""

        # Layer 2: Self-healing executor
        try:
            from utils.self_healing import SelfHealingExecutor

            # Use best available LLM client
            llm_client = self.qwen3_client or self.gemini3_client
            if llm_client:
                self.self_healing = SelfHealingExecutor(
                    llm_client=llm_client,
                    max_correction_attempts=3
                )
                logger.info("✅ Layer 2 (Self-healing) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Self-healing not available: {e}")

        # Layer 3: DSPy validator
        try:
            from utils.dspy_validators import DSPyValidator
            self.validator = DSPyValidator(max_backtracks=2)
            logger.info("✅ Layer 3 (DSPy Validation) initialized")
        except Exception as e:
            logger.warning(f"⚠️ DSPy validator not available: {e}")

    def _log_fallback_chain_status(self) -> None:
        """Log the status of the 3-tier fallback chain"""

        chain = []
        if self.qwen3_client:
            chain.append("Local Qwen3 (4k ctx, ¥0)")
        if self.alibaba_qwen_client:
            chain.append("Cloud Qwen3+ (32k ctx, ¥3)")
        if self.gemini3_client:
            chain.append("Gemini 3 Flash (defense)")

        if chain:
            logger.info(f"🔗 Fallback chain: {' → '.join(chain)}")
        else:
            logger.error("❌ No AI clients available!")

    async def execute_implementation(self, task: ImplementationTask) -> Dict[str, Any]:
        """
        Main implementation execution with 3-tier fallback chain

        Priority:
        1. Local Qwen3 (if context fits in 4k limit)
        2. Cloud Qwen3-plus (Kagemusha) for context overflow
        3. Gemini 3.0 Flash as final defense
        """

        start_time = time.time()
        logger.info(f"🏯 Taisho executing implementation: {task.content[:50]}...")

        try:
            # Gather context
            context = await self._gather_context(task)

            # Estimate context size
            context_size = self._estimate_context_size(task, context)
            logger.info(f"📏 Estimated context size: {context_size} tokens")

            # Plan implementation
            plan = await self._plan_implementation(task, context)

            # Execute with 3-tier fallback chain
            result, fallback_status = await self._execute_with_fallback(
                task, plan, context, context_size
            )

            # Validate results
            validation = await self._validate_implementation(result)

            # Git operations if successful
            if validation.get("valid", False):
                await self._commit_changes(task, result)

            # Update statistics
            elapsed_time = time.time() - start_time
            self._update_stats(fallback_status, elapsed_time)

            logger.info(f"✅ Taisho implementation complete ({fallback_status.value}) in {elapsed_time:.1f}s")

            return {
                "status": "completed",
                "result": result,
                "validation": validation,
                "mode": task.mode.value,
                "handled_by": "taisho",
                "fallback_status": fallback_status.value,
                "execution_time": elapsed_time
            }

        except Exception as e:
            logger.error(f"❌ Taisho implementation failed: {e}")
            self.execution_stats["total_failures"] += 1
            return {"error": str(e), "status": "failed"}

    def _estimate_context_size(self, task: ImplementationTask, context: Dict[str, Any]) -> int:
        """Estimate context size in tokens (rough approximation)"""

        # Rough estimate: 1 token ≈ 4 characters
        total_chars = len(task.content)

        for entry in context.get("memory_entries", []):
            total_chars += len(str(entry))

        for file_info in context.get("existing_files", []):
            if file_info.get("content"):
                total_chars += len(file_info["content"])

        return total_chars // 4

    async def _execute_with_fallback(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any],
        context_size: int
    ) -> tuple[Dict[str, Any], FallbackStatus]:
        """
        Execute with 3-tier fallback chain

        1. Try Local Qwen3 (if context fits)
        2. Fallback to Cloud Qwen3-plus (Kagemusha)
        3. Final fallback to Gemini 3 Flash
        """

        # Tier 1: Local Qwen3 (if context fits)
        if self.qwen3_client and context_size <= self.LOCAL_CONTEXT_LIMIT:
            try:
                result = await self._execute_with_qwen3(task, plan, context)
                if result.get("status") != "failed":
                    return result, FallbackStatus.LOCAL_SUCCESS
                logger.warning("⚠️ Local Qwen3 execution failed, activating Kagemusha")
            except Exception as e:
                logger.warning(f"⚠️ Local Qwen3 error: {e}, activating Kagemusha")

        # Context overflow or local failure
        if context_size > self.LOCAL_CONTEXT_LIMIT:
            self.execution_stats["context_overflows"] += 1
            logger.info(f"🏯 Context overflow ({context_size} > {self.LOCAL_CONTEXT_LIMIT}), activating Kagemusha")

        # Tier 2: Cloud Qwen3-plus (Kagemusha)
        if self.alibaba_qwen_client:
            try:
                result = await self._execute_with_kagemusha(task, plan, context)
                if result.get("status") != "failed":
                    self.execution_stats["cloud_fallback"] += 1
                    return result, FallbackStatus.CLOUD_SUCCESS
                logger.warning("⚠️ Kagemusha failed, activating Gemini final defense")
            except Exception as e:
                logger.warning(f"⚠️ Kagemusha error: {e}, activating Gemini final defense")

        # Tier 3: Gemini 3.0 Flash (Final Defense)
        if self.gemini3_client:
            try:
                result = await self._execute_with_gemini(task, plan, context)
                if result.get("status") != "failed":
                    self.execution_stats["gemini_fallback"] += 1
                    return result, FallbackStatus.GEMINI_SUCCESS
            except Exception as e:
                logger.error(f"❌ Gemini final defense failed: {e}")

        # All tiers failed
        return {"status": "failed", "error": "All fallback tiers exhausted"}, FallbackStatus.ALL_FAILED

    async def _execute_with_qwen3(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute implementation with Local Qwen3"""

        logger.info("🏯 Executing with Local Qwen3 (primary)")

        implementation_prompt = self._create_implementation_prompt(task, plan, context)

        response = await self.qwen3_client.generate(
            messages=[{"role": "user", "content": implementation_prompt}],
            max_tokens=3000  # Leave room for context in 4k limit
        )

        files_created = await self._parse_and_save_files(response)

        return {
            "status": "completed",
            "files_created": files_created,
            "implementation": response,
            "client": "local_qwen3"
        }

    async def _execute_with_kagemusha(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute implementation with Cloud Qwen3-plus (Kagemusha)"""

        logger.info("🏯 Executing with Kagemusha (Cloud Qwen3-plus, 32k context)")

        implementation_prompt = self._create_implementation_prompt(task, plan, context)

        response = await self.alibaba_qwen_client.generate(
            messages=[{"role": "user", "content": implementation_prompt}],
            max_tokens=4096,
            as_kagemusha=True,
            context_overflow=True
        )

        files_created = await self._parse_and_save_files(response)

        return {
            "status": "completed",
            "files_created": files_created,
            "implementation": response,
            "client": "kagemusha"
        }

    async def _execute_with_gemini(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute implementation with Gemini 3.0 Flash (Final Defense)"""

        logger.info("🛡️ Executing with Gemini 3.0 Flash (Final Defense)")

        implementation_prompt = self._create_implementation_prompt(task, plan, context)

        response = await self.gemini3_client.generate(
            prompt=implementation_prompt,
            max_output_tokens=4000,
            as_final_defense=True
        )

        files_created = await self._parse_and_save_files(response)

        return {
            "status": "completed",
            "files_created": files_created,
            "implementation": response,
            "client": "gemini3"
        }

    def _create_implementation_prompt(
        self,
        task: ImplementationTask,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Create implementation prompt for LLM"""

        return f"""
As Taisho (大将) in Bushidan v{self.VERSION}, implement this task following the plan:

Task: {task.content}
Mode: {task.mode.value}
Plan: {plan.get('plan_text', '')}

Generate complete, working code. Include:
- All necessary imports and dependencies
- Proper error handling
- Clean, readable code structure
- Basic documentation

Output each file separately with clear markers:
=== FILENAME: path/to/file.py ===
[file content]
=== END FILE ===
"""

    async def _gather_context(self, task: ImplementationTask) -> Dict[str, Any]:
        """Gather relevant context from Memory MCP and filesystem"""

        context = {
            "memory_entries": [],
            "existing_files": [],
            "project_structure": {}
        }

        try:
            if self.memory_mcp:
                memory_query = f"project context for: {task.content}"
                memory_entries = await self.memory_mcp.search(memory_query)
                context["memory_entries"] = memory_entries[:5] if memory_entries else []

            if self.filesystem_mcp and task.files_needed:
                for file_path in task.files_needed:
                    try:
                        content = await self.filesystem_mcp.read_file(file_path)
                        context["existing_files"].append({
                            "path": file_path,
                            "content": content[:2000]
                        })
                    except Exception:
                        context["existing_files"].append({
                            "path": file_path,
                            "content": None
                        })

            logger.info(f"📋 Context gathered: {len(context['memory_entries'])} memory, {len(context['existing_files'])} files")
            return context

        except Exception as e:
            logger.warning(f"⚠️ Context gathering failed: {e}")
            return context

    async def _plan_implementation(self, task: ImplementationTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan implementation approach"""

        # Use best available client for planning
        client = self.qwen3_client or self.gemini3_client
        if not client:
            return {"plan_text": "Direct implementation without explicit plan", "context": context}

        planning_prompt = f"""
As Taisho (大将), plan the implementation for this task:

Task: {task.content}
Mode: {task.mode.value}

Context:
{self._format_memory_context(context.get('memory_entries', []))}

Create a brief implementation plan with:
1. Files to create/modify
2. Key dependencies
3. Implementation approach
"""

        try:
            if hasattr(client, 'generate'):
                if self.qwen3_client:
                    response = await client.generate(
                        messages=[{"role": "user", "content": planning_prompt}],
                        max_tokens=500
                    )
                else:
                    response = await client.generate(
                        prompt=planning_prompt,
                        max_output_tokens=500
                    )
            else:
                response = "Direct implementation"

            return {"plan_text": response, "context": context}

        except Exception as e:
            logger.warning(f"⚠️ Planning failed: {e}")
            return {"plan_text": "Direct implementation", "context": context}

    async def _parse_and_save_files(self, implementation_text: str) -> List[str]:
        """Parse generated code and save files"""

        files_created = []

        if not self.filesystem_mcp:
            logger.warning("⚠️ Filesystem MCP not available")
            return files_created

        sections = implementation_text.split("=== FILENAME:")

        for section in sections[1:]:
            try:
                lines = section.split("\n")
                filename = lines[0].strip().split("===")[0].strip()

                content_lines = []
                for line in lines[1:]:
                    if "=== END FILE ===" in line:
                        break
                    content_lines.append(line)

                content = "\n".join(content_lines).strip()

                if content:
                    await self.filesystem_mcp.write_file(filename, content)
                    files_created.append(filename)
                    logger.info(f"📄 Created file: {filename}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to parse/save file: {e}")

        return files_created

    async def _validate_implementation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate implementation with Layer 3 DSPy validation"""

        validation = {
            "valid": True,
            "files_count": len(result.get("files_created", [])),
            "issues": [],
            "validation_details": []
        }

        if validation["files_count"] == 0 and result.get("status") != "failed":
            # Check if implementation text contains code
            impl_text = result.get("implementation", "")
            if "def " in impl_text or "class " in impl_text or "import " in impl_text:
                validation["valid"] = True
                validation["issues"].append("Code generated but not saved to files")
            else:
                validation["valid"] = False
                validation["issues"].append("No code generated")
            return validation

        # Layer 3: DSPy validation
        if self.validator:
            try:
                implementation_text = result.get("implementation", "")

                validation_rules = [
                    "code_block_present",
                    "no_japanese_comments",
                    "proper_formatting",
                    "complete_implementation"
                ]

                # Use best available client for validation
                llm_client = self.qwen3_client or self.gemini3_client

                if llm_client:
                    validation_passed, validated_output, attempts = await self.validator.validate_with_retry(
                        llm_client=llm_client,
                        output=implementation_text,
                        validation_rules=validation_rules,
                        original_prompt="Implementation task",
                        task_context={"task_type": "code_implementation"}
                    )

                    validation["valid"] = validation_passed
                    validation["validation_attempts"] = attempts

                    if not validation_passed:
                        validation["issues"].append(f"Validation failed after {attempts} attempts")

            except Exception as e:
                logger.warning(f"⚠️ Validation error: {e}")

        return validation

    async def execute_code_with_healing(
        self,
        code: str,
        task_description: str = "",
        allow_installation: bool = False
    ) -> Dict[str, Any]:
        """Execute code with Layer 2 self-healing"""

        if not self.self_healing:
            return {"status": "error", "error": "Self-healing not available"}

        logger.info("🔧 Executing code with self-healing enabled")

        try:
            execution_result = await self.self_healing.run_and_fix(
                code=code,
                task_description=task_description,
                language="python",
                allow_installation=allow_installation
            )

            if execution_result.success:
                logger.info(f"✅ Code executed (attempt {execution_result.attempt_number})")
                return {
                    "status": "success",
                    "stdout": execution_result.stdout,
                    "execution_time": execution_result.execution_time,
                    "attempts": execution_result.attempt_number
                }
            else:
                return {
                    "status": "failed",
                    "error": execution_result.stderr,
                    "attempts": execution_result.attempt_number
                }

        except Exception as e:
            return {"status": "system_error", "error": str(e)}

    async def _commit_changes(self, task: ImplementationTask, result: Dict[str, Any]) -> None:
        """Commit changes using Git MCP"""

        if not self.git_mcp:
            return

        try:
            files_created = result.get("files_created", [])
            for file_path in files_created:
                await self.git_mcp.add_file(file_path)

            commit_message = f"Implement: {task.content[:50]}\n\nGenerated by Taisho v{self.VERSION}"
            await self.git_mcp.commit(commit_message)

            logger.info(f"✅ Changes committed: {len(files_created)} files")

        except Exception as e:
            logger.warning(f"⚠️ Git commit failed: {e}")

    def _update_stats(self, fallback_status: FallbackStatus, elapsed_time: float) -> None:
        """Update execution statistics"""

        self.execution_stats["total_tasks"] += 1
        self.execution_stats["total_time_seconds"] += elapsed_time

        if fallback_status == FallbackStatus.LOCAL_SUCCESS:
            self.execution_stats["local_success"] += 1

    def _format_memory_context(self, memory_entries: List[Dict]) -> str:
        """Format memory entries for context"""
        if not memory_entries:
            return "No relevant memory entries."

        formatted = []
        for entry in memory_entries[:3]:
            formatted.append(f"- {entry.get('content', str(entry))[:200]}")

        return "\n".join(formatted)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive Taisho statistics"""

        stats = {
            "version": self.VERSION,
            "execution_stats": self.execution_stats,
            "fallback_chain": {
                "local_qwen3": self.qwen3_client is not None,
                "kagemusha": self.alibaba_qwen_client is not None,
                "gemini3": self.gemini3_client is not None
            },
            "error_handling": {
                "self_healing": self.self_healing is not None,
                "validator": self.validator is not None
            }
        }

        # Calculate success rate
        total = self.execution_stats["total_tasks"]
        if total > 0:
            local = self.execution_stats["local_success"]
            cloud = self.execution_stats["cloud_fallback"]
            gemini = self.execution_stats["gemini_fallback"]
            failures = self.execution_stats["total_failures"]

            stats["success_rate"] = round((local + cloud + gemini) / total * 100, 1)
            stats["local_rate"] = round(local / total * 100, 1)
            stats["fallback_rate"] = round((cloud + gemini) / total * 100, 1)

        # Add client statistics
        if self.qwen3_client and hasattr(self.qwen3_client, 'get_statistics'):
            stats["qwen3_statistics"] = self.qwen3_client.get_statistics()

        if self.alibaba_qwen_client and hasattr(self.alibaba_qwen_client, 'get_statistics'):
            stats["kagemusha_statistics"] = self.alibaba_qwen_client.get_statistics()

        return stats
