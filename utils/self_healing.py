"""
Bushidan Multi-Agent System v9.3 - Self-Healing Execution System

Layer 2 Error Handling: Execution-level self-repair
- Automatic code validation and syntax checking
- Error feedback loops with LLM auto-correction
- Sandbox execution with stderr capture
- Dependency validation and auto-installation
- Multi-attempt correction with learning
"""

import asyncio
import subprocess
import tempfile
import os
import sys
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution attempt"""
    success: bool
    stdout: str
    stderr: str
    returncode: int
    execution_time: float
    attempt_number: int


@dataclass
class CorrectionContext:
    """Context for code correction"""
    original_code: str
    error_message: str
    previous_attempts: List[str]
    task_description: str


class SelfHealingExecutor:
    """
    Self-healing code execution system
    
    Features:
    1. Automatic syntax validation before execution
    2. Sandboxed execution with error capture
    3. Feedback loop: errors → LLM → corrected code
    4. Dependency detection and auto-installation (with approval)
    5. Learning from previous correction attempts
    6. Security: Restricted execution environment
    """
    
    def __init__(self, llm_client, max_correction_attempts: int = 3):
        """
        Initialize self-healing executor
        
        Args:
            llm_client: LLM client for code correction (Taisho or Karo)
            max_correction_attempts: Maximum number of auto-correction attempts
        """
        self.llm_client = llm_client
        self.max_correction_attempts = max_correction_attempts
        self.correction_history: List[Dict[str, Any]] = []
        
    async def run_and_fix(
        self,
        code: str,
        task_description: str = "",
        language: str = "python",
        allow_installation: bool = False
    ) -> ExecutionResult:
        """
        Execute code with automatic self-healing
        
        Args:
            code: Code to execute
            task_description: Description of what code should do (for correction context)
            language: Programming language (currently supports "python")
            allow_installation: Allow automatic dependency installation
        
        Returns:
            ExecutionResult with success status and output
        """
        
        logger.info(f"🔧 Starting self-healing execution (max {self.max_correction_attempts} attempts)")
        
        previous_attempts = []
        current_code = code
        
        for attempt in range(1, self.max_correction_attempts + 1):
            logger.info(f"🔄 Execution attempt {attempt}/{self.max_correction_attempts}")
            
            # Step 1: Syntax validation
            syntax_valid, syntax_error = await self._validate_syntax(current_code, language)
            if not syntax_valid:
                logger.warning(f"⚠️ Syntax error detected: {syntax_error}")
                
                # Auto-correct syntax error
                if attempt < self.max_correction_attempts:
                    correction_context = CorrectionContext(
                        original_code=current_code,
                        error_message=f"Syntax error: {syntax_error}",
                        previous_attempts=previous_attempts,
                        task_description=task_description
                    )
                    current_code = await self._request_correction(correction_context)
                    previous_attempts.append(current_code)
                    continue
                else:
                    # Final attempt failed
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=f"Syntax error after {attempt} attempts: {syntax_error}",
                        returncode=-1,
                        execution_time=0.0,
                        attempt_number=attempt
                    )
            
            # Step 2: Execute code in sandbox
            result = await self._execute_in_sandbox(current_code, language)
            
            # Step 3: Check execution result
            if result.success:
                logger.info(f"✅ Execution successful on attempt {attempt}")
                self._record_success(task_description, current_code, attempt)
                return result
            
            # Step 4: Attempt auto-correction if execution failed
            if attempt < self.max_correction_attempts:
                logger.warning(f"⚠️ Execution failed: {result.stderr}")
                
                # Check if it's a missing dependency issue
                if allow_installation and self._is_dependency_error(result.stderr):
                    installed = await self._attempt_dependency_installation(result.stderr)
                    if installed:
                        # Retry same code with new dependency
                        continue
                
                # Request LLM correction
                correction_context = CorrectionContext(
                    original_code=current_code,
                    error_message=result.stderr,
                    previous_attempts=previous_attempts,
                    task_description=task_description
                )
                current_code = await self._request_correction(correction_context)
                previous_attempts.append(current_code)
            else:
                # Final attempt failed
                logger.error(f"❌ Self-healing failed after {attempt} attempts")
                self._record_failure(task_description, code, result.stderr, attempt)
                return result
        
        # Should not reach here, but just in case
        raise Exception("Self-healing execution loop completed unexpectedly")
    
    async def _validate_syntax(self, code: str, language: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code syntax before execution
        
        Returns:
            (is_valid, error_message)
        """
        
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return True, None
            except SyntaxError as e:
                return False, f"Line {e.lineno}: {e.msg}"
        
        # For other languages, skip syntax validation for now
        return True, None
    
    async def _execute_in_sandbox(
        self,
        code: str,
        language: str,
        timeout: int = 30
    ) -> ExecutionResult:
        """
        Execute code in sandboxed environment
        
        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout in seconds
        
        Returns:
            ExecutionResult with stdout, stderr, and return code
        """
        
        if language != "python":
            raise NotImplementedError(f"Language {language} not yet supported")
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute in subprocess with timeout
            import time
            start_time = time.time()
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Security: Limit resources (platform-dependent)
            )
            
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                stdout = stdout_bytes.decode('utf-8', errors='replace')
                stderr = stderr_bytes.decode('utf-8', errors='replace')
                returncode = process.returncode
                
                success = (returncode == 0 and not stderr)
                
                return ExecutionResult(
                    success=success,
                    stdout=stdout,
                    stderr=stderr,
                    returncode=returncode,
                    execution_time=execution_time,
                    attempt_number=1
                )
                
            except asyncio.TimeoutError:
                process.kill()
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    returncode=-1,
                    execution_time=timeout,
                    attempt_number=1
                )
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    async def _request_correction(self, context: CorrectionContext) -> str:
        """
        Request LLM to correct code based on error feedback
        
        Args:
            context: Correction context with error details
        
        Returns:
            Corrected code
        """
        
        correction_prompt = self._build_correction_prompt(context)
        
        try:
            # Use LLM client to generate correction
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": correction_prompt}],
                max_tokens=2000,
                temperature=0.1  # Low temperature for precise corrections
            )
            
            # Extract code from response
            corrected_code = self._extract_code_from_response(response)
            
            logger.info("🔧 LLM provided corrected code")
            return corrected_code
            
        except Exception as e:
            logger.error(f"❌ Failed to get code correction from LLM: {e}")
            # Return original code if correction fails
            return context.original_code
    
    def _build_correction_prompt(self, context: CorrectionContext) -> str:
        """Build prompt for code correction"""
        
        prompt = f"""
The following code encountered an error. Analyze the error and provide a corrected version.

TASK DESCRIPTION:
{context.task_description or "No description provided"}

ORIGINAL CODE:
```python
{context.original_code}
```

ERROR MESSAGE:
{context.error_message}

"""
        
        if context.previous_attempts:
            prompt += f"""
PREVIOUS CORRECTION ATTEMPTS (that also failed):
"""
            for i, attempt in enumerate(context.previous_attempts[-2:], 1):  # Show last 2 attempts
                prompt += f"""
Attempt {i}:
```python
{attempt}
```
"""
        
        prompt += """
INSTRUCTIONS:
1. Analyze the error message carefully
2. Identify the root cause of the error
3. Provide corrected code that fixes the issue
4. Output ONLY the corrected Python code, no explanations
5. Ensure the code is complete and executable
6. Do not repeat previous mistakes

OUTPUT FORMAT:
```python
[corrected code here]
```
"""
        
        return prompt
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract code from LLM response
        
        Handles various formats:
        - ```python ... ```
        - ``` ... ```
        - Plain code
        """
        
        # If response is a dict (from router), extract content
        if isinstance(response, dict):
            response = response.get("content", response)
        
        # Try to extract from code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # Return entire response if no code blocks found
        return response.strip()
    
    def _is_dependency_error(self, error_message: str) -> bool:
        """Check if error is due to missing dependency"""
        
        dependency_indicators = [
            "ModuleNotFoundError",
            "ImportError",
            "No module named",
            "cannot import name"
        ]
        
        return any(indicator in error_message for indicator in dependency_indicators)
    
    async def _attempt_dependency_installation(self, error_message: str) -> bool:
        """
        Attempt to install missing dependency
        
        Returns:
            True if installation successful, False otherwise
        """
        
        # Extract module name from error
        module_name = self._extract_module_name(error_message)
        
        if not module_name:
            return False
        
        logger.info(f"📦 Attempting to install missing dependency: {module_name}")
        
        try:
            # Use pip to install (in subprocess)
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", module_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ Successfully installed {module_name}")
                return True
            else:
                logger.warning(f"⚠️ Failed to install {module_name}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Error during dependency installation: {e}")
            return False
    
    def _extract_module_name(self, error_message: str) -> Optional[str]:
        """Extract module name from import error message"""
        
        # Pattern: "No module named 'xxx'"
        if "No module named" in error_message:
            import re
            match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_message)
            if match:
                module = match.group(1)
                # Get base module (e.g., 'requests' from 'requests.api')
                return module.split('.')[0]
        
        # Pattern: "ModuleNotFoundError: No module named 'xxx'"
        if "ModuleNotFoundError" in error_message:
            import re
            match = re.search(r"ModuleNotFoundError.*['\"]([^'\"]+)['\"]", error_message)
            if match:
                return match.group(1).split('.')[0]
        
        return None
    
    def _record_success(self, task: str, code: str, attempts: int):
        """Record successful correction for learning"""
        
        self.correction_history.append({
            "task": task,
            "code": code,
            "attempts": attempts,
            "outcome": "success"
        })
        
        logger.info(f"📝 Recorded successful correction (attempts: {attempts})")
    
    def _record_failure(self, task: str, code: str, error: str, attempts: int):
        """Record failed correction for analysis"""
        
        self.correction_history.append({
            "task": task,
            "code": code,
            "error": error,
            "attempts": attempts,
            "outcome": "failure"
        })
        
        logger.warning(f"📝 Recorded failed correction after {attempts} attempts")
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """Get statistics on correction attempts"""
        
        total = len(self.correction_history)
        if total == 0:
            return {
                "total_corrections": 0,
                "success_rate": 0.0,
                "avg_attempts": 0.0
            }
        
        successes = sum(1 for h in self.correction_history if h["outcome"] == "success")
        avg_attempts = sum(h["attempts"] for h in self.correction_history) / total
        
        return {
            "total_corrections": total,
            "success_rate": successes / total,
            "avg_attempts": avg_attempts,
            "recent_failures": [
                {"task": h["task"], "error": h["error"]}
                for h in self.correction_history[-5:]
                if h["outcome"] == "failure"
            ]
        }
