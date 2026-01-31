"""
Bushidan Multi-Agent System v9.3 - DSPy Quality Validators

Layer 3 Error Handling: Quality-level validation
- Assertion-based quality checks for LLM outputs
- Automatic retry with refined prompts on validation failure
- Rule-based validation (code style, language, completeness)
- Custom validation chains for different task types
- Learning from validation failures
"""

import re
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger


logger = get_logger(__name__)


class ValidationRule(Enum):
    """Predefined validation rules"""
    CODE_BLOCK_PRESENT = "code_block_present"
    NO_JAPANESE_COMMENTS = "no_japanese_comments"
    PROPER_FORMATTING = "proper_formatting"
    COMPLETE_IMPLEMENTATION = "complete_implementation"
    IMPORTS_PRESENT = "imports_present"
    ERROR_HANDLING = "error_handling"
    ENGLISH_ONLY = "english_only"
    JSON_VALID = "json_valid"


@dataclass
class ValidationResult:
    """Result of validation check"""
    passed: bool
    rule_name: str
    error_message: Optional[str] = None
    refinement_hint: Optional[str] = None


class DSPyValidator:
    """
    DSPy-inspired quality validation system
    
    Features:
    1. Rule-based validation with automatic retry
    2. Custom validation chains for different output types
    3. Refinement hints for LLM re-generation
    4. Backtracking support (limited retry attempts)
    5. Validation learning and statistics
    """
    
    def __init__(self, max_backtracks: int = 2):
        """
        Initialize DSPy validator
        
        Args:
            max_backtracks: Maximum number of validation retry attempts
        """
        self.max_backtracks = max_backtracks
        self.validation_history: List[Dict[str, Any]] = []
        self.custom_validators: Dict[str, Callable] = {}
        
        # Register built-in validators
        self._register_builtin_validators()
    
    def _register_builtin_validators(self):
        """Register built-in validation functions"""
        
        self.custom_validators = {
            ValidationRule.CODE_BLOCK_PRESENT.value: self._validate_code_block,
            ValidationRule.NO_JAPANESE_COMMENTS.value: self._validate_no_japanese,
            ValidationRule.PROPER_FORMATTING.value: self._validate_formatting,
            ValidationRule.COMPLETE_IMPLEMENTATION.value: self._validate_completeness,
            ValidationRule.IMPORTS_PRESENT.value: self._validate_imports,
            ValidationRule.ERROR_HANDLING.value: self._validate_error_handling,
            ValidationRule.ENGLISH_ONLY.value: self._validate_english_only,
            ValidationRule.JSON_VALID.value: self._validate_json,
        }
    
    async def validate_with_retry(
        self,
        llm_client,
        output: str,
        validation_rules: List[str],
        original_prompt: str,
        task_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, int]:
        """
        Validate output and retry with refinement if validation fails
        
        Args:
            llm_client: LLM client for regeneration
            output: LLM output to validate
            validation_rules: List of validation rule names to apply
            original_prompt: Original prompt sent to LLM
            task_context: Additional context for validation
        
        Returns:
            (validation_passed, final_output, attempts_used)
        """
        
        current_output = output
        attempts = 0
        
        for attempt in range(1, self.max_backtracks + 2):  # +1 for initial attempt
            attempts = attempt
            
            # Run all validation rules
            validation_results = await self._run_validation_rules(
                current_output,
                validation_rules,
                task_context
            )
            
            # Check if all validations passed
            all_passed = all(result.passed for result in validation_results)
            
            if all_passed:
                logger.info(f"✅ All validations passed (attempt {attempt})")
                self._record_validation_success(validation_rules, attempt)
                return True, current_output, attempts
            
            # If not final attempt, try to refine
            if attempt <= self.max_backtracks:
                logger.warning(f"⚠️ Validation failed on attempt {attempt}, refining...")
                
                # Build refinement prompt
                refinement_prompt = self._build_refinement_prompt(
                    original_prompt,
                    current_output,
                    validation_results
                )
                
                # Request refined output from LLM
                try:
                    refined_response = await llm_client.generate(
                        messages=[{"role": "user", "content": refinement_prompt}],
                        max_tokens=2000,
                        temperature=0.1  # Low temperature for precise corrections
                    )
                    
                    # Extract content if response is dict
                    if isinstance(refined_response, dict):
                        current_output = refined_response.get("content", refined_response)
                    else:
                        current_output = refined_response
                    
                except Exception as e:
                    logger.error(f"❌ Failed to get refined output: {e}")
                    break
            else:
                # Final attempt failed
                logger.error(f"❌ Validation failed after {attempt} attempts")
                self._record_validation_failure(validation_rules, validation_results, attempt)
                return False, current_output, attempts
        
        # Should not reach here
        return False, current_output, attempts
    
    async def _run_validation_rules(
        self,
        output: str,
        rule_names: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Run all specified validation rules"""
        
        results = []
        
        for rule_name in rule_names:
            validator = self.custom_validators.get(rule_name)
            
            if validator:
                try:
                    result = validator(output, context)
                    results.append(result)
                    
                    if not result.passed:
                        logger.warning(f"⚠️ Validation failed: {rule_name} - {result.error_message}")
                    
                except Exception as e:
                    logger.error(f"❌ Validator {rule_name} raised exception: {e}")
                    results.append(ValidationResult(
                        passed=False,
                        rule_name=rule_name,
                        error_message=f"Validator error: {e}",
                        refinement_hint="Please fix the internal validation error"
                    ))
            else:
                logger.warning(f"⚠️ Unknown validation rule: {rule_name}")
        
        return results
    
    def _build_refinement_prompt(
        self,
        original_prompt: str,
        failed_output: str,
        validation_results: List[ValidationResult]
    ) -> str:
        """Build prompt for output refinement based on validation failures"""
        
        failed_rules = [r for r in validation_results if not r.passed]
        
        refinement_prompt = f"""
Your previous output did not meet the required quality standards. Please refine it.

ORIGINAL REQUEST:
{original_prompt}

YOUR PREVIOUS OUTPUT:
{failed_output[:1000]}...  

VALIDATION FAILURES:
"""
        
        for result in failed_rules:
            refinement_prompt += f"""
❌ {result.rule_name}:
   Error: {result.error_message}
   Fix: {result.refinement_hint}
"""
        
        refinement_prompt += """

INSTRUCTIONS FOR REFINEMENT:
1. Address ALL validation failures listed above
2. Maintain the original intent and functionality
3. Output the complete refined version
4. Do not explain the changes, just provide the corrected output

REFINED OUTPUT:
"""
        
        return refinement_prompt
    
    # Built-in Validation Functions
    
    def _validate_code_block(self, output: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate that output contains proper code blocks"""
        
        if "```python" in output or "```" in output:
            return ValidationResult(
                passed=True,
                rule_name="code_block_present"
            )
        
        # Check if output looks like raw code (heuristic)
        code_indicators = ["def ", "class ", "import ", "from ", "    "]  # Indentation
        if any(indicator in output for indicator in code_indicators):
            # Has code but no markdown blocks
            return ValidationResult(
                passed=False,
                rule_name="code_block_present",
                error_message="Code is present but not wrapped in markdown code blocks (```python)",
                refinement_hint="Wrap all code in proper markdown code blocks: ```python\\n[code]\\n```"
            )
        
        return ValidationResult(
            passed=False,
            rule_name="code_block_present",
            error_message="No code block found in output",
            refinement_hint="Please include code wrapped in ```python ... ``` blocks"
        )
    
    def _validate_no_japanese(self, output: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate that output contains no Japanese characters in comments"""
        
        # Check for Japanese character ranges
        japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]'
        
        if re.search(japanese_pattern, output):
            return ValidationResult(
                passed=False,
                rule_name="no_japanese_comments",
                error_message="Japanese characters detected in output (comments must be in English)",
                refinement_hint="Rewrite all comments and strings to use English only. No Japanese characters allowed."
            )
        
        return ValidationResult(
            passed=True,
            rule_name="no_japanese_comments"
        )
    
    def _validate_formatting(self, output: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate proper code formatting"""
        
        # Extract code from markdown blocks
        code = self._extract_code(output)
        
        if not code:
            return ValidationResult(
                passed=False,
                rule_name="proper_formatting",
                error_message="Could not extract code for formatting validation",
                refinement_hint="Ensure code is properly formatted and wrapped in code blocks"
            )
        
        # Basic formatting checks
        issues = []
        
        # Check for reasonable line length (guideline, not strict)
        lines = code.split('\n')
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if len(long_lines) > len(lines) * 0.3:  # More than 30% lines too long
            issues.append("Many lines exceed 120 characters")
        
        # Check for basic structure
        if "def " not in code and "class " not in code and "import " not in code:
            issues.append("Code lacks basic structure (no functions, classes, or imports)")
        
        if issues:
            return ValidationResult(
                passed=False,
                rule_name="proper_formatting",
                error_message="; ".join(issues),
                refinement_hint="Improve code formatting: use proper line lengths, add structure"
            )
        
        return ValidationResult(
            passed=True,
            rule_name="proper_formatting"
        )
    
    def _validate_completeness(self, output: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate implementation completeness"""
        
        code = self._extract_code(output)
        
        if not code:
            return ValidationResult(
                passed=False,
                rule_name="complete_implementation",
                error_message="No code found to validate completeness",
                refinement_hint="Provide complete code implementation"
            )
        
        # Check for incomplete markers
        incomplete_markers = ["...", "TODO", "FIXME", "pass  # Implement", "NotImplementedError"]
        found_incomplete = [marker for marker in incomplete_markers if marker in code]
        
        if found_incomplete:
            return ValidationResult(
                passed=False,
                rule_name="complete_implementation",
                error_message=f"Incomplete implementation detected: {', '.join(found_incomplete)}",
                refinement_hint="Complete all implementation - remove TODOs, placeholders, and implement all functions"
            )
        
        return ValidationResult(
            passed=True,
            rule_name="complete_implementation"
        )
    
    def _validate_imports(self, output: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate that necessary imports are present"""
        
        code = self._extract_code(output)
        
        if not code:
            return ValidationResult(passed=True, rule_name="imports_present")
        
        # Check if code uses libraries but has no imports
        uses_common_libs = any([
            "requests." in code,
            "json." in code,
            "asyncio." in code,
            "Path(" in code,
            "datetime." in code
        ])
        
        has_imports = "import " in code or "from " in code
        
        if uses_common_libs and not has_imports:
            return ValidationResult(
                passed=False,
                rule_name="imports_present",
                error_message="Code appears to use libraries but has no import statements",
                refinement_hint="Add all necessary import statements at the top of the code"
            )
        
        return ValidationResult(
            passed=True,
            rule_name="imports_present"
        )
    
    def _validate_error_handling(self, output: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate presence of error handling"""
        
        code = self._extract_code(output)
        
        if not code:
            return ValidationResult(passed=True, rule_name="error_handling")
        
        # Check if code has functions but no error handling
        has_functions = "def " in code
        has_error_handling = "try:" in code or "except" in code or "raise " in code
        
        # If code is complex (multiple functions) but has no error handling
        function_count = code.count("def ")
        
        if has_functions and function_count >= 2 and not has_error_handling:
            return ValidationResult(
                passed=False,
                rule_name="error_handling",
                error_message="Complex code lacks error handling (no try/except blocks)",
                refinement_hint="Add appropriate try/except blocks for error handling"
            )
        
        return ValidationResult(
            passed=True,
            rule_name="error_handling"
        )
    
    def _validate_english_only(self, output: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate that all text is in English"""
        
        # Similar to no_japanese but more comprehensive
        non_latin_pattern = r'[\u0400-\u04FF\u0500-\u052F\u0370-\u03FF\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]'
        
        if re.search(non_latin_pattern, output):
            return ValidationResult(
                passed=False,
                rule_name="english_only",
                error_message="Non-English characters detected (use English only)",
                refinement_hint="Translate all text to English"
            )
        
        return ValidationResult(
            passed=True,
            rule_name="english_only"
        )
    
    def _validate_json(self, output: str, context: Optional[Dict] = None) -> ValidationResult:
        """Validate that output is valid JSON"""
        
        import json
        
        try:
            # Try to parse as JSON
            json.loads(output)
            return ValidationResult(
                passed=True,
                rule_name="json_valid"
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                passed=False,
                rule_name="json_valid",
                error_message=f"Invalid JSON: {e.msg} at line {e.lineno}",
                refinement_hint="Ensure output is valid JSON format"
            )
    
    def _extract_code(self, output: str) -> str:
        """Extract code from markdown blocks or raw output"""
        
        # Try Python code block
        if "```python" in output:
            start = output.find("```python") + 9
            end = output.find("```", start)
            if end != -1:
                return output[start:end].strip()
        
        # Try generic code block
        if "```" in output:
            start = output.find("```") + 3
            end = output.find("```", start)
            if end != -1:
                return output[start:end].strip()
        
        # Return entire output if no blocks
        return output.strip()
    
    def register_custom_validator(
        self,
        name: str,
        validator_func: Callable[[str, Optional[Dict]], ValidationResult]
    ):
        """Register a custom validation function"""
        
        self.custom_validators[name] = validator_func
        logger.info(f"✅ Registered custom validator: {name}")
    
    def _record_validation_success(self, rules: List[str], attempts: int):
        """Record successful validation"""
        
        self.validation_history.append({
            "rules": rules,
            "attempts": attempts,
            "outcome": "success"
        })
    
    def _record_validation_failure(
        self,
        rules: List[str],
        results: List[ValidationResult],
        attempts: int
    ):
        """Record validation failure"""
        
        failed_rules = [r.rule_name for r in results if not r.passed]
        
        self.validation_history.append({
            "rules": rules,
            "failed_rules": failed_rules,
            "attempts": attempts,
            "outcome": "failure"
        })
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        total = len(self.validation_history)
        if total == 0:
            return {
                "total_validations": 0,
                "success_rate": 0.0,
                "avg_attempts": 0.0
            }
        
        successes = sum(1 for h in self.validation_history if h["outcome"] == "success")
        avg_attempts = sum(h["attempts"] for h in self.validation_history) / total
        
        return {
            "total_validations": total,
            "success_rate": successes / total,
            "avg_attempts": avg_attempts,
            "common_failures": self._get_common_failure_rules()
        }
    
    def _get_common_failure_rules(self) -> List[str]:
        """Get most commonly failed validation rules"""
        
        from collections import Counter
        
        failed_rules = []
        for entry in self.validation_history:
            if entry["outcome"] == "failure":
                failed_rules.extend(entry.get("failed_rules", []))
        
        if not failed_rules:
            return []
        
        # Return top 5 most common
        counter = Counter(failed_rules)
        return [rule for rule, count in counter.most_common(5)]
