# Bushidan v9.3 Error Handling System

## Overview

Bushidan v9.3 implements a comprehensive 3-tier error handling architecture that transforms the system from "作品" (a work of art) to "転んでも自分で立ち上がる道具" (a tool that picks itself up when it falls).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Quality Validation (DSPy Validators)          │
│  ├─ 8 validation rules                                   │
│  ├─ Automatic refinement prompts                         │
│  └─ Max 2 backtracks                                     │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Self-Healing Execution                         │
│  ├─ Syntax validation                                    │
│  ├─ Sandboxed execution                                  │
│  ├─ Error → LLM → Correction loop                        │
│  └─ Max 3 correction attempts                            │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Infrastructure Resilience (LiteLLM Router)     │
│  ├─ Automatic retry (exponential backoff)                │
│  ├─ Fallback chain: Taisho → Groq → Gemini              │
│  ├─ Circuit breaker pattern                              │
│  └─ Max 3 retries per model                              │
└─────────────────────────────────────────────────────────┘
```

## Layer 1: Infrastructure Resilience

**File**: `utils/litellm_router.py`

### Purpose
Handle infrastructure-level failures (network issues, API timeouts, local Ollama crashes) transparently through automatic fallback chains.

### Features

#### 1. Automatic Fallback Chain
```python
Primary: Taisho (local Qwen3-Coder)
  ↓ fails
Fallback 1: Karo-Groq (fast cloud)
  ↓ fails
Fallback 2: Karo-Gemini (quality cloud)
```

#### 2. Exponential Backoff Retry
- Attempt 1: Immediate
- Attempt 2: Wait 1 second
- Attempt 3: Wait 2 seconds
- Attempt 4: Wait 4 seconds

#### 3. Circuit Breaker Pattern
- **Closed** (normal): All requests go through
- **Open** (failing): Block requests for 60 seconds after 5 failures
- **Half-Open** (testing): Allow 1 request to test recovery

### Usage Example

```python
from utils.litellm_router import LiteLLMRouter

router = LiteLLMRouter(config)

# Automatic fallback on failure
result = await router.completion(
    model="taisho-main",
    messages=[{"role": "user", "content": "Generate code"}],
    fallback_chain=["taisho-main", "karo-groq", "karo-gemini"]
)
# Returns: {"content": "...", "model": "karo-groq", "tier": "karo"}
# Automatically used Groq fallback if local Taisho failed

# Get statistics
stats = router.get_usage_stats()
# {"total_calls": 100, "fallbacks_used": 5, "retries_attempted": 12}
```

### Configuration

```python
ModelConfig(
    name="taisho-main",
    endpoint="http://localhost:11434",
    timeout=120,  # Local can be slower
    max_retries=3,
    cost_per_1k_tokens=0.0  # Free
)
```

## Layer 2: Self-Healing Execution

**File**: `utils/self_healing.py`

### Purpose
Automatically detect, analyze, and fix code execution errors through LLM-powered correction loops.

### Features

#### 1. Syntax Validation
Pre-execution syntax checking using Python's `compile()`:
```python
try:
    compile(code, "<string>", "exec")
except SyntaxError as e:
    # Auto-correct before execution
```

#### 2. Sandboxed Execution
Isolated subprocess execution with:
- 30-second timeout
- stdout/stderr capture
- Resource limits (platform-dependent)

#### 3. Error Feedback Loop
```
Execute code
  ↓ Error detected
Capture stderr
  ↓
Send to LLM: "Fix this error: {stderr}"
  ↓
Receive corrected code
  ↓
Execute again (max 3 attempts)
```

#### 4. Dependency Auto-Installation
Detects `ModuleNotFoundError` and attempts:
```bash
pip install [missing_module]
```

### Usage Example

```python
from utils.self_healing import SelfHealingExecutor

executor = SelfHealingExecutor(
    llm_client=qwen_client,
    max_correction_attempts=3
)

# Execute with automatic fixing
result = await executor.run_and_fix(
    code=generated_code,
    task_description="Calculate Fibonacci sequence",
    allow_installation=True  # Auto-install missing modules
)

if result.success:
    print(f"Success after {result.attempt_number} attempts")
    print(f"Output: {result.stdout}")
else:
    print(f"Failed after {result.attempt_number} attempts")
    print(f"Error: {result.stderr}")

# Get statistics
stats = executor.get_correction_stats()
# {"total_corrections": 20, "success_rate": 0.85, "avg_attempts": 1.8}
```

### Correction Prompt Template

```
以下のコードを実行したところエラーが発生しました。
コード:
{code}

エラー内容:
{error_message}

原因を分析し、修正されたコードのみを出力してください。
```

## Layer 3: Quality Validation

**File**: `utils/dspy_validators.py`

### Purpose
Enforce quality standards on LLM outputs through rule-based validation and automatic refinement.

### Built-in Validation Rules

#### 1. `code_block_present`
Ensures code is wrapped in proper markdown blocks:
```python
if "```python" in output or "```" in output:
    pass  # Valid
else:
    fail("Wrap code in ```python ... ``` blocks")
```

#### 2. `no_japanese_comments`
Detects Japanese characters (Hiragana, Katakana, Kanji):
```python
japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]'
if re.search(japanese_pattern, output):
    fail("Comments must be in English only")
```

#### 3. `proper_formatting`
Checks code structure:
- Line length guideline: 120 characters
- Presence of functions/classes/imports

#### 4. `complete_implementation`
Detects incomplete code:
```python
incomplete_markers = ["...", "TODO", "FIXME", "NotImplementedError"]
if any(marker in code for marker in incomplete_markers):
    fail("Complete all implementation - remove TODOs")
```

#### 5. `imports_present`
Validates necessary imports exist:
```python
if "requests." in code and "import requests" not in code:
    fail("Add missing import statements")
```

#### 6. `error_handling`
Checks for try/except in complex code:
```python
if function_count >= 2 and not has_error_handling:
    fail("Add appropriate try/except blocks")
```

#### 7. `english_only`
Comprehensive non-Latin character detection

#### 8. `json_valid`
Validates JSON format using `json.loads()`

### Validation with Automatic Retry

```python
from utils.dspy_validators import DSPyValidator

validator = DSPyValidator(max_backtracks=2)

# Validate with automatic refinement
passed, refined_output, attempts = await validator.validate_with_retry(
    llm_client=qwen_client,
    output=llm_response,
    validation_rules=[
        "code_block_present",
        "no_japanese_comments",
        "complete_implementation"
    ],
    original_prompt=task_prompt
)

if passed:
    print(f"Validation passed after {attempts} attempts")
else:
    print(f"Validation failed after {attempts} attempts")

# Get statistics
stats = validator.get_validation_stats()
# {"total_validations": 50, "success_rate": 0.92, "common_failures": [...]}
```

### Refinement Prompt Template

```
Your previous output did not meet the required quality standards. Please refine it.

ORIGINAL REQUEST:
{original_prompt}

YOUR PREVIOUS OUTPUT:
{failed_output}

VALIDATION FAILURES:
❌ no_japanese_comments:
   Error: Japanese characters detected
   Fix: Rewrite all comments to use English only

INSTRUCTIONS FOR REFINEMENT:
1. Address ALL validation failures listed above
2. Maintain the original intent and functionality
3. Output the complete refined version

REFINED OUTPUT:
```

### Custom Validators

Register domain-specific validation rules:

```python
def validate_api_security(output: str, context: dict) -> ValidationResult:
    """Check for hardcoded API keys"""
    if re.search(r'api_key\s*=\s*["\'][^"\']+["\']', output):
        return ValidationResult(
            passed=False,
            rule_name="api_security",
            error_message="Hardcoded API key detected",
            refinement_hint="Use environment variables for API keys"
        )
    return ValidationResult(passed=True, rule_name="api_security")

# Register custom validator
validator.register_custom_validator("api_security", validate_api_security)

# Use in validation
passed, output, attempts = await validator.validate_with_retry(
    llm_client=client,
    output=code,
    validation_rules=["code_block_present", "api_security"],
    original_prompt=prompt
)
```

## Integration with Core Components

### Enhanced QwenClient

**File**: `utils/qwen_client.py`

```python
class QwenClient:
    def __init__(self, config: Dict[str, Any], ...):
        # Initialize LiteLLM Router for fallback support
        self.router = LiteLLMRouter({
            "qwen_api_base": api_base,
            "gemini_api_key": config.get("gemini_api_key"),
            "groq_api_key": config.get("groq_api_key")
        })
    
    async def generate(self, messages, ..., enable_fallback=True):
        # Automatic fallback on failure
        result = await self.router.completion(
            model="taisho-main",
            fallback_chain=["taisho-main", "karo-groq", "karo-gemini"]
        )
        return result["content"]
```

### Enhanced Taisho

**File**: `core/taisho.py`

```python
class Taisho:
    def __init__(self, orchestrator):
        # Initialize error handling layers
        self.self_healing = SelfHealingExecutor(
            llm_client=self.qwen_client,
            max_correction_attempts=3
        )
        self.validator = DSPyValidator(max_backtracks=2)
    
    async def execute_code_with_healing(self, code, ...):
        """Layer 2: Execute code with self-healing"""
        return await self.self_healing.run_and_fix(code, ...)
    
    async def _validate_implementation(self, result):
        """Layer 3: Validate with DSPy rules"""
        passed, output, attempts = await self.validator.validate_with_retry(
            llm_client=self.qwen_client,
            output=result["implementation"],
            validation_rules=[
                "code_block_present",
                "no_japanese_comments",
                "complete_implementation",
                "imports_present"
            ],
            original_prompt="Implementation task"
        )
        return {"valid": passed, "attempts": attempts}
```

## Benefits

### 1. Reliability (99.9% Uptime Target)
- **Before**: Local Ollama crash = system down
- **After**: Automatic fallback to cloud APIs

### 2. Quality Assurance
- **Before**: Manual review of all LLM outputs
- **After**: Automated validation with 8 rules

### 3. Zero Manual Debugging
- **Before**: "大将の尻拭い" - cleaning up after Taisho
- **After**: Self-healing with automatic correction

### 4. Cost Optimization
- Primary: ¥0 (local Qwen3-Coder)
- Fallback: ¥130/month (Gemini/Groq)
- Only pay when local fails

### 5. 夜間の自動運転 (Overnight Autonomous Operation)
- Circuit breaker prevents hammering failed endpoints
- Automatic recovery when services come back online
- No human intervention required

## Statistics and Monitoring

### Router Statistics
```python
stats = router.get_usage_stats()
# {
#     "total_calls": 150,
#     "fallbacks_used": 5,          # 3.3% fallback rate
#     "retries_attempted": 12,
#     "circuit_breakers": {
#         "taisho-main": {"state": "closed", "failures": 0},
#         "karo-groq": {"state": "closed", "failures": 0}
#     }
# }
```

### Self-Healing Statistics
```python
stats = executor.get_correction_stats()
# {
#     "total_corrections": 20,
#     "success_rate": 0.85,         # 85% auto-fixed
#     "avg_attempts": 1.8,          # Most fixed in 1-2 attempts
#     "recent_failures": [...]
# }
```

### Validation Statistics
```python
stats = validator.get_validation_stats()
# {
#     "total_validations": 50,
#     "success_rate": 0.92,         # 92% pass validation
#     "avg_attempts": 1.3,          # Most pass on first try
#     "common_failures": [
#         "no_japanese_comments",   # Most common issue
#         "complete_implementation"
#     ]
# }
```

## Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key

# Optional
QWEN_API_BASE=http://localhost:11434  # Default
MAX_CORRECTION_ATTEMPTS=3             # Default
MAX_VALIDATION_BACKTRACKS=2           # Default
```

### System Configuration

```yaml
# config/settings.yaml
error_handling:
  layer1:
    circuit_breaker:
      failure_threshold: 5
      timeout_duration: 60
    retry:
      max_retries: 3
      base_delay: 1.0
  
  layer2:
    max_correction_attempts: 3
    execution_timeout: 30
    allow_auto_install: false  # Require approval
  
  layer3:
    max_backtracks: 2
    validation_rules:
      - code_block_present
      - no_japanese_comments
      - complete_implementation
      - imports_present
```

## Testing

### Unit Tests

```python
# Test Layer 1: Router fallback
async def test_router_fallback():
    router = LiteLLMRouter(config)
    # Mock local failure
    with mock_local_failure():
        result = await router.completion(
            model="taisho-main",
            fallback_chain=["taisho-main", "karo-groq"]
        )
        assert result["model"] == "karo-groq"

# Test Layer 2: Self-healing
async def test_self_healing():
    executor = SelfHealingExecutor(llm_client)
    bad_code = "print('hello'  # Missing closing paren"
    result = await executor.run_and_fix(bad_code)
    assert result.success
    assert result.attempt_number > 1

# Test Layer 3: Validation
async def test_validation():
    validator = DSPyValidator()
    bad_output = "print('日本語コメント')  # Japanese"
    passed, _, attempts = await validator.validate_with_retry(
        llm_client=client,
        output=bad_output,
        validation_rules=["no_japanese_comments"],
        original_prompt="Generate code"
    )
    assert passed
    assert attempts > 1
```

## Troubleshooting

### Issue: All models failing
**Symptom**: Exception "All models failed"
**Solution**: 
1. Check Ollama is running: `curl http://localhost:11434/v1/models`
2. Verify API keys: `echo $GEMINI_API_KEY`
3. Check router stats: `router.get_usage_stats()`

### Issue: Self-healing not fixing errors
**Symptom**: Code fails after 3 attempts
**Solution**:
1. Check correction stats: `executor.get_correction_stats()`
2. Increase max_attempts if needed
3. Review error messages in logs

### Issue: Validation always failing
**Symptom**: Validation fails after 2 backtracks
**Solution**:
1. Check validation stats: `validator.get_validation_stats()`
2. Review "common_failures" to identify patterns
3. Consider relaxing strict rules or improving prompts

## Future Enhancements

1. **Layer 1**: Add health checks and predictive circuit breaking
2. **Layer 2**: Support for more languages (JavaScript, Go, etc.)
3. **Layer 3**: Machine learning-based quality scoring
4. **All Layers**: Distributed tracing with OpenTelemetry

## References

- Original Gemini proposal: Issue #5 comment
- LiteLLM documentation: https://docs.litellm.ai/
- DSPy framework: https://github.com/stanfordnlp/dspy
- Circuit breaker pattern: Martin Fowler's "Release It!"
