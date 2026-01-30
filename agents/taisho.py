"""
将軍システム v8.0 - Enhanced Taisho (侍大将) Agent
現場指揮官: DeepSeek R1 Japanese + 全v8.0機能統合

Enhanced Features:
- Knowledge Base (RAG) integration
- Activity Memory (陣中日記) for decision consistency
- Sandbox execution validation (実証主義)
- Dynamic parameter optimization
- Thought process monitoring
- Ashigaru (足軽) coordination
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import httpx

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.knowledge_base import KnowledgeBase, KnowledgeEntry, DataSource
from core.activity_memory import ActivityMemory, TaskRecord, TaskComplexity, DecisionConfidence, SimilarTaskResult
from core.sandbox_executor import SandboxExecutor, CodeExecution, Language as SandboxLanguage
from ashigaru.ollama_web_search import OllamaWebSearch, SearchQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaishoTask:
    """Task structure for Taisho processing"""
    id: str
    description: str
    complexity: TaskComplexity
    priority: TaskPriority = TaskPriority.MEDIUM
    context: Optional[str] = None
    requires_execution: bool = False  # Whether code execution is needed
    requires_latest_info: bool = False  # Whether web search is needed
    max_processing_time: int = 300  # Max time in seconds
    requested_by: str = "unknown"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if not self.id:
            task_hash = hashlib.md5(f"{self.description}{self.created_at}".encode()).hexdigest()[:8]
            self.id = f"task_{task_hash}"


@dataclass
class TaishoResponse:
    """Response from Taisho processing"""
    task_id: str
    success: bool
    response: str
    reasoning: str
    confidence: DecisionConfidence
    processing_time_seconds: float
    tools_used: List[str]
    similar_tasks_found: int = 0
    knowledge_retrieved: int = 0
    code_executed: bool = False
    execution_successful: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class TaishoAgent:
    """
    将軍システム v8.0 Enhanced Taisho Agent
    
    DeepSeek R1 Japanese-based field commander with full v8.0 integration:
    - Knowledge Base consultation for latest information
    - Activity Memory for consistent decision-making
    - Sandbox execution for code validation
    - Dynamic parameter adjustment based on task complexity
    - Intelligent tool selection and coordination
    """
    
    def __init__(
        self,
        r1_endpoint: str = "http://192.168.1.11:11434",
        model_name: str = "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese",
        knowledge_base: Optional[KnowledgeBase] = None,
        activity_memory: Optional[ActivityMemory] = None,
        sandbox_executor: Optional[SandboxExecutor] = None,
        web_search: Optional[OllamaWebSearch] = None
    ):
        self.r1_endpoint = r1_endpoint.rstrip('/')
        self.model_name = model_name
        
        # v8.0 Core systems
        self.knowledge_base = knowledge_base
        self.activity_memory = activity_memory
        self.sandbox_executor = sandbox_executor
        self.web_search = web_search
        
        # Dynamic parameters for different task types
        self.parameter_configs = {
            TaskComplexity.SIMPLE: {
                "temperature": 0.3,
                "max_tokens": 1000,
                "timeout_seconds": 60
            },
            TaskComplexity.MEDIUM: {
                "temperature": 0.7,
                "max_tokens": 2000,
                "timeout_seconds": 120
            },
            TaskComplexity.COMPLEX: {
                "temperature": 0.9,
                "max_tokens": 4000,
                "timeout_seconds": 300
            },
            TaskComplexity.STRATEGIC: {
                "temperature": 0.5,  # More focused for strategic thinking
                "max_tokens": 3000,
                "timeout_seconds": 240
            }
        }
        
        # Audit mode parameters (for quality monitoring)
        self.audit_parameters = {
            "temperature": 0.1,  # Very focused
            "max_tokens": 1500,
            "timeout_seconds": 90
        }
        
        logger.info(f"TaishoAgent v8.0 initialized - Endpoint: {r1_endpoint}")
    
    async def process_task(self, task: TaishoTask, audit_mode: bool = False) -> TaishoResponse:
        """
        Process a task using the full v8.0 enhancement stack
        
        Args:
            task: Task to process
            audit_mode: Use strict parameters for audit/review tasks
            
        Returns:
            TaishoResponse with results and metadata
        """
        start_time = time.time()
        tools_used = []
        
        try:
            logger.info(f"Processing task {task.id}: {task.description[:50]}...")
            
            # Step 1: Check Activity Memory for similar tasks
            similar_tasks = []
            if self.activity_memory:
                logger.info("Checking activity memory for similar tasks...")
                similar_tasks = await self.activity_memory.find_similar_tasks(
                    task_summary=task.description,
                    task_type=task.complexity,
                    limit=3,
                    min_confidence=DecisionConfidence.MEDIUM
                )
                tools_used.append("activity_memory")
                
                if similar_tasks:
                    logger.info(f"Found {len(similar_tasks)} similar tasks")
            
            # Step 2: Retrieve relevant knowledge if needed
            knowledge_context = ""
            knowledge_retrieved = 0
            if self.knowledge_base:
                logger.info("Searching knowledge base...")
                knowledge_entries = await self.knowledge_base.search(
                    query=task.description,
                    max_results=3,
                    score_threshold=0.6
                )
                
                if knowledge_entries:
                    knowledge_retrieved = len(knowledge_entries)
                    knowledge_snippets = []
                    for entry in knowledge_entries:
                        snippet = f"【{entry.title}】: {entry.content[:200]}..."
                        knowledge_snippets.append(snippet)
                    knowledge_context = "\n".join(knowledge_snippets)
                    tools_used.append("knowledge_base")
                    logger.info(f"Retrieved {knowledge_retrieved} knowledge entries")
            
            # Step 3: Get latest information if required
            latest_info = ""
            if task.requires_latest_info and self.web_search:
                logger.info("Searching for latest information...")
                search_query = SearchQuery(
                    id=f"search_{task.id}",
                    query=task.description,
                    language="ja",
                    max_results=3,
                    requested_by="taisho",
                    context=task.context
                )
                
                search_response = await self.web_search.search(search_query)
                if search_response.success:
                    search_snippets = [result.snippet for result in search_response.results]
                    latest_info = "\n".join(search_snippets[:2])  # Top 2 results
                    tools_used.append("web_search")
                    logger.info("Latest information retrieved")
            
            # Step 4: Prepare enhanced context for R1
            enhanced_context = self._build_enhanced_context(
                task=task,
                similar_tasks=similar_tasks,
                knowledge_context=knowledge_context,
                latest_info=latest_info
            )
            
            # Step 5: Get R1 response with dynamic parameters
            parameters = self.audit_parameters if audit_mode else self.parameter_configs[task.complexity]
            r1_response = await self._query_r1(
                prompt=enhanced_context,
                parameters=parameters
            )
            
            # Step 6: Execute code validation if needed
            code_executed = False
            execution_successful = False
            if task.requires_execution and self.sandbox_executor:
                logger.info("Executing code validation...")
                code_executed, execution_successful = await self._validate_code_execution(task, r1_response)
                tools_used.append("sandbox_executor")
            
            # Step 7: Determine confidence and final response
            confidence = self._determine_confidence(
                similar_tasks_found=len(similar_tasks),
                knowledge_retrieved=knowledge_retrieved,
                execution_successful=execution_successful if code_executed else True,
                task_complexity=task.complexity
            )
            
            # Extract reasoning and response from R1 output
            reasoning, final_response = self._parse_r1_response(r1_response)
            
            processing_time = time.time() - start_time
            
            # Step 8: Record to Activity Memory
            if self.activity_memory:
                await self._record_activity(
                    task=task,
                    response=final_response,
                    reasoning=reasoning,
                    confidence=confidence,
                    processing_time=processing_time,
                    tools_used=tools_used,
                    similar_tasks=[t.task_id for t in similar_tasks],
                    success=True
                )
            
            # Create response object
            response = TaishoResponse(
                task_id=task.id,
                success=True,
                response=final_response,
                reasoning=reasoning,
                confidence=confidence,
                processing_time_seconds=processing_time,
                tools_used=tools_used,
                similar_tasks_found=len(similar_tasks),
                knowledge_retrieved=knowledge_retrieved,
                code_executed=code_executed,
                execution_successful=execution_successful
            )
            
            logger.info(f"Task {task.id} completed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Task {task.id} processing failed: {e}")
            
            processing_time = time.time() - start_time
            
            # Record failure to Activity Memory
            if self.activity_memory:
                await self._record_activity(
                    task=task,
                    response=f"処理エラー: {str(e)}",
                    reasoning="システムエラーにより処理が中断されました",
                    confidence=DecisionConfidence.LOW,
                    processing_time=processing_time,
                    tools_used=tools_used,
                    similar_tasks=[],
                    success=False,
                    error_message=str(e)
                )
            
            return TaishoResponse(
                task_id=task.id,
                success=False,
                response=f"申し訳ございませんが、処理中にエラーが発生しました: {str(e)}",
                reasoning="システムエラー",
                confidence=DecisionConfidence.LOW,
                processing_time_seconds=processing_time,
                tools_used=tools_used
            )
    
    def _build_enhanced_context(
        self,
        task: TaishoTask,
        similar_tasks: List[SimilarTaskResult],
        knowledge_context: str,
        latest_info: str
    ) -> str:
        """Build enhanced context prompt for R1"""
        
        context_parts = []
        
        # Base instruction
        context_parts.append("""あなたは将軍システムの侍大将（現場指揮官）です。DeepSeek R1 Japaneseとして、深い<think>タグでの思考プロセスを活用し、高品質な分析と判断を行ってください。

以下のタスクを処理してください：""")
        
        # Task description
        context_parts.append(f"""
【タスク】
{task.description}

複雑度: {task.complexity.value}
優先度: {task.priority.value}
""")
        
        # Context if provided
        if task.context:
            context_parts.append(f"""
【追加コンテキスト】
{task.context}
""")
        
        # Similar tasks from memory
        if similar_tasks:
            context_parts.append("\n【過去の類似タスク】")
            for i, similar in enumerate(similar_tasks[:2], 1):  # Top 2
                context_parts.append(f"""
{i}. {similar.task_summary}
   決定: {similar.final_decision}
   理由: {similar.decision_reasoning}
   信頼度: {similar.confidence_level.value}
   処理時間: {similar.processing_time_seconds:.1f}秒
   類似度: {similar.similarity_score:.2f}
""")
        
        # Knowledge base context
        if knowledge_context:
            context_parts.append(f"""
【関連知識】
{knowledge_context}
""")
        
        # Latest information
        if latest_info:
            context_parts.append(f"""
【最新情報】
{latest_info}
""")
        
        # Instructions
        context_parts.append("""
【回答形式】
<think>
ここで深く分析・検討してください：
1. タスクの本質的な要求を理解
2. 過去の類似タスクとの比較検討
3. 関連知識・最新情報の活用方法
4. 最適なアプローチの選択
5. 期待される結果とリスク評価
</think>

【判断】
明確で実用的な判断・推奨事項

【根拠】
判断に至った理由と考慮した要素
""")
        
        return "\n".join(context_parts)
    
    async def _query_r1(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Query DeepSeek R1 Japanese model"""
        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": parameters["temperature"],
                "max_tokens": parameters["max_tokens"],
                "stream": False
            }
            
            async with httpx.AsyncClient(timeout=parameters["timeout_seconds"]) as client:
                response = await client.post(
                    f"{self.r1_endpoint}/api/generate",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "応答を取得できませんでした。")
                else:
                    logger.error(f"R1 API error: {response.status_code} - {response.text}")
                    return "R1 APIエラーが発生しました。"
                    
        except httpx.TimeoutException:
            logger.error("R1 query timed out")
            return "処理がタイムアウトしました。"
        except Exception as e:
            logger.error(f"R1 query failed: {e}")
            return f"R1クエリエラー: {str(e)}"
    
    def _parse_r1_response(self, r1_output: str) -> Tuple[str, str]:
        """Parse R1 response to extract reasoning and final answer"""
        try:
            # Extract <think> content
            think_start = r1_output.find("<think>")
            think_end = r1_output.find("</think>")
            
            if think_start != -1 and think_end != -1:
                thinking_process = r1_output[think_start + 7:think_end].strip()
                remaining_text = r1_output[think_end + 8:].strip()
            else:
                thinking_process = "思考プロセス不明"
                remaining_text = r1_output
            
            # Extract judgment and reasoning
            judgment_start = remaining_text.find("【判断】")
            reasoning_start = remaining_text.find("【根拠】")
            
            if judgment_start != -1:
                if reasoning_start != -1:
                    judgment = remaining_text[judgment_start + 4:reasoning_start].strip()
                    reasoning = remaining_text[reasoning_start + 4:].strip()
                else:
                    judgment = remaining_text[judgment_start + 4:].strip()
                    reasoning = thinking_process
            else:
                # Fallback parsing
                lines = remaining_text.split('\n')
                judgment = remaining_text[:200] + "..." if len(remaining_text) > 200 else remaining_text
                reasoning = thinking_process
            
            return reasoning, judgment
            
        except Exception as e:
            logger.warning(f"Failed to parse R1 response: {e}")
            return "解析エラー", r1_output[:500] + "..." if len(r1_output) > 500 else r1_output
    
    async def _validate_code_execution(self, task: TaishoTask, r1_response: str) -> Tuple[bool, bool]:
        """Validate code execution in sandbox"""
        try:
            # Extract code from R1 response (simple heuristic)
            code_blocks = []
            lines = r1_response.split('\n')
            in_code_block = False
            current_code = []
            current_language = None
            
            for line in lines:
                if line.strip().startswith('```'):
                    if not in_code_block:
                        # Starting code block
                        in_code_block = True
                        language_hint = line.strip()[3:].lower()
                        if 'python' in language_hint:
                            current_language = SandboxLanguage.PYTHON
                        elif 'javascript' in language_hint or 'js' in language_hint:
                            current_language = SandboxLanguage.NODEJS
                        elif 'rust' in language_hint:
                            current_language = SandboxLanguage.RUST
                        else:
                            current_language = SandboxLanguage.PYTHON  # Default
                    else:
                        # Ending code block
                        in_code_block = False
                        if current_code and current_language:
                            code_blocks.append((current_language, '\n'.join(current_code)))
                        current_code = []
                        current_language = None
                elif in_code_block:
                    current_code.append(line)
            
            if not code_blocks:
                logger.info("No code blocks found in R1 response")
                return False, False
            
            # Execute first code block
            language, code = code_blocks[0]
            execution = CodeExecution(
                id=f"validation_{task.id}",
                language=language,
                code=code,
                description=f"Validation for task: {task.description}",
                timeout_seconds=30
            )
            
            result = await self.sandbox_executor.execute_code(execution)
            return True, result.success
            
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return False, False
    
    def _determine_confidence(
        self,
        similar_tasks_found: int,
        knowledge_retrieved: int,
        execution_successful: bool,
        task_complexity: TaskComplexity
    ) -> DecisionConfidence:
        """Determine confidence level based on available information"""
        
        confidence_score = 0
        
        # Base confidence by complexity
        if task_complexity == TaskComplexity.SIMPLE:
            confidence_score += 2
        elif task_complexity == TaskComplexity.MEDIUM:
            confidence_score += 1
        # Complex and Strategic start at 0
        
        # Boost from similar tasks
        if similar_tasks_found >= 2:
            confidence_score += 2
        elif similar_tasks_found == 1:
            confidence_score += 1
        
        # Boost from knowledge
        if knowledge_retrieved >= 2:
            confidence_score += 1
        
        # Boost from successful execution
        if execution_successful:
            confidence_score += 2
        
        # Map to confidence levels
        if confidence_score >= 5:
            return DecisionConfidence.VERY_HIGH
        elif confidence_score >= 3:
            return DecisionConfidence.HIGH
        elif confidence_score >= 1:
            return DecisionConfidence.MEDIUM
        else:
            return DecisionConfidence.LOW
    
    async def _record_activity(
        self,
        task: TaishoTask,
        response: str,
        reasoning: str,
        confidence: DecisionConfidence,
        processing_time: float,
        tools_used: List[str],
        similar_tasks: List[str],
        success: bool,
        error_message: Optional[str] = None
    ) -> bool:
        """Record activity to memory"""
        try:
            record = TaskRecord(
                id=f"activity_{task.id}",
                task_hash="",  # Will be auto-generated
                task_summary=task.description,
                task_type=task.complexity,
                think_summary=reasoning[:500],  # Condensed reasoning
                final_decision=response[:500],   # Condensed response
                decision_reasoning=reasoning,
                confidence_level=confidence,
                processing_time_seconds=processing_time,
                tools_used=tools_used,
                similar_tasks=similar_tasks,
                success=success,
                error_message=error_message,
                metadata={
                    "priority": task.priority.value,
                    "context": task.context,
                    "requested_by": task.requested_by
                }
            )
            
            return await self.activity_memory.record_activity(record)
            
        except Exception as e:
            logger.error(f"Failed to record activity: {e}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from activity memory"""
        try:
            if not self.activity_memory:
                return {"error": "Activity memory not available"}
            
            stats = await self.activity_memory.get_statistics()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            health_status = {
                "status": "unknown",
                "r1_accessible": False,
                "knowledge_base_status": "unknown",
                "activity_memory_status": "unknown",
                "sandbox_status": "unknown",
                "web_search_status": "unknown",
                "checked_at": datetime.utcnow().isoformat()
            }
            
            # Check R1 endpoint
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{self.r1_endpoint}/api/version")
                    health_status["r1_accessible"] = response.status_code == 200
            except:
                health_status["r1_accessible"] = False
            
            # Check subsystems
            if self.knowledge_base:
                kb_health = await self.knowledge_base.health_check()
                health_status["knowledge_base_status"] = kb_health.get("status", "unknown")
            
            if self.activity_memory:
                stats = await self.activity_memory.get_statistics()
                health_status["activity_memory_status"] = "healthy" if stats else "unhealthy"
            
            if self.sandbox_executor:
                sandbox_health = await self.sandbox_executor.health_check()
                health_status["sandbox_status"] = sandbox_health.get("status", "unknown")
            
            if self.web_search:
                search_health = await self.web_search.health_check()
                health_status["web_search_status"] = search_health.get("status", "unknown")
            
            # Overall status
            critical_systems = [health_status["r1_accessible"]]
            if all(critical_systems):
                health_status["status"] = "healthy"
            else:
                health_status["status"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }


# Factory function
def create_taisho_agent(
    config: Dict[str, Any],
    knowledge_base: Optional[KnowledgeBase] = None,
    activity_memory: Optional[ActivityMemory] = None,
    sandbox_executor: Optional[SandboxExecutor] = None,
    web_search: Optional[OllamaWebSearch] = None
) -> TaishoAgent:
    """Create TaishoAgent instance from configuration"""
    return TaishoAgent(
        r1_endpoint=config.get("url", "http://192.168.1.11:11434"),
        model_name=config.get("model", "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"),
        knowledge_base=knowledge_base,
        activity_memory=activity_memory,
        sandbox_executor=sandbox_executor,
        web_search=web_search
    )


# Example usage and testing
if __name__ == "__main__":
    async def test_taisho_agent():
        """Test the enhanced Taisho agent"""
        # Create agent (without subsystems for testing)
        agent = TaishoAgent()
        
        # Test task
        task = TaishoTask(
            id="test_task",
            description="I2S設定の最適化方法を教えてください",
            complexity=TaskComplexity.MEDIUM,
            priority=TaskPriority.HIGH,
            context="ESP32でオーディオ処理を行う際の設定について",
            requires_execution=False,
            requires_latest_info=True,
            requested_by="test_user"
        )
        
        # Process task
        response = await agent.process_task(task)
        
        print(f"Task processed: {response.success}")
        print(f"Response: {response.response}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Confidence: {response.confidence.value}")
        print(f"Processing time: {response.processing_time_seconds:.2f}s")
        print(f"Tools used: {response.tools_used}")
        
        # Health check
        health = await agent.health_check()
        print(f"Health: {health}")
    
    # Run test
    asyncio.run(test_taisho_agent())