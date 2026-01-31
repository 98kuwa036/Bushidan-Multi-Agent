"""
Bushidan Multi-Agent System v9.1 - DSPy Integration

DSPy integration for prompt optimization and translation layer.
Converts Japanese intentions into structured instructions for optimal LLM performance.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """Template for optimized prompts"""
    name: str
    template: str
    optimized_for: str  # "qwen3-coder", "gemini", "claude"
    complexity_level: str  # "simple", "medium", "complex", "strategic"


class DSPyClient:
    """
    DSPy Integration Client
    
    Features:
    - Automatic prompt optimization for different LLMs
    - Japanese -> Structured instruction translation
    - Context-aware prompt compilation
    - Model-specific optimization
    """
    
    def __init__(self):
        self.initialized = False
        self.prompt_templates = {}
        self.optimization_cache = {}
        
    async def initialize(self) -> None:
        """Initialize DSPy client"""
        logger.info("🔧 Initializing DSPy Client...")
        
        try:
            # Try to import DSPy (optional dependency)
            global dspy
            import dspy
            self.dspy_available = True
            logger.info("✅ DSPy library available")
        except ImportError:
            self.dspy_available = False
            logger.warning("⚠️ DSPy not available - using fallback optimization")
        
        # Load default prompt templates
        self._load_default_templates()
        
        self.initialized = True
        logger.info("✅ DSPy Client initialized")
    
    def _load_default_templates(self) -> None:
        """Load default optimized prompt templates"""
        
        self.prompt_templates = {
            # Templates optimized for Qwen3-Coder (大将)
            "qwen3_implementation": PromptTemplate(
                name="qwen3_implementation",
                template="""You are a skilled software engineer implementing: {task}

Requirements:
- Write complete, working code
- Include all necessary imports
- Add error handling where appropriate
- Follow best practices for {language}
- Generate clean, readable code

Context: {context}

Implementation:""",
                optimized_for="qwen3-coder",
                complexity_level="medium"
            ),
            
            "qwen3_complex": PromptTemplate(
                name="qwen3_complex",
                template="""You are architecting a complex system: {task}

System Requirements:
1. Analyze the requirements thoroughly
2. Design modular architecture
3. Implement core components
4. Add comprehensive error handling
5. Include testing approach

Context and constraints: {context}

Architecture and Implementation:""",
                optimized_for="qwen3-coder",
                complexity_level="complex"
            ),
            
            # Templates optimized for Gemini (家老)
            "gemini_coordination": PromptTemplate(
                name="gemini_coordination",
                template="""As Karo (family elder), coordinate this task: {task}

Your responsibilities:
- Break down the task into manageable subtasks
- Assign appropriate resources
- Ensure quality standards
- Coordinate between team members

Context: {context}

Coordination plan:""",
                optimized_for="gemini",
                complexity_level="medium"
            ),
            
            # Templates optimized for Claude (将軍)
            "claude_strategic": PromptTemplate(
                name="claude_strategic",
                template="""As Shogun (supreme commander), make strategic decisions for: {task}

Consider:
- Long-term implications
- Resource allocation
- Technical feasibility  
- Risk assessment
- Alignment with organizational goals

Context: {context}

Strategic decision:""",
                optimized_for="claude",
                complexity_level="strategic"
            )
        }
        
        logger.info(f"📚 Loaded {len(self.prompt_templates)} default templates")
    
    async def optimize_prompt(
        self, 
        task: str, 
        target_model: str = "qwen3-coder",
        complexity: str = "medium",
        context: Dict[str, Any] = None,
        language: str = "auto"
    ) -> str:
        """
        Optimize prompt for target model and complexity
        
        This is the core DSPy functionality - converting intentions
        into optimized, structured instructions.
        """
        
        if not self.initialized:
            await self.initialize()
        
        # Detect language if auto
        if language == "auto":
            language = self._detect_language(task)
        
        # Apply Japanese translation layer if needed
        if language == "japanese":
            task = await self._translate_japanese_intention(task)
        
        # Find appropriate template
        template_key = f"{target_model}_{complexity}"
        template = self._find_best_template(target_model, complexity)
        
        if not template:
            logger.warning(f"⚠️ No template found for {target_model}_{complexity}, using fallback")
            return await self._fallback_optimization(task, target_model, complexity, context)
        
        # Apply template with context
        optimized_prompt = template.template.format(
            task=task,
            context=self._format_context(context or {}),
            language=self._infer_programming_language(task, context)
        )
        
        # Apply model-specific optimizations
        optimized_prompt = self._apply_model_optimizations(optimized_prompt, target_model)
        
        logger.info(f"🔧 Optimized prompt for {target_model} ({complexity})")
        return optimized_prompt
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Japanese"""
        
        # Simple heuristic - check for Japanese characters
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309f' or  # Hiragana
                                                   '\u30a0' <= char <= '\u30ff' or  # Katakana
                                                   '\u4e00' <= char <= '\u9faf')     # Kanji
        
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars > 0 and japanese_chars / total_chars > 0.3:
            return "japanese"
        return "english"
    
    async def _translate_japanese_intention(self, japanese_task: str) -> str:
        """
        Translate Japanese intention into structured English instruction
        
        This is a key feature - bridging the language gap for local LLMs
        """
        
        # Enhanced translation patterns for common development tasks
        translation_patterns = {
            # Implementation patterns
            "実装": "implement",
            "作成": "create",
            "生成": "generate",
            "構築": "build",
            "開発": "develop",
            
            # Technical terms
            "システム": "system",
            "アプリケーション": "application", 
            "アプリ": "app",
            "データベース": "database",
            "API": "API",
            "ウェブ": "web",
            "サーバー": "server",
            "クライアント": "client",
            
            # Programming concepts
            "クラス": "class",
            "関数": "function",
            "メソッド": "method",
            "変数": "variable",
            "配列": "array",
            "辞書": "dictionary",
            "リスト": "list",
            
            # Actions
            "テスト": "test",
            "デバッグ": "debug", 
            "最適化": "optimize",
            "リファクタリング": "refactor",
            "修正": "fix",
            "改善": "improve"
        }
        
        english_task = japanese_task
        for japanese, english in translation_patterns.items():
            english_task = english_task.replace(japanese, english)
        
        # Add structured context for better understanding
        structured_task = f"Please {english_task.lower().strip()}"
        
        logger.info("🌐 Applied Japanese translation layer")
        return structured_task
    
    def _find_best_template(self, target_model: str, complexity: str) -> Optional[PromptTemplate]:
        """Find the best matching template"""
        
        # Try exact match first
        exact_key = f"{target_model}_{complexity}"
        for template in self.prompt_templates.values():
            if template.optimized_for == target_model and template.complexity_level == complexity:
                return template
        
        # Try model match with different complexity
        for template in self.prompt_templates.values():
            if template.optimized_for == target_model:
                return template
        
        # Fallback to any template for the complexity level
        for template in self.prompt_templates.values():
            if template.complexity_level == complexity:
                return template
        
        return None
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary for prompt injection"""
        
        if not context:
            return "No additional context provided."
        
        formatted = []
        for key, value in context.items():
            if value:
                formatted.append(f"{key}: {value}")
        
        return "\n".join(formatted) if formatted else "No additional context provided."
    
    def _infer_programming_language(self, task: str, context: Dict[str, Any] = None) -> str:
        """Infer programming language from task and context"""
        
        task_lower = task.lower()
        
        # Check context first
        if context:
            lang_context = context.get("language", "").lower()
            if lang_context:
                return lang_context
        
        # Language detection patterns
        language_patterns = {
            "python": ["python", "django", "flask", "pandas", "numpy", "pip", ".py"],
            "javascript": ["javascript", "js", "node", "npm", "react", "vue", "angular"],
            "typescript": ["typescript", "ts", "tsx"],
            "java": ["java", "spring", "maven", "gradle", ".java"],
            "go": ["golang", "go", ".go"],
            "rust": ["rust", "cargo", ".rs"],
            "c++": ["c++", "cpp", "cmake", ".cpp", ".cc"],
            "c": ["c programming", ".c"],
            "c#": ["c#", "csharp", "dotnet", ".cs"],
            "php": ["php", "laravel", "composer", ".php"],
            "ruby": ["ruby", "rails", "gem", ".rb"],
            "swift": ["swift", "ios", ".swift"],
            "kotlin": ["kotlin", "android", ".kt"]
        }
        
        for lang, patterns in language_patterns.items():
            if any(pattern in task_lower for pattern in patterns):
                return lang
        
        return "python"  # Default fallback
    
    def _apply_model_optimizations(self, prompt: str, target_model: str) -> str:
        """Apply model-specific optimizations"""
        
        optimizations = {
            "qwen3-coder": {
                "prefix": "You are an expert software engineer. ",
                "suffix": "\n\nGenerate complete, working code with proper structure and documentation."
            },
            "gemini": {
                "prefix": "You are a skilled project coordinator. ",
                "suffix": "\n\nProvide clear, actionable guidance with specific steps."
            },
            "claude": {
                "prefix": "You are a strategic decision maker. ",
                "suffix": "\n\nConsider all implications and provide comprehensive analysis."
            }
        }
        
        if target_model in optimizations:
            opt = optimizations[target_model]
            prompt = opt["prefix"] + prompt + opt["suffix"]
        
        return prompt
    
    async def _fallback_optimization(
        self, 
        task: str, 
        target_model: str, 
        complexity: str, 
        context: Dict[str, Any]
    ) -> str:
        """Fallback optimization when no template is available"""
        
        base_prompt = f"""Task: {task}

Complexity Level: {complexity}
Target System: {target_model}
Context: {self._format_context(context or {})}

Please provide a complete solution following best practices."""

        return self._apply_model_optimizations(base_prompt, target_model)
    
    async def compile_multi_agent_prompt(
        self,
        task: str,
        agents: List[str],  # ["shogun", "karo", "taisho"]
        context: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """
        Compile optimized prompts for multi-agent coordination
        
        This generates specific instructions for each agent in the hierarchy.
        """
        
        if not self.initialized:
            await self.initialize()
        
        compiled_prompts = {}
        
        # Map agents to their models and complexities
        agent_mapping = {
            "shogun": ("claude", "strategic"),
            "karo": ("gemini", "medium"), 
            "taisho": ("qwen3-coder", "medium"),
            "ashigaru": ("qwen3-coder", "simple")
        }
        
        for agent in agents:
            if agent in agent_mapping:
                model, complexity = agent_mapping[agent]
                
                # Create agent-specific context
                agent_context = (context or {}).copy()
                agent_context["role"] = agent
                agent_context["hierarchy_position"] = self._get_hierarchy_position(agent)
                
                # Generate optimized prompt
                compiled_prompts[agent] = await self.optimize_prompt(
                    task=task,
                    target_model=model,
                    complexity=complexity,
                    context=agent_context
                )
                
        logger.info(f"🔧 Compiled prompts for {len(compiled_prompts)} agents")
        return compiled_prompts
    
    def _get_hierarchy_position(self, agent: str) -> str:
        """Get hierarchy position description for agent"""
        
        positions = {
            "shogun": "Strategic decision maker and final authority",
            "karo": "Tactical coordinator and quality reviewer",
            "taisho": "Primary implementation specialist",
            "ashigaru": "Support specialist and parallel processor"
        }
        
        return positions.get(agent, "Team member")
    
    async def add_custom_template(self, template: PromptTemplate) -> None:
        """Add custom prompt template"""
        
        self.prompt_templates[template.name] = template
        logger.info(f"📝 Added custom template: {template.name}")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.prompt_templates.keys())
    
    async def shutdown(self) -> None:
        """Shutdown DSPy client"""
        logger.info("📴 DSPy Client shutdown complete")