"""
Bushidan Multi-Agent System v9.3 - Quality Metrics

Comprehensive quality measurement and tracking system.
Includes code complexity, security validation, and performance metrics.
"""

import re
import ast
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.logger import get_logger


logger = get_logger(__name__)


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplexityMetrics:
    """Code complexity metrics"""
    lines_of_code: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    max_nesting_depth: int
    number_of_functions: int
    average_function_length: float
    complexity_score: float  # 0-100
    risk_level: RiskLevel


@dataclass
class SecurityFindings:
    """Security validation results"""
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    security_score: float = 100.0  # 0-100


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    timestamp: str
    task_id: str
    complexity_metrics: ComplexityMetrics
    security_findings: SecurityFindings
    performance_indicators: Dict[str, Any]
    overall_quality_score: float  # 0-100
    recommendations: List[str]


class CodeComplexityAnalyzer:
    """
    Analyzes code complexity to determine appropriate review level
    
    Complexity factors:
    - Lines of code
    - Cyclomatic complexity
    - Nesting depth
    - Number of functions/classes
    - Security-sensitive operations
    """
    
    def __init__(self):
        self.security_keywords = [
            'password', 'token', 'secret', 'api_key', 'auth',
            'sql', 'exec', 'eval', 'subprocess', 'os.system',
            'pickle', 'yaml.load', 'input', 'raw_input'
        ]
        
    def analyze(self, code: str, language: str = "python") -> ComplexityMetrics:
        """
        Analyze code complexity
        
        Args:
            code: Source code to analyze
            language: Programming language (python, javascript, etc.)
            
        Returns:
            ComplexityMetrics with comprehensive analysis
        """
        
        if language == "python":
            return self._analyze_python(code)
        else:
            return self._analyze_generic(code)
    
    def _analyze_python(self, code: str) -> ComplexityMetrics:
        """Analyze Python code complexity"""
        
        try:
            tree = ast.parse(code)
            
            # Count lines of code (excluding blanks and comments)
            lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
            loc = len(lines)
            
            # Count functions and classes
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            num_functions = len(functions) + len(classes)
            
            # Calculate cyclomatic complexity (simplified)
            decision_points = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                    decision_points += 1
                elif isinstance(node, ast.BoolOp):
                    decision_points += len(node.values) - 1
            
            cyclomatic = decision_points + 1
            
            # Calculate max nesting depth
            max_depth = self._calculate_max_depth(tree)
            
            # Calculate average function length
            if functions:
                total_func_lines = sum(
                    len(ast.unparse(f).split('\n')) for f in functions
                )
                avg_func_length = total_func_lines / len(functions)
            else:
                avg_func_length = loc
            
            # Calculate cognitive complexity (simplified heuristic)
            cognitive = self._calculate_cognitive_complexity(tree)
            
            # Calculate overall complexity score (0-100, lower is simpler)
            complexity_score = self._calculate_complexity_score(
                loc, cyclomatic, cognitive, max_depth, avg_func_length
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(code, complexity_score)
            
            return ComplexityMetrics(
                lines_of_code=loc,
                cyclomatic_complexity=cyclomatic,
                cognitive_complexity=cognitive,
                max_nesting_depth=max_depth,
                number_of_functions=num_functions,
                average_function_length=avg_func_length,
                complexity_score=complexity_score,
                risk_level=risk_level
            )
            
        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code: {e}")
            # Return fallback metrics
            return self._analyze_generic(code)
    
    def _analyze_generic(self, code: str) -> ComplexityMetrics:
        """Fallback analysis for any language"""
        
        lines = [l.strip() for l in code.split('\n') if l.strip()]
        loc = len(lines)
        
        # Simple heuristics
        decision_keywords = ['if', 'else', 'for', 'while', 'switch', 'case']
        decision_count = sum(
            line.count(keyword) for line in lines for keyword in decision_keywords
        )
        
        cyclomatic = decision_count + 1
        cognitive = decision_count * 2  # Rough estimate
        
        # Estimate nesting by counting indentation
        max_depth = max((len(line) - len(line.lstrip())) // 4 for line in code.split('\n') if line.strip()) if lines else 0
        
        complexity_score = min(100, (loc / 10) + (cyclomatic * 5) + (max_depth * 10))
        risk_level = self._determine_risk_level(code, complexity_score)
        
        return ComplexityMetrics(
            lines_of_code=loc,
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            max_nesting_depth=max_depth,
            number_of_functions=0,
            average_function_length=float(loc),
            complexity_score=complexity_score,
            risk_level=risk_level
        )
    
    def _calculate_max_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        
        def depth(node, current=0):
            depths = [current]
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    depths.append(depth(child, current + 1))
                else:
                    depths.append(depth(child, current))
            return max(depths)
        
        return depth(tree)
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (simplified)"""
        
        cognitive = 0
        nesting_level = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                cognitive += 1 + nesting_level
            elif isinstance(node, ast.BoolOp):
                cognitive += len(node.values) - 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nesting_level += 1
        
        return cognitive
    
    def _calculate_complexity_score(
        self, 
        loc: int, 
        cyclomatic: int, 
        cognitive: int, 
        max_depth: int,
        avg_func_length: float
    ) -> float:
        """
        Calculate overall complexity score (0-100)
        
        Lower is simpler:
        - 0-20: Very simple
        - 21-40: Simple
        - 41-60: Moderate
        - 61-80: Complex
        - 81-100: Very complex
        """
        
        # Normalize each metric to 0-20 scale
        loc_score = min(20, (loc / 200) * 20)
        cyclomatic_score = min(20, (cyclomatic / 10) * 20)
        cognitive_score = min(20, (cognitive / 15) * 20)
        depth_score = min(20, (max_depth / 5) * 20)
        length_score = min(20, (avg_func_length / 50) * 20)
        
        total = loc_score + cyclomatic_score + cognitive_score + depth_score + length_score
        
        return round(total, 2)
    
    def _determine_risk_level(self, code: str, complexity_score: float) -> RiskLevel:
        """Determine risk level based on complexity and security keywords"""
        
        code_lower = code.lower()
        security_hits = sum(1 for keyword in self.security_keywords if keyword in code_lower)
        
        # High risk if many security keywords
        if security_hits >= 3:
            return RiskLevel.CRITICAL
        elif security_hits >= 2:
            return RiskLevel.HIGH
        
        # Otherwise, base on complexity
        if complexity_score >= 80:
            return RiskLevel.HIGH
        elif complexity_score >= 60:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


class SecurityValidator:
    """
    Validates code for common security vulnerabilities
    
    Checks for:
    - Hardcoded secrets
    - SQL injection risks
    - Command injection risks
    - Unsafe deserialization
    - XSS vulnerabilities
    """
    
    def __init__(self):
        self.patterns = {
            'hardcoded_secret': [
                (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
                (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
                (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token detected'),
                (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
            ],
            'sql_injection': [
                (r'execute\s*\(\s*["\'].*%s.*["\']', 'Potential SQL injection via string formatting'),
                (r'execute\s*\(\s*f["\'].*\{.*\}.*["\']', 'Potential SQL injection via f-string'),
                (r'\.format\s*\(.*\).*execute', 'Potential SQL injection via .format()'),
            ],
            'command_injection': [
                (r'os\.system\s*\(', 'Unsafe os.system() usage'),
                (r'subprocess\.call\s*\(.*shell\s*=\s*True', 'shell=True with subprocess'),
                (r'exec\s*\(', 'Dangerous exec() usage'),
                (r'eval\s*\(', 'Dangerous eval() usage'),
            ],
            'unsafe_deserialization': [
                (r'pickle\.loads?\s*\(', 'Unsafe pickle deserialization'),
                (r'yaml\.load\s*\((?!.*Loader)', 'Unsafe YAML loading'),
            ],
        }
    
    def validate(self, code: str) -> SecurityFindings:
        """
        Validate code for security issues
        
        Returns:
            SecurityFindings with vulnerabilities and risk level
        """
        
        vulnerabilities = []
        warnings = []
        
        for category, patterns in self.patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    line_number = code[:match.start()].count('\n') + 1
                    
                    finding = {
                        'category': category,
                        'description': description,
                        'line': line_number,
                        'code_snippet': match.group(0),
                        'severity': 'high' if category in ['sql_injection', 'command_injection'] else 'medium'
                    }
                    
                    if finding['severity'] == 'high':
                        vulnerabilities.append(finding)
                    else:
                        warnings.append(finding)
        
        # Determine overall risk level
        if len(vulnerabilities) >= 3:
            risk_level = RiskLevel.CRITICAL
        elif len(vulnerabilities) >= 1:
            risk_level = RiskLevel.HIGH
        elif len(warnings) >= 3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Calculate security score
        security_score = max(0, 100 - (len(vulnerabilities) * 20) - (len(warnings) * 5))
        
        return SecurityFindings(
            vulnerabilities=vulnerabilities,
            warnings=warnings,
            risk_level=risk_level,
            security_score=security_score
        )


class QualityMetricsCollector:
    """
    Comprehensive quality metrics collection and reporting
    """
    
    def __init__(self):
        self.complexity_analyzer = CodeComplexityAnalyzer()
        self.security_validator = SecurityValidator()
        self.reports: List[QualityReport] = []
    
    def collect_metrics(
        self, 
        task_id: str,
        code: str,
        language: str = "python",
        performance_data: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Collect comprehensive quality metrics
        
        Args:
            task_id: Unique task identifier
            code: Source code to analyze
            language: Programming language
            performance_data: Optional performance measurements
            
        Returns:
            Comprehensive QualityReport
        """
        
        # Analyze complexity
        complexity_metrics = self.complexity_analyzer.analyze(code, language)
        
        # Validate security
        security_findings = self.security_validator.validate(code)
        
        # Collect performance indicators
        performance_indicators = performance_data or {}
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            complexity_metrics,
            security_findings,
            performance_indicators
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            complexity_metrics,
            security_findings
        )
        
        report = QualityReport(
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            complexity_metrics=complexity_metrics,
            security_findings=security_findings,
            performance_indicators=performance_indicators,
            overall_quality_score=quality_score,
            recommendations=recommendations
        )
        
        self.reports.append(report)
        
        logger.info(
            f"📊 Quality metrics collected for {task_id}: "
            f"Score {quality_score:.1f}/100, "
            f"Risk {complexity_metrics.risk_level.value}"
        )
        
        return report
    
    def _calculate_quality_score(
        self,
        complexity: ComplexityMetrics,
        security: SecurityFindings,
        performance: Dict[str, Any]
    ) -> float:
        """
        Calculate overall quality score (0-100, higher is better)
        
        Weighting:
        - Complexity: 40%
        - Security: 40%
        - Performance: 20%
        """
        
        # Complexity score (invert since lower complexity is better)
        complexity_score = max(0, 100 - complexity.complexity_score) * 0.4
        
        # Security score
        security_score = security.security_score * 0.4
        
        # Performance score (simplified)
        performance_score = performance.get('score', 80) * 0.2
        
        total = complexity_score + security_score + performance_score
        
        return round(total, 2)
    
    def _generate_recommendations(
        self,
        complexity: ComplexityMetrics,
        security: SecurityFindings
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Complexity recommendations
        if complexity.cyclomatic_complexity > 10:
            recommendations.append(
                f"High cyclomatic complexity ({complexity.cyclomatic_complexity}). "
                "Consider breaking down complex functions."
            )
        
        if complexity.max_nesting_depth > 4:
            recommendations.append(
                f"Deep nesting detected (depth {complexity.max_nesting_depth}). "
                "Extract nested logic into separate functions."
            )
        
        if complexity.average_function_length > 50:
            recommendations.append(
                f"Long functions detected (avg {complexity.average_function_length:.0f} lines). "
                "Split into smaller, focused functions."
            )
        
        # Security recommendations
        for vuln in security.vulnerabilities:
            recommendations.append(
                f"🔒 {vuln['description']} at line {vuln['line']}"
            )
        
        if security.warnings:
            recommendations.append(
                f"Review {len(security.warnings)} security warnings before deployment"
            )
        
        return recommendations
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all reports"""
        
        if not self.reports:
            return {}
        
        avg_quality = sum(r.overall_quality_score for r in self.reports) / len(self.reports)
        
        risk_distribution = {
            'low': sum(1 for r in self.reports if r.complexity_metrics.risk_level == RiskLevel.LOW),
            'medium': sum(1 for r in self.reports if r.complexity_metrics.risk_level == RiskLevel.MEDIUM),
            'high': sum(1 for r in self.reports if r.complexity_metrics.risk_level == RiskLevel.HIGH),
            'critical': sum(1 for r in self.reports if r.complexity_metrics.risk_level == RiskLevel.CRITICAL),
        }
        
        return {
            'total_reports': len(self.reports),
            'average_quality_score': round(avg_quality, 2),
            'risk_distribution': risk_distribution,
            'recent_scores': [r.overall_quality_score for r in self.reports[-10:]]
        }
