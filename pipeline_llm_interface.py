"""
Pipeline LLM Interface - Natural Language Composition Interface

This module implements the LLM interface for creating and executing complex
analytical pipelines through natural language intent processing.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
import logging
from datetime import datetime

from pipeline_composition_framework import (
    CompositionBuilder,
    AnalysisComposition,
    CompositionStage,
    CompositionValidator
)
from pipeline_dataflow_engine import PipelineDataFlow
from pipeline_optimization_engine import WorkflowOptimizationEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Intent Parsing and Analysis
# ============================================================================

@dataclass
class AnalyticalStep:
    """Represents a single analytical step parsed from intent."""
    intent_category: str
    function: str
    required_capabilities: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1


@dataclass
class ParsedIntent:
    """Result of parsing analytical intent."""
    original_intent: str
    steps: List[AnalyticalStep]
    data_context_hints: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    output_format_preferences: List[str] = field(default_factory=list)


class AnalyticalIntentParser:
    """
    Parses natural language analytical intent into structured analytical steps.
    
    Handles complex intents like:
    - "Predict customer churn with feature importance and model performance"
    - "Time series forecasting with trend decomposition and seasonal adjustment"
    - "Geospatial clustering with visualization and statistical validation"
    """
    
    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.capability_keywords = self._initialize_capability_keywords()
        self.requirement_keywords = self._initialize_requirement_keywords()
    
    def parse(self, intent: str) -> ParsedIntent:
        """
        Parse natural language intent into structured analytical steps.
        
        Args:
            intent: Natural language description of analytical goals
            
        Returns:
            ParsedIntent with structured analytical steps
        """
        intent_lower = intent.lower()
        
        # Extract performance requirements
        performance_reqs = self._extract_performance_requirements(intent_lower)
        
        # Extract data context hints
        data_context = self._extract_data_context_hints(intent_lower)
        
        # Extract output format preferences
        output_prefs = self._extract_output_preferences(intent_lower)
        
        # Identify analytical steps
        steps = self._identify_analytical_steps(intent_lower)
        
        return ParsedIntent(
            original_intent=intent,
            steps=steps,
            data_context_hints=data_context,
            performance_requirements=performance_reqs,
            output_format_preferences=output_prefs
        )
    
    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for recognizing analytical intents."""
        return {
            'prediction': [
                'predict', 'forecast', 'estimate', 'model', 'classify'
            ],
            'analysis': [
                'analyze', 'examine', 'investigate', 'explore', 'study'
            ],
            'clustering': [
                'cluster', 'segment', 'group', 'categorize', 'partition'
            ],
            'visualization': [
                'visualize', 'plot', 'chart', 'graph', 'display', 'show'
            ],
            'statistical_analysis': [
                'statistics', 'statistical', 'correlation', 'regression', 'significance'
            ],
            'feature_engineering': [
                'features', 'feature engineering', 'feature selection', 'feature importance'
            ],
            'time_series': [
                'time series', 'temporal', 'trend', 'seasonality', 'decomposition'
            ],
            'geospatial': [
                'geospatial', 'geographic', 'location', 'spatial', 'coordinate'
            ],
            'optimization': [
                'optimize', 'optimization', 'minimize', 'maximize', 'best'
            ],
            'evaluation': [
                'evaluate', 'assessment', 'performance', 'metrics', 'validation'
            ]
        }
    
    def _initialize_capability_keywords(self) -> Dict[str, List[str]]:
        """Initialize keywords for capability requirements."""
        return {
            'interpretability': ['interpretable', 'explainable', 'interpretation', 'explain'],
            'scalability': ['scalable', 'large scale', 'big data', 'distributed'],
            'real_time': ['real time', 'real-time', 'streaming', 'online'],
            'high_accuracy': ['accurate', 'precision', 'high performance', 'best'],
            'robust': ['robust', 'stable', 'reliable', 'consistent'],
            'fast': ['fast', 'quick', 'rapid', 'efficient', 'speed']
        }
    
    def _initialize_requirement_keywords(self) -> Dict[str, List[str]]:
        """Initialize keywords for analytical requirements."""
        return {
            'cross_validation': ['cross validation', 'cv', 'validation'],
            'hyperparameter_tuning': ['hyperparameter', 'tuning', 'optimization'],
            'feature_selection': ['feature selection', 'feature importance'],
            'model_comparison': ['compare models', 'model comparison', 'benchmark'],
            'statistical_significance': ['significant', 'significance', 'p-value'],
            'confidence_intervals': ['confidence interval', 'uncertainty'],
            'visualization': ['plot', 'chart', 'visualization', 'graph']
        }
    
    def _extract_performance_requirements(self, intent: str) -> Dict[str, Any]:
        """Extract performance requirements from intent."""
        requirements = {}
        
        # Extract time constraints
        time_patterns = [
            (r'within (\d+) minutes?', 'max_execution_time_minutes'),
            (r'under (\d+) seconds?', 'max_execution_time_seconds'),
            (r'less than (\d+) hours?', 'max_execution_time_hours')
        ]
        
        for pattern, key in time_patterns:
            match = re.search(pattern, intent)
            if match:
                requirements[key] = int(match.group(1))
        
        # Extract memory constraints
        memory_patterns = [
            (r'using less than (\d+)gb', 'max_memory_gb'),
            (r'memory limit (\d+)mb', 'max_memory_mb')
        ]
        
        for pattern, key in memory_patterns:
            match = re.search(pattern, intent)
            if match:
                requirements[key] = int(match.group(1))
        
        return requirements
    
    def _extract_data_context_hints(self, intent: str) -> Dict[str, Any]:
        """Extract hints about data context and characteristics."""
        context = {}
        
        # Data size hints
        if any(word in intent for word in ['large', 'big', 'massive', 'millions']):
            context['expected_data_size'] = 'large'
        elif any(word in intent for word in ['small', 'limited', 'sample']):
            context['expected_data_size'] = 'small'
        
        # Data type hints
        if any(word in intent for word in ['time series', 'temporal', 'sequential']):
            context['data_type'] = 'time_series'
        elif any(word in intent for word in ['geospatial', 'geographic', 'location']):
            context['data_type'] = 'geospatial'
        elif any(word in intent for word in ['text', 'nlp', 'language']):
            context['data_type'] = 'text'
        
        # Quality expectations
        if any(word in intent for word in ['clean', 'preprocessed']):
            context['data_quality'] = 'high'
        elif any(word in intent for word in ['raw', 'messy', 'dirty']):
            context['data_quality'] = 'low'
        
        return context
    
    def _extract_output_preferences(self, intent: str) -> List[str]:
        """Extract preferred output formats."""
        preferences = []
        
        if any(word in intent for word in ['visualization', 'plot', 'chart', 'graph']):
            preferences.append('visualization')
        
        if any(word in intent for word in ['report', 'summary', 'document']):
            preferences.append('report')
        
        if any(word in intent for word in ['model', 'predictor', 'classifier']):
            preferences.append('model')
        
        if any(word in intent for word in ['data', 'dataset', 'table', 'results']):
            preferences.append('data')
        
        return preferences
    
    def _identify_analytical_steps(self, intent: str) -> List[AnalyticalStep]:
        """Identify sequence of analytical steps from intent."""
        steps = []
        
        # Identify main analytical categories
        identified_categories = []
        for category, keywords in self.intent_patterns.items():
            if any(keyword in intent for keyword in keywords):
                identified_categories.append(category)
        
        # Map categories to analytical steps
        for category in identified_categories:
            step = self._create_step_from_category(category, intent)
            if step:
                steps.append(step)
        
        # Add implicit steps based on identified categories
        implicit_steps = self._infer_implicit_steps(identified_categories, intent)
        steps.extend(implicit_steps)
        
        # Sort steps by logical order
        steps = self._order_analytical_steps(steps)
        
        return steps
    
    def _create_step_from_category(self, category: str, intent: str) -> Optional[AnalyticalStep]:
        """Create analytical step from identified category."""
        category_mappings = {
            'prediction': AnalyticalStep(
                intent_category='modeling',
                function='train_predictive_model',
                required_capabilities=['supervised_learning'],
                requirements=self._extract_requirements_for_category(category, intent)
            ),
            'clustering': AnalyticalStep(
                intent_category='clustering',
                function='perform_clustering',
                required_capabilities=['unsupervised_learning'],
                requirements=self._extract_requirements_for_category(category, intent)
            ),
            'statistical_analysis': AnalyticalStep(
                intent_category='analysis',
                function='statistical_analysis',
                required_capabilities=['statistical_testing'],
                requirements=self._extract_requirements_for_category(category, intent)
            ),
            'feature_engineering': AnalyticalStep(
                intent_category='transformation',
                function='engineer_features',
                required_capabilities=['feature_creation'],
                requirements=self._extract_requirements_for_category(category, intent)
            ),
            'time_series': AnalyticalStep(
                intent_category='time_series_analysis',
                function='time_series_analysis',
                required_capabilities=['temporal_modeling'],
                requirements=self._extract_requirements_for_category(category, intent)
            ),
            'visualization': AnalyticalStep(
                intent_category='visualization',
                function='create_visualization',
                required_capabilities=['plotting'],
                requirements=self._extract_requirements_for_category(category, intent)
            ),
            'evaluation': AnalyticalStep(
                intent_category='evaluation',
                function='evaluate_model',
                required_capabilities=['performance_metrics'],
                requirements=self._extract_requirements_for_category(category, intent)
            )
        }
        
        return category_mappings.get(category)
    
    def _extract_requirements_for_category(self, category: str, intent: str) -> List[str]:
        """Extract specific requirements for a category."""
        requirements = []
        
        for req_name, keywords in self.requirement_keywords.items():
            if any(keyword in intent for keyword in keywords):
                requirements.append(req_name)
        
        return requirements
    
    def _infer_implicit_steps(self, identified_categories: List[str], intent: str) -> List[AnalyticalStep]:
        """Infer implicit steps based on identified categories."""
        implicit_steps = []
        
        # If prediction is requested, we likely need data preparation
        if 'prediction' in identified_categories:
            if 'feature_engineering' not in identified_categories:
                implicit_steps.append(AnalyticalStep(
                    intent_category='transformation',
                    function='prepare_features',
                    required_capabilities=['data_preparation'],
                    priority=0  # High priority (early in pipeline)
                ))
            
            # Evaluation is often implicit with prediction
            if 'evaluation' not in identified_categories:
                implicit_steps.append(AnalyticalStep(
                    intent_category='evaluation',
                    function='evaluate_model',
                    required_capabilities=['performance_metrics'],
                    priority=3  # Lower priority (later in pipeline)
                ))
        
        # Data ingestion is almost always needed
        if not any(cat in ['data_ingestion'] for cat in identified_categories):
            implicit_steps.append(AnalyticalStep(
                intent_category='data_ingestion',
                function='load_data',
                required_capabilities=['data_loading'],
                priority=0  # Highest priority (first step)
            ))
        
        return implicit_steps
    
    def _order_analytical_steps(self, steps: List[AnalyticalStep]) -> List[AnalyticalStep]:
        """Order analytical steps in logical sequence."""
        # Define typical analytical workflow order
        category_order = {
            'data_ingestion': 0,
            'data_cleaning': 1,
            'transformation': 2,
            'analysis': 3,
            'clustering': 3,
            'time_series_analysis': 3,
            'modeling': 4,
            'evaluation': 5,
            'visualization': 6,
            'reporting': 7
        }
        
        # Sort by category order, then by priority
        steps.sort(key=lambda step: (
            category_order.get(step.intent_category, 5),
            step.priority
        ))
        
        return steps


# ============================================================================
# Tool Selection and Mapping
# ============================================================================

@dataclass
class Tool:
    """Represents an available analytical tool."""
    name: str
    category: str
    capabilities: List[str]
    default_parameters: Dict[str, Any]
    performance_characteristics: Dict[str, Any]
    compatibility_score_func: Optional[callable] = None


class ToolSelector:
    """Selects optimal tools for analytical steps based on requirements and context."""
    
    def __init__(self, tool_registry: 'ToolRegistry'):
        self.registry = tool_registry
        self.selection_weights = {
            'capability_match': 0.4,
            'performance_fit': 0.3,
            'data_compatibility': 0.2,
            'user_preference': 0.1
        }
    
    def select_best_tool(self, step: AnalyticalStep, data_context: Dict[str, Any]) -> Optional[Tool]:
        """Select the best tool for an analytical step."""
        candidate_tools = self.registry.find_tools_for_intent(
            step.intent_category,
            step.required_capabilities,
            data_context
        )
        
        if not candidate_tools:
            logger.warning(f"No tools found for step: {step.intent_category}")
            return None
        
        # Score each candidate tool
        scored_tools = []
        for tool in candidate_tools:
            score = self._score_tool(tool, step, data_context)
            scored_tools.append((tool, score))
        
        # Return highest scoring tool
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        best_tool, best_score = scored_tools[0]
        
        logger.info(f"Selected tool {best_tool.name} for {step.intent_category} (score: {best_score:.2f})")
        return best_tool
    
    def _score_tool(self, tool: Tool, step: AnalyticalStep, data_context: Dict[str, Any]) -> float:
        """Score a tool's suitability for an analytical step."""
        total_score = 0.0
        
        # Capability match score
        capability_score = self._score_capability_match(tool, step)
        total_score += capability_score * self.selection_weights['capability_match']
        
        # Performance fit score
        performance_score = self._score_performance_fit(tool, data_context)
        total_score += performance_score * self.selection_weights['performance_fit']
        
        # Data compatibility score
        compatibility_score = self._score_data_compatibility(tool, data_context)
        total_score += compatibility_score * self.selection_weights['data_compatibility']
        
        return total_score
    
    def _score_capability_match(self, tool: Tool, step: AnalyticalStep) -> float:
        """Score how well tool capabilities match step requirements."""
        if not step.required_capabilities:
            return 1.0
        
        matches = sum(1 for cap in step.required_capabilities if cap in tool.capabilities)
        return matches / len(step.required_capabilities)
    
    def _score_performance_fit(self, tool: Tool, data_context: Dict[str, Any]) -> float:
        """Score how well tool performance characteristics fit data context."""
        performance = tool.performance_characteristics
        score = 1.0  # Base score
        
        # Adjust based on data size
        data_size = data_context.get('expected_data_size', 'medium')
        if data_size == 'large':
            if performance.get('scalable', False):
                score += 0.2
            else:
                score -= 0.3
        
        # Adjust based on speed requirements
        if 'fast' in data_context.get('requirements', []):
            if performance.get('fast_execution', False):
                score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_data_compatibility(self, tool: Tool, data_context: Dict[str, Any]) -> float:
        """Score data type compatibility."""
        data_type = data_context.get('data_type', 'tabular')
        
        # Tool-specific compatibility rules
        compatibility_matrix = {
            ('time_series', 'time_series_analysis'): 1.0,
            ('geospatial', 'geospatial_analysis'): 1.0,
            ('text', 'nlp_analysis'): 1.0,
            ('tabular', 'statistical_analysis'): 0.9,
            ('tabular', 'machine_learning'): 0.9
        }
        
        key = (data_type, tool.category)
        return compatibility_matrix.get(key, 0.7)  # Default compatibility


# ============================================================================
# Main LLM Interface
# ============================================================================

class LLMCompositionInterface:
    """
    Natural language interface for LLM agents to create and execute complex analytical pipelines.
    
    Key Features:
    - Intent-based pipeline specification
    - Automatic tool selection based on analytical goals
    - Progressive disclosure of complexity
    - Rich result interpretation for downstream tools
    """
    
    def __init__(self, tool_registry: 'ToolRegistry', validator: CompositionValidator,
                 optimization_engine: Optional[WorkflowOptimizationEngine] = None):
        self.registry = tool_registry
        self.validator = validator
        self.optimization_engine = optimization_engine
        self.intent_parser = AnalyticalIntentParser()
        self.tool_selector = ToolSelector(tool_registry)
    
    def create_composition_from_intent(self, intent: str, data_context: Dict[str, Any]) -> AnalysisComposition:
        """
        Create pipeline composition from analytical intent expressed in natural language.
        
        Args:
            intent: Natural language description of analytical goals
            data_context: Context about the data being analyzed
            
        Returns:
            Complete AnalysisComposition ready for execution
        """
        logger.info(f"Creating composition from intent: {intent}")
        
        # Parse analytical intent
        parsed_intent = self.intent_parser.parse(intent)
        
        # Build composition using builder pattern
        composition_builder = CompositionBuilder()
        composition_builder.with_intent(intent)
        composition_builder.with_description(f"Auto-generated pipeline: {intent}")
        
        # Apply performance requirements if specified
        if parsed_intent.performance_requirements:
            perf_reqs = parsed_intent.performance_requirements
            composition_builder.with_performance_limits(
                max_execution_time_seconds=perf_reqs.get('max_execution_time_seconds'),
                max_memory_mb=perf_reqs.get('max_memory_mb')
            )
        
        # Create stages from analytical steps
        for step in parsed_intent.steps:
            # Select appropriate tool for this step
            best_tool = self.tool_selector.select_best_tool(step, data_context)
            
            if best_tool:
                # Infer parameters from step requirements and data context
                parameters = self._infer_parameters(best_tool, step, data_context)
                
                composition_builder.add_stage(
                    tool_name=best_tool.name,
                    function=step.function,
                    parameters=parameters
                )
            else:
                logger.warning(f"No suitable tool found for step: {step.intent_category}")
        
        # Build initial composition
        composition = composition_builder.build()
        
        # Validate and auto-fix common issues
        validation_result = self.validator.validate_composition(composition)
        
        if not validation_result.valid:
            composition = self._auto_fix_composition(composition, validation_result)
        
        # Apply optimizations if available
        if self.optimization_engine:
            optimized = self.optimization_engine.optimize_composition(composition)
            composition.stages = optimized.stages
            
            logger.info(f"Applied optimizations: {optimized.optimization_log}")
        
        logger.info(f"Created composition with {len(composition.stages)} stages")
        return composition
    
    def _infer_parameters(self, tool: Tool, step: AnalyticalStep, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently infer tool parameters based on data characteristics and analytical intent.
        """
        inferred_params = tool.default_parameters.copy()
        
        # Data-driven parameter inference
        data_size = data_context.get('expected_data_size', 'medium')
        if data_size == 'large':
            # Adjust for large datasets
            inferred_params['chunk_size'] = 10000
            inferred_params['streaming'] = True
            
            # Reduce complexity for scalability
            if 'max_iter' in inferred_params:
                inferred_params['max_iter'] = min(inferred_params.get('max_iter', 1000), 500)
            
        elif data_size == 'small':
            # Optimize for small datasets
            inferred_params['streaming'] = False
            
            # Increase complexity for better performance
            if 'max_iter' in inferred_params:
                inferred_params['max_iter'] = max(inferred_params.get('max_iter', 100), 1000)
        
        # Intent-driven parameter inference
        if step.intent_category == 'modeling':
            if 'high_accuracy' in step.requirements:
                inferred_params['cross_validation'] = True
                inferred_params['hyperparameter_tuning'] = True
            
            if 'interpretability' in step.requirements:
                inferred_params['feature_importance'] = True
                inferred_params['model_explanation'] = True
        
        # Data quality adjustments
        data_quality = data_context.get('data_quality', 'medium')
        if data_quality == 'low':
            inferred_params['robust'] = True
            inferred_params['handle_missing'] = True
        
        return inferred_params
    
    def _auto_fix_composition(self, composition: AnalysisComposition, 
                             validation_result: ValidationResult) -> AnalysisComposition:
        """Automatically fix common composition issues."""
        logger.info("Auto-fixing composition issues")
        
        # Apply auto-fixes from validation
        for fix in validation_result.auto_fixes:
            if fix['fix_type'] == 'add_conversion':
                stage_index = fix['stage_index']
                if stage_index < len(composition.stages):
                    composition.stages[stage_index].add_conversion(fix['conversion_path'])
                    logger.info(f"Applied auto-fix: {fix['message']}")
        
        return composition


# ============================================================================
# MCP Tool Integration
# ============================================================================

def create_analytical_pipeline(
    analytical_intent: str,
    data_source: str,
    performance_requirements: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    MCP tool implementation for creating and executing analytical pipelines.
    
    This function would be decorated with @mcp.tool in the actual MCP server.
    """
    try:
        # Initialize components (would be injected in real implementation)
        tool_registry = None  # MockToolRegistry()
        validator = None      # CompositionValidator(tool_registry)
        optimization_engine = None  # WorkflowOptimizationEngine(cache, scheduler)
        
        interface = LLMCompositionInterface(tool_registry, validator, optimization_engine)
        
        # Analyze data source for context
        data_context = _analyze_data_source(data_source)
        
        # Apply performance requirements
        if performance_requirements:
            data_context['performance_requirements'] = performance_requirements
        
        # Create composition from intent
        composition = interface.create_composition_from_intent(analytical_intent, data_context)
        
        # Execute pipeline (would use actual executor in real implementation)
        # result = pipeline_executor.execute_composition(composition)
        
        return {
            'success': True,
            'pipeline_created': True,
            'composition_id': composition.composition_id,
            'stages': len(composition.stages),
            'estimated_duration_seconds': composition.get_estimated_duration(),
            'estimated_memory_mb': composition.get_estimated_memory(),
            'analytical_intent': composition.analytical_intent,
            'stage_summary': [
                {'tool': stage.tool_name, 'function': stage.function}
                for stage in composition.stages
            ],
            'next_actions': [
                'Execute pipeline with execute_pipeline_composition()',
                'Monitor execution with get_pipeline_status()',
                'Retrieve results with get_pipeline_results()'
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to create analytical pipeline: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'suggestions': [
                'Check that analytical intent is clearly specified',
                'Verify data source is accessible',
                'Ensure performance requirements are reasonable'
            ]
        }


def _analyze_data_source(data_source: str) -> Dict[str, Any]:
    """Analyze data source to infer context."""
    context = {
        'data_source': data_source,
        'expected_data_size': 'medium',
        'data_type': 'tabular',
        'data_quality': 'medium'
    }
    
    # Infer characteristics from data source name/path
    if 'big' in data_source.lower() or 'large' in data_source.lower():
        context['expected_data_size'] = 'large'
    
    if any(word in data_source.lower() for word in ['time', 'temporal', 'series']):
        context['data_type'] = 'time_series'
    
    if any(word in data_source.lower() for word in ['geo', 'location', 'spatial']):
        context['data_type'] = 'geospatial'
    
    return context


# ============================================================================
# Example Usage
# ============================================================================

def demo_llm_interface():
    """Demonstrate the LLM interface capabilities."""
    intents = [
        "Predict customer churn with feature importance and model performance evaluation",
        "Analyze sales trends over time with seasonal decomposition and forecasting", 
        "Cluster customers by behavior with geospatial analysis and visualization",
        "Optimize marketing campaign performance with A/B testing and statistical analysis"
    ]
    
    parser = AnalyticalIntentParser()
    
    for intent in intents:
        print(f"\nIntent: {intent}")
        parsed = parser.parse(intent)
        print(f"Steps identified: {len(parsed.steps)}")
        for i, step in enumerate(parsed.steps):
            print(f"  {i+1}. {step.intent_category}: {step.function}")
        print(f"Performance requirements: {parsed.performance_requirements}")
        print(f"Data context hints: {parsed.data_context_hints}")


if __name__ == "__main__":
    demo_llm_interface()