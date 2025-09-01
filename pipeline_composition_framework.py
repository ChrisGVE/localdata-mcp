"""
Pipeline Composition Framework - Concrete Python Architecture

This module provides the core architectural components for pipeline composition
in LocalData MCP v2.0, building on the established first principles and framework patterns.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import json
import logging
from datetime import datetime, timedelta

# Import base framework components (from 32.2)
from analysis_pipeline_framework import (
    AnalysisPipelineBase, 
    CompositionMetadata,
    StreamingContext,
    MemoryManager
)

logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Structures for Composition
# ============================================================================

class ExecutionStrategy(Enum):
    """Execution strategies for pipeline stages."""
    DIRECT = "direct"
    STREAMING = "streaming"
    PARALLEL = "parallel"
    CACHED = "cached"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"      # Fail on any incompatibility
    LENIENT = "lenient"    # Auto-fix common issues
    PERMISSIVE = "permissive"  # Allow risky compositions with warnings


@dataclass
class CompositionStage:
    """
    Represents a single stage in an analytical pipeline composition.
    
    Designed for LLM-friendly configuration with automatic optimization.
    """
    tool_name: str
    function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Type compatibility
    expected_input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    
    # Conversion handling
    requires_conversion: bool = False
    conversion_path: Optional[str] = None
    
    # Execution configuration
    execution_strategy: ExecutionStrategy = ExecutionStrategy.DIRECT
    chunk_size: Optional[int] = None
    is_streaming: bool = False
    is_parallel_safe: bool = True
    
    # Error handling
    is_optional: bool = False
    fallback_tools: List[str] = field(default_factory=list)
    
    # Performance
    estimated_duration_seconds: float = 0.0
    estimated_memory_mb: float = 0.0
    is_deterministic: bool = True
    has_time_dependency: bool = False
    
    # Caching
    cacheable: bool = False
    cache_ttl_hours: Optional[int] = None
    
    def add_conversion(self, conversion_path: str):
        """Add automatic type conversion to this stage."""
        self.requires_conversion = True
        self.conversion_path = conversion_path
        
    def enable_streaming(self):
        """Enable streaming execution for this stage."""
        self.is_streaming = True
        self.execution_strategy = ExecutionStrategy.STREAMING
        
    def copy(self) -> 'CompositionStage':
        """Create a copy of this stage for modification."""
        return CompositionStage(
            tool_name=self.tool_name,
            function=self.function,
            parameters=self.parameters.copy(),
            expected_input_types=self.expected_input_types.copy(),
            output_types=self.output_types.copy(),
            requires_conversion=self.requires_conversion,
            conversion_path=self.conversion_path,
            execution_strategy=self.execution_strategy,
            chunk_size=self.chunk_size,
            is_streaming=self.is_streaming,
            is_parallel_safe=self.is_parallel_safe,
            is_optional=self.is_optional,
            fallback_tools=self.fallback_tools.copy(),
            estimated_duration_seconds=self.estimated_duration_seconds,
            estimated_memory_mb=self.estimated_memory_mb,
            is_deterministic=self.is_deterministic,
            has_time_dependency=self.has_time_dependency,
            cacheable=self.cacheable,
            cache_ttl_hours=self.cache_ttl_hours
        )


@dataclass
class AnalysisComposition:
    """
    Represents a complete analytical pipeline composition.
    
    Designed for complex multi-stage workflows with automatic optimization.
    """
    composition_id: str
    stages: List[CompositionStage]
    initial_data: Any = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    analytical_intent: str = ""
    
    # Performance constraints
    max_execution_time_seconds: Optional[int] = None
    max_memory_mb: Optional[int] = None
    
    # Execution configuration
    validation_level: ValidationLevel = ValidationLevel.STRICT
    enable_optimization: bool = True
    enable_caching: bool = True
    
    def get_stage_count(self) -> int:
        """Get total number of stages in composition."""
        return len(self.stages)
    
    def get_estimated_duration(self) -> float:
        """Get estimated total execution time."""
        return sum(stage.estimated_duration_seconds for stage in self.stages)
    
    def get_estimated_memory(self) -> float:
        """Get estimated peak memory usage."""
        return max(stage.estimated_memory_mb for stage in self.stages) if self.stages else 0.0


# ============================================================================
# Validation Results and Error Handling
# ============================================================================

@dataclass
class ValidationResult:
    """Result of composition validation."""
    valid: bool
    category: str
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    auto_fixes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversionResult:
    """Result of type conversion operation."""
    success: bool
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class CompositionError(Exception):
    """Base exception for composition-related errors."""
    def __init__(self, message: str, error_type: str, stage_index: Optional[int] = None):
        super().__init__(message)
        self.error_type = error_type
        self.stage_index = stage_index
        self.timestamp = datetime.now()


@dataclass
class ErrorHandlingResult:
    """Result of error handling operation."""
    recovery_possible: bool
    recovery_action: Optional[str] = None
    modified_stage: Optional[CompositionStage] = None
    error_message: Optional[str] = None
    message: Optional[str] = None
    diagnostic_info: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Core Composition Validator
# ============================================================================

class CompositionValidator:
    """
    Multi-layer validation system for pipeline compositions.
    
    Implements the validation architecture from 32.4 design.
    """
    
    def __init__(self, tool_registry: 'ToolRegistry'):
        self.registry = tool_registry
        self.compatibility_rules = self._initialize_compatibility_rules()
        
    def validate_composition(self, composition: AnalysisComposition) -> ValidationResult:
        """
        Comprehensive composition validation with multiple layers.
        
        Validation Layers:
        1. Data Type Compatibility - Can outputs flow to inputs?
        2. Semantic Compatibility - Does the analytical flow make sense? 
        3. Performance Feasibility - Will this exceed resource limits?
        4. Dependency Resolution - Are all required tools available?
        """
        all_results = []
        
        # Layer 1: Data Type Validation
        all_results.append(self._validate_data_types(composition))
        
        # Layer 2: Semantic Validation
        all_results.append(self._validate_semantic_flow(composition))
        
        # Layer 3: Performance Validation  
        all_results.append(self._validate_performance_limits(composition))
        
        # Layer 4: Dependency Validation
        all_results.append(self._validate_dependencies(composition))
        
        return self._aggregate_validation_results(all_results)
    
    def _validate_data_types(self, composition: AnalysisComposition) -> ValidationResult:
        """Validate data type compatibility between pipeline stages."""
        errors = []
        warnings = []
        auto_fixes = []
        
        for i in range(len(composition.stages) - 1):
            current_stage = composition.stages[i]
            next_stage = composition.stages[i + 1]
            
            # Get type information from registry
            try:
                output_types = self.registry.get_output_types(current_stage.tool_name)
                input_types = self.registry.get_input_types(next_stage.tool_name)
            except KeyError as e:
                errors.append({
                    'stage_index': i,
                    'error_type': 'unknown_tool',
                    'tool_name': str(e),
                    'message': f"Tool not found in registry: {e}"
                })
                continue
            
            if not self._types_compatible(output_types, input_types):
                # Check for conversion path
                conversion_path = self._find_conversion_path(output_types, input_types)
                
                if conversion_path:
                    # Auto-fix possible
                    auto_fixes.append({
                        'stage_index': i + 1,
                        'fix_type': 'add_conversion',
                        'conversion_path': conversion_path,
                        'message': f"Added automatic conversion: {conversion_path}"
                    })
                    
                    # Apply fix to composition
                    next_stage.add_conversion(conversion_path)
                    
                else:
                    # Incompatible types - hard error
                    errors.append({
                        'stage_from': i,
                        'stage_to': i + 1,
                        'output_types': output_types,
                        'input_types': input_types,
                        'error_type': 'type_incompatible',
                        'message': f"No conversion path from {output_types} to {input_types}"
                    })
        
        return ValidationResult(
            valid=len(errors) == 0,
            category='data_types',
            errors=errors,
            warnings=warnings,
            auto_fixes=auto_fixes
        )
    
    def _validate_semantic_flow(self, composition: AnalysisComposition) -> ValidationResult:
        """Validate that the analytical flow makes semantic sense."""
        errors = []
        warnings = []
        
        # Check analytical workflow logic
        for i in range(len(composition.stages) - 1):
            current_stage = composition.stages[i]
            next_stage = composition.stages[i + 1]
            
            current_category = self.registry.get_tool_category(current_stage.tool_name)
            next_category = self.registry.get_tool_category(next_stage.tool_name)
            
            if not self._semantic_flow_valid(current_category, next_category):
                warnings.append({
                    'stage_from': i,
                    'stage_to': i + 1,
                    'current_category': current_category,
                    'next_category': next_category,
                    'warning_type': 'unusual_flow',
                    'message': f"Unusual analytical flow: {current_category} → {next_category}"
                })
        
        return ValidationResult(
            valid=True,  # Semantic issues are warnings, not errors
            category='semantic_flow',
            errors=errors,
            warnings=warnings
        )
    
    def _validate_performance_limits(self, composition: AnalysisComposition) -> ValidationResult:
        """Validate that composition meets performance constraints."""
        errors = []
        warnings = []
        
        # Check execution time limits
        if composition.max_execution_time_seconds:
            estimated_duration = composition.get_estimated_duration()
            if estimated_duration > composition.max_execution_time_seconds:
                errors.append({
                    'error_type': 'execution_time_exceeded',
                    'estimated_seconds': estimated_duration,
                    'limit_seconds': composition.max_execution_time_seconds,
                    'message': f"Estimated execution time ({estimated_duration}s) exceeds limit ({composition.max_execution_time_seconds}s)"
                })
        
        # Check memory limits
        if composition.max_memory_mb:
            estimated_memory = composition.get_estimated_memory()
            if estimated_memory > composition.max_memory_mb:
                errors.append({
                    'error_type': 'memory_limit_exceeded',
                    'estimated_mb': estimated_memory,
                    'limit_mb': composition.max_memory_mb,
                    'message': f"Estimated memory usage ({estimated_memory}MB) exceeds limit ({composition.max_memory_mb}MB)"
                })
        
        return ValidationResult(
            valid=len(errors) == 0,
            category='performance_limits',
            errors=errors,
            warnings=warnings
        )
    
    def _validate_dependencies(self, composition: AnalysisComposition) -> ValidationResult:
        """Validate that all required tools and dependencies are available."""
        errors = []
        warnings = []
        
        for i, stage in enumerate(composition.stages):
            # Check tool availability
            if not self.registry.is_tool_available(stage.tool_name):
                errors.append({
                    'stage_index': i,
                    'error_type': 'tool_unavailable',
                    'tool_name': stage.tool_name,
                    'message': f"Tool {stage.tool_name} is not available"
                })
                continue
            
            # Check required parameters
            required_params = self.registry.get_required_parameters(stage.tool_name, stage.function)
            missing_params = set(required_params) - set(stage.parameters.keys())
            
            if missing_params:
                errors.append({
                    'stage_index': i,
                    'error_type': 'missing_parameters',
                    'tool_name': stage.tool_name,
                    'missing_parameters': list(missing_params),
                    'message': f"Missing required parameters for {stage.tool_name}: {missing_params}"
                })
        
        return ValidationResult(
            valid=len(errors) == 0,
            category='dependencies',
            errors=errors,
            warnings=warnings
        )
    
    def _initialize_compatibility_rules(self) -> Dict:
        """Initialize type compatibility rules."""
        return {
            'pandas.DataFrame': ['sklearn.base.BaseEstimator', 'scipy.stats', 'numpy.ndarray'],
            'numpy.ndarray': ['sklearn.base.BaseEstimator', 'scipy.optimize', 'pandas.DataFrame'],
            'sklearn.base.BaseEstimator': ['sklearn.metrics', 'dict'],
            'dict': ['pandas.DataFrame', 'json.serializable']
        }
    
    def _types_compatible(self, output_types: List[str], input_types: List[str]) -> bool:
        """Check if output types are compatible with input types."""
        for output_type in output_types:
            if output_type in input_types:
                return True
            
            # Check compatibility rules
            if output_type in self.compatibility_rules:
                compatible_types = self.compatibility_rules[output_type]
                if any(compatible_type in input_types for compatible_type in compatible_types):
                    return True
        
        return False
    
    def _find_conversion_path(self, output_types: List[str], input_types: List[str]) -> Optional[str]:
        """Find automatic conversion path between incompatible types."""
        conversions = {
            ('pandas.DataFrame', 'numpy.ndarray'): 'dataframe_to_array',
            ('numpy.ndarray', 'pandas.DataFrame'): 'array_to_dataframe', 
            ('sklearn.base.BaseEstimator', 'dict'): 'model_to_dict',
            ('dict', 'pandas.DataFrame'): 'dict_to_dataframe'
        }
        
        for output_type in output_types:
            for input_type in input_types:
                conversion_key = (output_type, input_type)
                if conversion_key in conversions:
                    return conversions[conversion_key]
        
        return None
    
    def _semantic_flow_valid(self, current_category: str, next_category: str) -> bool:
        """Check if analytical flow between categories makes semantic sense."""
        valid_flows = {
            'data_ingestion': ['data_cleaning', 'exploratory_analysis', 'transformation'],
            'data_cleaning': ['exploratory_analysis', 'transformation', 'modeling'],
            'exploratory_analysis': ['transformation', 'modeling', 'visualization'],
            'transformation': ['modeling', 'analysis', 'visualization'],
            'modeling': ['evaluation', 'prediction', 'interpretation'],
            'evaluation': ['interpretation', 'model_selection', 'reporting']
        }
        
        return next_category in valid_flows.get(current_category, [next_category])
    
    def _aggregate_validation_results(self, results: List[ValidationResult]) -> ValidationResult:
        """Aggregate multiple validation results into a single result."""
        all_errors = []
        all_warnings = []
        all_auto_fixes = []
        
        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_auto_fixes.extend(result.auto_fixes)
        
        return ValidationResult(
            valid=len(all_errors) == 0,
            category='comprehensive',
            errors=all_errors,
            warnings=all_warnings,
            auto_fixes=all_auto_fixes
        )


# ============================================================================
# Tool Registry Protocol
# ============================================================================

class ToolRegistry(Protocol):
    """Protocol defining the interface for tool registries."""
    
    def get_output_types(self, tool_name: str) -> List[str]:
        """Get output types for a tool."""
        ...
    
    def get_input_types(self, tool_name: str) -> List[str]:
        """Get input types for a tool."""
        ...
    
    def get_tool_category(self, tool_name: str) -> str:
        """Get analytical category for a tool."""
        ...
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if tool is available."""
        ...
    
    def get_required_parameters(self, tool_name: str, function: str) -> List[str]:
        """Get required parameters for a tool function."""
        ...
    
    def find_tools_for_intent(self, intent_category: str, capabilities: List[str], data_context: Dict) -> List['Tool']:
        """Find tools matching analytical intent."""
        ...


# ============================================================================
# Composition Builder
# ============================================================================

class CompositionBuilder:
    """
    Builder pattern for creating analytical pipeline compositions.
    
    Provides fluent interface for LLM-friendly composition creation.
    """
    
    def __init__(self):
        self.stages: List[CompositionStage] = []
        self.composition_id = self._generate_composition_id()
        self.description = ""
        self.analytical_intent = ""
        
    def add_stage(
        self,
        tool_name: str,
        function: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> 'CompositionBuilder':
        """Add a stage to the composition."""
        stage = CompositionStage(
            tool_name=tool_name,
            function=function,
            parameters=parameters or {}
        )
        self.stages.append(stage)
        return self
    
    def with_description(self, description: str) -> 'CompositionBuilder':
        """Set composition description."""
        self.description = description
        return self
    
    def with_intent(self, intent: str) -> 'CompositionBuilder':
        """Set analytical intent."""
        self.analytical_intent = intent
        return self
    
    def with_performance_limits(
        self,
        max_execution_time_seconds: Optional[int] = None,
        max_memory_mb: Optional[int] = None
    ) -> 'CompositionBuilder':
        """Set performance constraints."""
        self._max_execution_time = max_execution_time_seconds
        self._max_memory_mb = max_memory_mb
        return self
    
    def build(self) -> AnalysisComposition:
        """Build the final composition."""
        composition = AnalysisComposition(
            composition_id=self.composition_id,
            stages=self.stages,
            description=self.description,
            analytical_intent=self.analytical_intent
        )
        
        # Apply performance limits if set
        if hasattr(self, '_max_execution_time'):
            composition.max_execution_time_seconds = self._max_execution_time
        if hasattr(self, '_max_memory_mb'):
            composition.max_memory_mb = self._max_memory_mb
        
        return composition
    
    def _generate_composition_id(self) -> str:
        """Generate unique composition ID."""
        import uuid
        return f"comp_{uuid.uuid4().hex[:8]}"


# ============================================================================
# Example Usage and Integration Points
# ============================================================================

def create_example_composition() -> AnalysisComposition:
    """
    Example: Create a customer churn analysis composition.
    
    Workflow: Data Query → Feature Engineering → Model Training → Evaluation
    """
    return (CompositionBuilder()
            .with_intent("Analyze customer churn with feature importance and model performance")
            .with_description("End-to-end churn analysis pipeline")
            .add_stage("query_database", "execute_query", {
                "sql": "SELECT * FROM customers WHERE created_at > '2024-01-01'",
                "chunk_size": 10000
            })
            .add_stage("feature_engineering", "create_features", {
                "target_column": "churned",
                "include_interactions": True
            })
            .add_stage("logistic_regression", "train_model", {
                "cross_validation": True,
                "feature_importance": True
            })
            .add_stage("model_evaluation", "evaluate_performance", {
                "metrics": ["auc", "precision", "recall", "f1"]
            })
            .with_performance_limits(max_execution_time_seconds=300, max_memory_mb=2048)
            .build())


if __name__ == "__main__":
    # Example usage demonstration
    example_composition = create_example_composition()
    
    print(f"Created composition: {example_composition.composition_id}")
    print(f"Stages: {example_composition.get_stage_count()}")
    print(f"Estimated duration: {example_composition.get_estimated_duration()}s")
    print(f"Intent: {example_composition.analytical_intent}")