"""
Pipeline Optimization Engine - Workflow Optimization and Error Recovery

This module implements the workflow optimization engine and error handling system
for pipeline compositions, providing intelligent caching, parallelization, and
graceful error recovery.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import hashlib
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from pipeline_composition_framework import (
    CompositionStage,
    AnalysisComposition,
    CompositionError,
    ErrorHandlingResult
)

logger = logging.getLogger(__name__)


# ============================================================================
# Error Handling System
# ============================================================================

class ErrorRecoveryStrategy(Enum):
    """Available error recovery strategies."""
    RETRY_WITH_DIFFERENT_PARAMETERS = "retry_different_params"
    SWITCH_TO_STREAMING = "switch_streaming"  
    REDUCE_CHUNK_SIZE = "reduce_chunk_size"
    SUBSTITUTE_TOOL = "substitute_tool"
    SKIP_STAGE = "skip_stage"
    FAIL_FAST = "fail_fast"


@dataclass
class AlternativeTool:
    """Alternative tool for stage substitution."""
    name: str
    function: str
    default_parameters: Dict[str, Any]
    compatibility_score: float
    performance_factor: float  # Relative performance vs original tool


class CompositionErrorHandler:
    """
    Handles errors and implements graceful degradation for pipeline compositions.
    
    Error Recovery Strategies:
    1. Retry with different parameters
    2. Skip failed stage and continue with partial results  
    3. Substitute failed stage with alternative tool
    4. Switch to streaming execution for memory errors
    5. Provide diagnostic information for manual intervention
    """
    
    def __init__(self, tool_registry: 'ToolRegistry'):
        self.tool_registry = tool_registry
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.alternative_tools_cache = {}
        
    def handle_composition_error(self, error: CompositionError, 
                                context: 'CompositionContext',
                                stage: CompositionStage) -> ErrorHandlingResult:
        """
        Handle composition errors with appropriate recovery strategy.
        
        Args:
            error: The composition error that occurred
            context: Current composition execution context
            stage: The stage where the error occurred
            
        Returns:
            ErrorHandlingResult with recovery information
        """
        error_type = self._classify_error(error)
        
        logger.warning(f"Handling {error_type} error at stage {stage.tool_name}: {str(error)}")
        
        if error_type in self.recovery_strategies:
            recovery_func = self.recovery_strategies[error_type]
            return recovery_func(error, context, stage)
        else:
            return ErrorHandlingResult(
                recovery_possible=False,
                error_message=f"Unhandled error type: {error_type}",
                diagnostic_info=self._generate_diagnostic_info(error, context, stage)
            )
    
    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize error recovery strategy functions."""
        return {
            'memory_error': self._handle_memory_error,
            'type_conversion_error': self._handle_conversion_error, 
            'tool_execution_error': self._handle_tool_error,
            'dependency_missing': self._handle_missing_dependency,
            'parameter_invalid': self._handle_parameter_error,
            'timeout_error': self._handle_timeout_error
        }
    
    def _classify_error(self, error: CompositionError) -> str:
        """Classify error type for appropriate recovery strategy."""
        error_msg = str(error).lower()
        
        if 'memory' in error_msg or 'out of memory' in error_msg:
            return 'memory_error'
        elif 'conversion' in error_msg or 'type' in error_msg:
            return 'type_conversion_error'
        elif 'timeout' in error_msg or 'time limit' in error_msg:
            return 'timeout_error'
        elif 'parameter' in error_msg or 'argument' in error_msg:
            return 'parameter_invalid'
        elif 'not found' in error_msg or 'missing' in error_msg:
            return 'dependency_missing'
        else:
            return 'tool_execution_error'
    
    def _handle_memory_error(self, error: CompositionError, context, stage: CompositionStage) -> ErrorHandlingResult:
        """Handle memory errors by switching to streaming or reducing data size."""
        
        # Strategy 1: Switch to streaming execution
        if not stage.is_streaming:
            modified_stage = stage.copy()
            modified_stage.enable_streaming()
            modified_stage.chunk_size = 1000  # Conservative chunk size
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.SWITCH_TO_STREAMING.value,
                modified_stage=modified_stage,
                message="Switched to streaming execution to reduce memory usage"
            )
        
        # Strategy 2: Reduce chunk size if already streaming
        if stage.is_streaming and (stage.chunk_size or 1000) > 100:
            modified_stage = stage.copy()
            current_chunk_size = stage.chunk_size or 1000
            modified_stage.chunk_size = max(100, current_chunk_size // 2)
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.REDUCE_CHUNK_SIZE.value,
                modified_stage=modified_stage,
                message=f"Reduced chunk size to {modified_stage.chunk_size}"
            )
        
        # Strategy 3: Skip stage if optional
        if stage.is_optional:
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.SKIP_STAGE.value,
                message=f"Skipped optional stage {stage.tool_name} due to memory constraints"
            )
        
        return ErrorHandlingResult(
            recovery_possible=False,
            error_message="Memory error: unable to recover with current data size",
            diagnostic_info={
                'suggested_actions': [
                    'Reduce input data size',
                    'Increase available memory',
                    'Use sampling for analysis'
                ]
            }
        )
    
    def _handle_tool_error(self, error: CompositionError, context, stage: CompositionStage) -> ErrorHandlingResult:
        """Handle tool execution errors by trying alternative tools or parameters."""
        
        # Strategy 1: Try alternative tool for same function
        alternatives = self._find_alternative_tools(stage.tool_name, stage.function)
        
        if alternatives:
            best_alternative = alternatives[0]  # Highest compatibility score
            modified_stage = stage.copy()
            modified_stage.tool_name = best_alternative.name
            modified_stage.function = best_alternative.function
            modified_stage.parameters.update(best_alternative.default_parameters)
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.SUBSTITUTE_TOOL.value,
                modified_stage=modified_stage,
                message=f"Substituted {stage.tool_name} with {best_alternative.name}"
            )
        
        # Strategy 2: Retry with conservative parameters
        if stage.parameters:
            modified_stage = stage.copy()
            conservative_params = self._get_conservative_parameters(stage.tool_name, stage.parameters)
            modified_stage.parameters = conservative_params
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.RETRY_WITH_DIFFERENT_PARAMETERS.value,
                modified_stage=modified_stage,
                message="Retrying with more conservative parameters"
            )
        
        # Strategy 3: Skip stage if optional
        if stage.is_optional:
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.SKIP_STAGE.value,
                message=f"Skipped optional stage {stage.tool_name} due to execution failure"
            )
        
        return ErrorHandlingResult(
            recovery_possible=False,
            error_message=f"Critical stage {stage.tool_name} failed with no recovery options"
        )
    
    def _handle_conversion_error(self, error: CompositionError, context, stage: CompositionStage) -> ErrorHandlingResult:
        """Handle type conversion errors."""
        # Try simplified conversion parameters
        if stage.requires_conversion:
            modified_stage = stage.copy()
            # Add fallback conversion parameters
            modified_stage.parameters['preserve_categories'] = False
            modified_stage.parameters['force_conversion'] = True
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.RETRY_WITH_DIFFERENT_PARAMETERS.value,
                modified_stage=modified_stage,
                message="Retrying conversion with simplified parameters"
            )
        
        return ErrorHandlingResult(
            recovery_possible=False,
            error_message="Type conversion failed with no recovery options"
        )
    
    def _handle_parameter_error(self, error: CompositionError, context, stage: CompositionStage) -> ErrorHandlingResult:
        """Handle parameter validation errors."""
        # Get default parameters for the tool
        try:
            default_params = self.tool_registry.get_default_parameters(stage.tool_name, stage.function)
            
            modified_stage = stage.copy()
            modified_stage.parameters = default_params
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.RETRY_WITH_DIFFERENT_PARAMETERS.value,
                modified_stage=modified_stage,
                message="Reset to default parameters"
            )
        
        except Exception:
            return ErrorHandlingResult(
                recovery_possible=False,
                error_message="Parameter error with no default parameters available"
            )
    
    def _handle_timeout_error(self, error: CompositionError, context, stage: CompositionStage) -> ErrorHandlingResult:
        """Handle timeout errors."""
        # Switch to streaming if not already
        if not stage.is_streaming:
            modified_stage = stage.copy()
            modified_stage.enable_streaming()
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.SWITCH_TO_STREAMING.value,
                modified_stage=modified_stage,
                message="Switched to streaming to handle timeout"
            )
        
        return ErrorHandlingResult(
            recovery_possible=False,
            error_message="Timeout error with no recovery options"
        )
    
    def _handle_missing_dependency(self, error: CompositionError, context, stage: CompositionStage) -> ErrorHandlingResult:
        """Handle missing dependency errors."""
        # Try alternative tools
        alternatives = self._find_alternative_tools(stage.tool_name, stage.function)
        
        if alternatives:
            best_alternative = alternatives[0]
            modified_stage = stage.copy()
            modified_stage.tool_name = best_alternative.name
            
            return ErrorHandlingResult(
                recovery_possible=True,
                recovery_action=ErrorRecoveryStrategy.SUBSTITUTE_TOOL.value,
                modified_stage=modified_stage,
                message=f"Substituted unavailable tool {stage.tool_name} with {best_alternative.name}"
            )
        
        return ErrorHandlingResult(
            recovery_possible=False,
            error_message=f"Tool {stage.tool_name} unavailable with no alternatives"
        )
    
    def _find_alternative_tools(self, tool_name: str, function: str) -> List[AlternativeTool]:
        """Find alternative tools for a given tool and function."""
        cache_key = f"{tool_name}:{function}"
        
        if cache_key in self.alternative_tools_cache:
            return self.alternative_tools_cache[cache_key]
        
        alternatives = []
        
        # Example alternative mappings - would be populated from tool registry
        alternative_mappings = {
            'sklearn.LinearRegression': [
                AlternativeTool('sklearn.Ridge', 'train_model', {'alpha': 1.0}, 0.9, 1.1),
                AlternativeTool('sklearn.ElasticNet', 'train_model', {'alpha': 1.0, 'l1_ratio': 0.5}, 0.8, 1.2)
            ],
            'pandas.DataFrame.groupby': [
                AlternativeTool('numpy.aggregate', 'group_aggregate', {}, 0.7, 0.8)
            ]
        }
        
        if tool_name in alternative_mappings:
            alternatives = alternative_mappings[tool_name]
        
        # Sort by compatibility score
        alternatives.sort(key=lambda x: x.compatibility_score, reverse=True)
        
        self.alternative_tools_cache[cache_key] = alternatives
        return alternatives
    
    def _get_conservative_parameters(self, tool_name: str, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get more conservative parameters for a tool."""
        conservative_params = current_params.copy()
        
        # Common conservative adjustments
        if 'max_iter' in conservative_params:
            conservative_params['max_iter'] = min(conservative_params['max_iter'], 100)
        
        if 'n_estimators' in conservative_params:
            conservative_params['n_estimators'] = min(conservative_params['n_estimators'], 50)
        
        if 'learning_rate' in conservative_params:
            conservative_params['learning_rate'] = max(conservative_params['learning_rate'], 0.01)
        
        return conservative_params
    
    def _generate_diagnostic_info(self, error: CompositionError, context, stage: CompositionStage) -> Dict[str, Any]:
        """Generate diagnostic information for error analysis."""
        return {
            'error_type': error.error_type,
            'stage_index': error.stage_index,
            'tool_name': stage.tool_name,
            'function': stage.function,
            'parameters': stage.parameters,
            'timestamp': error.timestamp.isoformat(),
            'context_stage_count': len(context.stage_outputs) if hasattr(context, 'stage_outputs') else 0,
            'suggested_manual_actions': [
                'Check tool documentation for parameter requirements',
                'Validate input data format and size',
                'Ensure all required dependencies are installed',
                'Consider data preprocessing or sampling'
            ]
        }


# ============================================================================
# Workflow Optimization Engine  
# ============================================================================

@dataclass
class ParallelGroup:
    """Group of stages that can be executed in parallel."""
    stages: List[int]  # Stage indices
    max_concurrency: int
    synchronization_point: bool = True


@dataclass
class CacheEntry:
    """Cache entry for stage results."""
    key: str
    data: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    expiry: datetime
    size_mb: float


class ExpirationPolicy(Enum):
    """Cache expiration policies."""
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based" 
    USAGE_BASED = "usage_based"


@dataclass
class OptimizedComposition:
    """Optimized version of a composition."""
    stages: List[CompositionStage]
    parallel_groups: List[ParallelGroup]
    caching_strategy: Dict[int, str]  # Stage index -> cache policy
    optimization_log: List[str]
    estimated_performance_improvement: float


class WorkflowOptimizationEngine:
    """
    Optimizes pipeline compositions for performance through:
    1. Intelligent caching of intermediate results
    2. Parallel execution of independent stages
    3. Query optimization for database-heavy pipelines
    4. Memory-efficient data flow patterns
    """
    
    def __init__(self, cache_storage: 'CacheStorage', max_parallel_stages: int = 4):
        self.cache = CompositionCache(cache_storage)
        self.scheduler = ParallelScheduler(max_parallel_stages)
        self.optimization_rules = self._initialize_optimization_rules()
        
    def optimize_composition(self, composition: AnalysisComposition) -> OptimizedComposition:
        """
        Apply optimization strategies to improve composition performance.
        
        Args:
            composition: Original composition to optimize
            
        Returns:
            OptimizedComposition with performance improvements
        """
        optimization_log = []
        
        # Phase 1: Dependency Analysis
        dependency_graph = self._build_dependency_graph(composition)
        
        # Phase 2: Identify Optimization Opportunities  
        parallelizable_groups = self._identify_parallel_groups(dependency_graph)
        cacheable_stages = self._identify_cacheable_stages(composition)
        
        # Phase 3: Apply Optimization Rules
        optimized_stages = composition.stages.copy()
        
        for rule in self.optimization_rules:
            rule_result = rule.apply(composition, {
                'dependency_graph': dependency_graph,
                'parallel_groups': parallelizable_groups,
                'cacheable_stages': cacheable_stages
            })
            
            if rule_result.applied:
                optimized_stages = rule_result.optimized_stages
                optimization_log.append(rule_result.description)
        
        # Phase 4: Estimate Performance Improvement
        performance_improvement = self._estimate_performance_gain(
            composition.stages, optimized_stages, parallelizable_groups
        )
        
        return OptimizedComposition(
            stages=optimized_stages,
            parallel_groups=parallelizable_groups,
            caching_strategy=cacheable_stages,
            optimization_log=optimization_log,
            estimated_performance_improvement=performance_improvement
        )
    
    def _build_dependency_graph(self, composition: AnalysisComposition) -> Dict[int, Set[int]]:
        """Build dependency graph showing which stages depend on others."""
        dependencies = {}
        
        for i, stage in enumerate(composition.stages):
            dependencies[i] = set()
            
            # Sequential dependency - each stage depends on previous
            if i > 0:
                dependencies[i].add(i - 1)
            
            # TODO: Add more sophisticated dependency analysis
            # - Data dependency analysis
            # - Resource dependency analysis  
            # - Semantic dependency analysis
        
        return dependencies
    
    def _identify_parallel_groups(self, dependency_graph: Dict[int, Set[int]]) -> List[ParallelGroup]:
        """
        Identify stages that can be executed in parallel.
        
        Example:
        Query → [Feature Engineering, Statistical Analysis] → Modeling
        
        Feature Engineering and Statistical Analysis can run in parallel
        since they don't depend on each other.
        """
        parallel_groups = []
        levels = self._topological_levels(dependency_graph)
        
        for level in levels:
            if len(level) > 1:
                # Multiple independent stages at same level
                parallel_groups.append(ParallelGroup(
                    stages=level,
                    max_concurrency=min(len(level), 4),  # Limit concurrent stages
                    synchronization_point=True
                ))
        
        return parallel_groups
    
    def _topological_levels(self, dependency_graph: Dict[int, Set[int]]) -> List[List[int]]:
        """Get topological levels of dependency graph."""
        levels = []
        remaining = set(dependency_graph.keys())
        
        while remaining:
            # Find nodes with no remaining dependencies
            current_level = []
            for node in remaining:
                dependencies = dependency_graph[node]
                if not dependencies.intersection(remaining):
                    current_level.append(node)
            
            if not current_level:
                # Circular dependency - break with remaining nodes
                current_level = list(remaining)
            
            levels.append(current_level)
            remaining.difference_update(current_level)
        
        return levels
    
    def _identify_cacheable_stages(self, composition: AnalysisComposition) -> Dict[int, str]:
        """Identify stages that should be cached and their policies."""
        cacheable_stages = {}
        
        for i, stage in enumerate(composition.stages):
            if self._should_cache_stage(stage):
                cache_policy = self._determine_cache_policy(stage)
                cacheable_stages[i] = cache_policy
        
        return cacheable_stages
    
    def _should_cache_stage(self, stage: CompositionStage) -> bool:
        """Determine if a stage should be cached."""
        # Cache expensive operations
        expensive_operations = {
            'machine_learning_training',
            'large_dataset_aggregation', 
            'complex_statistical_analysis',
            'optimization_algorithms'
        }
        
        if hasattr(stage, 'operation_category') and stage.operation_category in expensive_operations:
            return True
        
        # Cache long-running operations (> 10 seconds estimated)
        if stage.estimated_duration_seconds > 10:
            return True
        
        # Cache deterministic operations with stable inputs
        if stage.is_deterministic and not stage.has_time_dependency:
            return True
        
        return False
    
    def _determine_cache_policy(self, stage: CompositionStage) -> str:
        """Determine appropriate cache policy for a stage."""
        if stage.has_time_dependency:
            return "short_term"  # 1 hour
        elif stage.estimated_duration_seconds > 60:
            return "long_term"  # 24 hours
        else:
            return "medium_term"  # 6 hours
    
    def _initialize_optimization_rules(self) -> List['OptimizationRule']:
        """Initialize optimization rules."""
        return [
            DatabaseQueryOptimizationRule(),
            StreamingOptimizationRule(),
            CacheOptimizationRule()
        ]
    
    def _estimate_performance_gain(self, original_stages: List[CompositionStage],
                                  optimized_stages: List[CompositionStage],
                                  parallel_groups: List[ParallelGroup]) -> float:
        """Estimate performance improvement from optimizations."""
        original_duration = sum(stage.estimated_duration_seconds for stage in original_stages)
        
        # Calculate parallel execution savings
        parallel_savings = 0.0
        for group in parallel_groups:
            group_stages = [optimized_stages[i] for i in group.stages]
            sequential_duration = sum(stage.estimated_duration_seconds for stage in group_stages)
            parallel_duration = max(stage.estimated_duration_seconds for stage in group_stages)
            parallel_savings += sequential_duration - parallel_duration
        
        # Calculate caching savings (estimated 50% reduction for cached stages)
        cache_savings = 0.0
        for stage in optimized_stages:
            if stage.cacheable:
                cache_savings += stage.estimated_duration_seconds * 0.5
        
        total_savings = parallel_savings + cache_savings
        return (total_savings / original_duration) * 100 if original_duration > 0 else 0.0


# ============================================================================
# Caching System
# ============================================================================

class CompositionCache:
    """
    Intelligent caching system for pipeline composition intermediate results.
    """
    
    def __init__(self, storage: 'CacheStorage'):
        self.storage = storage
        self.cache_policies = {
            'short_term': timedelta(hours=1),
            'medium_term': timedelta(hours=6),
            'long_term': timedelta(hours=24),
            'persistent': timedelta(days=7)
        }
        self.max_cache_size_mb = 1024  # 1GB cache limit
    
    def get_cache_key(self, stage: CompositionStage, input_data_hash: str) -> str:
        """Generate cache key based on stage configuration and input data."""
        stage_signature = {
            'tool_name': stage.tool_name,
            'function': stage.function,
            'parameters': stage.parameters,
            'version': getattr(stage, 'tool_version', '1.0')
        }
        
        signature_str = json.dumps(stage_signature, sort_keys=True)
        signature_hash = hashlib.sha256(signature_str.encode()).hexdigest()
        
        return f"comp:{stage.tool_name}:{signature_hash}:{input_data_hash}"
    
    def get_cached_result(self, cache_key: str) -> Optional[CacheEntry]:
        """Retrieve cached result if available and not expired."""
        entry = self.storage.get(cache_key)
        
        if entry and datetime.now() < entry.expiry:
            return entry
        
        # Remove expired entry
        if entry:
            self.storage.delete(cache_key)
        
        return None
    
    def cache_result(self, cache_key: str, data: Any, metadata: Dict[str, Any], 
                    policy: str = 'medium_term'):
        """Cache stage result with specified policy."""
        expiry = datetime.now() + self.cache_policies[policy]
        size_mb = self._estimate_size_mb(data)
        
        entry = CacheEntry(
            key=cache_key,
            data=data,
            metadata=metadata,
            timestamp=datetime.now(),
            expiry=expiry,
            size_mb=size_mb
        )
        
        # Check cache size limits
        if self._should_cache_entry(entry):
            self.storage.put(cache_key, entry)
    
    def _estimate_size_mb(self, data: Any) -> float:
        """Estimate data size in MB."""
        try:
            import sys
            size_bytes = sys.getsizeof(data)
            
            # For pandas DataFrames, get memory usage
            if hasattr(data, 'memory_usage'):
                size_bytes = data.memory_usage(deep=True).sum()
            
            return size_bytes / (1024 * 1024)
        except Exception:
            return 1.0  # Default estimate
    
    def _should_cache_entry(self, entry: CacheEntry) -> bool:
        """Determine if entry should be cached based on size and policies."""
        current_size = self.storage.get_total_size_mb()
        
        if current_size + entry.size_mb > self.max_cache_size_mb:
            # Try to make room by evicting old entries
            self._evict_old_entries(entry.size_mb)
            current_size = self.storage.get_total_size_mb()
        
        return current_size + entry.size_mb <= self.max_cache_size_mb
    
    def _evict_old_entries(self, space_needed_mb: float):
        """Evict old cache entries to make space."""
        # LRU eviction policy
        entries = self.storage.get_all_entries()
        entries.sort(key=lambda x: x.timestamp)  # Oldest first
        
        space_freed = 0.0
        for entry in entries:
            if space_freed >= space_needed_mb:
                break
            
            self.storage.delete(entry.key)
            space_freed += entry.size_mb


# ============================================================================
# Protocol Definitions
# ============================================================================

class CacheStorage(ABC):
    """Abstract interface for cache storage backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        pass
    
    @abstractmethod
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Store cache entry."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        pass
    
    @abstractmethod
    def get_total_size_mb(self) -> float:
        """Get total cache size in MB."""
        pass
    
    @abstractmethod
    def get_all_entries(self) -> List[CacheEntry]:
        """Get all cache entries."""
        pass


class ParallelScheduler:
    """Scheduler for parallel stage execution."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute_parallel_group(self, group: ParallelGroup, stage_executor_func: Callable) -> Dict[int, Any]:
        """Execute a group of stages in parallel."""
        futures = {}
        
        for stage_index in group.stages:
            future = self.executor.submit(stage_executor_func, stage_index)
            futures[stage_index] = future
        
        # Wait for all stages to complete
        results = {}
        for stage_index, future in futures.items():
            try:
                results[stage_index] = future.result()
            except Exception as e:
                logger.error(f"Parallel stage {stage_index} failed: {str(e)}")
                results[stage_index] = CompositionError(str(e), 'parallel_execution_error', stage_index)
        
        return results


# ============================================================================  
# Optimization Rules
# ============================================================================

class OptimizationRule(ABC):
    """Base class for optimization rules."""
    
    @abstractmethod
    def apply(self, composition: AnalysisComposition, context: Dict[str, Any]) -> 'OptimizationResult':
        """Apply optimization rule to composition."""
        pass


@dataclass
class OptimizationResult:
    """Result of applying an optimization rule."""
    applied: bool
    optimized_stages: List[CompositionStage]
    description: str


class DatabaseQueryOptimizationRule(OptimizationRule):
    """Optimize database query stages."""
    
    def apply(self, composition: AnalysisComposition, context: Dict[str, Any]) -> OptimizationResult:
        """Apply database query optimizations."""
        optimized_stages = []
        applied = False
        
        for stage in composition.stages:
            if stage.tool_name.startswith('query_') or 'database' in stage.tool_name:
                # Add query optimization hints
                optimized_stage = stage.copy()
                if 'sql' in stage.parameters:
                    # Add streaming hints for large queries
                    optimized_stage.parameters['streaming'] = True
                    optimized_stage.parameters['chunk_size'] = 10000
                    applied = True
                
                optimized_stages.append(optimized_stage)
            else:
                optimized_stages.append(stage)
        
        return OptimizationResult(
            applied=applied,
            optimized_stages=optimized_stages,
            description="Applied database query optimizations with streaming"
        )


class StreamingOptimizationRule(OptimizationRule):
    """Enable streaming for memory-intensive stages."""
    
    def apply(self, composition: AnalysisComposition, context: Dict[str, Any]) -> OptimizationResult:
        """Apply streaming optimizations."""
        optimized_stages = []
        applied = False
        
        for stage in composition.stages:
            if stage.estimated_memory_mb > 50 and not stage.is_streaming:
                optimized_stage = stage.copy()
                optimized_stage.enable_streaming()
                optimized_stage.chunk_size = 1000
                applied = True
                optimized_stages.append(optimized_stage)
            else:
                optimized_stages.append(stage)
        
        return OptimizationResult(
            applied=applied,
            optimized_stages=optimized_stages,
            description="Enabled streaming for memory-intensive stages"
        )


class CacheOptimizationRule(OptimizationRule):
    """Enable caching for expensive stages."""
    
    def apply(self, composition: AnalysisComposition, context: Dict[str, Any]) -> OptimizationResult:
        """Apply caching optimizations."""
        optimized_stages = []
        applied = False
        
        for stage in composition.stages:
            if stage.estimated_duration_seconds > 10 and not stage.cacheable:
                optimized_stage = stage.copy()
                optimized_stage.cacheable = True
                optimized_stage.cache_ttl_hours = 6  # 6 hour cache
                applied = True
                optimized_stages.append(optimized_stage)
            else:
                optimized_stages.append(stage)
        
        return OptimizationResult(
            applied=applied,
            optimized_stages=optimized_stages,
            description="Enabled caching for expensive stages"
        )


if __name__ == "__main__":
    # Example usage demonstration
    print("Pipeline Optimization Engine initialized")
    print("Available error recovery strategies:", [strategy.value for strategy in ErrorRecoveryStrategy])
    print("Available optimization rules: DatabaseQuery, Streaming, Cache")