"""
Automatic Shim Insertion Logic for LocalData MCP v2.0 Integration Shims Framework.

This module provides intelligent pipeline analysis and automatic shim insertion capabilities
for seamless cross-domain data science workflows with minimal user intervention.

Key Features:
- PipelineAnalyzer: Identify incompatible connections in pipeline chains
- ShimInjector: Automatic adapter insertion with optimal selection
- PipelineValidator: Complete pipeline composition verification
- Cost-based optimization for efficient shim selection
- Integration with existing compatibility matrix and shim registry

Design Principles:
- Intention-Driven Interface: Analyze pipelines by analytical goals
- Context-Aware Composition: Consider upstream/downstream context
- Progressive Disclosure: Simple analysis with detailed breakdowns available
- Streaming-First: Memory-efficient for large pipeline chains
- Modular Domain Integration: Seamless integration with existing infrastructure
"""

import logging
import time
import heapq
import networkx as nx
from typing import Any, Dict, List, Optional, Set, Tuple, Union, DefaultDict
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache

from .interfaces import (
    DataFormat, ConversionRequest, ConversionResult, ConversionPath,
    ConversionStep, ConversionCost, ValidationResult, ShimAdapter
)
from .compatibility_matrix import PipelineCompatibilityMatrix, CompatibilityLevel
from .shim_registry import ShimRegistry, EnhancedShimAdapter
from ...logging_manager import get_logger

logger = get_logger(__name__)


class AnalysisType(Enum):
    """Types of pipeline analysis that can be performed."""
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COST = "cost"
    COMPLETE = "complete"


class InjectionStrategy(Enum):
    """Strategies for shim injection."""
    MINIMAL = "minimal"          # Insert minimum necessary shims
    OPTIMAL = "optimal"          # Insert shims for best performance
    SAFE = "safe"               # Insert shims for maximum compatibility
    BALANCED = "balanced"       # Balance between performance and safety


@dataclass
class PipelineStep:
    """Representation of a single step in a pipeline."""
    step_id: str
    domain: str
    operation: str
    input_format: DataFormat
    output_format: DataFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConnection:
    """Connection between two pipeline steps."""
    source_step: PipelineStep
    target_step: PipelineStep
    data_flow: Dict[str, Any] = field(default_factory=dict)
    compatibility_score: float = 0.0
    requires_conversion: bool = True
    conversion_path: Optional[ConversionPath] = None


@dataclass
class IncompatibilityIssue:
    """Identified incompatibility in pipeline connection."""
    connection: PipelineConnection
    issue_type: str
    severity: str  # 'critical', 'warning', 'info'
    description: str
    suggested_solutions: List[str] = field(default_factory=list)
    cost_estimate: Optional[ConversionCost] = None


@dataclass
class ShimRecommendation:
    """Recommendation for shim insertion."""
    connection: PipelineConnection
    recommended_shim: EnhancedShimAdapter
    insertion_point: str  # 'before_target', 'after_source', 'intermediate'
    confidence: float
    expected_benefit: str
    cost_estimate: ConversionCost
    alternative_shims: List[Tuple[EnhancedShimAdapter, float]] = field(default_factory=list)


@dataclass 
class PipelineAnalysisResult:
    """Comprehensive result of pipeline analysis."""
    pipeline_id: str
    analysis_type: AnalysisType
    is_compatible: bool
    compatibility_score: float
    total_steps: int
    incompatible_connections: List[PipelineConnection]
    identified_issues: List[IncompatibilityIssue]
    shim_recommendations: List[ShimRecommendation]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: float = field(default_factory=time.time)
    execution_time: float = 0.0


@dataclass
class OptimizationCriteria:
    """Criteria for shim selection optimization."""
    prioritize_performance: bool = True
    prioritize_quality: bool = True
    prioritize_memory: bool = False
    max_cost_threshold: Optional[float] = None
    max_execution_time: Optional[float] = None
    quality_threshold: float = 0.8
    performance_weight: float = 0.4
    quality_weight: float = 0.4
    cost_weight: float = 0.2


class PipelineAnalyzer:
    """
    Core pipeline analyzer for identifying incompatible connections and data flow issues.
    
    Analyzes pipeline chains to identify format incompatibilities, performance bottlenecks,
    and optimization opportunities with detailed reporting and recommendations.
    """
    
    def __init__(self,
                 compatibility_matrix: PipelineCompatibilityMatrix,
                 shim_registry: ShimRegistry,
                 enable_caching: bool = True,
                 max_analysis_threads: int = 4):
        """
        Initialize PipelineAnalyzer.
        
        Args:
            compatibility_matrix: Matrix for format compatibility assessment
            shim_registry: Registry of available shim adapters
            enable_caching: Enable caching of analysis results
            max_analysis_threads: Maximum threads for parallel analysis
        """
        self.compatibility_matrix = compatibility_matrix
        self.shim_registry = shim_registry
        self.enable_caching = enable_caching
        self.max_analysis_threads = max_analysis_threads
        
        # Analysis cache
        if enable_caching:
            self._analysis_cache: Dict[str, PipelineAnalysisResult] = {}
            self._cache_lock = threading.RLock()
        
        # Thread pool for parallel analysis
        self._executor = ThreadPoolExecutor(max_workers=max_analysis_threads,
                                          thread_name_prefix="pipeline_analyzer")
        
        # Analysis statistics
        self._stats = {
            'total_analyses': 0,
            'cache_hits': 0,
            'issues_identified': 0,
            'shims_recommended': 0
        }
        
        logger.info("PipelineAnalyzer initialized",
                   caching_enabled=enable_caching,
                   max_threads=max_analysis_threads)
    
    def analyze_pipeline(self,
                        pipeline_steps: List[PipelineStep],
                        analysis_type: AnalysisType = AnalysisType.COMPLETE,
                        pipeline_id: Optional[str] = None) -> PipelineAnalysisResult:
        """
        Analyze pipeline for compatibility issues and optimization opportunities.
        
        Args:
            pipeline_steps: List of pipeline steps to analyze
            analysis_type: Type of analysis to perform
            pipeline_id: Optional pipeline identifier for caching
            
        Returns:
            Comprehensive analysis result with issues and recommendations
        """
        start_time = time.time()
        
        if pipeline_id is None:
            pipeline_id = f"pipeline_{int(time.time() * 1000)}"
        
        # Check cache
        cache_key = self._generate_cache_key(pipeline_steps, analysis_type, pipeline_id)
        if self.enable_caching:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self._stats['cache_hits'] += 1
                return cached_result
        
        self._stats['total_analyses'] += 1
        
        logger.info(f"Starting pipeline analysis",
                   pipeline_id=pipeline_id,
                   steps_count=len(pipeline_steps),
                   analysis_type=analysis_type.value)
        
        try:
            # Build pipeline graph
            connections = self._build_pipeline_connections(pipeline_steps)
            
            # Analyze connections based on type
            incompatible_connections = []
            identified_issues = []
            shim_recommendations = []
            
            if analysis_type in [AnalysisType.COMPATIBILITY, AnalysisType.COMPLETE]:
                incompatible_connections, issues = self._analyze_compatibility(connections)
                identified_issues.extend(issues)
            
            if analysis_type in [AnalysisType.COMPLETE]:
                performance_issues = self._analyze_performance(connections)
                identified_issues.extend(performance_issues)
                
                quality_issues = self._analyze_quality(connections)
                identified_issues.extend(quality_issues)
                
                cost_issues = self._analyze_cost(connections)
                identified_issues.extend(cost_issues)
            
            # Generate shim recommendations for incompatible connections
            if incompatible_connections:
                shim_recommendations = self._generate_shim_recommendations(incompatible_connections)
            
            # Calculate overall compatibility score
            compatibility_score = self._calculate_overall_compatibility_score(connections)
            
            # Create analysis result
            execution_time = time.time() - start_time
            result = PipelineAnalysisResult(
                pipeline_id=pipeline_id,
                analysis_type=analysis_type,
                is_compatible=len(incompatible_connections) == 0,
                compatibility_score=compatibility_score,
                total_steps=len(pipeline_steps),
                incompatible_connections=incompatible_connections,
                identified_issues=identified_issues,
                shim_recommendations=shim_recommendations,
                performance_metrics={
                    'total_connections': len(connections),
                    'analysis_threads_used': min(len(connections), self.max_analysis_threads),
                    'cache_key': cache_key
                },
                execution_time=execution_time
            )
            
            # Cache result
            if self.enable_caching:
                self._cache_result(cache_key, result)
            
            # Update statistics
            self._stats['issues_identified'] += len(identified_issues)
            self._stats['shims_recommended'] += len(shim_recommendations)
            
            logger.info(f"Pipeline analysis completed",
                       pipeline_id=pipeline_id,
                       is_compatible=result.is_compatible,
                       issues_count=len(identified_issues),
                       recommendations_count=len(shim_recommendations),
                       execution_time=execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline analysis failed: {e}", pipeline_id=pipeline_id)
            # Return error result
            return PipelineAnalysisResult(
                pipeline_id=pipeline_id,
                analysis_type=analysis_type,
                is_compatible=False,
                compatibility_score=0.0,
                total_steps=len(pipeline_steps),
                incompatible_connections=[],
                identified_issues=[
                    IncompatibilityIssue(
                        connection=None,
                        issue_type="analysis_error",
                        severity="critical",
                        description=f"Analysis failed: {str(e)}"
                    )
                ],
                shim_recommendations=[],
                execution_time=time.time() - start_time
            )
    
    def _build_pipeline_connections(self, pipeline_steps: List[PipelineStep]) -> List[PipelineConnection]:
        """Build connections between pipeline steps."""
        connections = []
        
        for i in range(len(pipeline_steps) - 1):
            source_step = pipeline_steps[i]
            target_step = pipeline_steps[i + 1]
            
            # Calculate compatibility between steps
            compatibility = self.compatibility_matrix.get_compatibility(
                source_step.output_format,
                target_step.input_format
            )
            
            connection = PipelineConnection(
                source_step=source_step,
                target_step=target_step,
                compatibility_score=compatibility.score,
                requires_conversion=compatibility.conversion_required,
                conversion_path=compatibility.conversion_path
            )
            
            connections.append(connection)
        
        return connections
    
    def _analyze_compatibility(self, connections: List[PipelineConnection]) -> Tuple[List[PipelineConnection], List[IncompatibilityIssue]]:
        """Analyze connections for compatibility issues."""
        incompatible_connections = []
        issues = []
        
        for connection in connections:
            if connection.compatibility_score < 0.6:  # Threshold for incompatibility
                incompatible_connections.append(connection)
                
                # Determine issue severity
                if connection.compatibility_score < 0.3:
                    severity = "critical"
                    description = f"Very low compatibility ({connection.compatibility_score:.2f}) between {connection.source_step.operation} and {connection.target_step.operation}"
                elif connection.compatibility_score < 0.6:
                    severity = "warning"
                    description = f"Moderate compatibility issues ({connection.compatibility_score:.2f}) between {connection.source_step.operation} and {connection.target_step.operation}"
                else:
                    severity = "info"
                    description = f"Minor compatibility concerns ({connection.compatibility_score:.2f}) between {connection.source_step.operation} and {connection.target_step.operation}"
                
                # Generate suggested solutions
                solutions = []
                if connection.conversion_path:
                    solutions.append(f"Use conversion path: {' -> '.join([step.adapter_id for step in connection.conversion_path.steps])}")
                solutions.append(f"Insert adapter for {connection.source_step.output_format.value} -> {connection.target_step.input_format.value}")
                
                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="format_incompatibility",
                    severity=severity,
                    description=description,
                    suggested_solutions=solutions,
                    cost_estimate=connection.conversion_path.total_cost if connection.conversion_path else None
                )
                
                issues.append(issue)
        
        return incompatible_connections, issues
    
    def _analyze_performance(self, connections: List[PipelineConnection]) -> List[IncompatibilityIssue]:
        """Analyze connections for performance issues."""
        issues = []
        
        for connection in connections:
            # Check for performance bottlenecks
            if connection.conversion_path and connection.conversion_path.total_cost.time_estimate_seconds > 10.0:
                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="performance_bottleneck",
                    severity="warning",
                    description=f"Slow conversion expected ({connection.conversion_path.total_cost.time_estimate_seconds:.1f}s) between {connection.source_step.operation} and {connection.target_step.operation}",
                    suggested_solutions=[
                        "Consider using a more efficient adapter",
                        "Implement parallel processing",
                        "Use streaming conversion"
                    ],
                    cost_estimate=connection.conversion_path.total_cost
                )
                issues.append(issue)
            
            # Check for memory issues
            if connection.conversion_path and connection.conversion_path.total_cost.memory_cost_mb > 1000:
                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="memory_intensive",
                    severity="warning",
                    description=f"High memory usage expected ({connection.conversion_path.total_cost.memory_cost_mb:.0f}MB) between {connection.source_step.operation} and {connection.target_step.operation}",
                    suggested_solutions=[
                        "Use streaming processing",
                        "Implement chunked conversion",
                        "Consider sparse representations"
                    ],
                    cost_estimate=connection.conversion_path.total_cost
                )
                issues.append(issue)
        
        return issues
    
    def _analyze_quality(self, connections: List[PipelineConnection]) -> List[IncompatibilityIssue]:
        """Analyze connections for quality issues."""
        issues = []
        
        for connection in connections:
            # Check for quality degradation
            if connection.conversion_path and connection.conversion_path.total_cost.quality_impact < -0.1:
                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="quality_degradation",
                    severity="warning",
                    description=f"Quality loss expected ({connection.conversion_path.total_cost.quality_impact:.2f}) in conversion between {connection.source_step.operation} and {connection.target_step.operation}",
                    suggested_solutions=[
                        "Use higher fidelity conversion options",
                        "Consider alternative data paths",
                        "Implement quality preservation strategies"
                    ],
                    cost_estimate=connection.conversion_path.total_cost
                )
                issues.append(issue)
        
        return issues
    
    def _analyze_cost(self, connections: List[PipelineConnection]) -> List[IncompatibilityIssue]:
        """Analyze connections for cost issues."""
        issues = []
        
        for connection in connections:
            # Check for high computational cost
            if connection.conversion_path and connection.conversion_path.total_cost.computational_cost > 0.8:
                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="high_computational_cost",
                    severity="info",
                    description=f"High computational cost ({connection.conversion_path.total_cost.computational_cost:.2f}) for conversion between {connection.source_step.operation} and {connection.target_step.operation}",
                    suggested_solutions=[
                        "Use more efficient algorithms",
                        "Consider approximate conversions",
                        "Implement caching strategies"
                    ],
                    cost_estimate=connection.conversion_path.total_cost
                )
                issues.append(issue)
        
        return issues
    
    def _generate_shim_recommendations(self, incompatible_connections: List[PipelineConnection]) -> List[ShimRecommendation]:
        """Generate shim recommendations for incompatible connections."""
        recommendations = []
        
        for connection in incompatible_connections:
            # Find compatible adapters for this conversion
            request = ConversionRequest(
                source_data=None,  # We don't have actual data for analysis
                source_format=connection.source_step.output_format,
                target_format=connection.target_step.input_format
            )
            
            compatible_adapters = self.shim_registry.get_compatible_adapters(request)
            
            if compatible_adapters:
                # Select best adapter (highest confidence)
                best_adapter, confidence = compatible_adapters[0]
                
                # Estimate cost
                try:
                    cost_estimate = best_adapter.estimate_cost(request)
                except Exception as e:
                    logger.warning(f"Could not estimate cost for adapter {best_adapter.adapter_id}: {e}")
                    cost_estimate = ConversionCost(
                        computational_cost=0.5,
                        memory_cost_mb=100,
                        time_estimate_seconds=1.0
                    )
                
                recommendation = ShimRecommendation(
                    connection=connection,
                    recommended_shim=best_adapter,
                    insertion_point="before_target",
                    confidence=confidence,
                    expected_benefit=f"Resolve format incompatibility with {confidence:.2f} confidence",
                    cost_estimate=cost_estimate,
                    alternative_shims=compatible_adapters[1:5]  # Top 5 alternatives
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_overall_compatibility_score(self, connections: List[PipelineConnection]) -> float:
        """Calculate overall pipeline compatibility score."""
        if not connections:
            return 1.0
        
        total_score = sum(connection.compatibility_score for connection in connections)
        return total_score / len(connections)
    
    def _generate_cache_key(self, pipeline_steps: List[PipelineStep], 
                           analysis_type: AnalysisType, pipeline_id: str) -> str:
        """Generate cache key for analysis results."""
        step_signature = "_".join([
            f"{step.step_id}:{step.domain}:{step.operation}:{step.input_format.value}:{step.output_format.value}"
            for step in pipeline_steps
        ])
        return f"{pipeline_id}_{analysis_type.value}_{hash(step_signature)}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[PipelineAnalysisResult]:
        """Get cached analysis result."""
        if not self.enable_caching:
            return None
        
        with self._cache_lock:
            return self._analysis_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: PipelineAnalysisResult) -> None:
        """Cache analysis result."""
        if not self.enable_caching:
            return
        
        with self._cache_lock:
            # Simple cache size management - keep last 100 results
            if len(self._analysis_cache) > 100:
                # Remove oldest entries
                sorted_items = sorted(self._analysis_cache.items(), 
                                    key=lambda x: x[1].analysis_timestamp)
                for old_key, _ in sorted_items[:20]:  # Remove 20 oldest
                    del self._analysis_cache[old_key]
            
            self._analysis_cache[cache_key] = result
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        cache_stats = {}
        if self.enable_caching:
            with self._cache_lock:
                cache_stats = {
                    'cache_size': len(self._analysis_cache),
                    'cache_hit_rate': self._stats['cache_hits'] / max(self._stats['total_analyses'], 1)
                }
        
        return {
            **self._stats,
            **cache_stats,
            'active_threads': self._executor._threads,
            'max_threads': self.max_analysis_threads
        }
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        if self.enable_caching:
            with self._cache_lock:
                self._analysis_cache.clear()
                logger.info("Analysis cache cleared")
    
    def shutdown(self) -> None:
        """Shutdown analyzer and cleanup resources."""
        self._executor.shutdown(wait=True)
        if self.enable_caching:
            self.clear_cache()
        logger.info("PipelineAnalyzer shutdown completed")


class ShimInjector:
    """
    Automatic shim injection system for incompatible pipeline connections.
    
    Intelligently inserts shim adapters at optimal positions in pipeline chains
    with cost-based optimization and chained conversion support.
    """
    
    def __init__(self,
                 shim_registry: ShimRegistry,
                 compatibility_matrix: PipelineCompatibilityMatrix,
                 optimization_criteria: Optional[OptimizationCriteria] = None):
        """
        Initialize ShimInjector.
        
        Args:
            shim_registry: Registry of available shim adapters
            compatibility_matrix: Matrix for compatibility assessment
            optimization_criteria: Criteria for shim selection optimization
        """
        self.shim_registry = shim_registry
        self.compatibility_matrix = compatibility_matrix
        self.optimization_criteria = optimization_criteria or OptimizationCriteria()
        
        # Injection statistics
        self._stats = {
            'total_injections': 0,
            'successful_injections': 0,
            'failed_injections': 0,
            'shims_inserted': 0
        }
        
        logger.info("ShimInjector initialized")
    
    def inject_shims_for_pipeline(self,
                                 pipeline_steps: List[PipelineStep],
                                 analysis_result: PipelineAnalysisResult,
                                 strategy: InjectionStrategy = InjectionStrategy.BALANCED) -> Tuple[List[PipelineStep], Dict[str, Any]]:
        """
        Inject shims into pipeline based on analysis results.
        
        Args:
            pipeline_steps: Original pipeline steps
            analysis_result: Analysis result with recommendations
            strategy: Strategy for shim injection
            
        Returns:
            Tuple of (modified_pipeline_steps, injection_metadata)
        """
        start_time = time.time()
        self._stats['total_injections'] += 1
        
        logger.info(f"Starting shim injection",
                   pipeline_id=analysis_result.pipeline_id,
                   strategy=strategy.value,
                   recommendations_count=len(analysis_result.shim_recommendations))
        
        try:
            # Select recommendations based on strategy
            selected_recommendations = self._select_recommendations_by_strategy(
                analysis_result.shim_recommendations, strategy
            )
            
            # Sort recommendations by insertion order
            ordered_recommendations = self._order_recommendations_for_injection(
                selected_recommendations, pipeline_steps
            )
            
            # Inject shims
            modified_steps = pipeline_steps.copy()
            injection_metadata = {
                'injections': [],
                'skipped': [],
                'errors': []
            }
            
            for recommendation in ordered_recommendations:
                try:
                    modified_steps, injection_info = self._inject_single_shim(
                        modified_steps, recommendation
                    )
                    injection_metadata['injections'].append(injection_info)
                    self._stats['shims_inserted'] += 1
                    
                except Exception as e:
                    error_info = {
                        'recommendation': recommendation,
                        'error': str(e),
                        'connection': f"{recommendation.connection.source_step.step_id} -> {recommendation.connection.target_step.step_id}"
                    }
                    injection_metadata['errors'].append(error_info)
                    logger.error(f"Failed to inject shim: {e}")
            
            self._stats['successful_injections'] += 1
            
            # Add overall metadata
            injection_metadata.update({
                'strategy': strategy.value,
                'total_recommendations': len(analysis_result.shim_recommendations),
                'selected_recommendations': len(selected_recommendations),
                'successful_injections': len(injection_metadata['injections']),
                'failed_injections': len(injection_metadata['errors']),
                'execution_time': time.time() - start_time
            })
            
            logger.info(f"Shim injection completed",
                       pipeline_id=analysis_result.pipeline_id,
                       shims_injected=len(injection_metadata['injections']),
                       errors=len(injection_metadata['errors']))
            
            return modified_steps, injection_metadata
            
        except Exception as e:
            self._stats['failed_injections'] += 1
            logger.error(f"Shim injection failed: {e}")
            
            return pipeline_steps, {
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _select_recommendations_by_strategy(self,
                                          recommendations: List[ShimRecommendation],
                                          strategy: InjectionStrategy) -> List[ShimRecommendation]:
        """Select recommendations based on injection strategy."""
        if strategy == InjectionStrategy.MINIMAL:
            # Only critical issues
            return [r for r in recommendations 
                   if any(issue.severity == "critical" 
                         for issue in r.connection.identified_issues 
                         if hasattr(r.connection, 'identified_issues'))]
        
        elif strategy == InjectionStrategy.OPTIMAL:
            # All recommendations, prioritize by performance benefit
            return sorted(recommendations, 
                         key=lambda r: (-r.confidence, r.cost_estimate.computational_cost))
        
        elif strategy == InjectionStrategy.SAFE:
            # All recommendations to ensure maximum compatibility
            return recommendations
        
        elif strategy == InjectionStrategy.BALANCED:
            # Filter by confidence threshold and reasonable cost
            return [r for r in recommendations 
                   if r.confidence > 0.6 and r.cost_estimate.computational_cost < 0.8]
        
        return recommendations
    
    def _order_recommendations_for_injection(self,
                                           recommendations: List[ShimRecommendation],
                                           pipeline_steps: List[PipelineStep]) -> List[ShimRecommendation]:
        """Order recommendations for optimal injection sequence."""
        # Create a mapping of step IDs to indices
        step_indices = {step.step_id: i for i, step in enumerate(pipeline_steps)}
        
        # Sort by pipeline position (upstream first to avoid index shifts)
        return sorted(recommendations, 
                     key=lambda r: step_indices.get(r.connection.source_step.step_id, 0))
    
    def _inject_single_shim(self,
                           pipeline_steps: List[PipelineStep],
                           recommendation: ShimRecommendation) -> Tuple[List[PipelineStep], Dict[str, Any]]:
        """Inject a single shim based on recommendation."""
        # Find the insertion point
        source_step_idx = None
        target_step_idx = None
        
        for i, step in enumerate(pipeline_steps):
            if step.step_id == recommendation.connection.source_step.step_id:
                source_step_idx = i
            if step.step_id == recommendation.connection.target_step.step_id:
                target_step_idx = i
        
        if source_step_idx is None or target_step_idx is None:
            raise ValueError("Could not find source or target step in pipeline")
        
        # Create shim step
        shim_step = PipelineStep(
            step_id=f"shim_{recommendation.recommended_shim.adapter_id}_{int(time.time() * 1000)}",
            domain="conversion",
            operation=f"convert_{recommendation.connection.source_step.output_format.value}_to_{recommendation.connection.target_step.input_format.value}",
            input_format=recommendation.connection.source_step.output_format,
            output_format=recommendation.connection.target_step.input_format,
            metadata={
                'shim_adapter': recommendation.recommended_shim.adapter_id,
                'injection_reason': 'format_incompatibility',
                'confidence': recommendation.confidence,
                'cost_estimate': recommendation.cost_estimate
            }
        )
        
        # Insert shim step
        if recommendation.insertion_point == "before_target":
            insertion_idx = target_step_idx
        elif recommendation.insertion_point == "after_source":
            insertion_idx = source_step_idx + 1
        else:  # intermediate
            insertion_idx = source_step_idx + 1
        
        modified_steps = pipeline_steps.copy()
        modified_steps.insert(insertion_idx, shim_step)
        
        injection_info = {
            'shim_step': shim_step,
            'insertion_index': insertion_idx,
            'recommendation': recommendation,
            'connection_resolved': f"{recommendation.connection.source_step.step_id} -> {recommendation.connection.target_step.step_id}"
        }
        
        return modified_steps, injection_info
    
    def optimize_shim_selection(self,
                               compatible_adapters: List[Tuple[EnhancedShimAdapter, float]],
                               conversion_request: ConversionRequest) -> Tuple[EnhancedShimAdapter, float]:
        """
        Optimize shim selection based on criteria.
        
        Args:
            compatible_adapters: List of compatible adapters with confidence scores
            conversion_request: The conversion request
            
        Returns:
            Tuple of (selected_adapter, optimization_score)
        """
        if not compatible_adapters:
            raise ValueError("No compatible adapters provided")
        
        best_adapter = None
        best_score = -1.0
        
        for adapter, confidence in compatible_adapters:
            try:
                cost = adapter.estimate_cost(conversion_request)
                
                # Calculate optimization score
                performance_score = 1.0 - cost.computational_cost  # Lower cost = higher score
                quality_score = confidence  # Confidence as quality proxy
                cost_score = 1.0 - (cost.memory_cost_mb / 1000.0)  # Normalize memory cost
                cost_score = max(0.0, min(1.0, cost_score))  # Clamp to [0,1]
                
                # Weighted combination
                optimization_score = (
                    self.optimization_criteria.performance_weight * performance_score +
                    self.optimization_criteria.quality_weight * quality_score +
                    self.optimization_criteria.cost_weight * cost_score
                )
                
                # Apply thresholds
                if (self.optimization_criteria.quality_threshold and 
                    confidence < self.optimization_criteria.quality_threshold):
                    continue
                
                if (self.optimization_criteria.max_cost_threshold and 
                    cost.computational_cost > self.optimization_criteria.max_cost_threshold):
                    continue
                
                if optimization_score > best_score:
                    best_score = optimization_score
                    best_adapter = adapter
                    
            except Exception as e:
                logger.warning(f"Error evaluating adapter {adapter.adapter_id}: {e}")
                continue
        
        if best_adapter is None:
            # Fallback to highest confidence adapter
            best_adapter, confidence = compatible_adapters[0]
            best_score = confidence
        
        return best_adapter, best_score
    
    def get_injection_statistics(self) -> Dict[str, Any]:
        """Get injection statistics."""
        return self._stats.copy()


class PipelineValidator:
    """
    Complete pipeline composition validator with execution plan generation.
    
    Validates entire pipeline chains, checks for circular dependencies,
    identifies bottlenecks, and generates optimized execution plans.
    """
    
    def __init__(self,
                 compatibility_matrix: PipelineCompatibilityMatrix,
                 shim_registry: ShimRegistry,
                 analyzer: PipelineAnalyzer,
                 injector: ShimInjector):
        """
        Initialize PipelineValidator.
        
        Args:
            compatibility_matrix: Matrix for compatibility assessment
            shim_registry: Registry of available shim adapters  
            analyzer: Pipeline analyzer for issue detection
            injector: Shim injector for automatic fixes
        """
        self.compatibility_matrix = compatibility_matrix
        self.shim_registry = shim_registry
        self.analyzer = analyzer
        self.injector = injector
        
        # Validation statistics
        self._stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'pipelines_fixed': 0
        }
        
        logger.info("PipelineValidator initialized")
    
    def validate_and_fix_pipeline(self,
                                 pipeline_steps: List[PipelineStep],
                                 auto_fix: bool = True,
                                 validation_level: str = "strict") -> Dict[str, Any]:
        """
        Validate pipeline and optionally auto-fix issues.
        
        Args:
            pipeline_steps: Pipeline steps to validate
            auto_fix: Whether to automatically fix detected issues
            validation_level: 'strict', 'moderate', or 'lenient'
            
        Returns:
            Comprehensive validation result with fixes applied
        """
        start_time = time.time()
        self._stats['total_validations'] += 1
        
        pipeline_id = f"validation_{int(time.time() * 1000)}"
        
        logger.info(f"Starting pipeline validation",
                   pipeline_id=pipeline_id,
                   steps_count=len(pipeline_steps),
                   auto_fix=auto_fix,
                   validation_level=validation_level)
        
        try:
            validation_result = {
                'pipeline_id': pipeline_id,
                'original_steps_count': len(pipeline_steps),
                'validation_level': validation_level,
                'auto_fix_enabled': auto_fix,
                'is_valid': False,
                'validation_errors': [],
                'validation_warnings': [],
                'structural_issues': [],
                'compatibility_issues': [],
                'performance_issues': [],
                'fixes_applied': [],
                'final_pipeline': pipeline_steps.copy(),
                'execution_plan': None,
                'validation_score': 0.0,
                'execution_time': 0.0
            }
            
            # 1. Structural validation
            structural_issues = self._validate_pipeline_structure(pipeline_steps)
            validation_result['structural_issues'] = structural_issues
            
            if structural_issues and validation_level == "strict":
                validation_result['validation_errors'].extend([
                    f"Structural issue: {issue}" for issue in structural_issues
                ])
            
            # 2. Compatibility analysis
            analysis_result = self.analyzer.analyze_pipeline(
                pipeline_steps, 
                AnalysisType.COMPLETE,
                pipeline_id
            )
            
            validation_result['compatibility_issues'] = analysis_result.identified_issues
            validation_result['validation_score'] = analysis_result.compatibility_score
            
            # 3. Auto-fix if enabled and issues found
            fixed_pipeline = pipeline_steps.copy()
            if auto_fix and not analysis_result.is_compatible:
                try:
                    fixed_pipeline, injection_metadata = self.injector.inject_shims_for_pipeline(
                        pipeline_steps, analysis_result, InjectionStrategy.BALANCED
                    )
                    
                    validation_result['fixes_applied'] = injection_metadata.get('injections', [])
                    validation_result['final_pipeline'] = fixed_pipeline
                    
                    if injection_metadata.get('injections'):
                        self._stats['pipelines_fixed'] += 1
                    
                    # Re-analyze fixed pipeline
                    fixed_analysis = self.analyzer.analyze_pipeline(
                        fixed_pipeline, 
                        AnalysisType.COMPATIBILITY,
                        f"{pipeline_id}_fixed"
                    )
                    
                    validation_result['is_valid'] = fixed_analysis.is_compatible
                    validation_result['validation_score'] = fixed_analysis.compatibility_score
                    
                except Exception as e:
                    logger.error(f"Auto-fix failed: {e}")
                    validation_result['validation_errors'].append(f"Auto-fix failed: {str(e)}")
            else:
                validation_result['is_valid'] = analysis_result.is_compatible
            
            # 4. Generate execution plan
            if validation_result['is_valid'] or validation_level == "lenient":
                execution_plan = self._generate_execution_plan(fixed_pipeline)
                validation_result['execution_plan'] = execution_plan
            
            # 5. Performance validation
            performance_issues = self._validate_performance(fixed_pipeline)
            validation_result['performance_issues'] = performance_issues
            
            if performance_issues and validation_level == "strict":
                validation_result['validation_warnings'].extend([
                    f"Performance issue: {issue}" for issue in performance_issues
                ])
            
            self._stats['successful_validations'] += 1
            
            validation_result['execution_time'] = time.time() - start_time
            
            logger.info(f"Pipeline validation completed",
                       pipeline_id=pipeline_id,
                       is_valid=validation_result['is_valid'],
                       fixes_applied=len(validation_result['fixes_applied']),
                       final_score=validation_result['validation_score'])
            
            return validation_result
            
        except Exception as e:
            self._stats['failed_validations'] += 1
            logger.error(f"Pipeline validation failed: {e}")
            
            return {
                'pipeline_id': pipeline_id,
                'is_valid': False,
                'validation_errors': [f"Validation failed: {str(e)}"],
                'execution_time': time.time() - start_time
            }
    
    def _validate_pipeline_structure(self, pipeline_steps: List[PipelineStep]) -> List[str]:
        """Validate structural integrity of pipeline."""
        issues = []
        
        if not pipeline_steps:
            issues.append("Pipeline is empty")
            return issues
        
        if len(pipeline_steps) < 2:
            issues.append("Pipeline must have at least 2 steps")
        
        # Check for duplicate step IDs
        step_ids = [step.step_id for step in pipeline_steps]
        if len(step_ids) != len(set(step_ids)):
            issues.append("Duplicate step IDs found")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(pipeline_steps):
            issues.append("Circular dependencies detected")
        
        # Validate format flow continuity
        for i in range(len(pipeline_steps) - 1):
            current_step = pipeline_steps[i]
            next_step = pipeline_steps[i + 1]
            
            if current_step.output_format != next_step.input_format:
                # This is expected and should be handled by compatibility analysis
                continue
        
        return issues
    
    def _has_circular_dependencies(self, pipeline_steps: List[PipelineStep]) -> bool:
        """Check for circular dependencies in pipeline."""
        # Create directed graph
        graph = nx.DiGraph()
        
        for step in pipeline_steps:
            graph.add_node(step.step_id)
        
        # Add edges based on step sequence
        for i in range(len(pipeline_steps) - 1):
            graph.add_edge(pipeline_steps[i].step_id, pipeline_steps[i + 1].step_id)
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(graph))
            return len(cycles) > 0
        except:
            return False
    
    def _validate_performance(self, pipeline_steps: List[PipelineStep]) -> List[str]:
        """Validate pipeline performance characteristics."""
        issues = []
        
        # Check for potential bottlenecks
        memory_intensive_steps = 0
        compute_intensive_steps = 0
        
        for step in pipeline_steps:
            # Check for known memory-intensive operations
            if any(keyword in step.operation.lower() 
                  for keyword in ['sparse', 'matrix', 'large', 'dense']):
                memory_intensive_steps += 1
            
            # Check for compute-intensive operations
            if any(keyword in step.operation.lower() 
                  for keyword in ['model', 'train', 'optimize', 'search']):
                compute_intensive_steps += 1
        
        if memory_intensive_steps > len(pipeline_steps) * 0.5:
            issues.append("High number of memory-intensive operations may cause performance issues")
        
        if compute_intensive_steps > len(pipeline_steps) * 0.3:
            issues.append("High number of compute-intensive operations may cause long execution times")
        
        return issues
    
    def _generate_execution_plan(self, pipeline_steps: List[PipelineStep]) -> Dict[str, Any]:
        """Generate optimized execution plan for pipeline."""
        plan = {
            'steps': [],
            'estimated_total_time': 0.0,
            'estimated_total_memory': 0.0,
            'parallel_opportunities': [],
            'optimization_suggestions': []
        }
        
        for i, step in enumerate(pipeline_steps):
            step_plan = {
                'step_index': i,
                'step_id': step.step_id,
                'operation': step.operation,
                'domain': step.domain,
                'input_format': step.input_format.value,
                'output_format': step.output_format.value,
                'estimated_time': self._estimate_step_time(step),
                'estimated_memory': self._estimate_step_memory(step),
                'dependencies': [pipeline_steps[i-1].step_id] if i > 0 else [],
                'can_parallelize': self._can_step_parallelize(step)
            }
            
            plan['steps'].append(step_plan)
            plan['estimated_total_time'] += step_plan['estimated_time']
            plan['estimated_total_memory'] = max(plan['estimated_total_memory'], 
                                               step_plan['estimated_memory'])
        
        # Identify parallel opportunities
        plan['parallel_opportunities'] = self._identify_parallel_opportunities(plan['steps'])
        
        # Generate optimization suggestions
        plan['optimization_suggestions'] = self._generate_optimization_suggestions(plan)
        
        return plan
    
    def _estimate_step_time(self, step: PipelineStep) -> float:
        """Estimate execution time for a pipeline step."""
        # Simple heuristic based on operation type
        base_time = 1.0  # seconds
        
        if 'convert' in step.operation.lower():
            base_time *= 2.0
        elif 'model' in step.operation.lower():
            base_time *= 10.0
        elif 'analysis' in step.operation.lower():
            base_time *= 3.0
        
        return base_time
    
    def _estimate_step_memory(self, step: PipelineStep) -> float:
        """Estimate memory usage for a pipeline step."""
        # Simple heuristic based on operation type
        base_memory = 100.0  # MB
        
        if 'sparse' in step.operation.lower():
            base_memory *= 0.5
        elif 'dense' in step.operation.lower():
            base_memory *= 5.0
        elif 'matrix' in step.operation.lower():
            base_memory *= 3.0
        
        return base_memory
    
    def _can_step_parallelize(self, step: PipelineStep) -> bool:
        """Check if step can be parallelized."""
        # Simple heuristic
        parallel_operations = ['analysis', 'transform', 'process']
        return any(op in step.operation.lower() for op in parallel_operations)
    
    def _identify_parallel_opportunities(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify opportunities for parallel execution."""
        opportunities = []
        
        # Look for independent steps that can run in parallel
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            
            if (current_step['can_parallelize'] and 
                next_step['can_parallelize'] and
                len(next_step['dependencies']) == 1):
                
                opportunity = {
                    'type': 'parallel_sequence',
                    'steps': [current_step['step_id'], next_step['step_id']],
                    'potential_time_saving': min(current_step['estimated_time'], 
                                               next_step['estimated_time']) * 0.8
                }
                opportunities.append(opportunity)
        
        return opportunities
    
    def _generate_optimization_suggestions(self, plan: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions for execution plan."""
        suggestions = []
        
        total_time = plan['estimated_total_time']
        total_memory = plan['estimated_total_memory']
        
        if total_time > 300:  # 5 minutes
            suggestions.append("Consider implementing parallel processing to reduce execution time")
        
        if total_memory > 2000:  # 2GB
            suggestions.append("Consider using streaming processing to reduce memory usage")
        
        if len(plan['parallel_opportunities']) > 0:
            suggestions.append(f"Found {len(plan['parallel_opportunities'])} opportunities for parallel execution")
        
        # Check for consecutive conversion steps
        conversion_steps = [step for step in plan['steps'] if 'convert' in step['operation'].lower()]
        if len(conversion_steps) > 2:
            suggestions.append("Multiple consecutive conversions detected - consider combining or optimizing")
        
        return suggestions
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self._stats.copy()


# Factory Functions

def create_pipeline_analyzer(compatibility_matrix: PipelineCompatibilityMatrix,
                           shim_registry: ShimRegistry,
                           **kwargs) -> PipelineAnalyzer:
    """Create a PipelineAnalyzer with standard configuration."""
    return PipelineAnalyzer(
        compatibility_matrix=compatibility_matrix,
        shim_registry=shim_registry,
        **kwargs
    )


def create_shim_injector(shim_registry: ShimRegistry,
                        compatibility_matrix: PipelineCompatibilityMatrix,
                        **kwargs) -> ShimInjector:
    """Create a ShimInjector with standard configuration."""
    return ShimInjector(
        shim_registry=shim_registry,
        compatibility_matrix=compatibility_matrix,
        **kwargs
    )


def create_pipeline_validator(compatibility_matrix: PipelineCompatibilityMatrix,
                             shim_registry: ShimRegistry,
                             **kwargs) -> PipelineValidator:
    """Create a PipelineValidator with complete analysis capabilities."""
    analyzer = create_pipeline_analyzer(compatibility_matrix, shim_registry)
    injector = create_shim_injector(shim_registry, compatibility_matrix)
    
    return PipelineValidator(
        compatibility_matrix=compatibility_matrix,
        shim_registry=shim_registry,
        analyzer=analyzer,
        injector=injector,
        **kwargs
    )


def create_optimization_criteria(**kwargs) -> OptimizationCriteria:
    """Create OptimizationCriteria with custom parameters."""
    return OptimizationCriteria(**kwargs)


# Utility Functions

def create_pipeline_step(step_id: str, 
                        domain: str,
                        operation: str,
                        input_format: DataFormat,
                        output_format: DataFormat,
                        **kwargs) -> PipelineStep:
    """Factory function to create a PipelineStep."""
    return PipelineStep(
        step_id=step_id,
        domain=domain,
        operation=operation,
        input_format=input_format,
        output_format=output_format,
        **kwargs
    )


def analyze_and_fix_pipeline(pipeline_steps: List[PipelineStep],
                           compatibility_matrix: PipelineCompatibilityMatrix,
                           shim_registry: ShimRegistry,
                           auto_fix: bool = True) -> Dict[str, Any]:
    """
    High-level utility function to analyze and fix a pipeline in one call.
    
    Args:
        pipeline_steps: Pipeline steps to analyze and fix
        compatibility_matrix: Compatibility matrix for analysis
        shim_registry: Registry of available shims
        auto_fix: Whether to automatically fix detected issues
        
    Returns:
        Complete analysis and fix results
    """
    validator = create_pipeline_validator(compatibility_matrix, shim_registry)
    
    return validator.validate_and_fix_pipeline(
        pipeline_steps=pipeline_steps,
        auto_fix=auto_fix,
        validation_level="balanced"
    )