"""
Core pipeline analyzer for identifying incompatible connections and data flow issues.

Analyzes pipeline chains to identify format incompatibilities, performance bottlenecks,
and optimization opportunities with detailed reporting and recommendations.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..interfaces import ConversionRequest, ConversionCost
from ..compatibility_matrix import PipelineCompatibilityMatrix
from ..shim_registry import ShimRegistry
from ....logging_manager import get_logger

from .types import (
    AnalysisType,
    PipelineStep,
    PipelineConnection,
    IncompatibilityIssue,
    ShimRecommendation,
    PipelineAnalysisResult,
)

logger = get_logger(__name__)


class PipelineAnalyzer:
    """
    Core pipeline analyzer for identifying incompatible connections and data flow issues.

    Analyzes pipeline chains to identify format incompatibilities, performance bottlenecks,
    and optimization opportunities with detailed reporting and recommendations.
    """

    def __init__(
        self,
        compatibility_matrix: PipelineCompatibilityMatrix,
        shim_registry: ShimRegistry,
        enable_caching: bool = True,
        max_analysis_threads: int = 4,
    ):
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
        self._executor = ThreadPoolExecutor(
            max_workers=max_analysis_threads, thread_name_prefix="pipeline_analyzer"
        )

        # Analysis statistics
        self._stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "issues_identified": 0,
            "shims_recommended": 0,
        }

        logger.info(
            "PipelineAnalyzer initialized",
            caching_enabled=enable_caching,
            max_threads=max_analysis_threads,
        )

    def analyze_pipeline(
        self,
        pipeline_steps: List[PipelineStep],
        analysis_type: AnalysisType = AnalysisType.COMPLETE,
        pipeline_id: Optional[str] = None,
    ) -> PipelineAnalysisResult:
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
                self._stats["cache_hits"] += 1
                return cached_result

        self._stats["total_analyses"] += 1

        logger.info(
            f"Starting pipeline analysis",
            pipeline_id=pipeline_id,
            steps_count=len(pipeline_steps),
            analysis_type=analysis_type.value,
        )

        try:
            # Build pipeline graph
            connections = self._build_pipeline_connections(pipeline_steps)

            # Analyze connections based on type
            incompatible_connections = []
            identified_issues = []
            shim_recommendations = []

            if analysis_type in [AnalysisType.COMPATIBILITY, AnalysisType.COMPLETE]:
                incompatible_connections, issues = self._analyze_compatibility(
                    connections
                )
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
                shim_recommendations = self._generate_shim_recommendations(
                    incompatible_connections
                )

            # Calculate overall compatibility score
            compatibility_score = self._calculate_overall_compatibility_score(
                connections
            )

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
                    "total_connections": len(connections),
                    "analysis_threads_used": min(
                        len(connections), self.max_analysis_threads
                    ),
                    "cache_key": cache_key,
                },
                execution_time=execution_time,
            )

            # Cache result
            if self.enable_caching:
                self._cache_result(cache_key, result)

            # Update statistics
            self._stats["issues_identified"] += len(identified_issues)
            self._stats["shims_recommended"] += len(shim_recommendations)

            logger.info(
                f"Pipeline analysis completed",
                pipeline_id=pipeline_id,
                is_compatible=result.is_compatible,
                issues_count=len(identified_issues),
                recommendations_count=len(shim_recommendations),
                execution_time=execution_time,
            )

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
                        description=f"Analysis failed: {str(e)}",
                    )
                ],
                shim_recommendations=[],
                execution_time=time.time() - start_time,
            )

    def _build_pipeline_connections(
        self, pipeline_steps: List[PipelineStep]
    ) -> List[PipelineConnection]:
        """Build connections between pipeline steps."""
        connections = []

        for i in range(len(pipeline_steps) - 1):
            source_step = pipeline_steps[i]
            target_step = pipeline_steps[i + 1]

            # Calculate compatibility between steps
            compatibility = self.compatibility_matrix.get_compatibility(
                source_step.output_format, target_step.input_format
            )

            connection = PipelineConnection(
                source_step=source_step,
                target_step=target_step,
                compatibility_score=compatibility.score,
                requires_conversion=compatibility.conversion_required,
                conversion_path=compatibility.conversion_path,
            )

            connections.append(connection)

        return connections

    def _analyze_compatibility(
        self, connections: List[PipelineConnection]
    ) -> Tuple[List[PipelineConnection], List[IncompatibilityIssue]]:
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
                    solutions.append(
                        f"Use conversion path: {' -> '.join([step.adapter_id for step in connection.conversion_path.steps])}"
                    )
                solutions.append(
                    f"Insert adapter for {connection.source_step.output_format.value} -> {connection.target_step.input_format.value}"
                )

                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="format_incompatibility",
                    severity=severity,
                    description=description,
                    suggested_solutions=solutions,
                    cost_estimate=connection.conversion_path.total_cost
                    if connection.conversion_path
                    else None,
                )

                issues.append(issue)

        return incompatible_connections, issues

    def _analyze_performance(
        self, connections: List[PipelineConnection]
    ) -> List[IncompatibilityIssue]:
        """Analyze connections for performance issues."""
        issues = []

        for connection in connections:
            # Check for performance bottlenecks
            if (
                connection.conversion_path
                and connection.conversion_path.total_cost.time_estimate_seconds > 10.0
            ):
                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="performance_bottleneck",
                    severity="warning",
                    description=f"Slow conversion expected ({connection.conversion_path.total_cost.time_estimate_seconds:.1f}s) between {connection.source_step.operation} and {connection.target_step.operation}",
                    suggested_solutions=[
                        "Consider using a more efficient adapter",
                        "Implement parallel processing",
                        "Use streaming conversion",
                    ],
                    cost_estimate=connection.conversion_path.total_cost,
                )
                issues.append(issue)

            # Check for memory issues
            if (
                connection.conversion_path
                and connection.conversion_path.total_cost.memory_cost_mb > 1000
            ):
                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="memory_intensive",
                    severity="warning",
                    description=f"High memory usage expected ({connection.conversion_path.total_cost.memory_cost_mb:.0f}MB) between {connection.source_step.operation} and {connection.target_step.operation}",
                    suggested_solutions=[
                        "Use streaming processing",
                        "Implement chunked conversion",
                        "Consider sparse representations",
                    ],
                    cost_estimate=connection.conversion_path.total_cost,
                )
                issues.append(issue)

        return issues

    def _analyze_quality(
        self, connections: List[PipelineConnection]
    ) -> List[IncompatibilityIssue]:
        """Analyze connections for quality issues."""
        issues = []

        for connection in connections:
            # Check for quality degradation
            if (
                connection.conversion_path
                and connection.conversion_path.total_cost.quality_impact < -0.1
            ):
                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="quality_degradation",
                    severity="warning",
                    description=f"Quality loss expected ({connection.conversion_path.total_cost.quality_impact:.2f}) in conversion between {connection.source_step.operation} and {connection.target_step.operation}",
                    suggested_solutions=[
                        "Use higher fidelity conversion options",
                        "Consider alternative data paths",
                        "Implement quality preservation strategies",
                    ],
                    cost_estimate=connection.conversion_path.total_cost,
                )
                issues.append(issue)

        return issues

    def _analyze_cost(
        self, connections: List[PipelineConnection]
    ) -> List[IncompatibilityIssue]:
        """Analyze connections for cost issues."""
        issues = []

        for connection in connections:
            # Check for high computational cost
            if (
                connection.conversion_path
                and connection.conversion_path.total_cost.computational_cost > 0.8
            ):
                issue = IncompatibilityIssue(
                    connection=connection,
                    issue_type="high_computational_cost",
                    severity="info",
                    description=f"High computational cost ({connection.conversion_path.total_cost.computational_cost:.2f}) for conversion between {connection.source_step.operation} and {connection.target_step.operation}",
                    suggested_solutions=[
                        "Use more efficient algorithms",
                        "Consider approximate conversions",
                        "Implement caching strategies",
                    ],
                    cost_estimate=connection.conversion_path.total_cost,
                )
                issues.append(issue)

        return issues

    def _generate_shim_recommendations(
        self, incompatible_connections: List[PipelineConnection]
    ) -> List[ShimRecommendation]:
        """Generate shim recommendations for incompatible connections."""
        recommendations = []

        for connection in incompatible_connections:
            # Find compatible adapters for this conversion
            request = ConversionRequest(
                source_data=None,  # We don't have actual data for analysis
                source_format=connection.source_step.output_format,
                target_format=connection.target_step.input_format,
            )

            compatible_adapters = self.shim_registry.get_compatible_adapters(request)

            if compatible_adapters:
                # Select best adapter (highest confidence)
                best_adapter, confidence = compatible_adapters[0]

                # Estimate cost
                try:
                    cost_estimate = best_adapter.estimate_cost(request)
                except Exception as e:
                    logger.warning(
                        f"Could not estimate cost for adapter {best_adapter.adapter_id}: {e}"
                    )
                    cost_estimate = ConversionCost(
                        computational_cost=0.5,
                        memory_cost_mb=100,
                        time_estimate_seconds=1.0,
                    )

                recommendation = ShimRecommendation(
                    connection=connection,
                    recommended_shim=best_adapter,
                    insertion_point="before_target",
                    confidence=confidence,
                    expected_benefit=f"Resolve format incompatibility with {confidence:.2f} confidence",
                    cost_estimate=cost_estimate,
                    alternative_shims=compatible_adapters[1:5],  # Top 5 alternatives
                )

                recommendations.append(recommendation)

        return recommendations

    def _calculate_overall_compatibility_score(
        self, connections: List[PipelineConnection]
    ) -> float:
        """Calculate overall pipeline compatibility score."""
        if not connections:
            return 1.0

        total_score = sum(connection.compatibility_score for connection in connections)
        return total_score / len(connections)

    def _generate_cache_key(
        self,
        pipeline_steps: List[PipelineStep],
        analysis_type: AnalysisType,
        pipeline_id: str,
    ) -> str:
        """Generate cache key for analysis results."""
        step_signature = "_".join(
            [
                f"{step.step_id}:{step.domain}:{step.operation}:{step.input_format.value}:{step.output_format.value}"
                for step in pipeline_steps
            ]
        )
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
                sorted_items = sorted(
                    self._analysis_cache.items(), key=lambda x: x[1].analysis_timestamp
                )
                for old_key, _ in sorted_items[:20]:  # Remove 20 oldest
                    del self._analysis_cache[old_key]

            self._analysis_cache[cache_key] = result

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        cache_stats = {}
        if self.enable_caching:
            with self._cache_lock:
                cache_stats = {
                    "cache_size": len(self._analysis_cache),
                    "cache_hit_rate": self._stats["cache_hits"]
                    / max(self._stats["total_analyses"], 1),
                }

        return {
            **self._stats,
            **cache_stats,
            "active_threads": self._executor._threads,
            "max_threads": self.max_analysis_threads,
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
