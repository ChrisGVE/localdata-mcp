"""
Automatic shim injection system for incompatible pipeline connections.

Intelligently inserts shim adapters at optimal positions in pipeline chains
with cost-based optimization and chained conversion support.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from ....logging_manager import get_logger
from ..compatibility_matrix import PipelineCompatibilityMatrix
from ..interfaces import ConversionCost, ConversionRequest
from ..shim_registry import EnhancedShimAdapter, ShimRegistry
from .types import (
    InjectionStrategy,
    OptimizationCriteria,
    PipelineAnalysisResult,
    PipelineStep,
    ShimRecommendation,
)

logger = get_logger(__name__)


class ShimInjector:
    """
    Automatic shim injection system for incompatible pipeline connections.

    Intelligently inserts shim adapters at optimal positions in pipeline chains
    with cost-based optimization and chained conversion support.
    """

    def __init__(
        self,
        shim_registry: ShimRegistry,
        compatibility_matrix: PipelineCompatibilityMatrix,
        optimization_criteria: Optional[OptimizationCriteria] = None,
    ):
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
            "total_injections": 0,
            "successful_injections": 0,
            "failed_injections": 0,
            "shims_inserted": 0,
        }

        logger.info("ShimInjector initialized")

    def inject_shims_for_pipeline(
        self,
        pipeline_steps: List[PipelineStep],
        analysis_result: PipelineAnalysisResult,
        strategy: InjectionStrategy = InjectionStrategy.BALANCED,
    ) -> Tuple[List[PipelineStep], Dict[str, Any]]:
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
        self._stats["total_injections"] += 1

        logger.info(
            f"Starting shim injection",
            pipeline_id=analysis_result.pipeline_id,
            strategy=strategy.value,
            recommendations_count=len(analysis_result.shim_recommendations),
        )

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
            injection_metadata = {"injections": [], "skipped": [], "errors": []}

            for recommendation in ordered_recommendations:
                try:
                    modified_steps, injection_info = self._inject_single_shim(
                        modified_steps, recommendation
                    )
                    injection_metadata["injections"].append(injection_info)
                    self._stats["shims_inserted"] += 1

                except Exception as e:
                    error_info = {
                        "recommendation": recommendation,
                        "error": str(e),
                        "connection": f"{recommendation.connection.source_step.step_id} -> {recommendation.connection.target_step.step_id}",
                    }
                    injection_metadata["errors"].append(error_info)
                    logger.error(f"Failed to inject shim: {e}")

            self._stats["successful_injections"] += 1

            # Add overall metadata
            injection_metadata.update(
                {
                    "strategy": strategy.value,
                    "total_recommendations": len(analysis_result.shim_recommendations),
                    "selected_recommendations": len(selected_recommendations),
                    "successful_injections": len(injection_metadata["injections"]),
                    "failed_injections": len(injection_metadata["errors"]),
                    "execution_time": time.time() - start_time,
                }
            )

            logger.info(
                f"Shim injection completed",
                pipeline_id=analysis_result.pipeline_id,
                shims_injected=len(injection_metadata["injections"]),
                errors=len(injection_metadata["errors"]),
            )

            return modified_steps, injection_metadata

        except Exception as e:
            self._stats["failed_injections"] += 1
            logger.error(f"Shim injection failed: {e}")

            return pipeline_steps, {
                "error": str(e),
                "execution_time": time.time() - start_time,
            }

    def _select_recommendations_by_strategy(
        self, recommendations: List[ShimRecommendation], strategy: InjectionStrategy
    ) -> List[ShimRecommendation]:
        """Select recommendations based on injection strategy."""
        if strategy == InjectionStrategy.MINIMAL:
            # Only critical issues
            return [
                r
                for r in recommendations
                if any(
                    issue.severity == "critical"
                    for issue in r.connection.identified_issues
                    if hasattr(r.connection, "identified_issues")
                )
            ]

        elif strategy == InjectionStrategy.OPTIMAL:
            # All recommendations, prioritize by performance benefit
            return sorted(
                recommendations,
                key=lambda r: (-r.confidence, r.cost_estimate.computational_cost),
            )

        elif strategy == InjectionStrategy.SAFE:
            # All recommendations to ensure maximum compatibility
            return recommendations

        elif strategy == InjectionStrategy.BALANCED:
            # Filter by confidence threshold and reasonable cost
            return [
                r
                for r in recommendations
                if r.confidence > 0.6 and r.cost_estimate.computational_cost < 0.8
            ]

        return recommendations

    def _order_recommendations_for_injection(
        self,
        recommendations: List[ShimRecommendation],
        pipeline_steps: List[PipelineStep],
    ) -> List[ShimRecommendation]:
        """Order recommendations for optimal injection sequence."""
        # Create a mapping of step IDs to indices
        step_indices = {step.step_id: i for i, step in enumerate(pipeline_steps)}

        # Sort by pipeline position (upstream first to avoid index shifts)
        return sorted(
            recommendations,
            key=lambda r: step_indices.get(r.connection.source_step.step_id, 0),
        )

    def _inject_single_shim(
        self, pipeline_steps: List[PipelineStep], recommendation: ShimRecommendation
    ) -> Tuple[List[PipelineStep], Dict[str, Any]]:
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
                "shim_adapter": recommendation.recommended_shim.adapter_id,
                "injection_reason": "format_incompatibility",
                "confidence": recommendation.confidence,
                "cost_estimate": recommendation.cost_estimate,
            },
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
            "shim_step": shim_step,
            "insertion_index": insertion_idx,
            "recommendation": recommendation,
            "connection_resolved": f"{recommendation.connection.source_step.step_id} -> {recommendation.connection.target_step.step_id}",
        }

        return modified_steps, injection_info

    def optimize_shim_selection(
        self,
        compatible_adapters: List[Tuple[EnhancedShimAdapter, float]],
        conversion_request: ConversionRequest,
    ) -> Tuple[EnhancedShimAdapter, float]:
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
                performance_score = (
                    1.0 - cost.computational_cost
                )  # Lower cost = higher score
                quality_score = confidence  # Confidence as quality proxy
                cost_score = 1.0 - (
                    cost.memory_cost_mb / 1000.0
                )  # Normalize memory cost
                cost_score = max(0.0, min(1.0, cost_score))  # Clamp to [0,1]

                # Weighted combination
                optimization_score = (
                    self.optimization_criteria.performance_weight * performance_score
                    + self.optimization_criteria.quality_weight * quality_score
                    + self.optimization_criteria.cost_weight * cost_score
                )

                # Apply thresholds
                if (
                    self.optimization_criteria.quality_threshold
                    and confidence < self.optimization_criteria.quality_threshold
                ):
                    continue

                if (
                    self.optimization_criteria.max_cost_threshold
                    and cost.computational_cost
                    > self.optimization_criteria.max_cost_threshold
                ):
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
