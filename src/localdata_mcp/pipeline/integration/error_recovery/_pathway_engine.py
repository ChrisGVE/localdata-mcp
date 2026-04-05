"""
AlternativePathwayEngine: intelligent discovery and evaluation of alternative conversion pathways.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from ..interfaces import (
    ConversionCost,
    ConversionError,
    ConversionPath,
    ConversionRequest,
    ConversionStep,
    DataFormat,
)
from ..shim_registry import EnhancedShimAdapter, ShimRegistry
from ....logging_manager import get_logger
from ._types import (
    AlternativePathway,
    ErrorContext,
    PathwayCost,
    QualityAssessment,
)

logger = get_logger(__name__)


class AlternativePathwayEngine:
    """
    Intelligent discovery and evaluation of alternative conversion pathways.

    Finds alternative routes when direct conversion fails, with cost-benefit
    analysis and quality assessment for each pathway option.
    """

    def __init__(
        self,
        registry: Optional[ShimRegistry] = None,
        max_pathway_depth: int = 5,
        enable_pathway_caching: bool = True,
        quality_threshold: float = 0.6,
    ):
        """
        Initialize AlternativePathwayEngine.

        Args:
            registry: ShimRegistry for adapter discovery
            max_pathway_depth: Maximum number of conversion steps in pathway
            enable_pathway_caching: Cache successful pathways for reuse
            quality_threshold: Minimum quality threshold for pathways
        """
        self.registry = registry
        self.max_pathway_depth = max_pathway_depth
        self.enable_pathway_caching = enable_pathway_caching
        self.quality_threshold = quality_threshold

        # Pathway caching (new enhanced system)
        self._pathway_cache: Dict[str, List[ConversionPath]] = {}
        self._successful_pathways: Dict[str, ConversionPath] = {}

        # Performance tracking
        self._pathway_success_rates: Dict[str, float] = defaultdict(float)
        self._pathway_performance: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Legacy pathway analysis cache (for backward compatibility)
        self.pathway_cache: Dict[str, List[AlternativePathway]] = {}
        self.success_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Analysis statistics
        self.stats = {
            "pathways_analyzed": 0,
            "alternatives_found": 0,
            "successful_alternatives": 0,
            "cache_hits": 0,
        }

        # Thread safety
        self._lock = threading.RLock()

        logger.info(
            "AlternativePathwayEngine initialized",
            max_depth=max_pathway_depth,
            caching_enabled=enable_pathway_caching,
            quality_threshold=quality_threshold,
        )

    def find_alternative_pathways(
        self,
        failed_conversion: ConversionRequest,
        error_context: ErrorContext,
        max_alternatives: int = 5,
    ) -> List[AlternativePathway]:
        """
        Find alternative pathways for failed conversion.

        Args:
            failed_conversion: The conversion that failed
            error_context: Context of the failure
            max_alternatives: Maximum number of alternatives to return

        Returns:
            List of alternative pathways with feasibility analysis
        """
        cache_key = self._generate_pathway_cache_key(failed_conversion, error_context)

        with self._lock:
            self.stats["pathways_analyzed"] += 1

            # Check cache first
            if cache_key in self.pathway_cache:
                self.stats["cache_hits"] += 1
                return self.pathway_cache[cache_key][:max_alternatives]

        logger.info(
            "Analyzing alternative pathways",
            source_format=failed_conversion.source_format.value,
            target_format=failed_conversion.target_format.value,
            error_type=error_context.error_type,
        )

        alternatives = []

        # Strategy 1: Find alternative intermediate formats
        intermediate_alternatives = self._find_intermediate_format_alternatives(
            failed_conversion
        )
        alternatives.extend(intermediate_alternatives)

        # Strategy 2: Find alternative target formats with similar capabilities
        target_alternatives = self._find_alternative_target_formats(failed_conversion)
        alternatives.extend(target_alternatives)

        # Strategy 3: Find alternative source format preprocessors
        source_alternatives = self._find_source_preprocessing_alternatives(
            failed_conversion
        )
        alternatives.extend(source_alternatives)

        # Strategy 4: Find pipeline restructuring alternatives
        restructuring_alternatives = self._find_restructuring_alternatives(
            failed_conversion, error_context
        )
        alternatives.extend(restructuring_alternatives)

        # Analyze feasibility and cost-benefit for each alternative
        for alternative in alternatives:
            self._analyze_alternative_feasibility(
                alternative, failed_conversion, error_context
            )

        # Sort by feasibility score
        alternatives.sort(key=lambda a: a.feasibility_score, reverse=True)

        # Cache results
        with self._lock:
            self.pathway_cache[cache_key] = alternatives
            self.stats["alternatives_found"] += len(alternatives)

        logger.info(
            f"Found {len(alternatives)} alternative pathways",
            top_feasibility=alternatives[0].feasibility_score if alternatives else 0.0,
        )

        return alternatives[:max_alternatives]

    def _find_intermediate_format_alternatives(
        self, failed_conversion: ConversionRequest
    ) -> List[AlternativePathway]:
        """Find alternatives using different intermediate formats."""
        alternatives = []

        # Get all possible formats as intermediates
        potential_intermediates = [
            fmt
            for fmt in DataFormat
            if fmt
            not in [failed_conversion.source_format, failed_conversion.target_format]
        ]

        for intermediate_format in potential_intermediates:
            # Check if source -> intermediate -> target path exists
            source_to_intermediate = self._check_conversion_feasibility(
                failed_conversion.source_format, intermediate_format
            )

            if not source_to_intermediate:
                continue

            intermediate_to_target = self._check_conversion_feasibility(
                intermediate_format, failed_conversion.target_format
            )

            if not intermediate_to_target:
                continue

            # Create alternative pathway
            alternative = AlternativePathway(
                description=f"Use {intermediate_format.value} as intermediate format",
                alternative_formats=[intermediate_format],
                required_changes=[
                    f"Add conversion step: {failed_conversion.source_format.value} -> {intermediate_format.value}",
                    f"Add conversion step: {intermediate_format.value} -> {failed_conversion.target_format.value}",
                ],
                implementation_complexity="medium",
            )

            alternatives.append(alternative)

        return alternatives

    def _find_alternative_target_formats(
        self, failed_conversion: ConversionRequest
    ) -> List[AlternativePathway]:
        """Find alternative target formats with similar analytical capabilities."""
        alternatives = []

        # Map formats to their analytical capabilities
        capability_groups = {
            "tabular_analysis": [
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.CSV,
                DataFormat.PARQUET,
            ],
            "numerical_computation": [DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE],
            "statistical_analysis": [
                DataFormat.STATISTICAL_RESULT,
                DataFormat.PANDAS_DATAFRAME,
            ],
            "time_series_analysis": [
                DataFormat.TIME_SERIES,
                DataFormat.PANDAS_DATAFRAME,
            ],
            "machine_learning": [
                DataFormat.NUMPY_ARRAY,
                DataFormat.SCIPY_SPARSE,
                DataFormat.PANDAS_DATAFRAME,
            ],
        }

        # Find capability group of target format
        target_capabilities = []
        for capability, formats in capability_groups.items():
            if failed_conversion.target_format in formats:
                target_capabilities.append(capability)

        # Find alternative formats with similar capabilities
        alternative_targets = set()
        for capability in target_capabilities:
            alternative_targets.update(capability_groups.get(capability, []))

        # Remove original target format
        alternative_targets.discard(failed_conversion.target_format)

        for alt_format in alternative_targets:
            # Check if conversion is feasible
            if self._check_conversion_feasibility(
                failed_conversion.source_format, alt_format
            ):
                alternative = AlternativePathway(
                    description=f"Use {alt_format.value} instead of {failed_conversion.target_format.value}",
                    alternative_formats=[alt_format],
                    required_changes=[
                        f"Change target format to {alt_format.value}",
                        "Update downstream pipeline to handle new format",
                    ],
                    implementation_complexity="low",
                )
                alternatives.append(alternative)

        return alternatives

    def _find_source_preprocessing_alternatives(
        self, failed_conversion: ConversionRequest
    ) -> List[AlternativePathway]:
        """Find alternatives involving source data preprocessing."""
        alternatives = []

        # Common preprocessing alternatives based on source format
        preprocessing_options = {
            DataFormat.PANDAS_DATAFRAME: [
                ("sparse_representation", "Convert to sparse matrix representation"),
                ("chunked_processing", "Process data in chunks to reduce memory usage"),
                ("column_subset", "Use only essential columns for conversion"),
                (
                    "data_type_optimization",
                    "Optimize data types to reduce memory usage",
                ),
            ],
            DataFormat.NUMPY_ARRAY: [
                ("dtype_conversion", "Convert to more memory-efficient data types"),
                ("array_reshaping", "Reshape array for better compatibility"),
                ("sparse_conversion", "Convert to sparse array if appropriate"),
            ],
            DataFormat.CSV: [
                ("streaming_reader", "Use streaming CSV reader for large files"),
                ("column_selection", "Read only required columns"),
                ("data_type_specification", "Specify data types during reading"),
            ],
        }

        source_options = preprocessing_options.get(failed_conversion.source_format, [])

        for option_id, description in source_options:
            alternative = AlternativePathway(
                description=f"Preprocess source data: {description}",
                alternative_formats=[
                    failed_conversion.source_format
                ],  # Same format, preprocessed
                required_changes=[
                    f"Add preprocessing step: {description}",
                    "Modify source data handling in pipeline",
                ],
                implementation_complexity="low"
                if "streaming" not in option_id
                else "medium",
            )
            alternatives.append(alternative)

        return alternatives

    def _find_restructuring_alternatives(
        self, failed_conversion: ConversionRequest, error_context: ErrorContext
    ) -> List[AlternativePathway]:
        """Find alternatives involving pipeline restructuring."""
        alternatives = []

        # Restructuring strategies based on error type
        if error_context.error_type == ConversionError.Type.MEMORY_EXCEEDED:
            alternatives.append(
                AlternativePathway(
                    description="Implement streaming-based conversion pipeline",
                    required_changes=[
                        "Break conversion into streaming chunks",
                        "Implement memory-efficient intermediate storage",
                        "Add progress tracking and resumption capability",
                    ],
                    implementation_complexity="high",
                    estimated_effort=8.0,
                )
            )

        elif error_context.error_type == ConversionError.Type.TIMEOUT:
            alternatives.append(
                AlternativePathway(
                    description="Implement asynchronous conversion pipeline",
                    required_changes=[
                        "Convert to asynchronous processing",
                        "Add job queue for conversion tasks",
                        "Implement result caching and retrieval",
                    ],
                    implementation_complexity="high",
                    estimated_effort=12.0,
                )
            )

        elif error_context.error_type == ConversionError.Type.ADAPTER_NOT_FOUND:
            alternatives.append(
                AlternativePathway(
                    description="Implement custom adapter for specific conversion",
                    required_changes=[
                        "Develop custom conversion adapter",
                        "Register adapter in shim registry",
                        "Add unit tests and validation",
                    ],
                    implementation_complexity="high",
                    estimated_effort=16.0,
                )
            )

        return alternatives

    def _check_conversion_feasibility(
        self, source_format: DataFormat, target_format: DataFormat
    ) -> bool:
        """Check if conversion between formats is feasible."""
        # This would typically use the compatibility matrix
        # For now, implement basic feasibility rules

        # Direct format compatibility
        if source_format == target_format:
            return True

        # Tabular data conversions
        tabular_formats = {
            DataFormat.PANDAS_DATAFRAME,
            DataFormat.CSV,
            DataFormat.PARQUET,
        }
        if source_format in tabular_formats and target_format in tabular_formats:
            return True

        # Numerical data conversions
        numerical_formats = {DataFormat.NUMPY_ARRAY, DataFormat.SCIPY_SPARSE}
        if source_format in numerical_formats and target_format in numerical_formats:
            return True

        # DataFrame to numerical
        if (
            source_format == DataFormat.PANDAS_DATAFRAME
            and target_format in numerical_formats
        ):
            return True

        # Assume other conversions may be possible through adapters
        return True

    def _analyze_alternative_feasibility(
        self,
        alternative: AlternativePathway,
        failed_conversion: ConversionRequest,
        error_context: ErrorContext,
    ) -> None:
        """Analyze feasibility and cost-benefit for an alternative pathway."""
        # Base feasibility score
        feasibility_score = 0.5

        # Adjust based on complexity
        complexity_adjustments = {"low": 0.3, "medium": 0.0, "high": -0.2}
        feasibility_score += complexity_adjustments.get(
            alternative.implementation_complexity, 0.0
        )

        # Adjust based on error type compatibility
        error_type_compatibility = {
            ConversionError.Type.MEMORY_EXCEEDED: {
                "sparse": 0.4,
                "streaming": 0.5,
                "chunked": 0.4,
            },
            ConversionError.Type.TIMEOUT: {
                "asynchronous": 0.3,
                "chunked": 0.2,
                "streaming": 0.2,
            },
            ConversionError.Type.ADAPTER_NOT_FOUND: {
                "custom": 0.2,
                "intermediate": 0.3,
                "alternative": 0.4,
            },
        }

        error_adjustments = error_type_compatibility.get(error_context.error_type, {})
        for keyword, adjustment in error_adjustments.items():
            if keyword in alternative.description.lower():
                feasibility_score += adjustment
                break

        # Clamp score to valid range
        alternative.feasibility_score = max(0.0, min(1.0, feasibility_score))

        # Calculate cost-benefit ratio (simplified)
        estimated_cost = alternative.estimated_effort * 10  # Cost in arbitrary units
        estimated_benefit = feasibility_score * 100  # Benefit in arbitrary units
        alternative.cost_benefit_ratio = estimated_benefit / max(estimated_cost, 1)

        # Estimate quality loss (simplified)
        quality_loss_factors = {
            "sparse": 0.05,
            "chunked": 0.02,
            "preprocessing": 0.01,
            "alternative_format": 0.10,
            "custom": 0.00,
        }

        quality_loss = 0.0
        for keyword, loss in quality_loss_factors.items():
            if keyword in alternative.description.lower():
                quality_loss = max(quality_loss, loss)

        alternative.expected_quality_loss = quality_loss

    def _generate_pathway_cache_key(
        self, conversion_request: ConversionRequest, error_context: ErrorContext
    ) -> str:
        """Generate cache key for pathway analysis."""
        return f"{conversion_request.source_format.value}_{conversion_request.target_format.value}_{error_context.error_type}"

    def record_pathway_success(
        self, pathway: AlternativePathway, success_metrics: Dict[str, Any]
    ) -> None:
        """Record successful use of an alternative pathway."""
        with self._lock:
            self.stats["successful_alternatives"] += 1

            pattern_key = (
                f"{pathway.alternative_formats}_{pathway.implementation_complexity}"
            )
            self.success_patterns[pattern_key].append(
                {
                    "pathway_id": pathway.pathway_id,
                    "feasibility_score": pathway.feasibility_score,
                    "success_metrics": success_metrics,
                    "timestamp": time.time(),
                }
            )

        logger.info(
            "Alternative pathway success recorded",
            pathway_id=pathway.pathway_id,
            feasibility_score=pathway.feasibility_score,
        )

    def get_pathway_statistics(self) -> Dict[str, Any]:
        """Get alternative pathway analysis statistics."""
        with self._lock:
            return {
                **self.stats,
                "cache_size": len(self.pathway_cache),
                "success_patterns": {
                    k: len(v) for k, v in self.success_patterns.items()
                },
                "cache_hit_rate": self.stats["cache_hits"]
                / max(self.stats["pathways_analyzed"], 1),
            }

    def clear_pathway_cache(self) -> None:
        """Clear pathway analysis cache."""
        with self._lock:
            self.pathway_cache.clear()
            self._pathway_cache.clear()
            logger.info("Alternative pathway cache cleared")

    # Enhanced pathway discovery methods

    def find_alternative_pathways_enhanced(
        self, failed_request: ConversionRequest
    ) -> List[ConversionPath]:
        """
        Find alternative conversion pathways for a failed conversion (enhanced version).

        Args:
            failed_request: Original conversion request that failed

        Returns:
            List of alternative conversion pathways, sorted by viability
        """
        start_time = time.time()

        source_format = failed_request.source_format
        target_format = failed_request.target_format

        # Check cache first
        cache_key = f"{source_format.value}->{target_format.value}"
        if self.enable_pathway_caching and cache_key in self._pathway_cache:
            cached_pathways = self._pathway_cache[cache_key]
            logger.debug(
                f"Using cached pathways for {cache_key}: {len(cached_pathways)} found"
            )
            return cached_pathways

        logger.info(
            f"Finding alternative pathways: {source_format.value} -> {target_format.value}"
        )

        # Find pathways using graph search
        pathways = self._discover_pathways_bfs(
            source_format, target_format, failed_request
        )

        # Evaluate and sort pathways
        evaluated_pathways = []
        for pathway in pathways:
            cost = self.assess_pathway_cost(pathway)
            quality = self.assess_quality_degradation(pathway)

            # Calculate overall viability score
            viability_score = self._calculate_viability_score(pathway, cost, quality)
            pathway.success_probability = viability_score

            if quality.expected_quality_score >= self.quality_threshold:
                evaluated_pathways.append(pathway)

        # Sort by viability (success probability)
        evaluated_pathways.sort(key=lambda p: p.success_probability, reverse=True)

        # Cache successful search
        if self.enable_pathway_caching:
            self._pathway_cache[cache_key] = evaluated_pathways

        discovery_time = time.time() - start_time
        logger.info(
            f"Found {len(evaluated_pathways)} alternative pathways in {discovery_time:.2f}s"
        )

        return evaluated_pathways

    def assess_pathway_cost(self, pathway: ConversionPath) -> PathwayCost:
        """
        Assess the computational cost of a conversion pathway.

        Args:
            pathway: Conversion pathway to assess

        Returns:
            Detailed cost assessment
        """
        total_computational_cost = 0.0
        total_time_overhead = 0.0
        total_memory_overhead = 0.0
        min_reliability = 1.0

        # Sum costs across all steps
        for step in pathway.steps:
            step_cost = step.estimated_cost
            total_computational_cost += step_cost.computational_cost
            total_time_overhead += step_cost.time_estimate_seconds
            total_memory_overhead += step_cost.memory_cost_mb
            min_reliability = min(min_reliability, step.confidence)

        # Calculate quality degradation (increases with pathway length)
        quality_degradation = min(len(pathway.steps) * 0.05, 0.3)  # Max 30% degradation

        # Confidence decreases with pathway complexity
        confidence = max(min_reliability * (0.95 ** len(pathway.steps)), 0.1)

        return PathwayCost(
            computational_cost=total_computational_cost,
            quality_degradation=quality_degradation,
            time_overhead=total_time_overhead,
            memory_overhead=total_memory_overhead,
            reliability_score=min_reliability,
            confidence=confidence,
        )

    def assess_quality_degradation(self, pathway: ConversionPath) -> QualityAssessment:
        """
        Assess quality degradation for a conversion pathway.

        Args:
            pathway: Conversion pathway to assess

        Returns:
            Quality assessment with degradation analysis
        """
        # Base quality starts high and degrades with each conversion step
        base_quality = 1.0
        cumulative_degradation = 0.0
        metadata_preservation = 1.0
        data_fidelity = 1.0
        risk_factors = []

        for i, step in enumerate(pathway.steps):
            # Each step introduces some quality loss
            step_degradation = 0.02 + (0.01 * i)  # Increasing degradation per step
            cumulative_degradation += step_degradation

            # Specific format conversions have known quality impacts
            degradation_factor = self._get_format_degradation_factor(
                step.source_format, step.target_format
            )
            cumulative_degradation += degradation_factor

            # Metadata preservation decreases with conversions
            if self._loses_metadata(step.source_format, step.target_format):
                metadata_preservation *= 0.9
                risk_factors.append(
                    f"Metadata loss in {step.source_format.value} -> {step.target_format.value}"
                )

            # Data fidelity assessment
            if self._loses_precision(step.source_format, step.target_format):
                data_fidelity *= 0.95
                risk_factors.append(
                    f"Precision loss in {step.source_format.value} -> {step.target_format.value}"
                )

        expected_quality = max(base_quality - cumulative_degradation, 0.1)

        return QualityAssessment(
            expected_quality_score=expected_quality,
            quality_degradation=cumulative_degradation,
            metadata_preservation=metadata_preservation,
            data_fidelity=data_fidelity,
            risk_factors=risk_factors,
        )

    def cache_successful_pathway(self, pathway: ConversionPath) -> None:
        """
        Cache a successful conversion pathway for future reuse.

        Args:
            pathway: Successfully executed conversion pathway
        """
        if not self.enable_pathway_caching:
            return

        with self._lock:
            cache_key = f"{pathway.source_format.value}->{pathway.target_format.value}"

            # Store successful pathway
            self._successful_pathways[pathway.path_id] = pathway

            # Update success rate
            current_rate = self._pathway_success_rates.get(cache_key, 0.0)
            self._pathway_success_rates[cache_key] = min(current_rate + 0.1, 1.0)

            # Update cache with this successful pathway prioritized
            if cache_key in self._pathway_cache:
                cached_pathways = self._pathway_cache[cache_key]
                # Move successful pathway to front if it exists
                for i, cached_pathway in enumerate(cached_pathways):
                    if cached_pathway.path_id == pathway.path_id:
                        cached_pathways.insert(0, cached_pathways.pop(i))
                        break
                else:
                    # Add new successful pathway to front
                    cached_pathways.insert(0, pathway)
            else:
                self._pathway_cache[cache_key] = [pathway]

            logger.info(f"Cached successful pathway {pathway.path_id} for {cache_key}")

    def _discover_pathways_bfs(
        self,
        source_format: DataFormat,
        target_format: DataFormat,
        original_request: ConversionRequest,
    ) -> List[ConversionPath]:
        """Discover pathways using breadth-first search."""

        if not self.registry:
            logger.warning("No registry available for pathway discovery")
            return []

        pathways = []
        visited = set()

        # BFS queue: (current_format, path_so_far, total_cost)
        queue = deque([(source_format, [], self._create_zero_cost())])

        while queue and len(pathways) < 10:  # Limit number of pathways
            current_format, path, current_cost = queue.popleft()

            # Skip if already visited this format in this path
            if current_format in [step.target_format for step in path]:
                continue

            # Skip if path is too deep
            if len(path) >= self.max_pathway_depth:
                continue

            # Check if we reached target
            if current_format == target_format and path:
                pathway = ConversionPath(
                    source_format=source_format,
                    target_format=target_format,
                    steps=path,
                    total_cost=current_cost,
                    success_probability=0.8 ** len(path),  # Decreases with steps
                )
                pathways.append(pathway)
                continue

            # Get available adapters from current format
            available_adapters = self._get_adapters_from_format(current_format)

            for adapter in available_adapters:
                supported_conversions = adapter.get_supported_conversions()

                for source_fmt, target_fmt in supported_conversions:
                    if source_fmt == current_format and target_fmt != current_format:
                        # Create dummy request for cost estimation
                        dummy_request = ConversionRequest(
                            source_data=None,
                            source_format=source_fmt,
                            target_format=target_fmt,
                            context=original_request.context,
                        )

                        try:
                            step_cost = adapter.estimate_cost(dummy_request)
                            confidence = adapter.can_convert(dummy_request)
                        except Exception:
                            # Use default values if estimation fails
                            step_cost = self._create_default_cost()
                            confidence = 0.5

                        new_step = ConversionStep(
                            adapter_id=adapter.adapter_id,
                            source_format=source_fmt,
                            target_format=target_fmt,
                            estimated_cost=step_cost,
                            confidence=confidence,
                        )

                        new_path = path + [new_step]
                        new_cost = self._add_costs(current_cost, step_cost)

                        queue.append((target_fmt, new_path, new_cost))

        return pathways

    def _get_adapters_from_format(
        self, format_type: DataFormat
    ) -> List[EnhancedShimAdapter]:
        """Get adapters that can convert from the given format."""
        if not self.registry:
            return []

        available_adapters = []
        for adapter in self.registry.get_active_adapters():
            supported_conversions = adapter.get_supported_conversions()
            for source_fmt, _ in supported_conversions:
                if source_fmt == format_type:
                    available_adapters.append(adapter)
                    break

        return available_adapters

    def _calculate_viability_score(
        self, pathway: ConversionPath, cost: PathwayCost, quality: QualityAssessment
    ) -> float:
        """Calculate overall viability score for pathway."""

        # Base score from quality
        quality_score = quality.expected_quality_score * 0.4

        # Reliability score
        reliability_score = cost.reliability_score * 0.3

        # Cost efficiency (inverse of computational cost, capped)
        cost_efficiency = max(1.0 - min(cost.computational_cost, 1.0), 0.1) * 0.2

        # Path simplicity (prefer shorter paths)
        simplicity_score = max(1.0 - (len(pathway.steps) - 1) * 0.1, 0.1) * 0.1

        return quality_score + reliability_score + cost_efficiency + simplicity_score

    def _get_format_degradation_factor(
        self, source: DataFormat, target: DataFormat
    ) -> float:
        """Get quality degradation factor for specific format conversions."""
        degradation_map = {
            # High precision to low precision
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_LIST): 0.05,
            (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_LIST): 0.03,
            (DataFormat.SCIPY_SPARSE, DataFormat.PANDAS_DATAFRAME): 0.02,
            # Complex to simple
            (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME): 0.04,
            (DataFormat.CATEGORICAL, DataFormat.NUMPY_ARRAY): 0.06,
            # Structured to unstructured
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_DICT): 0.01,
        }

        return degradation_map.get((source, target), 0.01)  # Default small degradation

    def _loses_metadata(self, source: DataFormat, target: DataFormat) -> bool:
        """Check if conversion loses metadata."""
        metadata_losing_conversions = {
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_LIST),
            (DataFormat.TIME_SERIES, DataFormat.NUMPY_ARRAY),
            (DataFormat.CATEGORICAL, DataFormat.NUMPY_ARRAY),
        }
        return (source, target) in metadata_losing_conversions

    def _loses_precision(self, source: DataFormat, target: DataFormat) -> bool:
        """Check if conversion loses numerical precision."""
        precision_losing_conversions = {
            (DataFormat.NUMPY_ARRAY, DataFormat.PYTHON_LIST),
            (DataFormat.SCIPY_SPARSE, DataFormat.PYTHON_LIST),
            (DataFormat.PANDAS_DATAFRAME, DataFormat.PYTHON_DICT),
        }
        return (source, target) in precision_losing_conversions

    def _create_zero_cost(self) -> ConversionCost:
        """Create zero-cost baseline."""
        return ConversionCost(
            computational_cost=0.0,
            memory_cost_mb=0.0,
            time_estimate_seconds=0.0,
            io_operations=0,
            network_operations=0,
            quality_impact=0.0,
        )

    def _create_default_cost(self) -> ConversionCost:
        """Create default cost estimate."""
        return ConversionCost(
            computational_cost=0.1,
            memory_cost_mb=10.0,
            time_estimate_seconds=1.0,
            io_operations=0,
            network_operations=0,
            quality_impact=0.02,
        )

    def _add_costs(
        self, cost1: ConversionCost, cost2: ConversionCost
    ) -> ConversionCost:
        """Add two conversion costs together."""
        return ConversionCost(
            computational_cost=cost1.computational_cost + cost2.computational_cost,
            memory_cost_mb=cost1.memory_cost_mb + cost2.memory_cost_mb,
            time_estimate_seconds=cost1.time_estimate_seconds
            + cost2.time_estimate_seconds,
            io_operations=cost1.io_operations + cost2.io_operations,
            network_operations=cost1.network_operations + cost2.network_operations,
            quality_impact=cost1.quality_impact + cost2.quality_impact,
        )

    def get_pathway_statistics(self) -> Dict[str, Any]:
        """Get pathway discovery and performance statistics."""
        with self._lock:
            return {
                "cached_pathways": len(self._pathway_cache),
                "successful_pathways": len(self._successful_pathways),
                "average_success_rates": dict(self._pathway_success_rates),
                "pathway_performance": dict(self._pathway_performance),
                **self.stats,
                "cache_size": len(self.pathway_cache),
                "success_patterns": {
                    k: len(v) for k, v in self.success_patterns.items()
                },
                "cache_hit_rate": self.stats["cache_hits"]
                / max(self.stats["pathways_analyzed"], 1),
            }
