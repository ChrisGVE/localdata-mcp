"""
Data types and structures for pipeline analysis.

Enums, dataclasses, and type definitions used across the pipeline analyzer,
shim injector, and pipeline validator components.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces import DataFormat, ConversionCost, ConversionPath
from ..shim_registry import EnhancedShimAdapter


class AnalysisType(Enum):
    """Types of pipeline analysis that can be performed."""

    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COST = "cost"
    COMPLETE = "complete"


class InjectionStrategy(Enum):
    """Strategies for shim injection."""

    MINIMAL = "minimal"  # Insert minimum necessary shims
    OPTIMAL = "optimal"  # Insert shims for best performance
    SAFE = "safe"  # Insert shims for maximum compatibility
    BALANCED = "balanced"  # Balance between performance and safety


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
    alternative_shims: List[Tuple[EnhancedShimAdapter, float]] = field(
        default_factory=list
    )


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
