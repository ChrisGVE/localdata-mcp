"""Data models for response metadata.

Defines enums for data quality and query complexity levels, and dataclasses for
statistical summaries, schema information, chunk availability, data quality metrics,
and enhanced response metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..token_manager import TokenEstimation


class DataQualityLevel(Enum):
    """Data quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """Query complexity levels for processing estimation."""

    SIMPLE = "simple"  # Basic SELECT, small results
    MODERATE = "moderate"  # JOINs, aggregations, medium results
    COMPLEX = "complex"  # Multi-table JOINs, subqueries, large results
    INTENSIVE = "intensive"  # Complex analytics, very large results


@dataclass
class StatisticalSummary:
    """Statistical summary for a dataset or column."""

    # Basic statistics
    total_rows: int
    non_null_rows: int
    null_percentage: float

    # Data type distribution
    data_types: Dict[str, int]  # dtype -> count of columns

    # Numeric column statistics (if any numeric columns exist)
    numeric_summary: Optional[Dict[str, Any]] = None

    # Text column statistics (if any text columns exist)
    text_summary: Optional[Dict[str, Any]] = None

    # Sample data preview
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)

    # Data quality indicators
    duplicate_rows: int = 0
    duplicate_percentage: float = 0.0

    # Memory and storage estimates
    estimated_memory_usage_mb: float = 0.0


@dataclass
class SchemaInformation:
    """Detailed schema information for data understanding."""

    # Column information
    columns: List[Dict[str, Any]]  # name, dtype, nullable, sample_values
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)

    # Index information
    indexes: List[Dict[str, Any]] = field(default_factory=list)

    # Constraints
    constraints: List[Dict[str, Any]] = field(default_factory=list)

    # Table metadata
    table_name: Optional[str] = None
    last_modified: Optional[float] = None


@dataclass
class ChunkAvailability:
    """Information about available data chunks for progressive loading."""

    total_chunks: int
    available_chunks: List[int]  # Which chunks are ready/cached
    chunk_size: int  # Rows per chunk
    chunk_overlap: int = 0

    # Chunk metadata
    chunk_metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Loading status
    loading_status: Dict[int, str] = field(default_factory=dict)  # chunk_id -> status
    estimated_load_time: Dict[int, float] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality assessment."""

    overall_quality: DataQualityLevel
    quality_score: float  # 0.0 to 1.0

    # Specific quality dimensions
    completeness: float  # Percentage of non-null values
    consistency: float  # Data format consistency
    validity: float  # Values within expected ranges/formats
    accuracy: float  # Estimated accuracy (based on patterns)

    # Quality issues detected
    issues: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class EnhancedResponseMetadata:
    """Comprehensive response metadata for intelligent LLM decision-making."""

    # Basic response information
    query_id: str
    timestamp: float

    # Enhanced size and complexity estimates
    query_complexity_score: float  # 0.0 to 1.0
    query_complexity_level: QueryComplexity
    estimated_processing_time: float  # Seconds
    memory_footprint: float  # MB

    # Token and response size analysis (from existing TokenManager)
    token_estimation: TokenEstimation

    # Data characteristics and quality
    statistical_summary: StatisticalSummary
    data_quality_metrics: DataQualityMetrics
    schema_information: SchemaInformation

    # Progressive loading capabilities
    chunk_availability: ChunkAvailability
    supports_streaming: bool
    supports_cancellation: bool

    # Caching information
    is_cached: bool = False
    cache_key: Optional[str] = None
    cache_expiry: Optional[float] = None

    # LLM guidance
    recommended_action: str = "proceed"  # proceed, chunk, sample, cancel
    action_rationale: str = ""
    llm_friendly_summary: str = ""

    # Structured error classification (populated on error)
    error_classification: Optional[Dict[str, Any]] = None
