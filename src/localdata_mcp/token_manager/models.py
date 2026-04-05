"""Data models and constants for the token management system.

This module defines the dataclasses used for token estimation results,
chunking recommendations, and response metadata, along with model
context window sizes and risk threshold constants.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenEstimation:
    """Complete token estimation results with rich metadata."""

    # Core estimation
    total_tokens: int
    tokens_per_row: float
    confidence: float  # 0.0-1.0 based on sample quality

    # Column analysis
    numeric_columns: List[str]
    text_columns: List[str]
    other_columns: List[str]
    column_token_breakdown: Dict[str, int]

    # Performance metadata
    estimation_method: str  # 'full', 'sampled', 'extrapolated'
    sample_size: int
    total_rows: int

    # JSON overhead
    json_overhead_per_row: int
    json_overhead_total: int

    # Risk assessment
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    memory_risk: str  # 'low', 'medium', 'high'

    # Context window analysis
    fits_in_context: Dict[str, bool] = field(default_factory=dict)  # model -> fits
    recommended_chunk_size: Optional[int] = None


@dataclass
class ChunkingRecommendation:
    """Chunking strategy recommendations for large responses."""

    should_chunk: bool
    recommended_chunk_size: int
    estimated_chunks: int
    chunk_overlap_rows: int
    strategy: str  # 'row_based', 'column_based', 'mixed'

    # Metadata for LLM
    chunk_size_rationale: str
    performance_impact: str
    memory_benefits: str


@dataclass
class ResponseMetadata:
    """Rich metadata provided to LLMs for intelligent decision-making."""

    # Size estimates
    estimated_tokens: int
    estimated_memory_mb: float
    response_size_category: str  # 'small', 'medium', 'large', 'xlarge'

    # Data characteristics
    row_count: int
    column_count: int
    data_density: str  # 'sparse', 'moderate', 'dense'
    text_heavy: bool

    # Processing recommendations
    chunking_recommendation: Optional[ChunkingRecommendation]
    streaming_recommended: bool
    sampling_options: Dict[str, Any]

    # Context window compatibility
    model_compatibility: Dict[str, Dict[str, Any]]

    # Performance indicators
    processing_complexity: str  # 'low', 'medium', 'high'
    estimated_response_time: float


# Context window sizes for popular models
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 16385,
    "claude-3-haiku": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3.5-sonnet": 200000,
    "gemini-pro": 32768,
    "default": 8192,
}

# Token thresholds for risk assessment
TOKEN_RISK_THRESHOLDS: Dict[str, int] = {
    "low": 1000,
    "medium": 10000,
    "high": 50000,
    "critical": 100000,
}
