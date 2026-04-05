"""Data models for the query analysis system.

Contains the QueryAnalysis dataclass that holds all analysis results
including resource estimates, complexity metrics, and recommendations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class QueryAnalysis:
    """Results of query analysis containing resource estimates and metadata."""

    # Basic query information
    query: str
    query_hash: str
    validated_query: str

    # Row count analysis from COUNT(*)
    estimated_rows: int
    count_query_time: float

    # Sample analysis from LIMIT 1
    sample_row: Optional[pd.Series]
    sample_query_time: float
    column_count: int
    column_types: Dict[str, str]

    # Memory estimation
    estimated_row_size_bytes: float
    estimated_total_memory_mb: float
    memory_risk_level: str  # 'low', 'medium', 'high', 'critical'

    # Token estimation
    estimated_tokens_per_row: int
    estimated_total_tokens: int
    token_risk_level: str  # 'low', 'medium', 'high', 'critical'

    # Timeout estimation
    estimated_execution_time_seconds: float
    timeout_risk_level: str  # 'low', 'medium', 'high', 'critical'

    # Query complexity analysis
    complexity_score: int  # 1-10 scale
    has_joins: bool
    has_aggregations: bool
    has_subqueries: bool
    has_window_functions: bool

    # Recommendations
    recommendations: List[str]
    should_chunk: bool
    recommended_chunk_size: Optional[int]

    # Analysis metadata
    analysis_time_seconds: float
    timestamp: float
