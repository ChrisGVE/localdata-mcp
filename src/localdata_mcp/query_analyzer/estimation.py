"""Resource estimation functions for query analysis.

Provides standalone functions for estimating memory usage, token counts,
execution time, and fallback row estimation. These are extracted from
the QueryAnalyzer class to keep module sizes manageable.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..size_estimator import get_size_estimator
from ..token_manager import get_token_manager

logger = logging.getLogger(__name__)

# Memory risk thresholds (MB)
MEMORY_THRESHOLDS: Dict[str, int] = {
    "low": 10,  # < 10MB
    "medium": 50,  # 10-50MB
    "high": 200,  # 50-200MB
    "critical": 500,  # > 200MB
}

# Token risk thresholds
TOKEN_THRESHOLDS: Dict[str, int] = {
    "low": 1000,  # < 1K tokens
    "medium": 10000,  # 1K-10K tokens
    "high": 50000,  # 10K-50K tokens
    "critical": 100000,  # > 50K tokens
}

# Timeout risk thresholds (seconds)
TIMEOUT_THRESHOLDS: Dict[str, int] = {
    "low": 1,  # < 1 second
    "medium": 5,  # 1-5 seconds
    "high": 30,  # 5-30 seconds
    "critical": 60,  # > 30 seconds
}


def _classify_risk(value: float, thresholds: Dict[str, int]) -> str:
    """Classify a numeric value into a risk level using thresholds.

    Args:
        value: The numeric value to classify.
        thresholds: Mapping of risk level to threshold value.

    Returns:
        Risk level string: 'low', 'medium', 'high', or 'critical'.
    """
    risk_level = "low"
    for level, threshold in thresholds.items():
        if value > threshold:
            risk_level = level
    return risk_level


def estimate_memory_usage(
    row_count: int,
    sample_row: Optional[pd.Series],
    column_info: Dict[str, Any],
    *,
    engine: Optional[Engine] = None,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Estimate memory usage based on row count and sample data.

    When *engine* and *query* are provided the ``SizeEstimator`` is
    attempted first for a type-aware byte estimate.  The existing
    sample-based heuristic is used as a fallback.

    Args:
        row_count: Estimated number of rows.
        sample_row: Sample row data.
        column_info: Column metadata.
        engine: Optional SQLAlchemy engine for SizeEstimator.
        query: Optional SQL query for SizeEstimator.

    Returns:
        Dictionary with memory usage estimates.
    """
    # Try SizeEstimator for a type-aware estimate
    if engine is not None and query is not None:
        try:
            estimator = get_size_estimator(engine)
            est = estimator.estimate_result_size(query, estimated_rows=row_count)
            if est.estimated_bytes_per_row > 0:
                buffer_factor = 1.5
                mem_bytes = est.estimated_total_bytes * buffer_factor
                mem_mb = mem_bytes / (1024 * 1024)
                risk_level = _classify_risk(mem_mb, MEMORY_THRESHOLDS)
                return {
                    "row_size": est.estimated_bytes_per_row,
                    "total_memory": mem_mb,
                    "risk_level": risk_level,
                }
        except Exception as exc:
            logger.debug("SizeEstimator failed, falling back: %s", exc)

    # Fallback: sample-based heuristic
    if sample_row is None or row_count == 0:
        return {"row_size": 0, "total_memory": 0, "risk_level": "low"}

    # Calculate sample row size in bytes
    row_size_bytes = 0

    for col_name, value in sample_row.items():
        if pd.isna(value):
            row_size_bytes += 8  # NULL value overhead
        elif isinstance(value, str):
            row_size_bytes += len(value.encode("utf-8"))
        elif isinstance(value, (int, float)):
            row_size_bytes += 8  # Numeric types
        elif isinstance(value, bool):
            row_size_bytes += 1  # Boolean
        else:
            # Complex types - estimate based on string representation
            row_size_bytes += len(str(value).encode("utf-8"))

    # Add DataFrame overhead (index, column headers, etc.)
    overhead_per_row = 24  # Estimated pandas overhead per row
    total_row_size = row_size_bytes + overhead_per_row

    # Apply buffer factor (1.5x) for memory allocation overhead
    buffer_factor = 1.5
    estimated_memory_bytes = row_count * total_row_size * buffer_factor
    estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)

    risk_level = _classify_risk(estimated_memory_mb, MEMORY_THRESHOLDS)

    return {
        "row_size": total_row_size,
        "total_memory": estimated_memory_mb,
        "risk_level": risk_level,
    }


def estimate_token_count(
    row_count: int,
    sample_row: Optional[pd.Series],
    column_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate token count using enhanced TokenManager.

    Args:
        row_count: Estimated number of rows.
        sample_row: Sample row data.
        column_info: Column metadata.

    Returns:
        Dictionary with token count estimates.
    """
    if sample_row is None or row_count == 0:
        return {"tokens_per_row": 0, "total_tokens": 0, "risk_level": "low"}

    # Create DataFrame from sample for TokenManager
    sample_df = pd.DataFrame([sample_row])

    # Use TokenManager for intelligent estimation
    token_manager = get_token_manager()
    estimation = token_manager.estimate_tokens_for_query_result(row_count, sample_df)

    return {
        "tokens_per_row": estimation.tokens_per_row,
        "total_tokens": estimation.total_tokens,
        "risk_level": estimation.risk_level,
    }


def estimate_execution_time(
    row_count: int,
    complexity_analysis: Dict[str, Any],
    engine: Engine,
) -> Dict[str, Any]:
    """Estimate query execution time based on complexity and row count.

    Args:
        row_count: Estimated number of rows.
        complexity_analysis: Query complexity metadata.
        engine: Database engine connection.

    Returns:
        Dictionary with execution time estimates.
    """
    # Base time estimation (very rough heuristics)
    base_time = 0.001  # 1ms base time

    # Time per row based on complexity
    if complexity_analysis["score"] <= 3:
        time_per_1k_rows = 0.01  # Simple queries
    elif complexity_analysis["score"] <= 6:
        time_per_1k_rows = 0.05  # Medium complexity
    else:
        time_per_1k_rows = 0.1  # High complexity

    # Additional time for specific features
    complexity_multiplier = 1.0

    if complexity_analysis["has_joins"]:
        complexity_multiplier *= 1.5

    if complexity_analysis["has_aggregations"]:
        complexity_multiplier *= 1.3

    if complexity_analysis["has_subqueries"]:
        complexity_multiplier *= 1.4

    if complexity_analysis["has_window_functions"]:
        complexity_multiplier *= 1.6

    # Calculate estimated time
    estimated_time = (
        base_time + (row_count / 1000.0) * time_per_1k_rows * complexity_multiplier
    )

    risk_level = _classify_risk(estimated_time, TIMEOUT_THRESHOLDS)

    return {"estimated_time": estimated_time, "risk_level": risk_level}


def fallback_row_estimation(query: str, engine: Engine) -> int:
    """Fallback row estimation when COUNT(*) fails.

    Args:
        query: SQL query.
        engine: Database engine connection.

    Returns:
        Estimated row count (conservative estimate).
    """
    try:
        # Try EXPLAIN if supported
        explain_query = f"EXPLAIN {query}"
        with engine.connect() as conn:
            result = conn.execute(text(explain_query))
            # This is database-specific - for now, return conservative estimate
            return 1000  # Conservative default

    except Exception:
        # Last resort: return very conservative estimate
        logger.warning("All row estimation methods failed, using conservative default")
        return 100
