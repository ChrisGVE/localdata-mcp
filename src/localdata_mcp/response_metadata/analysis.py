"""Data analysis helpers for response metadata generation.

Provides functions for statistical summarization, schema extraction,
query complexity scoring, processing estimates, and LLM recommendations.
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..query_analyzer import QueryAnalysis
from ..token_manager import TokenEstimation
from .models import (
    ChunkAvailability,
    DataQualityLevel,
    DataQualityMetrics,
    QueryComplexity,
    SchemaInformation,
    StatisticalSummary,
)

# Re-export for backward compatibility so generator.py can import from here
from .quality import assess_data_quality  # noqa: F401


def generate_statistical_summary(df: pd.DataFrame) -> StatisticalSummary:
    """Generate comprehensive statistical summary for a DataFrame.

    Args:
        df: DataFrame to summarize.

    Returns:
        Statistical summary dataclass.
    """
    if df.empty:
        return StatisticalSummary(
            total_rows=0, non_null_rows=0, null_percentage=0.0, data_types={}
        )

    total_rows = len(df)
    non_null_rows = len(df.dropna())
    null_percentage = (
        ((total_rows - non_null_rows) / total_rows * 100) if total_rows > 0 else 0.0
    )

    data_types = df.dtypes.value_counts().to_dict()
    data_types = {str(k): int(v) for k, v in data_types.items()}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = None
    if numeric_cols:
        numeric_summary = {
            "columns": numeric_cols,
            "statistics": df[numeric_cols].describe().to_dict(),
        }

    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    text_summary = None
    if text_cols:
        text_summary = {
            "columns": text_cols,
            "avg_length": {
                col: df[col].astype(str).str.len().mean() for col in text_cols[:5]
            },
            "unique_values": {col: df[col].nunique() for col in text_cols[:5]},
        }

    sample_rows = df.head(5).to_dict(orient="records")
    duplicate_rows = len(df) - len(df.drop_duplicates())
    duplicate_percentage = (duplicate_rows / len(df) * 100) if len(df) > 0 else 0.0
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return StatisticalSummary(
        total_rows=total_rows,
        non_null_rows=non_null_rows,
        null_percentage=null_percentage,
        data_types=data_types,
        numeric_summary=numeric_summary,
        text_summary=text_summary,
        sample_rows=sample_rows,
        duplicate_rows=duplicate_rows,
        duplicate_percentage=duplicate_percentage,
        estimated_memory_usage_mb=memory_mb,
    )


def generate_schema_information(
    df: pd.DataFrame, db_name: Optional[str] = None
) -> SchemaInformation:
    """Generate detailed schema information from a DataFrame.

    Args:
        df: DataFrame to inspect.
        db_name: Optional database/table name.

    Returns:
        Schema information dataclass.
    """
    columns = []

    for col in df.columns:
        col_info = {
            "name": str(col),
            "dtype": str(df[col].dtype),
            "nullable": df[col].isnull().any(),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().astype(str).head(3).tolist(),
        }
        columns.append(col_info)

    return SchemaInformation(columns=columns, table_name=db_name)


def assess_query_complexity(
    query: str,
    df: pd.DataFrame,
    query_analysis: Optional[QueryAnalysis] = None,
) -> Tuple[QueryComplexity, float]:
    """Assess query complexity level and score.

    Args:
        query: SQL query string.
        df: Result DataFrame (used for row-count factor).
        query_analysis: Optional pre-computed query analysis.

    Returns:
        Tuple of (complexity level enum, numeric score 0.0-1.0).
    """
    complexity_score = 0.0
    query_lower = query.lower()

    row_factor = min(0.4, len(df) / 100000)
    complexity_score += row_factor

    pattern_score = 0.0
    if "join" in query_lower:
        pattern_score += 0.2
    if query_lower.count("join") > 1:
        pattern_score += 0.1
    if "group by" in query_lower:
        pattern_score += 0.1
    if "order by" in query_lower:
        pattern_score += 0.05
    if "having" in query_lower:
        pattern_score += 0.1
    if any(keyword in query_lower for keyword in ["window", "partition", "over"]):
        pattern_score += 0.15

    complexity_score += min(0.6, pattern_score)

    if complexity_score < 0.2:
        complexity_level = QueryComplexity.SIMPLE
    elif complexity_score < 0.5:
        complexity_level = QueryComplexity.MODERATE
    elif complexity_score < 0.8:
        complexity_level = QueryComplexity.COMPLEX
    else:
        complexity_level = QueryComplexity.INTENSIVE

    return complexity_level, complexity_score


def estimate_processing_time(
    df: pd.DataFrame,
    complexity_level: QueryComplexity,
    token_estimation: TokenEstimation,
) -> float:
    """Estimate processing time in seconds.

    Args:
        df: Result DataFrame.
        complexity_level: Assessed query complexity.
        token_estimation: Token estimation from the token manager.

    Returns:
        Estimated processing time in seconds.
    """
    base_time = 0.1
    size_time = len(df) * 0.0001

    complexity_multipliers = {
        QueryComplexity.SIMPLE: 1.0,
        QueryComplexity.MODERATE: 2.0,
        QueryComplexity.COMPLEX: 4.0,
        QueryComplexity.INTENSIVE: 8.0,
    }

    complexity_time = size_time * complexity_multipliers[complexity_level]
    token_time = token_estimation.total_tokens * 0.000001

    return round(base_time + complexity_time + token_time, 3)


def calculate_memory_footprint(
    df: pd.DataFrame, token_estimation: TokenEstimation
) -> float:
    """Calculate memory footprint in MB.

    Args:
        df: Result DataFrame.
        token_estimation: Token estimation from the token manager.

    Returns:
        Estimated memory usage in megabytes.
    """
    df_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    token_memory = token_estimation.total_tokens * 4 / (1024 * 1024)

    return round(df_memory + token_memory, 2)


def generate_chunk_availability(
    df: pd.DataFrame, token_estimation: TokenEstimation
) -> ChunkAvailability:
    """Generate chunk availability information.

    Args:
        df: Result DataFrame.
        token_estimation: Token estimation from the token manager.

    Returns:
        Chunk availability dataclass.
    """
    chunk_size = token_estimation.recommended_chunk_size or 1000
    total_chunks = math.ceil(len(df) / chunk_size) if chunk_size > 0 else 1
    available_chunks = list(range(total_chunks))

    return ChunkAvailability(
        total_chunks=total_chunks,
        available_chunks=available_chunks,
        chunk_size=chunk_size,
    )


def determine_recommended_action(
    token_estimation: TokenEstimation,
    data_quality: DataQualityMetrics,
    complexity_level: QueryComplexity,
    row_count: int,
) -> Tuple[str, str]:
    """Determine recommended action for LLM.

    Args:
        token_estimation: Token estimation from the token manager.
        data_quality: Assessed data quality metrics.
        complexity_level: Assessed query complexity.
        row_count: Number of rows in the result.

    Returns:
        Tuple of (action string, rationale string).
    """
    if token_estimation.total_tokens > 100000:
        return (
            "chunk",
            f"Dataset is very large ({token_estimation.total_tokens:,} tokens). "
            "Recommend chunking for better handling.",
        )

    if token_estimation.total_tokens > 20000:
        return (
            "sample",
            f"Dataset is large ({token_estimation.total_tokens:,} tokens). "
            "Consider sampling or chunking.",
        )

    if data_quality.overall_quality == DataQualityLevel.POOR:
        return (
            "review",
            f"Data quality is {data_quality.overall_quality.value} "
            f"(score: {data_quality.quality_score:.2f}). Review before processing.",
        )

    if complexity_level == QueryComplexity.INTENSIVE:
        return (
            "stream",
            f"Query complexity is {complexity_level.value}. "
            "Consider streaming processing.",
        )

    return (
        "proceed",
        f"Dataset is manageable ({token_estimation.total_tokens:,} tokens, "
        f"{row_count:,} rows). Safe to proceed.",
    )


def generate_llm_friendly_summary(
    df: pd.DataFrame,
    token_estimation: TokenEstimation,
    data_quality: DataQualityMetrics,
    complexity_level: QueryComplexity,
) -> str:
    """Generate a human-readable summary for LLM understanding.

    Args:
        df: Result DataFrame.
        token_estimation: Token estimation from the token manager.
        data_quality: Assessed data quality metrics.
        complexity_level: Assessed query complexity.

    Returns:
        Human-readable summary string.
    """
    size_desc = "small" if len(df) < 1000 else "medium" if len(df) < 10000 else "large"
    quality_desc = data_quality.overall_quality.value
    complexity_desc = complexity_level.value

    summary = (
        f"This is a {size_desc} dataset with {len(df):,} rows and {len(df.columns)} columns. "
        f"Data quality is {quality_desc} with {data_quality.quality_score:.1%} quality score. "
        f"Query complexity is {complexity_desc}. "
        f"Estimated response size: {token_estimation.total_tokens:,} tokens "
        f"({token_estimation.total_tokens / 1000:.1f}K). "
    )

    if token_estimation.total_tokens > 10000:
        summary += "Consider chunking or sampling for optimal processing. "

    if data_quality.recommendations:
        summary += (
            f"Data recommendations: {'; '.join(data_quality.recommendations[:2])}."
        )

    return summary
