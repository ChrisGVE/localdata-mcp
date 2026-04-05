"""Token estimation helper functions for column analysis and assessment.

This module contains the private helper methods extracted from TokenManager
that handle column type analysis, text token estimation, risk assessment,
chunking recommendations, and model compatibility analysis.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import pandas as pd
import tiktoken

from .models import (
    MODEL_CONTEXT_WINDOWS,
    TOKEN_RISK_THRESHOLDS,
    ChunkingRecommendation,
    TokenEstimation,
)

logger = logging.getLogger(__name__)


def analyze_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Analyze DataFrame columns by type for token estimation strategy."""
    numeric_cols: List[str] = []
    text_cols: List[str] = []
    other_cols: List[str] = []

    for col in df.columns:
        dtype = df[col].dtype

        # Boolean types first (before numeric check)
        if pd.api.types.is_bool_dtype(dtype):
            other_cols.append(col)
        # Numeric types (int, float, etc.) - but not boolean
        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        # String/object types that likely contain text
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            text_cols.append(col)
        # Everything else (datetime, etc.)
        else:
            other_cols.append(col)

    return {"numeric": numeric_cols, "text": text_cols, "other": other_cols}


def estimate_text_column_tokens(
    df: pd.DataFrame,
    text_columns: List[str],
    sample_size: int,
    encoding: Optional[tiktoken.Encoding],
) -> Dict[str, int]:
    """Estimate tokens for text columns using sampling."""
    column_tokens: Dict[str, int] = {}

    for col in text_columns:
        if col not in df.columns:
            continue

        # Sample data for analysis
        sample_data = df[col].dropna().head(sample_size)

        if sample_data.empty:
            column_tokens[col] = 1  # Placeholder for null/empty
            continue

        total_tokens = 0
        valid_samples = 0

        for value in sample_data:
            if pd.isna(value) or value == "":
                total_tokens += 1  # "null" or empty string
            else:
                str_value = str(value)
                if encoding:
                    try:
                        token_count = len(encoding.encode(str_value))
                        total_tokens += token_count
                    except Exception:
                        # Fallback: 1 token per 4 characters
                        total_tokens += max(1, len(str_value) // 4)
                else:
                    # Fallback method
                    total_tokens += max(1, len(str_value) // 4)

            valid_samples += 1

        # Average tokens per non-null value in this column
        avg_tokens = total_tokens / valid_samples if valid_samples > 0 else 1
        column_tokens[col] = int(avg_tokens)

    return column_tokens


def estimate_other_column_tokens(column: pd.Series) -> int:
    """Estimate tokens for non-numeric, non-text columns."""
    dtype = column.dtype

    # Boolean columns
    if pd.api.types.is_bool_dtype(dtype):
        return 1  # "true"/"false" = 1 token each

    # Datetime columns
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 3  # Typical datetime string ~3 tokens

    # Default
    return 2  # Conservative estimate


def calculate_json_overhead(df: pd.DataFrame) -> float:
    """Calculate JSON serialization overhead per row."""
    # Field names + brackets, commas, quotes
    # Roughly: {"field1": value1, "field2": value2, ...}
    # Each field adds ~2 tokens for quotes and structure
    return len(df.columns) * 2 + 2  # +2 for opening/closing braces


def calculate_confidence(
    df: pd.DataFrame,
    column_analysis: Dict[str, List[str]],
    confidence_boost: float,
    sample_size: int,
) -> float:
    """Calculate confidence in token estimation."""
    base_confidence = 0.8  # Start with high confidence

    # Reduce confidence for sampling
    if len(df) > sample_size:
        sampling_factor = min(1.0, sample_size / len(df))
        base_confidence *= 0.6 + 0.4 * sampling_factor

    # Reduce confidence for text-heavy data (more variable)
    text_ratio = (
        len(column_analysis["text"]) / len(df.columns) if len(df.columns) > 0 else 0
    )
    base_confidence *= 1.0 - 0.3 * text_ratio

    # Apply confidence boost
    final_confidence = min(1.0, base_confidence + confidence_boost)

    return round(final_confidence, 2)


def assess_token_risk(total_tokens: int) -> str:
    """Assess token count risk level."""
    for risk_level, threshold in TOKEN_RISK_THRESHOLDS.items():
        if total_tokens < threshold:
            return risk_level
    return "critical"


def assess_memory_risk(total_tokens: int, row_count: int) -> str:
    """Assess memory usage risk."""
    # Rough memory estimate (tokens * 4 bytes)
    estimated_mb = total_tokens * 4 / (1024 * 1024)

    if estimated_mb < 50:
        return "low"
    elif estimated_mb < 200:
        return "medium"
    else:
        return "high"


def assess_context_compatibility(total_tokens: int) -> Dict[str, bool]:
    """Check if response fits in various model context windows."""
    compatibility: Dict[str, bool] = {}

    for model, window_size in MODEL_CONTEXT_WINDOWS.items():
        if model == "default":
            continue
        # Reserve 20% for prompt and response space
        available_tokens = int(window_size * 0.8)
        compatibility[model] = total_tokens <= available_tokens

    return compatibility


def calculate_recommended_chunk_size(
    total_tokens: int, tokens_per_row: float
) -> Optional[int]:
    """Calculate recommended chunk size in rows."""
    if total_tokens <= 5000:
        return None  # No chunking needed

    # Target ~5000 tokens per chunk for good balance
    target_tokens_per_chunk = 5000
    rows_per_chunk = (
        int(target_tokens_per_chunk / tokens_per_row) if tokens_per_row > 0 else 100
    )

    # Minimum 10 rows, maximum 10000 rows per chunk
    return max(10, min(10000, rows_per_chunk))


def categorize_response_size(total_tokens: int, row_count: int) -> str:
    """Categorize response size for LLM understanding."""
    if total_tokens < 1000:
        return "small"
    elif total_tokens < 10000:
        return "medium"
    elif total_tokens < 50000:
        return "large"
    else:
        return "xlarge"


def assess_data_density(estimation: TokenEstimation) -> str:
    """Assess how dense the data is (text content vs structure)."""
    total_cols = (
        len(estimation.numeric_columns)
        + len(estimation.text_columns)
        + len(estimation.other_columns)
    )
    if total_cols == 0:
        return "sparse"

    # Calculate ratio of content tokens to overhead
    content_ratio = (
        estimation.tokens_per_row - estimation.json_overhead_per_row
    ) / estimation.tokens_per_row

    if content_ratio < 0.3:
        return "sparse"  # Mostly structure
    elif content_ratio < 0.7:
        return "moderate"
    else:
        return "dense"  # Mostly content


def generate_chunking_recommendation(
    estimation: TokenEstimation,
) -> ChunkingRecommendation:
    """Generate intelligent chunking recommendations."""
    should_chunk = estimation.total_tokens > 5000

    if not should_chunk:
        return ChunkingRecommendation(
            should_chunk=False,
            recommended_chunk_size=0,
            estimated_chunks=1,
            chunk_overlap_rows=0,
            strategy="none",
            chunk_size_rationale="Response is small enough to send as single chunk",
            performance_impact="Minimal",
            memory_benefits="None needed",
        )

    chunk_size = estimation.recommended_chunk_size or 1000
    estimated_chunks = math.ceil(estimation.total_rows / chunk_size)

    # Strategy depends on data characteristics
    strategy = "row_based"  # Most common strategy

    return ChunkingRecommendation(
        should_chunk=True,
        recommended_chunk_size=chunk_size,
        estimated_chunks=estimated_chunks,
        chunk_overlap_rows=0,  # No overlap needed for database results
        strategy=strategy,
        chunk_size_rationale=(
            f"Optimized for ~5000 tokens per chunk with "
            f"{estimation.tokens_per_row:.1f} tokens per row"
        ),
        performance_impact="Reduced memory usage, streaming capability",
        memory_benefits=(
            f"Reduces peak memory from "
            f"~{estimation.total_tokens * 4 // 1024 // 1024}MB to "
            f"~{int(chunk_size * estimation.tokens_per_row * 4) // 1024 // 1024}MB per chunk"
        ),
    )


def analyze_model_compatibility(
    estimation: TokenEstimation,
) -> Dict[str, Dict[str, Any]]:
    """Analyze compatibility with different language models."""
    compatibility: Dict[str, Dict[str, Any]] = {}

    for model, window_size in MODEL_CONTEXT_WINDOWS.items():
        if model == "default":
            continue

        available_tokens = int(window_size * 0.8)  # Reserve space
        fits = estimation.total_tokens <= available_tokens

        compatibility[model] = {
            "fits_in_context": fits,
            "context_window": window_size,
            "available_tokens": available_tokens,
            "utilization_percent": (estimation.total_tokens / available_tokens) * 100,
            "recommended_chunk_count": max(
                1, math.ceil(estimation.total_tokens / available_tokens)
            ),
        }

    return compatibility


def generate_sampling_options(
    estimation: TokenEstimation,
) -> Dict[str, Any]:
    """Generate sampling options for large datasets."""
    if estimation.total_rows <= 1000:
        return {
            "recommended": False,
            "reason": "Dataset is small enough to return in full",
        }

    # Calculate sampling options
    sample_sizes = [100, 500, 1000, 5000]
    options: Dict[str, Any] = {}

    for size in sample_sizes:
        if size < estimation.total_rows:
            sample_tokens = int(size * estimation.tokens_per_row)
            options[f"sample_{size}"] = {
                "rows": size,
                "estimated_tokens": sample_tokens,
                "percentage": (size / estimation.total_rows) * 100,
            }

    return {
        "recommended": estimation.total_tokens > 20000,
        "options": options,
        "reason": "Large dataset - sampling can provide quick overview",
    }


def assess_processing_complexity(estimation: TokenEstimation) -> str:
    """Assess processing complexity for performance estimation."""
    # Based on size and text content
    complexity_score = 0

    # Size factor
    if estimation.total_rows > 10000:
        complexity_score += 2
    elif estimation.total_rows > 1000:
        complexity_score += 1

    # Text processing factor
    text_ratio = len(estimation.text_columns) / max(
        1, len(estimation.text_columns) + len(estimation.numeric_columns)
    )
    if text_ratio > 0.5:
        complexity_score += 2
    elif text_ratio > 0.2:
        complexity_score += 1

    # Token density factor
    if estimation.tokens_per_row > 100:
        complexity_score += 1

    if complexity_score >= 4:
        return "high"
    elif complexity_score >= 2:
        return "medium"
    else:
        return "low"


def estimate_response_time(estimation: TokenEstimation) -> float:
    """Estimate response time based on data characteristics."""
    # Base time for small datasets
    base_time = 0.1

    # Add time based on row count
    row_time = estimation.total_rows * 0.0001  # 0.1ms per row

    # Add time for text processing
    text_time = len(estimation.text_columns) * estimation.total_rows * 0.0001

    # Add time for JSON serialization
    json_time = estimation.total_tokens * 0.000001  # 1us per token

    return round(base_time + row_time + text_time + json_time, 2)


def create_empty_estimation() -> TokenEstimation:
    """Create estimation for empty dataset."""
    return TokenEstimation(
        total_tokens=0,
        tokens_per_row=0.0,
        confidence=1.0,
        numeric_columns=[],
        text_columns=[],
        other_columns=[],
        column_token_breakdown={},
        estimation_method="empty",
        sample_size=0,
        total_rows=0,
        json_overhead_per_row=0,
        json_overhead_total=0,
        risk_level="low",
        memory_risk="low",
        fits_in_context={
            model: True for model in MODEL_CONTEXT_WINDOWS if model != "default"
        },
        recommended_chunk_size=None,
    )
