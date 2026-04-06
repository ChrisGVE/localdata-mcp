"""TokenManager class and singleton accessor.

This module contains the main TokenManager class that provides intelligent
token management with DataFrame-based estimation, and the get_token_manager
singleton factory function.
"""

import logging
from typing import Optional

import pandas as pd
import tiktoken

from .estimation import (
    analyze_column_types,
    analyze_model_compatibility,
    assess_context_compatibility,
    assess_data_density,
    assess_memory_risk,
    assess_processing_complexity,
    assess_token_risk,
    calculate_confidence,
    calculate_json_overhead,
    calculate_recommended_chunk_size,
    categorize_response_size,
    create_empty_estimation,
    estimate_other_column_tokens,
    estimate_response_time,
    estimate_text_column_tokens,
    generate_chunking_recommendation,
    generate_sampling_options,
)
from .models import ChunkingRecommendation, ResponseMetadata, TokenEstimation

logger = logging.getLogger(__name__)


class TokenManager:
    """Intelligent token management with DataFrame-based estimation."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        sample_size: int = 100,
    ) -> None:
        """Initialize the token manager.

        Args:
            encoding_name: Tiktoken encoding to use (default: cl100k_base for GPT-4)
            sample_size: Number of rows to sample for text analysis
        """
        self.sample_size = sample_size
        self.encoding: Optional[tiktoken.Encoding] = None

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.debug("Initialized TokenManager with %s encoding", encoding_name)
        except Exception as e:
            logger.warning("Failed to initialize tiktoken encoder: %s", e)
            logger.warning("Token estimation will use fallback methods")

    def estimate_tokens_from_dataframe(
        self,
        df: pd.DataFrame,
        confidence_boost: float = 0.0,
    ) -> TokenEstimation:
        """Estimate tokens from a complete DataFrame using efficient sampling.

        Args:
            df: DataFrame to analyze
            confidence_boost: Additional confidence for known complete data

        Returns:
            Complete token estimation with metadata
        """
        if df.empty:
            return create_empty_estimation()

        # Analyze DataFrame structure
        column_analysis = self._analyze_column_types(df)

        # Estimate tokens per column type
        column_tokens: dict[str, int] = {}
        total_tokens_per_row: float = 0

        # Handle numeric columns (1 token per value)
        for col in column_analysis["numeric"]:
            col_tokens = 1  # Numeric values = 1 token each
            column_tokens[col] = col_tokens
            total_tokens_per_row += col_tokens

        # Handle text columns with sampling
        text_token_analysis = self._estimate_text_column_tokens(
            df, column_analysis["text"]
        )
        for col, tokens in text_token_analysis.items():
            column_tokens[col] = tokens
            total_tokens_per_row += tokens

        # Handle other columns
        for col in column_analysis["other"]:
            col_tokens = self._estimate_other_column_tokens(df[col])
            column_tokens[col] = col_tokens
            total_tokens_per_row += col_tokens

        # Calculate JSON serialization overhead
        json_overhead = self._calculate_json_overhead(df)
        total_tokens_per_row += json_overhead

        # Calculate total tokens
        total_tokens = int(len(df) * total_tokens_per_row)

        # Determine confidence based on sampling
        confidence = self._calculate_confidence(df, column_analysis, confidence_boost)

        return TokenEstimation(
            total_tokens=total_tokens,
            tokens_per_row=total_tokens_per_row,
            confidence=confidence,
            numeric_columns=column_analysis["numeric"],
            text_columns=column_analysis["text"],
            other_columns=column_analysis["other"],
            column_token_breakdown=column_tokens,
            estimation_method="sampled" if len(df) > self.sample_size else "full",
            sample_size=min(len(df), self.sample_size),
            total_rows=len(df),
            json_overhead_per_row=json_overhead,
            json_overhead_total=int(len(df) * json_overhead),
            risk_level=assess_token_risk(total_tokens),
            memory_risk=assess_memory_risk(total_tokens, len(df)),
            fits_in_context=assess_context_compatibility(total_tokens),
            recommended_chunk_size=calculate_recommended_chunk_size(
                total_tokens, total_tokens_per_row
            ),
        )

    def estimate_tokens_for_query_result(
        self,
        row_count: int,
        sample_df: pd.DataFrame,
    ) -> TokenEstimation:
        """Estimate tokens for a large query result using sample data.

        This is used by QueryAnalyzer for pre-execution estimation.

        Args:
            row_count: Total number of rows expected
            sample_df: Sample DataFrame (e.g., from LIMIT 1)

        Returns:
            Token estimation extrapolated from sample
        """
        if sample_df.empty or row_count == 0:
            return create_empty_estimation()

        # Get per-row estimation from sample
        sample_estimation = self.estimate_tokens_from_dataframe(sample_df)
        tokens_per_row = sample_estimation.tokens_per_row

        # Extrapolate to full dataset
        total_tokens = int(row_count * tokens_per_row)

        # Reduce confidence for extrapolation
        extrapolation_confidence = max(0.1, sample_estimation.confidence * 0.6)

        return TokenEstimation(
            total_tokens=total_tokens,
            tokens_per_row=tokens_per_row,
            confidence=extrapolation_confidence,
            numeric_columns=sample_estimation.numeric_columns,
            text_columns=sample_estimation.text_columns,
            other_columns=sample_estimation.other_columns,
            column_token_breakdown=sample_estimation.column_token_breakdown,
            estimation_method="extrapolated",
            sample_size=len(sample_df),
            total_rows=row_count,
            json_overhead_per_row=sample_estimation.json_overhead_per_row,
            json_overhead_total=int(
                row_count * sample_estimation.json_overhead_per_row
            ),
            risk_level=assess_token_risk(total_tokens),
            memory_risk=assess_memory_risk(total_tokens, row_count),
            fits_in_context=assess_context_compatibility(total_tokens),
            recommended_chunk_size=calculate_recommended_chunk_size(
                total_tokens, tokens_per_row
            ),
        )

    def get_response_metadata(
        self,
        estimation: TokenEstimation,
        include_chunking: bool = True,
    ) -> ResponseMetadata:
        """Generate rich metadata for LLM decision-making.

        Args:
            estimation: Token estimation results
            include_chunking: Whether to include chunking recommendations

        Returns:
            Complete response metadata
        """
        # Categorize response size
        size_category = categorize_response_size(
            estimation.total_tokens, estimation.total_rows
        )

        # Assess data characteristics
        text_heavy = len(estimation.text_columns) > len(estimation.numeric_columns)
        data_density = assess_data_density(estimation)

        # Generate chunking recommendation
        chunking_rec: Optional[ChunkingRecommendation] = None
        if include_chunking and estimation.total_tokens > 5000:
            chunking_rec = generate_chunking_recommendation(estimation)

        # Model compatibility analysis
        model_compat = analyze_model_compatibility(estimation)

        # Estimate memory usage (rough approximation)
        estimated_memory = estimation.total_tokens * 4 / (1024 * 1024)

        return ResponseMetadata(
            estimated_tokens=estimation.total_tokens,
            estimated_memory_mb=estimated_memory,
            response_size_category=size_category,
            row_count=estimation.total_rows,
            column_count=(
                len(estimation.numeric_columns)
                + len(estimation.text_columns)
                + len(estimation.other_columns)
            ),
            data_density=data_density,
            text_heavy=text_heavy,
            chunking_recommendation=chunking_rec,
            streaming_recommended=(
                estimation.total_tokens > 10000 or estimation.total_rows > 1000
            ),
            sampling_options=generate_sampling_options(estimation),
            model_compatibility=model_compat,
            processing_complexity=assess_processing_complexity(estimation),
            estimated_response_time=estimate_response_time(estimation),
        )

    # --- Delegate to estimation module (preserve method API for tests) ---

    def _analyze_column_types(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Analyze DataFrame columns by type for token estimation."""
        return analyze_column_types(df)

    def _estimate_text_column_tokens(
        self,
        df: pd.DataFrame,
        text_columns: list[str],
    ) -> dict[str, int]:
        """Estimate tokens for text columns using sampling."""
        return estimate_text_column_tokens(
            df, text_columns, self.sample_size, self.encoding
        )

    def _estimate_other_column_tokens(self, column: pd.Series) -> int:
        """Estimate tokens for non-numeric, non-text columns."""
        return estimate_other_column_tokens(column)

    def _calculate_json_overhead(self, df: pd.DataFrame) -> float:
        """Calculate JSON serialization overhead per row."""
        return calculate_json_overhead(df)

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        column_analysis: dict[str, list[str]],
        confidence_boost: float,
    ) -> float:
        """Calculate confidence in token estimation."""
        return calculate_confidence(
            df, column_analysis, confidence_boost, self.sample_size
        )

    def _assess_token_risk(self, total_tokens: int) -> str:
        """Assess token count risk level."""
        return assess_token_risk(total_tokens)

    def _assess_memory_risk(self, total_tokens: int, row_count: int) -> str:
        """Assess memory usage risk."""
        return assess_memory_risk(total_tokens, row_count)

    def _assess_context_compatibility(self, total_tokens: int) -> dict[str, bool]:
        """Check if response fits in various model context windows."""
        return assess_context_compatibility(total_tokens)

    def _generate_chunking_recommendation(
        self, estimation: TokenEstimation
    ) -> ChunkingRecommendation:
        """Generate intelligent chunking recommendations."""
        return generate_chunking_recommendation(estimation)


# Singleton instance for global use
_token_manager: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    """Get the global TokenManager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager
