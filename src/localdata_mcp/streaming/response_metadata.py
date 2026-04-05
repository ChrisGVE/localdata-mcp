"""Response metadata generation for streaming results.

Provides standalone functions for generating rich response and
analysis metadata using the TokenManager, extracted from
StreamingQueryExecutor.
"""

from typing import Any, Dict

import pandas as pd

from ..logging_manager import get_logger
from ..token_manager import get_token_manager

logger = get_logger(__name__)


def generate_response_metadata(
    sample_df: pd.DataFrame, total_rows: int
) -> Dict[str, Any]:
    """Generate rich response metadata using TokenManager.

    Args:
        sample_df: First chunk/sample of the data.
        total_rows: Total number of rows in the complete result.

    Returns:
        Dictionary with enhanced metadata for LLM decision-making.
    """
    try:
        # Validate input data
        if sample_df is None or len(sample_df) == 0 or total_rows <= 0:
            return {
                "token_analysis": {
                    "estimated_total_tokens": 0,
                    "risk_level": "low",
                    "note": "No data to analyze",
                }
            }

        mgr = get_token_manager()

        # Get estimation for the complete result
        estimation = mgr.estimate_tokens_for_query_result(total_rows, sample_df)

        # Get rich response metadata
        response_meta = mgr.get_response_metadata(estimation)

        # Create enhanced metadata structure
        return format_response_metadata(estimation, response_meta)

    except Exception as e:
        logger.warning(f"Failed to generate response metadata: {e}")
        return {
            "token_analysis": {
                "error": f"Token analysis failed: {str(e)}",
                "estimated_total_tokens": 0,
                "risk_level": "unknown",
            }
        }


def format_response_metadata(estimation, response_meta) -> Dict[str, Any]:
    """Format estimation and response metadata into structured dict.

    Args:
        estimation: Estimation result from the manager.
        response_meta: Response metadata from the manager.

    Returns:
        Dictionary with analysis, data characteristics,
        chunking recommendations, model compatibility, performance
        indicators, and sampling options.
    """
    return {
        "token_analysis": {
            "estimated_total_tokens": estimation.total_tokens,
            "tokens_per_row": round(estimation.tokens_per_row, 2),
            "confidence": estimation.confidence,
            "risk_level": estimation.risk_level,
            "memory_risk": estimation.memory_risk,
        },
        "data_characteristics": {
            "response_size_category": response_meta.response_size_category,
            "data_density": response_meta.data_density,
            "text_heavy": response_meta.text_heavy,
            "column_breakdown": {
                "numeric_columns": estimation.numeric_columns,
                "text_columns": estimation.text_columns,
                "other_columns": estimation.other_columns,
            },
        },
        "chunking_recommendation": {
            "should_chunk": response_meta.chunking_recommendation.should_chunk
            if response_meta.chunking_recommendation
            else False,
            "recommended_chunk_size": response_meta.chunking_recommendation.recommended_chunk_size
            if response_meta.chunking_recommendation
            else None,
            "strategy": response_meta.chunking_recommendation.strategy
            if response_meta.chunking_recommendation
            else None,
            "rationale": response_meta.chunking_recommendation.chunk_size_rationale
            if response_meta.chunking_recommendation
            else None,
        },
        "model_compatibility": {
            model: info["fits_in_context"]
            for model, info in response_meta.model_compatibility.items()
        },
        "performance_indicators": {
            "processing_complexity": response_meta.processing_complexity,
            "estimated_response_time": response_meta.estimated_response_time,
            "estimated_memory_mb": response_meta.estimated_memory_mb,
            "streaming_recommended": response_meta.streaming_recommended,
        },
        "sampling_options": response_meta.sampling_options,
    }
