"""Response metadata generation for datasets.

Provides the ResponseMetadataGenerator factory class and a singleton accessor
for generating comprehensive response metadata from query results.
"""

import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

import pandas as pd

from ..query_analyzer import QueryAnalysis
from ..token_manager import get_token_manager
from .analysis import (
    assess_data_quality,
    assess_query_complexity,
    calculate_memory_footprint,
    determine_recommended_action,
    estimate_processing_time,
    generate_chunk_availability,
    generate_llm_friendly_summary,
    generate_schema_information,
    generate_statistical_summary,
)
from .models import EnhancedResponseMetadata

logger = logging.getLogger(__name__)


class ResponseMetadataGenerator:
    """Factory for generating enhanced response metadata."""

    def __init__(self) -> None:
        self.token_manager = get_token_manager()
        self._cache: Dict[str, Tuple[EnhancedResponseMetadata, float]] = {}

    def generate_metadata(
        self,
        query_id: str,
        df: pd.DataFrame,
        query: str,
        query_analysis: Optional[QueryAnalysis] = None,
        db_name: Optional[str] = None,
    ) -> EnhancedResponseMetadata:
        """Generate comprehensive response metadata for a dataset.

        Args:
            query_id: Unique identifier for the query
            df: DataFrame to analyze
            query: Original SQL query
            query_analysis: Pre-computed query analysis
            db_name: Database name

        Returns:
            Comprehensive response metadata
        """
        cache_key = self._generate_cache_key(query_id, query, df)
        if cache_key in self._cache:
            cached_metadata, cache_time = self._cache[cache_key]
            if time.time() - cache_time < 300:
                logger.debug(f"Using cached metadata for query {query_id}")
                return cached_metadata

        logger.info(f"Generating enhanced response metadata for query {query_id}")

        token_estimation = self.token_manager.estimate_tokens_from_dataframe(df)
        statistical_summary = generate_statistical_summary(df)
        data_quality = assess_data_quality(df)
        schema_info = generate_schema_information(df, db_name)

        complexity_level, complexity_score = assess_query_complexity(
            query, df, query_analysis
        )
        processing_time = estimate_processing_time(
            df, complexity_level, token_estimation
        )
        memory_footprint = calculate_memory_footprint(df, token_estimation)
        chunk_availability = generate_chunk_availability(df, token_estimation)

        recommended_action, action_rationale = determine_recommended_action(
            token_estimation, data_quality, complexity_level, len(df)
        )
        llm_summary = generate_llm_friendly_summary(
            df, token_estimation, data_quality, complexity_level
        )

        metadata = EnhancedResponseMetadata(
            query_id=query_id,
            timestamp=time.time(),
            query_complexity_score=complexity_score,
            query_complexity_level=complexity_level,
            estimated_processing_time=processing_time,
            memory_footprint=memory_footprint,
            token_estimation=token_estimation,
            statistical_summary=statistical_summary,
            data_quality_metrics=data_quality,
            schema_information=schema_info,
            chunk_availability=chunk_availability,
            supports_streaming=len(df) > 1000,
            supports_cancellation=True,
            cache_key=cache_key,
            recommended_action=recommended_action,
            action_rationale=action_rationale,
            llm_friendly_summary=llm_summary,
        )

        self._cache[cache_key] = (metadata, time.time())

        logger.info(
            f"Generated metadata for {len(df)} rows, {len(df.columns)} columns, "
            f"complexity: {complexity_level.value}, "
            f"quality: {data_quality.overall_quality.value}"
        )

        return metadata

    def _generate_cache_key(self, query_id: str, query: str, df: pd.DataFrame) -> str:
        """Generate cache key for metadata."""
        key_data = f"{query}_{len(df)}_{len(df.columns)}_{df.dtypes.to_string()}"
        return hashlib.md5(key_data.encode()).hexdigest()


# Singleton instance
_metadata_generator: Optional[ResponseMetadataGenerator] = None


def get_metadata_generator() -> ResponseMetadataGenerator:
    """Get the global ResponseMetadataGenerator instance."""
    global _metadata_generator
    if _metadata_generator is None:
        _metadata_generator = ResponseMetadataGenerator()
    return _metadata_generator
