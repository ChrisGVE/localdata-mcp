"""LLM Communication Protocol for progressive data interaction.

Provides the LLMCommunicationProtocol class for intelligent, chunked communication
between the data layer and LLM agents.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ..token_manager import get_token_manager
from .models import EnhancedResponseMetadata

logger = logging.getLogger(__name__)


class LLMCommunicationProtocol:
    """Protocol for progressive data interaction and intelligent communication with LLMs."""

    def __init__(
        self, response_metadata: EnhancedResponseMetadata, data_source: Any = None
    ):
        """Initialize the communication protocol.

        Args:
            response_metadata: Comprehensive metadata about the response
            data_source: Source of data (QueryBuffer, file path, database connection, etc.)
        """
        self.metadata = response_metadata
        self.data_source = data_source
        self.cancelled = False
        self.active_chunks: Dict[int, pd.DataFrame] = {}

        logger.debug(
            f"Initialized LLM communication protocol for query {response_metadata.query_id}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary suitable for LLM understanding.

        Returns:
            Dictionary with key information for LLM decision-making
        """
        summary = {
            "query_id": self.metadata.query_id,
            "data_overview": {
                "total_rows": self.metadata.statistical_summary.total_rows,
                "columns": len(self.metadata.schema_information.columns),
                "estimated_tokens": self.metadata.token_estimation.total_tokens,
                "memory_footprint_mb": self.metadata.memory_footprint,
                "data_quality": self.metadata.data_quality_metrics.overall_quality.value,
            },
            "complexity_assessment": {
                "query_complexity": self.metadata.query_complexity_level.value,
                "complexity_score": self.metadata.query_complexity_score,
                "processing_time_estimate": self.metadata.estimated_processing_time,
            },
            "loading_options": {
                "supports_chunking": self.metadata.chunk_availability.total_chunks > 1,
                "supports_streaming": self.metadata.supports_streaming,
                "supports_cancellation": self.metadata.supports_cancellation,
                "recommended_action": self.metadata.recommended_action,
            },
            "sample_data": self.metadata.statistical_summary.sample_rows[
                :3
            ],  # First 3 rows
            "schema_preview": [
                {
                    "column": col["name"],
                    "type": col["dtype"],
                    "sample_values": col.get("sample_values", [])[:3],
                }
                for col in self.metadata.schema_information.columns[
                    :10
                ]  # First 10 columns
            ],
            "recommendations": {
                "action": self.metadata.recommended_action,
                "rationale": self.metadata.action_rationale,
                "chunking_strategy": self.metadata.token_estimation.recommended_chunk_size,
                "quality_recommendations": self.metadata.data_quality_metrics.recommendations[
                    :3
                ],
            },
        }

        return summary

    def request_chunk(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Request a specific chunk of data.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            Chunk data with metadata, or None if chunk not available
        """
        if self.cancelled:
            logger.warning(f"Cannot request chunk {chunk_id}: operation cancelled")
            return None

        if chunk_id not in self.metadata.chunk_availability.available_chunks:
            logger.warning(f"Chunk {chunk_id} not available")
            return None

        logger.info(f"Loading chunk {chunk_id} for query {self.metadata.query_id}")

        try:
            chunk_data = self._load_chunk_from_source(chunk_id)

            if chunk_data is not None:
                self.active_chunks[chunk_id] = chunk_data

                token_manager = get_token_manager()
                chunk_token_estimate = token_manager.estimate_tokens_from_dataframe(
                    chunk_data
                )

                return {
                    "chunk_id": chunk_id,
                    "data": chunk_data.to_dict(orient="records"),
                    "metadata": {
                        "rows": len(chunk_data),
                        "columns": len(chunk_data.columns),
                        "estimated_tokens": chunk_token_estimate.total_tokens,
                        "memory_mb": chunk_data.memory_usage(deep=True).sum()
                        / (1024 * 1024),
                        "chunk_size": self.metadata.chunk_availability.chunk_size,
                        "total_chunks": self.metadata.chunk_availability.total_chunks,
                    },
                }
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_id}: {e}")
            return None

        return None

    def request_multiple_chunks(
        self, chunk_ids: List[int]
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        """Request multiple chunks efficiently.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            Dictionary mapping chunk_id to chunk data (or None if failed)
        """
        results = {}

        for chunk_id in chunk_ids:
            if self.cancelled:
                break
            results[chunk_id] = self.request_chunk(chunk_id)

        return results

    def cancel_operation(self, reason: str = "User requested") -> bool:
        """Cancel the ongoing operation.

        Args:
            reason: Reason for cancellation

        Returns:
            True if cancellation was successful
        """
        if not self.metadata.supports_cancellation:
            logger.warning(
                f"Operation {self.metadata.query_id} does not support cancellation"
            )
            return False

        self.cancelled = True
        logger.info(f"Operation {self.metadata.query_id} cancelled: {reason}")

        # Clean up active chunks
        self.active_chunks.clear()

        return True

    def get_schema_details(self) -> Dict[str, Any]:
        """Get detailed schema information for data understanding.

        Returns:
            Detailed schema information
        """
        return {
            "columns": self.metadata.schema_information.columns,
            "primary_keys": self.metadata.schema_information.primary_keys,
            "foreign_keys": self.metadata.schema_information.foreign_keys,
            "indexes": self.metadata.schema_information.indexes,
            "constraints": self.metadata.schema_information.constraints,
            "table_metadata": {
                "name": self.metadata.schema_information.table_name,
                "last_modified": self.metadata.schema_information.last_modified,
            },
        }

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality assessment.

        Returns:
            Data quality report with metrics and recommendations
        """
        return {
            "overall_quality": self.metadata.data_quality_metrics.overall_quality.value,
            "quality_score": self.metadata.data_quality_metrics.quality_score,
            "dimensions": {
                "completeness": self.metadata.data_quality_metrics.completeness,
                "consistency": self.metadata.data_quality_metrics.consistency,
                "validity": self.metadata.data_quality_metrics.validity,
                "accuracy": self.metadata.data_quality_metrics.accuracy,
            },
            "issues": self.metadata.data_quality_metrics.issues,
            "recommendations": self.metadata.data_quality_metrics.recommendations,
            "statistical_summary": {
                "null_percentage": self.metadata.statistical_summary.null_percentage,
                "duplicate_percentage": self.metadata.statistical_summary.duplicate_percentage,
                "data_types": self.metadata.statistical_summary.data_types,
            },
        }

    def _load_chunk_from_source(self, chunk_id: int) -> Optional[pd.DataFrame]:
        """Load a chunk from the data source.

        This is a placeholder implementation. In practice, this would:
        - Load from QueryBuffer if data is cached
        - Execute chunked query if loading from database
        - Read chunk from file if data source is a file

        Args:
            chunk_id: ID of chunk to load

        Returns:
            DataFrame chunk or None if loading failed
        """
        logger.debug(f"Loading chunk {chunk_id} from data source")
        return None
