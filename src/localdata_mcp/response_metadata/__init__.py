"""Enhanced Response Metadata and LLM Communication Protocol for LocalData MCP.

This sub-package provides comprehensive response metadata that enables intelligent
LLM decision-making about large datasets, including progressive data loading, query
complexity analysis, and rich statistical summaries.
"""

from .generator import ResponseMetadataGenerator, get_metadata_generator
from .models import (
    ChunkAvailability,
    DataQualityLevel,
    DataQualityMetrics,
    EnhancedResponseMetadata,
    QueryComplexity,
    SchemaInformation,
    StatisticalSummary,
)
from .protocol import LLMCommunicationProtocol

__all__ = [
    "ChunkAvailability",
    "DataQualityLevel",
    "DataQualityMetrics",
    "EnhancedResponseMetadata",
    "LLMCommunicationProtocol",
    "QueryComplexity",
    "ResponseMetadataGenerator",
    "SchemaInformation",
    "StatisticalSummary",
    "get_metadata_generator",
]
