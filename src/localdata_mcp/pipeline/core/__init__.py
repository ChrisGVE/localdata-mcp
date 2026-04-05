"""
Core Pipeline Framework for LocalData MCP v2.0

This package provides the foundational pipeline classes:
- DataSciencePipeline: Enhanced sklearn Pipeline with metadata tracking
- DataFrameStreamingSource: Streaming data source for pandas DataFrames
- PipelineComposer: Multi-pipeline workflow orchestration
- StreamingDataPipeline: Memory-bounded streaming pipeline
- SklearnStreamingAdapter: Bridge between sklearn and StreamingQueryExecutor
"""

from .pipeline_class import DataSciencePipeline, DataFrameStreamingSource
from .composer import PipelineComposer
from .streaming import StreamingDataPipeline, SklearnStreamingAdapter

__all__ = [
    "DataSciencePipeline",
    "DataFrameStreamingSource",
    "PipelineComposer",
    "StreamingDataPipeline",
    "SklearnStreamingAdapter",
]
