"""
Core Pipeline Framework for LocalData MCP v2.0

This package provides the foundational pipeline classes:
- DataSciencePipeline: Enhanced sklearn Pipeline with metadata tracking
- DataFrameStreamingSource: Streaming data source for pandas DataFrames
- PipelineComposer: Multi-pipeline workflow orchestration
- StreamingDataPipeline: Memory-bounded streaming pipeline
- SklearnStreamingAdapter: Bridge between sklearn and StreamingQueryExecutor
"""

from .composer import PipelineComposer
from .pipeline_class import DataFrameStreamingSource, DataSciencePipeline
from .streaming import SklearnStreamingAdapter, StreamingDataPipeline

__all__ = [
    "DataSciencePipeline",
    "DataFrameStreamingSource",
    "PipelineComposer",
    "StreamingDataPipeline",
    "SklearnStreamingAdapter",
]
