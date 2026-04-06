"""Streaming File Processing Pipeline for LocalData MCP.

This sub-package implements memory-efficient streaming file processors that replace
batch DataFrame loading with streaming approaches. Works with the streaming
executor architecture to provide memory-bounded file processing.
"""

from .base import FileProcessorFactory, ProcessingProgress, StreamingFileProcessor
from .engine import create_streaming_file_engine
from .processors import (
    StreamingCSVProcessor,
    StreamingExcelProcessor,
    StreamingJSONProcessor,
    StreamingNumbersProcessor,
    StreamingODSProcessor,
)
from .sources import (
    StreamingCSVSource,
    StreamingExcelSource,
    StreamingJSONSource,
    StreamingNumbersSource,
    StreamingODSSource,
)

__all__ = [
    "ProcessingProgress",
    "StreamingFileProcessor",
    "FileProcessorFactory",
    "StreamingExcelProcessor",
    "StreamingJSONProcessor",
    "StreamingCSVProcessor",
    "StreamingODSProcessor",
    "StreamingNumbersProcessor",
    "StreamingExcelSource",
    "StreamingJSONSource",
    "StreamingCSVSource",
    "StreamingODSSource",
    "StreamingNumbersSource",
    "create_streaming_file_engine",
]
