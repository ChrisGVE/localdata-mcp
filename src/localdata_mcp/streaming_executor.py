"""Backward-compatibility shim — real implementation in streaming/ sub-package.

Also re-exports internal names so existing mock.patch() targets keep working.
"""

from .streaming import (  # noqa: F401
    ChunkMetrics,
    MemoryStatus,
    ResultBuffer,
    StreamingDataSource,
    StreamingFileSource,
    StreamingQueryExecutor,
    StreamingSQLSource,
    create_streaming_source,
)

# Re-export internal helpers used as mock.patch() targets in tests
from .logging_manager import get_logging_manager  # noqa: F401
from .size_estimator import get_size_estimator  # noqa: F401
from .token_manager import get_token_manager  # noqa: F401

__all__ = [
    "ChunkMetrics",
    "MemoryStatus",
    "ResultBuffer",
    "StreamingDataSource",
    "StreamingFileSource",
    "StreamingQueryExecutor",
    "StreamingSQLSource",
    "create_streaming_source",
]
