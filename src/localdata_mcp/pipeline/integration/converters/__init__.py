"""
Core Data Format Converters for LocalData MCP v2.0 Integration Shims Framework.

This package provides bidirectional data format conversion capabilities between
common data science formats with comprehensive metadata preservation and
memory-efficient processing.

Key Features:
- PandasConverter: DataFrame <-> other formats with index/column preservation
- NumpyConverter: Array <-> other formats with shape/dtype preservation
- SparseMatrixConverter: Sparse matrices <-> other formats with density management
- Streaming-first architecture for memory efficiency
- Sklearn-compatible fit/transform patterns
- Comprehensive error handling and quality scoring
"""

from ._common import (
    ConversionContextInternal,
    ConversionQuality,
    ConversionOptions,
)
from ._pandas import PandasConverter
from ._numpy import NumpyConverter
from ._sparse import SparseMatrixConverter
from ._factories import (
    create_pandas_converter,
    create_numpy_converter,
    create_sparse_converter,
    create_memory_efficient_options,
    create_high_fidelity_options,
    create_streaming_options,
)

__all__ = [
    # Core converters
    "PandasConverter",
    "NumpyConverter",
    "SparseMatrixConverter",
    # Conversion options and utilities
    "ConversionOptions",
    "ConversionQuality",
    "ConversionContextInternal",
    # Factory functions
    "create_pandas_converter",
    "create_numpy_converter",
    "create_sparse_converter",
    "create_memory_efficient_options",
    "create_high_fidelity_options",
    "create_streaming_options",
]
