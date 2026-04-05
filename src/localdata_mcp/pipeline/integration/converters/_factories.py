"""
Factory functions for creating converters and conversion options.

Provides convenient constructors for common converter configurations.
"""

from typing import Optional

from ._common import ConversionOptions, ConversionQuality
from ._pandas import PandasConverter
from ._numpy import NumpyConverter
from ._sparse import SparseMatrixConverter


# Factory functions for easy converter creation


def create_pandas_converter(
    conversion_options: Optional[ConversionOptions] = None, **kwargs
) -> PandasConverter:
    """Create a PandasConverter with optional configuration."""
    return PandasConverter(conversion_options=conversion_options, **kwargs)


def create_numpy_converter(
    conversion_options: Optional[ConversionOptions] = None, **kwargs
) -> NumpyConverter:
    """Create a NumpyConverter with optional configuration."""
    return NumpyConverter(conversion_options=conversion_options, **kwargs)


def create_sparse_converter(
    conversion_options: Optional[ConversionOptions] = None, **kwargs
) -> SparseMatrixConverter:
    """Create a SparseMatrixConverter with optional configuration."""
    return SparseMatrixConverter(conversion_options=conversion_options, **kwargs)


# Utility functions for conversion options


def create_memory_efficient_options() -> ConversionOptions:
    """Create conversion options optimized for memory efficiency."""
    return ConversionOptions(
        chunk_size_rows=5000,
        memory_efficient=True,
        sparse_density_threshold=0.05,
        quality_target=ConversionQuality.MODERATE,
    )


def create_high_fidelity_options() -> ConversionOptions:
    """Create conversion options optimized for data preservation."""
    return ConversionOptions(
        preserve_index=True,
        preserve_columns=True,
        handle_mixed_types=True,
        quality_target=ConversionQuality.HIGH_FIDELITY,
    )


def create_streaming_options(chunk_size: int = 10000) -> ConversionOptions:
    """Create conversion options optimized for streaming processing."""
    return ConversionOptions(
        chunk_size_rows=chunk_size,
        memory_efficient=True,
        quality_target=ConversionQuality.HIGH_FIDELITY,
    )
