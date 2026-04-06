"""
Common types and utilities shared across converter modules.

Provides dataclasses and enums used by PandasConverter, NumpyConverter,
and SparseMatrixConverter.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


@dataclass
class ConversionContextInternal:
    """Internal conversion context for tracking conversion operations."""

    request_id: str
    start_time: float = field(default_factory=time.time)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class ConversionQuality(Enum):
    """Quality levels for data conversion."""

    LOSSLESS = "lossless"  # No data loss
    HIGH_FIDELITY = "high_fidelity"  # Minimal data loss
    MODERATE = "moderate"  # Some data loss acceptable
    LOW = "low"  # Significant data loss


@dataclass
class ConversionOptions:
    """Options for controlling conversion behavior."""

    preserve_index: bool = True
    preserve_columns: bool = True
    handle_mixed_types: bool = True
    categorical_threshold: float = (
        0.1  # Unique ratio below which to treat as categorical
    )
    sparse_density_threshold: float = 0.1  # Below which to use sparse representation
    chunk_size_rows: int = 10000
    memory_efficient: bool = True
    quality_target: ConversionQuality = ConversionQuality.HIGH_FIDELITY
