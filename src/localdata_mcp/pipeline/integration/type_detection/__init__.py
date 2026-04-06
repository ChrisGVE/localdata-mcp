"""
Type detection for the pipeline integration layer.

Provides format-specific detection, confidence scoring, and schema inference.
"""

from ._detectors import (
    CategoricalDetector,
    FormatSpecificDetector,
    NumpyArrayDetector,
    PandasDataFrameDetector,
    TimeSeriesDetector,
)
from ._engine import TypeDetectionEngine, detect_data_format
from ._types import FormatDetectionResult, SchemaInfo

__all__ = [
    "FormatDetectionResult",
    "SchemaInfo",
    "FormatSpecificDetector",
    "PandasDataFrameDetector",
    "NumpyArrayDetector",
    "TimeSeriesDetector",
    "CategoricalDetector",
    "TypeDetectionEngine",
    "detect_data_format",
]
