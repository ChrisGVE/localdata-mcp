"""
Type detection for the pipeline integration layer.

Provides format-specific detection, confidence scoring, and schema inference.
"""

from ._types import FormatDetectionResult, SchemaInfo
from ._detectors import (
    FormatSpecificDetector,
    PandasDataFrameDetector,
    NumpyArrayDetector,
    TimeSeriesDetector,
    CategoricalDetector,
)
from ._engine import TypeDetectionEngine, detect_data_format

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
