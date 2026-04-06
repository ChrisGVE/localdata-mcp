"""
Preprocessing Stage Pipeline - Data Cleaning and Feature Engineering

This package implements the preprocessing stage of the Core Pipeline Framework,
providing data cleaning, normalization, and feature engineering with progressive
disclosure architecture and streaming compatibility.

Key Features:
- Progressive complexity levels (minimal, auto, comprehensive, custom)
- Streaming-compatible chunk-by-chunk processing
- Intelligent transformation selection based on data characteristics
- Detailed transformation logging and metadata generation
"""

from ._cleaning_pipeline import DataCleaningPipeline
from ._dataclasses import (
    CleaningOperation,
    DataQualityMetrics,
    TransformationStrategy,
)
from ._encoding_pipeline import CategoricalEncodingPipeline
from ._preprocessing_pipeline import DataPreprocessingPipeline
from ._scaling_pipeline import FeatureScalingPipeline

__all__ = [
    "DataQualityMetrics",
    "CleaningOperation",
    "TransformationStrategy",
    "DataPreprocessingPipeline",
    "DataCleaningPipeline",
    "FeatureScalingPipeline",
    "CategoricalEncodingPipeline",
]
