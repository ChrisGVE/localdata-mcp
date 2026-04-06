"""Type detection - Main detection engine."""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ....logging_manager import get_logger
from ...type_conversion import TypeInferenceEngine
from ..interfaces import DataFormat, TypeDetector, ValidationResult
from ._detectors import (
    CategoricalDetector,
    FormatSpecificDetector,
    NumpyArrayDetector,
    PandasDataFrameDetector,
    TimeSeriesDetector,
)
from ._types import FormatDetectionResult, SchemaInfo

logger = get_logger(__name__)


class TypeDetectionEngine(TypeDetector):
    """
    Enhanced type detection engine building on existing TypeInferenceEngine.

    Provides comprehensive format detection across all supported data types
    with confidence scoring and detailed schema inference.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        enable_schema_inference: bool = True,
        max_sample_size: int = 10000,
    ):
        self.confidence_threshold = confidence_threshold
        self.enable_schema_inference = enable_schema_inference
        self.max_sample_size = max_sample_size
        self._base_engine = TypeInferenceEngine()
        self._detectors: List[FormatSpecificDetector] = [
            PandasDataFrameDetector(),
            NumpyArrayDetector(),
            TimeSeriesDetector(),
            CategoricalDetector(),
        ]
        self._detection_cache: Dict[str, FormatDetectionResult] = {}
        self._cache_max_size = 100

        logger.info(
            "TypeDetectionEngine initialized",
            confidence_threshold=confidence_threshold,
            num_detectors=len(self._detectors),
            enable_schema_inference=enable_schema_inference,
        )

    def detect_format(self, data: Any) -> FormatDetectionResult:
        """Detect the format of input data with confidence scoring."""
        start_time = time.time()

        cache_key = self._generate_cache_key(data)
        if cache_key in self._detection_cache:
            logger.debug("Returned cached format detection result")
            return self._detection_cache[cache_key]

        sample_data = self._sample_data(data)
        sample_size = len(sample_data) if hasattr(sample_data, "__len__") else 1

        detection_results = []
        detection_details = {}

        for detector in self._detectors:
            try:
                confidence, details = detector.detect_format(sample_data)
                if confidence > 0:
                    detection_results.append(
                        (detector.get_target_format(), confidence, detector)
                    )
                    detection_details[detector.get_target_format().value] = details
            except Exception as e:
                logger.warning(f"Detector {detector.__class__.__name__} failed: {e}")

        detection_results.sort(key=lambda x: x[1], reverse=True)

        if detection_results and detection_results[0][1] >= self.confidence_threshold:
            detected_format = detection_results[0][0]
            confidence_score = detection_results[0][1]
            best_detector = detection_results[0][2]
        else:
            detected_format, confidence_score = self._fallback_detection(sample_data)
            best_detector = None

        schema_info = None
        if self.enable_schema_inference and best_detector:
            try:
                schema_info = best_detector.extract_schema(sample_data)
            except Exception as e:
                logger.warning(f"Schema extraction failed: {e}")

        alternative_formats = [(fmt, conf) for fmt, conf, _ in detection_results[1:5]]

        result = FormatDetectionResult(
            detected_format=detected_format,
            confidence_score=confidence_score,
            alternative_formats=alternative_formats,
            detection_details=detection_details,
            schema_info=schema_info,
            detection_time=time.time() - start_time,
            sample_size=sample_size,
        )

        self._cache_result(cache_key, result)

        logger.info(
            "Format detection completed",
            detected_format=detected_format.value,
            confidence=confidence_score,
            detection_time=result.detection_time,
        )
        return result

    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for detection."""
        return self.confidence_threshold

    def validate_format_compatibility(
        self, data: Any, expected_format: DataFormat
    ) -> ValidationResult:
        """Validate if data is compatible with expected format."""
        detection_result = self.detect_format(data)
        errors = []
        warnings = []

        if detection_result.detected_format != expected_format:
            alternative_formats = {
                fmt for fmt, _ in detection_result.alternative_formats
            }
            if expected_format in alternative_formats:
                warnings.append(
                    f"Expected format {expected_format.value} is possible "
                    f"but not the most confident detection"
                )
            else:
                errors.append(
                    f"Data format {detection_result.detected_format.value} "
                    f"incompatible with expected {expected_format.value}"
                )

        if detection_result.schema_info:
            schema_warnings = self._validate_schema_quality(
                detection_result.schema_info
            )
            warnings.extend(schema_warnings)

        is_valid = len(errors) == 0
        score = detection_result.confidence_score if is_valid else 0.0

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            errors=errors,
            warnings=warnings,
            details={
                "detected_format": detection_result.detected_format.value,
                "detection_confidence": detection_result.confidence_score,
                "alternative_formats": detection_result.alternative_formats,
            },
        )

    def infer_conversion_requirements(
        self, data: Any, target_format: DataFormat
    ) -> Dict[str, Any]:
        """Infer requirements for converting data to target format."""
        detection_result = self.detect_format(data)

        requirements: Dict[str, Any] = {
            "source_format": detection_result.detected_format,
            "target_format": target_format,
            "conversion_needed": detection_result.detected_format != target_format,
            "confidence": detection_result.confidence_score,
            "estimated_complexity": "low",
        }

        if requirements["conversion_needed"]:
            complexity = self._assess_conversion_complexity(
                detection_result.detected_format,
                target_format,
                detection_result.schema_info,
            )
            requirements["estimated_complexity"] = complexity
            recommendations = self._generate_conversion_recommendations(
                detection_result, target_format
            )
            requirements["recommendations"] = recommendations

        return requirements

    def _sample_data(self, data: Any) -> Any:
        """Sample data for efficient analysis."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if len(data) > self.max_sample_size:
                return data.sample(n=self.max_sample_size, random_state=42)
        elif isinstance(data, np.ndarray):
            if data.size > self.max_sample_size:
                if data.ndim == 1:
                    indices = np.random.choice(
                        len(data), self.max_sample_size, replace=False
                    )
                    return data[indices]
                else:
                    max_rows = min(
                        (
                            self.max_sample_size // data.shape[1]
                            if data.ndim > 1
                            else self.max_sample_size
                        ),
                        data.shape[0],
                    )
                    indices = np.random.choice(data.shape[0], max_rows, replace=False)
                    return data[indices]
        elif hasattr(data, "__len__") and len(data) > self.max_sample_size:
            import random

            return random.sample(list(data), self.max_sample_size)
        return data

    def _fallback_detection(self, data: Any) -> Tuple[DataFormat, float]:
        """Fallback detection for unrecognized formats."""
        if isinstance(data, dict):
            return DataFormat.PYTHON_DICT, 0.6
        elif isinstance(data, list):
            return DataFormat.PYTHON_LIST, 0.6
        elif isinstance(data, str):
            if data.startswith(("http://", "https://")):
                return DataFormat.JSON, 0.5
            elif data.startswith(("{", "[")):
                return DataFormat.JSON, 0.7
            else:
                return DataFormat.PYTHON_DICT, 0.4
        else:
            return DataFormat.UNKNOWN, 0.1

    def _generate_cache_key(self, data: Any) -> str:
        """Generate cache key for data."""
        key_components = [
            str(type(data).__name__),
            str(getattr(data, "shape", None)),
            str(getattr(data, "dtypes", None)),
            str(id(data)),
        ]
        key_string = "_".join(str(c) for c in key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: FormatDetectionResult):
        """Cache detection result with size management."""
        if len(self._detection_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._detection_cache))
            del self._detection_cache[oldest_key]
        self._detection_cache[cache_key] = result

    def _validate_schema_quality(self, schema: SchemaInfo) -> List[str]:
        """Validate schema quality and return warnings."""
        warnings = []
        if "completeness" in schema.quality_metrics:
            if schema.quality_metrics["completeness"] < 0.8:
                warnings.append(
                    f"Low data completeness: {schema.quality_metrics['completeness']:.2%}"
                )
        if "consistency" in schema.quality_metrics:
            if schema.quality_metrics["consistency"] < 0.7:
                warnings.append(
                    f"Low data consistency: {schema.quality_metrics['consistency']:.2%}"
                )
        if schema.null_info and "null_percentage" in schema.null_info:
            if schema.null_info["null_percentage"] > 50:
                warnings.append(
                    f"High null value percentage: {schema.null_info['null_percentage']:.1f}%"
                )
        return warnings

    def _assess_conversion_complexity(
        self,
        source_format: DataFormat,
        target_format: DataFormat,
        schema_info: Optional[SchemaInfo],
    ) -> str:
        """Assess complexity of format conversion."""
        if source_format == target_format:
            return "none"

        complexity_map = {
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY): "low",
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME): "low",
            (DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES): "low",
            (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME): "low",
            (DataFormat.PANDAS_DATAFRAME, DataFormat.CATEGORICAL): "medium",
            (DataFormat.CATEGORICAL, DataFormat.PANDAS_DATAFRAME): "medium",
            (DataFormat.JSON, DataFormat.PANDAS_DATAFRAME): "high",
            (DataFormat.PANDAS_DATAFRAME, DataFormat.JSON): "medium",
        }

        conversion_key = (source_format, target_format)
        if conversion_key in complexity_map:
            return complexity_map[conversion_key]

        simple_formats = {DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY}
        complex_formats = {
            DataFormat.JSON,
            DataFormat.HIERARCHICAL,
            DataFormat.STREAMING,
        }

        if source_format in complex_formats or target_format in complex_formats:
            return "high"
        elif source_format in simple_formats and target_format in simple_formats:
            return "low"
        else:
            return "medium"

    def _generate_conversion_recommendations(
        self, detection_result: FormatDetectionResult, target_format: DataFormat
    ) -> List[str]:
        """Generate specific recommendations for conversion."""
        recommendations: List[str] = []
        source_format = detection_result.detected_format

        if (
            source_format == DataFormat.PANDAS_DATAFRAME
            and target_format == DataFormat.NUMPY_ARRAY
        ):
            recommendations.extend(
                [
                    "Use .values attribute for direct conversion",
                    "Consider handling missing values before conversion",
                    "Ensure all columns have compatible numeric types",
                ]
            )
        elif (
            source_format == DataFormat.NUMPY_ARRAY
            and target_format == DataFormat.PANDAS_DATAFRAME
        ):
            recommendations.extend(
                [
                    "Provide appropriate column names",
                    "Consider setting proper index",
                    "Specify data types if needed",
                ]
            )
        elif target_format == DataFormat.TIME_SERIES:
            recommendations.extend(
                [
                    "Ensure datetime index is properly set",
                    "Consider frequency inference for regular time series",
                    "Handle missing time periods if necessary",
                ]
            )
        elif target_format == DataFormat.CATEGORICAL:
            recommendations.extend(
                [
                    "Identify high-cardinality columns for categorical conversion",
                    "Consider ordered vs unordered categories",
                    "Validate category levels before conversion",
                ]
            )

        if detection_result.schema_info:
            schema = detection_result.schema_info
            if "completeness" in schema.quality_metrics:
                if schema.quality_metrics["completeness"] < 0.9:
                    recommendations.append("Address missing values before conversion")
            if schema.null_info and "null_percentage" in schema.null_info:
                if schema.null_info["null_percentage"] > 20:
                    recommendations.append(
                        "High null percentage detected - consider imputation strategies"
                    )

        if detection_result.confidence_score < 0.8:
            recommendations.append(
                "Low format detection confidence - manually verify data format"
            )

        return recommendations


def detect_data_format(
    data: Any,
    confidence_threshold: float = 0.7,
    include_schema: bool = True,
) -> FormatDetectionResult:
    """Convenient function for data format detection."""
    engine = TypeDetectionEngine(
        confidence_threshold=confidence_threshold,
        enable_schema_inference=include_schema,
    )
    return engine.detect_format(data)
