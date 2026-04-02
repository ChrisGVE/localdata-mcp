"""
Enhanced Type Detection System for Integration Shims Framework.

This module provides advanced type detection capabilities that build upon the existing 
TypeInferenceEngine, adding format-specific detection, confidence scoring, and 
schema inference for seamless cross-domain integration.

Key Features:
- Enhanced TypeDetectionEngine with format-specific detectors
- FormatDetectionResult with detailed analysis and confidence scoring
- SchemaInfo extraction and validation
- Integration with existing TypeInferenceEngine
- Support for complex data formats and nested structures
- Streaming-compatible detection for large datasets
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, date

from .interfaces import DataFormat, ValidationResult, TypeDetector
from ..type_conversion import TypeInferenceEngine, DataType, TypeInferenceResult
from ...logging_manager import get_logger

logger = get_logger(__name__)


@dataclass
class FormatDetectionResult:
    """Result of data format detection analysis."""
    
    detected_format: DataFormat
    confidence_score: float
    alternative_formats: List[Tuple[DataFormat, float]] = field(default_factory=list)
    detection_details: Dict[str, Any] = field(default_factory=dict)
    schema_info: Optional['SchemaInfo'] = None
    warnings: List[str] = field(default_factory=list)
    detection_time: float = 0.0
    sample_size: int = 0


@dataclass 
class SchemaInfo:
    """Comprehensive schema information for detected data formats."""
    
    data_format: DataFormat
    structure_type: str  # 'tabular', 'array', 'nested', 'scalar'
    
    # Tabular data schema
    columns: Optional[Dict[str, str]] = None  # column_name -> data_type
    column_types: Optional[Dict[str, DataType]] = None
    
    # Array/tensor schema
    shape: Optional[Tuple[int, ...]] = None
    element_type: Optional[str] = None
    
    # General properties
    size_info: Dict[str, Any] = field(default_factory=dict)
    null_info: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    creation_time: datetime = field(default_factory=datetime.now)
    inference_confidence: float = 1.0
    additional_properties: Dict[str, Any] = field(default_factory=dict)


class FormatSpecificDetector(ABC):
    """Abstract base class for format-specific detectors."""
    
    @abstractmethod
    def detect_format(self, data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Detect if data matches this detector's format.
        
        Returns:
            Tuple of (confidence_score, detection_details)
        """
        pass
    
    @abstractmethod
    def get_target_format(self) -> DataFormat:
        """Get the DataFormat this detector identifies."""
        pass
    
    @abstractmethod
    def extract_schema(self, data: Any) -> SchemaInfo:
        """Extract schema information for detected format."""
        pass


class PandasDataFrameDetector(FormatSpecificDetector):
    """Detector for pandas DataFrame format."""
    
    def detect_format(self, data: Any) -> Tuple[float, Dict[str, Any]]:
        """Detect pandas DataFrame format."""
        if isinstance(data, pd.DataFrame):
            details = {
                'shape': data.shape,
                'dtypes': data.dtypes.to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'index_type': type(data.index).__name__,
                'columns_count': len(data.columns)
            }
            return 1.0, details
        
        return 0.0, {}
    
    def get_target_format(self) -> DataFormat:
        return DataFormat.PANDAS_DATAFRAME
    
    def extract_schema(self, data: pd.DataFrame) -> SchemaInfo:
        """Extract schema from DataFrame."""
        # Use existing TypeInferenceEngine for column analysis
        type_engine = TypeInferenceEngine()
        
        column_types = {}
        column_strings = {}
        
        for col in data.columns:
            inference_result = type_engine.infer_type(data[col])
            column_types[col] = inference_result.inferred_type
            column_strings[col] = str(data[col].dtype)
        
        # Calculate quality metrics
        quality_metrics = {
            'completeness': 1.0 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])),
            'consistency': self._calculate_consistency_score(data),
            'uniqueness': self._calculate_uniqueness_score(data)
        }
        
        return SchemaInfo(
            data_format=DataFormat.PANDAS_DATAFRAME,
            structure_type='tabular',
            columns=column_strings,
            column_types=column_types,
            shape=data.shape,
            size_info={
                'rows': data.shape[0],
                'columns': data.shape[1],
                'memory_bytes': data.memory_usage(deep=True).sum()
            },
            null_info={
                'total_nulls': data.isnull().sum().sum(),
                'null_columns': data.columns[data.isnull().any()].tolist(),
                'null_percentage': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            },
            quality_metrics=quality_metrics,
            inference_confidence=np.mean([result.confidence_score for result in 
                                        [type_engine.infer_type(data[col]) for col in data.columns]])
        )
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        # This is a simplified consistency check
        # Could be enhanced with more sophisticated methods
        consistency_scores = []
        
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check string pattern consistency
                non_null_data = data[col].dropna().astype(str)
                if len(non_null_data) > 0:
                    # Simple heuristic: consistency based on length variation
                    lengths = non_null_data.str.len()
                    length_cv = lengths.std() / lengths.mean() if lengths.mean() > 0 else 1
                    consistency_scores.append(max(0, 1 - length_cv))
                else:
                    consistency_scores.append(1.0)
            else:
                # For numeric data, consistency is generally high
                consistency_scores.append(0.9)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_uniqueness_score(self, data: pd.DataFrame) -> float:
        """Calculate data uniqueness score."""
        uniqueness_scores = []
        
        for col in data.columns:
            unique_ratio = data[col].nunique() / len(data) if len(data) > 0 else 1
            uniqueness_scores.append(unique_ratio)
        
        return np.mean(uniqueness_scores) if uniqueness_scores else 1.0


class NumpyArrayDetector(FormatSpecificDetector):
    """Detector for numpy array format."""
    
    def detect_format(self, data: Any) -> Tuple[float, Dict[str, Any]]:
        """Detect numpy array format."""
        if isinstance(data, np.ndarray):
            details = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'ndim': data.ndim,
                'size': data.size,
                'memory_bytes': data.nbytes,
                'is_contiguous': data.flags.c_contiguous
            }
            return 1.0, details
        
        return 0.0, {}
    
    def get_target_format(self) -> DataFormat:
        return DataFormat.NUMPY_ARRAY
    
    def extract_schema(self, data: np.ndarray) -> SchemaInfo:
        """Extract schema from numpy array."""
        quality_metrics = {
            'completeness': 1.0 - (np.isnan(data).sum() / data.size if np.issubdtype(data.dtype, np.floating) else 0.0),
            'consistency': 1.0,  # Arrays are inherently consistent in type
            'density': (data != 0).sum() / data.size if data.size > 0 else 1.0
        }
        
        return SchemaInfo(
            data_format=DataFormat.NUMPY_ARRAY,
            structure_type='array',
            shape=data.shape,
            element_type=str(data.dtype),
            size_info={
                'shape': data.shape,
                'size': data.size,
                'ndim': data.ndim,
                'memory_bytes': data.nbytes
            },
            null_info={
                'nan_count': np.isnan(data).sum() if np.issubdtype(data.dtype, np.floating) else 0,
                'inf_count': np.isinf(data).sum() if np.issubdtype(data.dtype, np.floating) else 0
            },
            quality_metrics=quality_metrics,
            inference_confidence=1.0,
            additional_properties={
                'is_contiguous': data.flags.c_contiguous,
                'byte_order': data.dtype.byteorder,
                'is_writeable': data.flags.writeable
            }
        )


class TimeSeriesDetector(FormatSpecificDetector):
    """Detector for time series data format."""
    
    def detect_format(self, data: Any) -> Tuple[float, Dict[str, Any]]:
        """Detect time series format."""
        confidence = 0.0
        details = {}
        
        if isinstance(data, pd.DataFrame):
            # Check for datetime index
            has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
            
            # Check for datetime columns
            datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Check for temporal patterns in column names
            temporal_keywords = ['time', 'date', 'timestamp', 'period', 'year', 'month', 'day']
            temporal_columns = [col for col in data.columns 
                              if any(keyword in str(col).lower() for keyword in temporal_keywords)]
            
            # Calculate confidence
            if has_datetime_index:
                confidence += 0.7
            if datetime_columns:
                confidence += 0.3 * (len(datetime_columns) / len(data.columns))
            if temporal_columns:
                confidence += 0.2 * (len(temporal_columns) / len(data.columns))
            
            # Check for regular time intervals (for time series)
            if has_datetime_index and len(data) > 2:
                try:
                    intervals = data.index.to_series().diff().dropna()
                    if len(intervals.unique()) <= 3:  # Regular intervals
                        confidence += 0.2
                except:
                    pass
            
            confidence = min(confidence, 1.0)
            
            details = {
                'has_datetime_index': has_datetime_index,
                'datetime_columns': datetime_columns,
                'temporal_columns': temporal_columns,
                'index_type': type(data.index).__name__,
                'potential_frequency': self._infer_frequency(data) if has_datetime_index else None
            }
        
        return confidence, details
    
    def get_target_format(self) -> DataFormat:
        return DataFormat.TIME_SERIES
    
    def extract_schema(self, data: pd.DataFrame) -> SchemaInfo:
        """Extract schema from time series data."""
        # Use pandas detector for base schema
        base_detector = PandasDataFrameDetector()
        base_schema = base_detector.extract_schema(data)
        
        # Enhance with time series specific information
        base_schema.data_format = DataFormat.TIME_SERIES
        base_schema.additional_properties.update({
            'temporal_index': isinstance(data.index, pd.DatetimeIndex),
            'frequency': self._infer_frequency(data),
            'time_range': (data.index.min(), data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else None,
            'missing_time_periods': self._detect_missing_periods(data)
        })
        
        return base_schema
    
    def _infer_frequency(self, data: pd.DataFrame) -> Optional[str]:
        """Infer time series frequency."""
        if isinstance(data.index, pd.DatetimeIndex):
            try:
                return pd.infer_freq(data.index)
            except:
                return None
        return None
    
    def _detect_missing_periods(self, data: pd.DataFrame) -> int:
        """Detect missing time periods."""
        if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
            return 0
        
        try:
            freq = pd.infer_freq(data.index)
            if freq:
                expected_periods = pd.date_range(
                    start=data.index.min(),
                    end=data.index.max(),
                    freq=freq
                )
                return len(expected_periods) - len(data)
        except:
            pass
        
        return 0


class CategoricalDetector(FormatSpecificDetector):
    """Detector for categorical data format."""
    
    def detect_format(self, data: Any) -> Tuple[float, Dict[str, Any]]:
        """Detect categorical format."""
        confidence = 0.0
        details = {}
        
        if isinstance(data, pd.DataFrame):
            categorical_columns = []
            
            for col in data.columns:
                # Check if already categorical
                if data[col].dtype.name == 'category':
                    categorical_columns.append(col)
                else:
                    # Check if should be categorical
                    unique_ratio = data[col].nunique() / len(data) if len(data) > 0 else 0
                    if unique_ratio < 0.1 and data[col].nunique() < 50:  # Low cardinality
                        categorical_columns.append(col)
            
            if categorical_columns:
                confidence = len(categorical_columns) / len(data.columns)
            
            details = {
                'categorical_columns': categorical_columns,
                'cardinality_info': {col: data[col].nunique() for col in categorical_columns},
                'total_categorical_ratio': confidence
            }
        
        elif isinstance(data, pd.Series):
            if data.dtype.name == 'category':
                confidence = 1.0
            else:
                unique_ratio = data.nunique() / len(data) if len(data) > 0 else 0
                if unique_ratio < 0.1 and data.nunique() < 50:
                    confidence = 0.8
            
            details = {
                'unique_values': data.nunique(),
                'unique_ratio': unique_ratio,
                'sample_values': data.unique()[:10].tolist()
            }
        
        return confidence, details
    
    def get_target_format(self) -> DataFormat:
        return DataFormat.CATEGORICAL
    
    def extract_schema(self, data: Any) -> SchemaInfo:
        """Extract schema from categorical data."""
        if isinstance(data, pd.DataFrame):
            base_detector = PandasDataFrameDetector()
            schema = base_detector.extract_schema(data)
            schema.data_format = DataFormat.CATEGORICAL
        else:
            # Handle Series or other data types
            schema = SchemaInfo(
                data_format=DataFormat.CATEGORICAL,
                structure_type='array',
                shape=(len(data),) if hasattr(data, '__len__') else None,
                size_info={'length': len(data) if hasattr(data, '__len__') else 0}
            )
        
        return schema


class TypeDetectionEngine(TypeDetector):
    """
    Enhanced type detection engine building on existing TypeInferenceEngine.
    
    Provides comprehensive format detection across all supported data types
    with confidence scoring and detailed schema inference.
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.7,
                 enable_schema_inference: bool = True,
                 max_sample_size: int = 10000):
        """
        Initialize TypeDetectionEngine.
        
        Args:
            confidence_threshold: Minimum confidence for format detection
            enable_schema_inference: Enable detailed schema extraction
            max_sample_size: Maximum data sample size for analysis
        """
        self.confidence_threshold = confidence_threshold
        self.enable_schema_inference = enable_schema_inference
        self.max_sample_size = max_sample_size
        
        # Initialize base type inference engine
        self._base_engine = TypeInferenceEngine()
        
        # Initialize format-specific detectors
        self._detectors: List[FormatSpecificDetector] = [
            PandasDataFrameDetector(),
            NumpyArrayDetector(),
            TimeSeriesDetector(),
            CategoricalDetector(),
        ]
        
        # Cache for recently detected formats
        self._detection_cache: Dict[str, FormatDetectionResult] = {}
        self._cache_max_size = 100
        
        logger.info("TypeDetectionEngine initialized",
                   confidence_threshold=confidence_threshold,
                   num_detectors=len(self._detectors),
                   enable_schema_inference=enable_schema_inference)
    
    def detect_format(self, data: Any) -> FormatDetectionResult:
        """
        Detect the format of input data with confidence scoring.
        
        Args:
            data: Input data to analyze
            
        Returns:
            FormatDetectionResult with detected format and details
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(data)
        if cache_key in self._detection_cache:
            cached_result = self._detection_cache[cache_key]
            logger.debug("Returned cached format detection result")
            return cached_result
        
        # Sample data for efficient analysis
        sample_data = self._sample_data(data)
        sample_size = len(sample_data) if hasattr(sample_data, '__len__') else 1
        
        # Run all detectors
        detection_results = []
        detection_details = {}
        
        for detector in self._detectors:
            try:
                confidence, details = detector.detect_format(sample_data)
                if confidence > 0:
                    detection_results.append((detector.get_target_format(), confidence, detector))
                    detection_details[detector.get_target_format().value] = details
            except Exception as e:
                logger.warning(f"Detector {detector.__class__.__name__} failed: {e}")
        
        # Sort by confidence
        detection_results.sort(key=lambda x: x[1], reverse=True)
        
        # Determine best format
        if detection_results and detection_results[0][1] >= self.confidence_threshold:
            detected_format = detection_results[0][0]
            confidence_score = detection_results[0][1]
            best_detector = detection_results[0][2]
        else:
            # Fallback detection based on basic type analysis
            detected_format, confidence_score = self._fallback_detection(sample_data)
            best_detector = None
        
        # Extract schema if enabled
        schema_info = None
        if self.enable_schema_inference and best_detector:
            try:
                schema_info = best_detector.extract_schema(sample_data)
            except Exception as e:
                logger.warning(f"Schema extraction failed: {e}")
        
        # Prepare alternative formats
        alternative_formats = [(fmt, conf) for fmt, conf, _ in detection_results[1:5]]  # Top 4 alternatives
        
        # Create result
        result = FormatDetectionResult(
            detected_format=detected_format,
            confidence_score=confidence_score,
            alternative_formats=alternative_formats,
            detection_details=detection_details,
            schema_info=schema_info,
            detection_time=time.time() - start_time,
            sample_size=sample_size
        )
        
        # Cache result
        self._cache_result(cache_key, result)
        
        logger.info("Format detection completed",
                   detected_format=detected_format.value,
                   confidence=confidence_score,
                   detection_time=result.detection_time)
        
        return result
    
    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for detection."""
        return self.confidence_threshold
    
    def validate_format_compatibility(self, 
                                     data: Any, 
                                     expected_format: DataFormat) -> ValidationResult:
        """
        Validate if data is compatible with expected format.
        
        Args:
            data: Data to validate
            expected_format: Expected data format
            
        Returns:
            ValidationResult with compatibility assessment
        """
        detection_result = self.detect_format(data)
        
        errors = []
        warnings = []
        
        if detection_result.detected_format != expected_format:
            # Check if detected format is in alternatives
            alternative_formats = {fmt for fmt, _ in detection_result.alternative_formats}
            
            if expected_format in alternative_formats:
                warnings.append(f"Expected format {expected_format.value} is possible but not the most confident detection")
            else:
                errors.append(f"Data format {detection_result.detected_format.value} incompatible with expected {expected_format.value}")
        
        # Additional compatibility checks based on schema
        if detection_result.schema_info:
            schema_warnings = self._validate_schema_quality(detection_result.schema_info)
            warnings.extend(schema_warnings)
        
        is_valid = len(errors) == 0
        score = detection_result.confidence_score if is_valid else 0.0
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            errors=errors,
            warnings=warnings,
            details={
                'detected_format': detection_result.detected_format.value,
                'detection_confidence': detection_result.confidence_score,
                'alternative_formats': detection_result.alternative_formats
            }
        )
    
    def infer_conversion_requirements(self, 
                                     data: Any,
                                     target_format: DataFormat) -> Dict[str, Any]:
        """
        Infer requirements for converting data to target format.
        
        Args:
            data: Source data
            target_format: Target format
            
        Returns:
            Dictionary with conversion requirements and recommendations
        """
        detection_result = self.detect_format(data)
        
        requirements = {
            'source_format': detection_result.detected_format,
            'target_format': target_format,
            'conversion_needed': detection_result.detected_format != target_format,
            'confidence': detection_result.confidence_score,
            'estimated_complexity': 'low'  # Default
        }
        
        if requirements['conversion_needed']:
            # Determine conversion complexity
            complexity = self._assess_conversion_complexity(
                detection_result.detected_format, 
                target_format,
                detection_result.schema_info
            )
            requirements['estimated_complexity'] = complexity
            
            # Add specific conversion recommendations
            recommendations = self._generate_conversion_recommendations(
                detection_result, target_format
            )
            requirements['recommendations'] = recommendations
        
        return requirements
    
    # Private helper methods
    
    def _sample_data(self, data: Any) -> Any:
        """Sample data for efficient analysis."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if len(data) > self.max_sample_size:
                return data.sample(n=self.max_sample_size, random_state=42)
        elif isinstance(data, np.ndarray):
            if data.size > self.max_sample_size:
                # Sample array preserving shape structure
                if data.ndim == 1:
                    indices = np.random.choice(len(data), self.max_sample_size, replace=False)
                    return data[indices]
                else:
                    # For multidimensional arrays, sample along first dimension
                    max_rows = min(self.max_sample_size // data.shape[1] if data.ndim > 1 else self.max_sample_size, data.shape[0])
                    indices = np.random.choice(data.shape[0], max_rows, replace=False)
                    return data[indices]
        elif hasattr(data, '__len__') and len(data) > self.max_sample_size:
            # For other sequences
            import random
            return random.sample(list(data), self.max_sample_size)
        
        return data
    
    def _fallback_detection(self, data: Any) -> Tuple[DataFormat, float]:
        """Fallback detection for unrecognized formats."""
        # Basic type-based detection
        if isinstance(data, dict):
            return DataFormat.PYTHON_DICT, 0.6
        elif isinstance(data, list):
            return DataFormat.PYTHON_LIST, 0.6
        elif isinstance(data, str):
            # Try to detect specific string formats
            if data.startswith(('http://', 'https://')):
                return DataFormat.JSON, 0.5  # Could be URL data
            elif data.startswith(('{', '[')):
                return DataFormat.JSON, 0.7  # Likely JSON string
            else:
                return DataFormat.PYTHON_DICT, 0.4  # Generic string data
        else:
            return DataFormat.UNKNOWN, 0.1
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate cache key for data."""
        # Simple hash-based key generation
        import hashlib
        
        key_components = [
            str(type(data).__name__),
            str(getattr(data, 'shape', None)),
            str(getattr(data, 'dtypes', None)),
            str(id(data))  # Memory location as fallback
        ]
        
        key_string = "_".join(str(c) for c in key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: FormatDetectionResult):
        """Cache detection result with size management."""
        if len(self._detection_cache) >= self._cache_max_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self._detection_cache))
            del self._detection_cache[oldest_key]
        
        self._detection_cache[cache_key] = result
    
    def _validate_schema_quality(self, schema: SchemaInfo) -> List[str]:
        """Validate schema quality and return warnings."""
        warnings = []
        
        if 'completeness' in schema.quality_metrics:
            if schema.quality_metrics['completeness'] < 0.8:
                warnings.append(f"Low data completeness: {schema.quality_metrics['completeness']:.2%}")
        
        if 'consistency' in schema.quality_metrics:
            if schema.quality_metrics['consistency'] < 0.7:
                warnings.append(f"Low data consistency: {schema.quality_metrics['consistency']:.2%}")
        
        # Check for excessive null values
        if schema.null_info and 'null_percentage' in schema.null_info:
            if schema.null_info['null_percentage'] > 50:
                warnings.append(f"High null value percentage: {schema.null_info['null_percentage']:.1f}%")
        
        return warnings
    
    def _assess_conversion_complexity(self, 
                                     source_format: DataFormat,
                                     target_format: DataFormat,
                                     schema_info: Optional[SchemaInfo]) -> str:
        """Assess complexity of format conversion."""
        if source_format == target_format:
            return 'none'
        
        # Define conversion complexity matrix
        complexity_map = {
            # Same family conversions are typically low complexity
            (DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY): 'low',
            (DataFormat.NUMPY_ARRAY, DataFormat.PANDAS_DATAFRAME): 'low',
            
            # Time series conversions
            (DataFormat.PANDAS_DATAFRAME, DataFormat.TIME_SERIES): 'low',
            (DataFormat.TIME_SERIES, DataFormat.PANDAS_DATAFRAME): 'low',
            
            # Categorical conversions
            (DataFormat.PANDAS_DATAFRAME, DataFormat.CATEGORICAL): 'medium',
            (DataFormat.CATEGORICAL, DataFormat.PANDAS_DATAFRAME): 'medium',
            
            # Complex format conversions
            (DataFormat.JSON, DataFormat.PANDAS_DATAFRAME): 'high',
            (DataFormat.PANDAS_DATAFRAME, DataFormat.JSON): 'medium',
        }
        
        conversion_key = (source_format, target_format)
        if conversion_key in complexity_map:
            return complexity_map[conversion_key]
        
        # Default complexity assessment
        simple_formats = {DataFormat.PANDAS_DATAFRAME, DataFormat.NUMPY_ARRAY}
        complex_formats = {DataFormat.JSON, DataFormat.HIERARCHICAL, DataFormat.STREAMING}
        
        if source_format in complex_formats or target_format in complex_formats:
            return 'high'
        elif source_format in simple_formats and target_format in simple_formats:
            return 'low'
        else:
            return 'medium'
    
    def _generate_conversion_recommendations(self, 
                                           detection_result: FormatDetectionResult,
                                           target_format: DataFormat) -> List[str]:
        """Generate specific recommendations for conversion."""
        recommendations = []
        source_format = detection_result.detected_format
        
        # Format-specific recommendations
        if source_format == DataFormat.PANDAS_DATAFRAME and target_format == DataFormat.NUMPY_ARRAY:
            recommendations.extend([
                "Use .values attribute for direct conversion",
                "Consider handling missing values before conversion",
                "Ensure all columns have compatible numeric types"
            ])
        
        elif source_format == DataFormat.NUMPY_ARRAY and target_format == DataFormat.PANDAS_DATAFRAME:
            recommendations.extend([
                "Provide appropriate column names",
                "Consider setting proper index",
                "Specify data types if needed"
            ])
        
        elif target_format == DataFormat.TIME_SERIES:
            recommendations.extend([
                "Ensure datetime index is properly set",
                "Consider frequency inference for regular time series",
                "Handle missing time periods if necessary"
            ])
        
        elif target_format == DataFormat.CATEGORICAL:
            recommendations.extend([
                "Identify high-cardinality columns for categorical conversion",
                "Consider ordered vs unordered categories",
                "Validate category levels before conversion"
            ])
        
        # Schema-based recommendations
        if detection_result.schema_info:
            schema = detection_result.schema_info
            
            if 'completeness' in schema.quality_metrics:
                if schema.quality_metrics['completeness'] < 0.9:
                    recommendations.append("Address missing values before conversion")
            
            if schema.null_info and 'null_percentage' in schema.null_info:
                if schema.null_info['null_percentage'] > 20:
                    recommendations.append("High null percentage detected - consider imputation strategies")
        
        # Confidence-based recommendations
        if detection_result.confidence_score < 0.8:
            recommendations.append("Low format detection confidence - manually verify data format")
        
        return recommendations


# Utility function for easy format detection
def detect_data_format(data: Any, 
                      confidence_threshold: float = 0.7,
                      include_schema: bool = True) -> FormatDetectionResult:
    """
    Convenient function for data format detection.
    
    Args:
        data: Data to analyze
        confidence_threshold: Minimum confidence threshold
        include_schema: Whether to include schema inference
        
    Returns:
        FormatDetectionResult
    """
    engine = TypeDetectionEngine(
        confidence_threshold=confidence_threshold,
        enable_schema_inference=include_schema
    )
    return engine.detect_format(data)