"""Type detection - Format-specific detectors."""

from typing import Any, Dict, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..interfaces import DataFormat
from ...type_conversion import TypeInferenceEngine
from ._types import SchemaInfo


class FormatSpecificDetector(ABC):
    """Abstract base class for format-specific detectors."""

    @abstractmethod
    def detect_format(self, data: Any) -> Tuple[float, Dict[str, Any]]:
        """Detect if data matches this detector's format."""
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
                "shape": data.shape,
                "dtypes": data.dtypes.to_dict(),
                "memory_usage": data.memory_usage(deep=True).sum(),
                "index_type": type(data.index).__name__,
                "columns_count": len(data.columns),
            }
            return 1.0, details
        return 0.0, {}

    def get_target_format(self) -> DataFormat:
        return DataFormat.PANDAS_DATAFRAME

    def extract_schema(self, data: pd.DataFrame) -> SchemaInfo:
        """Extract schema from DataFrame."""
        type_engine = TypeInferenceEngine()
        column_types = {}
        column_strings = {}

        for col in data.columns:
            inference_result = type_engine.infer_type(data[col])
            column_types[col] = inference_result.inferred_type
            column_strings[col] = str(data[col].dtype)

        quality_metrics = {
            "completeness": 1.0
            - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])),
            "consistency": self._calculate_consistency_score(data),
            "uniqueness": self._calculate_uniqueness_score(data),
        }

        return SchemaInfo(
            data_format=DataFormat.PANDAS_DATAFRAME,
            structure_type="tabular",
            columns=column_strings,
            column_types=column_types,
            shape=data.shape,
            size_info={
                "rows": data.shape[0],
                "columns": data.shape[1],
                "memory_bytes": data.memory_usage(deep=True).sum(),
            },
            null_info={
                "total_nulls": data.isnull().sum().sum(),
                "null_columns": data.columns[data.isnull().any()].tolist(),
                "null_percentage": (
                    data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                )
                * 100,
            },
            quality_metrics=quality_metrics,
            inference_confidence=np.mean(
                [
                    result.confidence_score
                    for result in [
                        type_engine.infer_type(data[col]) for col in data.columns
                    ]
                ]
            ),
        )

    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        consistency_scores = []
        for col in data.columns:
            if data[col].dtype == "object":
                non_null_data = data[col].dropna().astype(str)
                if len(non_null_data) > 0:
                    lengths = non_null_data.str.len()
                    length_cv = (
                        lengths.std() / lengths.mean() if lengths.mean() > 0 else 1
                    )
                    consistency_scores.append(max(0, 1 - length_cv))
                else:
                    consistency_scores.append(1.0)
            else:
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
                "shape": data.shape,
                "dtype": str(data.dtype),
                "ndim": data.ndim,
                "size": data.size,
                "memory_bytes": data.nbytes,
                "is_contiguous": data.flags.c_contiguous,
            }
            return 1.0, details
        return 0.0, {}

    def get_target_format(self) -> DataFormat:
        return DataFormat.NUMPY_ARRAY

    def extract_schema(self, data: np.ndarray) -> SchemaInfo:
        """Extract schema from numpy array."""
        quality_metrics = {
            "completeness": 1.0
            - (
                np.isnan(data).sum() / data.size
                if np.issubdtype(data.dtype, np.floating)
                else 0.0
            ),
            "consistency": 1.0,
            "density": (data != 0).sum() / data.size if data.size > 0 else 1.0,
        }
        return SchemaInfo(
            data_format=DataFormat.NUMPY_ARRAY,
            structure_type="array",
            shape=data.shape,
            element_type=str(data.dtype),
            size_info={
                "shape": data.shape,
                "size": data.size,
                "ndim": data.ndim,
                "memory_bytes": data.nbytes,
            },
            null_info={
                "nan_count": np.isnan(data).sum()
                if np.issubdtype(data.dtype, np.floating)
                else 0,
                "inf_count": np.isinf(data).sum()
                if np.issubdtype(data.dtype, np.floating)
                else 0,
            },
            quality_metrics=quality_metrics,
            inference_confidence=1.0,
            additional_properties={
                "is_contiguous": data.flags.c_contiguous,
                "byte_order": data.dtype.byteorder,
                "is_writeable": data.flags.writeable,
            },
        )


class TimeSeriesDetector(FormatSpecificDetector):
    """Detector for time series data format."""

    def detect_format(self, data: Any) -> Tuple[float, Dict[str, Any]]:
        """Detect time series format."""
        confidence = 0.0
        details = {}

        if isinstance(data, pd.DataFrame):
            has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
            datetime_columns = data.select_dtypes(
                include=["datetime64"]
            ).columns.tolist()
            temporal_keywords = [
                "time",
                "date",
                "timestamp",
                "period",
                "year",
                "month",
                "day",
            ]
            temporal_columns = [
                col
                for col in data.columns
                if any(keyword in str(col).lower() for keyword in temporal_keywords)
            ]

            if has_datetime_index:
                confidence += 0.7
            if datetime_columns:
                confidence += 0.3 * (len(datetime_columns) / len(data.columns))
            if temporal_columns:
                confidence += 0.2 * (len(temporal_columns) / len(data.columns))

            if has_datetime_index and len(data) > 2:
                try:
                    intervals = data.index.to_series().diff().dropna()
                    if len(intervals.unique()) <= 3:
                        confidence += 0.2
                except:
                    pass

            confidence = min(confidence, 1.0)
            details = {
                "has_datetime_index": has_datetime_index,
                "datetime_columns": datetime_columns,
                "temporal_columns": temporal_columns,
                "index_type": type(data.index).__name__,
                "potential_frequency": self._infer_frequency(data)
                if has_datetime_index
                else None,
            }

        return confidence, details

    def get_target_format(self) -> DataFormat:
        return DataFormat.TIME_SERIES

    def extract_schema(self, data: pd.DataFrame) -> SchemaInfo:
        """Extract schema from time series data."""
        base_detector = PandasDataFrameDetector()
        base_schema = base_detector.extract_schema(data)
        base_schema.data_format = DataFormat.TIME_SERIES
        base_schema.additional_properties.update(
            {
                "temporal_index": isinstance(data.index, pd.DatetimeIndex),
                "frequency": self._infer_frequency(data),
                "time_range": (data.index.min(), data.index.max())
                if isinstance(data.index, pd.DatetimeIndex)
                else None,
                "missing_time_periods": self._detect_missing_periods(data),
            }
        )
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
                    start=data.index.min(), end=data.index.max(), freq=freq
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
                if data[col].dtype.name == "category":
                    categorical_columns.append(col)
                else:
                    unique_ratio = (
                        data[col].nunique() / len(data) if len(data) > 0 else 0
                    )
                    if unique_ratio <= 0.2 and data[col].nunique() < 50:
                        categorical_columns.append(col)

            if categorical_columns:
                confidence = len(categorical_columns) / len(data.columns)

            details = {
                "categorical_columns": categorical_columns,
                "cardinality_info": {
                    col: data[col].nunique() for col in categorical_columns
                },
                "total_categorical_ratio": confidence,
            }

        elif isinstance(data, pd.Categorical):
            confidence = 1.0
            details = {
                "unique_values": len(data.categories),
                "unique_ratio": len(data.categories) / len(data)
                if len(data) > 0
                else 0,
                "sample_values": list(data.categories[:10]),
            }

        elif isinstance(data, pd.Series):
            if data.dtype.name == "category":
                confidence = 1.0
            else:
                unique_ratio = data.nunique() / len(data) if len(data) > 0 else 0
                if unique_ratio <= 0.2 and data.nunique() < 50:
                    confidence = 0.8

            details = {
                "unique_values": data.nunique(),
                "unique_ratio": unique_ratio
                if data.dtype.name != "category"
                else (data.nunique() / len(data) if len(data) > 0 else 0),
                "sample_values": data.unique()[:10].tolist(),
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
            schema = SchemaInfo(
                data_format=DataFormat.CATEGORICAL,
                structure_type="array",
                shape=(len(data),) if hasattr(data, "__len__") else None,
                size_info={"length": len(data) if hasattr(data, "__len__") else 0},
            )
        return schema
