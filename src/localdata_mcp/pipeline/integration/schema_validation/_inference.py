"""
SchemaInferenceEngine class.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from ..interfaces import DataFormat
from ..type_detection import TypeDetectionEngine
from ._schema import DataSchema
from ._types import (
    SchemaConstraint,
    SchemaInferenceResult,
    ValidationRuleType,
)


class SchemaInferenceEngine:
    """Engine for automatically inferring data schemas from samples."""

    def __init__(self, type_detection_engine: Optional[TypeDetectionEngine] = None):
        self.type_detection_engine = type_detection_engine or TypeDetectionEngine()
        self.inference_strategies = {
            DataFormat.PANDAS_DATAFRAME: self._infer_dataframe_schema,
            DataFormat.NUMPY_ARRAY: self._infer_numpy_schema,
            DataFormat.TIME_SERIES: self._infer_timeseries_schema,
            DataFormat.SCIPY_SPARSE: self._infer_sparse_schema,
        }

    def infer_schema(
        self,
        data: Any,
        data_format: Optional[DataFormat] = None,
        sample_size: Optional[int] = None,
        confidence_threshold: float = 0.7,
    ) -> SchemaInferenceResult:
        """
        Infer schema from data sample with confidence scoring.

        Args:
            data: Data sample to analyze
            data_format: Optional explicit format hint
            sample_size: Maximum sample size for analysis
            confidence_threshold: Minimum confidence for schema acceptance

        Returns:
            SchemaInferenceResult with inferred schema and metrics
        """
        start_time = time.time()

        # Detect data format if not provided
        if data_format is None:
            format_result = self.type_detection_engine.detect_format(data)
            data_format = format_result.detected_format

        # Sample data if necessary
        sampled_data, actual_sample_size = self._sample_data(data, sample_size)

        # Infer schema using appropriate strategy
        inference_func = self.inference_strategies.get(
            data_format, self._infer_generic_schema
        )
        schema, confidence, details = inference_func(sampled_data)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(sampled_data, schema)

        inference_time = time.time() - start_time

        result = SchemaInferenceResult(
            inferred_schema=schema,
            confidence_score=confidence,
            inference_details=details,
            sample_size=actual_sample_size,
            inference_time=inference_time,
            data_quality_score=quality_metrics.get("data_quality", 1.0),
            completeness_score=quality_metrics.get("completeness", 1.0),
            consistency_score=quality_metrics.get("consistency", 1.0),
        )

        # Add warnings for low confidence
        if confidence < confidence_threshold:
            result.warnings.append(
                f"Schema inference confidence ({confidence:.2f}) below threshold ({confidence_threshold})"
            )

        return result

    def _sample_data(self, data: Any, sample_size: Optional[int]) -> Tuple[Any, int]:
        """Sample data for schema inference."""
        if sample_size is None:
            return data, self._get_data_size(data)

        if isinstance(data, pd.DataFrame):
            if len(data) <= sample_size:
                return data, len(data)
            else:
                return data.sample(n=sample_size), sample_size
        elif isinstance(data, np.ndarray):
            if data.shape[0] <= sample_size:
                return data, data.shape[0]
            else:
                indices = np.random.choice(data.shape[0], sample_size, replace=False)
                return data[indices], sample_size
        else:
            return data, self._get_data_size(data)

    def _get_data_size(self, data: Any) -> int:
        """Get size of data sample."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return len(data)
        elif isinstance(data, np.ndarray):
            return data.shape[0]
        elif isinstance(data, (list, tuple)):
            return len(data)
        else:
            return 1

    def _infer_dataframe_schema(
        self, data: pd.DataFrame
    ) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Infer schema from pandas DataFrame."""
        schema_id = f"df_schema_{int(time.time())}"

        fields = {}
        constraints = []

        for col in data.columns:
            column_data = data[col]

            # Infer column type
            dtype = column_data.dtype
            if pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_integer_dtype(dtype):
                    field_type = "int"
                else:
                    field_type = "float"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                field_type = "datetime"
            elif pd.api.types.is_bool_dtype(dtype):
                field_type = "bool"
            else:
                field_type = "str"

            # Analyze column properties
            null_count = column_data.isnull().sum()
            total_count = len(column_data)
            nullable = null_count > 0

            field_schema = {
                "type": field_type,
                "nullable": nullable,
                "null_percentage": null_count / total_count if total_count > 0 else 0,
            }

            # Add constraints for numeric fields
            if field_type in ["int", "float"]:
                non_null_data = column_data.dropna()
                if len(non_null_data) > 0:
                    field_schema["min"] = float(non_null_data.min())
                    field_schema["max"] = float(non_null_data.max())
                    field_schema["mean"] = float(non_null_data.mean())
                    field_schema["std"] = (
                        float(non_null_data.std()) if len(non_null_data) > 1 else 0.0
                    )

                    # Add range constraints
                    min_constraint = SchemaConstraint(
                        constraint_id=f"{col}_min",
                        constraint_type=ValidationRuleType.RANGE_CHECK,
                        field_name=col,
                        constraint_value=field_schema["min"],
                        description=f"Minimum value constraint for {col}",
                        is_required=False,
                    )
                    constraints.append(min_constraint)

            # Add null constraints
            if not nullable:
                null_constraint = SchemaConstraint(
                    constraint_id=f"{col}_not_null",
                    constraint_type=ValidationRuleType.NULL_CHECK,
                    field_name=col,
                    constraint_value=False,
                    description=f"Not null constraint for {col}",
                )
                constraints.append(null_constraint)

            fields[col] = field_schema

        schema = DataSchema(
            schema_id=schema_id,
            schema_name="Inferred DataFrame Schema",
            data_format=DataFormat.PANDAS_DATAFRAME,
            fields=fields,
            constraints=constraints,
            description=f"Auto-inferred schema for DataFrame with {len(fields)} columns",
        )

        # Calculate confidence based on data quality
        confidence = self._calculate_dataframe_confidence(data, fields)

        details = {
            "column_count": len(fields),
            "row_count": len(data),
            "numeric_columns": sum(
                1 for f in fields.values() if f["type"] in ["int", "float"]
            ),
            "categorical_columns": sum(
                1 for f in fields.values() if f["type"] == "str"
            ),
            "datetime_columns": sum(
                1 for f in fields.values() if f["type"] == "datetime"
            ),
        }

        return schema, confidence, details

    def _infer_numpy_schema(
        self, data: np.ndarray
    ) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Infer schema from numpy array."""
        schema_id = f"np_schema_{int(time.time())}"

        # Determine array properties
        shape = data.shape
        dtype = data.dtype

        # Map numpy dtype to schema type
        if np.issubdtype(dtype, np.integer):
            element_type = "int"
        elif np.issubdtype(dtype, np.floating):
            element_type = "float"
        elif np.issubdtype(dtype, np.bool_):
            element_type = "bool"
        elif np.issubdtype(dtype, np.datetime64):
            element_type = "datetime"
        else:
            element_type = "object"

        fields = {
            "array_data": {
                "type": element_type,
                "shape": shape,
                "ndim": data.ndim,
                "size": data.size,
                "nullable": False,
            }
        }

        # Add statistical constraints for numeric data
        constraints = []
        if element_type in ["int", "float"]:
            flat_data = data.flatten()
            finite_data = flat_data[np.isfinite(flat_data)]

            if len(finite_data) > 0:
                min_val = float(finite_data.min())
                max_val = float(finite_data.max())

                constraints.append(
                    SchemaConstraint(
                        constraint_id="array_min",
                        constraint_type=ValidationRuleType.RANGE_CHECK,
                        field_name="array_data",
                        constraint_value=min_val,
                        description="Minimum value constraint for array data",
                    )
                )

        schema = DataSchema(
            schema_id=schema_id,
            schema_name="Inferred NumPy Array Schema",
            data_format=DataFormat.NUMPY_ARRAY,
            fields=fields,
            constraints=constraints,
            description=f"Auto-inferred schema for {element_type} array with shape {shape}",
        )

        confidence = 0.9 if element_type != "object" else 0.7

        details = {
            "shape": shape,
            "dtype": str(dtype),
            "element_type": element_type,
            "total_elements": data.size,
            "memory_usage": data.nbytes,
        }

        return schema, confidence, details

    def _infer_timeseries_schema(
        self, data: Any
    ) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Infer schema from time series data."""
        # This is a placeholder implementation
        # In practice, this would analyze time series specific properties
        schema_id = f"ts_schema_{int(time.time())}"

        if isinstance(data, pd.DataFrame):
            return self._infer_dataframe_schema(data)
        elif isinstance(data, pd.Series):
            fields = {
                "timestamp": {"type": "datetime", "nullable": False},
                "value": {"type": "float", "nullable": True},
            }

            schema = DataSchema(
                schema_id=schema_id,
                schema_name="Inferred Time Series Schema",
                data_format=DataFormat.TIME_SERIES,
                fields=fields,
                description="Auto-inferred schema for time series data",
            )

            return schema, 0.8, {"series_length": len(data)}
        else:
            return self._infer_generic_schema(data)

    def _infer_sparse_schema(
        self, data: sparse.spmatrix
    ) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Infer schema from scipy sparse matrix."""
        schema_id = f"sparse_schema_{int(time.time())}"

        fields = {
            "sparse_data": {
                "type": "float",
                "shape": data.shape,
                "format": data.format,
                "nnz": data.nnz,
                "density": (
                    data.nnz / (data.shape[0] * data.shape[1])
                    if data.shape[0] * data.shape[1] > 0
                    else 0
                ),
                "nullable": True,
            }
        }

        schema = DataSchema(
            schema_id=schema_id,
            schema_name="Inferred Sparse Matrix Schema",
            data_format=DataFormat.SCIPY_SPARSE,
            fields=fields,
            description=f"Auto-inferred schema for sparse matrix ({data.format}, {data.shape})",
        )

        confidence = 0.85
        details = {
            "shape": data.shape,
            "format": data.format,
            "nnz": data.nnz,
            "density": fields["sparse_data"]["density"],
        }

        return schema, confidence, details

    def _infer_generic_schema(
        self, data: Any
    ) -> Tuple[DataSchema, float, Dict[str, Any]]:
        """Fallback schema inference for unknown data types."""
        schema_id = f"generic_schema_{int(time.time())}"

        fields = {
            "data": {
                "type": "object",
                "python_type": type(data).__name__,
                "nullable": True,
            }
        }

        schema = DataSchema(
            schema_id=schema_id,
            schema_name="Generic Inferred Schema",
            data_format=DataFormat.UNKNOWN,
            fields=fields,
            description=f"Generic schema for {type(data).__name__} data",
        )

        confidence = 0.3  # Low confidence for generic schema
        details = {"data_type": type(data).__name__}

        return schema, confidence, details

    def _calculate_dataframe_confidence(
        self, data: pd.DataFrame, fields: Dict[str, Dict]
    ) -> float:
        """Calculate confidence score for DataFrame schema inference."""
        if len(data) == 0:
            return 0.1

        # Base confidence
        confidence = 0.8

        # Adjust based on data quality indicators
        total_cells = len(data) * len(data.columns)
        null_cells = data.isnull().sum().sum()
        null_ratio = null_cells / total_cells if total_cells > 0 else 0

        # Lower confidence for high null ratios
        if null_ratio > 0.5:
            confidence -= 0.3
        elif null_ratio > 0.2:
            confidence -= 0.1

        # Adjust based on type consistency
        type_consistency_score = self._calculate_type_consistency(data)
        confidence += (type_consistency_score - 0.5) * 0.4

        return max(0.1, min(1.0, confidence))

    def _calculate_type_consistency(self, data: pd.DataFrame) -> float:
        """Calculate type consistency score for DataFrame."""
        if len(data.columns) == 0:
            return 1.0

        consistent_columns = 0
        for col in data.columns:
            column_data = data[col].dropna()
            if len(column_data) == 0:
                continue

            # Check if all non-null values have consistent types
            first_type = type(column_data.iloc[0])
            if all(isinstance(val, first_type) for val in column_data):
                consistent_columns += 1

        return consistent_columns / len(data.columns)

    def _calculate_quality_metrics(
        self, data: Any, schema: DataSchema
    ) -> Dict[str, float]:
        """Calculate data quality metrics."""
        metrics = {"data_quality": 1.0, "completeness": 1.0, "consistency": 1.0}

        if isinstance(data, pd.DataFrame):
            # Calculate completeness (ratio of non-null values)
            total_cells = len(data) * len(data.columns)
            non_null_cells = total_cells - data.isnull().sum().sum()
            metrics["completeness"] = (
                non_null_cells / total_cells if total_cells > 0 else 1.0
            )

            # Calculate consistency (type consistency score)
            metrics["consistency"] = self._calculate_type_consistency(data)

            # Overall data quality combines completeness and consistency
            metrics["data_quality"] = (
                metrics["completeness"] + metrics["consistency"]
            ) / 2

        return metrics
