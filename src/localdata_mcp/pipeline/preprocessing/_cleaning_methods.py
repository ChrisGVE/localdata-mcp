"""
Data Cleaning Pipeline Implementation Methods - Part 1.

Contains mixin methods for the DataCleaningPipeline class:
quality assessment, type inference, missing value handling,
and advanced outlier detection.
"""

import re
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

from ...logging_manager import get_logger
from ..missing_value_handler import MissingValueHandler

logger = get_logger(__name__)


def _assess_initial_quality(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assess initial data quality before cleaning."""
    self._quality_metrics_before = self._calculate_comprehensive_quality_metrics(data)

    metadata = {
        "parameters": {},
        "records_affected": 0,
        "quality_assessment": {
            "overall_score": self._quality_metrics_before.overall_quality_score,
            "completeness": self._quality_metrics_before.completeness_score,
            "consistency": self._quality_metrics_before.consistency_score,
            "validity": self._quality_metrics_before.validity_score,
            "accuracy": self._quality_metrics_before.accuracy_score,
        },
    }

    logger.info(
        "Initial quality assessment completed",
        overall_score=self._quality_metrics_before.overall_quality_score,
    )

    return data.copy(), metadata


def _basic_type_inference(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Basic data type inference and conversion."""
    result_data = data.copy()
    type_conversions = {}
    records_affected = 0

    for col in data.columns:
        if data[col].dtype == "object":
            # Try numeric conversion
            try:
                numeric_series = pd.to_numeric(data[col], errors="coerce")
                success_rate = numeric_series.notna().sum() / len(numeric_series)
                if success_rate > 0.8:
                    result_data[col] = numeric_series
                    type_conversions[col] = {
                        "from": "object",
                        "to": str(numeric_series.dtype),
                    }
                    records_affected += len(result_data)
                    continue
            except Exception:
                pass

            # Try datetime conversion
            try:
                datetime_series = pd.to_datetime(data[col], errors="coerce")
                success_rate = datetime_series.notna().sum() / len(datetime_series)
                if success_rate > 0.8:
                    result_data[col] = datetime_series
                    type_conversions[col] = {
                        "from": "object",
                        "to": "datetime64[ns]",
                    }
                    records_affected += len(result_data)
            except Exception:
                pass

    metadata = {
        "parameters": {"inference_threshold": 0.8},
        "records_affected": records_affected,
        "type_conversions": type_conversions,
        "reversibility_data": {"original_dtypes": dict(data.dtypes)},
    }

    return result_data, metadata


def _comprehensive_type_inference(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Comprehensive intelligent data type inference."""
    from ._dataclasses import TransformationStrategy

    result_data = data.copy()
    strategies = TransformationStrategy.data_type_inference_strategy(data)
    type_conversions = {}
    records_affected = 0

    for col, strategy in strategies.items():
        if strategy == "datetime":
            try:
                datetime_series = pd.to_datetime(
                    data[col], infer_datetime_format=True, errors="coerce"
                )
                if datetime_series.notna().sum() / len(datetime_series) > 0.7:
                    result_data[col] = datetime_series
                    type_conversions[col] = {
                        "from": str(data[col].dtype),
                        "to": "datetime64[ns]",
                    }
                    records_affected += len(result_data)
            except Exception:
                pass
        elif strategy == "numeric":
            try:
                numeric_series = pd.to_numeric(data[col], errors="coerce")
                if numeric_series.notna().sum() / len(numeric_series) > 0.7:
                    result_data[col] = numeric_series
                    type_conversions[col] = {
                        "from": str(data[col].dtype),
                        "to": str(numeric_series.dtype),
                    }
                    records_affected += len(result_data)
            except Exception:
                pass
        elif strategy == "categorical":
            if data[col].nunique() < len(data) * 0.5:
                result_data[col] = data[col].astype("category")
                type_conversions[col] = {
                    "from": str(data[col].dtype),
                    "to": "category",
                }
                records_affected += len(result_data)

    metadata = {
        "parameters": {"strategies": strategies},
        "records_affected": records_affected,
        "type_conversions": type_conversions,
        "reversibility_data": {"original_dtypes": dict(data.dtypes)},
    }

    return result_data, metadata


def _handle_basic_missing_values(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Basic missing value handling using simple strategies."""
    result_data = data.copy()
    imputation_log = {}
    records_affected = 0

    numeric_cols = data.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            median_val = data[col].median()
            result_data[col].fillna(median_val, inplace=True)
            imputation_log[col] = {"method": "median", "value": median_val}
            records_affected += data[col].isnull().sum()

    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode_val = (
                data[col].mode().iloc[0] if not data[col].mode().empty else "unknown"
            )
            result_data[col].fillna(mode_val, inplace=True)
            imputation_log[col] = {"method": "mode", "value": mode_val}
            records_affected += data[col].isnull().sum()

    metadata = {
        "parameters": {"strategy": "basic"},
        "records_affected": records_affected,
        "imputation_log": imputation_log,
        "reversibility_data": {"missing_positions": data.isnull().to_dict()},
    }

    return result_data, metadata


def _intelligent_missing_value_handling(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Intelligent missing value handling using MissingValueHandler."""
    if self.cleaning_intensity == "minimal":
        complexity = "minimal"
    elif self.cleaning_intensity == "comprehensive":
        complexity = "comprehensive"
    else:
        complexity = "auto"

    missing_handler = MissingValueHandler(
        analytical_intention=(f"Handle missing values for {self.analytical_intention}"),
        strategy="auto",
        complexity=complexity,
        cross_validation=complexity in ["auto", "comprehensive"],
        metadata_tracking=True,
        streaming_config=self.streaming_config,
        custom_parameters={
            "quality_thresholds": {
                "min_accuracy": 0.7,
                "max_mse_increase": 0.2,
                "min_correlation_preservation": 0.8,
                "max_distribution_deviation": 0.1,
            }
        },
    )

    try:
        result_data, handler_metadata = missing_handler.analyze(data)

        imputation_results = handler_metadata.get("imputation_results", {})
        missing_analysis = handler_metadata.get("missing_value_analysis", {})

        records_affected = imputation_results.get(
            "original_missing_values", 0
        ) - imputation_results.get("final_missing_values", 0)

        metadata = {
            "parameters": {
                "strategy": "sophisticated_sklearn",
                "complexity": complexity,
                "pattern_detected": missing_analysis.get("pattern_type", "unknown"),
                "pattern_confidence": missing_analysis.get("pattern_confidence", 0.0),
            },
            "records_affected": records_affected,
            "imputation_log": {
                "missing_pattern": missing_analysis.get("pattern_type", "unknown"),
                "strategy_used": handler_metadata.get("imputation_pipeline", {}).get(
                    "strategy", "auto"
                ),
                "cross_validation": handler_metadata.get("imputation_pipeline", {}).get(
                    "cross_validation", False
                ),
                "imputation_complete": imputation_results.get(
                    "imputation_complete", False
                ),
                "columns_imputed": imputation_results.get("columns_imputed", 0),
            },
            "quality_assessment": handler_metadata.get("quality_assessment", {}),
            "reversibility_data": {
                "missing_positions": data.isnull().to_dict(),
                "imputation_artifacts": handler_metadata.get(
                    "composition_context", {}
                ).get("imputation_artifacts", {}),
            },
            "sophisticated_handler_metadata": handler_metadata,
        }

        logger.info(
            "Sophisticated missing value handling completed",
            pattern_type=missing_analysis.get("pattern_type"),
            records_affected=records_affected,
            imputation_complete=imputation_results.get("imputation_complete", False),
        )

        return result_data, metadata

    except Exception as e:
        logger.error(f"Sophisticated missing value handling failed: {e}")
        return _handle_basic_missing_values(self, data)
