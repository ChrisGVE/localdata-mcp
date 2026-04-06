"""
Data Cleaning Pipeline Implementation Methods - Part 3.

Contains mixin methods for the DataCleaningPipeline class:
comprehensive validation, consistency enhancement, feature cleanup,
final optimization, and final quality assessment.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ...logging_manager import get_logger

logger = get_logger(__name__)


def _comprehensive_data_validation(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Comprehensive data validation with configurable business rules."""
    result_data = data.copy()
    validation_log = {}
    records_affected = 0

    # Apply basic validation first
    result_data, basic_metadata = self._basic_data_validation(result_data)
    validation_log.update(basic_metadata["validation_log"])
    records_affected += basic_metadata["records_affected"]

    # Apply custom business rules
    for rule in self.business_rules:
        try:
            rule_type = rule.get("type")
            column = rule.get("column")
            parameters = rule.get("parameters", {})

            if rule_type == "range_validation" and column in result_data.columns:
                min_val = parameters.get("min")
                max_val = parameters.get("max")

                if min_val is not None:
                    violation_mask = result_data[column] < min_val
                    if violation_mask.sum() > 0:
                        action = parameters.get("action", "set_to_min")
                        if action == "set_to_min":
                            result_data.loc[violation_mask, column] = min_val
                        elif action == "set_to_null":
                            result_data.loc[violation_mask, column] = pd.NA
                        records_affected += violation_mask.sum()

                if max_val is not None:
                    violation_mask = result_data[column] > max_val
                    if violation_mask.sum() > 0:
                        action = parameters.get("action", "set_to_max")
                        if action == "set_to_max":
                            result_data.loc[violation_mask, column] = max_val
                        elif action == "set_to_null":
                            result_data.loc[violation_mask, column] = pd.NA
                        records_affected += violation_mask.sum()

                validation_log[f"{column}_range"] = {
                    "rule": "range_validation",
                    "parameters": parameters,
                    "violations": (
                        violation_mask.sum() if "violation_mask" in locals() else 0
                    ),
                }

            elif rule_type == "pattern_validation" and column in result_data.columns:
                pattern = parameters.get("pattern")
                if pattern:
                    violation_mask = (
                        ~result_data[column].astype(str).str.match(pattern, na=False)
                    )
                    if violation_mask.sum() > 0:
                        action = parameters.get("action", "set_to_null")
                        if action == "set_to_null":
                            result_data.loc[violation_mask, column] = pd.NA
                        records_affected += violation_mask.sum()

                    validation_log[f"{column}_pattern"] = {
                        "rule": "pattern_validation",
                        "pattern": pattern,
                        "violations": violation_mask.sum(),
                    }

        except Exception as e:
            logger.warning(f"Business rule validation failed: {e}")

    metadata = {
        "parameters": {
            "validation_level": "comprehensive",
            "business_rules_count": len(self.business_rules),
        },
        "records_affected": records_affected,
        "validation_log": validation_log,
    }

    return result_data, metadata


def _data_consistency_enhancement(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enhance data consistency across columns and relationships."""
    result_data = data.copy()
    consistency_log = {}
    records_affected = 0

    categorical_cols = result_data.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if result_data[col].dtype == "object":
            original_unique = result_data[col].nunique()
            result_data[col] = result_data[col].astype(str).str.strip().str.lower()
            new_unique = result_data[col].nunique()

            if original_unique != new_unique:
                consistency_log[col] = {
                    "operation": "case_standardization",
                    "before_unique": original_unique,
                    "after_unique": new_unique,
                    "values_consolidated": original_unique - new_unique,
                }
                records_affected += len(result_data)

    date_cols = result_data.select_dtypes(include=["datetime64"]).columns
    if len(date_cols) >= 2:
        start_cols = [
            c for c in date_cols if "start" in c.lower() or "begin" in c.lower()
        ]
        end_cols = [c for c in date_cols if "end" in c.lower() or "finish" in c.lower()]

        for start_col in start_cols:
            for end_col in end_cols:
                inconsistent_mask = result_data[start_col] > result_data[end_col]
                if inconsistent_mask.sum() > 0:
                    temp = result_data.loc[inconsistent_mask, start_col].copy()
                    result_data.loc[inconsistent_mask, start_col] = result_data.loc[
                        inconsistent_mask, end_col
                    ]
                    result_data.loc[inconsistent_mask, end_col] = temp

                    consistency_log[f"{start_col}_{end_col}"] = {
                        "operation": "date_order_fix",
                        "inconsistencies": inconsistent_mask.sum(),
                    }
                    records_affected += inconsistent_mask.sum()

    metadata = {
        "parameters": {"operation": "consistency_enhancement"},
        "records_affected": records_affected,
        "consistency_log": consistency_log,
    }

    return result_data, metadata


def _feature_engineering_cleanup(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove constant/near-constant columns and highly correlated features."""
    result_data = data.copy()
    cleanup_log = {}

    constant_cols = [
        col for col in result_data.columns if result_data[col].nunique() <= 1
    ]
    if constant_cols:
        result_data = result_data.drop(columns=constant_cols)
        cleanup_log["constant_columns_removed"] = constant_cols

    near_constant_cols = []
    for col in result_data.columns:
        if result_data[col].dtype in ["object", "category"]:
            if len(result_data[col].value_counts()) > 0:
                most_frequent_pct = (
                    result_data[col].value_counts(normalize=True).iloc[0]
                )
                if most_frequent_pct > 0.95:
                    near_constant_cols.append(col)

    if near_constant_cols:
        result_data = result_data.drop(columns=near_constant_cols)
        cleanup_log["near_constant_columns_removed"] = near_constant_cols

    numeric_cols = result_data.select_dtypes(include=["number"]).columns
    highly_corr_features = []

    if len(numeric_cols) > 1:
        try:
            corr_matrix = result_data[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            highly_corr_features = [
                column for column in upper_tri.columns if any(upper_tri[column] > 0.95)
            ]
            if highly_corr_features:
                result_data = result_data.drop(columns=highly_corr_features)
                cleanup_log["highly_correlated_features_removed"] = highly_corr_features
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")

    total_removed = (
        len(constant_cols) + len(near_constant_cols) + len(highly_corr_features)
    )

    metadata = {
        "parameters": {
            "correlation_threshold": 0.95,
            "constant_threshold": 0.95,
        },
        "records_affected": 0,
        "cleanup_log": cleanup_log,
        "columns_removed": total_removed,
    }

    return result_data, metadata


def _final_quality_optimization(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Final optimization pass for memory efficiency."""
    result_data = data.copy()
    records_affected = 0
    original_memory = result_data.memory_usage(deep=True).sum()

    for col in result_data.select_dtypes(include=["int64"]).columns:
        col_min = result_data[col].min()
        col_max = result_data[col].max()
        if col_min >= -128 and col_max <= 127:
            result_data[col] = result_data[col].astype("int8")
        elif col_min >= -32768 and col_max <= 32767:
            result_data[col] = result_data[col].astype("int16")
        elif col_min >= -2147483648 and col_max <= 2147483647:
            result_data[col] = result_data[col].astype("int32")

    for col in result_data.select_dtypes(include=["float64"]).columns:
        if result_data[col].dtype == "float64":
            float32_version = result_data[col].astype("float32")
            try:
                if np.allclose(
                    result_data[col], float32_version, rtol=1e-6, equal_nan=True
                ):
                    result_data[col] = float32_version
            except Exception:
                pass

    final_memory = result_data.memory_usage(deep=True).sum()
    memory_reduction = (
        (original_memory - final_memory) / original_memory * 100
        if original_memory > 0
        else 0
    )

    optimization_log = {
        "memory_optimization": {
            "original_memory_mb": original_memory / (1024 * 1024),
            "optimized_memory_mb": final_memory / (1024 * 1024),
            "reduction_percentage": memory_reduction,
        }
    }

    metadata = {
        "parameters": {"operation": "final_optimization"},
        "records_affected": records_affected,
        "optimization_log": optimization_log,
    }

    return result_data, metadata


def _assess_final_quality(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assess final data quality after cleaning."""
    self._quality_metrics_after = self._calculate_comprehensive_quality_metrics(data)

    if self._quality_metrics_before:
        improvement = {
            "overall": (
                self._quality_metrics_after.overall_quality_score
                - self._quality_metrics_before.overall_quality_score
            ),
            "completeness": (
                self._quality_metrics_after.completeness_score
                - self._quality_metrics_before.completeness_score
            ),
            "consistency": (
                self._quality_metrics_after.consistency_score
                - self._quality_metrics_before.consistency_score
            ),
            "validity": (
                self._quality_metrics_after.validity_score
                - self._quality_metrics_before.validity_score
            ),
            "accuracy": (
                self._quality_metrics_after.accuracy_score
                - self._quality_metrics_before.accuracy_score
            ),
        }
    else:
        improvement = {"error": "no_before_metrics"}

    metadata = {
        "parameters": {},
        "records_affected": 0,
        "final_quality_assessment": {
            "overall_score": self._quality_metrics_after.overall_quality_score,
            "completeness": self._quality_metrics_after.completeness_score,
            "consistency": self._quality_metrics_after.consistency_score,
            "validity": self._quality_metrics_after.validity_score,
            "accuracy": self._quality_metrics_after.accuracy_score,
            "improvement": improvement,
        },
    }

    logger.info(
        "Final quality assessment completed",
        final_score=self._quality_metrics_after.overall_quality_score,
        improvement=improvement.get("overall", 0),
    )

    return data.copy(), metadata
