"""
Data Cleaning Pipeline Implementation Methods - Part 2.

Contains mixin methods for the DataCleaningPipeline class:
outlier detection, duplicate handling, validation, consistency,
feature cleanup, optimization, and final quality assessment.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

from ...logging_manager import get_logger

logger = get_logger(__name__)


def _advanced_outlier_detection(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Advanced outlier detection using IsolationForest and LOF."""
    result_data = data.copy()
    numeric_cols = data.select_dtypes(include=["number"]).columns
    outlier_log = {}
    records_affected = 0

    if len(numeric_cols) == 0:
        metadata = {
            "parameters": {"method": "none", "reason": "no_numeric_columns"},
            "records_affected": 0,
            "outlier_log": {},
        }
        return result_data, metadata

    numeric_data = data[numeric_cols].fillna(data[numeric_cols].median())

    try:
        isolation_forest = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=100
        )
        isolation_outliers = isolation_forest.fit_predict(numeric_data)
        isolation_mask = isolation_outliers == -1

        lof = LocalOutlierFactor(
            n_neighbors=min(20, len(data) // 10 + 1), contamination=0.1
        )
        lof_outliers = lof.fit_predict(numeric_data)
        lof_mask = lof_outliers == -1

        combined_outliers = isolation_mask & lof_mask
        moderate_outliers = isolation_mask | lof_mask

        action = self.custom_parameters.get("outlier_action", "flag")

        if action == "remove":
            result_data = result_data[~combined_outliers]
            records_affected = combined_outliers.sum()
        elif action == "cap":
            for col in numeric_cols:
                q1, q3 = data[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_indices = moderate_outliers
                result_data.loc[outlier_indices, col] = np.clip(
                    result_data.loc[outlier_indices, col],
                    lower_bound,
                    upper_bound,
                )
            records_affected = moderate_outliers.sum()
        else:  # flag
            result_data["outlier_isolation"] = isolation_mask
            result_data["outlier_lof"] = lof_mask
            result_data["outlier_combined"] = combined_outliers
            records_affected = moderate_outliers.sum()

        outlier_log = {
            "isolation_forest_outliers": isolation_mask.sum(),
            "lof_outliers": lof_mask.sum(),
            "combined_outliers": combined_outliers.sum(),
            "moderate_outliers": moderate_outliers.sum(),
            "action_taken": action,
        }

    except Exception as e:
        logger.warning(f"Advanced outlier detection failed, using IQR: {e}")

        for col in numeric_cols:
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            outliers_count = outlier_mask.sum()

            if outliers_count > 0:
                result_data[col] = np.clip(data[col], lower_bound, upper_bound)
                records_affected += outliers_count
                outlier_log[col] = {
                    "method": "iqr_fallback",
                    "outliers_detected": outliers_count,
                    "bounds": {"lower": lower_bound, "upper": upper_bound},
                }

    metadata = {
        "parameters": {"method": "advanced_sklearn", "contamination": 0.1},
        "records_affected": records_affected,
        "outlier_log": outlier_log,
        "reversibility_data": {
            "outlier_indices": combined_outliers
            if "combined_outliers" in locals()
            else []
        },
    }

    return result_data, metadata


def _remove_exact_duplicates(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove exact duplicate rows."""
    initial_rows = len(data)
    result_data = data.drop_duplicates()
    records_affected = initial_rows - len(result_data)

    metadata = {
        "parameters": {"method": "exact"},
        "records_affected": records_affected,
        "duplicate_info": {
            "initial_rows": initial_rows,
            "final_rows": len(result_data),
            "duplicates_removed": records_affected,
        },
    }

    return result_data, metadata


def _sophisticated_duplicate_detection(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Sophisticated duplicate detection including fuzzy matching."""
    result_data = data.copy()
    records_affected = 0
    duplicate_log = {}

    initial_rows = len(result_data)
    result_data = result_data.drop_duplicates()
    exact_duplicates = initial_rows - len(result_data)
    records_affected += exact_duplicates

    text_cols = result_data.select_dtypes(include=["object"]).columns
    fuzzy_duplicates = 0

    if len(text_cols) > 0 and len(result_data) < 10000 and fuzz:
        try:
            for col in text_cols:
                if result_data[col].dtype == "object":
                    unique_values = result_data[col].dropna().unique()
                    if len(unique_values) < 1000:
                        duplicates_to_remove = set()
                        for i, val1 in enumerate(unique_values):
                            if val1 in duplicates_to_remove:
                                continue
                            for j, val2 in enumerate(unique_values[i + 1 :], i + 1):
                                if val2 in duplicates_to_remove:
                                    continue
                                similarity = fuzz.ratio(str(val1), str(val2))
                                if similarity > 85:
                                    count1 = (result_data[col] == val1).sum()
                                    count2 = (result_data[col] == val2).sum()
                                    if count1 >= count2:
                                        result_data[col] = result_data[col].replace(
                                            val2, val1
                                        )
                                        duplicates_to_remove.add(val2)
                                    else:
                                        result_data[col] = result_data[col].replace(
                                            val1, val2
                                        )
                                        duplicates_to_remove.add(val1)
                                    fuzzy_duplicates += min(count1, count2)

                        duplicate_log[col] = {
                            "fuzzy_duplicates_merged": len(duplicates_to_remove)
                        }

        except Exception as e:
            logger.warning(f"Fuzzy duplicate detection failed: {e}")
    elif not fuzz:
        logger.warning("fuzzywuzzy not available, skipping fuzzy duplicate detection")

    records_affected += fuzzy_duplicates

    metadata = {
        "parameters": {"method": "sophisticated", "fuzzy_threshold": 85},
        "records_affected": records_affected,
        "duplicate_info": {
            "exact_duplicates": exact_duplicates,
            "fuzzy_duplicates": fuzzy_duplicates,
            "total_affected": records_affected,
        },
        "duplicate_log": duplicate_log,
    }

    return result_data, metadata


def _basic_data_validation(
    self, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Basic data validation with simple business rules."""
    result_data = data.copy()
    validation_log = {}
    records_affected = 0

    numeric_cols = data.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        if "age" in col.lower() or "count" in col.lower() or "quantity" in col.lower():
            negative_mask = data[col] < 0
            if negative_mask.sum() > 0:
                result_data.loc[negative_mask, col] = 0
                validation_log[col] = {
                    "rule": "non_negative",
                    "violations": negative_mask.sum(),
                    "action": "set_to_zero",
                }
                records_affected += negative_mask.sum()

    date_cols = data.select_dtypes(include=["datetime64"]).columns
    for col in date_cols:
        if "birth" in col.lower() or "created" in col.lower():
            future_mask = data[col] > pd.Timestamp.now()
            if future_mask.sum() > 0:
                result_data.loc[future_mask, col] = pd.NaT
                validation_log[col] = {
                    "rule": "no_future_dates",
                    "violations": future_mask.sum(),
                    "action": "set_to_nat",
                }
                records_affected += future_mask.sum()

    metadata = {
        "parameters": {"validation_level": "basic"},
        "records_affected": records_affected,
        "validation_log": validation_log,
    }

    return result_data, metadata
