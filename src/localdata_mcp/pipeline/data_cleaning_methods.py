"""
Data Cleaning Pipeline Implementation Methods - Part 2

This module contains the implementation methods for the DataCleaningPipeline class.
These methods are separated from the main preprocessing.py file due to size constraints.
"""

import time
import warnings
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from difflib import SequenceMatcher
import re

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

from ..logging_manager import get_logger

logger = get_logger(__name__)

# This file contains methods that should be mixed into the DataCleaningPipeline class
# The methods are defined here and can be imported and added to the main class

def _assess_initial_quality(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
            "accuracy": self._quality_metrics_before.accuracy_score
        }
    }
    
    logger.info("Initial quality assessment completed",
               overall_score=self._quality_metrics_before.overall_quality_score)
    
    return data.copy(), metadata

def _basic_type_inference(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Basic data type inference and conversion."""
    result_data = data.copy()
    type_conversions = {}
    records_affected = 0
    
    for col in data.columns:
        if data[col].dtype == 'object':
            # Try numeric conversion
            try:
                numeric_series = pd.to_numeric(data[col], errors='coerce')
                # If more than 80% convert successfully, use numeric type
                success_rate = numeric_series.notna().sum() / len(numeric_series)
                if success_rate > 0.8:
                    result_data[col] = numeric_series
                    type_conversions[col] = {'from': 'object', 'to': str(numeric_series.dtype)}
                    records_affected += len(result_data)
                    continue
            except:
                pass
            
            # Try datetime conversion
            try:
                datetime_series = pd.to_datetime(data[col], errors='coerce')
                success_rate = datetime_series.notna().sum() / len(datetime_series)
                if success_rate > 0.8:
                    result_data[col] = datetime_series
                    type_conversions[col] = {'from': 'object', 'to': 'datetime64[ns]'}
                    records_affected += len(result_data)
            except:
                pass
    
    metadata = {
        "parameters": {"inference_threshold": 0.8},
        "records_affected": records_affected,
        "type_conversions": type_conversions,
        "reversibility_data": {"original_dtypes": dict(data.dtypes)}
    }
    
    return result_data, metadata

def _comprehensive_type_inference(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Comprehensive intelligent data type inference."""
    from .preprocessing import TransformationStrategy
    
    result_data = data.copy()
    strategies = TransformationStrategy.data_type_inference_strategy(data)
    type_conversions = {}
    records_affected = 0
    
    for col, strategy in strategies.items():
        if strategy == "datetime":
            try:
                # Multiple datetime format attempts
                datetime_series = pd.to_datetime(data[col], infer_datetime_format=True, errors='coerce')
                if datetime_series.notna().sum() / len(datetime_series) > 0.7:
                    result_data[col] = datetime_series
                    type_conversions[col] = {'from': str(data[col].dtype), 'to': 'datetime64[ns]'}
                    records_affected += len(result_data)
            except:
                pass
        elif strategy == "numeric":
            try:
                numeric_series = pd.to_numeric(data[col], errors='coerce')
                if numeric_series.notna().sum() / len(numeric_series) > 0.7:
                    result_data[col] = numeric_series
                    type_conversions[col] = {'from': str(data[col].dtype), 'to': str(numeric_series.dtype)}
                    records_affected += len(result_data)
            except:
                pass
        elif strategy == "categorical":
            # Convert to category if cardinality is reasonable
            if data[col].nunique() < len(data) * 0.5:
                result_data[col] = data[col].astype('category')
                type_conversions[col] = {'from': str(data[col].dtype), 'to': 'category'}
                records_affected += len(result_data)
    
    metadata = {
        "parameters": {"strategies": strategies},
        "records_affected": records_affected,
        "type_conversions": type_conversions,
        "reversibility_data": {"original_dtypes": dict(data.dtypes)}
    }
    
    return result_data, metadata

def _handle_basic_missing_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Basic missing value handling using simple strategies."""
    result_data = data.copy()
    imputation_log = {}
    records_affected = 0
    
    # Simple median imputation for numeric
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            median_val = data[col].median()
            result_data[col].fillna(median_val, inplace=True)
            imputation_log[col] = {'method': 'median', 'value': median_val}
            records_affected += data[col].isnull().sum()
    
    # Simple mode imputation for categorical
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else 'unknown'
            result_data[col].fillna(mode_val, inplace=True)
            imputation_log[col] = {'method': 'mode', 'value': mode_val}
            records_affected += data[col].isnull().sum()
    
    metadata = {
        "parameters": {"strategy": "basic"},
        "records_affected": records_affected,
        "imputation_log": imputation_log,
        "reversibility_data": {"missing_positions": data.isnull().to_dict()}
    }
    
    return result_data, metadata

def _intelligent_missing_value_handling(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Intelligent missing value handling using advanced sklearn methods."""
    result_data = data.copy()
    imputation_log = {}
    records_affected = 0
    
    # Advanced imputation for numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1 and any(data[col].isnull().sum() > 0 for col in numeric_cols):
        try:
            # Use KNN imputation for numeric columns
            knn_imputer = KNNImputer(n_neighbors=min(5, len(data)//10 + 1))
            numeric_data = data[numeric_cols]
            imputed_numeric = knn_imputer.fit_transform(numeric_data)
            
            for i, col in enumerate(numeric_cols):
                if numeric_data[col].isnull().sum() > 0:
                    result_data[col] = imputed_numeric[:, i]
                    imputation_log[col] = {'method': 'knn', 'n_neighbors': knn_imputer.n_neighbors}
                    records_affected += numeric_data[col].isnull().sum()
        except Exception as e:
            # Fallback to median imputation
            for col in numeric_cols:
                if data[col].isnull().sum() > 0:
                    median_val = data[col].median()
                    result_data[col].fillna(median_val, inplace=True)
                    imputation_log[col] = {'method': 'median_fallback', 'value': median_val}
                    records_affected += data[col].isnull().sum()
    
    # Mode imputation for categorical with frequency-based selection
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            # Use the most frequent value, but if it's very rare, use 'unknown'
            value_counts = data[col].value_counts()
            if len(value_counts) > 0 and value_counts.iloc[0] / len(data) > 0.05:
                mode_val = value_counts.index[0]
                method = 'mode'
            else:
                mode_val = 'unknown'
                method = 'unknown_substitution'
            
            result_data[col].fillna(mode_val, inplace=True)
            imputation_log[col] = {'method': method, 'value': mode_val}
            records_affected += data[col].isnull().sum()
    
    metadata = {
        "parameters": {"strategy": "intelligent"},
        "records_affected": records_affected,
        "imputation_log": imputation_log,
        "reversibility_data": {"missing_positions": data.isnull().to_dict()}
    }
    
    return result_data, metadata

def _advanced_outlier_detection(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Advanced outlier detection using sklearn IsolationForest and LocalOutlierFactor."""
    result_data = data.copy()
    numeric_cols = data.select_dtypes(include=['number']).columns
    outlier_log = {}
    records_affected = 0
    
    if len(numeric_cols) == 0:
        metadata = {
            "parameters": {"method": "none", "reason": "no_numeric_columns"},
            "records_affected": 0,
            "outlier_log": {}
        }
        return result_data, metadata
    
    # Use different methods based on data characteristics
    numeric_data = data[numeric_cols].fillna(data[numeric_cols].median())
    
    try:
        # IsolationForest for global anomaly detection
        isolation_forest = IsolationForest(
            contamination=0.1,  # Assume 10% outliers max
            random_state=42,
            n_estimators=100
        )
        
        isolation_outliers = isolation_forest.fit_predict(numeric_data)
        isolation_mask = isolation_outliers == -1
        
        # LocalOutlierFactor for local anomaly detection
        lof = LocalOutlierFactor(
            n_neighbors=min(20, len(data)//10 + 1),
            contamination=0.1
        )
        
        lof_outliers = lof.fit_predict(numeric_data)
        lof_mask = lof_outliers == -1
        
        # Combine both methods (intersection for high confidence)
        combined_outliers = isolation_mask & lof_mask
        moderate_outliers = isolation_mask | lof_mask
        
        # Different actions based on outlier confidence
        action = self.custom_parameters.get('outlier_action', 'flag')
        
        if action == 'remove':
            # Remove high-confidence outliers
            result_data = result_data[~combined_outliers]
            records_affected = combined_outliers.sum()
        elif action == 'cap':
            # Cap outliers to reasonable bounds
            for col in numeric_cols:
                q1, q3 = data[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_indices = moderate_outliers
                result_data.loc[outlier_indices, col] = np.clip(
                    result_data.loc[outlier_indices, col], 
                    lower_bound, upper_bound
                )
            records_affected = moderate_outliers.sum()
        else:  # flag
            # Add outlier flags
            result_data['outlier_isolation'] = isolation_mask
            result_data['outlier_lof'] = lof_mask
            result_data['outlier_combined'] = combined_outliers
            records_affected = moderate_outliers.sum()
        
        outlier_log = {
            "isolation_forest_outliers": isolation_mask.sum(),
            "lof_outliers": lof_mask.sum(),
            "combined_outliers": combined_outliers.sum(),
            "moderate_outliers": moderate_outliers.sum(),
            "action_taken": action
        }
        
    except Exception as e:
        # Fallback to IQR method
        logger.warning(f"Advanced outlier detection failed, using IQR method: {e}")
        
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
                    'method': 'iqr_fallback',
                    'outliers_detected': outliers_count,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
    
    metadata = {
        "parameters": {"method": "advanced_sklearn", "contamination": 0.1},
        "records_affected": records_affected,
        "outlier_log": outlier_log,
        "reversibility_data": {"outlier_indices": combined_outliers if 'combined_outliers' in locals() else []}
    }
    
    return result_data, metadata