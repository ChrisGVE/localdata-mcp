"""
Data Cleaning Pipeline Implementation Methods - Part 2 (Continued)

This module contains the remaining implementation methods for the DataCleaningPipeline class.
"""

import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

from ..logging_manager import get_logger

logger = get_logger(__name__)

def _remove_exact_duplicates(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
            "duplicates_removed": records_affected
        }
    }
    
    return result_data, metadata

def _sophisticated_duplicate_detection(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Sophisticated duplicate detection including fuzzy matching."""
    result_data = data.copy()
    records_affected = 0
    duplicate_log = {}
    
    # First, remove exact duplicates
    initial_rows = len(result_data)
    result_data = result_data.drop_duplicates()
    exact_duplicates = initial_rows - len(result_data)
    records_affected += exact_duplicates
    
    # Then, look for fuzzy duplicates in text columns
    text_cols = result_data.select_dtypes(include=['object']).columns
    fuzzy_duplicates = 0
    
    if len(text_cols) > 0 and len(result_data) < 10000 and fuzz:  # Only for manageable datasets
        try:
            # Fuzzy matching for string columns
            for col in text_cols:
                if result_data[col].dtype == 'object':
                    unique_values = result_data[col].dropna().unique()
                    if len(unique_values) < 1000:  # Only for reasonable number of unique values
                        
                        duplicates_to_remove = set()
                        for i, val1 in enumerate(unique_values):
                            if val1 in duplicates_to_remove:
                                continue
                            for j, val2 in enumerate(unique_values[i+1:], i+1):
                                if val2 in duplicates_to_remove:
                                    continue
                                
                                # Use fuzzywuzzy for similarity
                                similarity = fuzz.ratio(str(val1), str(val2))
                                if similarity > 85:  # 85% similarity threshold
                                    # Keep the more frequent value
                                    count1 = (result_data[col] == val1).sum()
                                    count2 = (result_data[col] == val2).sum()
                                    
                                    if count1 >= count2:
                                        result_data[col] = result_data[col].replace(val2, val1)
                                        duplicates_to_remove.add(val2)
                                    else:
                                        result_data[col] = result_data[col].replace(val1, val2)
                                        duplicates_to_remove.add(val1)
                                    
                                    fuzzy_duplicates += min(count1, count2)
                        
                        duplicate_log[col] = {
                            'fuzzy_duplicates_merged': len(duplicates_to_remove)
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
            "total_affected": records_affected
        },
        "duplicate_log": duplicate_log
    }
    
    return result_data, metadata

def _basic_data_validation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Basic data validation with simple business rules."""
    result_data = data.copy()
    validation_log = {}
    records_affected = 0
    
    # Basic validation rules
    numeric_cols = data.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        # Check for negative values where they might not make sense
        if 'age' in col.lower() or 'count' in col.lower() or 'quantity' in col.lower():
            negative_mask = data[col] < 0
            if negative_mask.sum() > 0:
                result_data.loc[negative_mask, col] = 0  # Set to 0
                validation_log[col] = {
                    'rule': 'non_negative',
                    'violations': negative_mask.sum(),
                    'action': 'set_to_zero'
                }
                records_affected += negative_mask.sum()
    
    # Date validation
    date_cols = data.select_dtypes(include=['datetime64']).columns
    for col in date_cols:
        # Check for future dates where they might not make sense
        if 'birth' in col.lower() or 'created' in col.lower():
            future_mask = data[col] > pd.Timestamp.now()
            if future_mask.sum() > 0:
                result_data.loc[future_mask, col] = pd.NaT  # Set to NaT
                validation_log[col] = {
                    'rule': 'no_future_dates',
                    'violations': future_mask.sum(),
                    'action': 'set_to_nat'
                }
                records_affected += future_mask.sum()
    
    metadata = {
        "parameters": {"validation_level": "basic"},
        "records_affected": records_affected,
        "validation_log": validation_log
    }
    
    return result_data, metadata

def _comprehensive_data_validation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Comprehensive data validation with configurable business rules."""
    result_data = data.copy()
    validation_log = {}
    records_affected = 0
    
    # Apply basic validation first
    result_data, basic_metadata = self._basic_data_validation(result_data)
    validation_log.update(basic_metadata['validation_log'])
    records_affected += basic_metadata['records_affected']
    
    # Apply custom business rules
    for rule in self.business_rules:
        try:
            rule_type = rule.get('type')
            column = rule.get('column')
            parameters = rule.get('parameters', {})
            
            if rule_type == 'range_validation' and column in result_data.columns:
                min_val = parameters.get('min')
                max_val = parameters.get('max')
                
                if min_val is not None:
                    violation_mask = result_data[column] < min_val
                    if violation_mask.sum() > 0:
                        action = parameters.get('action', 'set_to_min')
                        if action == 'set_to_min':
                            result_data.loc[violation_mask, column] = min_val
                        elif action == 'set_to_null':
                            result_data.loc[violation_mask, column] = pd.NA
                        records_affected += violation_mask.sum()
                
                if max_val is not None:
                    violation_mask = result_data[column] > max_val
                    if violation_mask.sum() > 0:
                        action = parameters.get('action', 'set_to_max')
                        if action == 'set_to_max':
                            result_data.loc[violation_mask, column] = max_val
                        elif action == 'set_to_null':
                            result_data.loc[violation_mask, column] = pd.NA
                        records_affected += violation_mask.sum()
                
                validation_log[f"{column}_range"] = {
                    'rule': 'range_validation',
                    'parameters': parameters,
                    'violations': violation_mask.sum() if 'violation_mask' in locals() else 0
                }
            
            elif rule_type == 'pattern_validation' and column in result_data.columns:
                pattern = parameters.get('pattern')
                if pattern:
                    violation_mask = ~result_data[column].astype(str).str.match(pattern, na=False)
                    if violation_mask.sum() > 0:
                        action = parameters.get('action', 'set_to_null')
                        if action == 'set_to_null':
                            result_data.loc[violation_mask, column] = pd.NA
                        records_affected += violation_mask.sum()
                    
                    validation_log[f"{column}_pattern"] = {
                        'rule': 'pattern_validation',
                        'pattern': pattern,
                        'violations': violation_mask.sum()
                    }
                    
        except Exception as e:
            logger.warning(f"Business rule validation failed: {e}")
    
    metadata = {
        "parameters": {"validation_level": "comprehensive", "business_rules_count": len(self.business_rules)},
        "records_affected": records_affected,
        "validation_log": validation_log
    }
    
    return result_data, metadata

def _data_consistency_enhancement(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enhance data consistency across columns and relationships."""
    result_data = data.copy()
    consistency_log = {}
    records_affected = 0
    
    # Standardize categorical values
    categorical_cols = result_data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        # Standardize case
        if result_data[col].dtype == 'object':
            original_unique = result_data[col].nunique()
            result_data[col] = result_data[col].astype(str).str.strip().str.lower()
            new_unique = result_data[col].nunique()
            
            if original_unique != new_unique:
                consistency_log[col] = {
                    'operation': 'case_standardization',
                    'before_unique': original_unique,
                    'after_unique': new_unique,
                    'values_consolidated': original_unique - new_unique
                }
                records_affected += len(result_data)
    
    # Check for logical inconsistencies (example: end_date before start_date)
    date_cols = result_data.select_dtypes(include=['datetime64']).columns
    if len(date_cols) >= 2:
        # Look for start/end date pairs
        start_cols = [col for col in date_cols if 'start' in col.lower() or 'begin' in col.lower()]
        end_cols = [col for col in date_cols if 'end' in col.lower() or 'finish' in col.lower()]
        
        for start_col in start_cols:
            for end_col in end_cols:
                inconsistent_mask = result_data[start_col] > result_data[end_col]
                if inconsistent_mask.sum() > 0:
                    # Swap the values
                    temp = result_data.loc[inconsistent_mask, start_col].copy()
                    result_data.loc[inconsistent_mask, start_col] = result_data.loc[inconsistent_mask, end_col]
                    result_data.loc[inconsistent_mask, end_col] = temp
                    
                    consistency_log[f"{start_col}_{end_col}"] = {
                        'operation': 'date_order_fix',
                        'inconsistencies': inconsistent_mask.sum()
                    }
                    records_affected += inconsistent_mask.sum()
    
    metadata = {
        "parameters": {"operation": "consistency_enhancement"},
        "records_affected": records_affected,
        "consistency_log": consistency_log
    }
    
    return result_data, metadata

def _feature_engineering_cleanup(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean up features by removing constant/near-constant columns and highly correlated features."""
    result_data = data.copy()
    cleanup_log = {}
    
    # Remove constant columns
    constant_cols = []
    for col in result_data.columns:
        if result_data[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        result_data = result_data.drop(columns=constant_cols)
        cleanup_log['constant_columns_removed'] = constant_cols
    
    # Remove near-constant columns (> 95% same value)
    near_constant_cols = []
    for col in result_data.columns:
        if result_data[col].dtype in ['object', 'category']:
            if len(result_data[col].value_counts()) > 0:
                most_frequent_pct = result_data[col].value_counts(normalize=True).iloc[0]
                if most_frequent_pct > 0.95:
                    near_constant_cols.append(col)
    
    if near_constant_cols:
        result_data = result_data.drop(columns=near_constant_cols)
        cleanup_log['near_constant_columns_removed'] = near_constant_cols
    
    # Remove highly correlated numeric features
    numeric_cols = result_data.select_dtypes(include=['number']).columns
    highly_corr_features = []
    
    if len(numeric_cols) > 1:
        try:
            corr_matrix = result_data[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation > 0.95
            highly_corr_features = [
                column for column in upper_tri.columns 
                if any(upper_tri[column] > 0.95)
            ]
            
            if highly_corr_features:
                result_data = result_data.drop(columns=highly_corr_features)
                cleanup_log['highly_correlated_features_removed'] = highly_corr_features
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
    
    total_removed = len(constant_cols) + len(near_constant_cols) + len(highly_corr_features)
    
    metadata = {
        "parameters": {"correlation_threshold": 0.95, "constant_threshold": 0.95},
        "records_affected": 0,  # This affects columns, not records
        "cleanup_log": cleanup_log,
        "columns_removed": total_removed
    }
    
    return result_data, metadata

def _final_quality_optimization(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Final optimization pass to ensure optimal data quality."""
    result_data = data.copy()
    optimization_log = {}
    records_affected = 0
    
    # Optimize data types for memory efficiency
    original_memory = result_data.memory_usage(deep=True).sum()
    
    # Convert int64 to smaller int types where possible
    for col in result_data.select_dtypes(include=['int64']).columns:
        col_min = result_data[col].min()
        col_max = result_data[col].max()
        
        if col_min >= -128 and col_max <= 127:
            result_data[col] = result_data[col].astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            result_data[col] = result_data[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            result_data[col] = result_data[col].astype('int32')
    
    # Convert float64 to float32 where precision allows
    for col in result_data.select_dtypes(include=['float64']).columns:
        if result_data[col].dtype == 'float64':
            # Check if conversion to float32 would lose significant precision
            float32_version = result_data[col].astype('float32')
            try:
                if np.allclose(result_data[col], float32_version, rtol=1e-6, equal_nan=True):
                    result_data[col] = float32_version
            except:
                pass  # Skip if comparison fails
    
    # Final data quality score
    final_memory = result_data.memory_usage(deep=True).sum()
    memory_reduction = (original_memory - final_memory) / original_memory * 100 if original_memory > 0 else 0
    
    optimization_log = {
        'memory_optimization': {
            'original_memory_mb': original_memory / (1024 * 1024),
            'optimized_memory_mb': final_memory / (1024 * 1024),
            'reduction_percentage': memory_reduction
        }
    }
    
    metadata = {
        "parameters": {"operation": "final_optimization"},
        "records_affected": records_affected,
        "optimization_log": optimization_log
    }
    
    return result_data, metadata

def _assess_final_quality(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assess final data quality after cleaning."""
    self._quality_metrics_after = self._calculate_comprehensive_quality_metrics(data)
    
    # Calculate improvement
    if self._quality_metrics_before:
        improvement = {
            "overall": self._quality_metrics_after.overall_quality_score - self._quality_metrics_before.overall_quality_score,
            "completeness": self._quality_metrics_after.completeness_score - self._quality_metrics_before.completeness_score,
            "consistency": self._quality_metrics_after.consistency_score - self._quality_metrics_before.consistency_score,
            "validity": self._quality_metrics_after.validity_score - self._quality_metrics_before.validity_score,
            "accuracy": self._quality_metrics_after.accuracy_score - self._quality_metrics_before.accuracy_score
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
            "improvement": improvement
        }
    }
    
    logger.info("Final quality assessment completed",
               final_score=self._quality_metrics_after.overall_quality_score,
               improvement=improvement.get('overall', 0))
    
    return data.copy(), metadata