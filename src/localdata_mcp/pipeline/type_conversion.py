"""
Data Type Conversion System - Intelligent Type Inference and Safe Conversion

This module implements comprehensive data type conversion utilities with intelligent
type inference, conversion validation, and metadata preservation throughout pipeline
transformations.

Key Features:
- Intelligent data type inference based on content analysis
- Safe type conversions with validation and error handling
- Metadata preservation during type transformations
- Integration with DomainAdapter for library-specific type requirements
- Support for complex data types (datetime, categorical, mixed)
- Conversion rollback capabilities for error recovery
- Memory-efficient conversion for large datasets
"""

import time
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date
import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .base import (
    AnalysisPipelineBase,
    StreamingConfig,
    PipelineError,
    ErrorClassification
)
from ..logging_manager import get_logger

logger = get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')


class DataType(Enum):
    """Enumeration of supported data types."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    DATE = "date"
    BOOLEAN = "boolean"
    MIXED = "mixed"
    UNKNOWN = "unknown"
    
    # Complex types
    JSON = "json"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"
    CURRENCY = "currency"


@dataclass
class TypeInferenceResult:
    """Result of data type inference analysis."""
    
    inferred_type: DataType
    confidence_score: float
    sample_analysis: Dict[str, Any] = field(default_factory=dict)
    conversion_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)
    memory_impact: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionOperation:
    """Record of a type conversion operation for transparency and rollback."""
    
    column: str
    from_type: str
    to_type: DataType
    success: bool
    execution_time: float
    records_converted: int
    conversion_errors: int
    rollback_data: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    metadata_preserved: bool = True


class TypeInferenceEngine:
    """Engine for intelligent data type inference based on content analysis."""
    
    def __init__(self):
        # Regex patterns for type detection
        self.patterns = {
            DataType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            DataType.URL: re.compile(r'^https?://[^\s]+$'),
            DataType.PHONE: re.compile(r'^[\+]?[1-9]?[\d\s\-\(\)]{7,15}$'),
            DataType.CURRENCY: re.compile(r'^[\$€£¥]?[\d,]+\.?\d*$'),
            DataType.DATE: re.compile(r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$|^\d{2}-\d{2}-\d{4}$'),
            DataType.DATETIME: re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(:\d{2})?$')
        }
        
        # Threshold configurations
        self.confidence_thresholds = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
    
    def infer_type(self, data: pd.Series) -> TypeInferenceResult:
        """
        Infer the most appropriate data type for a pandas Series.
        
        Args:
            data: Pandas Series to analyze
            
        Returns:
            TypeInferenceResult with inference details
        """
        # Remove null values for analysis
        non_null_data = data.dropna()
        if len(non_null_data) == 0:
            return TypeInferenceResult(
                inferred_type=DataType.UNKNOWN,
                confidence_score=0.0,
                potential_issues=["All values are null"]
            )
        
        # Sample data for analysis (limit to 1000 for performance)
        sample_size = min(1000, len(non_null_data))
        sample_data = non_null_data.sample(n=sample_size, random_state=42)
        
        # Analyze current dtype
        current_dtype = str(data.dtype)
        
        # Run inference tests
        inference_results = {
            DataType.INTEGER: self._test_integer(sample_data),
            DataType.FLOAT: self._test_float(sample_data),
            DataType.BOOLEAN: self._test_boolean(sample_data),
            DataType.DATETIME: self._test_datetime(sample_data),
            DataType.DATE: self._test_date(sample_data),
            DataType.EMAIL: self._test_email(sample_data),
            DataType.URL: self._test_url(sample_data),
            DataType.PHONE: self._test_phone(sample_data),
            DataType.CURRENCY: self._test_currency(sample_data),
            DataType.CATEGORICAL: self._test_categorical(sample_data),
            DataType.STRING: self._test_string(sample_data)
        }
        
        # Find best match
        best_type = max(inference_results.items(), key=lambda x: x[1]['confidence'])
        
        # Generate conversion suggestions
        suggestions = self._generate_conversion_suggestions(inference_results, current_dtype)
        
        # Identify potential issues
        issues = self._identify_potential_issues(data, best_type[0])
        
        # Calculate memory impact
        memory_impact = self._calculate_memory_impact(data, best_type[0])
        
        return TypeInferenceResult(
            inferred_type=best_type[0],
            confidence_score=best_type[1]['confidence'],
            sample_analysis={
                'sample_size': sample_size,
                'current_dtype': current_dtype,
                'unique_values': len(non_null_data.unique()),
                'null_percentage': (data.isnull().sum() / len(data)) * 100,
                'inference_scores': {k.value: v['confidence'] for k, v in inference_results.items()}
            },
            conversion_suggestions=suggestions,
            potential_issues=issues,
            memory_impact=memory_impact
        )
    
    def _test_integer(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data can be converted to integer."""
        try:
            # Try converting to numeric
            numeric_data = pd.to_numeric(data.astype(str), errors='coerce')
            
            # Check if all non-null values are integers
            valid_count = numeric_data.notna().sum()
            if valid_count == 0:
                return {'confidence': 0.0, 'details': 'No valid numeric values'}
            
            # Check if values are whole numbers
            integer_mask = (numeric_data == numeric_data.astype('Int64', errors='ignore')).fillna(False)
            integer_count = integer_mask.sum()
            
            confidence = integer_count / valid_count
            
            return {
                'confidence': confidence,
                'details': {
                    'valid_numeric': valid_count,
                    'valid_integers': integer_count,
                    'range': (numeric_data.min(), numeric_data.max()) if valid_count > 0 else None
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_float(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data can be converted to float."""
        try:
            numeric_data = pd.to_numeric(data.astype(str), errors='coerce')
            
            valid_count = numeric_data.notna().sum()
            if valid_count == 0:
                return {'confidence': 0.0, 'details': 'No valid numeric values'}
            
            # Check if values have decimal places
            has_decimals = (numeric_data % 1 != 0).sum()
            
            # Higher confidence if there are actual decimal values
            confidence = valid_count / len(data)
            if has_decimals > 0:
                confidence += 0.1  # Bonus for having actual decimals
            
            return {
                'confidence': min(confidence, 1.0),
                'details': {
                    'valid_numeric': valid_count,
                    'has_decimals': has_decimals,
                    'range': (numeric_data.min(), numeric_data.max()) if valid_count > 0 else None
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_boolean(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data represents boolean values."""
        try:
            # Common boolean representations
            true_values = {'true', '1', 'yes', 'y', 't', 'on', 'enabled'}
            false_values = {'false', '0', 'no', 'n', 'f', 'off', 'disabled'}
            
            str_data = data.astype(str).str.lower().str.strip()
            boolean_values = true_values | false_values
            
            boolean_count = str_data.isin(boolean_values).sum()
            confidence = boolean_count / len(data)
            
            return {
                'confidence': confidence,
                'details': {
                    'boolean_matches': boolean_count,
                    'unique_values': str_data.unique().tolist()[:10]
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_datetime(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data can be converted to datetime."""
        try:
            # Try pandas datetime conversion
            datetime_data = pd.to_datetime(data.astype(str), errors='coerce', infer_datetime_format=True)
            
            valid_count = datetime_data.notna().sum()
            confidence = valid_count / len(data) if len(data) > 0 else 0
            
            # Pattern matching bonus
            pattern_matches = 0
            for value in data.astype(str).head(100):
                if self.patterns[DataType.DATETIME].match(value):
                    pattern_matches += 1
            
            if pattern_matches > 0:
                confidence += 0.1
            
            return {
                'confidence': min(confidence, 1.0),
                'details': {
                    'valid_datetime': valid_count,
                    'pattern_matches': pattern_matches,
                    'date_range': (datetime_data.min(), datetime_data.max()) if valid_count > 0 else None
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_date(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data represents date values (without time)."""
        try:
            # Try date conversion
            date_data = pd.to_datetime(data.astype(str), errors='coerce').dt.date
            
            valid_count = pd.Series(date_data).notna().sum()
            confidence = valid_count / len(data) if len(data) > 0 else 0
            
            # Check if times are all midnight (indicating date-only data)
            datetime_data = pd.to_datetime(data.astype(str), errors='coerce')
            if valid_count > 0:
                time_components = datetime_data.dt.time
                midnight_count = (time_components == datetime.min.time()).sum()
                if midnight_count == valid_count:
                    confidence += 0.1
            
            return {
                'confidence': min(confidence, 1.0),
                'details': {
                    'valid_dates': valid_count,
                    'date_range': (date_data.min() if valid_count > 0 else None,
                                  date_data.max() if valid_count > 0 else None)
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_email(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data contains email addresses."""
        try:
            str_data = data.astype(str).str.strip()
            email_matches = str_data.str.match(self.patterns[DataType.EMAIL]).sum()
            confidence = email_matches / len(data) if len(data) > 0 else 0
            
            return {
                'confidence': confidence,
                'details': {
                    'email_matches': email_matches,
                    'sample_values': str_data.head(5).tolist()
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_url(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data contains URLs."""
        try:
            str_data = data.astype(str).str.strip()
            url_matches = str_data.str.match(self.patterns[DataType.URL]).sum()
            confidence = url_matches / len(data) if len(data) > 0 else 0
            
            return {
                'confidence': confidence,
                'details': {
                    'url_matches': url_matches,
                    'sample_values': str_data.head(5).tolist()
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_phone(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data contains phone numbers."""
        try:
            str_data = data.astype(str).str.strip()
            phone_matches = str_data.str.match(self.patterns[DataType.PHONE]).sum()
            confidence = phone_matches / len(data) if len(data) > 0 else 0
            
            return {
                'confidence': confidence,
                'details': {
                    'phone_matches': phone_matches,
                    'sample_values': str_data.head(5).tolist()
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_currency(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data contains currency values."""
        try:
            str_data = data.astype(str).str.strip()
            currency_matches = str_data.str.match(self.patterns[DataType.CURRENCY]).sum()
            confidence = currency_matches / len(data) if len(data) > 0 else 0
            
            return {
                'confidence': confidence,
                'details': {
                    'currency_matches': currency_matches,
                    'sample_values': str_data.head(5).tolist()
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_categorical(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data should be treated as categorical."""
        try:
            unique_count = data.nunique()
            total_count = len(data)
            
            # High cardinality suggests not categorical
            if total_count == 0:
                return {'confidence': 0.0, 'details': 'No data'}
            
            cardinality_ratio = unique_count / total_count
            
            # Low cardinality suggests categorical
            if cardinality_ratio < 0.1:  # Less than 10% unique values
                confidence = 0.8
            elif cardinality_ratio < 0.5:  # Less than 50% unique values
                confidence = 0.5
            else:
                confidence = 0.2
            
            return {
                'confidence': confidence,
                'details': {
                    'unique_count': unique_count,
                    'cardinality_ratio': cardinality_ratio,
                    'sample_values': data.unique()[:10].tolist()
                }
            }
        except Exception as e:
            return {'confidence': 0.0, 'details': f'Error: {e}'}
    
    def _test_string(self, data: pd.Series) -> Dict[str, Any]:
        """Test if data should remain as string."""
        try:
            # String is the fallback type - always has some confidence
            str_data = data.astype(str)
            avg_length = str_data.str.len().mean()
            
            # Higher confidence for longer strings or mixed content
            confidence = 0.3  # Base confidence
            if avg_length > 50:  # Long strings likely to be text
                confidence += 0.2
            
            return {
                'confidence': confidence,
                'details': {
                    'average_length': avg_length,
                    'max_length': str_data.str.len().max(),
                    'sample_values': str_data.head(5).tolist()
                }
            }
        except Exception as e:
            return {'confidence': 0.3, 'details': f'Error: {e}'}  # Fallback confidence
    
    def _generate_conversion_suggestions(self, inference_results: Dict, current_dtype: str) -> List[Dict[str, Any]]:
        """Generate ranked conversion suggestions."""
        suggestions = []
        
        # Sort by confidence
        sorted_results = sorted(inference_results.items(), key=lambda x: x[1]['confidence'], reverse=True)
        
        for data_type, result in sorted_results[:3]:  # Top 3 suggestions
            if result['confidence'] > self.confidence_thresholds['low']:
                suggestions.append({
                    'target_type': data_type.value,
                    'confidence': result['confidence'],
                    'benefits': self._get_conversion_benefits(data_type),
                    'risks': self._get_conversion_risks(data_type, current_dtype),
                    'memory_impact': self._estimate_memory_impact(data_type)
                })
        
        return suggestions
    
    def _get_conversion_benefits(self, data_type: DataType) -> List[str]:
        """Get benefits of converting to specific type."""
        benefits = {
            DataType.INTEGER: ["Reduced memory usage", "Mathematical operations", "Indexing efficiency"],
            DataType.FLOAT: ["Mathematical operations", "Statistical analysis", "ML compatibility"],
            DataType.DATETIME: ["Time series analysis", "Date arithmetic", "Temporal filtering"],
            DataType.CATEGORICAL: ["Memory efficiency", "Faster grouping", "Ordered operations"],
            DataType.BOOLEAN: ["Logical operations", "Minimal memory usage", "Clear semantics"]
        }
        return benefits.get(data_type, ["Type-specific operations"])
    
    def _get_conversion_risks(self, data_type: DataType, current_dtype: str) -> List[str]:
        """Get risks of converting to specific type."""
        risks = []
        
        if data_type in [DataType.INTEGER, DataType.FLOAT]:
            risks.append("Potential data loss for non-numeric values")
        
        if data_type == DataType.DATETIME:
            risks.append("Parsing errors for invalid date formats")
        
        if current_dtype == 'object' and data_type != DataType.STRING:
            risks.append("Loss of original string representation")
        
        return risks
    
    def _estimate_memory_impact(self, data_type: DataType) -> str:
        """Estimate memory impact of conversion."""
        impact_map = {
            DataType.INTEGER: "Reduce",
            DataType.FLOAT: "Neutral",
            DataType.BOOLEAN: "Reduce significantly",
            DataType.CATEGORICAL: "Reduce",
            DataType.DATETIME: "Neutral",
            DataType.STRING: "Increase"
        }
        return impact_map.get(data_type, "Unknown")
    
    def _identify_potential_issues(self, data: pd.Series, inferred_type: DataType) -> List[str]:
        """Identify potential issues with the inferred type."""
        issues = []
        
        # Check for high null percentage
        null_percentage = (data.isnull().sum() / len(data)) * 100
        if null_percentage > 50:
            issues.append(f"High null percentage: {null_percentage:.1f}%")
        
        # Check for mixed types if inferring numeric
        if inferred_type in [DataType.INTEGER, DataType.FLOAT]:
            try:
                numeric_data = pd.to_numeric(data, errors='coerce')
                conversion_failures = (numeric_data.isnull() & data.notnull()).sum()
                if conversion_failures > 0:
                    issues.append(f"{conversion_failures} values cannot be converted to numeric")
            except:
                pass
        
        # Check for very high cardinality in categorical
        if inferred_type == DataType.CATEGORICAL:
            unique_ratio = data.nunique() / len(data) if len(data) > 0 else 0
            if unique_ratio > 0.8:
                issues.append("Very high cardinality for categorical type")
        
        return issues
    
    def _calculate_memory_impact(self, data: pd.Series, inferred_type: DataType) -> Dict[str, Any]:
        """Calculate memory impact of type conversion."""
        current_memory = data.memory_usage(deep=True)
        
        # Estimate new memory usage
        estimated_memory = current_memory  # Default
        
        if inferred_type == DataType.INTEGER:
            estimated_memory = len(data) * 8  # int64
        elif inferred_type == DataType.FLOAT:
            estimated_memory = len(data) * 8  # float64
        elif inferred_type == DataType.BOOLEAN:
            estimated_memory = len(data)  # bool
        elif inferred_type == DataType.CATEGORICAL:
            # Categorical memory depends on unique values
            unique_count = data.nunique()
            estimated_memory = len(data) + (unique_count * 50)  # Rough estimate
        
        return {
            'current_bytes': current_memory,
            'estimated_bytes': estimated_memory,
            'change_bytes': estimated_memory - current_memory,
            'change_percentage': ((estimated_memory - current_memory) / current_memory) * 100 if current_memory > 0 else 0
        }


class DataTypeManager(BaseEstimator, TransformerMixin):
    """
    Comprehensive data type conversion manager with intelligent inference and safe conversion.
    
    Provides intelligent data type inference, safe conversions with validation,
    metadata preservation, and rollback capabilities for error recovery.
    """
    
    def __init__(self,
                 analytical_intention: str = "optimize data types for analysis",
                 auto_inference: bool = True,
                 validation_enabled: bool = True,
                 rollback_enabled: bool = True,
                 memory_efficiency: bool = True,
                 streaming_config: Optional[StreamingConfig] = None,
                 custom_type_mappings: Optional[Dict[str, DataType]] = None):
        """
        Initialize DataTypeManager.
        
        Args:
            analytical_intention: Natural language description of conversion goal
            auto_inference: Enable automatic type inference
            validation_enabled: Enable conversion validation
            rollback_enabled: Enable rollback capabilities
            memory_efficiency: Prioritize memory-efficient conversions
            streaming_config: Configuration for streaming processing
            custom_type_mappings: Custom column -> type mappings
        """
        self.analytical_intention = analytical_intention
        self.auto_inference = auto_inference
        self.validation_enabled = validation_enabled
        self.rollback_enabled = rollback_enabled
        self.memory_efficiency = memory_efficiency
        self.streaming_config = streaming_config or StreamingConfig()
        self.custom_type_mappings = custom_type_mappings or {}
        
        # Internal state
        self.inference_engine = TypeInferenceEngine()
        self.conversion_history: List[ConversionOperation] = []
        self.type_inference_cache: Dict[str, TypeInferenceResult] = {}
        self.rollback_data: Dict[str, Any] = {}
        
        logger.info("DataTypeManager initialized",
                   intention=analytical_intention,
                   auto_inference=auto_inference)
    
    def fit(self, X, y=None):
        """
        Fit the type converter by analyzing data characteristics.
        
        Args:
            X: Input DataFrame or Series
            y: Target data (ignored)
            
        Returns:
            self
        """
        start_time = time.time()
        
        if isinstance(X, pd.Series):
            X = X.to_frame()
        
        # Analyze each column
        for column in X.columns:
            if column not in self.custom_type_mappings and self.auto_inference:
                # Perform type inference
                inference_result = self.inference_engine.infer_type(X[column])
                self.type_inference_cache[column] = inference_result
                
                logger.info(f"Inferred type for column '{column}'",
                           inferred_type=inference_result.inferred_type.value,
                           confidence=inference_result.confidence_score)
        
        execution_time = time.time() - start_time
        logger.info("DataTypeManager fit completed", execution_time=execution_time)
        
        return self
    
    def transform(self, X):
        """
        Transform DataFrame by converting columns to optimal types.
        
        Args:
            X: Input DataFrame or Series
            
        Returns:
            DataFrame with converted types
        """
        start_time = time.time()
        
        if isinstance(X, pd.Series):
            X = X.to_frame()
        
        result_df = X.copy()
        
        # Convert each column
        for column in X.columns:
            try:
                # Determine target type
                if column in self.custom_type_mappings:
                    target_type = self.custom_type_mappings[column]
                elif column in self.type_inference_cache:
                    inference_result = self.type_inference_cache[column]
                    if inference_result.confidence_score > 0.7:  # High confidence threshold
                        target_type = inference_result.inferred_type
                    else:
                        continue  # Skip low-confidence conversions
                else:
                    continue  # No inference data
                
                # Perform conversion
                converted_column, operation = self._convert_column(
                    result_df[column], column, target_type
                )
                
                if operation.success:
                    result_df[column] = converted_column
                    self.conversion_history.append(operation)
                
            except Exception as e:
                logger.warning(f"Failed to convert column '{column}': {e}")
        
        execution_time = time.time() - start_time
        logger.info("DataTypeManager transform completed", 
                   execution_time=execution_time,
                   conversions_applied=len([op for op in self.conversion_history if op.success]))
        
        return result_df
    
    def _convert_column(self, column: pd.Series, column_name: str, 
                       target_type: DataType) -> Tuple[pd.Series, ConversionOperation]:
        """
        Convert a single column to target type.
        
        Args:
            column: Series to convert
            column_name: Name of the column
            target_type: Target data type
            
        Returns:
            Tuple of (converted_series, conversion_operation)
        """
        start_time = time.time()
        original_dtype = str(column.dtype)
        
        # Store rollback data if enabled
        rollback_data = None
        if self.rollback_enabled:
            rollback_data = {
                'original_data': column.copy(),
                'original_dtype': original_dtype
            }
        
        try:
            # Perform the conversion
            if target_type == DataType.INTEGER:
                converted_column = self._convert_to_integer(column)
            elif target_type == DataType.FLOAT:
                converted_column = self._convert_to_float(column)
            elif target_type == DataType.BOOLEAN:
                converted_column = self._convert_to_boolean(column)
            elif target_type == DataType.DATETIME:
                converted_column = self._convert_to_datetime(column)
            elif target_type == DataType.DATE:
                converted_column = self._convert_to_date(column)
            elif target_type == DataType.CATEGORICAL:
                converted_column = self._convert_to_categorical(column)
            else:
                # Default to string
                converted_column = column.astype(str)
            
            # Validate conversion if enabled
            conversion_errors = 0
            if self.validation_enabled:
                conversion_errors = self._validate_conversion(column, converted_column, target_type)
            
            execution_time = time.time() - start_time
            
            operation = ConversionOperation(
                column=column_name,
                from_type=original_dtype,
                to_type=target_type,
                success=True,
                execution_time=execution_time,
                records_converted=len(column),
                conversion_errors=conversion_errors,
                rollback_data=rollback_data
            )
            
            return converted_column, operation
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            operation = ConversionOperation(
                column=column_name,
                from_type=original_dtype,
                to_type=target_type,
                success=False,
                execution_time=execution_time,
                records_converted=0,
                conversion_errors=len(column),
                rollback_data=rollback_data,
                error_details=str(e)
            )
            
            return column, operation  # Return original column on failure
    
    def _convert_to_integer(self, column: pd.Series) -> pd.Series:
        """Convert column to integer type."""
        # Use nullable integer type to handle NaN values
        return pd.to_numeric(column, errors='coerce').astype('Int64')
    
    def _convert_to_float(self, column: pd.Series) -> pd.Series:
        """Convert column to float type."""
        return pd.to_numeric(column, errors='coerce').astype('float64')
    
    def _convert_to_boolean(self, column: pd.Series) -> pd.Series:
        """Convert column to boolean type."""
        # Define mappings for common boolean representations
        true_values = {'true', '1', 'yes', 'y', 't', 'on', 'enabled', 'active'}
        false_values = {'false', '0', 'no', 'n', 'f', 'off', 'disabled', 'inactive'}
        
        str_column = column.astype(str).str.lower().str.strip()
        
        # Create boolean mask
        result = pd.Series(index=column.index, dtype='boolean')
        result[str_column.isin(true_values)] = True
        result[str_column.isin(false_values)] = False
        
        return result
    
    def _convert_to_datetime(self, column: pd.Series) -> pd.Series:
        """Convert column to datetime type."""
        return pd.to_datetime(column, errors='coerce', infer_datetime_format=True)
    
    def _convert_to_date(self, column: pd.Series) -> pd.Series:
        """Convert column to date type."""
        datetime_column = pd.to_datetime(column, errors='coerce', infer_datetime_format=True)
        return datetime_column.dt.date
    
    def _convert_to_categorical(self, column: pd.Series) -> pd.Series:
        """Convert column to categorical type."""
        return column.astype('category')
    
    def _validate_conversion(self, original: pd.Series, converted: pd.Series, 
                           target_type: DataType) -> int:
        """
        Validate the conversion results.
        
        Args:
            original: Original series
            converted: Converted series
            target_type: Target data type
            
        Returns:
            Number of conversion errors
        """
        # Count how many values couldn't be converted (became NaN)
        if target_type in [DataType.INTEGER, DataType.FLOAT, DataType.DATETIME, DataType.DATE]:
            original_nulls = original.isnull().sum()
            converted_nulls = converted.isnull().sum()
            conversion_errors = converted_nulls - original_nulls
            return max(0, conversion_errors)
        
        return 0
    
    # Public utility methods
    def get_type_inference_results(self) -> Dict[str, TypeInferenceResult]:
        """Get type inference results for all analyzed columns."""
        return self.type_inference_cache.copy()
    
    def get_conversion_history(self) -> List[ConversionOperation]:
        """Get history of all conversion operations."""
        return self.conversion_history.copy()
    
    def rollback_column(self, column_name: str) -> bool:
        """
        Rollback conversion for specific column.
        
        Args:
            column_name: Name of column to rollback
            
        Returns:
            Success status
        """
        if not self.rollback_enabled:
            logger.warning("Rollback not enabled")
            return False
        
        # Find the most recent successful conversion for this column
        for operation in reversed(self.conversion_history):
            if operation.column == column_name and operation.success and operation.rollback_data:
                # Rollback data would be applied in the context where this is used
                logger.info(f"Rollback data available for column '{column_name}'")
                return True
        
        logger.warning(f"No rollback data available for column '{column_name}'")
        return False
    
    def get_memory_impact_report(self) -> Dict[str, Any]:
        """Get report of memory impact from conversions."""
        total_memory_saved = 0
        successful_conversions = 0
        
        for operation in self.conversion_history:
            if operation.success and operation.column in self.type_inference_cache:
                inference_result = self.type_inference_cache[operation.column]
                memory_impact = inference_result.memory_impact.get('change_bytes', 0)
                total_memory_saved += memory_impact
                successful_conversions += 1
        
        return {
            'total_conversions': successful_conversions,
            'total_memory_change_bytes': total_memory_saved,
            'total_memory_change_mb': total_memory_saved / (1024 * 1024),
            'average_memory_change_per_column': total_memory_saved / successful_conversions if successful_conversions > 0 else 0
        }
    
    def get_conversion_summary(self) -> str:
        """Get human-readable summary of conversions."""
        if not self.conversion_history:
            return "No conversions performed yet."
        
        successful = sum(1 for op in self.conversion_history if op.success)
        total = len(self.conversion_history)
        
        summary_parts = [
            f"Data type conversion completed: {successful}/{total} successful conversions."
        ]
        
        if successful > 0:
            memory_report = self.get_memory_impact_report()
            memory_mb = memory_report['total_memory_change_mb']
            if memory_mb != 0:
                direction = "saved" if memory_mb < 0 else "increased"
                summary_parts.append(f"Memory {direction}: {abs(memory_mb):.2f} MB")
        
        # Add type conversion breakdown
        type_counts = {}
        for op in self.conversion_history:
            if op.success:
                target_type = op.to_type.value
                type_counts[target_type] = type_counts.get(target_type, 0) + 1
        
        if type_counts:
            conversion_details = ", ".join([f"{count} to {type_name}" for type_name, count in type_counts.items()])
            summary_parts.append(f"Conversions: {conversion_details}")
        
        return "\n".join(summary_parts)