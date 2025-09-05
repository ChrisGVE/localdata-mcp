"""
Phase 1 Tool Transformers - sklearn-compatible wrappers for profile_table, detect_data_types, and analyze_distributions.

This module implements sklearn BaseEstimator and TransformerMixin interfaces for the existing Phase 1 tools
to enable seamless pipeline integration while maintaining 100% API compatibility.

Key Features:
- Full sklearn pipeline compatibility
- Preserved streaming capabilities 
- Memory-efficient processing
- Comprehensive parameter validation
- Backward compatible interfaces
"""

import json
import time
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ..logging_manager import get_logger
from .base import PipelineState, DataSourceType

logger = get_logger(__name__)


class ProfileTableTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for comprehensive data profiling.
    
    Wraps the existing profile_table functionality in a sklearn pipeline-compatible interface
    while preserving all original capabilities including streaming, sampling, and distribution analysis.
    
    Parameters:
    -----------
    sample_size : int, default=10000
        Number of rows to sample for analysis (0 = all rows)
    include_distributions : bool, default=True
        Whether to include distribution analysis for numeric columns
    connection_name : str, default=None
        Database connection name (if using database source)
    table_name : str, default=None
        Table name to profile (mutually exclusive with query)
    query : str, default=None
        Custom SQL query to profile (mutually exclusive with table_name)
        
    Attributes:
    -----------
    profile_ : dict
        Generated data profile after fitting
    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during fit
    n_features_in_ : int
        Number of features seen during fit
    state_ : PipelineState
        Current transformer state
    """
    
    def __init__(self, 
                 sample_size: int = 10000,
                 include_distributions: bool = True,
                 connection_name: Optional[str] = None,
                 table_name: Optional[str] = None,
                 query: Optional[str] = None):
        self.sample_size = sample_size
        self.include_distributions = include_distributions
        self.connection_name = connection_name
        self.table_name = table_name
        self.query = query
        
        # Internal state
        self.state_ = PipelineState.INITIALIZED
        self.profile_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y=None):
        """
        Fit the profiler by analyzing the input data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or pandas.DataFrame
            Training data to profile
        y : array-like of shape (n_samples,), default=None
            Target values (ignored)
            
        Returns:
        --------
        self : ProfileTableTransformer
            Fitted transformer
        """
        # Input validation and conversion
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self.feature_names_in_ = np.array(X.columns)
        else:
            X = check_array(X, accept_sparse=False, force_all_finite=False)
            df = pd.DataFrame(X)
            self.feature_names_in_ = np.array([f'feature_{i}' for i in range(X.shape[1])])
            
        self.n_features_in_ = df.shape[1]
        self.state_ = PipelineState.EXECUTING
        
        try:
            # Apply sampling if specified
            if self.sample_size > 0 and len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)
                
            # Generate comprehensive profile using the existing logic
            self.profile_ = self._generate_data_profile(df, self.include_distributions)
            
            # Add metadata
            self.profile_['metadata'] = {
                'source_type': 'transformer_input',
                'sample_size': self.sample_size,
                'actual_rows_analyzed': len(df),
                'profiling_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'include_distributions': self.include_distributions,
                'pipeline_state': self.state_.value
            }
            
            self.state_ = PipelineState.FITTED
            logger.info(f"ProfileTableTransformer fitted successfully with {len(df)} rows, {df.shape[1]} columns")
            
        except Exception as e:
            self.state_ = PipelineState.ERROR
            logger.error(f"Error fitting ProfileTableTransformer: {e}")
            raise
            
        return self
        
    def transform(self, X):
        """
        Transform is identity for profiling - returns input unchanged.
        Profile data is available via get_profile() method.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X : array-like of shape (n_samples, n_features)
            Unchanged input data
        """
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            return X
        else:
            return check_array(X, accept_sparse=False, force_all_finite=False)
            
    def get_profile(self) -> Dict[str, Any]:
        """
        Get the generated data profile.
        
        Returns:
        --------
        profile : dict
            Comprehensive data profile with statistics and quality metrics
        """
        check_is_fitted(self)
        return self.profile_
        
    def get_profile_json(self) -> str:
        """
        Get the generated data profile as JSON string (backward compatibility).
        
        Returns:
        --------
        profile_json : str
            Comprehensive data profile in JSON format
        """
        check_is_fitted(self)
        return json.dumps(self.profile_, indent=2, default=str)
        
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation (sklearn compatibility).
        
        Parameters:
        -----------
        input_features : array-like of str or None, default=None
            Input features. If None, uses feature_names_in_.
            
        Returns:
        --------
        feature_names_out : ndarray of str
            Transformed feature names (same as input for profiling)
        """
        check_is_fitted(self)
        
        if input_features is None:
            return self.feature_names_in_.copy()
        else:
            return np.array(input_features)
            
    def get_composition_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for pipeline composition and tool chaining.
        
        Returns:
        --------
        composition_metadata : dict
            Metadata for downstream pipeline composition including:
            - Statistical summaries for each column
            - Data quality scores
            - Processing hints for downstream tools
            - Recommended next steps
        """
        check_is_fitted(self)
        
        if not self.profile_:
            return {}
            
        # Extract composition-relevant metadata
        metadata = {
            'tool_type': 'profiler',
            'processing_stage': 'data_understanding',
            'data_shape': {
                'rows': self.profile_['summary']['total_rows'],
                'columns': self.profile_['summary']['total_columns']
            },
            'data_quality': self.profile_.get('data_quality', {}),
            'column_types': {},
            'processing_hints': {},
            'recommended_next_steps': []
        }
        
        # Extract column-level information
        for col_name, col_info in self.profile_.get('columns', {}).items():
            metadata['column_types'][col_name] = {
                'data_type': col_info.get('data_type', 'unknown'),
                'null_percentage': col_info.get('null_percentage', 0),
                'unique_percentage': col_info.get('unique_percentage', 0),
                'has_outliers': col_info.get('outliers', {}).get('count', 0) > 0 if 'outliers' in col_info else False
            }
            
            # Generate processing hints
            hints = []
            if col_info.get('null_percentage', 0) > 5:
                hints.append('missing_value_imputation')
            if 'outliers' in col_info and col_info['outliers'].get('count', 0) > 0:
                hints.append('outlier_handling')
            if pd.api.types.is_numeric_dtype(col_info.get('data_type', '')):
                hints.append('scaling_normalization')
            if col_info.get('unique_percentage', 0) > 95:
                hints.append('potential_identifier')
                
            metadata['processing_hints'][col_name] = hints
        
        # Generate recommended next steps
        overall_quality = metadata['data_quality'].get('overall_score', 100)
        if overall_quality < 80:
            metadata['recommended_next_steps'].append('data_cleaning')
        if any('missing_value_imputation' in hints for hints in metadata['processing_hints'].values()):
            metadata['recommended_next_steps'].append('missing_value_treatment')
        if any('outlier_handling' in hints for hints in metadata['processing_hints'].values()):
            metadata['recommended_next_steps'].append('outlier_analysis')
        if any('scaling_normalization' in hints for hints in metadata['processing_hints'].values()):
            metadata['recommended_next_steps'].append('feature_scaling')
            
        return metadata
        
    def _generate_data_profile(self, df: pd.DataFrame, include_distributions: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive data profile for a DataFrame.
        
        This method replicates the existing _generate_data_profile logic
        to maintain 100% compatibility with the original implementation.
        
        Args:
            df: Pandas DataFrame to profile
            include_distributions: Whether to include distribution analysis
            
        Returns:
            Dictionary containing comprehensive profile data
        """
        profile = {
            'summary': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'completeness_score': ((df.notna().sum().sum()) / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
            },
            'columns': {}
        }
        
        # Profile each column
        for column in df.columns:
            col_data = df[column]
            col_profile = {
                'data_type': str(col_data.dtype),
                'non_null_count': int(col_data.notna().sum()),
                'null_count': int(col_data.isnull().sum()),
                'null_percentage': float((col_data.isnull().sum() / len(col_data)) * 100) if len(col_data) > 0 else 0,
                'unique_count': int(col_data.nunique()),
                'unique_percentage': float((col_data.nunique() / len(col_data)) * 100) if len(col_data) > 0 else 0,
                'memory_usage_bytes': int(col_data.memory_usage(deep=True))
            }
            
            # Add basic statistics for non-null values
            non_null_data = col_data.dropna()
            
            if len(non_null_data) > 0:
                # Most common values
                value_counts = non_null_data.value_counts().head(5)
                col_profile['top_values'] = {
                    str(val): int(count) for val, count in value_counts.items()
                }
                
                # Type-specific analysis
                if pd.api.types.is_numeric_dtype(col_data):
                    col_profile.update(self._profile_numeric_column(non_null_data, include_distributions))
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    col_profile.update(self._profile_datetime_column(non_null_data))
                else:
                    col_profile.update(self._profile_text_column(non_null_data))
            
            profile['columns'][column] = col_profile
        
        # Calculate data quality metrics
        profile['data_quality'] = self._calculate_data_quality_metrics(df)
        
        return profile
        
    def _profile_numeric_column(self, series: pd.Series, include_distributions: bool) -> Dict[str, Any]:
        """Profile numeric column with statistical analysis."""
        profile = {
            'min_value': float(series.min()),
            'max_value': float(series.max()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std_deviation': float(series.std()) if len(series) > 1 else 0,
            'variance': float(series.var()) if len(series) > 1 else 0,
            'skewness': float(series.skew()) if len(series) > 2 else 0,
            'kurtosis': float(series.kurtosis()) if len(series) > 3 else 0
        }
        
        # Quartiles
        try:
            quartiles = series.quantile([0.25, 0.5, 0.75])
            q25_val = float(quartiles[0.25])
            q75_val = float(quartiles[0.75])
            
            # Add quartiles both as top-level values for easy access and nested for detail
            profile['q25'] = q25_val  # 25th percentile (Q1)
            profile['q75'] = q75_val  # 75th percentile (Q3)
            
            profile['quartiles'] = {
                'q1': q25_val,
                'q2': float(quartiles[0.5]),
                'q3': q75_val,
                'iqr': float(q75_val - q25_val)
            }
        except Exception:
            profile['quartiles'] = None
            profile['q25'] = None
            profile['q75'] = None
        
        # Outlier detection using IQR method
        if profile['quartiles']:
            q1, q3 = profile['quartiles']['q1'], profile['quartiles']['q3']
            iqr = profile['quartiles']['iqr']
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            profile['outliers'] = {
                'count': int(len(outliers)),
                'percentage': float((len(outliers) / len(series)) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        
        # Distribution analysis if requested
        if include_distributions:
            try:
                # Create histogram data
                hist, bin_edges = np.histogram(series.values, bins=20)
                profile['histogram'] = {
                    'counts': [int(x) for x in hist],
                    'bin_edges': [float(x) for x in bin_edges]
                }
                
                # Percentile distribution
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                profile['percentiles'] = {
                    f'p{p}': float(series.quantile(p/100)) for p in percentiles
                }
            except Exception as e:
                logger.warning(f"Could not generate distribution data: {e}")
                profile['histogram'] = None
                profile['percentiles'] = None
        
        return profile
        
    def _profile_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column with temporal analysis."""
        profile = {
            'min_date': str(series.min()),
            'max_date': str(series.max()),
            'date_range_days': (series.max() - series.min()).days,
        }
        
        # Extract time components for analysis
        try:
            profile['year_range'] = {
                'min_year': int(series.dt.year.min()),
                'max_year': int(series.dt.year.max())
            }
            profile['month_distribution'] = series.dt.month.value_counts().to_dict()
            profile['weekday_distribution'] = series.dt.day_name().value_counts().to_dict()
        except Exception as e:
            logger.warning(f"Could not analyze datetime components: {e}")
            
        return profile
        
    def _profile_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile text column with string analysis."""
        str_series = series.astype(str)
        
        profile = {
            'min_length': int(str_series.str.len().min()),
            'max_length': int(str_series.str.len().max()),
            'avg_length': float(str_series.str.len().mean()),
            'std_length': float(str_series.str.len().std()) if len(str_series) > 1 else 0
        }
        
        # Pattern analysis
        try:
            profile['patterns'] = {
                'contains_digits': int(str_series.str.contains(r'\d', na=False).sum()),
                'contains_letters': int(str_series.str.contains(r'[a-zA-Z]', na=False).sum()),
                'contains_special_chars': int(str_series.str.contains(r'[^a-zA-Z0-9\s]', na=False).sum()),
                'all_uppercase': int(str_series.str.isupper().sum()),
                'all_lowercase': int(str_series.str.islower().sum())
            }
        except Exception as e:
            logger.warning(f"Could not analyze text patterns: {e}")
            profile['patterns'] = {}
            
        return profile
        
    def _calculate_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics."""
        total_cells = len(df) * len(df.columns)
        
        if total_cells == 0:
            return {
                'completeness': 0,
                'consistency': 0,
                'validity': 0,
                'accuracy': 0,
                'overall_score': 0
            }
        
        # Completeness: percentage of non-null values
        non_null_cells = df.notna().sum().sum()
        completeness = (non_null_cells / total_cells) * 100
        
        # Consistency: low variance in data types and formats per column
        consistency_scores = []
        for column in df.columns:
            col_data = df[column].dropna()
            if len(col_data) > 0:
                # Simple consistency metric based on data type uniformity
                if pd.api.types.is_numeric_dtype(col_data):
                    consistency_scores.append(95)  # Numeric data generally consistent
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    consistency_scores.append(90)  # Datetime data generally consistent
                else:
                    # Text data consistency based on length variance
                    lengths = col_data.astype(str).str.len()
                    if len(lengths) > 1:
                        cv = lengths.std() / lengths.mean() if lengths.mean() > 0 else 1
                        consistency_scores.append(max(0, 100 - (cv * 50)))
                    else:
                        consistency_scores.append(100)
            else:
                consistency_scores.append(0)
        
        consistency = np.mean(consistency_scores) if consistency_scores else 0
        
        # Validity: percentage of values that conform to expected patterns
        validity_scores = []
        for column in df.columns:
            col_data = df[column].dropna()
            if len(col_data) > 0:
                # Simple validity check - non-empty strings for text, finite numbers for numeric
                if pd.api.types.is_numeric_dtype(col_data):
                    valid_count = np.isfinite(col_data).sum()
                    validity_scores.append((valid_count / len(col_data)) * 100)
                else:
                    # For text, check for non-empty strings
                    str_data = col_data.astype(str)
                    valid_count = (str_data.str.len() > 0).sum()
                    validity_scores.append((valid_count / len(str_data)) * 100)
            else:
                validity_scores.append(0)
        
        validity = np.mean(validity_scores) if validity_scores else 0
        
        # Accuracy: placeholder (would need reference data for true accuracy)
        accuracy = min(completeness, consistency, validity)  # Conservative estimate
        
        # Overall score: weighted average
        overall_score = (completeness * 0.3 + consistency * 0.25 + validity * 0.25 + accuracy * 0.2)
        
        return {
            'completeness': round(completeness, 2),
            'consistency': round(consistency, 2),
            'validity': round(validity, 2),
            'accuracy': round(accuracy, 2),
            'overall_score': round(overall_score, 2)
        }


class DataTypeDetectorTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for advanced data type detection.
    
    Wraps the existing detect_data_types functionality in a sklearn pipeline-compatible interface
    while preserving semantic type detection, confidence scoring, and pattern recognition.
    
    Parameters:
    -----------
    sample_size : int, default=1000
        Number of rows to sample for type detection
    confidence_threshold : float, default=0.8
        Minimum confidence threshold for type detection
    include_semantic_types : bool, default=True
        Whether to include semantic type detection (email, phone, etc.)
        
    Attributes:
    -----------
    detected_types_ : dict
        Detected data types after fitting
    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during fit
    n_features_in_ : int
        Number of features seen during fit
    state_ : PipelineState
        Current transformer state
    """
    
    def __init__(self,
                 sample_size: int = 1000,
                 confidence_threshold: float = 0.8,
                 include_semantic_types: bool = True):
        self.sample_size = sample_size
        self.confidence_threshold = confidence_threshold
        self.include_semantic_types = include_semantic_types
        
        # Internal state
        self.state_ = PipelineState.INITIALIZED
        self.detected_types_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y=None):
        """
        Fit the type detector by analyzing the input data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or pandas.DataFrame
            Training data to analyze types
        y : array-like of shape (n_samples,), default=None
            Target values (ignored)
            
        Returns:
        --------
        self : DataTypeDetectorTransformer
            Fitted transformer
        """
        # Input validation and conversion
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self.feature_names_in_ = np.array(X.columns)
        else:
            X = check_array(X, accept_sparse=False, force_all_finite=False)
            df = pd.DataFrame(X)
            self.feature_names_in_ = np.array([f'feature_{i}' for i in range(X.shape[1])])
            
        self.n_features_in_ = df.shape[1]
        self.state_ = PipelineState.EXECUTING
        
        try:
            # Apply sampling if specified
            if self.sample_size > 0 and len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)
                
            # Detect data types using existing logic
            self.detected_types_ = self._detect_column_types(df)
            
            # Add metadata
            self.detected_types_['metadata'] = {
                'source_type': 'transformer_input',
                'sample_size': self.sample_size,
                'actual_rows_analyzed': len(df),
                'detection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'confidence_threshold': self.confidence_threshold,
                'include_semantic_types': self.include_semantic_types,
                'pipeline_state': self.state_.value
            }
            
            self.state_ = PipelineState.FITTED
            logger.info(f"DataTypeDetectorTransformer fitted successfully with {len(df)} rows, {df.shape[1]} columns")
            
        except Exception as e:
            self.state_ = PipelineState.ERROR
            logger.error(f"Error fitting DataTypeDetectorTransformer: {e}")
            raise
            
        return self
        
    def transform(self, X):
        """
        Transform is identity for type detection - returns input unchanged.
        Detected types are available via get_detected_types() method.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X : array-like of shape (n_samples, n_features)
            Unchanged input data
        """
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            return X
        else:
            return check_array(X, accept_sparse=False, force_all_finite=False)
            
    def get_detected_types(self) -> Dict[str, Any]:
        """
        Get the detected data types.
        
        Returns:
        --------
        types : dict
            Detected data types with confidence scores and semantic information
        """
        check_is_fitted(self)
        return self.detected_types_
        
    def get_detected_types_json(self) -> str:
        """
        Get the detected data types as JSON string (backward compatibility).
        
        Returns:
        --------
        types_json : str
            Detected data types in JSON format
        """
        check_is_fitted(self)
        return json.dumps(self.detected_types_, indent=2, default=str)
        
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation (sklearn compatibility).
        
        Parameters:
        -----------
        input_features : array-like of str or None, default=None
            Input features. If None, uses feature_names_in_.
            
        Returns:
        --------
        feature_names_out : ndarray of str
            Transformed feature names (same as input for type detection)
        """
        check_is_fitted(self)
        
        if input_features is None:
            return self.feature_names_in_.copy()
        else:
            return np.array(input_features)
            
    def get_composition_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for pipeline composition and tool chaining.
        
        Returns:
        --------
        composition_metadata : dict
            Metadata for downstream pipeline composition including:
            - Detected types and confidence scores
            - Semantic type information
            - Type conversion recommendations
            - Processing hints for downstream tools
        """
        check_is_fitted(self)
        
        if not self.detected_types_:
            return {}
            
        # Extract composition-relevant metadata
        metadata = {
            'tool_type': 'type_detector',
            'processing_stage': 'data_understanding',
            'overall_confidence': self.detected_types_.get('summary', {}).get('detection_confidence', 0),
            'type_conversions': {},
            'semantic_types': {},
            'processing_hints': {},
            'recommended_next_steps': []
        }
        
        # Extract column-level type information
        for col_name, col_info in self.detected_types_.get('columns', {}).items():
            detected_type = col_info.get('detected_type', 'unknown')
            confidence = col_info.get('confidence', 0)
            semantic_type = col_info.get('semantic_type')
            
            # Type conversion recommendations
            if detected_type == 'numeric_string' and confidence > self.confidence_threshold:
                metadata['type_conversions'][col_name] = {
                    'from': 'string',
                    'to': 'numeric',
                    'confidence': confidence,
                    'conversion_function': 'pd.to_numeric'
                }
            elif detected_type == 'date_string' and confidence > self.confidence_threshold:
                metadata['type_conversions'][col_name] = {
                    'from': 'string',
                    'to': 'datetime',
                    'confidence': confidence,
                    'conversion_function': 'pd.to_datetime'
                }
            elif detected_type == 'boolean_string' and confidence > self.confidence_threshold:
                metadata['type_conversions'][col_name] = {
                    'from': 'string',
                    'to': 'boolean',
                    'confidence': confidence,
                    'conversion_function': 'astype(bool)'
                }
                
            # Semantic type information
            if semantic_type:
                metadata['semantic_types'][col_name] = {
                    'type': semantic_type,
                    'validation_required': True,
                    'special_handling': self._get_semantic_handling_hints(semantic_type)
                }
                
            # Processing hints
            hints = []
            if confidence < self.confidence_threshold:
                hints.append('manual_type_verification')
            if semantic_type == 'email':
                hints.append('email_validation')
            elif semantic_type == 'phone':
                hints.append('phone_formatting')
            elif semantic_type == 'url':
                hints.append('url_validation')
            elif detected_type in ['integer', 'float']:
                hints.append('numeric_analysis')
            elif detected_type == 'datetime':
                hints.append('temporal_analysis')
                
            metadata['processing_hints'][col_name] = hints
        
        # Generate recommended next steps
        if len(metadata['type_conversions']) > 0:
            metadata['recommended_next_steps'].append('type_conversion')
        if any(info['type'] in ['email', 'phone', 'url'] for info in metadata['semantic_types'].values()):
            metadata['recommended_next_steps'].append('data_validation')
        if metadata['overall_confidence'] < 0.9:
            metadata['recommended_next_steps'].append('manual_type_review')
            
        return metadata
        
    def _get_semantic_handling_hints(self, semantic_type: str) -> List[str]:
        """Get special handling hints for semantic types."""
        hints_map = {
            'email': ['validation', 'privacy_masking', 'domain_analysis'],
            'phone': ['formatting', 'country_code_detection', 'privacy_masking'],
            'url': ['validation', 'domain_extraction', 'security_check'],
            'zip_code': ['geographic_analysis', 'validation'],
        }
        return hints_map.get(semantic_type, [])
        
    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data types for all columns in DataFrame.
        
        This method replicates the existing detect_data_types logic
        to maintain 100% compatibility with the original implementation.
        """
        results = {
            'summary': {
                'total_columns': len(df.columns),
                'columns_analyzed': len(df.columns),
                'detection_confidence': 0.0
            },
            'columns': {}
        }
        
        confidence_scores = []
        
        for column in df.columns:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                results['columns'][column] = {
                    'detected_type': 'unknown',
                    'confidence': 0.0,
                    'pandas_dtype': str(df[column].dtype),
                    'semantic_type': None,
                    'patterns': [],
                    'sample_values': []
                }
                confidence_scores.append(0.0)
                continue
                
            # Get basic type information
            pandas_dtype = str(col_data.dtype)
            sample_values = [str(val) for val in col_data.head(5).tolist()]
            
            # Detect primary type
            detected_type, confidence = self._detect_primary_type(col_data)
            
            # Detect semantic type if enabled
            semantic_type = None
            if self.include_semantic_types:
                semantic_type = self._detect_semantic_type(col_data)
                
            # Detect patterns
            patterns = self._detect_patterns(col_data)
            
            results['columns'][column] = {
                'detected_type': detected_type,
                'confidence': confidence,
                'pandas_dtype': pandas_dtype,
                'semantic_type': semantic_type,
                'patterns': patterns,
                'sample_values': sample_values
            }
            
            confidence_scores.append(confidence)
        
        # Calculate overall confidence
        results['summary']['detection_confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return results
        
    def _detect_primary_type(self, series: pd.Series) -> Tuple[str, float]:
        """Detect primary data type with confidence score."""
        # Start with pandas dtype
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return 'integer', 0.95
            else:
                return 'float', 0.95
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime', 0.95
        elif pd.api.types.is_bool_dtype(series):
            return 'boolean', 0.95
        
        # For object dtype, do deeper analysis
        str_series = series.astype(str)
        
        # Check for numeric strings
        try:
            pd.to_numeric(str_series)
            return 'numeric_string', 0.9
        except:
            pass
            
        # Check for date strings
        try:
            pd.to_datetime(str_series)
            return 'date_string', 0.85
        except:
            pass
            
        # Check for boolean strings
        bool_values = {'true', 'false', 'yes', 'no', '1', '0', 'y', 'n'}
        if set(str_series.str.lower().unique()).issubset(bool_values):
            return 'boolean_string', 0.8
            
        # Default to string
        return 'string', 0.7
        
    def _detect_semantic_type(self, series: pd.Series) -> Optional[str]:
        """Detect semantic type (email, phone, URL, etc.)."""
        str_series = series.astype(str)
        
        # Email pattern
        if str_series.str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$').any():
            return 'email'
            
        # Phone pattern (simple)
        if str_series.str.match(r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$').any():
            return 'phone'
            
        # URL pattern
        if str_series.str.match(r'^https?://').any():
            return 'url'
            
        # ZIP code pattern
        if str_series.str.match(r'^\d{5}(-\d{4})?$').any():
            return 'zip_code'
            
        return None
        
    def _detect_patterns(self, series: pd.Series) -> List[str]:
        """Detect common patterns in the data."""
        patterns = []
        str_series = series.astype(str)
        
        # Check for common patterns
        if str_series.str.contains(r'^\d+$', na=False).any():
            patterns.append('digits_only')
            
        if str_series.str.contains(r'^[A-Z]+$', na=False).any():
            patterns.append('uppercase_only')
            
        if str_series.str.contains(r'^[a-z]+$', na=False).any():
            patterns.append('lowercase_only')
            
        if str_series.str.contains(r'^\d{4}-\d{2}-\d{2}$', na=False).any():
            patterns.append('date_iso')
            
        if str_series.str.contains(r'^[A-Z]{2,}$', na=False).any():
            patterns.append('abbreviation')
            
        return patterns


class DistributionAnalyzerTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for distribution analysis.
    
    Wraps the existing analyze_distributions functionality in a sklearn pipeline-compatible interface
    while preserving histogram generation, percentile calculations, and statistical pattern detection.
    
    Parameters:
    -----------
    sample_size : int, default=10000
        Number of rows to sample for distribution analysis
    bins : int, default=20
        Number of bins for histogram generation
    percentiles : list, default=None
        List of percentiles to calculate (if None, uses default set)
        
    Attributes:
    -----------
    distributions_ : dict
        Analyzed distributions after fitting
    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during fit
    n_features_in_ : int
        Number of features seen during fit
    state_ : PipelineState
        Current transformer state
    """
    
    def __init__(self,
                 sample_size: int = 10000,
                 bins: int = 20,
                 percentiles: Optional[List[float]] = None):
        self.sample_size = sample_size
        self.bins = bins
        self.percentiles = percentiles or [1, 5, 10, 25, 50, 75, 90, 95, 99]
        
        # Internal state
        self.state_ = PipelineState.INITIALIZED
        self.distributions_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y=None):
        """
        Fit the distribution analyzer by analyzing the input data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features) or pandas.DataFrame
            Training data to analyze distributions
        y : array-like of shape (n_samples,), default=None
            Target values (ignored)
            
        Returns:
        --------
        self : DistributionAnalyzerTransformer
            Fitted transformer
        """
        # Input validation and conversion
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self.feature_names_in_ = np.array(X.columns)
        else:
            X = check_array(X, accept_sparse=False, force_all_finite=False)
            df = pd.DataFrame(X)
            self.feature_names_in_ = np.array([f'feature_{i}' for i in range(X.shape[1])])
            
        self.n_features_in_ = df.shape[1]
        self.state_ = PipelineState.EXECUTING
        
        try:
            # Apply sampling if specified
            if self.sample_size > 0 and len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)
                
            # Analyze distributions using existing logic
            self.distributions_ = self._analyze_distributions(df)
            
            # Add metadata
            self.distributions_['metadata'] = {
                'source_type': 'transformer_input',
                'sample_size': self.sample_size,
                'actual_rows_analyzed': len(df),
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'bins': self.bins,
                'percentiles': self.percentiles,
                'pipeline_state': self.state_.value
            }
            
            self.state_ = PipelineState.FITTED
            logger.info(f"DistributionAnalyzerTransformer fitted successfully with {len(df)} rows, {df.shape[1]} columns")
            
        except Exception as e:
            self.state_ = PipelineState.ERROR
            logger.error(f"Error fitting DistributionAnalyzerTransformer: {e}")
            raise
            
        return self
        
    def transform(self, X):
        """
        Transform is identity for distribution analysis - returns input unchanged.
        Distribution data is available via get_distributions() method.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X : array-like of shape (n_samples, n_features)
            Unchanged input data
        """
        check_is_fitted(self)
        
        if isinstance(X, pd.DataFrame):
            return X
        else:
            return check_array(X, accept_sparse=False, force_all_finite=False)
            
    def get_distributions(self) -> Dict[str, Any]:
        """
        Get the analyzed distributions.
        
        Returns:
        --------
        distributions : dict
            Distribution analysis with histograms, percentiles, and patterns
        """
        check_is_fitted(self)
        return self.distributions_
        
    def get_distributions_json(self) -> str:
        """
        Get the distribution analysis as JSON string (backward compatibility).
        
        Returns:
        --------
        distributions_json : str
            Distribution analysis in JSON format
        """
        check_is_fitted(self)
        return json.dumps(self.distributions_, indent=2, default=str)
        
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation (sklearn compatibility).
        
        Parameters:
        -----------
        input_features : array-like of str or None, default=None
            Input features. If None, uses feature_names_in_.
            
        Returns:
        --------
        feature_names_out : ndarray of str
            Transformed feature names (same as input for distribution analysis)
        """
        check_is_fitted(self)
        
        if input_features is None:
            return self.feature_names_in_.copy()
        else:
            return np.array(input_features)
            
    def get_composition_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for pipeline composition and tool chaining.
        
        Returns:
        --------
        composition_metadata : dict
            Metadata for downstream pipeline composition including:
            - Distribution characteristics and patterns
            - Statistical summaries for each column
            - Normality and outlier information
            - Processing hints for downstream tools
        """
        check_is_fitted(self)
        
        if not self.distributions_:
            return {}
            
        # Extract composition-relevant metadata
        metadata = {
            'tool_type': 'distribution_analyzer',
            'processing_stage': 'statistical_analysis',
            'summary': self.distributions_.get('summary', {}),
            'distribution_patterns': {},
            'normality_tests': {},
            'outlier_information': {},
            'processing_hints': {},
            'recommended_next_steps': []
        }
        
        # Extract distribution information for each column
        for col_name, dist_info in self.distributions_.get('distributions', {}).items():
            if dist_info.get('type') == 'numeric':
                # Numeric distribution patterns
                shape_metrics = dist_info.get('shape_metrics', {})
                metadata['distribution_patterns'][col_name] = {
                    'type': 'numeric',
                    'is_normal': shape_metrics.get('is_normal_distributed', False),
                    'skewness': shape_metrics.get('skewness', 0),
                    'kurtosis': shape_metrics.get('kurtosis', 0),
                    'outliers_count': shape_metrics.get('outliers_count', 0)
                }
                
                # Normality test results
                metadata['normality_tests'][col_name] = {
                    'is_normal': shape_metrics.get('is_normal_distributed', False),
                    'recommendation': 'parametric_tests' if shape_metrics.get('is_normal_distributed') else 'non_parametric_tests'
                }
                
                # Outlier information
                outliers_count = shape_metrics.get('outliers_count', 0)
                if outliers_count > 0:
                    metadata['outlier_information'][col_name] = {
                        'count': outliers_count,
                        'percentage': (outliers_count / dist_info.get('summary_stats', {}).get('count', 1)) * 100,
                        'treatment_needed': outliers_count > dist_info.get('summary_stats', {}).get('count', 0) * 0.05  # More than 5%
                    }
                    
                # Processing hints for numeric columns
                hints = []
                if not shape_metrics.get('is_normal_distributed', True):
                    hints.extend(['log_transformation', 'box_cox_transformation'])
                if outliers_count > 0:
                    hints.extend(['outlier_treatment', 'robust_scaling'])
                if abs(shape_metrics.get('skewness', 0)) > 1:
                    hints.append('skewness_correction')
                hints.append('standardization')
                
                metadata['processing_hints'][col_name] = hints
                
            elif dist_info.get('type') == 'categorical':
                # Categorical distribution patterns
                dist_metrics = dist_info.get('distribution_metrics', {})
                metadata['distribution_patterns'][col_name] = {
                    'type': 'categorical',
                    'entropy': dist_metrics.get('entropy', 0),
                    'uniformity': dist_metrics.get('uniformity_score', 0),
                    'concentration_ratio': dist_metrics.get('concentration_ratio', 0)
                }
                
                # Processing hints for categorical columns
                hints = []
                if dist_metrics.get('entropy', 0) > 3:  # High entropy
                    hints.append('dimensionality_reduction')
                if dist_metrics.get('concentration_ratio', 0) > 0.8:  # Highly concentrated
                    hints.append('rare_category_handling')
                hints.extend(['one_hot_encoding', 'label_encoding'])
                
                metadata['processing_hints'][col_name] = hints
        
        # Generate recommended next steps based on patterns
        numeric_cols = [col for col, info in metadata['distribution_patterns'].items() if info.get('type') == 'numeric']
        categorical_cols = [col for col, info in metadata['distribution_patterns'].items() if info.get('type') == 'categorical']
        
        if numeric_cols:
            metadata['recommended_next_steps'].extend(['feature_scaling', 'correlation_analysis'])
            if any(info.get('outliers_count', 0) > 0 for info in metadata['outlier_information'].values()):
                metadata['recommended_next_steps'].append('outlier_treatment')
            if any(not info.get('is_normal') for info in metadata['normality_tests'].values()):
                metadata['recommended_next_steps'].append('normality_transformation')
                
        if categorical_cols:
            metadata['recommended_next_steps'].extend(['categorical_encoding', 'feature_engineering'])
            
        if numeric_cols and categorical_cols:
            metadata['recommended_next_steps'].append('mixed_type_preprocessing')
            
        return metadata
        
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze distributions for all columns in DataFrame.
        
        This method replicates the existing analyze_distributions logic
        to maintain 100% compatibility with the original implementation.
        """
        results = {
            'summary': {
                'total_columns': len(df.columns),
                'numeric_columns': 0,
                'categorical_columns': 0,
                'analyzed_columns': 0
            },
            'distributions': {}
        }
        
        for column in df.columns:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                results['distributions'][column] = {
                    'type': 'empty',
                    'analysis': 'No data available for analysis'
                }
                continue
                
            results['summary']['analyzed_columns'] += 1
            
            # Analyze based on data type
            if pd.api.types.is_numeric_dtype(col_data):
                results['summary']['numeric_columns'] += 1
                results['distributions'][column] = self._analyze_numeric_distribution(col_data)
            else:
                results['summary']['categorical_columns'] += 1
                results['distributions'][column] = self._analyze_categorical_distribution(col_data)
        
        return results
        
    def _analyze_numeric_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column distribution."""
        analysis = {
            'type': 'numeric',
            'summary_stats': {
                'count': len(series),
                'mean': float(series.mean()),
                'std': float(series.std()) if len(series) > 1 else 0,
                'min': float(series.min()),
                'max': float(series.max())
            }
        }
        
        # Percentiles
        try:
            percentile_values = series.quantile([p/100 for p in self.percentiles])
            analysis['percentiles'] = {
                f'p{p}': float(percentile_values[p/100]) for p in self.percentiles
            }
        except Exception as e:
            logger.warning(f"Could not calculate percentiles: {e}")
            analysis['percentiles'] = {}
        
        # Histogram
        try:
            hist, bin_edges = np.histogram(series.values, bins=self.bins)
            analysis['histogram'] = {
                'bins': self.bins,
                'counts': [int(x) for x in hist],
                'bin_edges': [float(x) for x in bin_edges],
                'bin_centers': [float((bin_edges[i] + bin_edges[i+1]) / 2) for i in range(len(bin_edges)-1)]
            }
        except Exception as e:
            logger.warning(f"Could not generate histogram: {e}")
            analysis['histogram'] = None
        
        # Distribution shape analysis
        try:
            analysis['shape_metrics'] = {
                'skewness': float(series.skew()) if len(series) > 2 else 0,
                'kurtosis': float(series.kurtosis()) if len(series) > 3 else 0,
                'is_normal_distributed': self._test_normality(series),
                'outliers_count': self._count_outliers(series)
            }
        except Exception as e:
            logger.warning(f"Could not analyze distribution shape: {e}")
            analysis['shape_metrics'] = {}
        
        return analysis
        
    def _analyze_categorical_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column distribution."""
        value_counts = series.value_counts()
        
        analysis = {
            'type': 'categorical',
            'summary_stats': {
                'count': len(series),
                'unique_values': len(value_counts),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            }
        }
        
        # Value distribution
        analysis['value_distribution'] = {
            str(val): int(count) for val, count in value_counts.head(20).items()
        }
        
        # Distribution metrics
        analysis['distribution_metrics'] = {
            'entropy': self._calculate_entropy(value_counts),
            'concentration_ratio': float(value_counts.iloc[0] / len(series)) if len(value_counts) > 0 else 0,
            'uniformity_score': self._calculate_uniformity(value_counts)
        }
        
        return analysis
        
    def _test_normality(self, series: pd.Series) -> bool:
        """Simple normality test based on skewness and kurtosis."""
        try:
            skew = abs(series.skew())
            kurt = abs(series.kurtosis())
            # Simple heuristic: normal if skewness < 1 and kurtosis < 3
            return skew < 1.0 and kurt < 3.0
        except:
            return False
            
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        try:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return len(outliers)
        except:
            return 0
            
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution."""
        try:
            probabilities = value_counts / value_counts.sum()
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            return float(entropy)
        except:
            return 0.0
            
    def _calculate_uniformity(self, value_counts: pd.Series) -> float:
        """Calculate uniformity score (0 = highly skewed, 1 = perfectly uniform)."""
        try:
            if len(value_counts) <= 1:
                return 1.0
            expected_count = value_counts.sum() / len(value_counts)
            deviations = [(count - expected_count) ** 2 for count in value_counts]
            mse = sum(deviations) / len(value_counts)
            # Normalize to 0-1 scale
            max_possible_mse = (value_counts.sum() ** 2) / len(value_counts)
            uniformity = 1 - (mse / max_possible_mse) if max_possible_mse > 0 else 1
            return float(max(0, min(1, uniformity)))
        except:
            return 0.0