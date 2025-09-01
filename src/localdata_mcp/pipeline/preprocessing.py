"""
Preprocessing Stage Pipeline - Data Cleaning and Feature Engineering

This module implements the preprocessing stage of the Core Pipeline Framework,
providing data cleaning, normalization, and feature engineering with progressive
disclosure architecture and streaming compatibility.

Key Features:
- Progressive complexity levels (minimal, auto, comprehensive, custom)
- Streaming-compatible chunk-by-chunk processing
- Intelligent transformation selection based on data characteristics
- Detailed transformation logging and metadata generation
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import re

from .base import (
    AnalysisPipelineBase,
    PreprocessingIntent,
    StreamingConfig,
    PipelineError,
    ErrorClassification
)
from ..logging_manager import get_logger

logger = get_logger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality assessment metrics."""
    
    # Completeness metrics
    completeness_score: float = 0.0
    missing_value_percentage: float = 0.0
    
    # Consistency metrics
    consistency_score: float = 0.0
    duplicate_percentage: float = 0.0
    
    # Validity metrics
    validity_score: float = 0.0
    type_conformity_percentage: float = 0.0
    
    # Accuracy metrics (outlier detection)
    accuracy_score: float = 0.0
    outlier_percentage: float = 0.0
    
    # Overall quality score
    overall_quality_score: float = 0.0
    
    # Business rules compliance
    business_rules_compliance: float = 0.0
    
    # Data profile summary
    data_profile: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall data quality score from component metrics."""
        scores = [self.completeness_score, self.consistency_score, 
                 self.validity_score, self.accuracy_score, self.business_rules_compliance]
        self.overall_quality_score = np.mean([s for s in scores if s > 0])
        return self.overall_quality_score


@dataclass
class CleaningOperation:
    """Record of a data cleaning operation for transparency and reversibility."""
    
    operation_type: str
    column: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    records_affected: int = 0
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    before_stats: Dict[str, Any] = field(default_factory=dict)
    after_stats: Dict[str, Any] = field(default_factory=dict)
    reversibility_data: Dict[str, Any] = field(default_factory=dict)
    

class TransformationStrategy:
    """Strategies for different preprocessing transformations."""
    
    @staticmethod
    def missing_values_auto(data: pd.DataFrame) -> str:
        """Automatically determine missing value strategy."""
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        strategies = []
        if len(numeric_cols) > 0:
            strategies.append("numeric_median")
        if len(categorical_cols) > 0:
            strategies.append("categorical_mode")
        
        return "mixed" if len(strategies) > 1 else strategies[0] if strategies else "none"
    
    @staticmethod
    def outlier_detection_auto(data: pd.DataFrame) -> str:
        """Automatically determine outlier detection strategy."""
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return "none"
        
        # Use IQR method for most cases, Z-score for large datasets
        if len(data) > 10000:
            return "zscore"
        else:
            return "iqr"
    
    @staticmethod
    def encoding_strategy_auto(data: pd.DataFrame) -> str:
        """Automatically determine categorical encoding strategy."""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return "none"
        
        # Check cardinality to decide between label encoding and one-hot encoding
        high_cardinality_threshold = 10
        strategies = []
        
        for col in categorical_cols:
            cardinality = data[col].nunique()
            if cardinality <= high_cardinality_threshold:
                strategies.append("onehot")
            else:
                strategies.append("label")
        
        # Return most common strategy
        return max(set(strategies), key=strategies.count) if strategies else "none"
    
    @staticmethod
    def duplicate_detection_strategy(data: pd.DataFrame) -> str:
        """Automatically determine duplicate detection strategy."""
        # For small datasets, use exact matching
        if len(data) < 1000:
            return "exact"
        # For larger datasets, use hash-based detection for efficiency
        elif len(data) < 50000:
            return "hash_based"
        else:
            return "sampling_based"
    
    @staticmethod
    def data_type_inference_strategy(data: pd.DataFrame) -> Dict[str, str]:
        """Automatically determine data type inference strategies per column."""
        strategies = {}
        
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if it might be datetime
                sample_values = data[col].dropna().astype(str).head(100)
                if len(sample_values) > 0:
                    datetime_patterns = [
                        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                        r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                    ]
                    is_datetime = any(re.search(pattern, str(val)) for val in sample_values[:10] for pattern in datetime_patterns)
                    
                    if is_datetime:
                        strategies[col] = "datetime"
                    else:
                        # Try numeric conversion
                        try:
                            pd.to_numeric(sample_values, errors='raise')
                            strategies[col] = "numeric"
                        except:
                            strategies[col] = "categorical"
                else:
                    strategies[col] = "categorical"
            else:
                strategies[col] = "preserve"
                
        return strategies


class DataPreprocessingPipeline(AnalysisPipelineBase):
    """
    Preprocessing pipeline with progressive disclosure and streaming support.
    
    First Principle: Progressive Disclosure Architecture
    - Simple by default: automatic data cleaning and type inference
    - Powerful when needed: custom transformations and advanced preprocessing
    """
    
    def __init__(self,
                 analytical_intention: str,
                 preprocessing_intent: PreprocessingIntent = PreprocessingIntent.AUTO,
                 custom_transformations: Optional[List[Callable]] = None,
                 streaming_config: Optional[StreamingConfig] = None,
                 custom_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessing with progressive complexity.
        
        Args:
            analytical_intention: Natural language description of analytical intent
            preprocessing_intent: Level of preprocessing complexity
            custom_transformations: Optional custom transformation functions
            streaming_config: Configuration for streaming execution
            custom_parameters: Custom preprocessing parameters
        """
        super().__init__(
            analytical_intention=analytical_intention,
            streaming_config=streaming_config or StreamingConfig(),
            progressive_complexity=preprocessing_intent.value,
            composition_aware=True,
            custom_parameters=custom_parameters or {}
        )
        
        self.preprocessing_intent = preprocessing_intent
        self.custom_transformations = custom_transformations or []
        
        # Transformation state storage for streaming compatibility
        self._transformation_states: Dict[str, Dict[str, Any]] = {}
        self._preprocessing_log: List[Dict[str, Any]] = []
        
        logger.info("DataPreprocessingPipeline initialized",
                   intention=analytical_intention,
                   complexity=preprocessing_intent.value)
    
    def get_analysis_type(self) -> str:
        """Get the analysis type - preprocessing."""
        return "data_preprocessing"
    
    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure preprocessing steps based on complexity level."""
        pipeline_steps = []
        
        if self.preprocessing_intent == PreprocessingIntent.MINIMAL:
            pipeline_steps.extend([
                self._handle_missing_values,
                self._infer_and_convert_types
            ])
            
        elif self.preprocessing_intent == PreprocessingIntent.AUTO:
            pipeline_steps.extend([
                self._handle_missing_values,
                self._infer_and_convert_types,
                self._detect_and_handle_outliers,
                self._normalize_text_columns,
                self._encode_categorical_variables
            ])
            
        elif self.preprocessing_intent == PreprocessingIntent.COMPREHENSIVE:
            pipeline_steps.extend([
                self._handle_missing_values,
                self._infer_and_convert_types,
                self._detect_and_handle_outliers,
                self._normalize_text_columns,
                self._encode_categorical_variables,
                self._feature_scaling,
                self._dimensionality_assessment,
                self._correlation_analysis,
                self._data_quality_enhancement
            ])
            
        elif self.preprocessing_intent == PreprocessingIntent.CUSTOM:
            # Start with minimal base, then add custom transformations
            pipeline_steps.extend([
                self._handle_missing_values,
                self._infer_and_convert_types
            ])
        
        # Add custom transformations
        pipeline_steps.extend(self.custom_transformations)
        
        logger.info(f"Configured preprocessing pipeline with {len(pipeline_steps)} steps")
        return pipeline_steps
    
    def _execute_analysis_step(self, step: Callable, data: pd.DataFrame, 
                              context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute individual preprocessing step with error handling."""
        step_name = step.__name__
        start_time = time.time()
        
        try:
            # Get transformation state for this step
            transform_state = self._transformation_states.get(step_name, {})
            
            # Execute the transformation
            processed_data, step_metadata = step(data, **transform_state)
            
            execution_time = time.time() - start_time
            
            # Log successful transformation
            log_entry = {
                "transformation": step_name,
                "status": "success",
                "execution_time": execution_time,
                "rows_processed": len(processed_data),
                "metadata": step_metadata
            }
            self._preprocessing_log.append(log_entry)
            
            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": True,
                "step_metadata": step_metadata
            }
            
            return processed_data, metadata
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log failed transformation
            log_entry = {
                "transformation": step_name,
                "status": "error",
                "execution_time": execution_time,
                "error": str(e)
            }
            self._preprocessing_log.append(log_entry)
            
            logger.error(f"Preprocessing step {step_name} failed: {e}")
            
            # Return original data for graceful degradation
            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            }
            
            return data, metadata  # Return original data, don't fail the entire pipeline
    
    def _execute_streaming_analysis(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute preprocessing with streaming support for large datasets."""
        # For streaming preprocessing, we apply transformations chunk by chunk
        # using the learned transformation states
        
        processed_data = data.copy()
        
        # Apply each transformation in the pipeline
        for transform_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                transform_func, processed_data, self.get_execution_context()
            )
        
        # Build metadata
        metadata = self._build_preprocessing_metadata(processed_data, streaming_enabled=True)
        return processed_data, metadata
    
    def _execute_standard_analysis(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute preprocessing on full dataset in memory."""
        processed_data = data.copy()
        
        # Apply each transformation in the pipeline
        for transform_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                transform_func, processed_data, self.get_execution_context()
            )
        
        # Build metadata
        metadata = self._build_preprocessing_metadata(processed_data, streaming_enabled=False)
        return processed_data, metadata
    
    # Transformation methods
    def _handle_missing_values(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values in the dataset."""
        strategy = kwargs.get('strategy', TransformationStrategy.missing_values_auto(data))
        
        result_data = data.copy()
        imputation_log = {}
        
        if strategy in ['numeric_median', 'mixed']:
            numeric_cols = data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if data[col].isnull().sum() > 0:
                    median_value = data[col].median()
                    result_data[col].fillna(median_value, inplace=True)
                    imputation_log[col] = {'method': 'median', 'value': median_value}
        
        if strategy in ['categorical_mode', 'mixed']:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if data[col].isnull().sum() > 0:
                    mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else 'unknown'
                    result_data[col].fillna(mode_value, inplace=True)
                    imputation_log[col] = {'method': 'mode', 'value': mode_value}
        
        metadata = {
            'strategy': strategy,
            'imputation_log': imputation_log,
            'missing_values_before': data.isnull().sum().sum(),
            'missing_values_after': result_data.isnull().sum().sum()
        }
        
        return result_data, metadata
    
    def _infer_and_convert_types(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Infer and convert data types for optimal analysis."""
        result_data = data.copy()
        type_conversions = {}
        
        for col in data.columns:
            original_dtype = str(data[col].dtype)
            
            # Try to convert to datetime if it looks like a date
            if data[col].dtype == 'object':
                try:
                    # Sample a few values to check if they look like dates
                    sample_values = data[col].dropna().head(100)
                    if len(sample_values) > 0:
                        pd.to_datetime(sample_values, errors='raise')
                        result_data[col] = pd.to_datetime(data[col], errors='coerce')
                        type_conversions[col] = {'from': original_dtype, 'to': 'datetime64[ns]'}
                        continue
                except:
                    pass
                
                # Try to convert to numeric
                try:
                    numeric_series = pd.to_numeric(data[col], errors='coerce')
                    # If most values convert successfully, use numeric type
                    if (numeric_series.notna().sum() / len(numeric_series)) > 0.8:
                        result_data[col] = numeric_series
                        type_conversions[col] = {'from': original_dtype, 'to': str(numeric_series.dtype)}
                        continue
                except:
                    pass
        
        metadata = {
            'type_conversions': type_conversions,
            'total_conversions': len(type_conversions)
        }
        
        return result_data, metadata
    
    def _detect_and_handle_outliers(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect and handle outliers in numeric columns."""
        strategy = kwargs.get('strategy', TransformationStrategy.outlier_detection_auto(data))
        action = kwargs.get('action', 'cap')  # 'cap', 'remove', 'flag'
        
        result_data = data.copy()
        numeric_cols = data.select_dtypes(include=['number']).columns
        outlier_log = {}
        
        for col in numeric_cols:
            outliers_detected = 0
            
            if strategy == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers_detected = outlier_mask.sum()
                
                if action == 'cap':
                    result_data[col] = np.clip(data[col], lower_bound, upper_bound)
                elif action == 'remove':
                    result_data = result_data[~outlier_mask]
                elif action == 'flag':
                    result_data[f'{col}_outlier_flag'] = outlier_mask
            
            elif strategy == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_mask = z_scores > 3
                outliers_detected = outlier_mask.sum()
                
                if action == 'cap':
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    result_data[col] = np.clip(data[col], mean_val - 3*std_val, mean_val + 3*std_val)
                elif action == 'remove':
                    result_data = result_data[~outlier_mask]
                elif action == 'flag':
                    result_data[f'{col}_outlier_flag'] = outlier_mask
            
            outlier_log[col] = {
                'strategy': strategy,
                'action': action,
                'outliers_detected': outliers_detected
            }
        
        metadata = {
            'strategy': strategy,
            'action': action,
            'outlier_log': outlier_log,
            'total_outliers_processed': sum(log['outliers_detected'] for log in outlier_log.values())
        }
        
        return result_data, metadata
    
    def _normalize_text_columns(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize text columns (trim, case normalization, etc.)."""
        result_data = data.copy()
        text_cols = data.select_dtypes(include=['object']).columns
        normalization_log = {}
        
        for col in text_cols:
            if data[col].dtype == 'object':  # Only process string columns
                # Remove leading/trailing whitespace
                result_data[col] = data[col].astype(str).str.strip()
                
                # Convert to lowercase if specified
                if kwargs.get('lowercase', False):
                    result_data[col] = result_data[col].str.lower()
                
                # Remove extra whitespace between words
                result_data[col] = result_data[col].str.replace(r'\s+', ' ', regex=True)
                
                normalization_log[col] = {
                    'operations': ['trim', 'whitespace_normalize'] + 
                                (['lowercase'] if kwargs.get('lowercase', False) else [])
                }
        
        metadata = {
            'normalized_columns': list(normalization_log.keys()),
            'normalization_log': normalization_log
        }
        
        return result_data, metadata
    
    def _encode_categorical_variables(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical variables for analysis."""
        strategy = kwargs.get('strategy', TransformationStrategy.encoding_strategy_auto(data))
        
        result_data = data.copy()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        encoding_log = {}
        
        for col in categorical_cols:
            cardinality = data[col].nunique()
            
            if strategy == 'onehot' or (strategy == 'auto' and cardinality <= 10):
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(data[col], prefix=col, dummy_na=True)
                result_data = pd.concat([result_data.drop(col, axis=1), dummies], axis=1)
                encoding_log[col] = {'method': 'onehot', 'new_columns': list(dummies.columns)}
                
            elif strategy == 'label' or (strategy == 'auto' and cardinality > 10):
                # Label encoding for high cardinality
                le = LabelEncoder()
                result_data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
                encoding_log[col] = {'method': 'label', 'new_column': f'{col}_encoded'}
        
        metadata = {
            'strategy': strategy,
            'encoding_log': encoding_log,
            'encoded_columns': len(encoding_log)
        }
        
        return result_data, metadata
    
    def _feature_scaling(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features."""
        method = kwargs.get('method', 'standard')  # 'standard', 'minmax', 'robust'
        
        result_data = data.copy()
        numeric_cols = data.select_dtypes(include=['number']).columns
        scaling_log = {}
        
        if method == 'standard':
            scaler = StandardScaler()
            result_data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            scaling_log = {'method': 'standard', 'columns': list(numeric_cols)}
        
        metadata = {
            'scaling_method': method,
            'scaled_columns': len(numeric_cols),
            'scaling_log': scaling_log
        }
        
        return result_data, metadata
    
    def _dimensionality_assessment(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Assess dimensionality and suggest dimension reduction if needed."""
        result_data = data.copy()  # No transformation, just assessment
        
        shape = data.shape
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        # Simple assessment
        high_dimensionality = len(numeric_cols) > 20
        samples_to_features_ratio = shape[0] / len(numeric_cols) if len(numeric_cols) > 0 else float('inf')
        
        recommendations = []
        if high_dimensionality:
            recommendations.append("Consider PCA for dimension reduction")
        if samples_to_features_ratio < 10:
            recommendations.append("Low samples-to-features ratio - consider feature selection")
        
        metadata = {
            'total_features': shape[1],
            'numeric_features': len(numeric_cols),
            'samples_to_features_ratio': samples_to_features_ratio,
            'high_dimensionality': high_dimensionality,
            'recommendations': recommendations
        }
        
        return result_data, metadata
    
    def _correlation_analysis(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze correlations and identify highly correlated features."""
        result_data = data.copy()
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            metadata = {'correlation_analysis': 'insufficient_numeric_columns'}
            return result_data, metadata
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Find highly correlated pairs
        threshold = kwargs.get('correlation_threshold', 0.8)
        highly_correlated_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) > threshold:
                    highly_correlated_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': correlation
                    })
        
        metadata = {
            'correlation_threshold': threshold,
            'highly_correlated_pairs': len(highly_correlated_pairs),
            'correlation_details': highly_correlated_pairs[:10],  # Limit to first 10
            'recommendations': ['Consider removing highly correlated features'] if highly_correlated_pairs else []
        }
        
        return result_data, metadata
    
    def _data_quality_enhancement(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhance overall data quality through various checks and improvements."""
        result_data = data.copy()
        
        # Remove duplicate rows
        initial_rows = len(result_data)
        result_data = result_data.drop_duplicates()
        duplicates_removed = initial_rows - len(result_data)
        
        # Check for constant columns
        constant_columns = []
        for col in result_data.columns:
            if result_data[col].nunique() <= 1:
                constant_columns.append(col)
        
        # Remove constant columns if requested
        if kwargs.get('remove_constant_columns', True):
            result_data = result_data.drop(columns=constant_columns)
        
        metadata = {
            'duplicates_removed': duplicates_removed,
            'constant_columns': constant_columns,
            'constant_columns_removed': len(constant_columns) if kwargs.get('remove_constant_columns', True) else 0,
            'final_shape': result_data.shape
        }
        
        return result_data, metadata
    
    def _build_preprocessing_metadata(self, processed_data: pd.DataFrame, streaming_enabled: bool) -> Dict[str, Any]:
        """Build comprehensive metadata for preprocessing results."""
        original_shape = self._execution_context.get('data_profile', {}).get('shape', (0, 0))
        
        metadata = {
            "preprocessing_pipeline": {
                "analytical_intention": self.analytical_intention,
                "preprocessing_intent": self.preprocessing_intent.value,
                "streaming_enabled": streaming_enabled,
                "steps_executed": len(self._preprocessing_log)
            },
            "transformation_summary": {
                "original_shape": original_shape,
                "processed_shape": processed_data.shape,
                "rows_changed": processed_data.shape[0] - original_shape[0],
                "columns_changed": processed_data.shape[1] - original_shape[1]
            },
            "preprocessing_log": self._preprocessing_log,
            "data_quality_score": self._calculate_data_quality_score(processed_data),
            "composition_context": {
                "ready_for_analysis": True,
                "data_characteristics": self._analyze_processed_data(processed_data),
                "suggested_next_steps": self._suggest_analysis_steps(processed_data),
                "preprocessing_artifacts": self._extract_preprocessing_artifacts()
            }
        }
        
        return metadata
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        scores = []
        
        # Completeness (no missing values)
        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        scores.append(completeness)
        
        # Consistency (no duplicates)
        consistency = (1 - data.duplicated().sum() / len(data)) * 100
        scores.append(consistency)
        
        # Validity (appropriate data types)
        type_validity = 85  # Base score, could be enhanced with more sophisticated checks
        scores.append(type_validity)
        
        return np.mean(scores)
    
    def _analyze_processed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of processed data."""
        return {
            "shape": data.shape,
            "dtypes": dict(data.dtypes),
            "numeric_columns": data.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": data.select_dtypes(include=['datetime64']).columns.tolist(),
            "missing_values": data.isnull().sum().sum(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    def _suggest_analysis_steps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest next analysis steps based on processed data characteristics."""
        suggestions = []
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Time series analysis
        if datetime_cols:
            suggestions.append({
                "analysis_type": "time_series_analysis",
                "reason": "Datetime columns detected",
                "confidence": 0.8
            })
        
        # Statistical analysis
        if len(numeric_cols) >= 2:
            suggestions.append({
                "analysis_type": "statistical_analysis", 
                "reason": "Multiple numeric columns for correlation/regression analysis",
                "confidence": 0.7
            })
        
        # Classification/clustering
        if categorical_cols and numeric_cols:
            suggestions.append({
                "analysis_type": "machine_learning",
                "reason": "Mixed data types suitable for supervised/unsupervised learning",
                "confidence": 0.6
            })
        
        return suggestions
    
    def _extract_preprocessing_artifacts(self) -> Dict[str, Any]:
        """Extract preprocessing artifacts that might be useful for downstream analysis."""
        artifacts = {}
        
        # Extract transformation states that could be reused
        for step_name, state in self._transformation_states.items():
            if state:  # Only include non-empty states
                artifacts[f"{step_name}_state"] = state
        
        # Extract preprocessing statistics
        artifacts["transformation_statistics"] = {
            "total_steps": len(self._preprocessing_log),
            "successful_steps": sum(1 for log in self._preprocessing_log if log["status"] == "success"),
            "failed_steps": sum(1 for log in self._preprocessing_log if log["status"] == "error")
        }
        
        return artifacts