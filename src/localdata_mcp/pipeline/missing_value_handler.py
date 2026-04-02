"""
Missing Value Handler Implementation - Advanced sklearn.impute Integration

This module implements sophisticated missing value handling with multiple imputation
strategies, automatic strategy selection, cross-validation assessment, and comprehensive
quality metrics.
"""

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import tracemalloc

import pandas as pd
import numpy as np
# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import psutil
import os

from .base import AnalysisPipelineBase, StreamingConfig
from ..logging_manager import get_logger

logger = get_logger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


@dataclass
class MissingValuePattern:
    """Analysis of missing value patterns in the dataset."""
    
    pattern_type: str  # "MCAR", "MAR", "MNAR"
    missing_percentage: float
    column_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ImputationQuality:
    """Quality metrics for imputation methods."""
    
    strategy_name: str
    accuracy_score: float
    mse: float
    mae: float
    distribution_preservation: float
    correlation_preservation: float
    confidence_interval: Tuple[float, float]
    execution_time: float
    memory_usage: float
    

@dataclass
class ImputationMetadata:
    """Comprehensive metadata for imputation operations."""
    
    selected_strategy: str
    strategy_confidence: float
    missing_pattern: MissingValuePattern
    quality_assessment: Dict[str, ImputationQuality]
    cross_validation_results: Dict[str, Any] = field(default_factory=dict)
    imputation_log: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    reversibility_data: Dict[str, Any] = field(default_factory=dict)


class MissingValueHandler(AnalysisPipelineBase):
    """
    Sophisticated missing value handling with sklearn.impute integration.
    
    This handler provides:
    - Multiple imputation strategies with automatic selection
    - Missing value pattern analysis (MCAR, MAR, MNAR)
    - Cross-validation assessment of imputation quality
    - Progressive disclosure from simple to expert-level control
    - Full transparency and reversibility
    """
    
    def __init__(self,
                 analytical_intention: str = "handle missing values intelligently",
                 strategy: str = "auto",  # "auto", "simple", "knn", "iterative", "custom"
                 complexity: str = "auto",  # "minimal", "auto", "comprehensive", "custom"
                 cross_validation: bool = True,
                 metadata_tracking: bool = True,
                 streaming_config: Optional[StreamingConfig] = None,
                 custom_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize missing value handler with intelligent strategy selection.
        
        Args:
            analytical_intention: Natural language description of imputation goal
            strategy: Imputation strategy selection approach
            complexity: Level of analysis complexity
            cross_validation: Enable quality assessment via cross-validation
            metadata_tracking: Enable detailed imputation metadata tracking
            streaming_config: Configuration for streaming execution
            custom_parameters: Additional custom parameters
        """
        super().__init__(
            analytical_intention=analytical_intention,
            streaming_config=streaming_config or StreamingConfig(),
            progressive_complexity=complexity,
            composition_aware=True,
            custom_parameters=custom_parameters or {}
        )
        
        self.strategy = strategy
        self.complexity = complexity
        self.cross_validation = cross_validation
        self.metadata_tracking = metadata_tracking
        
        # Strategy configurations
        self.strategy_configs = {
            'simple': {'numeric': 'median', 'categorical': 'most_frequent'},
            'knn': {'n_neighbors': 5, 'weights': 'uniform'},
            'iterative': {'estimator': RandomForestRegressor(n_estimators=10, random_state=42),
                         'random_state': 42, 'max_iter': 10},
            'custom': self.custom_parameters.get('strategy_config', {})
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_accuracy': 0.7,
            'max_mse_increase': 0.2,
            'min_correlation_preservation': 0.8,
            'max_distribution_deviation': 0.1
        }
        
        # Imputation tracking
        self._missing_pattern: Optional[MissingValuePattern] = None
        self._imputation_metadata: Optional[ImputationMetadata] = None
        self._fitted_imputers: Dict[str, Any] = {}
        self._original_data: Optional[pd.DataFrame] = None
        
        logger.info("MissingValueHandler initialized",
                   strategy=strategy,
                   complexity=complexity)
    
    def get_analysis_type(self) -> str:
        """Get the analysis type - missing value handling."""
        return "missing_value_handling"
    
    def _configure_analysis_pipeline(self) -> List[Callable]:
        """Configure missing value handling pipeline based on complexity level."""
        pipeline_steps = []
        
        # Always start with pattern analysis
        pipeline_steps.append(self._analyze_missing_patterns)
        
        if self.complexity == "minimal":
            pipeline_steps.extend([
                self._simple_imputation
            ])
            
        elif self.complexity == "auto":
            pipeline_steps.extend([
                self._intelligent_strategy_selection,
                self._apply_selected_strategy,
                self._assess_imputation_quality
            ])
            
        elif self.complexity == "comprehensive":
            pipeline_steps.extend([
                self._evaluate_all_strategies,
                self._cross_validate_strategies,
                self._ensemble_imputation,
                self._comprehensive_quality_assessment
            ])
            
        elif self.complexity == "custom":
            # Load custom pipeline from parameters
            custom_steps = self.custom_parameters.get('pipeline_steps', [])
            pipeline_steps.extend(custom_steps)
        
        # Always end with metadata compilation
        if self.metadata_tracking:
            pipeline_steps.append(self._compile_imputation_metadata)
        
        logger.info(f"Configured imputation pipeline with {len(pipeline_steps)} steps")
        return pipeline_steps
    
    def _execute_analysis_step(self, step: Callable, data: pd.DataFrame,
                              context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute individual imputation step with comprehensive logging."""
        step_name = step.__name__
        start_time = time.time()
        
        # Store original data for quality assessment
        if self._original_data is None:
            self._original_data = data.copy()
        
        try:
            # Execute the imputation step
            processed_data, step_metadata = step(data)
            
            execution_time = time.time() - start_time
            
            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": True,
                "step_metadata": step_metadata,
                "missing_values_before": data.isnull().sum().sum(),
                "missing_values_after": processed_data.isnull().sum().sum()
            }
            
            logger.info(f"Imputation step {step_name} completed successfully",
                       execution_time=execution_time,
                       missing_reduced=metadata["missing_values_before"] - metadata["missing_values_after"])
            
            return processed_data, metadata
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(f"Imputation step {step_name} failed: {e}")
            
            metadata = {
                "step": step_name,
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            }
            
            return data, metadata  # Return original data
    
    def _execute_streaming_analysis(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute imputation with streaming support for large datasets."""
        processed_data = data.copy()
        
        # Apply each imputation step in the pipeline
        for impute_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                impute_func, processed_data, self.get_execution_context()
            )
        
        # Build comprehensive metadata
        metadata = self._build_imputation_metadata(processed_data, streaming_enabled=True)
        return processed_data, metadata
    
    def _execute_standard_analysis(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute imputation on full dataset in memory."""
        processed_data = data.copy()
        
        # Apply each imputation step in the pipeline
        for impute_func in self._analysis_pipeline:
            processed_data, step_metadata = self._execute_analysis_step(
                impute_func, processed_data, self.get_execution_context()
            )
        
        # Build comprehensive metadata
        metadata = self._build_imputation_metadata(processed_data, streaming_enabled=False)
        return processed_data, metadata
    
    # ===========================================
    # MISSING VALUE PATTERN ANALYSIS
    # ===========================================
    
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze missing value patterns to guide imputation strategy."""
        missing_info = {}
        
        # Calculate missing percentages per column
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        # Analyze missing patterns across columns
        missing_pattern_matrix = data.isnull()
        pattern_correlations = missing_pattern_matrix.corr()
        
        # Classify missing pattern type
        pattern_type = self._classify_missing_pattern(data, pattern_correlations)
        
        # Column-specific patterns
        column_patterns = {}
        for col in data.columns:
            if missing_counts[col] > 0:
                column_patterns[col] = {
                    'missing_count': missing_counts[col],
                    'missing_percentage': missing_percentages[col],
                    'data_type': str(data[col].dtype),
                    'unique_values': data[col].nunique(),
                    'pattern_with_others': self._analyze_column_pattern(data, col)
                }
        
        # Temporal patterns (if datetime columns exist)
        temporal_patterns = self._analyze_temporal_patterns(data)
        
        # Generate recommendations based on patterns
        recommendations = self._generate_pattern_recommendations(
            pattern_type, missing_percentages, column_patterns
        )
        
        # Store pattern analysis
        self._missing_pattern = MissingValuePattern(
            pattern_type=pattern_type,
            missing_percentage=missing_percentages.mean(),
            column_patterns=column_patterns,
            correlation_matrix=pattern_correlations,
            temporal_patterns=temporal_patterns,
            confidence_score=self._calculate_pattern_confidence(pattern_correlations),
            recommendations=recommendations
        )
        
        metadata = {
            "pattern_analysis": {
                "pattern_type": pattern_type,
                "overall_missing_percentage": missing_percentages.mean(),
                "columns_with_missing": len(column_patterns),
                "recommendations": recommendations,
                "confidence_score": self._missing_pattern.confidence_score
            }
        }
        
        return data.copy(), metadata
    
    def _classify_missing_pattern(self, data: pd.DataFrame, 
                                 pattern_correlations: pd.DataFrame) -> str:
        """Classify the type of missing data pattern."""
        # Analyze correlation strengths
        strong_correlations = (pattern_correlations.abs() > 0.5).sum().sum() - len(pattern_correlations)
        moderate_correlations = ((pattern_correlations.abs() > 0.3) & (pattern_correlations.abs() <= 0.5)).sum().sum()
        
        total_possible = len(pattern_correlations) * (len(pattern_correlations) - 1)
        
        if strong_correlations > total_possible * 0.1:
            return "MNAR"  # Missing Not At Random
        elif moderate_correlations > total_possible * 0.2:
            return "MAR"   # Missing At Random
        else:
            return "MCAR"  # Missing Completely At Random
    
    def _analyze_column_pattern(self, data: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze missing pattern for a specific column."""
        missing_mask = data[column].isnull()
        
        pattern_info = {
            'correlates_with': [],
            'distribution_bias': None,
            'temporal_pattern': None
        }
        
        # Check correlation with other columns' missing patterns
        for other_col in data.columns:
            if other_col != column and data[other_col].isnull().sum() > 0:
                correlation = missing_mask.corr(data[other_col].isnull())
                if abs(correlation) > 0.3:
                    pattern_info['correlates_with'].append({
                        'column': other_col,
                        'correlation': correlation
                    })
        
        # Check for distribution bias
        numeric_cols = data.select_dtypes(include=['number']).columns
        for num_col in numeric_cols:
            if num_col != column and len(data[~missing_mask]) > 10 and len(data[missing_mask]) > 10:
                present_data = data[~missing_mask][num_col].dropna()
                all_data = data[num_col].dropna()
                
                if len(present_data) > 0 and len(all_data) > 0:
                    try:
                        # Compare distributions
                        stat, p_value = stats.ks_2samp(present_data, all_data)
                        if p_value < 0.05:  # Significant difference
                            pattern_info['distribution_bias'] = {
                                'affected_column': num_col,
                                'p_value': p_value,
                                'mean_difference': present_data.mean() - all_data.mean()
                            }
                    except:
                        pass
        
        return pattern_info
    
    def _analyze_temporal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in missing values."""
        temporal_patterns = {}
        
        # Check for datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        for dt_col in datetime_cols:
            if data[dt_col].isnull().sum() == 0:  # Use as time reference
                for col in data.columns:
                    if col != dt_col and data[col].isnull().sum() > 0:
                        try:
                            # Analyze missing pattern over time
                            missing_by_time = data.groupby(pd.Grouper(key=dt_col, freq='D'))[col].apply(
                                lambda x: x.isnull().sum() / len(x) if len(x) > 0 else 0
                            ).fillna(0)
                            
                            if len(missing_by_time) > 1 and missing_by_time.var() > 0.01:  # Significant temporal variation
                                temporal_patterns[col] = {
                                    'time_reference': dt_col,
                                    'temporal_variance': missing_by_time.var(),
                                    'peak_missing_periods': missing_by_time.nlargest(min(3, len(missing_by_time))).index.tolist()
                                }
                        except:
                            pass
        
        return temporal_patterns
    
    def _generate_pattern_recommendations(self, pattern_type: str, 
                                        missing_percentages: pd.Series,
                                        column_patterns: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate imputation strategy recommendations based on missing patterns."""
        recommendations = []
        
        # Overall missing percentage recommendations
        overall_missing = missing_percentages.mean()
        
        if overall_missing < 5:
            recommendations.append("Low missing data rate - simple imputation methods suitable")
        elif overall_missing < 20:
            recommendations.append("Moderate missing data - consider KNN or iterative imputation")
        else:
            recommendations.append("High missing data rate - advanced methods and careful validation needed")
        
        # Pattern-specific recommendations
        if pattern_type == "MCAR":
            recommendations.append("MCAR pattern detected - any imputation method appropriate")
        elif pattern_type == "MAR":
            recommendations.append("MAR pattern detected - multivariate methods recommended (KNN, Iterative)")
        elif pattern_type == "MNAR":
            recommendations.append("MNAR pattern detected - domain knowledge required, consider explicit missing indicators")
        
        # Column-specific recommendations
        high_missing_cols = [col for col, perc in missing_percentages.items() if perc > 50]
        if high_missing_cols:
            recommendations.append(f"Columns with >50% missing: {high_missing_cols} - consider removal or domain-specific imputation")
        
        return recommendations
    
    def _calculate_pattern_confidence(self, pattern_correlations: pd.DataFrame) -> float:
        """Calculate confidence score for pattern classification."""
        # Base confidence on correlation strength and consistency
        abs_correlations = pattern_correlations.abs().fillna(0)
        
        # Remove diagonal (self-correlations)
        mask = np.eye(len(abs_correlations), dtype=bool)
        off_diagonal = abs_correlations.values[~mask]
        
        if len(off_diagonal) == 0:
            return 0.5
        
        # Higher correlations = higher confidence in pattern classification
        mean_correlation = np.mean(off_diagonal)
        correlation_variance = np.var(off_diagonal)
        
        # Confidence increases with mean correlation but decreases with high variance
        confidence = mean_correlation * (1 - correlation_variance)
        
        return max(0.0, min(1.0, confidence))
    
    # ===========================================
    # IMPUTATION STRATEGY IMPLEMENTATIONS
    # ===========================================
    
    def _simple_imputation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Simple imputation using median/mode strategies."""
        result_data = data.copy()
        imputation_log = {}
        start_time = time.time()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Numeric columns - median imputation
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                imputer = SimpleImputer(strategy='median')
                result_data[col] = imputer.fit_transform(data[[col]]).ravel()
                self._fitted_imputers[f"{col}_simple"] = imputer
                imputation_log[col] = {
                    'strategy': 'simple_median',
                    'missing_before': data[col].isnull().sum(),
                    'missing_after': result_data[col].isnull().sum(),
                    'imputed_value': imputer.statistics_[0]
                }
        
        # Categorical columns - most frequent imputation
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                result_data[col] = imputer.fit_transform(data[[col]]).ravel()
                self._fitted_imputers[f"{col}_simple"] = imputer
                imputation_log[col] = {
                    'strategy': 'simple_mode',
                    'missing_before': data[col].isnull().sum(),
                    'missing_after': result_data[col].isnull().sum(),
                    'imputed_value': imputer.statistics_[0]
                }
        
        # Calculate memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = time.time() - start_time
        
        metadata = {
            'strategy': 'simple',
            'imputation_log': imputation_log,
            'execution_time': execution_time,
            'memory_usage_mb': peak / (1024 * 1024),
            'total_imputed_values': sum(log['missing_before'] - log['missing_after'] for log in imputation_log.values())
        }
        
        return result_data, metadata
    
    def _intelligent_strategy_selection(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Intelligently select optimal imputation strategy based on data characteristics."""
        if self._missing_pattern is None:
            # Run pattern analysis if not already done
            _, _ = self._analyze_missing_patterns(data)
        
        selected_strategy = self._select_optimal_strategy(data, self._missing_pattern)
        
        metadata = {
            'selected_strategy': selected_strategy,
            'selection_rationale': self._get_strategy_rationale(selected_strategy, self._missing_pattern),
            'confidence': self._calculate_strategy_confidence(selected_strategy, self._missing_pattern)
        }
        
        # Store selected strategy for next step
        self.custom_parameters['auto_selected_strategy'] = selected_strategy
        
        return data.copy(), metadata
    
    def _select_optimal_strategy(self, data: pd.DataFrame, pattern: MissingValuePattern) -> str:
        """Select optimal imputation strategy based on missing value pattern analysis."""
        
        # Factor 1: Missing percentage
        overall_missing = pattern.missing_percentage
        
        # Factor 2: Data size and complexity
        n_rows, n_cols = data.shape
        numeric_cols = len(data.select_dtypes(include=['number']).columns)
        
        # Factor 3: Pattern type
        pattern_type = pattern.pattern_type
        
        # Factor 4: Computational constraints
        is_large_dataset = n_rows > 50000 or n_cols > 100
        
        # Decision logic
        if overall_missing < 5 and not is_large_dataset:
            return "simple"
        elif pattern_type == "MCAR" and overall_missing < 15:
            return "knn" if not is_large_dataset else "simple"
        elif pattern_type == "MAR" and numeric_cols > 2 and not is_large_dataset:
            return "iterative"
        elif pattern_type == "MNAR" or overall_missing > 30:
            return "custom"  # Requires domain knowledge
        elif is_large_dataset:
            return "knn"  # Good balance of quality and speed
        else:
            return "iterative"  # Default advanced method
    
    def _get_strategy_rationale(self, strategy: str, pattern: MissingValuePattern) -> Dict[str, Any]:
        """Get rationale for strategy selection."""
        rationale = {
            'strategy': strategy,
            'factors': {
                'missing_percentage': pattern.missing_percentage,
                'pattern_type': pattern.pattern_type,
                'pattern_confidence': pattern.confidence_score,
                'recommendations_followed': []
            }
        }
        
        # Match selected strategy to recommendations
        for rec in pattern.recommendations:
            if strategy.upper() in rec.upper() or strategy in rec.lower():
                rationale['factors']['recommendations_followed'].append(rec)
        
        return rationale
    
    def _calculate_strategy_confidence(self, strategy: str, pattern: MissingValuePattern) -> float:
        """Calculate confidence in strategy selection."""
        base_confidence = pattern.confidence_score
        
        # Adjust based on strategy-pattern alignment
        pattern_type = pattern.pattern_type
        missing_pct = pattern.missing_percentage
        
        # Strategy-specific confidence adjustments
        adjustments = {
            'simple': 0.9 if missing_pct < 5 else 0.6,
            'knn': 0.8 if pattern_type in ['MCAR', 'MAR'] else 0.5,
            'iterative': 0.9 if pattern_type == 'MAR' else 0.7,
            'custom': 0.6  # Always uncertain without domain knowledge
        }
        
        strategy_confidence = adjustments.get(strategy, 0.5)
        
        # Combine pattern confidence with strategy confidence
        combined_confidence = (base_confidence + strategy_confidence) / 2
        
        return max(0.0, min(1.0, combined_confidence))
    
    def _apply_selected_strategy(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply the selected imputation strategy."""
        selected_strategy = self.custom_parameters.get('auto_selected_strategy', 'simple')
        
        if selected_strategy == "simple":
            return self._simple_imputation(data)
        elif selected_strategy == "knn":
            return self._knn_imputation(data)
        elif selected_strategy == "iterative":
            return self._iterative_imputation(data)
        elif selected_strategy == "custom":
            return self._custom_imputation(data)
        else:
            # Fallback to simple
            return self._simple_imputation(data)
    
    def _knn_imputation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """KNN-based imputation for multivariate missing value handling."""
        result_data = data.copy()
        imputation_log = {}
        start_time = time.time()
        
        tracemalloc.start()
        
        # Handle numeric columns with KNN
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0 and any(data[col].isnull().sum() > 0 for col in numeric_cols):
            
            # Determine optimal number of neighbors
            n_neighbors = min(self.strategy_configs['knn']['n_neighbors'], len(data) // 10 + 1)
            n_neighbors = max(1, n_neighbors)  # Ensure at least 1 neighbor
            
            try:
                knn_imputer = KNNImputer(
                    n_neighbors=n_neighbors,
                    weights=self.strategy_configs['knn']['weights']
                )
                
                numeric_data = data[numeric_cols]
                imputed_numeric = knn_imputer.fit_transform(numeric_data)
                
                for i, col in enumerate(numeric_cols):
                    if numeric_data[col].isnull().sum() > 0:
                        result_data[col] = imputed_numeric[:, i]
                        self._fitted_imputers[f"{col}_knn"] = knn_imputer
                        imputation_log[col] = {
                            'strategy': 'knn',
                            'n_neighbors': n_neighbors,
                            'missing_before': data[col].isnull().sum(),
                            'missing_after': result_data[col].isnull().sum()
                        }
                        
            except Exception as e:
                logger.warning(f"KNN imputation failed, falling back to median: {e}")
                # Fallback to simple imputation
                for col in numeric_cols:
                    if data[col].isnull().sum() > 0:
                        result_data[col].fillna(data[col].median(), inplace=True)
                        imputation_log[col] = {
                            'strategy': 'median_fallback',
                            'missing_before': data[col].isnull().sum(),
                            'missing_after': result_data[col].isnull().sum(),
                            'fallback_reason': str(e)
                        }
        
        # Handle categorical columns with most frequent
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                most_frequent = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'unknown'
                result_data[col].fillna(most_frequent, inplace=True)
                imputation_log[col] = {
                    'strategy': 'most_frequent',
                    'missing_before': data[col].isnull().sum(),
                    'missing_after': result_data[col].isnull().sum(),
                    'imputed_value': most_frequent
                }
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = time.time() - start_time
        
        metadata = {
            'strategy': 'knn',
            'parameters': {'n_neighbors': n_neighbors},
            'imputation_log': imputation_log,
            'execution_time': execution_time,
            'memory_usage_mb': peak / (1024 * 1024),
            'total_imputed_values': sum(log['missing_before'] - log['missing_after'] for log in imputation_log.values())
        }
        
        return result_data, metadata
    
    def _iterative_imputation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Iterative imputation using machine learning models."""
        result_data = data.copy()
        imputation_log = {}
        start_time = time.time()
        
        tracemalloc.start()
        
        # Prepare data for iterative imputation
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric columns with IterativeImputer
        if len(numeric_cols) > 1 and any(data[col].isnull().sum() > 0 for col in numeric_cols):
            try:
                iterative_imputer = IterativeImputer(
                    estimator=self.strategy_configs['iterative']['estimator'],
                    random_state=self.strategy_configs['iterative']['random_state'],
                    max_iter=self.strategy_configs['iterative']['max_iter']
                )
                
                numeric_data = data[numeric_cols]
                imputed_numeric = iterative_imputer.fit_transform(numeric_data)
                
                for i, col in enumerate(numeric_cols):
                    if numeric_data[col].isnull().sum() > 0:
                        result_data[col] = imputed_numeric[:, i]
                        self._fitted_imputers[f"{col}_iterative"] = iterative_imputer
                        imputation_log[col] = {
                            'strategy': 'iterative',
                            'estimator': str(iterative_imputer.estimator),
                            'n_iter': iterative_imputer.n_iter_,
                            'missing_before': data[col].isnull().sum(),
                            'missing_after': result_data[col].isnull().sum()
                        }
                        
            except Exception as e:
                logger.warning(f"Iterative imputation failed, falling back to median: {e}")
                # Fallback to simple imputation
                for col in numeric_cols:
                    if data[col].isnull().sum() > 0:
                        result_data[col].fillna(data[col].median(), inplace=True)
                        imputation_log[col] = {
                            'strategy': 'median_fallback',
                            'missing_before': data[col].isnull().sum(),
                            'missing_after': result_data[col].isnull().sum(),
                            'fallback_reason': str(e)
                        }
        
        # Handle categorical columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                # Use frequency-based imputation with threshold
                value_counts = data[col].value_counts()
                if len(value_counts) > 0 and value_counts.iloc[0] / len(data) > 0.05:
                    most_frequent = value_counts.index[0]
                    strategy_name = 'most_frequent'
                else:
                    most_frequent = 'unknown'
                    strategy_name = 'unknown_substitution'
                
                result_data[col].fillna(most_frequent, inplace=True)
                imputation_log[col] = {
                    'strategy': strategy_name,
                    'missing_before': data[col].isnull().sum(),
                    'missing_after': result_data[col].isnull().sum(),
                    'imputed_value': most_frequent
                }
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = time.time() - start_time
        
        metadata = {
            'strategy': 'iterative',
            'parameters': self.strategy_configs['iterative'],
            'imputation_log': imputation_log,
            'execution_time': execution_time,
            'memory_usage_mb': peak / (1024 * 1024),
            'total_imputed_values': sum(log['missing_before'] - log['missing_after'] for log in imputation_log.values())
        }
        
        return result_data, metadata
    
    def _custom_imputation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Custom imputation based on user-defined parameters."""
        # For now, implement domain-specific strategies
        result_data = data.copy()
        imputation_log = {}
        
        # Add missing value indicators for high-missing columns
        high_missing_cols = []
        for col in data.columns:
            missing_pct = (data[col].isnull().sum() / len(data)) * 100
            if missing_pct > 30:  # High missing rate
                indicator_col = f"{col}_was_missing"
                result_data[indicator_col] = data[col].isnull().astype(int)
                high_missing_cols.append(col)
                imputation_log[indicator_col] = {
                    'strategy': 'missing_indicator',
                    'original_column': col,
                    'missing_percentage': missing_pct
                }
        
        # Apply conservative imputation to original columns
        simple_data, simple_metadata = self._simple_imputation(data)
        result_data[data.columns] = simple_data[data.columns]
        
        # Combine logs
        imputation_log.update(simple_metadata.get('imputation_log', {}))
        
        metadata = {
            'strategy': 'custom',
            'high_missing_columns': high_missing_cols,
            'missing_indicators_added': len(high_missing_cols),
            'imputation_log': imputation_log,
            'execution_time': simple_metadata.get('execution_time', 0)
        }
        
        return result_data, metadata
    
    # ===========================================
    # QUALITY ASSESSMENT METHODS
    # ===========================================
    
    def _assess_imputation_quality(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Assess quality of imputation using various metrics."""
        if self._original_data is None:
            return data.copy(), {'quality_assessment': 'no_original_data'}
        
        quality_metrics = {}
        
        # Compare distributions before and after imputation
        numeric_cols = self._original_data.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if self._original_data[col].isnull().sum() > 0:  # Column had missing values
                original_values = self._original_data[col].dropna()
                imputed_values = data[col]
                
                # Distribution preservation
                try:
                    ks_stat, p_value = stats.ks_2samp(original_values, imputed_values)
                    distribution_preservation = 1 - ks_stat  # Higher is better
                except:
                    distribution_preservation = 0.5
                
                # Correlation preservation (if possible)
                correlation_preservation = self._assess_correlation_preservation(
                    self._original_data, data, col
                )
                
                quality_metrics[col] = {
                    'distribution_preservation': distribution_preservation,
                    'correlation_preservation': correlation_preservation,
                    'ks_statistic': ks_stat if 'ks_stat' in locals() else None,
                    'ks_p_value': p_value if 'p_value' in locals() else None
                }
        
        # Overall quality score
        if quality_metrics:
            overall_quality = np.mean([
                metrics['distribution_preservation'] for metrics in quality_metrics.values()
            ])
        else:
            overall_quality = 1.0  # No imputation needed
        
        metadata = {
            'quality_assessment': quality_metrics,
            'overall_quality_score': overall_quality,
            'quality_threshold_met': overall_quality >= self.quality_thresholds.get('min_correlation_preservation', 0.8)
        }
        
        return data.copy(), metadata
    
    def _assess_correlation_preservation(self, original_data: pd.DataFrame, 
                                       imputed_data: pd.DataFrame, target_col: str) -> float:
        """Assess how well correlations are preserved after imputation."""
        numeric_cols = original_data.select_dtypes(include=['number']).columns
        other_cols = [col for col in numeric_cols if col != target_col]
        
        if len(other_cols) == 0:
            return 1.0
        
        try:
            # Calculate correlations before imputation (using only complete cases)
            complete_cases = original_data[list(numeric_cols)].dropna()
            if len(complete_cases) < 10:
                return 0.5  # Not enough data
            
            original_corrs = complete_cases.corr()[target_col].drop(target_col)
            
            # Calculate correlations after imputation
            imputed_corrs = imputed_data[list(numeric_cols)].corr()[target_col].drop(target_col)
            
            # Compare correlations
            correlation_diff = np.abs(original_corrs - imputed_corrs).mean()
            correlation_preservation = max(0.0, 1.0 - correlation_diff)
            
            return correlation_preservation
            
        except:
            return 0.5  # Default if calculation fails
    
    # ===========================================
    # COMPREHENSIVE ANALYSIS METHODS
    # ===========================================
    
    def _evaluate_all_strategies(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Evaluate all available imputation strategies."""
        strategies = ['simple', 'knn', 'iterative']
        evaluation_results = {}
        
        for strategy in strategies:
            try:
                # Apply strategy
                if strategy == 'simple':
                    imputed_data, strategy_metadata = self._simple_imputation(data)
                elif strategy == 'knn':
                    imputed_data, strategy_metadata = self._knn_imputation(data)
                elif strategy == 'iterative':
                    imputed_data, strategy_metadata = self._iterative_imputation(data)
                
                # Assess quality
                _, quality_metadata = self._assess_imputation_quality(imputed_data)
                
                evaluation_results[strategy] = {
                    'strategy_metadata': strategy_metadata,
                    'quality_metadata': quality_metadata,
                    'overall_score': quality_metadata.get('overall_quality_score', 0)
                }
                
            except Exception as e:
                evaluation_results[strategy] = {
                    'error': str(e),
                    'overall_score': 0
                }
        
        # Select best strategy
        best_strategy = max(evaluation_results.keys(), 
                           key=lambda k: evaluation_results[k].get('overall_score', 0))
        
        metadata = {
            'evaluation_results': evaluation_results,
            'best_strategy': best_strategy,
            'best_score': evaluation_results[best_strategy].get('overall_score', 0)
        }
        
        # Store best strategy selection
        self.custom_parameters['evaluated_best_strategy'] = best_strategy
        
        return data.copy(), metadata
    
    def _cross_validate_strategies(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cross-validate imputation strategies for robust quality assessment."""
        if not self.cross_validation:
            return data.copy(), {'cross_validation': 'disabled'}
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        cv_results = {}
        
        # Only cross-validate on numeric columns with sufficient data
        testable_cols = [col for col in numeric_cols if data[col].isnull().sum() > 0 and data[col].isnull().sum() < len(data) * 0.5]
        
        if len(testable_cols) == 0:
            return data.copy(), {'cross_validation': 'no_suitable_columns'}
        
        strategies = ['simple', 'knn', 'iterative']
        
        for strategy in strategies:
            strategy_scores = []
            
            for col in testable_cols[:3]:  # Limit to first 3 columns for performance
                try:
                    # Create artificial missing values for testing
                    col_scores = self._cross_validate_column(data, col, strategy)
                    strategy_scores.extend(col_scores)
                except Exception as e:
                    logger.warning(f"Cross-validation failed for {strategy} on {col}: {e}")
            
            if strategy_scores:
                cv_results[strategy] = {
                    'mean_score': np.mean(strategy_scores),
                    'std_score': np.std(strategy_scores),
                    'scores': strategy_scores
                }
        
        metadata = {
            'cross_validation_results': cv_results,
            'tested_columns': testable_cols[:3],
            'best_cv_strategy': max(cv_results.keys(), key=lambda k: cv_results[k]['mean_score']) if cv_results else None
        }
        
        return data.copy(), metadata
    
    def _cross_validate_column(self, data: pd.DataFrame, column: str, strategy: str) -> List[float]:
        """Cross-validate imputation for a single column."""
        # Get complete cases for this column
        complete_data = data[data[column].notna()]
        
        if len(complete_data) < 20:  # Need minimum data for CV
            return [0.5]
        
        # 5-fold cross-validation
        kf = KFold(n_splits=min(5, len(complete_data) // 4), shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in kf.split(complete_data):
            try:
                # Create train/test split
                train_data = complete_data.iloc[train_idx].copy()
                test_data = complete_data.iloc[test_idx].copy()
                
                # Artificially remove values from test set
                test_values = test_data[column].copy()
                test_data[column] = np.nan
                
                # Combine train and test data for imputation
                cv_data = pd.concat([train_data, test_data])
                
                # Apply imputation strategy
                if strategy == 'simple':
                    imputed_data, _ = self._simple_imputation(cv_data)
                elif strategy == 'knn':
                    imputed_data, _ = self._knn_imputation(cv_data)
                elif strategy == 'iterative':
                    imputed_data, _ = self._iterative_imputation(cv_data)
                
                # Extract imputed values for test set
                imputed_test_values = imputed_data[column].iloc[len(train_data):]
                
                # Calculate accuracy (correlation with true values)
                if len(imputed_test_values) > 1 and len(test_values) > 1:
                    correlation = np.corrcoef(imputed_test_values, test_values)[0, 1]
                    scores.append(max(0, correlation))
                
            except Exception as e:
                scores.append(0.0)  # Failed fold
        
        return scores
    
    def _ensemble_imputation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Ensemble imputation combining multiple strategies."""
        # Get evaluation results
        eval_strategy = self.custom_parameters.get('evaluated_best_strategy', 'simple')
        cv_strategy = self.custom_parameters.get('best_cv_strategy')
        
        # Use the best validated strategy
        final_strategy = cv_strategy if cv_strategy else eval_strategy
        
        # Apply the selected strategy
        if final_strategy == 'simple':
            return self._simple_imputation(data)
        elif final_strategy == 'knn':
            return self._knn_imputation(data)
        elif final_strategy == 'iterative':
            return self._iterative_imputation(data)
        else:
            return self._simple_imputation(data)  # Fallback
    
    def _comprehensive_quality_assessment(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Comprehensive quality assessment with detailed metrics."""
        _, basic_quality = self._assess_imputation_quality(data)
        
        # Additional comprehensive metrics
        comprehensive_metrics = {
            'basic_quality': basic_quality,
            'imputation_coverage': self._calculate_imputation_coverage(data),
            'data_integrity': self._assess_data_integrity(data),
            'imputation_artifacts': self._detect_imputation_artifacts(data)
        }
        
        return data.copy(), comprehensive_metrics
    
    def _calculate_imputation_coverage(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate imputation coverage statistics."""
        if self._original_data is None:
            return {'coverage': 'no_original_data'}
        
        original_missing = self._original_data.isnull().sum().sum()
        current_missing = data.isnull().sum().sum()
        
        coverage = {
            'original_missing_values': original_missing,
            'remaining_missing_values': current_missing,
            'imputation_rate': (original_missing - current_missing) / original_missing if original_missing > 0 else 1.0,
            'complete_imputation': current_missing == 0
        }
        
        return coverage
    
    def _assess_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data integrity after imputation."""
        integrity_checks = {}
        
        # Check for reasonable value ranges
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if self._original_data is not None and col in self._original_data.columns:
                orig_min, orig_max = self._original_data[col].min(), self._original_data[col].max()
                curr_min, curr_max = data[col].min(), data[col].max()
                
                integrity_checks[col] = {
                    'values_within_original_range': curr_min >= orig_min and curr_max <= orig_max,
                    'range_expansion': (curr_max - curr_min) / (orig_max - orig_min) if orig_max != orig_min else 1.0
                }
        
        return integrity_checks
    
    def _detect_imputation_artifacts(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential artifacts from imputation."""
        artifacts = {}
        
        # Check for repeated values (potential over-imputation)
        for col in data.columns:
            if self._original_data is not None and col in self._original_data.columns:
                if self._original_data[col].isnull().sum() > 0:
                    # Check for artificial concentration of values
                    value_counts = data[col].value_counts()
                    most_common_freq = value_counts.iloc[0] / len(data) if len(value_counts) > 0 else 0
                    
                    artifacts[col] = {
                        'high_concentration_detected': most_common_freq > 0.3,
                        'most_common_frequency': most_common_freq,
                        'unique_values_ratio': data[col].nunique() / len(data)
                    }
        
        return artifacts
    
    def _compile_imputation_metadata(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compile comprehensive imputation metadata."""
        if not self.metadata_tracking:
            return data.copy(), {'metadata_compilation': 'disabled'}
        
        # Compile all metadata from the imputation process
        metadata = {
            'imputation_summary': {
                'missing_pattern': self._missing_pattern.__dict__ if self._missing_pattern else {},
                'total_original_missing': self._original_data.isnull().sum().sum() if self._original_data is not None else 0,
                'total_final_missing': data.isnull().sum().sum(),
                'imputation_complete': data.isnull().sum().sum() == 0
            },
            'fitted_imputers': list(self._fitted_imputers.keys()),
            'quality_thresholds': self.quality_thresholds,
            'configuration': {
                'strategy': self.strategy,
                'complexity': self.complexity,
                'cross_validation_enabled': self.cross_validation
            }
        }
        
        return data.copy(), metadata
    
    # ===========================================
    # UTILITY METHODS
    # ===========================================
    
    def _build_imputation_metadata(self, processed_data: pd.DataFrame, streaming_enabled: bool) -> Dict[str, Any]:
        """Build comprehensive metadata for imputation results."""
        metadata = {
            "imputation_pipeline": {
                "analytical_intention": self.analytical_intention,
                "strategy": self.strategy,
                "complexity": self.complexity,
                "streaming_enabled": streaming_enabled,
                "cross_validation": self.cross_validation
            },
            "missing_value_analysis": {
                "pattern_type": self._missing_pattern.pattern_type if self._missing_pattern else "unknown",
                "pattern_confidence": self._missing_pattern.confidence_score if self._missing_pattern else 0.0,
                "recommendations_followed": self._missing_pattern.recommendations if self._missing_pattern else []
            },
            "imputation_results": {
                "original_missing_values": self._original_data.isnull().sum().sum() if self._original_data is not None else 0,
                "final_missing_values": processed_data.isnull().sum().sum(),
                "imputation_complete": processed_data.isnull().sum().sum() == 0,
                "columns_imputed": len(self._fitted_imputers)
            },
            "quality_assessment": {
                "quality_thresholds": self.quality_thresholds,
                "imputation_artifacts_detected": False  # Would be set by quality assessment
            },
            "composition_context": {
                "ready_for_analysis": processed_data.isnull().sum().sum() == 0,
                "data_characteristics": self._analyze_imputed_data(processed_data),
                "suggested_next_steps": self._suggest_post_imputation_steps(processed_data),
                "imputation_artifacts": self._extract_imputation_artifacts()
            }
        }
        
        return metadata
    
    def _analyze_imputed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of imputed data."""
        return {
            "shape": data.shape,
            "dtypes": dict(data.dtypes),
            "missing_values": data.isnull().sum().sum(),
            "numeric_columns": data.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024),
            "imputation_complete": data.isnull().sum().sum() == 0
        }
    
    def _suggest_post_imputation_steps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest next steps after imputation."""
        suggestions = []
        
        if data.isnull().sum().sum() == 0:
            suggestions.append({
                "analysis_type": "data_validation",
                "reason": "Complete imputation achieved - validate data quality",
                "confidence": 0.9
            })
            
            suggestions.append({
                "analysis_type": "outlier_detection", 
                "reason": "Imputation may have introduced outliers",
                "confidence": 0.7
            })
            
            suggestions.append({
                "analysis_type": "feature_engineering",
                "reason": "Complete dataset ready for feature creation",
                "confidence": 0.8
            })
        else:
            suggestions.append({
                "analysis_type": "advanced_imputation",
                "reason": "Some missing values remain - consider domain-specific methods",
                "confidence": 0.6
            })
        
        return suggestions
    
    def _extract_imputation_artifacts(self) -> Dict[str, Any]:
        """Extract imputation artifacts for potential reuse or analysis."""
        artifacts = {
            "fitted_imputers": self._fitted_imputers,
            "imputation_strategy_used": self.strategy,
            "missing_value_pattern": self._missing_pattern.__dict__ if self._missing_pattern else {},
            "quality_thresholds": self.quality_thresholds
        }
        
        return artifacts
    
    # Public utility methods
    def get_imputation_summary(self) -> str:
        """Get human-readable summary of imputation operations."""
        if self._original_data is None:
            return "No imputation performed yet."
        
        original_missing = self._original_data.isnull().sum().sum()
        pattern_type = self._missing_pattern.pattern_type if self._missing_pattern else "unknown"
        
        summary_parts = [
            f"Missing value imputation completed using {self.strategy} strategy.",
            f"Missing value pattern detected: {pattern_type}",
            f"Original missing values: {original_missing:,}",
            f"Imputation strategies applied: {len(self._fitted_imputers)}"
        ]
        
        if self._missing_pattern:
            summary_parts.append(f"Pattern confidence: {self._missing_pattern.confidence_score:.2f}")
        
        return "\n".join(summary_parts)
    
    def is_imputation_complete(self) -> bool:
        """Check if imputation is complete (no missing values remain)."""
        return len(self._fitted_imputers) > 0  # At least some imputation was performed
    
    def get_missing_value_pattern(self) -> Optional[MissingValuePattern]:
        """Get the analyzed missing value pattern."""
        return self._missing_pattern
    
    def get_fitted_imputers(self) -> Dict[str, Any]:
        """Get fitted imputers for reuse or inspection."""
        return self._fitted_imputers.copy()