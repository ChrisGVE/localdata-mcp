"""
Pre-built Domain Shims for LocalData MCP v2.0 Integration Framework.

This module provides pre-built domain shims for common data science workflows,
enabling seamless integration between statistical, regression, time series, 
and pattern recognition domains.

Key Features:
- StatisticalShim: Bridge statistical analysis with other domains
- RegressionShim: Connect regression modeling with other domains  
- TimeSeriesShim: Enable time series integration across domains
- PatternRecognitionShim: Bridge pattern recognition with other domains
- Domain-specific parameter mapping and result normalization
- Intelligent semantic understanding of data transformations

Design Principles:
- Intention-Driven Interface: Shims understand analytical goals, not just data formats
- Context-Aware Composition: Preserve semantic meaning across domain boundaries
- Progressive Disclosure: Simple defaults with advanced customization options
- Streaming-First: Memory-efficient processing for large datasets
- Modular Domain Integration: Easy extension to new domains
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import abstractmethod

from .shim_registry import EnhancedShimAdapter, AdapterConfig
from .interfaces import (
    DataFormat, ConversionRequest, ConversionResult, ConversionCost,
    ConversionContext, MemoryConstraints, PerformanceRequirements,
    ValidationResult, ConversionError
)
from .converters import PandasConverter, NumpyConverter, ConversionOptions
from .type_detection import TypeDetectionEngine, FormatDetectionResult
from .metadata_manager import MetadataManager
from ...logging_manager import get_logger

logger = get_logger(__name__)


class DomainShimType(Enum):
    """Types of domain shims available."""
    STATISTICAL = "statistical"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    PATTERN_RECOGNITION = "pattern_recognition"


@dataclass
class DomainMapping:
    """Mapping configuration for cross-domain transformations."""
    source_domain: str
    target_domain: str
    parameter_mappings: Dict[str, str] = field(default_factory=dict)
    result_transformations: Dict[str, str] = field(default_factory=dict)
    semantic_hints: Dict[str, Any] = field(default_factory=dict)
    quality_preservation: float = 1.0  # 0-1 score for information preservation


@dataclass
class SemanticContext:
    """Semantic context for domain-aware transformations."""
    analytical_goal: str  # Primary analysis intention
    domain_context: str  # Source domain context
    target_use_case: str  # Target domain use case
    data_characteristics: Dict[str, Any] = field(default_factory=dict)
    transformation_hints: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)


class BaseDomainShim(EnhancedShimAdapter):
    """
    Base class for all domain-specific shims.
    
    Provides common functionality for domain integration including parameter
    mapping, result normalization, and semantic context preservation.
    """
    
    def __init__(self, adapter_id: str, domain_type: DomainShimType, 
                 config: Optional[AdapterConfig] = None,
                 **kwargs):
        """
        Initialize BaseDomainShim.
        
        Args:
            adapter_id: Unique identifier for this shim
            domain_type: Type of domain this shim handles
            config: Optional adapter configuration
            **kwargs: Additional arguments
        """
        super().__init__(adapter_id, config, **kwargs)
        
        self.domain_type = domain_type
        self.supported_mappings: List[DomainMapping] = []
        
        # Initialize domain-specific components
        self._pandas_converter = PandasConverter()
        self._numpy_converter = NumpyConverter()
        self._type_detector = TypeDetectionEngine()
        self._metadata_manager = MetadataManager()
        
        # Domain knowledge base
        self._domain_schemas = {}
        self._parameter_maps = {}
        self._result_normalizers = {}
        
        # Initialize domain-specific configurations
        self._initialize_domain_knowledge()
        
        logger.info(f"BaseDomainShim initialized",
                   adapter_id=adapter_id,
                   domain_type=domain_type.value)
    
    def _initialize_impl(self) -> bool:
        """Initialize domain-specific components."""
        try:
            # Initialize converters
            self._pandas_converter.initialize()
            self._numpy_converter.initialize()
            
            # Activate converters
            self._pandas_converter.activate()
            self._numpy_converter.activate()
            
            # Initialize domain-specific knowledge
            self._load_domain_mappings()
            
            logger.info(f"Domain shim '{self.adapter_id}' initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize domain shim '{self.adapter_id}': {e}")
            return False
    
    def _activate_impl(self) -> bool:
        """Activate domain shim for processing."""
        try:
            # Ensure all components are active
            if (self._pandas_converter.state.value != 'active' or 
                self._numpy_converter.state.value != 'active'):
                return False
            
            logger.info(f"Domain shim '{self.adapter_id}' activated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate domain shim '{self.adapter_id}': {e}")
            return False
    
    @abstractmethod
    def _initialize_domain_knowledge(self) -> None:
        """Initialize domain-specific knowledge and mappings."""
        pass
    
    @abstractmethod
    def _load_domain_mappings(self) -> None:
        """Load domain-specific conversion mappings."""
        pass
    
    def can_convert(self, request: ConversionRequest) -> float:
        """
        Evaluate conversion capability with semantic understanding.
        
        Args:
            request: Conversion request to evaluate
            
        Returns:
            Confidence score (0-1) for handling this conversion
        """
        confidence = 0.0
        
        try:
            # Check if this is a domain-relevant conversion
            source_domain = self._extract_domain_from_format(request.source_format)
            target_domain = self._extract_domain_from_format(request.target_format)
            
            # Higher confidence for domain-specific conversions
            if (source_domain == self.domain_type.value or 
                target_domain == self.domain_type.value):
                confidence += 0.7
            
            # Check for supported mapping
            mapping = self._find_domain_mapping(source_domain, target_domain)
            if mapping:
                confidence += 0.2
            
            # Consider data characteristics
            if hasattr(request, 'context') and request.context:
                semantic_match = self._evaluate_semantic_match(request.context)
                confidence += semantic_match * 0.1
            
            logger.debug(f"Conversion confidence for '{self.adapter_id}': {confidence}",
                        source_format=request.source_format.value,
                        target_format=request.target_format.value)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error evaluating conversion capability: {e}")
            return 0.0
    
    def _convert_impl(self, request: ConversionRequest) -> ConversionResult:
        """
        Implement domain-aware conversion with semantic preservation.
        
        Args:
            request: Conversion request
            
        Returns:
            Conversion result with domain-specific enhancements
        """
        start_time = time.time()
        
        try:
            # Extract semantic context
            semantic_context = self._extract_semantic_context(request)
            
            # Find appropriate domain mapping
            source_domain = self._extract_domain_from_format(request.source_format)
            target_domain = self._extract_domain_from_format(request.target_format)
            mapping = self._find_domain_mapping(source_domain, target_domain)
            
            if not mapping:
                # Fallback to generic conversion
                return self._generic_conversion(request, semantic_context)
            
            # Perform domain-specific conversion
            converted_data = self._perform_domain_conversion(
                request, mapping, semantic_context
            )
            
            # Normalize results for target domain
            normalized_data = self._normalize_results(
                converted_data, mapping, semantic_context
            )
            
            # Preserve and enhance metadata
            enhanced_metadata = self._enhance_metadata(request, mapping, semantic_context)
            
            # Calculate quality metrics
            quality_score = self._calculate_conversion_quality(
                request, normalized_data, mapping, semantic_context
            )
            
            execution_time = time.time() - start_time
            
            result = ConversionResult(
                converted_data=normalized_data,
                success=True,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.target_format,
                metadata=enhanced_metadata,
                performance_metrics={
                    'execution_time': execution_time,
                    'adapter_id': self.adapter_id,
                    'domain_mapping': mapping.source_domain + '_to_' + mapping.target_domain,
                    'semantic_context': semantic_context.analytical_goal,
                    'quality_preservation': mapping.quality_preservation
                },
                quality_score=quality_score,
                request_id=request.request_id,
                execution_time=execution_time
            )
            
            logger.info("Domain conversion completed successfully",
                       request_id=request.request_id,
                       adapter_id=self.adapter_id,
                       execution_time=execution_time,
                       quality_score=quality_score)
            
            return result
            
        except Exception as e:
            logger.error(f"Domain conversion failed: {e}")
            return self._create_error_result(request, str(e), time.time() - start_time)
    
    def _extract_domain_from_format(self, data_format: DataFormat) -> str:
        """Extract domain name from data format."""
        format_value = data_format.value.lower()
        
        if 'statistical' in format_value:
            return 'statistical'
        elif 'regression' in format_value:
            return 'regression'
        elif 'time_series' in format_value or 'forecast' in format_value:
            return 'time_series'
        elif ('clustering' in format_value or 'pattern' in format_value or 
              'classification' in format_value):
            return 'pattern_recognition'
        else:
            return 'generic'
    
    def _find_domain_mapping(self, source_domain: str, target_domain: str) -> Optional[DomainMapping]:
        """Find appropriate domain mapping for conversion."""
        for mapping in self.supported_mappings:
            if (mapping.source_domain == source_domain and 
                mapping.target_domain == target_domain):
                return mapping
        return None
    
    def _extract_semantic_context(self, request: ConversionRequest) -> SemanticContext:
        """Extract semantic context from conversion request."""
        context = request.context if hasattr(request, 'context') and request.context else None
        
        analytical_goal = "data_transformation"
        domain_context = self.domain_type.value
        target_use_case = "analysis"
        
        if context:
            analytical_goal = getattr(context, 'user_intention', analytical_goal)
            domain_context = getattr(context, 'source_domain', domain_context)
            target_use_case = getattr(context, 'target_domain', target_use_case)
        
        return SemanticContext(
            analytical_goal=analytical_goal,
            domain_context=domain_context,
            target_use_case=target_use_case,
            data_characteristics=self._analyze_data_characteristics(request.source_data),
            transformation_hints={},
            quality_requirements={'preservation': 0.9, 'accuracy': 0.95}
        )
    
    def _analyze_data_characteristics(self, data: Any) -> Dict[str, Any]:
        """Analyze characteristics of source data."""
        characteristics = {}
        
        try:
            if isinstance(data, pd.DataFrame):
                characteristics.update({
                    'data_type': 'dataframe',
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict(),
                    'missing_values': data.isnull().sum().to_dict(),
                    'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
                    'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns)
                })
            elif isinstance(data, np.ndarray):
                characteristics.update({
                    'data_type': 'numpy_array',
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'dimensions': data.ndim,
                    'size': data.size
                })
            elif isinstance(data, dict):
                characteristics.update({
                    'data_type': 'dictionary',
                    'keys': list(data.keys()),
                    'values_types': [type(v).__name__ for v in data.values()]
                })
            else:
                characteristics['data_type'] = type(data).__name__
            
        except Exception as e:
            logger.warning(f"Failed to analyze data characteristics: {e}")
            characteristics['analysis_error'] = str(e)
        
        return characteristics
    
    def _evaluate_semantic_match(self, context: ConversionContext) -> float:
        """Evaluate how well this shim matches the semantic context."""
        # Base implementation - can be overridden by specific shims
        return 0.5
    
    @abstractmethod
    def _perform_domain_conversion(self, request: ConversionRequest, 
                                  mapping: DomainMapping, 
                                  semantic_context: SemanticContext) -> Any:
        """Perform domain-specific conversion logic."""
        pass
    
    @abstractmethod
    def _normalize_results(self, data: Any, mapping: DomainMapping, 
                          semantic_context: SemanticContext) -> Any:
        """Normalize conversion results for target domain."""
        pass
    
    def _generic_conversion(self, request: ConversionRequest, 
                           semantic_context: SemanticContext) -> ConversionResult:
        """Fallback to generic conversion when no domain mapping exists."""
        try:
            # Use pandas converter as fallback
            return self._pandas_converter.convert(request)
        except Exception as e:
            logger.error(f"Generic conversion failed: {e}")
            return self._create_error_result(request, f"Generic conversion failed: {e}", 0.0)
    
    def _enhance_metadata(self, request: ConversionRequest, 
                         mapping: DomainMapping, 
                         semantic_context: SemanticContext) -> Dict[str, Any]:
        """Enhance metadata with domain-specific information."""
        enhanced_metadata = request.metadata.copy()
        
        enhanced_metadata.update({
            'domain_shim': {
                'adapter_id': self.adapter_id,
                'domain_type': self.domain_type.value,
                'mapping_used': {
                    'source_domain': mapping.source_domain,
                    'target_domain': mapping.target_domain,
                    'quality_preservation': mapping.quality_preservation
                },
                'semantic_context': {
                    'analytical_goal': semantic_context.analytical_goal,
                    'domain_context': semantic_context.domain_context,
                    'target_use_case': semantic_context.target_use_case
                }
            }
        })
        
        return enhanced_metadata
    
    def _calculate_conversion_quality(self, request: ConversionRequest, 
                                    converted_data: Any,
                                    mapping: DomainMapping,
                                    semantic_context: SemanticContext) -> float:
        """Calculate quality score for domain conversion."""
        base_quality = mapping.quality_preservation
        
        # Adjust based on data characteristics preservation
        if 'shape' in semantic_context.data_characteristics:
            original_shape = semantic_context.data_characteristics['shape']
            if hasattr(converted_data, 'shape'):
                if original_shape == converted_data.shape:
                    base_quality += 0.05
                else:
                    base_quality -= 0.1
        
        # Adjust based on semantic context match
        if semantic_context.analytical_goal in mapping.semantic_hints:
            base_quality += 0.05
        
        return min(max(base_quality, 0.0), 1.0)
    
    def _create_error_result(self, request: ConversionRequest, 
                            error_message: str, execution_time: float) -> ConversionResult:
        """Create error result for failed conversions."""
        return ConversionResult(
            converted_data=request.source_data,
            success=False,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=request.source_format,
            errors=[f"Domain conversion error: {error_message}"],
            performance_metrics={
                'execution_time': execution_time,
                'adapter_id': self.adapter_id,
                'error': error_message
            },
            quality_score=0.0,
            request_id=request.request_id,
            execution_time=execution_time
        )
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Estimate computational cost of domain conversion."""
        # Base cost estimation
        base_cost = 0.3  # Medium computational cost
        memory_cost = 50.0  # MB
        time_estimate = 2.0  # seconds
        
        # Adjust based on data size
        if hasattr(request.source_data, 'shape'):
            data_size = np.prod(request.source_data.shape)
            if data_size > 100000:
                base_cost += 0.2
                memory_cost += data_size / 1000
                time_estimate += data_size / 50000
        
        # Adjust based on domain complexity
        source_domain = self._extract_domain_from_format(request.source_format)
        target_domain = self._extract_domain_from_format(request.target_format)
        
        if source_domain != target_domain:
            base_cost += 0.1
            time_estimate += 0.5
        
        return ConversionCost(
            computational_cost=min(base_cost, 1.0),
            memory_cost_mb=memory_cost,
            time_estimate_seconds=time_estimate,
            io_operations=1,
            network_operations=0,
            quality_impact=0.05  # Small quality impact from domain conversion
        )
    
    def get_supported_conversions(self) -> List[Tuple[DataFormat, DataFormat]]:
        """Get list of supported conversion paths for this domain shim."""
        conversions = []
        
        # Add domain-specific conversions based on mappings
        for mapping in self.supported_mappings:
            source_formats = self._get_formats_for_domain(mapping.source_domain)
            target_formats = self._get_formats_for_domain(mapping.target_domain)
            
            for source_format in source_formats:
                for target_format in target_formats:
                    conversions.append((source_format, target_format))
        
        return conversions
    
    def _get_formats_for_domain(self, domain: str) -> List[DataFormat]:
        """Get data formats associated with a domain."""
        domain_formats = {
            'statistical': [
                DataFormat.STATISTICAL_RESULT,
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.NUMPY_ARRAY
            ],
            'regression': [
                DataFormat.REGRESSION_MODEL,
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.NUMPY_ARRAY
            ],
            'time_series': [
                DataFormat.TIME_SERIES,
                DataFormat.FORECAST_RESULT,
                DataFormat.PANDAS_DATAFRAME
            ],
            'pattern_recognition': [
                DataFormat.PATTERN_RECOGNITION_RESULT,
                DataFormat.CLUSTERING_RESULT,
                DataFormat.NUMPY_ARRAY
            ],
            'generic': [
                DataFormat.PANDAS_DATAFRAME,
                DataFormat.NUMPY_ARRAY,
                DataFormat.PYTHON_DICT,
                DataFormat.PYTHON_LIST
            ]
        }
        
        return domain_formats.get(domain, domain_formats['generic'])


class StatisticalShim(BaseDomainShim):
    """
    Bridge statistical analysis with other domains.
    
    Handles conversions between statistical analysis results and other domain formats,
    including correlation matrices, hypothesis test results, and descriptive statistics.
    """
    
    def __init__(self, adapter_id: str = "statistical_shim", 
                 config: Optional[AdapterConfig] = None, **kwargs):
        """Initialize StatisticalShim."""
        super().__init__(
            adapter_id=adapter_id,
            domain_type=DomainShimType.STATISTICAL,
            config=config,
            **kwargs
        )
    
    def _initialize_domain_knowledge(self) -> None:
        """Initialize statistical domain knowledge."""
        self._domain_schemas = {
            'correlation_matrix': {
                'type': 'symmetric_matrix',
                'value_range': [-1, 1],
                'diagonal_ones': True
            },
            'hypothesis_test': {
                'required_fields': ['statistic', 'p_value'],
                'optional_fields': ['degrees_of_freedom', 'effect_size', 'confidence_interval']
            },
            'descriptive_stats': {
                'required_fields': ['mean', 'std', 'count'],
                'optional_fields': ['median', 'min', 'max', 'skewness', 'kurtosis']
            }
        }
        
        self._parameter_maps = {
            'statistical_to_regression': {
                'correlation_coefficient': 'feature_correlation',
                'p_value': 'significance',
                'confidence_interval': 'prediction_interval'
            },
            'statistical_to_time_series': {
                'trend_coefficient': 'trend',
                'seasonal_component': 'seasonality',
                'residuals': 'error_terms'
            }
        }
    
    def _load_domain_mappings(self) -> None:
        """Load statistical domain mappings."""
        # Statistical to Regression mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='statistical',
            target_domain='regression',
            parameter_mappings={
                'correlation_matrix': 'feature_correlation_matrix',
                'test_statistics': 'model_diagnostics',
                'confidence_intervals': 'prediction_intervals',
                'p_values': 'feature_significance'
            },
            result_transformations={
                'statistical_summary': 'regression_features',
                'hypothesis_results': 'model_validation'
            },
            semantic_hints={
                'correlation_analysis': 'feature_selection',
                'normality_test': 'assumption_checking',
                'outlier_detection': 'data_preprocessing'
            },
            quality_preservation=0.95
        ))
        
        # Statistical to Time Series mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='statistical',
            target_domain='time_series',
            parameter_mappings={
                'trend_statistics': 'trend_parameters',
                'seasonal_tests': 'seasonality_detection',
                'stationarity_tests': 'differencing_requirements'
            },
            result_transformations={
                'time_series_stats': 'ts_characteristics',
                'decomposition_results': 'component_analysis'
            },
            semantic_hints={
                'trend_analysis': 'trend_modeling',
                'seasonal_decomposition': 'seasonal_adjustment',
                'autocorrelation': 'lag_analysis'
            },
            quality_preservation=0.90
        ))
        
        # Statistical to Pattern Recognition mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='statistical',
            target_domain='pattern_recognition',
            parameter_mappings={
                'principal_components': 'dimensionality_reduction',
                'cluster_statistics': 'clustering_validation',
                'distribution_parameters': 'model_parameters'
            },
            result_transformations={
                'statistical_features': 'pattern_features',
                'correlation_structure': 'similarity_matrix'
            },
            semantic_hints={
                'pca_analysis': 'dimensionality_reduction',
                'cluster_validation': 'clustering_evaluation',
                'anomaly_detection': 'outlier_identification'
            },
            quality_preservation=0.85
        ))
    
    def _perform_domain_conversion(self, request: ConversionRequest, 
                                  mapping: DomainMapping, 
                                  semantic_context: SemanticContext) -> Any:
        """Perform statistical domain conversion."""
        source_data = request.source_data
        
        if mapping.target_domain == 'regression':
            return self._convert_statistical_to_regression(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == 'time_series':
            return self._convert_statistical_to_time_series(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == 'pattern_recognition':
            return self._convert_statistical_to_pattern_recognition(
                source_data, mapping, semantic_context
            )
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target domain: {mapping.target_domain}"
            )
    
    def _convert_statistical_to_regression(self, data: Any, mapping: DomainMapping,
                                         semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert statistical results to regression-friendly format."""
        if isinstance(data, dict):
            # Handle statistical test results
            if 'correlation_matrix' in data:
                # Convert correlation matrix to feature correlation
                corr_matrix = data['correlation_matrix']
                if isinstance(corr_matrix, pd.DataFrame):
                    return {
                        'feature_correlation_matrix': corr_matrix.values,
                        'feature_names': list(corr_matrix.columns),
                        'correlation_significance': data.get('p_values', None)
                    }
            
            # Handle hypothesis test results
            if 'test_statistic' in data and 'p_value' in data:
                return {
                    'model_diagnostics': {
                        'test_statistic': data['test_statistic'],
                        'significance': data['p_value'],
                        'degrees_of_freedom': data.get('degrees_of_freedom'),
                        'effect_size': data.get('effect_size')
                    }
                }
        
        elif isinstance(data, pd.DataFrame):
            # Convert statistical summary to regression features
            if data.index.name in ['mean', 'std', 'count'] or any(
                stat in str(data.index) for stat in ['mean', 'std', 'median']
            ):
                return {
                    'descriptive_statistics': data.to_dict(),
                    'feature_names': list(data.columns),
                    'summary_type': 'regression_preprocessing'
                }
        
        # Fallback: return as-is with regression context
        return {
            'statistical_input': data,
            'conversion_type': 'statistical_to_regression',
            'metadata': semantic_context.__dict__
        }
    
    def _convert_statistical_to_time_series(self, data: Any, mapping: DomainMapping,
                                          semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert statistical results to time series format."""
        if isinstance(data, dict):
            # Handle time series statistical tests
            if 'stationarity_test' in data:
                return {
                    'stationarity_info': {
                        'test_statistic': data.get('test_statistic'),
                        'p_value': data.get('p_value'),
                        'is_stationary': data.get('p_value', 1.0) < 0.05,
                        'differencing_suggested': data.get('p_value', 1.0) > 0.05
                    }
                }
            
            # Handle autocorrelation analysis
            if 'autocorrelation' in data or 'acf' in data:
                return {
                    'autocorrelation_structure': {
                        'acf_values': data.get('autocorrelation', data.get('acf')),
                        'pacf_values': data.get('partial_autocorrelation', data.get('pacf')),
                        'significant_lags': data.get('significant_lags', [])
                    }
                }
        
        elif isinstance(data, pd.DataFrame):
            # Handle time series decomposition statistics
            if any(col.lower() in ['trend', 'seasonal', 'residual'] 
                   for col in data.columns):
                return {
                    'decomposition_statistics': data.to_dict(),
                    'components': list(data.columns),
                    'time_index': data.index.tolist() if hasattr(data, 'index') else None
                }
        
        # Fallback
        return {
            'statistical_input': data,
            'conversion_type': 'statistical_to_time_series',
            'temporal_context': semantic_context.transformation_hints
        }
    
    def _convert_statistical_to_pattern_recognition(self, data: Any, mapping: DomainMapping,
                                                  semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert statistical results to pattern recognition format."""
        if isinstance(data, dict):
            # Handle PCA results
            if 'principal_components' in data or 'explained_variance' in data:
                return {
                    'dimensionality_reduction': {
                        'components': data.get('principal_components'),
                        'explained_variance_ratio': data.get('explained_variance_ratio'),
                        'cumulative_variance': data.get('cumulative_variance'),
                        'n_components': data.get('n_components')
                    }
                }
            
            # Handle clustering validation statistics
            if 'silhouette_score' in data or 'inertia' in data:
                return {
                    'clustering_validation': {
                        'silhouette_score': data.get('silhouette_score'),
                        'inertia': data.get('inertia'),
                        'calinski_harabasz_score': data.get('calinski_harabasz_score'),
                        'davies_bouldin_score': data.get('davies_bouldin_score')
                    }
                }
        
        elif isinstance(data, pd.DataFrame):
            # Handle correlation matrix for similarity analysis
            if data.shape[0] == data.shape[1] and all(
                data.iloc[i, i] == 1.0 for i in range(min(3, data.shape[0]))
            ):
                # Likely a correlation matrix
                return {
                    'similarity_matrix': data.values,
                    'feature_names': list(data.columns),
                    'matrix_type': 'correlation'
                }
        
        # Fallback
        return {
            'statistical_input': data,
            'conversion_type': 'statistical_to_pattern_recognition',
            'pattern_hints': semantic_context.transformation_hints
        }
    
    def _normalize_results(self, data: Any, mapping: DomainMapping, 
                          semantic_context: SemanticContext) -> Any:
        """Normalize statistical conversion results for target domain."""
        if not isinstance(data, dict):
            return data
        
        # Add domain-specific metadata
        data['domain_conversion'] = {
            'source': 'statistical',
            'target': mapping.target_domain,
            'semantic_goal': semantic_context.analytical_goal,
            'quality_preservation': mapping.quality_preservation
        }
        
        # Ensure consistent structure
        if mapping.target_domain == 'regression':
            # Ensure regression-compatible structure
            if 'feature_correlation_matrix' in data:
                # Add metadata for regression feature selection
                data['feature_selection_hints'] = {
                    'highly_correlated_features': [],
                    'independent_features': [],
                    'multicollinearity_warnings': []
                }
        
        return data


class RegressionShim(BaseDomainShim):
    """
    Connect regression modeling with other domains.
    
    Transforms regression model outputs for time series forecasting, pattern recognition,
    and business intelligence, handling model coefficients, predictions, and residuals.
    """
    
    def __init__(self, adapter_id: str = "regression_shim", 
                 config: Optional[AdapterConfig] = None, **kwargs):
        """Initialize RegressionShim."""
        super().__init__(
            adapter_id=adapter_id,
            domain_type=DomainShimType.REGRESSION,
            config=config,
            **kwargs
        )
    
    def _initialize_domain_knowledge(self) -> None:
        """Initialize regression domain knowledge."""
        self._domain_schemas = {
            'linear_model': {
                'required_fields': ['coefficients', 'intercept', 'r2_score'],
                'optional_fields': ['std_error', 'p_values', 'confidence_intervals']
            },
            'model_diagnostics': {
                'required_fields': ['residuals', 'fitted_values'],
                'optional_fields': ['leverage', 'cooks_distance', 'standardized_residuals']
            },
            'predictions': {
                'required_fields': ['predicted_values'],
                'optional_fields': ['prediction_intervals', 'confidence_intervals']
            }
        }
    
    def _load_domain_mappings(self) -> None:
        """Load regression domain mappings."""
        # Regression to Time Series mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='regression',
            target_domain='time_series',
            parameter_mappings={
                'model_coefficients': 'trend_parameters',
                'residuals': 'error_component',
                'predictions': 'forecasted_values',
                'confidence_intervals': 'forecast_intervals'
            },
            result_transformations={
                'regression_model': 'trend_model',
                'fitted_values': 'fitted_trend',
                'model_diagnostics': 'forecast_diagnostics'
            },
            semantic_hints={
                'trend_modeling': 'linear_trend',
                'seasonal_regression': 'seasonal_components',
                'autoregression': 'ar_model'
            },
            quality_preservation=0.92
        ))
        
        # Regression to Pattern Recognition mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='regression',
            target_domain='pattern_recognition',
            parameter_mappings={
                'feature_importance': 'feature_weights',
                'model_coefficients': 'component_loadings',
                'residual_analysis': 'anomaly_scores'
            },
            result_transformations={
                'regression_features': 'pattern_features',
                'model_predictions': 'classification_scores'
            },
            semantic_hints={
                'feature_selection': 'dimensionality_reduction',
                'outlier_detection': 'anomaly_detection',
                'classification': 'supervised_learning'
            },
            quality_preservation=0.88
        ))
        
        # Regression to Statistical mapping  
        self.supported_mappings.append(DomainMapping(
            source_domain='regression',
            target_domain='statistical',
            parameter_mappings={
                'model_statistics': 'test_statistics',
                'p_values': 'statistical_significance',
                'residuals': 'error_distribution'
            },
            result_transformations={
                'regression_summary': 'statistical_summary',
                'anova_table': 'hypothesis_test_results'
            },
            semantic_hints={
                'significance_testing': 'hypothesis_testing',
                'model_validation': 'assumption_testing',
                'residual_analysis': 'normality_testing'
            },
            quality_preservation=0.94
        ))
    
    def _perform_domain_conversion(self, request: ConversionRequest, 
                                  mapping: DomainMapping, 
                                  semantic_context: SemanticContext) -> Any:
        """Perform regression domain conversion."""
        source_data = request.source_data
        
        if mapping.target_domain == 'time_series':
            return self._convert_regression_to_time_series(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == 'pattern_recognition':
            return self._convert_regression_to_pattern_recognition(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == 'statistical':
            return self._convert_regression_to_statistical(
                source_data, mapping, semantic_context
            )
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target domain: {mapping.target_domain}"
            )
    
    def _convert_regression_to_time_series(self, data: Any, mapping: DomainMapping,
                                         semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert regression results to time series format."""
        if isinstance(data, dict):
            # Handle regression model for trend analysis
            if 'coefficients' in data and 'fitted_values' in data:
                return {
                    'trend_model': {
                        'trend_parameters': data['coefficients'],
                        'fitted_trend': data['fitted_values'],
                        'residuals': data.get('residuals'),
                        'model_type': 'linear_trend'
                    },
                    'forecast_info': {
                        'trend_strength': abs(data['coefficients'][0]) if len(data['coefficients']) > 0 else 0,
                        'model_r2': data.get('r2_score', 0),
                        'forecast_reliability': data.get('r2_score', 0)
                    }
                }
            
            # Handle predictions for forecasting
            if 'predictions' in data or 'predicted_values' in data:
                predictions = data.get('predictions', data.get('predicted_values'))
                return {
                    'forecasted_values': predictions,
                    'forecast_intervals': data.get('confidence_intervals'),
                    'forecast_method': 'regression_based',
                    'forecast_horizon': len(predictions) if hasattr(predictions, '__len__') else 1
                }
        
        elif isinstance(data, pd.DataFrame):
            # Handle regression results DataFrame
            if 'fitted' in data.columns or 'predicted' in data.columns:
                fitted_col = 'fitted' if 'fitted' in data.columns else 'predicted'
                result = {
                    'fitted_values': data[fitted_col].values,
                    'time_index': data.index.tolist() if hasattr(data, 'index') else None
                }
                
                if 'residuals' in data.columns:
                    result['residuals'] = data['residuals'].values
                
                return result
        
        # Fallback
        return {
            'regression_input': data,
            'conversion_type': 'regression_to_time_series',
            'temporal_context': semantic_context.transformation_hints
        }
    
    def _convert_regression_to_pattern_recognition(self, data: Any, mapping: DomainMapping,
                                                 semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert regression results to pattern recognition format."""
        if isinstance(data, dict):
            # Handle feature importance for dimensionality reduction
            if 'feature_importance' in data or 'coefficients' in data:
                importance = data.get('feature_importance', np.abs(data.get('coefficients', [])))
                return {
                    'feature_weights': importance,
                    'feature_names': data.get('feature_names', []),
                    'importance_type': 'regression_coefficients',
                    'dimensionality_reduction_ready': True
                }
            
            # Handle residuals for anomaly detection
            if 'residuals' in data:
                residuals = np.array(data['residuals'])
                return {
                    'anomaly_scores': np.abs(residuals),
                    'residual_statistics': {
                        'mean': np.mean(residuals),
                        'std': np.std(residuals),
                        'outlier_threshold': np.mean(residuals) + 2 * np.std(residuals)
                    },
                    'anomaly_method': 'residual_based'
                }
            
            # Handle model predictions for classification
            if 'predictions' in data and 'model_type' in data:
                if data['model_type'] in ['logistic', 'classification']:
                    return {
                        'classification_scores': data['predictions'],
                        'prediction_probabilities': data.get('probabilities'),
                        'classification_method': 'regression_based'
                    }
        
        # Fallback
        return {
            'regression_input': data,
            'conversion_type': 'regression_to_pattern_recognition',
            'pattern_hints': semantic_context.transformation_hints
        }
    
    def _convert_regression_to_statistical(self, data: Any, mapping: DomainMapping,
                                         semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert regression results to statistical format."""
        if isinstance(data, dict):
            # Handle model summary for statistical analysis
            if 'coefficients' in data and 'p_values' in data:
                return {
                    'hypothesis_test_results': {
                        'test_statistics': data['coefficients'],
                        'p_values': data['p_values'],
                        'significance_level': 0.05,
                        'significant_features': [
                            i for i, p in enumerate(data['p_values']) 
                            if p < 0.05
                        ] if isinstance(data['p_values'], (list, np.ndarray)) else []
                    }
                }
            
            # Handle ANOVA table
            if 'anova_table' in data or ('f_statistic' in data and 'f_pvalue' in data):
                return {
                    'anova_results': {
                        'f_statistic': data.get('f_statistic'),
                        'f_pvalue': data.get('f_pvalue'),
                        'degrees_of_freedom': data.get('df_model', data.get('df')),
                        'sum_of_squares': data.get('sum_of_squares')
                    }
                }
            
            # Handle model diagnostics
            if 'residuals' in data:
                residuals = np.array(data['residuals'])
                return {
                    'residual_analysis': {
                        'normality_test': self._test_normality(residuals),
                        'homoscedasticity': self._test_homoscedasticity(residuals),
                        'autocorrelation': self._test_autocorrelation(residuals)
                    }
                }
        
        # Fallback
        return {
            'regression_input': data,
            'conversion_type': 'regression_to_statistical',
            'statistical_context': semantic_context.transformation_hints
        }
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test normality of residuals."""
        from scipy.stats import shapiro, jarque_bera
        
        try:
            # Shapiro-Wilk test (for small samples)
            if len(residuals) <= 5000:
                stat, p_value = shapiro(residuals)
                test_name = 'shapiro_wilk'
            else:
                # Jarque-Bera test (for larger samples)
                stat, p_value = jarque_bera(residuals)
                test_name = 'jarque_bera'
            
            return {
                'test_name': test_name,
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except Exception as e:
            return {'error': str(e), 'is_normal': False}
    
    def _test_homoscedasticity(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test homoscedasticity (constant variance) of residuals."""
        try:
            # Simple Breusch-Pagan style test
            abs_residuals = np.abs(residuals)
            mean_abs_residual = np.mean(abs_residuals)
            
            # Test if variance changes over time/fitted values
            n_groups = min(10, len(residuals) // 10)
            group_size = len(residuals) // n_groups
            
            group_variances = []
            for i in range(n_groups):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < n_groups - 1 else len(residuals)
                group_var = np.var(residuals[start_idx:end_idx])
                group_variances.append(group_var)
            
            # Simple test: variance of group variances
            variance_of_variances = np.var(group_variances)
            is_homoscedastic = variance_of_variances < (np.mean(group_variances) * 0.5)
            
            return {
                'test_name': 'group_variance_test',
                'variance_of_variances': variance_of_variances,
                'is_homoscedastic': is_homoscedastic
            }
        except Exception as e:
            return {'error': str(e), 'is_homoscedastic': False}
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for autocorrelation in residuals."""
        try:
            from scipy.stats import pearsonr
            
            if len(residuals) < 2:
                return {'error': 'Insufficient data', 'has_autocorrelation': False}
            
            # Test lag-1 autocorrelation
            lag1_corr, p_value = pearsonr(residuals[:-1], residuals[1:])
            
            return {
                'test_name': 'lag1_autocorrelation',
                'correlation': lag1_corr,
                'p_value': p_value,
                'has_autocorrelation': abs(lag1_corr) > 0.3 and p_value < 0.05
            }
        except Exception as e:
            return {'error': str(e), 'has_autocorrelation': False}
    
    def _normalize_results(self, data: Any, mapping: DomainMapping, 
                          semantic_context: SemanticContext) -> Any:
        """Normalize regression conversion results for target domain."""
        if not isinstance(data, dict):
            return data
        
        # Add domain-specific metadata
        data['domain_conversion'] = {
            'source': 'regression',
            'target': mapping.target_domain,
            'semantic_goal': semantic_context.analytical_goal,
            'quality_preservation': mapping.quality_preservation
        }
        
        # Ensure consistent structure based on target domain
        if mapping.target_domain == 'time_series':
            # Ensure time series compatible structure
            if 'fitted_values' in data:
                data['forecast_diagnostics'] = {
                    'method': 'regression_based_forecast',
                    'reliability': data.get('model_r2', 0.8),
                    'assumptions_met': True  # Simplified
                }
        
        elif mapping.target_domain == 'pattern_recognition':
            # Ensure pattern recognition compatible structure
            if 'feature_weights' in data:
                data['feature_selection_info'] = {
                    'selection_method': 'regression_importance',
                    'n_features_selected': len(data.get('feature_weights', [])),
                    'selection_threshold': 0.1
                }
        
        return data


class TimeSeriesShim(BaseDomainShim):
    """
    Enable time series integration across domains.
    
    Converts time series data for statistical analysis and regression models,
    handling temporal indexing, seasonality, trends, and forecast intervals.
    """
    
    def __init__(self, adapter_id: str = "time_series_shim", 
                 config: Optional[AdapterConfig] = None, **kwargs):
        """Initialize TimeSeriesShim."""
        super().__init__(
            adapter_id=adapter_id,
            domain_type=DomainShimType.TIME_SERIES,
            config=config,
            **kwargs
        )
    
    def _initialize_domain_knowledge(self) -> None:
        """Initialize time series domain knowledge."""
        self._domain_schemas = {
            'time_series_data': {
                'required_fields': ['values', 'timestamps'],
                'optional_fields': ['frequency', 'seasonal_period', 'trend']
            },
            'forecast_result': {
                'required_fields': ['forecasted_values', 'forecast_horizon'],
                'optional_fields': ['confidence_intervals', 'prediction_intervals', 'forecast_method']
            },
            'decomposition': {
                'components': ['trend', 'seasonal', 'residual', 'irregular'],
                'methods': ['additive', 'multiplicative']
            }
        }
    
    def _load_domain_mappings(self) -> None:
        """Load time series domain mappings."""
        # Time Series to Statistical mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='time_series',
            target_domain='statistical',
            parameter_mappings={
                'trend_component': 'trend_statistics',
                'seasonal_component': 'seasonal_statistics',
                'forecast_errors': 'residual_analysis',
                'autocorrelation_function': 'correlation_analysis'
            },
            result_transformations={
                'time_series_decomposition': 'component_statistics',
                'forecast_accuracy': 'prediction_validation'
            },
            semantic_hints={
                'trend_analysis': 'regression_analysis',
                'seasonality_detection': 'periodicity_test',
                'stationarity_testing': 'unit_root_test'
            },
            quality_preservation=0.93
        ))
        
        # Time Series to Regression mapping  
        self.supported_mappings.append(DomainMapping(
            source_domain='time_series',
            target_domain='regression',
            parameter_mappings={
                'lagged_features': 'regression_features',
                'trend_features': 'linear_predictors',
                'seasonal_features': 'categorical_predictors',
                'forecast_horizon': 'prediction_horizon'
            },
            result_transformations={
                'time_series_features': 'feature_matrix',
                'temporal_patterns': 'predictor_variables'
            },
            semantic_hints={
                'autoregression': 'lag_regression',
                'trend_modeling': 'linear_regression',
                'seasonal_adjustment': 'dummy_variables'
            },
            quality_preservation=0.90
        ))
        
        # Time Series to Pattern Recognition mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='time_series',
            target_domain='pattern_recognition',
            parameter_mappings={
                'temporal_features': 'pattern_features',
                'seasonal_patterns': 'recurring_patterns',
                'anomaly_scores': 'outlier_scores',
                'change_points': 'pattern_breaks'
            },
            result_transformations={
                'time_series_patterns': 'sequence_patterns',
                'temporal_clusters': 'pattern_clusters'
            },
            semantic_hints={
                'pattern_mining': 'sequence_analysis',
                'anomaly_detection': 'outlier_identification',
                'clustering': 'temporal_clustering'
            },
            quality_preservation=0.87
        ))
    
    def _perform_domain_conversion(self, request: ConversionRequest, 
                                  mapping: DomainMapping, 
                                  semantic_context: SemanticContext) -> Any:
        """Perform time series domain conversion."""
        source_data = request.source_data
        
        if mapping.target_domain == 'statistical':
            return self._convert_time_series_to_statistical(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == 'regression':
            return self._convert_time_series_to_regression(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == 'pattern_recognition':
            return self._convert_time_series_to_pattern_recognition(
                source_data, mapping, semantic_context
            )
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target domain: {mapping.target_domain}"
            )
    
    def _convert_time_series_to_statistical(self, data: Any, mapping: DomainMapping,
                                          semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert time series data to statistical format."""
        if isinstance(data, pd.DataFrame) and hasattr(data.index, 'freq'):
            # Handle time series DataFrame
            values = data.values if len(data.columns) == 1 else data.iloc[:, 0].values
            
            # Calculate basic statistics
            result = {
                'time_series_statistics': {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values),
                    'trend_slope': self._calculate_trend_slope(values),
                    'stationarity_score': self._assess_stationarity(values)
                }
            }
            
            # Add autocorrelation analysis
            acf_values = self._calculate_autocorrelation(values)
            if acf_values is not None:
                result['autocorrelation_analysis'] = {
                    'acf_values': acf_values[:min(20, len(acf_values))].tolist(),
                    'significant_lags': self._find_significant_lags(acf_values)
                }
            
            return result
        
        elif isinstance(data, dict) and 'values' in data:
            # Handle structured time series data
            values = np.array(data['values'])
            
            result = {
                'time_series_statistics': {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'variance': np.var(values),
                    'skewness': self._calculate_skewness(values),
                    'kurtosis': self._calculate_kurtosis(values)
                }
            }
            
            # Add decomposition statistics if available
            if 'trend' in data or 'seasonal' in data:
                result['decomposition_statistics'] = {
                    'trend_strength': self._calculate_component_strength(data.get('trend')),
                    'seasonal_strength': self._calculate_component_strength(data.get('seasonal')),
                    'residual_variance': np.var(data.get('residual', [0]))
                }
            
            return result
        
        # Fallback
        return {
            'time_series_input': data,
            'conversion_type': 'time_series_to_statistical',
            'statistical_context': semantic_context.transformation_hints
        }
    
    def _convert_time_series_to_regression(self, data: Any, mapping: DomainMapping,
                                         semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert time series data to regression format."""
        if isinstance(data, pd.DataFrame):
            # Create lagged features for regression
            values = data.values if len(data.columns) == 1 else data.iloc[:, 0].values
            
            # Generate lagged features
            n_lags = min(10, len(values) // 4)  # Use up to 10 lags or 1/4 of data length
            lagged_features = self._create_lagged_features(values, n_lags)
            
            # Create trend features
            trend_features = self._create_trend_features(len(values))
            
            # Combine features
            feature_matrix = np.column_stack([lagged_features, trend_features])
            
            # Create target variable (next period values)
            target_values = values[n_lags:]  # Skip first n_lags values
            
            return {
                'feature_matrix': feature_matrix,
                'target_variable': target_values,
                'feature_names': (
                    [f'lag_{i+1}' for i in range(n_lags)] + 
                    ['trend', 'trend_squared']
                ),
                'temporal_info': {
                    'n_lags': n_lags,
                    'sample_size': len(target_values),
                    'feature_type': 'time_series_derived'
                }
            }
        
        elif isinstance(data, dict) and 'forecast_result' in data:
            # Handle forecast results
            forecasts = data['forecast_result']
            
            result = {
                'prediction_data': {
                    'predicted_values': forecasts.get('forecasted_values'),
                    'prediction_intervals': forecasts.get('confidence_intervals'),
                    'prediction_method': forecasts.get('forecast_method', 'time_series')
                }
            }
            
            if 'actual_values' in data:
                result['validation_data'] = {
                    'actual_values': data['actual_values'],
                    'prediction_errors': self._calculate_prediction_errors(
                        data['actual_values'], forecasts.get('forecasted_values')
                    )
                }
            
            return result
        
        # Fallback
        return {
            'time_series_input': data,
            'conversion_type': 'time_series_to_regression',
            'regression_context': semantic_context.transformation_hints
        }
    
    def _convert_time_series_to_pattern_recognition(self, data: Any, mapping: DomainMapping,
                                                  semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert time series data to pattern recognition format."""
        if isinstance(data, pd.DataFrame):
            values = data.values if len(data.columns) == 1 else data.iloc[:, 0].values
            
            # Extract temporal features for pattern recognition
            temporal_features = self._extract_temporal_features(values)
            
            # Detect patterns and anomalies
            pattern_info = self._detect_patterns(values)
            
            return {
                'temporal_features': temporal_features,
                'pattern_detection': pattern_info,
                'sequence_characteristics': {
                    'length': len(values),
                    'variability': np.std(values),
                    'trend_direction': 'increasing' if temporal_features['trend_slope'] > 0 else 'decreasing',
                    'seasonality_present': pattern_info.get('seasonal_detected', False)
                }
            }
        
        elif isinstance(data, dict) and 'seasonal_patterns' in data:
            # Handle seasonal pattern data
            seasonal_data = data['seasonal_patterns']
            
            return {
                'recurring_patterns': {
                    'seasonal_components': seasonal_data,
                    'pattern_strength': self._calculate_pattern_strength(seasonal_data),
                    'pattern_frequency': self._estimate_pattern_frequency(seasonal_data)
                },
                'pattern_type': 'seasonal_time_series'
            }
        
        # Fallback
        return {
            'time_series_input': data,
            'conversion_type': 'time_series_to_pattern_recognition',
            'pattern_context': semantic_context.transformation_hints
        }
    
    # Helper methods for time series analysis
    
    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate the trend slope of time series."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.corrcoef(x, values)[0, 1] * (np.std(values) / np.std(x))
        return slope
    
    def _assess_stationarity(self, values: np.ndarray) -> float:
        """Assess stationarity of time series (0-1 score)."""
        if len(values) < 10:
            return 0.5
        
        # Simple rolling statistics approach
        window_size = min(len(values) // 4, 20)
        rolling_mean = pd.Series(values).rolling(window_size).mean()
        rolling_std = pd.Series(values).rolling(window_size).std()
        
        # Check stability of rolling statistics
        mean_stability = 1.0 - np.std(rolling_mean.dropna()) / np.mean(rolling_mean.dropna())
        std_stability = 1.0 - np.std(rolling_std.dropna()) / np.mean(rolling_std.dropna())
        
        return np.mean([mean_stability, std_stability])
    
    def _calculate_autocorrelation(self, values: np.ndarray) -> Optional[np.ndarray]:
        """Calculate autocorrelation function."""
        try:
            n = len(values)
            max_lags = min(n // 4, 40)  # Up to 40 lags or 1/4 of data
            
            # Center the data
            centered_values = values - np.mean(values)
            
            # Calculate autocorrelation
            acf = np.correlate(centered_values, centered_values, mode='full')
            acf = acf[n-1:]  # Take positive lags
            acf = acf / acf[0]  # Normalize
            
            return acf[:max_lags]
        except Exception:
            return None
    
    def _find_significant_lags(self, acf_values: np.ndarray, significance_level: float = 0.05) -> List[int]:
        """Find statistically significant autocorrelation lags."""
        n = len(acf_values)
        threshold = 1.96 / np.sqrt(n)  # 95% confidence bounds
        
        significant_lags = []
        for i, acf_val in enumerate(acf_values[1:], 1):  # Skip lag 0
            if abs(acf_val) > threshold:
                significant_lags.append(i)
        
        return significant_lags
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of time series."""
        try:
            from scipy.stats import skew
            return skew(values)
        except Exception:
            # Manual calculation
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0.0
            return np.mean(((values - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of time series."""
        try:
            from scipy.stats import kurtosis
            return kurtosis(values, fisher=True)  # Excess kurtosis
        except Exception:
            # Manual calculation
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0.0
            return np.mean(((values - mean_val) / std_val) ** 4) - 3
    
    def _calculate_component_strength(self, component: Optional[Any]) -> float:
        """Calculate strength of a time series component."""
        if component is None:
            return 0.0
        
        component_array = np.array(component)
        if len(component_array) == 0:
            return 0.0
        
        return np.std(component_array) / (np.std(component_array) + 1e-8)  # Avoid division by zero
    
    def _create_lagged_features(self, values: np.ndarray, n_lags: int) -> np.ndarray:
        """Create lagged features for regression."""
        n = len(values)
        lagged_features = np.zeros((n - n_lags, n_lags))
        
        for i in range(n_lags):
            lagged_features[:, i] = values[n_lags-i-1:n-i-1]
        
        return lagged_features
    
    def _create_trend_features(self, n_points: int) -> np.ndarray:
        """Create trend features for regression."""
        trend = np.arange(n_points)
        trend_squared = trend ** 2
        
        # Normalize
        trend = (trend - np.mean(trend)) / np.std(trend)
        trend_squared = (trend_squared - np.mean(trend_squared)) / np.std(trend_squared)
        
        return np.column_stack([trend, trend_squared])
    
    def _calculate_prediction_errors(self, actual: Any, predicted: Any) -> Dict[str, float]:
        """Calculate prediction error metrics."""
        try:
            actual_array = np.array(actual)
            predicted_array = np.array(predicted)
            
            if len(actual_array) != len(predicted_array):
                min_len = min(len(actual_array), len(predicted_array))
                actual_array = actual_array[:min_len]
                predicted_array = predicted_array[:min_len]
            
            errors = actual_array - predicted_array
            
            return {
                'mae': np.mean(np.abs(errors)),
                'mse': np.mean(errors ** 2),
                'rmse': np.sqrt(np.mean(errors ** 2)),
                'mape': np.mean(np.abs(errors / (actual_array + 1e-8))) * 100
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_temporal_features(self, values: np.ndarray) -> Dict[str, float]:
        """Extract temporal features for pattern recognition."""
        features = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'trend_slope': self._calculate_trend_slope(values),
            'skewness': self._calculate_skewness(values),
            'kurtosis': self._calculate_kurtosis(values)
        }
        
        # Add change point features
        change_points = self._detect_change_points(values)
        features['n_change_points'] = len(change_points)
        features['change_point_strength'] = np.mean([abs(cp) for cp in change_points]) if change_points else 0.0
        
        return features
    
    def _detect_patterns(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect patterns in time series."""
        pattern_info = {}
        
        # Simple seasonal pattern detection
        if len(values) >= 12:  # Need at least one potential cycle
            seasonal_detected = self._detect_seasonality(values)
            pattern_info['seasonal_detected'] = seasonal_detected
            
            if seasonal_detected:
                pattern_info['estimated_period'] = self._estimate_seasonal_period(values)
        
        # Trend detection
        trend_slope = self._calculate_trend_slope(values)
        pattern_info['trend_detected'] = abs(trend_slope) > 0.1
        pattern_info['trend_direction'] = 'increasing' if trend_slope > 0 else 'decreasing'
        
        return pattern_info
    
    def _detect_seasonality(self, values: np.ndarray) -> bool:
        """Simple seasonality detection."""
        try:
            # Test common seasonal periods
            for period in [4, 7, 12, 24]:  # Quarterly, weekly, monthly, daily patterns
                if len(values) < 2 * period:
                    continue
                
                # Calculate autocorrelation at seasonal lag
                acf = self._calculate_autocorrelation(values)
                if acf is not None and len(acf) > period:
                    if abs(acf[period]) > 0.3:  # Threshold for seasonal correlation
                        return True
            
            return False
        except Exception:
            return False
    
    def _estimate_seasonal_period(self, values: np.ndarray) -> int:
        """Estimate seasonal period."""
        best_period = 12  # Default
        max_correlation = 0.0
        
        for period in range(2, min(len(values) // 2, 50)):
            try:
                acf = self._calculate_autocorrelation(values)
                if acf is not None and len(acf) > period:
                    correlation = abs(acf[period])
                    if correlation > max_correlation:
                        max_correlation = correlation
                        best_period = period
            except Exception:
                continue
        
        return best_period
    
    def _detect_change_points(self, values: np.ndarray) -> List[float]:
        """Simple change point detection."""
        if len(values) < 4:
            return []
        
        change_points = []
        window_size = min(len(values) // 10, 20)
        
        for i in range(window_size, len(values) - window_size):
            left_mean = np.mean(values[i-window_size:i])
            right_mean = np.mean(values[i:i+window_size])
            
            change_magnitude = abs(right_mean - left_mean)
            if change_magnitude > 2 * np.std(values):
                change_points.append(change_magnitude)
        
        return change_points
    
    def _calculate_pattern_strength(self, pattern_data: Any) -> float:
        """Calculate strength of detected patterns."""
        if isinstance(pattern_data, (list, np.ndarray)):
            pattern_array = np.array(pattern_data)
            return np.std(pattern_array) / (np.mean(np.abs(pattern_array)) + 1e-8)
        return 0.5
    
    def _estimate_pattern_frequency(self, pattern_data: Any) -> int:
        """Estimate frequency of patterns."""
        if isinstance(pattern_data, (list, np.ndarray)):
            return len(pattern_data)
        return 1
    
    def _normalize_results(self, data: Any, mapping: DomainMapping, 
                          semantic_context: SemanticContext) -> Any:
        """Normalize time series conversion results for target domain."""
        if not isinstance(data, dict):
            return data
        
        # Add domain-specific metadata
        data['domain_conversion'] = {
            'source': 'time_series',
            'target': mapping.target_domain,
            'semantic_goal': semantic_context.analytical_goal,
            'quality_preservation': mapping.quality_preservation
        }
        
        # Ensure consistent structure based on target domain
        if mapping.target_domain == 'statistical':
            # Add time series specific statistical context
            data['temporal_statistics_info'] = {
                'analysis_type': 'time_series_statistical',
                'temporal_structure_preserved': True,
                'autocorrelation_available': 'autocorrelation_analysis' in data
            }
        
        elif mapping.target_domain == 'regression':
            # Add regression-ready metadata
            if 'feature_matrix' in data:
                data['regression_info'] = {
                    'feature_engineering_method': 'time_series_transformation',
                    'temporal_dependencies': True,
                    'lag_structure_preserved': True
                }
        
        return data


class PatternRecognitionShim(BaseDomainShim):
    """
    Bridge pattern recognition with other domains.
    
    Converts clustering results to statistical summaries, transforms dimensionality
    reduction outputs for visualization, and bridges classification results to business metrics.
    """
    
    def __init__(self, adapter_id: str = "pattern_recognition_shim", 
                 config: Optional[AdapterConfig] = None, **kwargs):
        """Initialize PatternRecognitionShim."""
        super().__init__(
            adapter_id=adapter_id,
            domain_type=DomainShimType.PATTERN_RECOGNITION,
            config=config,
            **kwargs
        )
    
    def _initialize_domain_knowledge(self) -> None:
        """Initialize pattern recognition domain knowledge."""
        self._domain_schemas = {
            'clustering_result': {
                'required_fields': ['cluster_labels', 'centroids'],
                'optional_fields': ['silhouette_scores', 'inertia', 'n_clusters']
            },
            'classification_result': {
                'required_fields': ['predictions', 'classes'],
                'optional_fields': ['probabilities', 'confidence_scores', 'feature_importance']
            },
            'dimensionality_reduction': {
                'required_fields': ['transformed_data', 'n_components'],
                'optional_fields': ['explained_variance_ratio', 'components', 'reconstruction_error']
            }
        }
    
    def _load_domain_mappings(self) -> None:
        """Load pattern recognition domain mappings."""
        # Pattern Recognition to Statistical mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='pattern_recognition',
            target_domain='statistical',
            parameter_mappings={
                'cluster_centroids': 'group_means',
                'silhouette_scores': 'cluster_validity',
                'feature_importance': 'variable_importance',
                'classification_accuracy': 'test_statistics'
            },
            result_transformations={
                'clustering_summary': 'group_statistics',
                'classification_metrics': 'hypothesis_test_results',
                'dimensionality_reduction': 'principal_component_analysis'
            },
            semantic_hints={
                'cluster_analysis': 'group_comparison',
                'feature_selection': 'variable_selection',
                'anomaly_detection': 'outlier_analysis'
            },
            quality_preservation=0.91
        ))
        
        # Pattern Recognition to Regression mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='pattern_recognition',
            target_domain='regression',
            parameter_mappings={
                'feature_weights': 'regression_coefficients',
                'transformed_features': 'predictor_variables',
                'anomaly_scores': 'leverage_scores',
                'cluster_assignments': 'categorical_predictors'
            },
            result_transformations={
                'pattern_features': 'regression_features',
                'classification_scores': 'continuous_predictors'
            },
            semantic_hints={
                'feature_extraction': 'feature_engineering',
                'dimensionality_reduction': 'predictor_reduction',
                'clustering': 'categorical_encoding'
            },
            quality_preservation=0.88
        ))
        
        # Pattern Recognition to Time Series mapping
        self.supported_mappings.append(DomainMapping(
            source_domain='pattern_recognition',
            target_domain='time_series',
            parameter_mappings={
                'sequential_patterns': 'temporal_patterns',
                'change_points': 'structural_breaks',
                'pattern_frequency': 'seasonal_components',
                'anomaly_detection': 'outlier_identification'
            },
            result_transformations={
                'pattern_sequences': 'time_series_segments',
                'clustering_temporal': 'regime_identification'
            },
            semantic_hints={
                'sequence_mining': 'pattern_analysis',
                'temporal_clustering': 'regime_detection',
                'change_detection': 'structural_break_analysis'
            },
            quality_preservation=0.85
        ))
    
    def _perform_domain_conversion(self, request: ConversionRequest, 
                                  mapping: DomainMapping, 
                                  semantic_context: SemanticContext) -> Any:
        """Perform pattern recognition domain conversion."""
        source_data = request.source_data
        
        if mapping.target_domain == 'statistical':
            return self._convert_pattern_recognition_to_statistical(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == 'regression':
            return self._convert_pattern_recognition_to_regression(
                source_data, mapping, semantic_context
            )
        elif mapping.target_domain == 'time_series':
            return self._convert_pattern_recognition_to_time_series(
                source_data, mapping, semantic_context
            )
        else:
            raise ConversionError(
                ConversionError.Type.CONVERSION_FAILED,
                f"Unsupported target domain: {mapping.target_domain}"
            )
    
    def _convert_pattern_recognition_to_statistical(self, data: Any, mapping: DomainMapping,
                                                  semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert pattern recognition results to statistical format."""
        if isinstance(data, dict):
            # Handle clustering results
            if 'cluster_labels' in data and 'centroids' in data:
                cluster_labels = np.array(data['cluster_labels'])
                centroids = np.array(data['centroids'])
                
                # Calculate cluster statistics
                unique_labels = np.unique(cluster_labels)
                cluster_stats = {}
                
                for label in unique_labels:
                    cluster_mask = cluster_labels == label
                    cluster_stats[f'cluster_{label}'] = {
                        'size': np.sum(cluster_mask),
                        'proportion': np.mean(cluster_mask),
                        'centroid': centroids[label].tolist() if label < len(centroids) else None
                    }
                
                return {
                    'group_statistics': cluster_stats,
                    'clustering_validity': {
                        'n_clusters': len(unique_labels),
                        'silhouette_score': data.get('silhouette_score'),
                        'inertia': data.get('inertia'),
                        'cluster_sizes': [stats['size'] for stats in cluster_stats.values()]
                    }
                }
            
            # Handle classification results
            if 'predictions' in data and 'probabilities' in data:
                predictions = np.array(data['predictions'])
                probabilities = np.array(data['probabilities'])
                
                # Calculate classification statistics
                unique_classes = np.unique(predictions)
                class_stats = {}
                
                for cls in unique_classes:
                    class_mask = predictions == cls
                    class_probs = probabilities[class_mask] if len(probabilities.shape) == 1 else probabilities[class_mask, cls]
                    
                    class_stats[f'class_{cls}'] = {
                        'frequency': np.sum(class_mask),
                        'proportion': np.mean(class_mask),
                        'avg_confidence': np.mean(class_probs) if len(class_probs) > 0 else 0.0,
                        'confidence_std': np.std(class_probs) if len(class_probs) > 0 else 0.0
                    }
                
                return {
                    'classification_statistics': class_stats,
                    'prediction_confidence': {
                        'overall_confidence': np.mean(np.max(probabilities, axis=1)) if len(probabilities.shape) > 1 else np.mean(probabilities),
                        'uncertainty_measure': 1.0 - np.mean(np.max(probabilities, axis=1)) if len(probabilities.shape) > 1 else 1.0 - np.mean(probabilities)
                    }
                }
            
            # Handle dimensionality reduction results
            if 'transformed_data' in data and 'explained_variance_ratio' in data:
                return {
                    'principal_component_analysis': {
                        'explained_variance_ratio': data['explained_variance_ratio'],
                        'cumulative_variance': np.cumsum(data['explained_variance_ratio']).tolist(),
                        'n_components': data.get('n_components', len(data['explained_variance_ratio'])),
                        'dimensionality_reduction_ratio': data.get('n_components', 0) / data.get('original_dimensions', 1)
                    }
                }
        
        # Fallback
        return {
            'pattern_recognition_input': data,
            'conversion_type': 'pattern_recognition_to_statistical',
            'statistical_context': semantic_context.transformation_hints
        }
    
    def _convert_pattern_recognition_to_regression(self, data: Any, mapping: DomainMapping,
                                                 semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert pattern recognition results to regression format."""
        if isinstance(data, dict):
            # Handle feature importance from classification/clustering
            if 'feature_importance' in data or 'feature_weights' in data:
                importance = data.get('feature_importance', data.get('feature_weights'))
                
                return {
                    'feature_selection': {
                        'feature_importance': importance,
                        'feature_names': data.get('feature_names', []),
                        'selection_method': 'pattern_recognition_based',
                        'importance_threshold': np.percentile(importance, 75) if isinstance(importance, (list, np.ndarray)) else 0.5
                    }
                }
            
            # Handle transformed features from dimensionality reduction
            if 'transformed_data' in data:
                transformed_data = np.array(data['transformed_data'])
                
                return {
                    'predictor_variables': {
                        'transformed_features': transformed_data,
                        'n_components': transformed_data.shape[1] if len(transformed_data.shape) > 1 else 1,
                        'transformation_method': data.get('method', 'dimensionality_reduction'),
                        'explained_variance': data.get('explained_variance_ratio')
                    }
                }
            
            # Handle cluster assignments as categorical predictors
            if 'cluster_labels' in data:
                cluster_labels = np.array(data['cluster_labels'])
                
                # Create dummy variables for clusters
                unique_clusters = np.unique(cluster_labels)
                dummy_matrix = np.zeros((len(cluster_labels), len(unique_clusters)))
                
                for i, cluster in enumerate(unique_clusters):
                    dummy_matrix[:, i] = (cluster_labels == cluster).astype(int)
                
                return {
                    'categorical_predictors': {
                        'cluster_dummies': dummy_matrix,
                        'cluster_labels': cluster_labels,
                        'n_clusters': len(unique_clusters),
                        'cluster_names': [f'cluster_{c}' for c in unique_clusters]
                    }
                }
        
        # Fallback
        return {
            'pattern_recognition_input': data,
            'conversion_type': 'pattern_recognition_to_regression',
            'regression_context': semantic_context.transformation_hints
        }
    
    def _convert_pattern_recognition_to_time_series(self, data: Any, mapping: DomainMapping,
                                                  semantic_context: SemanticContext) -> Dict[str, Any]:
        """Convert pattern recognition results to time series format."""
        if isinstance(data, dict):
            # Handle sequential patterns
            if 'sequential_patterns' in data or 'temporal_patterns' in data:
                patterns = data.get('sequential_patterns', data.get('temporal_patterns'))
                
                return {
                    'pattern_analysis': {
                        'detected_patterns': patterns,
                        'pattern_frequency': self._calculate_pattern_frequency(patterns),
                        'pattern_strength': self._calculate_pattern_strength(patterns),
                        'temporal_structure': 'sequential'
                    }
                }
            
            # Handle change point detection
            if 'change_points' in data or 'anomaly_scores' in data:
                change_points = data.get('change_points', [])
                anomaly_scores = data.get('anomaly_scores', [])
                
                return {
                    'structural_break_analysis': {
                        'change_points': change_points,
                        'anomaly_scores': anomaly_scores,
                        'n_change_points': len(change_points),
                        'break_detection_method': 'pattern_recognition_based'
                    }
                }
            
            # Handle temporal clustering
            if 'cluster_labels' in data and 'temporal_info' in data:
                cluster_labels = data['cluster_labels']
                temporal_info = data['temporal_info']
                
                # Identify regime changes based on cluster transitions
                regime_changes = []
                if len(cluster_labels) > 1:
                    for i in range(1, len(cluster_labels)):
                        if cluster_labels[i] != cluster_labels[i-1]:
                            regime_changes.append(i)
                
                return {
                    'regime_identification': {
                        'regime_labels': cluster_labels,
                        'regime_changes': regime_changes,
                        'n_regimes': len(np.unique(cluster_labels)),
                        'temporal_index': temporal_info.get('timestamps', list(range(len(cluster_labels))))
                    }
                }
        
        elif isinstance(data, pd.DataFrame):
            # Handle DataFrame with temporal patterns
            if 'timestamp' in data.columns or hasattr(data.index, 'freq'):
                # Extract pattern features for time series
                pattern_features = {}
                
                for col in data.columns:
                    if col != 'timestamp':
                        values = data[col].values
                        pattern_features[col] = {
                            'pattern_strength': np.std(values),
                            'trend': self._calculate_trend_slope(values),
                            'volatility': self._calculate_volatility(values)
                        }
                
                return {
                    'temporal_pattern_features': pattern_features,
                    'time_index': data.index.tolist() if hasattr(data, 'index') else None
                }
        
        # Fallback
        return {
            'pattern_recognition_input': data,
            'conversion_type': 'pattern_recognition_to_time_series',
            'temporal_context': semantic_context.transformation_hints
        }
    
    def _calculate_pattern_frequency(self, patterns: Any) -> Dict[str, int]:
        """Calculate frequency of detected patterns."""
        if isinstance(patterns, (list, np.ndarray)):
            pattern_array = np.array(patterns)
            unique, counts = np.unique(pattern_array, return_counts=True)
            return {f'pattern_{i}': count for i, count in enumerate(counts)}
        return {}
    
    def _calculate_pattern_strength(self, patterns: Any) -> float:
        """Calculate strength of detected patterns."""
        if isinstance(patterns, (list, np.ndarray)):
            pattern_array = np.array(patterns)
            return np.std(pattern_array) / (np.mean(np.abs(pattern_array)) + 1e-8)
        return 0.5
    
    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate trend slope."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        return np.corrcoef(x, values)[0, 1] * (np.std(values) / np.std(x))
    
    def _calculate_volatility(self, values: np.ndarray) -> float:
        """Calculate volatility measure."""
        if len(values) < 2:
            return 0.0
        
        returns = np.diff(values) / (values[:-1] + 1e-8)
        return np.std(returns)
    
    def _normalize_results(self, data: Any, mapping: DomainMapping, 
                          semantic_context: SemanticContext) -> Any:
        """Normalize pattern recognition conversion results for target domain."""
        if not isinstance(data, dict):
            return data
        
        # Add domain-specific metadata
        data['domain_conversion'] = {
            'source': 'pattern_recognition',
            'target': mapping.target_domain,
            'semantic_goal': semantic_context.analytical_goal,
            'quality_preservation': mapping.quality_preservation
        }
        
        # Ensure consistent structure based on target domain
        if mapping.target_domain == 'statistical':
            # Add pattern recognition specific statistical context
            data['pattern_statistics_info'] = {
                'analysis_type': 'pattern_recognition_statistical',
                'clustering_available': any('cluster' in key for key in data.keys()),
                'classification_available': any('classification' in key for key in data.keys()),
                'dimensionality_reduction_available': any('pca' in key.lower() or 'component' in key for key in data.keys())
            }
        
        elif mapping.target_domain == 'regression':
            # Add regression-ready metadata
            if 'feature_selection' in data or 'predictor_variables' in data:
                data['regression_info'] = {
                    'feature_engineering_method': 'pattern_recognition_transformation',
                    'dimensionality_reduced': 'transformed_features' in str(data),
                    'categorical_encoding_applied': 'cluster_dummies' in str(data)
                }
        
        return data


# Factory functions for creating domain shims

def create_statistical_shim(adapter_id: str = "statistical_shim",
                           config: Optional[AdapterConfig] = None,
                           **kwargs) -> StatisticalShim:
    """Create a StatisticalShim with optional configuration."""
    return StatisticalShim(adapter_id=adapter_id, config=config, **kwargs)


def create_regression_shim(adapter_id: str = "regression_shim",
                          config: Optional[AdapterConfig] = None,
                          **kwargs) -> RegressionShim:
    """Create a RegressionShim with optional configuration."""
    return RegressionShim(adapter_id=adapter_id, config=config, **kwargs)


def create_time_series_shim(adapter_id: str = "time_series_shim",
                           config: Optional[AdapterConfig] = None,
                           **kwargs) -> TimeSeriesShim:
    """Create a TimeSeriesShim with optional configuration."""
    return TimeSeriesShim(adapter_id=adapter_id, config=config, **kwargs)


def create_pattern_recognition_shim(adapter_id: str = "pattern_recognition_shim",
                                   config: Optional[AdapterConfig] = None,
                                   **kwargs) -> PatternRecognitionShim:
    """Create a PatternRecognitionShim with optional configuration."""
    return PatternRecognitionShim(adapter_id=adapter_id, config=config, **kwargs)


# Utility functions for domain shim management

def create_all_domain_shims(registry: Optional['ShimRegistry'] = None,
                           auto_register: bool = True) -> Dict[str, BaseDomainShim]:
    """
    Create all available domain shims.
    
    Args:
        registry: Optional registry to register shims with
        auto_register: Whether to automatically register shims with registry
        
    Returns:
        Dictionary mapping shim names to shim instances
    """
    shims = {
        'statistical': create_statistical_shim(),
        'regression': create_regression_shim(),
        'time_series': create_time_series_shim(),
        'pattern_recognition': create_pattern_recognition_shim()
    }
    
    if registry and auto_register:
        for shim_name, shim in shims.items():
            registry.register_adapter(shim)
            logger.info(f"Registered domain shim: {shim_name}")
    
    return shims


def get_compatible_domain_shims(source_domain: str, target_domain: str,
                               available_shims: Optional[Dict[str, BaseDomainShim]] = None) -> List[BaseDomainShim]:
    """
    Find compatible domain shims for a specific domain conversion.
    
    Args:
        source_domain: Source domain name
        target_domain: Target domain name
        available_shims: Optional dictionary of available shims
        
    Returns:
        List of compatible domain shims
    """
    if available_shims is None:
        available_shims = create_all_domain_shims(auto_register=False)
    
    compatible_shims = []
    
    for shim in available_shims.values():
        for mapping in shim.supported_mappings:
            if (mapping.source_domain == source_domain and 
                mapping.target_domain == target_domain):
                compatible_shims.append(shim)
                break
    
    return compatible_shims


def validate_domain_shim_configuration(shims: Dict[str, BaseDomainShim]) -> ValidationResult:
    """
    Validate domain shim configuration for completeness and consistency.
    
    Args:
        shims: Dictionary of domain shims to validate
        
    Returns:
        Validation result with any configuration issues
    """
    errors = []
    warnings = []
    
    expected_shims = {'statistical', 'regression', 'time_series', 'pattern_recognition'}
    available_shims = set(shims.keys())
    
    # Check for missing shims
    missing_shims = expected_shims - available_shims
    if missing_shims:
        warnings.append(f"Missing domain shims: {list(missing_shims)}")
    
    # Check shim configurations
    for shim_name, shim in shims.items():
        if not shim.supported_mappings:
            errors.append(f"Domain shim '{shim_name}' has no supported mappings")
        
        # Check for bidirectional mappings
        source_domains = {m.source_domain for m in shim.supported_mappings}
        target_domains = {m.target_domain for m in shim.supported_mappings}
        
        if len(source_domains) == 1 and len(target_domains) == 1:
            warnings.append(f"Domain shim '{shim_name}' only supports unidirectional conversion")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        score=1.0 - len(errors) * 0.2 - len(warnings) * 0.1,
        errors=errors,
        warnings=warnings,
        details={
            'total_shims': len(shims),
            'expected_shims': len(expected_shims),
            'available_shims': list(available_shims),
            'missing_shims': list(missing_shims)
        }
    )