"""
Domain Adapter System - Non-sklearn Library Integration

This module implements the DomainAdapter base class that provides standardized
interface for integrating external libraries (scipy, statsmodels, etc.) with 
the sklearn-based pipeline architecture.

Key Features:
- Abstract interface for wrapping non-sklearn libraries
- Automatic sklearn-compatible API generation
- Metadata preservation across library boundaries
- Error handling and graceful degradation when libraries unavailable
- Progressive disclosure parameter mapping
- Integration with composition metadata system
"""

import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
from dataclasses import dataclass, field

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
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


@dataclass
class LibraryIntegrationConfig:
    """Configuration for external library integration."""
    
    library_name: str
    required_version: Optional[str] = None
    import_path: str = ""
    fallback_available: bool = True
    graceful_degradation: bool = True
    progressive_complexity: bool = True
    streaming_compatible: bool = False
    metadata_preservation: bool = True
    
    # Integration parameters
    parameter_mappings: Dict[str, Any] = field(default_factory=dict)
    output_transformers: List[Callable] = field(default_factory=list)
    error_handlers: Dict[str, Callable] = field(default_factory=dict)


@dataclass
class AdapterMetadata:
    """Metadata for domain adapter operations."""
    
    adapter_type: str
    library_name: str
    library_version: Optional[str] = None
    integration_successful: bool = True
    fallback_used: bool = False
    
    # Execution metrics
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Parameters and configuration
    original_parameters: Dict[str, Any] = field(default_factory=dict)
    mapped_parameters: Dict[str, Any] = field(default_factory=dict)
    sklearn_equivalent: Optional[str] = None
    
    # Quality and compatibility
    compatibility_score: float = 1.0
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings_generated: List[str] = field(default_factory=list)


class DomainAdapter(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for integrating non-sklearn libraries with the pipeline architecture.
    
    This adapter provides a standardized interface for wrapping external libraries
    while maintaining sklearn compatibility and enabling composition with other
    pipeline components.
    
    First Principle: Modular Domain Integration
    - Each domain (scipy.stats, statsmodels, etc.) gets its own adapter
    - Adapters provide sklearn-compatible interface while preserving library-specific features
    - Graceful degradation when libraries are unavailable
    """
    
    def __init__(self,
                 library_config: LibraryIntegrationConfig,
                 analytical_intention: str = "integrate external library",
                 progressive_complexity: str = "auto",
                 streaming_config: Optional[StreamingConfig] = None,
                 custom_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize domain adapter.
        
        Args:
            library_config: Configuration for library integration
            analytical_intention: Natural language description of adapter purpose
            progressive_complexity: Complexity level for parameter mapping
            streaming_config: Configuration for streaming support
            custom_parameters: Additional custom parameters
        """
        self.library_config = library_config
        self.analytical_intention = analytical_intention
        self.progressive_complexity = progressive_complexity
        self.streaming_config = streaming_config or StreamingConfig()
        self.custom_parameters = custom_parameters or {}
        
        # Adapter state
        self._library_available = False
        self._library_module = None
        self._adapter_metadata = AdapterMetadata(
            adapter_type=self.__class__.__name__,
            library_name=library_config.library_name
        )
        self._fitted_state = {}
        
        # Initialize library integration
        self._initialize_library_integration()
        
        logger.info("DomainAdapter initialized",
                   adapter_type=self.__class__.__name__,
                   library=library_config.library_name,
                   available=self._library_available)
    
    def _initialize_library_integration(self):
        """Initialize integration with external library."""
        try:
            # Import the library
            self._library_module = self._import_library()
            self._library_available = True
            
            # Get version if available
            if hasattr(self._library_module, '__version__'):
                self._adapter_metadata.library_version = self._library_module.__version__
            
            logger.info(f"Successfully integrated {self.library_config.library_name}",
                       version=self._adapter_metadata.library_version)
            
        except ImportError as e:
            self._library_available = False
            self._adapter_metadata.integration_successful = False
            
            if self.library_config.graceful_degradation:
                logger.warning(f"Library {self.library_config.library_name} not available, "
                             f"will use fallback: {e}")
            else:
                raise PipelineError(
                    f"Required library {self.library_config.library_name} not available: {e}",
                    ErrorClassification.CONFIGURATION_ERROR
                )
    
    @abstractmethod
    def _import_library(self) -> Any:
        """
        Import the external library.
        
        Returns:
            The imported library module
        """
        pass
    
    @abstractmethod
    def _create_library_instance(self, **kwargs) -> Any:
        """
        Create instance of the external library's main class/function.
        
        Args:
            **kwargs: Parameters for library instantiation
            
        Returns:
            Instance of library class or prepared function
        """
        pass
    
    @abstractmethod
    def _map_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Map sklearn-style parameters to library-specific parameters.
        
        Args:
            **kwargs: sklearn-style parameters
            
        Returns:
            Dictionary of library-specific parameters
        """
        pass
    
    @abstractmethod
    def _transform_output(self, library_output: Any) -> Any:
        """
        Transform library output to sklearn-compatible format.
        
        Args:
            library_output: Raw output from external library
            
        Returns:
            sklearn-compatible output (typically DataFrame or ndarray)
        """
        pass
    
    def fit(self, X, y=None, **fit_params):
        """
        Fit the adapter using external library.
        
        Args:
            X: Input data
            y: Target data (optional)
            **fit_params: Additional fitting parameters
            
        Returns:
            self
        """
        start_time = time.time()
        
        try:
            if not self._library_available:
                return self._fallback_fit(X, y, **fit_params)
            
            # Map parameters
            mapped_params = self._map_parameters(**fit_params)
            self._adapter_metadata.original_parameters = fit_params
            self._adapter_metadata.mapped_parameters = mapped_params
            
            # Create and fit library instance
            library_instance = self._create_library_instance(**mapped_params)
            
            # Fit using library-specific method
            fitted_instance = self._fit_library_instance(library_instance, X, y, **mapped_params)
            
            # Store fitted state
            self._fitted_state = {
                'library_instance': fitted_instance,
                'fit_parameters': mapped_params,
                'input_shape': X.shape if hasattr(X, 'shape') else None
            }
            
            execution_time = time.time() - start_time
            self._adapter_metadata.execution_time = execution_time
            
            logger.info(f"Adapter {self.__class__.__name__} fitted successfully",
                       execution_time=execution_time)
            
            return self
            
        except Exception as e:
            if self.library_config.graceful_degradation:
                logger.warning(f"Library fitting failed, using fallback: {e}")
                self._adapter_metadata.fallback_used = True
                return self._fallback_fit(X, y, **fit_params)
            else:
                raise PipelineError(
                    f"Adapter {self.__class__.__name__} fitting failed: {e}",
                    ErrorClassification.EXECUTION_ERROR
                )
    
    def transform(self, X):
        """
        Transform data using external library.
        
        Args:
            X: Input data to transform
            
        Returns:
            Transformed data in sklearn-compatible format
        """
        start_time = time.time()
        
        try:
            if not self._library_available or self._adapter_metadata.fallback_used:
                return self._fallback_transform(X)
            
            # Use fitted library instance to transform
            library_instance = self._fitted_state.get('library_instance')
            if library_instance is None:
                raise PipelineError(
                    "Adapter not fitted. Call fit() first.",
                    ErrorClassification.USAGE_ERROR
                )
            
            # Transform using library-specific method
            library_output = self._transform_library_instance(library_instance, X)
            
            # Transform output to sklearn-compatible format
            sklearn_output = self._transform_output(library_output)
            
            execution_time = time.time() - start_time
            
            logger.info(f"Adapter {self.__class__.__name__} transform completed",
                       execution_time=execution_time)
            
            return sklearn_output
            
        except Exception as e:
            if self.library_config.graceful_degradation:
                logger.warning(f"Library transform failed, using fallback: {e}")
                self._adapter_metadata.fallback_used = True
                return self._fallback_transform(X)
            else:
                raise PipelineError(
                    f"Adapter {self.__class__.__name__} transform failed: {e}",
                    ErrorClassification.EXECUTION_ERROR
                )
    
    def _fit_library_instance(self, library_instance: Any, X, y=None, **params) -> Any:
        """
        Fit library instance with data.
        
        Args:
            library_instance: Instance from _create_library_instance
            X: Input data
            y: Target data (optional)
            **params: Additional parameters
            
        Returns:
            Fitted library instance
        """
        # Default implementation - many libraries auto-fit or don't need explicit fit
        return library_instance
    
    def _transform_library_instance(self, library_instance: Any, X) -> Any:
        """
        Transform data using fitted library instance.
        
        Args:
            library_instance: Fitted library instance
            X: Input data
            
        Returns:
            Library-specific output
        """
        # Default implementation - call library instance directly
        if callable(library_instance):
            return library_instance(X)
        else:
            # Try common method names
            for method_name in ['transform', 'predict', 'apply', 'run']:
                if hasattr(library_instance, method_name):
                    method = getattr(library_instance, method_name)
                    if callable(method):
                        return method(X)
            
            raise PipelineError(
                f"Don't know how to transform data with {type(library_instance)}",
                ErrorClassification.INTEGRATION_ERROR
            )
    
    def _fallback_fit(self, X, y=None, **fit_params):
        """Fallback fitting when external library unavailable."""
        logger.info(f"Using fallback fit for {self.__class__.__name__}")
        # Default: no-op fit
        return self
    
    def _fallback_transform(self, X):
        """Fallback transformation when external library unavailable."""
        logger.info(f"Using fallback transform for {self.__class__.__name__}")
        # Default: return input unchanged
        return X
    
    # sklearn compatibility methods
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            return None
        # Default: preserve input feature names
        return input_features
    
    def get_params(self, deep=True):
        """Get parameters for this adapter."""
        params = {
            'library_config': self.library_config,
            'analytical_intention': self.analytical_intention,
            'progressive_complexity': self.progressive_complexity,
            'streaming_config': self.streaming_config,
            'custom_parameters': self.custom_parameters
        }
        return params
    
    def set_params(self, **params):
        """Set parameters for this adapter."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        return self
    
    # Metadata and introspection methods
    def get_adapter_metadata(self) -> AdapterMetadata:
        """Get comprehensive adapter metadata."""
        return self._adapter_metadata
    
    def is_library_available(self) -> bool:
        """Check if external library is available."""
        return self._library_available
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get information about integrated library."""
        return {
            'name': self.library_config.library_name,
            'version': self._adapter_metadata.library_version,
            'available': self._library_available,
            'fallback_used': self._adapter_metadata.fallback_used,
            'import_path': self.library_config.import_path
        }
    
    def get_sklearn_equivalent(self) -> Optional[str]:
        """Get equivalent sklearn class/function if available."""
        return self._adapter_metadata.sklearn_equivalent
    
    # Composition integration methods
    def get_composition_metadata(self) -> Dict[str, Any]:
        """Get metadata for pipeline composition."""
        return {
            "adapter_info": {
                "adapter_type": self.__class__.__name__,
                "library_name": self.library_config.library_name,
                "library_version": self._adapter_metadata.library_version,
                "analytical_intention": self.analytical_intention
            },
            "integration_status": {
                "library_available": self._library_available,
                "integration_successful": self._adapter_metadata.integration_successful,
                "fallback_used": self._adapter_metadata.fallback_used,
                "compatibility_score": self._adapter_metadata.compatibility_score
            },
            "capabilities": {
                "streaming_compatible": self.library_config.streaming_compatible,
                "progressive_complexity": self.library_config.progressive_complexity,
                "metadata_preservation": self.library_config.metadata_preservation
            },
            "execution_metrics": {
                "last_execution_time": self._adapter_metadata.execution_time,
                "memory_usage_mb": self._adapter_metadata.memory_usage_mb,
                "warnings_count": len(self._adapter_metadata.warnings_generated)
            },
            "composition_context": {
                "ready_for_chaining": self._library_available or self.library_config.fallback_available,
                "suggested_next_steps": self._generate_composition_suggestions(),
                "integration_artifacts": {
                    "parameter_mappings": self._adapter_metadata.mapped_parameters,
                    "sklearn_equivalent": self._adapter_metadata.sklearn_equivalent
                }
            }
        }
    
    def _generate_composition_suggestions(self) -> List[Dict[str, Any]]:
        """Generate suggestions for pipeline composition."""
        suggestions = []
        
        if self._library_available:
            suggestions.append({
                "analysis_type": "domain_specific_analysis",
                "reason": f"{self.library_config.library_name} successfully integrated",
                "confidence": 0.9,
                "next_adapter": None
            })
        
        if self.library_config.fallback_available:
            suggestions.append({
                "analysis_type": "fallback_analysis",
                "reason": "Fallback methods available if library fails",
                "confidence": 0.6,
                "next_adapter": "sklearn_equivalent"
            })
        
        return suggestions


class SciPyStatsAdapter(DomainAdapter):
    """
    Adapter for scipy.stats statistical functions.
    
    Integrates scipy.stats functions with the pipeline architecture while
    maintaining sklearn compatibility and providing progressive disclosure.
    """
    
    def __init__(self, 
                 stat_function: str = "describe",
                 **kwargs):
        """
        Initialize scipy.stats adapter.
        
        Args:
            stat_function: Name of scipy.stats function to use
            **kwargs: Additional parameters for DomainAdapter
        """
        library_config = LibraryIntegrationConfig(
            library_name="scipy.stats",
            import_path="scipy.stats",
            fallback_available=True,
            graceful_degradation=True,
            progressive_complexity=True,
            streaming_compatible=True,
            metadata_preservation=True
        )
        
        super().__init__(library_config, **kwargs)
        self.stat_function = stat_function
        self._adapter_metadata.sklearn_equivalent = "None (statistical analysis)"
    
    def _import_library(self):
        """Import scipy.stats."""
        import scipy.stats
        return scipy.stats
    
    def _create_library_instance(self, **kwargs):
        """Create scipy.stats function instance."""
        stat_func = getattr(self._library_module, self.stat_function)
        return stat_func
    
    def _map_parameters(self, **kwargs):
        """Map parameters for scipy.stats functions."""
        # Most scipy.stats functions don't need parameter mapping
        return kwargs
    
    def _transform_output(self, library_output):
        """Transform scipy.stats output to DataFrame format."""
        if hasattr(library_output, '_asdict'):
            # Named tuple output (like describe)
            result_dict = library_output._asdict()
            return pd.DataFrame([result_dict])
        elif isinstance(library_output, (list, tuple)):
            # Multiple values
            return pd.DataFrame({'values': library_output})
        else:
            # Single value
            return pd.DataFrame({'result': [library_output]})
    
    def _fallback_fit(self, X, y=None, **fit_params):
        """Fallback using basic pandas/numpy statistics."""
        logger.info("Using pandas/numpy fallback for scipy.stats")
        return self
    
    def _fallback_transform(self, X):
        """Fallback transformation using basic statistics."""
        if isinstance(X, pd.DataFrame):
            # Basic descriptive statistics
            return X.describe().T.reset_index()
        else:
            # Convert to DataFrame and describe
            df = pd.DataFrame(X)
            return df.describe().T.reset_index()


# Utility functions for adapter management
def create_domain_adapter(library_name: str, 
                         adapter_class: Optional[Type[DomainAdapter]] = None,
                         **kwargs) -> DomainAdapter:
    """
    Factory function to create domain adapters.
    
    Args:
        library_name: Name of the library to integrate
        adapter_class: Specific adapter class to use
        **kwargs: Additional parameters for adapter
        
    Returns:
        Configured domain adapter instance
    """
    # Registry of available adapters
    adapter_registry = {
        'scipy.stats': SciPyStatsAdapter,
        # Future adapters can be added here
        # 'statsmodels': StatsmodelsAdapter,
        # 'sklearn_extra': SklearnExtraAdapter,
    }
    
    if adapter_class is not None:
        return adapter_class(**kwargs)
    elif library_name in adapter_registry:
        adapter_cls = adapter_registry[library_name]
        return adapter_cls(**kwargs)
    else:
        raise ValueError(f"No adapter available for library: {library_name}")


def list_available_adapters() -> Dict[str, Dict[str, Any]]:
    """
    List all available domain adapters.
    
    Returns:
        Dictionary with adapter information
    """
    adapters = {
        'scipy.stats': {
            'class': 'SciPyStatsAdapter',
            'description': 'Statistical functions from scipy.stats',
            'streaming_compatible': True,
            'fallback_available': True
        }
        # Future adapters would be listed here
    }
    
    return adapters


def check_library_availability(library_name: str) -> Dict[str, Any]:
    """
    Check if a library is available for integration.
    
    Args:
        library_name: Name of the library to check
        
    Returns:
        Dictionary with availability information
    """
    try:
        if library_name == 'scipy.stats':
            import scipy.stats
            version = getattr(scipy.stats, '__version__', 'unknown')
        else:
            # Generic import attempt
            __import__(library_name)
            module = __import__(library_name)
            version = getattr(module, '__version__', 'unknown')
        
        return {
            'available': True,
            'version': version,
            'import_successful': True
        }
        
    except ImportError as e:
        return {
            'available': False,
            'version': None,
            'import_successful': False,
            'error': str(e)
        }