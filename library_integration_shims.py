"""
Library Integration Shims - Non-sklearn Library Integration Architecture
LocalData MCP v2.0

This module provides integration adapters (shims) that enable non-sklearn libraries 
to participate seamlessly in the unified pipeline architecture while maintaining:
- Streaming compatibility for memory-constrained environments
- Graceful degradation when specialized libraries are unavailable  
- Rich metadata propagation for downstream tool composition
- Error handling with intelligent fallbacks

Design Principles:
1. Intention-Driven Interface - LLM agents express analytical intentions
2. Context-Aware Composition - Results designed for downstream composition
3. Progressive Disclosure - Simple by default, powerful when needed
4. Streaming-First - Memory constraints as architectural requirements
5. Modular Integration - Seamless cross-domain analytical workflows
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Protocol, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import importlib
import functools
import warnings
from datetime import datetime

# Import core pipeline components
from pipeline_composition_framework import (
    CompositionStage, 
    AnalysisComposition,
    ExecutionStrategy,
    ValidationResult,
    ConversionResult,
    CompositionError
)

logger = logging.getLogger(__name__)


# ============================================================================
# Library Integration Core Architecture
# ============================================================================

class LibraryCategory(Enum):
    """Categories of external libraries for integration patterns."""
    GEOSPATIAL = "geospatial"
    TIME_SERIES = "time_series"  
    STATISTICS = "statistics"
    OPTIMIZATION = "optimization"
    FUZZY_MATCHING = "fuzzy_matching"
    SURVIVAL_ANALYSIS = "survival_analysis"
    GRAPH_ANALYTICS = "graph_analytics"
    AUDIO_PROCESSING = "audio_processing"
    SPECIALIZED_DOMAIN = "specialized_domain"


class IntegrationStrategy(Enum):
    """Integration strategies for different library patterns."""
    SKLEARN_WRAPPER = "sklearn_wrapper"        # Wrap as sklearn transformer/estimator
    FUNCTION_ADAPTER = "function_adapter"      # Adapt function-based APIs  
    STREAMING_BUFFER = "streaming_buffer"      # Buffer data for batch libraries
    PROGRESSIVE_RESULTS = "progressive_results" # Process results progressively
    FALLBACK_CHAIN = "fallback_chain"          # Chain of fallback implementations


@dataclass
class LibraryDependency:
    """Represents a library dependency with fallback options."""
    name: str
    import_path: str
    min_version: Optional[str] = None
    fallback_libraries: List[str] = field(default_factory=list)
    sklearn_equivalent: Optional[str] = None
    is_optional: bool = True
    installation_hint: str = ""
    
    def __post_init__(self):
        if not self.installation_hint:
            self.installation_hint = f"pip install {self.name}"


@dataclass
class IntegrationMetadata:
    """Metadata for library integration results."""
    library_used: str
    library_version: Optional[str] = None
    integration_strategy: IntegrationStrategy = IntegrationStrategy.FUNCTION_ADAPTER
    data_transformations: List[str] = field(default_factory=list)
    performance_notes: List[str] = field(default_factory=list)
    streaming_compatible: bool = True
    fallback_used: bool = False
    original_parameters: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Base Library Adapter
# ============================================================================

class BaseLibraryAdapter(ABC):
    """
    Base class for all library integration adapters.
    
    Provides common functionality for:
    - Dependency management with graceful degradation
    - Data type conversion between library formats
    - Streaming compatibility through buffering
    - Error handling with fallback strategies
    - Metadata propagation for pipeline composition
    """
    
    def __init__(self, 
                 library_category: LibraryCategory,
                 dependencies: List[LibraryDependency]):
        self.category = library_category
        self.dependencies = dependencies
        self._imported_libraries = {}
        self._available_libraries = {}
        self._check_dependencies()
        
    @abstractmethod
    def get_supported_functions(self) -> Dict[str, Callable]:
        """Return dictionary of supported functions by name."""
        pass
    
    @abstractmethod 
    def adapt_function_call(self, 
                          function_name: str,
                          data: Any,
                          parameters: Dict[str, Any]) -> Tuple[Any, IntegrationMetadata]:
        """Adapt a function call to the target library's API."""
        pass
    
    def _check_dependencies(self):
        """Check which dependencies are available and import them."""
        for dep in self.dependencies:
            try:
                # Attempt to import the library
                module = importlib.import_module(dep.import_path)
                self._imported_libraries[dep.name] = module
                self._available_libraries[dep.name] = True
                
                # Check version if specified
                if dep.min_version and hasattr(module, '__version__'):
                    # Simplified version check - could be enhanced
                    self._available_libraries[f"{dep.name}_version"] = module.__version__
                    
                logger.info(f"Successfully imported {dep.name}")
                
            except ImportError as e:
                self._available_libraries[dep.name] = False
                logger.info(f"Optional library {dep.name} not available: {e}")
                
                if not dep.is_optional:
                    raise CompositionError(
                        f"Required library {dep.name} is not available. {dep.installation_hint}",
                        error_type="missing_dependency"
                    )
    
    def is_library_available(self, library_name: str) -> bool:
        """Check if a specific library is available."""
        return self._available_libraries.get(library_name, False)
    
    def get_library(self, library_name: str) -> Any:
        """Get imported library module."""
        if not self.is_library_available(library_name):
            raise CompositionError(
                f"Library {library_name} is not available",
                error_type="library_not_available"
            )
        return self._imported_libraries[library_name]
    
    def convert_input_data(self, data: Any, target_format: str) -> Tuple[Any, List[str]]:
        """Convert input data to target library's expected format."""
        transformations = []
        
        # Common conversion patterns
        if target_format == "pandas.DataFrame":
            if hasattr(data, 'to_dataframe'):
                data = data.to_dataframe()
                transformations.append("to_dataframe")
            elif hasattr(data, 'values'):  # numpy array
                import pandas as pd
                data = pd.DataFrame(data)
                transformations.append("array_to_dataframe")
                
        elif target_format == "numpy.ndarray":
            if hasattr(data, 'values'):  # pandas DataFrame
                data = data.values
                transformations.append("dataframe_to_array")
            elif hasattr(data, 'to_numpy'):
                data = data.to_numpy()
                transformations.append("to_numpy")
                
        elif target_format == "geopandas.GeoDataFrame":
            if self.is_library_available("geopandas"):
                geopandas = self.get_library("geopandas")
                if not isinstance(data, geopandas.GeoDataFrame):
                    # Try to convert pandas DataFrame to GeoDataFrame
                    data = geopandas.GeoDataFrame(data)
                    transformations.append("dataframe_to_geodataframe")
        
        return data, transformations
    
    def convert_output_data(self, result: Any, target_format: str = "pandas.DataFrame") -> Tuple[Any, List[str]]:
        """Convert output data to standard pipeline format."""
        transformations = []
        
        # Handle common result types
        if target_format == "pandas.DataFrame":
            if hasattr(result, 'to_dataframe'):
                result = result.to_dataframe()
                transformations.append("result_to_dataframe")
            elif isinstance(result, dict):
                import pandas as pd
                result = pd.DataFrame([result])
                transformations.append("dict_to_dataframe")
        
        return result, transformations
    
    def handle_streaming_execution(self,
                                 function_name: str,
                                 data_stream: Any,
                                 parameters: Dict[str, Any],
                                 chunk_size: int = 10000) -> Tuple[Any, IntegrationMetadata]:
        """Handle streaming execution for batch-oriented libraries."""
        results = []
        metadata_list = []
        
        # Process data in chunks if needed
        if hasattr(data_stream, 'iterrows') or hasattr(data_stream, '__iter__'):
            # Handle pandas DataFrame or iterable
            for chunk in self._chunk_data(data_stream, chunk_size):
                chunk_result, chunk_metadata = self.adapt_function_call(
                    function_name, chunk, parameters
                )
                results.append(chunk_result)
                metadata_list.append(chunk_metadata)
        else:
            # Single execution
            return self.adapt_function_call(function_name, data_stream, parameters)
        
        # Combine results
        combined_result = self._combine_chunk_results(results)
        combined_metadata = self._combine_chunk_metadata(metadata_list)
        
        return combined_result, combined_metadata
    
    def _chunk_data(self, data: Any, chunk_size: int):
        """Generator to chunk data for streaming processing."""
        if hasattr(data, 'iloc'):  # pandas DataFrame
            for i in range(0, len(data), chunk_size):
                yield data.iloc[i:i + chunk_size]
        elif hasattr(data, '__getitem__') and hasattr(data, '__len__'):  # array-like
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
        else:
            yield data  # Single item
    
    def _combine_chunk_results(self, results: List[Any]) -> Any:
        """Combine results from chunked processing."""
        if not results:
            return None
            
        first_result = results[0]
        
        # Handle pandas DataFrames
        if hasattr(first_result, 'concat'):
            import pandas as pd
            return pd.concat(results, ignore_index=True)
        elif hasattr(first_result, 'append'):
            combined = first_result.copy()
            for result in results[1:]:
                combined = combined.append(result, ignore_index=True)
            return combined
        elif isinstance(first_result, list):
            combined = []
            for result in results:
                combined.extend(result)
            return combined
        elif isinstance(first_result, dict):
            # Merge dictionaries - last one wins for conflicts
            combined = {}
            for result in results:
                combined.update(result)
            return combined
        
        # Default: return last result
        return results[-1]
    
    def _combine_chunk_metadata(self, metadata_list: List[IntegrationMetadata]) -> IntegrationMetadata:
        """Combine metadata from chunked processing."""
        if not metadata_list:
            return IntegrationMetadata(library_used="unknown")
        
        first_metadata = metadata_list[0]
        combined_transformations = []
        combined_performance_notes = []
        
        for metadata in metadata_list:
            combined_transformations.extend(metadata.data_transformations)
            combined_performance_notes.extend(metadata.performance_notes)
        
        return IntegrationMetadata(
            library_used=first_metadata.library_used,
            library_version=first_metadata.library_version,
            integration_strategy=first_metadata.integration_strategy,
            data_transformations=list(set(combined_transformations)),
            performance_notes=combined_performance_notes,
            streaming_compatible=first_metadata.streaming_compatible,
            fallback_used=first_metadata.fallback_used,
            original_parameters=first_metadata.original_parameters
        )


# ============================================================================
# Dependency Manager
# ============================================================================

class DependencyManager:
    """
    Centralized dependency management for library integration.
    
    Provides:
    - Runtime library detection
    - Graceful degradation strategies  
    - Installation guidance
    - Fallback chain management
    """
    
    def __init__(self):
        self.library_status = {}
        self.fallback_chains = {}
        self._initialize_fallback_chains()
    
    def register_library_group(self, 
                             category: LibraryCategory,
                             primary_library: str,
                             fallback_chain: List[str]):
        """Register a fallback chain for a library category."""
        self.fallback_chains[category] = {
            'primary': primary_library,
            'fallbacks': fallback_chain
        }
    
    def get_available_library(self, category: LibraryCategory) -> Optional[str]:
        """Get the best available library for a category."""
        if category not in self.fallback_chains:
            return None
        
        chain_info = self.fallback_chains[category]
        
        # Try primary library first
        if self._check_library_available(chain_info['primary']):
            return chain_info['primary']
        
        # Try fallback libraries
        for fallback in chain_info['fallbacks']:
            if self._check_library_available(fallback):
                logger.info(f"Using fallback library {fallback} for {category.value}")
                return fallback
        
        return None
    
    def _check_library_available(self, library_name: str) -> bool:
        """Check if a library is available for import."""
        if library_name in self.library_status:
            return self.library_status[library_name]
        
        try:
            importlib.import_module(library_name)
            self.library_status[library_name] = True
            return True
        except ImportError:
            self.library_status[library_name] = False
            return False
    
    def _initialize_fallback_chains(self):
        """Initialize default fallback chains for major library categories."""
        # Geospatial processing
        self.register_library_group(
            LibraryCategory.GEOSPATIAL,
            primary_library="geopandas",
            fallback_chain=["shapely", "sklearn.cluster"]  # Basic spatial clustering
        )
        
        # Time series analysis
        self.register_library_group(
            LibraryCategory.TIME_SERIES,
            primary_library="statsmodels",
            fallback_chain=["scipy.stats", "sklearn.linear_model"]
        )
        
        # Statistical analysis
        self.register_library_group(
            LibraryCategory.STATISTICS,
            primary_library="scipy.stats",
            fallback_chain=["sklearn.metrics", "numpy"]
        )
        
        # Optimization
        self.register_library_group(
            LibraryCategory.OPTIMIZATION,
            primary_library="scipy.optimize",
            fallback_chain=["sklearn.model_selection"]  # Grid search as fallback
        )
        
        # Fuzzy matching
        self.register_library_group(
            LibraryCategory.FUZZY_MATCHING,
            primary_library="fuzzywuzzy",
            fallback_chain=["difflib"]  # Standard library fallback
        )


# ============================================================================
# Dependency Injection Decorator
# ============================================================================

def requires_library(library_name: str, 
                    fallback_function: Optional[Callable] = None,
                    installation_hint: str = ""):
    """
    Decorator for functions that require optional libraries.
    
    Provides automatic fallback to alternative implementations
    when specialized libraries are not available.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Check if library is available
                importlib.import_module(library_name)
                return func(*args, **kwargs)
            except ImportError:
                if fallback_function:
                    logger.warning(f"Library {library_name} not available, using fallback")
                    return fallback_function(*args, **kwargs)
                else:
                    hint = installation_hint or f"pip install {library_name}"
                    raise CompositionError(
                        f"Required library {library_name} not available. Install with: {hint}",
                        error_type="missing_dependency"
                    )
        return wrapper
    return decorator


# ============================================================================
# Integration Result Container
# ============================================================================

@dataclass
class LibraryIntegrationResult:
    """
    Container for results from library integration operations.
    
    Designed for downstream tool composition with rich metadata.
    """
    data: Any
    metadata: IntegrationMetadata
    performance_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    # Composition support
    composition_context: Dict[str, Any] = field(default_factory=dict)
    downstream_hints: Dict[str, Any] = field(default_factory=dict)
    
    def add_composition_hint(self, tool_name: str, parameters: Dict[str, Any]):
        """Add hints for downstream tool composition."""
        self.downstream_hints[tool_name] = parameters
    
    def add_performance_note(self, note: str):
        """Add performance observation."""
        if 'notes' not in self.performance_info:
            self.performance_info['notes'] = []
        self.performance_info['notes'].append(note)
    
    def to_composition_stage(self, tool_name: str, function: str) -> CompositionStage:
        """Convert result to composition stage for pipeline building."""
        return CompositionStage(
            tool_name=tool_name,
            function=function,
            parameters=self.metadata.original_parameters,
            output_types=[type(self.data).__name__],
            is_streaming=self.metadata.streaming_compatible,
            estimated_memory_mb=self.performance_info.get('memory_mb', 0.0),
            estimated_duration_seconds=self.performance_info.get('duration_seconds', 0.0)
        )


if __name__ == "__main__":
    # Example usage demonstration
    dependency_manager = DependencyManager()
    
    # Check what's available for geospatial processing
    geo_library = dependency_manager.get_available_library(LibraryCategory.GEOSPATIAL)
    print(f"Best available geospatial library: {geo_library}")
    
    # Check time series capabilities
    ts_library = dependency_manager.get_available_library(LibraryCategory.TIME_SERIES)
    print(f"Best available time series library: {ts_library}")