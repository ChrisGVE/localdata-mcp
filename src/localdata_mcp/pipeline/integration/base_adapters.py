"""
Base adapter implementations for the Integration Shims Framework.

This module provides foundational converter classes with sklearn-compatible 
transformer interfaces, streaming capabilities, and comprehensive error handling.

Key Features:
- BaseShimAdapter with sklearn-compatible fit/transform pattern
- StreamingShimAdapter for memory-efficient large dataset processing  
- CachingShimAdapter for performance optimization
- Utility adapters for common conversion patterns
- Comprehensive validation and error handling
"""

import time
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import lru_cache
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .interfaces import (
    DataFormat, ConversionRequest, ConversionResult, ConversionCost,
    ShimAdapter, ConversionError, ValidationResult, MemoryConstraints,
    PerformanceRequirements
)
from ..type_conversion import TypeInferenceEngine
from ...logging_manager import get_logger

logger = get_logger(__name__)


@dataclass
class ConversionContext:
    """Extended context for conversion operations."""
    request_id: str
    start_time: float = field(default_factory=time.time)
    memory_usage_start: Optional[float] = None
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseShimAdapter(ShimAdapter, BaseEstimator, TransformerMixin):
    """
    Base implementation of ShimAdapter with sklearn-compatible interface.
    
    Provides common functionality including:
    - Sklearn-compatible fit/transform pattern
    - Comprehensive validation and error handling
    - Performance monitoring and logging
    - Context preservation throughout conversion
    """
    
    def __init__(self, 
                 adapter_id: str,
                 supported_conversions: Optional[List[Tuple[DataFormat, DataFormat]]] = None,
                 memory_constraints: Optional[MemoryConstraints] = None,
                 performance_requirements: Optional[PerformanceRequirements] = None,
                 enable_caching: bool = True,
                 enable_validation: bool = True):
        """
        Initialize BaseShimAdapter.
        
        Args:
            adapter_id: Unique identifier for this adapter
            supported_conversions: List of (source, target) format tuples
            memory_constraints: Memory usage constraints
            performance_requirements: Performance requirements
            enable_caching: Enable result caching
            enable_validation: Enable input/output validation
        """
        super().__init__(adapter_id)
        self.supported_conversions = supported_conversions or []
        self.memory_constraints = memory_constraints
        self.performance_requirements = performance_requirements
        self.enable_caching = enable_caching
        self.enable_validation = enable_validation
        
        # Internal state
        self._fitted = False
        self._fit_metadata: Dict[str, Any] = {}
        self._conversion_cache: Dict[str, ConversionResult] = {}
        self._performance_stats: Dict[str, List[float]] = {}
        
        # Initialize type inference engine for data analysis
        self._type_engine = TypeInferenceEngine()
        
        logger.info(f"Initialized {self.__class__.__name__}", 
                   adapter_id=adapter_id,
                   enable_caching=enable_caching,
                   enable_validation=enable_validation)
    
    def fit(self, X: Any, y: Optional[Any] = None) -> 'BaseShimAdapter':
        """
        Fit the adapter by analyzing data characteristics.
        
        This method analyzes input data to optimize conversion strategies,
        cache common patterns, and validate adapter configuration.
        
        Args:
            X: Input data to analyze
            y: Target data (ignored, for sklearn compatibility)
            
        Returns:
            self (fitted adapter)
        """
        start_time = time.time()
        
        try:
            # Detect input format if needed
            if hasattr(X, '__iter__') and not isinstance(X, (str, bytes)):
                # Sample data for analysis
                sample_data = self._sample_data_for_analysis(X)
                
                # Analyze data characteristics
                self._fit_metadata = self._analyze_data_characteristics(sample_data)
                
                logger.info("Data analysis completed for fitting",
                           adapter_id=self.adapter_id,
                           data_size=len(sample_data) if hasattr(sample_data, '__len__') else 'unknown',
                           analysis_time=time.time() - start_time)
            
            self._fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit adapter {self.adapter_id}: {e}")
            raise ConversionError(
                ConversionError.Type.ADAPTER_NOT_FOUND,
                f"Adapter fitting failed: {e}"
            )
    
    def transform(self, X: Any) -> Any:
        """
        Transform input data using fitted adapter.
        
        Args:
            X: Input data to transform
            
        Returns:
            Transformed data
        """
        if not self._fitted:
            # Auto-fit if not already fitted
            self.fit(X)
        
        # This is a base implementation - subclasses should override
        # for specific transformation logic
        return X
    
    def can_convert(self, request: ConversionRequest) -> float:
        """
        Evaluate capability to handle conversion request.
        
        Args:
            request: Conversion request to evaluate
            
        Returns:
            Confidence score (0-1)
        """
        # Check if conversion is in supported list
        conversion_tuple = (request.source_format, request.target_format)
        if conversion_tuple in self.supported_conversions:
            base_confidence = 0.8
        elif request.source_format == request.target_format:
            # Pass-through conversion
            base_confidence = 1.0
        else:
            base_confidence = 0.0
        
        # Adjust based on data characteristics if fitted
        if self._fitted and base_confidence > 0:
            # Consider memory constraints
            if self.memory_constraints:
                data_size = self._estimate_data_size(request.source_data)
                if (self.memory_constraints.max_memory_mb and 
                    data_size > self.memory_constraints.max_memory_mb * 1024 * 1024):
                    if not self.memory_constraints.prefer_streaming:
                        base_confidence *= 0.5
            
            # Consider performance requirements
            if self.performance_requirements:
                estimated_time = self._estimate_conversion_time(request)
                if (self.performance_requirements.max_execution_time_seconds and
                    estimated_time > self.performance_requirements.max_execution_time_seconds):
                    base_confidence *= 0.7
        
        return min(base_confidence, 1.0)
    
    def convert(self, request: ConversionRequest) -> ConversionResult:
        """
        Perform data conversion with comprehensive error handling.
        
        Args:
            request: Conversion request with all parameters
            
        Returns:
            Conversion result with converted data and metadata
        """
        context = ConversionContext(request_id=request.request_id)
        
        try:
            # Validate request if enabled
            if self.enable_validation:
                validation_result = self.validate_request(request)
                if not validation_result.is_valid:
                    return self._create_error_result(
                        request, 
                        ConversionError.Type.SCHEMA_INVALID,
                        f"Request validation failed: {'; '.join(validation_result.errors)}",
                        context
                    )
            
            # Check cache if enabled
            if self.enable_caching:
                cache_key = self._generate_cache_key(request)
                if cache_key in self._conversion_cache:
                    cached_result = self._conversion_cache[cache_key]
                    logger.info("Returned cached conversion result", 
                               request_id=request.request_id)
                    return cached_result
            
            # Ensure adapter is fitted
            if not self._fitted:
                self.fit(request.source_data)
            
            # Perform the actual conversion
            converted_data = self._perform_conversion(request, context)
            
            # Calculate performance metrics
            execution_time = time.time() - context.start_time
            self._update_performance_stats(execution_time)
            
            # Create successful result
            result = ConversionResult(
                converted_data=converted_data,
                success=True,
                original_format=request.source_format,
                target_format=request.target_format,
                actual_format=request.target_format,
                metadata=self._preserve_metadata(request, context),
                performance_metrics={
                    'execution_time': execution_time,
                    'adapter_id': self.adapter_id,
                    **context.performance_metrics
                },
                quality_score=self._calculate_quality_score(request, converted_data, context),
                warnings=context.warnings,
                request_id=request.request_id,
                execution_time=execution_time
            )
            
            # Cache result if enabled
            if self.enable_caching:
                self._conversion_cache[cache_key] = result
            
            logger.info("Conversion completed successfully",
                       request_id=request.request_id,
                       adapter_id=self.adapter_id,
                       execution_time=execution_time)
            
            return result
            
        except ConversionError:
            raise
        except Exception as e:
            logger.error(f"Conversion failed unexpectedly: {e}",
                        request_id=request.request_id,
                        adapter_id=self.adapter_id)
            return self._create_error_result(
                request,
                ConversionError.Type.CONVERSION_FAILED,
                str(e),
                context
            )
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """
        Estimate computational cost of conversion.
        
        Args:
            request: Conversion request to estimate
            
        Returns:
            Estimated cost breakdown
        """
        data_size = self._estimate_data_size(request.source_data)
        
        # Base cost estimation
        computational_cost = 0.1  # Low base cost
        memory_cost_mb = data_size / (1024 * 1024) * 1.2  # 20% overhead
        time_estimate = data_size / (10 * 1024 * 1024)  # 10MB/sec processing rate
        
        # Adjust based on conversion complexity
        if request.source_format != request.target_format:
            computational_cost += 0.3
            time_estimate *= 2
        
        # Consider memory constraints
        if (self.memory_constraints and 
            self.memory_constraints.prefer_streaming and
            data_size > 100 * 1024 * 1024):  # 100MB threshold
            # Streaming reduces memory but increases time
            memory_cost_mb *= 0.3
            time_estimate *= 1.5
            computational_cost += 0.2
        
        return ConversionCost(
            computational_cost=min(computational_cost, 1.0),
            memory_cost_mb=memory_cost_mb,
            time_estimate_seconds=time_estimate,
            io_operations=1 if data_size > 1024 * 1024 else 0,  # Large data needs I/O
            network_operations=0,
            quality_impact=0.0
        )
    
    def get_supported_conversions(self) -> List[Tuple[DataFormat, DataFormat]]:
        """Return list of supported conversion paths."""
        return self.supported_conversions.copy()
    
    # Protected methods for subclass implementation
    
    def _perform_conversion(self, request: ConversionRequest, 
                          context: ConversionContext) -> Any:
        """
        Perform the actual data conversion.
        
        This is the main method that subclasses should override to implement
        specific conversion logic.
        
        Args:
            request: Conversion request
            context: Conversion context for tracking
            
        Returns:
            Converted data
        """
        # Base implementation is pass-through
        if request.source_format == request.target_format:
            return request.source_data
        
        raise NotImplementedError(
            f"Conversion from {request.source_format} to {request.target_format} "
            f"not implemented in {self.__class__.__name__}"
        )
    
    def _sample_data_for_analysis(self, data: Any, max_samples: int = 1000) -> Any:
        """Sample data for efficient analysis during fitting."""
        if isinstance(data, pd.DataFrame):
            return data.head(max_samples) if len(data) > max_samples else data
        elif isinstance(data, pd.Series):
            return data.head(max_samples) if len(data) > max_samples else data
        elif isinstance(data, np.ndarray):
            return data[:max_samples] if len(data) > max_samples else data
        elif hasattr(data, '__len__') and len(data) > max_samples:
            return data[:max_samples]
        else:
            return data
    
    def _analyze_data_characteristics(self, data: Any) -> Dict[str, Any]:
        """Analyze data characteristics for optimization."""
        characteristics = {
            'analysis_timestamp': time.time(),
            'data_type': type(data).__name__,
        }
        
        if isinstance(data, pd.DataFrame):
            characteristics.update({
                'shape': data.shape,
                'dtypes': data.dtypes.to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'null_counts': data.isnull().sum().to_dict()
            })
        elif isinstance(data, pd.Series):
            characteristics.update({
                'length': len(data),
                'dtype': str(data.dtype),
                'memory_usage': data.memory_usage(deep=True),
                'null_count': data.isnull().sum()
            })
        elif isinstance(data, np.ndarray):
            characteristics.update({
                'shape': data.shape,
                'dtype': str(data.dtype),
                'memory_usage': data.nbytes
            })
        
        return characteristics
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate data size in bytes."""
        if hasattr(data, 'memory_usage'):
            return int(data.memory_usage(deep=True).sum())
        elif hasattr(data, 'nbytes'):
            return int(data.nbytes)
        elif hasattr(data, '__len__'):
            # Rough estimation for other sequences
            return len(data) * 64  # Assume 64 bytes per item
        else:
            return 1024  # Default size
    
    def _estimate_conversion_time(self, request: ConversionRequest) -> float:
        """Estimate conversion time based on data and operation."""
        data_size = self._estimate_data_size(request.source_data)
        base_time = data_size / (50 * 1024 * 1024)  # 50MB/sec base rate
        
        # Adjust based on conversion complexity
        if request.source_format != request.target_format:
            base_time *= 2
        
        return max(base_time, 0.001)  # Minimum 1ms
    
    def _preserve_metadata(self, request: ConversionRequest, 
                          context: ConversionContext) -> Dict[str, Any]:
        """Preserve and enhance metadata during conversion."""
        preserved_metadata = request.metadata.copy()
        
        # Add conversion metadata
        preserved_metadata.update({
            'conversion_adapter': self.adapter_id,
            'conversion_timestamp': time.time(),
            'conversion_request_id': request.request_id,
            'source_format': request.source_format.value,
            'target_format': request.target_format.value,
        })
        
        # Add fit metadata if available
        if self._fit_metadata:
            preserved_metadata['fit_metadata'] = self._fit_metadata
        
        return preserved_metadata
    
    def _calculate_quality_score(self, request: ConversionRequest, 
                                converted_data: Any, context: ConversionContext) -> float:
        """Calculate conversion quality score."""
        base_score = 1.0
        
        # Reduce score for warnings
        if context.warnings:
            base_score -= len(context.warnings) * 0.1
        
        # Pass-through conversions are perfect
        if request.source_format == request.target_format:
            return 1.0
        
        # Reduce score based on potential data loss indicators
        # This is a basic implementation - subclasses can enhance
        
        return max(base_score, 0.0)
    
    def _generate_cache_key(self, request: ConversionRequest) -> str:
        """Generate cache key for conversion request."""
        # Create hash of key request components
        key_data = f"{request.source_format.value}_{request.target_format.value}_{id(request.source_data)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _update_performance_stats(self, execution_time: float):
        """Update performance statistics."""
        if 'execution_times' not in self._performance_stats:
            self._performance_stats['execution_times'] = []
        
        self._performance_stats['execution_times'].append(execution_time)
        
        # Keep only recent stats
        if len(self._performance_stats['execution_times']) > 100:
            self._performance_stats['execution_times'] = \
                self._performance_stats['execution_times'][-100:]
    
    def _create_error_result(self, request: ConversionRequest, 
                           error_type: ConversionError.Type,
                           message: str, context: ConversionContext) -> ConversionResult:
        """Create error result for failed conversions."""
        execution_time = time.time() - context.start_time
        
        return ConversionResult(
            converted_data=request.source_data,  # Return original data
            success=False,
            original_format=request.source_format,
            target_format=request.target_format,
            actual_format=request.source_format,  # No conversion happened
            errors=[f"{error_type.value}: {message}"],
            performance_metrics={
                'execution_time': execution_time,
                'adapter_id': self.adapter_id,
                'error_type': error_type.value
            },
            quality_score=0.0,
            warnings=context.warnings,
            request_id=request.request_id,
            execution_time=execution_time
        )


class StreamingShimAdapter(BaseShimAdapter):
    """
    Streaming-optimized adapter for memory-efficient processing of large datasets.
    
    Extends BaseShimAdapter with chunked processing capabilities to handle
    datasets that don't fit in memory.
    """
    
    def __init__(self, 
                 adapter_id: str,
                 chunk_size: int = 10000,
                 **kwargs):
        """
        Initialize StreamingShimAdapter.
        
        Args:
            adapter_id: Unique identifier for this adapter
            chunk_size: Number of records per processing chunk
            **kwargs: Additional arguments passed to BaseShimAdapter
        """
        # Force streaming-friendly memory constraints
        if 'memory_constraints' not in kwargs or kwargs['memory_constraints'] is None:
            kwargs['memory_constraints'] = MemoryConstraints(
                prefer_streaming=True,
                chunk_size=chunk_size,
                memory_efficient=True
            )
        
        super().__init__(adapter_id, **kwargs)
        self.chunk_size = chunk_size
        
        logger.info(f"Initialized StreamingShimAdapter", 
                   adapter_id=adapter_id,
                   chunk_size=chunk_size)
    
    def _perform_conversion(self, request: ConversionRequest, 
                          context: ConversionContext) -> Any:
        """
        Perform streaming conversion for large datasets.
        
        Args:
            request: Conversion request
            context: Conversion context for tracking
            
        Returns:
            Converted data
        """
        data_size = self._estimate_data_size(request.source_data)
        
        # Use streaming processing for large datasets
        if data_size > 100 * 1024 * 1024:  # 100MB threshold
            return self._stream_convert(request, context)
        else:
            # Use regular processing for small data
            return super()._perform_conversion(request, context)
    
    def _stream_convert(self, request: ConversionRequest, 
                       context: ConversionContext) -> Any:
        """
        Perform chunked streaming conversion.
        
        Args:
            request: Conversion request
            context: Conversion context
            
        Returns:
            Converted data assembled from chunks
        """
        # This is a base implementation - subclasses should implement
        # specific streaming logic for their conversion types
        
        if isinstance(request.source_data, pd.DataFrame):
            return self._stream_convert_dataframe(request, context)
        else:
            # Fallback to regular conversion
            context.warnings.append("Streaming not supported for this data type, using regular conversion")
            return super()._perform_conversion(request, context)
    
    def _stream_convert_dataframe(self, request: ConversionRequest,
                                 context: ConversionContext) -> pd.DataFrame:
        """Stream convert a DataFrame in chunks."""
        source_df = request.source_data
        converted_chunks = []
        
        total_chunks = (len(source_df) + self.chunk_size - 1) // self.chunk_size
        
        for i in range(0, len(source_df), self.chunk_size):
            chunk = source_df.iloc[i:i + self.chunk_size]
            
            # Create chunk-specific request
            chunk_request = ConversionRequest(
                source_data=chunk,
                source_format=request.source_format,
                target_format=request.target_format,
                metadata=request.metadata,
                context=request.context,
                request_id=f"{request.request_id}_chunk_{i//self.chunk_size}"
            )
            
            # Convert chunk using base implementation
            converted_chunk = super()._perform_conversion(chunk_request, context)
            converted_chunks.append(converted_chunk)
            
            # Update progress
            progress = ((i // self.chunk_size) + 1) / total_chunks
            context.performance_metrics[f'chunk_{i//self.chunk_size}_progress'] = progress
        
        # Combine chunks
        result = pd.concat(converted_chunks, ignore_index=True)
        context.performance_metrics['total_chunks_processed'] = len(converted_chunks)
        
        return result


class CachingShimAdapter(BaseShimAdapter):
    """
    Performance-optimized adapter with intelligent caching strategies.
    
    Extends BaseShimAdapter with advanced caching mechanisms including
    LRU eviction, cache warming, and intelligent cache key generation.
    """
    
    def __init__(self,
                 adapter_id: str, 
                 cache_size_mb: int = 256,
                 cache_ttl_seconds: int = 3600,
                 **kwargs):
        """
        Initialize CachingShimAdapter.
        
        Args:
            adapter_id: Unique identifier for this adapter
            cache_size_mb: Maximum cache size in MB
            cache_ttl_seconds: Cache time-to-live in seconds
            **kwargs: Additional arguments passed to BaseShimAdapter
        """
        kwargs['enable_caching'] = True  # Force caching enabled
        super().__init__(adapter_id, **kwargs)
        
        self.cache_size_mb = cache_size_mb
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache_access_times: Dict[str, float] = {}
        self._cache_sizes: Dict[str, int] = {}
        self._total_cache_size = 0
        
        logger.info(f"Initialized CachingShimAdapter",
                   adapter_id=adapter_id,
                   cache_size_mb=cache_size_mb,
                   cache_ttl_seconds=cache_ttl_seconds)
    
    def _generate_cache_key(self, request: ConversionRequest) -> str:
        """Generate more sophisticated cache key."""
        # Include more request details for better cache precision
        key_components = [
            request.source_format.value,
            request.target_format.value,
            str(hash(str(request.source_data))),  # Data hash
            str(request.metadata),
            str(request.context.user_intention) if request.context else "",
        ]
        
        key_data = "_".join(key_components)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def convert(self, request: ConversionRequest) -> ConversionResult:
        """Enhanced convert with advanced caching."""
        # Clean expired cache entries first
        self._cleanup_expired_cache()
        
        # Check cache with TTL validation
        cache_key = self._generate_cache_key(request)
        if cache_key in self._conversion_cache:
            cache_time = self._cache_access_times.get(cache_key, 0)
            if time.time() - cache_time < self.cache_ttl_seconds:
                # Update access time
                self._cache_access_times[cache_key] = time.time()
                
                cached_result = self._conversion_cache[cache_key]
                logger.info("Returned cached conversion result with TTL validation",
                           request_id=request.request_id,
                           cache_age_seconds=time.time() - cache_time)
                return cached_result
            else:
                # Remove expired entry
                self._remove_from_cache(cache_key)
        
        # Perform conversion
        result = super().convert(request)
        
        # Cache result with size management
        if result.success:
            self._add_to_cache(cache_key, result)
        
        return result
    
    def _add_to_cache(self, cache_key: str, result: ConversionResult):
        """Add result to cache with size management."""
        result_size = self._estimate_result_size(result)
        
        # Check if we need to evict entries
        while (self._total_cache_size + result_size > 
               self.cache_size_mb * 1024 * 1024 and 
               self._conversion_cache):
            self._evict_lru_entry()
        
        # Add to cache
        self._conversion_cache[cache_key] = result
        self._cache_access_times[cache_key] = time.time()
        self._cache_sizes[cache_key] = result_size
        self._total_cache_size += result_size
        
        logger.debug(f"Added result to cache",
                    cache_key=cache_key[:16] + "...",
                    result_size_mb=result_size / (1024 * 1024),
                    total_cache_size_mb=self._total_cache_size / (1024 * 1024))
    
    def _remove_from_cache(self, cache_key: str):
        """Remove entry from cache."""
        if cache_key in self._conversion_cache:
            size = self._cache_sizes.pop(cache_key, 0)
            self._total_cache_size -= size
            del self._conversion_cache[cache_key]
            self._cache_access_times.pop(cache_key, None)
    
    def _evict_lru_entry(self):
        """Evict least recently used cache entry."""
        if not self._cache_access_times:
            return
        
        # Find LRU entry
        lru_key = min(self._cache_access_times.keys(),
                     key=lambda k: self._cache_access_times[k])
        
        logger.debug(f"Evicting LRU cache entry", cache_key=lru_key[:16] + "...")
        self._remove_from_cache(lru_key)
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self._cache_access_times.items()
            if current_time - access_time > self.cache_ttl_seconds
        ]
        
        for key in expired_keys:
            self._remove_from_cache(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _estimate_result_size(self, result: ConversionResult) -> int:
        """Estimate size of conversion result."""
        base_size = 1024  # Base overhead
        
        if hasattr(result.converted_data, 'memory_usage'):
            base_size += int(result.converted_data.memory_usage(deep=True).sum())
        elif hasattr(result.converted_data, 'nbytes'):
            base_size += int(result.converted_data.nbytes)
        elif hasattr(result.converted_data, '__len__'):
            base_size += len(result.converted_data) * 64
        
        return base_size


class PassThroughAdapter(BaseShimAdapter):
    """
    Utility adapter for pass-through conversions (no actual conversion needed).
    
    Useful for same-format conversions or when data is already in target format.
    """
    
    def __init__(self, adapter_id: str = "pass_through"):
        """Initialize PassThroughAdapter."""
        # Support all format-to-same-format conversions
        all_formats = list(DataFormat)
        supported_conversions = [(fmt, fmt) for fmt in all_formats]
        
        super().__init__(
            adapter_id=adapter_id,
            supported_conversions=supported_conversions,
            enable_caching=False  # No need to cache pass-through
        )
    
    def can_convert(self, request: ConversionRequest) -> float:
        """Return 1.0 confidence for same-format conversions."""
        if request.source_format == request.target_format:
            return 1.0
        return 0.0
    
    def _perform_conversion(self, request: ConversionRequest, 
                          context: ConversionContext) -> Any:
        """Return data unchanged for pass-through conversion."""
        if request.source_format != request.target_format:
            raise ConversionError(
                ConversionError.Type.TYPE_MISMATCH,
                f"PassThroughAdapter cannot convert {request.source_format} to {request.target_format}"
            )
        
        return request.source_data
    
    def estimate_cost(self, request: ConversionRequest) -> ConversionCost:
        """Return minimal cost for pass-through."""
        return ConversionCost(
            computational_cost=0.01,
            memory_cost_mb=0.0,
            time_estimate_seconds=0.001,
            io_operations=0,
            network_operations=0,
            quality_impact=0.0
        )


class ValidationAdapter(BaseShimAdapter):
    """
    Utility adapter that adds comprehensive validation to any conversion.
    
    Can wrap other adapters to add validation layers without changing
    their core functionality.
    """
    
    def __init__(self,
                 adapter_id: str,
                 wrapped_adapter: Optional[ShimAdapter] = None,
                 strict_validation: bool = True,
                 **kwargs):
        """
        Initialize ValidationAdapter.
        
        Args:
            adapter_id: Unique identifier for this adapter
            wrapped_adapter: Optional adapter to wrap with validation
            strict_validation: Enable strict validation mode
            **kwargs: Additional arguments passed to BaseShimAdapter
        """
        kwargs['enable_validation'] = True  # Force validation enabled
        super().__init__(adapter_id, **kwargs)
        
        self.wrapped_adapter = wrapped_adapter
        self.strict_validation = strict_validation
    
    def can_convert(self, request: ConversionRequest) -> float:
        """Delegate to wrapped adapter if available."""
        if self.wrapped_adapter:
            return self.wrapped_adapter.can_convert(request)
        return super().can_convert(request)
    
    def validate_request(self, request: ConversionRequest) -> ValidationResult:
        """Enhanced request validation."""
        # Start with base validation
        base_result = super().validate_request(request)
        
        errors = base_result.errors.copy()
        warnings = base_result.warnings.copy()
        
        # Additional validation checks
        if request.source_data is None:
            errors.append("Source data is None")
        
        # Validate format specifications
        if request.format_spec:
            if request.format_spec.memory_constraints:
                if (request.format_spec.memory_constraints.max_memory_mb and
                    request.format_spec.memory_constraints.max_memory_mb <= 0):
                    errors.append("Invalid memory constraint: max_memory_mb must be positive")
            
            if request.format_spec.performance_requirements:
                perf_req = request.format_spec.performance_requirements
                if (perf_req.max_execution_time_seconds and
                    perf_req.max_execution_time_seconds <= 0):
                    errors.append("Invalid performance requirement: max_execution_time_seconds must be positive")
        
        # Validate data compatibility with formats
        if self.strict_validation:
            # Additional strict validation logic
            data_size = self._estimate_data_size(request.source_data)
            if data_size > 1024 * 1024 * 1024:  # 1GB
                warnings.append("Large dataset detected - consider streaming processing")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=1.0 if len(errors) == 0 else max(0.0, 1.0 - len(errors) * 0.2),
            errors=errors,
            warnings=warnings,
            details={
                'validation_type': 'enhanced',
                'strict_mode': self.strict_validation,
                'wrapped_adapter': self.wrapped_adapter.adapter_id if self.wrapped_adapter else None
            }
        )
    
    def _perform_conversion(self, request: ConversionRequest, 
                          context: ConversionContext) -> Any:
        """Perform conversion with validation."""
        if self.wrapped_adapter:
            # Use wrapped adapter for conversion
            wrapped_result = self.wrapped_adapter.convert(request)
            if not wrapped_result.success:
                raise ConversionError(
                    ConversionError.Type.CONVERSION_FAILED,
                    f"Wrapped adapter conversion failed: {'; '.join(wrapped_result.errors)}"
                )
            
            # Validate the result
            self._validate_conversion_result(request, wrapped_result.converted_data, context)
            return wrapped_result.converted_data
        else:
            # Use base implementation
            result = super()._perform_conversion(request, context)
            self._validate_conversion_result(request, result, context)
            return result
    
    def _validate_conversion_result(self, request: ConversionRequest, 
                                   converted_data: Any, context: ConversionContext):
        """Validate conversion result."""
        # Check if conversion actually changed the format as expected
        if request.source_format != request.target_format:
            # Perform format detection on result to verify conversion
            # This is a basic check - could be enhanced with more sophisticated detection
            if type(converted_data) == type(request.source_data):
                context.warnings.append("Converted data type matches source type - conversion may not have occurred")
        
        # Check for data loss indicators
        if hasattr(request.source_data, '__len__') and hasattr(converted_data, '__len__'):
            if len(converted_data) != len(request.source_data):
                context.warnings.append(f"Data length changed during conversion: {len(request.source_data)} -> {len(converted_data)}")
        
        # Memory usage validation
        source_size = self._estimate_data_size(request.source_data)
        converted_size = self._estimate_data_size(converted_data)
        
        if converted_size > source_size * 3:  # 3x increase threshold
            context.warnings.append(f"Significant memory increase during conversion: {source_size} -> {converted_size} bytes")