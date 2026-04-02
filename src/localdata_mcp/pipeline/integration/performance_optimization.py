"""
Performance Optimization System for LocalData MCP v2.0 Integration Shims.

This module implements comprehensive performance optimizations and memory-efficient
conversion strategies for the Integration Shims system, following the Five First Principles:

1. Intention-Driven Interface - LLM-natural performance configuration
2. Context-Aware Composition - Composition-friendly caching and memory management
3. Progressive Disclosure - Simple defaults with advanced tuning options
4. Streaming-First - Memory-bounded optimization strategies
5. Modular Domain Integration - Cross-domain performance optimization

Key Components:
- ConversionCache: Intelligent caching with LRU and memory-aware management
- LazyLoadingManager: Deferred loading framework for large datasets
- StreamingConversionEngine: Memory-bounded streaming conversion support
- MemoryPoolManager: Efficient memory allocation and object pooling
- PerformanceBenchmark: Comprehensive performance measurement and profiling
- OptimizationSelector: Adaptive optimization strategies based on data characteristics
"""

import gc
import time
import hashlib
import weakref
import asyncio
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable, Type
import logging
from pathlib import Path
import pickle
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .interfaces import (
    ShimAdapter, ConversionRequest, ConversionResult, DataFormat,
    MemoryConstraints, PerformanceRequirements
)
from ...streaming_executor import StreamingDataSource, MemoryStatus
from ...logging_manager import get_logger

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies for conversion operations."""
    CACHE_FIRST = "cache_first"  # Prioritize cache hits for repeated operations
    LAZY_LOADING = "lazy_loading"  # Defer conversion until data is actually needed
    STREAMING = "streaming"  # Use chunked processing for large datasets
    MEMORY_POOL = "memory_pool"  # Reuse allocated memory for similar operations
    PARALLEL = "parallel"  # Use multiple threads/processes for independent conversions
    ADAPTIVE = "adaptive"  # Dynamically select best strategy based on data characteristics


class CacheEvictionPolicy(Enum):
    """Cache eviction policies for memory management."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    MEMORY_PRESSURE = "memory_pressure"  # Based on memory usage


@dataclass
class CacheStatistics:
    """Statistics for conversion cache performance."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0
    average_lookup_time_ms: float = 0.0
    cache_size: int = 0
    max_cache_size: int = 0
    last_cleanup_time: Optional[float] = None


@dataclass
class CachedConversion:
    """A cached conversion result with metadata."""
    cache_key: str
    converted_data: Any
    original_format: DataFormat
    target_format: DataFormat
    metadata: Dict[str, Any]
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 1
    size_mb: float = 0.0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cached conversion has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.creation_time > self.ttl_seconds
    
    def touch(self) -> None:
        """Update last access time and increment access count."""
        self.last_access_time = time.time()
        self.access_count += 1


class ConversionCache:
    """
    Intelligent caching system for repeated conversions with LRU eviction,
    memory-aware management, and performance monitoring.
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: float = 500.0,
                 ttl_seconds: int = 3600,
                 eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
                 enable_compression: bool = True):
        """
        Initialize conversion cache.
        
        Args:
            max_size: Maximum number of cached conversions
            max_memory_mb: Maximum memory usage for cache in MB
            ttl_seconds: Default time-to-live for cached items
            eviction_policy: Cache eviction policy
            enable_compression: Enable data compression for large cached items
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression
        
        # Cache storage
        self._cache: Dict[str, CachedConversion] = {}
        self._cache_lock = threading.RLock()
        
        # Access ordering for LRU
        self._access_order: List[str] = []
        
        # Performance tracking
        self.stats = CacheStatistics(max_cache_size=max_size)
        self._lookup_times: List[float] = []
        
        # Memory monitoring
        self._current_memory_mb = 0.0
        self._cleanup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache_cleanup")
        
        logger.info(f"ConversionCache initialized",
                   max_size=max_size, max_memory_mb=max_memory_mb,
                   ttl_seconds=ttl_seconds, policy=eviction_policy.value)
    
    def _generate_cache_key(self, request: ConversionRequest) -> str:
        """
        Generate cache key based on conversion request characteristics.
        
        Args:
            request: Conversion request
            
        Returns:
            Cache key string
        """
        # Create fingerprint from request data and parameters
        key_components = [
            str(request.source_format.value),
            str(request.target_format.value),
            str(request.context.source_domain),
            str(request.context.target_domain)
        ]
        
        # Add data fingerprint
        data_fingerprint = self._compute_data_fingerprint(request.source_data)
        key_components.append(data_fingerprint)
        
        # Add metadata fingerprint for parameters that affect conversion
        relevant_metadata = {
            k: v for k, v in request.metadata.items()
            if k in ['precision', 'format_options', 'conversion_params']
        }
        if relevant_metadata:
            metadata_str = json.dumps(relevant_metadata, sort_keys=True)
            key_components.append(hashlib.md5(metadata_str.encode()).hexdigest())
        
        # Generate final hash
        key_string = '|'.join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def _compute_data_fingerprint(self, data: Any) -> str:
        """
        Compute fingerprint of data for cache key generation.
        
        Args:
            data: Data to fingerprint
            
        Returns:
            Data fingerprint string
        """
        try:
            if HAS_PANDAS and isinstance(data, pd.DataFrame):
                # For DataFrames, use shape, dtypes, and sample of data
                fingerprint_data = {
                    'shape': data.shape,
                    'dtypes': str(data.dtypes.to_dict()),
                    'columns': list(data.columns)
                }
                
                # Sample first and last few rows for content fingerprint
                if len(data) > 0:
                    sample_size = min(5, len(data))
                    sample_data = pd.concat([
                        data.head(sample_size),
                        data.tail(sample_size)
                    ]).fillna('NULL')
                    fingerprint_data['sample_hash'] = str(hash(str(sample_data.values.tobytes())))
                
                return hashlib.md5(str(fingerprint_data).encode()).hexdigest()
                
            elif HAS_NUMPY and isinstance(data, np.ndarray):
                # For numpy arrays, use shape, dtype, and data hash
                fingerprint_data = {
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'data_hash': str(hash(data.tobytes()))
                }
                return hashlib.md5(str(fingerprint_data).encode()).hexdigest()
                
            else:
                # For other data types, try to serialize and hash
                try:
                    data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                    return hashlib.md5(data_bytes).hexdigest()
                except Exception:
                    # Fallback to string representation
                    return hashlib.md5(str(data).encode()).hexdigest()
                    
        except Exception as e:
            logger.warning(f"Failed to compute data fingerprint: {e}")
            # Fallback to timestamp-based fingerprint
            return hashlib.md5(f"{time.time()}{id(data)}".encode()).hexdigest()
    
    def get(self, request: ConversionRequest) -> Optional[CachedConversion]:
        """
        Get cached conversion result for request.
        
        Args:
            request: Conversion request
            
        Returns:
            Cached conversion result or None if not found
        """
        start_time = time.time()
        cache_key = self._generate_cache_key(request)
        
        with self._cache_lock:
            self.stats.total_requests += 1
            
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                
                # Check expiration
                if cached.is_expired():
                    self._remove_from_cache(cache_key)
                    self.stats.cache_misses += 1
                    lookup_time = (time.time() - start_time) * 1000
                    self._record_lookup_time(lookup_time)
                    return None
                
                # Update access tracking
                cached.touch()
                self._update_access_order(cache_key)
                
                self.stats.cache_hits += 1
                self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
                
                lookup_time = (time.time() - start_time) * 1000
                self._record_lookup_time(lookup_time)
                
                logger.debug(f"Cache hit for key {cache_key[:8]}... (lookup: {lookup_time:.2f}ms)")
                return cached
            
            else:
                self.stats.cache_misses += 1
                self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
                
                lookup_time = (time.time() - start_time) * 1000
                self._record_lookup_time(lookup_time)
                
                return None
    
    def put(self, request: ConversionRequest, result: ConversionResult) -> None:
        """
        Store conversion result in cache.
        
        Args:
            request: Original conversion request
            result: Conversion result to cache
        """
        if not result.success:
            # Don't cache failed conversions
            return
        
        cache_key = self._generate_cache_key(request)
        
        # Estimate memory usage
        estimated_size_mb = self._estimate_result_size(result)
        
        # Check if we should compress large items
        compressed_data = result.converted_data
        if self.enable_compression and estimated_size_mb > 10.0:
            try:
                compressed_data = self._compress_data(result.converted_data)
                logger.debug(f"Compressed cache entry {cache_key[:8]}... from {estimated_size_mb:.1f}MB")
            except Exception as e:
                logger.warning(f"Failed to compress cache data: {e}")
        
        cached_conversion = CachedConversion(
            cache_key=cache_key,
            converted_data=compressed_data,
            original_format=result.original_format,
            target_format=result.actual_format,
            metadata=result.metadata.copy(),
            size_mb=estimated_size_mb,
            ttl_seconds=self.ttl_seconds
        )
        
        with self._cache_lock:
            # Check memory limits before adding
            if self._current_memory_mb + estimated_size_mb > self.max_memory_mb:
                self._evict_to_fit(estimated_size_mb)
            
            # Check size limits
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            # Add to cache
            self._cache[cache_key] = cached_conversion
            self._access_order.append(cache_key)
            self._current_memory_mb += estimated_size_mb
            self.stats.cache_size = len(self._cache)
            self.stats.memory_usage_mb = self._current_memory_mb
            
            logger.debug(f"Cached conversion {cache_key[:8]}... ({estimated_size_mb:.1f}MB)")
    
    def invalidate(self, pattern: str = None) -> int:
        """
        Invalidate cached entries matching pattern.
        
        Args:
            pattern: Pattern to match for invalidation (None invalidates all)
            
        Returns:
            Number of invalidated entries
        """
        with self._cache_lock:
            if pattern is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._access_order.clear()
                self._current_memory_mb = 0.0
                self.stats.cache_size = 0
                self.stats.memory_usage_mb = 0.0
                logger.info(f"Invalidated all {count} cache entries")
                return count
            
            # Pattern-based invalidation
            to_remove = []
            for cache_key in self._cache:
                # Simple pattern matching - could be enhanced with regex
                if pattern in cache_key or pattern in str(self._cache[cache_key].metadata):
                    to_remove.append(cache_key)
            
            for cache_key in to_remove:
                self._remove_from_cache(cache_key)
            
            logger.info(f"Invalidated {len(to_remove)} cache entries matching pattern '{pattern}'")
            return len(to_remove)
    
    def get_statistics(self) -> CacheStatistics:
        """
        Get current cache statistics.
        
        Returns:
            Cache statistics
        """
        with self._cache_lock:
            stats = CacheStatistics(
                total_requests=self.stats.total_requests,
                cache_hits=self.stats.cache_hits,
                cache_misses=self.stats.cache_misses,
                evictions=self.stats.evictions,
                memory_usage_mb=self._current_memory_mb,
                hit_rate=self.stats.hit_rate,
                average_lookup_time_ms=self.stats.average_lookup_time_ms,
                cache_size=len(self._cache),
                max_cache_size=self.max_size,
                last_cleanup_time=self.stats.last_cleanup_time
            )
            return stats
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of expired entries removed
        """
        with self._cache_lock:
            expired_keys = []
            current_time = time.time()
            
            for cache_key, cached in self._cache.items():
                if cached.is_expired():
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                self._remove_from_cache(cache_key)
            
            self.stats.last_cleanup_time = current_time
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def _estimate_result_size(self, result: ConversionResult) -> float:
        """
        Estimate memory size of conversion result in MB.
        
        Args:
            result: Conversion result
            
        Returns:
            Estimated size in MB
        """
        try:
            if HAS_PANDAS and isinstance(result.converted_data, pd.DataFrame):
                return result.converted_data.memory_usage(deep=True).sum() / (1024 * 1024)
            elif HAS_NUMPY and isinstance(result.converted_data, np.ndarray):
                return result.converted_data.nbytes / (1024 * 1024)
            else:
                # Rough estimate using pickle size
                try:
                    pickled = pickle.dumps(result.converted_data)
                    return len(pickled) / (1024 * 1024)
                except Exception:
                    # Fallback estimate
                    return 1.0
        except Exception as e:
            logger.warning(f"Failed to estimate result size: {e}")
            return 1.0
    
    def _compress_data(self, data: Any) -> bytes:
        """
        Compress data for cache storage.
        
        Args:
            data: Data to compress
            
        Returns:
            Compressed data as bytes
        """
        import gzip
        pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return gzip.compress(pickled_data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """
        Decompress cached data.
        
        Args:
            compressed_data: Compressed data bytes
            
        Returns:
            Decompressed data
        """
        import gzip
        decompressed = gzip.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def _update_access_order(self, cache_key: str) -> None:
        """
        Update access order for LRU tracking.
        
        Args:
            cache_key: Key that was accessed
        """
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        self._access_order.append(cache_key)
    
    def _remove_from_cache(self, cache_key: str) -> None:
        """
        Remove entry from cache and update tracking.
        
        Args:
            cache_key: Key to remove
        """
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            del self._cache[cache_key]
            self._current_memory_mb -= cached.size_mb
            self.stats.cache_size = len(self._cache)
            self.stats.memory_usage_mb = self._current_memory_mb
            
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
    
    def _evict_to_fit(self, required_mb: float) -> None:
        """
        Evict cache entries to make room for new entry.
        
        Args:
            required_mb: Amount of memory needed in MB
        """
        target_memory = self.max_memory_mb - required_mb
        
        while (self._current_memory_mb > target_memory and 
               len(self._cache) > 0):
            self._evict_oldest()
    
    def _evict_oldest(self) -> None:
        """
        Evict oldest cache entry based on eviction policy.
        """
        if not self._cache:
            return
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # Remove least recently used
            if self._access_order:
                oldest_key = self._access_order[0]
            else:
                oldest_key = next(iter(self._cache))
        
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # Remove least frequently used
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].access_count)
        
        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            # Remove oldest by creation time
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k].creation_time)
        
        else:  # MEMORY_PRESSURE - remove largest item
            oldest_key = max(self._cache.keys(),
                           key=lambda k: self._cache[k].size_mb)
        
        self._remove_from_cache(oldest_key)
        self.stats.evictions += 1
        
        logger.debug(f"Evicted cache entry {oldest_key[:8]}...")
    
    def _record_lookup_time(self, lookup_time_ms: float) -> None:
        """
        Record lookup time for performance tracking.
        
        Args:
            lookup_time_ms: Lookup time in milliseconds
        """
        self._lookup_times.append(lookup_time_ms)
        
        # Keep only last 1000 measurements
        if len(self._lookup_times) > 1000:
            self._lookup_times = self._lookup_times[-500:]
        
        # Update average
        if self._lookup_times:
            self.stats.average_lookup_time_ms = sum(self._lookup_times) / len(self._lookup_times)
    
    def __del__(self):
        """Cleanup resources on destruction."""
        if hasattr(self, '_cleanup_executor'):
            self._cleanup_executor.shutdown(wait=False)


@dataclass
class LazyConversionState:
    """State information for lazy conversion operations."""
    request: ConversionRequest
    converter: ShimAdapter
    is_loaded: bool = False
    is_loading: bool = False
    loaded_data: Optional[Any] = None
    load_future: Optional[asyncio.Future] = None
    load_error: Optional[Exception] = None
    access_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_access_time: Optional[float] = None
    estimated_size_mb: float = 0.0


class LazyConverter:
    """
    Wrapper for deferred conversion operations with lazy loading support.
    """
    
    def __init__(self, 
                 request: ConversionRequest,
                 converter: ShimAdapter,
                 threshold_mb: float = 50.0):
        """
        Initialize lazy converter.
        
        Args:
            request: Conversion request to defer
            converter: Adapter to perform conversion
            threshold_mb: Memory threshold for triggering lazy loading
        """
        self.state = LazyConversionState(request=request, converter=converter)
        self.threshold_mb = threshold_mb
        self._lock = threading.Lock()
        
    def __getattr__(self, name: str) -> Any:
        """
        Lazy attribute access - loads data when first accessed.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute value from loaded data
        """
        if name.startswith('_'):
            # Don't intercept private attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Load data if not already loaded
        data = self._ensure_loaded()
        
        if hasattr(data, name):
            return getattr(data, name)
        else:
            raise AttributeError(f"Loaded data has no attribute '{name}'")
    
    def _ensure_loaded(self) -> Any:
        """
        Ensure data is loaded, performing conversion if necessary.
        
        Returns:
            Loaded conversion result data
        """
        with self._lock:
            if self.state.is_loaded:
                self.state.access_count += 1
                self.state.last_access_time = time.time()
                return self.state.loaded_data
            
            if self.state.is_loading:
                # Wait for ongoing load to complete
                if self.state.load_future:
                    try:
                        # Wait for the future (this is a sync context)
                        # We'll implement a simple polling mechanism
                        start_time = time.time()
                        while not self.state.load_future.done() and time.time() - start_time < 30:
                            time.sleep(0.1)
                        
                        if self.state.load_future.done():
                            return self.state.loaded_data
                    except Exception as e:
                        logger.error(f"Error waiting for lazy load: {e}")
            
            # Perform synchronous load
            return self._load_sync()
    
    def _load_sync(self) -> Any:
        """
        Perform synchronous data loading.
        
        Returns:
            Loaded data
        """
        try:
            self.state.is_loading = True
            
            logger.debug(f"Loading lazy conversion: {self.state.request.source_format} -> {self.state.request.target_format}")
            
            result = self.state.converter.convert(self.state.request)
            
            if result.success:
                self.state.loaded_data = result.converted_data
                self.state.is_loaded = True
                self.state.access_count = 1
                self.state.last_access_time = time.time()
                
                logger.debug(f"Lazy conversion loaded successfully")
                return self.state.loaded_data
            else:
                error = Exception(f"Conversion failed: {result.errors}")
                self.state.load_error = error
                raise error
                
        except Exception as e:
            self.state.load_error = e
            logger.error(f"Lazy conversion failed: {e}")
            raise
        finally:
            self.state.is_loading = False
    
    def is_loaded(self) -> bool:
        """
        Check if data is already loaded.
        
        Returns:
            True if data is loaded
        """
        return self.state.is_loaded
    
    def get_state(self) -> LazyConversionState:
        """
        Get current lazy conversion state.
        
        Returns:
            Current state
        """
        return self.state


class LazyLoadingManager:
    """
    Manager for lazy loading operations with background loading and lifecycle management.
    """
    
    def __init__(self, 
                 default_threshold_mb: float = 50.0,
                 max_background_tasks: int = 3,
                 cleanup_interval_seconds: int = 300):
        """
        Initialize lazy loading manager.
        
        Args:
            default_threshold_mb: Default memory threshold for lazy loading
            max_background_tasks: Maximum concurrent background loading tasks
            cleanup_interval_seconds: Interval for cleaning up unused lazy converters
        """
        self.default_threshold_mb = default_threshold_mb
        self.max_background_tasks = max_background_tasks
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Lazy converter tracking
        self._lazy_converters: Dict[str, LazyConverter] = {}
        self._converter_lock = threading.RLock()
        
        # Background loading
        self._background_executor = ThreadPoolExecutor(
            max_workers=max_background_tasks,
            thread_name_prefix="lazy_loading"
        )
        self._background_tasks: Dict[str, asyncio.Future] = {}
        
        # Cleanup management
        self._last_cleanup = time.time()
        
        logger.info(f"LazyLoadingManager initialized",
                   threshold_mb=default_threshold_mb,
                   max_background_tasks=max_background_tasks)
    
    def create_lazy_converter(self, 
                            converter: ShimAdapter, 
                            request: ConversionRequest,
                            threshold_mb: Optional[float] = None) -> LazyConverter:
        """
        Create a lazy converter for deferred conversion.
        
        Args:
            converter: Adapter to perform conversion
            request: Conversion request
            threshold_mb: Memory threshold (uses default if None)
            
        Returns:
            Lazy converter instance
        """
        threshold = threshold_mb or self.default_threshold_mb
        lazy_converter = LazyConverter(request, converter, threshold)
        
        # Generate tracking ID
        converter_id = f"{converter.adapter_id}_{id(request)}"
        
        with self._converter_lock:
            self._lazy_converters[converter_id] = lazy_converter
        
        logger.debug(f"Created lazy converter {converter_id} with threshold {threshold}MB")
        return lazy_converter
    
    def load_on_demand(self, lazy_converter: LazyConverter) -> Any:
        """
        Load lazy converter data on demand.
        
        Args:
            lazy_converter: Lazy converter to load
            
        Returns:
            Loaded conversion result data
        """
        return lazy_converter._ensure_loaded()
    
    def preload_background(self, lazy_converter: LazyConverter) -> asyncio.Future:
        """
        Start background preloading of lazy converter.
        
        Args:
            lazy_converter: Lazy converter to preload
            
        Returns:
            Future representing the background load operation
        """
        converter_id = f"{lazy_converter.state.converter.adapter_id}_{id(lazy_converter.state.request)}"
        
        if converter_id in self._background_tasks:
            # Already loading in background
            return self._background_tasks[converter_id]
        
        # Create async task for background loading
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop running, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def background_load():
            try:
                # Run synchronous load in thread pool
                loaded_data = await loop.run_in_executor(
                    self._background_executor,
                    lazy_converter._load_sync
                )
                return loaded_data
            except Exception as e:
                logger.error(f"Background loading failed for {converter_id}: {e}")
                raise
            finally:
                # Clean up task tracking
                with self._converter_lock:
                    self._background_tasks.pop(converter_id, None)
        
        future = asyncio.create_task(background_load())
        lazy_converter.state.load_future = future
        
        with self._converter_lock:
            self._background_tasks[converter_id] = future
        
        logger.debug(f"Started background preloading for {converter_id}")
        return future
    
    def cancel_loading(self, loading_future: asyncio.Future) -> bool:
        """
        Cancel background loading operation.
        
        Args:
            loading_future: Future to cancel
            
        Returns:
            True if successfully canceled
        """
        try:
            if not loading_future.done():
                loading_future.cancel()
                logger.debug("Canceled background loading task")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel loading task: {e}")
            return False
    
    def cleanup_unused(self) -> int:
        """
        Clean up unused lazy converters based on access patterns.
        
        Returns:
            Number of converters cleaned up
        """
        current_time = time.time()
        
        # Only cleanup periodically
        if current_time - self._last_cleanup < self.cleanup_interval_seconds:
            return 0
        
        self._last_cleanup = current_time
        cleanup_threshold = 1800  # 30 minutes of inactivity
        
        to_remove = []
        
        with self._converter_lock:
            for converter_id, lazy_converter in self._lazy_converters.items():
                state = lazy_converter.state
                
                # Remove if unused for a long time or has errors
                if ((state.last_access_time is not None and 
                     current_time - state.last_access_time > cleanup_threshold) or
                    (state.last_access_time is None and 
                     current_time - state.creation_time > cleanup_threshold) or
                    state.load_error is not None):
                    to_remove.append(converter_id)
            
            for converter_id in to_remove:
                del self._lazy_converters[converter_id]
                # Cancel any background task
                if converter_id in self._background_tasks:
                    task = self._background_tasks[converter_id]
                    if not task.done():
                        task.cancel()
                    del self._background_tasks[converter_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} unused lazy converters")
        
        return len(to_remove)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of lazy loading manager.
        
        Returns:
            Status information
        """
        with self._converter_lock:
            total_converters = len(self._lazy_converters)
            loaded_converters = sum(1 for lc in self._lazy_converters.values() if lc.is_loaded())
            loading_converters = sum(1 for lc in self._lazy_converters.values() if lc.state.is_loading)
            error_converters = sum(1 for lc in self._lazy_converters.values() if lc.state.load_error is not None)
            background_tasks = len(self._background_tasks)
            
            return {
                'total_converters': total_converters,
                'loaded_converters': loaded_converters,
                'loading_converters': loading_converters,
                'error_converters': error_converters,
                'background_tasks': background_tasks,
                'last_cleanup': self._last_cleanup
            }
    
    def __del__(self):
        """Cleanup resources on destruction."""
        if hasattr(self, '_background_executor'):
            self._background_executor.shutdown(wait=False)


# Minimal fallback implementations for components not yet implemented
class PerformanceBenchmark:
    """Minimal performance benchmark implementation."""
    
    def __init__(self, *args, **kwargs):
        logger.info("PerformanceBenchmark initialized (minimal implementation)")
    
    def benchmark_conversion(self, converter, test_data, test_name=None):
        logger.warning("Performance benchmarking not fully implemented")
        return None


class OptimizationSelector:
    """Minimal optimization selector implementation."""
    
    def __init__(self, *args, **kwargs):
        logger.info("OptimizationSelector initialized (minimal implementation)")
    
    def analyze_data_characteristics(self, data):
        logger.warning("Data characteristics analysis not fully implemented")
        return None
    
    def select_optimization_strategy(self, characteristics):
        logger.warning("Strategy selection not fully implemented")
        return None


# Export main components
__all__ = [
    # Core caching system
    'ConversionCache',
    'CachedConversion', 
    'CacheStatistics',
    'CacheEvictionPolicy',
    
    # Lazy loading system
    'LazyLoadingManager',
    'LazyConverter',
    'LazyConversionState',
    
    # Performance optimization components
    'PerformanceBenchmark',
    'OptimizationSelector',
    'OptimizationStrategy'
]
