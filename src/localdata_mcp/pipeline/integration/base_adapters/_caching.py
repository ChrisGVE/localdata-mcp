"""
Performance-optimized adapter with intelligent caching strategies.

Extends BaseShimAdapter with advanced caching mechanisms including
LRU eviction, cache warming, and intelligent cache key generation.
"""

import time
import hashlib
from typing import Dict

from ..interfaces import ConversionRequest, ConversionResult
from ._core import BaseShimAdapter
from ....logging_manager import get_logger

logger = get_logger(__name__)


class CachingShimAdapter(BaseShimAdapter):
    """
    Performance-optimized adapter with intelligent caching strategies.

    Extends BaseShimAdapter with advanced caching mechanisms including
    LRU eviction, cache warming, and intelligent cache key generation.
    """

    def __init__(
        self,
        adapter_id: str,
        cache_size_mb: int = 256,
        cache_ttl_seconds: int = 3600,
        **kwargs,
    ):
        """
        Initialize CachingShimAdapter.

        Args:
            adapter_id: Unique identifier for this adapter
            cache_size_mb: Maximum cache size in MB
            cache_ttl_seconds: Cache time-to-live in seconds
            **kwargs: Additional arguments passed to BaseShimAdapter
        """
        kwargs["enable_caching"] = True  # Force caching enabled
        super().__init__(adapter_id, **kwargs)

        self.cache_size_mb = cache_size_mb
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache_access_times: Dict[str, float] = {}
        self._cache_sizes: Dict[str, int] = {}
        self._total_cache_size = 0

        logger.info(
            f"Initialized CachingShimAdapter",
            adapter_id=adapter_id,
            cache_size_mb=cache_size_mb,
            cache_ttl_seconds=cache_ttl_seconds,
        )

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
                logger.info(
                    "Returned cached conversion result with TTL validation",
                    request_id=request.request_id,
                    cache_age_seconds=time.time() - cache_time,
                )
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
        while (
            self._total_cache_size + result_size > self.cache_size_mb * 1024 * 1024
            and self._conversion_cache
        ):
            self._evict_lru_entry()

        # Add to cache
        self._conversion_cache[cache_key] = result
        self._cache_access_times[cache_key] = time.time()
        self._cache_sizes[cache_key] = result_size
        self._total_cache_size += result_size

        logger.debug(
            f"Added result to cache",
            cache_key=cache_key[:16] + "...",
            result_size_mb=result_size / (1024 * 1024),
            total_cache_size_mb=self._total_cache_size / (1024 * 1024),
        )

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
        lru_key = min(
            self._cache_access_times.keys(), key=lambda k: self._cache_access_times[k]
        )

        logger.debug(f"Evicting LRU cache entry", cache_key=lru_key[:16] + "...")
        self._remove_from_cache(lru_key)

    def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key
            for key, access_time in self._cache_access_times.items()
            if current_time - access_time > self.cache_ttl_seconds
        ]

        for key in expired_keys:
            self._remove_from_cache(key)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _estimate_result_size(self, result: ConversionResult) -> int:
        """Estimate size of conversion result."""
        base_size = 1024  # Base overhead

        if hasattr(result.converted_data, "memory_usage"):
            base_size += int(result.converted_data.memory_usage(deep=True).sum())
        elif hasattr(result.converted_data, "nbytes"):
            base_size += int(result.converted_data.nbytes)
        elif hasattr(result.converted_data, "__len__"):
            base_size += len(result.converted_data) * 64

        return base_size
