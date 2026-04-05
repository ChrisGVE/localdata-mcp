"""
Lazy Loading System for the Performance Optimization sub-package.

Provides deferred loading framework for large datasets with background
loading and lifecycle management.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..interfaces import ConversionRequest, ShimAdapter
from ....logging_manager import get_logger

logger = get_logger(__name__)


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

    def __init__(
        self,
        request: ConversionRequest,
        converter: ShimAdapter,
        threshold_mb: float = 50.0,
    ):
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
        if name.startswith("_"):
            # Don't intercept private attributes
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

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
                        while (
                            not self.state.load_future.done()
                            and time.time() - start_time < 30
                        ):
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

            logger.debug(
                f"Loading lazy conversion: "
                f"{self.state.request.source_format} -> "
                f"{self.state.request.target_format}"
            )

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
    Manager for lazy loading operations with background loading
    and lifecycle management.
    """

    def __init__(
        self,
        default_threshold_mb: float = 50.0,
        max_background_tasks: int = 3,
        cleanup_interval_seconds: int = 300,
    ):
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
            thread_name_prefix="lazy_loading",
        )
        self._background_tasks: Dict[str, asyncio.Future] = {}

        # Cleanup management
        self._last_cleanup = time.time()

        logger.info(
            f"LazyLoadingManager initialized",
            threshold_mb=default_threshold_mb,
            max_background_tasks=max_background_tasks,
        )

    def create_lazy_converter(
        self,
        converter: ShimAdapter,
        request: ConversionRequest,
        threshold_mb: Optional[float] = None,
    ) -> LazyConverter:
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

        logger.debug(
            f"Created lazy converter {converter_id} with threshold {threshold}MB"
        )
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
        converter_id = (
            f"{lazy_converter.state.converter.adapter_id}"
            f"_{id(lazy_converter.state.request)}"
        )

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
                    lazy_converter._load_sync,
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
                if (
                    (
                        state.last_access_time is not None
                        and current_time - state.last_access_time > cleanup_threshold
                    )
                    or (
                        state.last_access_time is None
                        and current_time - state.creation_time > cleanup_threshold
                    )
                    or state.load_error is not None
                ):
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
            loaded_converters = sum(
                1 for lc in self._lazy_converters.values() if lc.is_loaded()
            )
            loading_converters = sum(
                1 for lc in self._lazy_converters.values() if lc.state.is_loading
            )
            error_converters = sum(
                1
                for lc in self._lazy_converters.values()
                if lc.state.load_error is not None
            )
            background_tasks = len(self._background_tasks)

            return {
                "total_converters": total_converters,
                "loaded_converters": loaded_converters,
                "loading_converters": loading_converters,
                "error_converters": error_converters,
                "background_tasks": background_tasks,
                "last_cleanup": self._last_cleanup,
            }

    def __del__(self):
        """Cleanup resources on destruction."""
        if hasattr(self, "_background_executor"):
            self._background_executor.shutdown(wait=False)
